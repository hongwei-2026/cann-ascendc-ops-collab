"""CPU-only Q001 regressions. These do not compile or validate AscendC on a device."""
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import unittest

import ml_dtypes
import numpy as np

CODE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("quant_reference", CODE / "samples/reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def ceildiv(a, b):
    return a // b + (a % b != 0)


def tiling(batch, total, width, cores=64, ub=262144):
    budget = ub - 8192 - 2048 * 8
    if budget < 32 * 27:
        raise ValueError("UB cannot fit minimum aligned buffers")
    maximum = min(budget // 27 // 32 * 32, 8192)
    tile = ceildiv(min(max(width, 1), maximum), 32) * 32
    used = max(min(cores, max(batch, 1)), min(cores, ceildiv(total, 32768) if total else 1))
    return used, ceildiv(batch, used) if batch else 1, tile


def copy_ranges(total, cores):
    count = ceildiv(ceildiv(total, 32), cores)
    for block in range(cores):
        start = block * count * 32
        end = min(start + count * 32, total)
        if start < end:
            yield start, end


def quantize(values, scales, zero_points, reciprocal):
    work = values.astype(np.float32)
    if reciprocal:
        work = np.multiply(work, scales.astype(np.float32), dtype=np.float32)
    else:
        work = np.divide(work, scales.astype(np.float32), dtype=np.float32)
    if zero_points is not None:
        work = np.add(work, zero_points.astype(np.float32), dtype=np.float32)
    # Execute the specified conversion chain, including the int16 RINT intermediate.
    i16 = np.rint(np.clip(work, -128, 127)).astype(np.int16)
    return i16.astype(np.float16).astype(np.int8)


def scalar_oracle(var, indices, updates, scales, zero_points, reciprocal):
    out = var.copy()
    for coordinate in np.ndindex(updates.shape):
        batch = coordinate[0]
        if batch >= len(indices):
            continue
        column = coordinate[-1]
        value = np.float32(updates[coordinate])
        scale = np.float32(scales[column])
        value = np.float32(value * scale) if reciprocal else np.float32(value / scale)
        if zero_points is not None:
            value = np.float32(value + np.float32(zero_points[column]))
        destination = list(coordinate)
        destination[-2] += int(indices[batch])
        out[tuple(destination)] = np.int8(round(min(max(float(value), -128.0), 127.0)))
    return out


def tiled_schedule(var, indices, updates, scales, zero_points, reciprocal, cores, ub, inplace=False):
    batch, width = updates.shape[0], updates.shape[-1]
    used, factor, tile = tiling(batch, var.size, width, cores, ub)
    out = var if inplace else np.empty_like(var)
    if not inplace:
        for first, last in copy_ranges(var.size, used):
            out.reshape(-1)[first:last] = var.reshape(-1)[first:last]
    # All copy ranges have finished before any scatter, matching SyncAll.
    for block in range(used):
        first = block * factor
        length = max(0, min(factor, batch - first))
        cached = max(0, min(length, len(indices) - first))
        cache = indices[first:first+cached].copy() if cached <= 2048 else None
        for column in range(0, width, tile):
            end = min(width, column + tile)
            for p in range(first, first + length):
                if p >= len(indices):
                    break
                index = int(indices[p] if cache is None else cache[p-first])
                for mid in np.ndindex(updates.shape[1:-2]):
                    for k in range(updates.shape[-2]):
                        src = (p,) + mid + (k, slice(column, end))
                        dst = (p,) + mid + (index + k, slice(column, end))
                        zp = None if zero_points is None else zero_points[column:end]
                        out[dst] = quantize(updates[src], scales[column:end], zp, reciprocal)
    return out


class Q001LocalTest(unittest.TestCase):
    def test_fp32_division_boundary(self):
        var = np.zeros((1, 1, 1), np.int8)
        updates = np.ones((1, 1, 1), np.float16)
        scale = np.array([2.0/3.0], np.float32)
        self.assertEqual(int(reference.impl(var, [0], updates, scale).item()), 2)
        # The old fp64 reference rounded this case differently.
        self.assertEqual(int(np.rint(np.float64(1)/np.float64(scale[0]))), 1)

    def test_direct_half_conversion_would_double_round(self):
        scale = np.array([1.0/2.5002], np.float32)
        value = np.float32(1)/scale[0]
        self.assertEqual(int(np.rint(value)), 3)
        self.assertEqual(int(np.rint(np.float16(value))), 2)
        actual = reference.impl(np.zeros((1,1,1),np.int8), [0], np.ones((1,1,1),np.float16), scale)
        self.assertEqual(int(actual.item()), 3)

    def test_ties_saturation_and_no_input_mutation(self):
        values = np.array([-1000,-128.5,-127.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,126.5,127.5,1000],np.float16)
        updates = values.reshape(1,1,-1)
        var = np.full((1,3,len(values)), -99, np.int8)
        old = var.copy()
        out = reference.impl(var,[1],updates,np.ones(len(values),np.float32))
        np.testing.assert_array_equal(out[0,1],[-128,-128,-128,-2,-2,0,0,2,2,126,127,127])
        np.testing.assert_array_equal(out[0,0],old[0,0])
        np.testing.assert_array_equal(out[0,2],old[0,2])
        np.testing.assert_array_equal(var,old)

    def test_reciprocal_multiplies_directly(self):
        scale = np.array([0,2,0.25,3],np.float32)
        updates = np.array([1,2,4,-2],np.float16).reshape(1,1,4)
        out = reference.impl(np.zeros((1,1,4),np.int8),[0],updates,scale,reciprocal_scale=True)
        np.testing.assert_array_equal(out.reshape(-1),[0,4,1,-6])

    def test_copy_coverage_and_disjoint_aligned_boundaries(self):
        totals=[0,1,7,31,32,33,63,64,65,32767,32768,32769,65537,2**31+17,2**40+3]
        for total,cores in itertools.product(totals,[1,2,3,8,48,64]):
            previous=0
            for first,last in copy_ranges(total,cores):
                self.assertEqual(first,previous)
                self.assertEqual(first % 32,0)
                self.assertLess(first,last)
                self.assertLessEqual(last,total)
                previous=last
            self.assertEqual(previous,total)
        self.assertGreater(tiling(1,16*32768,33)[0],1)

    def test_ub_and_dma_padding(self):
        for ub,width in itertools.product([32768,65536,196608,262144], [0,1,7,8,15,16,17,31,32,33,8191,8192,8193,65537]):
            _,_,tile=tiling(64,1<<20,width,64,ub)
            self.assertEqual(tile % 32,0)
            self.assertLessEqual(tile*27+2048*8+8192,ub)
            for element_size in [1,2,4,8]:
                for length in [1,7,8,15,16,17,31,32,min(width,tile)]:
                    if 0 < length <= tile:
                        self.assertLessEqual(ceildiv(length*element_size,32)*32,tile*element_size)
        for cache in [1,3,7,2047,2048]:
            self.assertLessEqual(ceildiv(cache*8,32)*32,2048*8)

    def test_reference_matches_independent_scalar_fp32_oracle(self):
        rng=np.random.default_rng(2330)
        for dtype,reciprocal,with_zp in itertools.product([np.float16,ml_dtypes.bfloat16],[False,True],[False,True]):
            for middle in [(),(2,),(2,2)]:
                width=33
                shape=(3,)+middle+(2,width)
                var=rng.integers(-128,128,(3,)+middle+(5,width),dtype=np.int8)
                updates=rng.uniform(-300,300,shape).astype(dtype)
                scales=rng.uniform(0.1,3,width).astype(np.float32 if dtype==np.float16 else dtype)
                zp=None if not with_zp else (rng.integers(-30,30,width,dtype=np.int32) if dtype==np.float16 else rng.uniform(-5,5,width).astype(dtype))
                indices=np.array([0,3,1],dtype=np.int32 if dtype==np.float16 else np.int64)
                expected=scalar_oracle(var,indices,updates,scales,zp,reciprocal)
                actual=reference.impl(var,indices,updates,scales,zp,reciprocal_scale=reciprocal)
                np.testing.assert_array_equal(actual,expected)

    def test_schedule_equivalence_modes_tails_and_index_lengths(self):
        rng=np.random.default_rng(1001)
        case_count=0
        for dtype,reciprocal,with_zp,width in itertools.product([np.float16,ml_dtypes.bfloat16],[False,True],[False,True],[1,7,17,31,32,33,65,129,8193]):
            batch,middle,rows=(3,2,2) if width<1000 else (1,1,1)
            var=rng.integers(-128,128,(batch,middle,rows+3,width),dtype=np.int8)
            updates=rng.uniform(-500,500,(batch,middle,rows,width)).astype(dtype)
            scales=rng.uniform(0.01,5,width).astype(np.float32 if dtype==np.float16 else dtype)
            zp=None if not with_zp else (rng.integers(-50,50,width,dtype=np.int32) if dtype==np.float16 else rng.uniform(-10,10,width).astype(dtype))
            for count in [0,max(0,batch-1),batch,batch+2]:
                indices=rng.integers(0,4,count,dtype=np.int32 if dtype==np.float16 else np.int64)
                expected=reference.impl(var,indices,updates,scales,zp,reciprocal_scale=reciprocal)
                for inplace in [False,True]:
                    actual=tiled_schedule(var.copy(),indices,updates,scales,zp,reciprocal,4,32768,inplace)
                    np.testing.assert_array_equal(actual,expected)
                    case_count+=1
        print('Schedule comparisons:',case_count)

    def test_indices_cache_cap_and_fallback(self):
        for dtype,batch in itertools.product([np.float16,ml_dtypes.bfloat16],[2048,2049]):
            var=np.full((batch,2,1),-99,np.int8)
            updates=(np.arange(batch,dtype=np.float32)%127).astype(dtype).reshape(batch,1,1)
            indices=np.ones(batch,dtype=np.int32 if dtype==np.float16 else np.int64)
            scales=np.ones(1,dtype=np.float32 if dtype==np.float16 else dtype)
            expected=reference.impl(var,indices,updates,scales)
            actual=tiled_schedule(var,indices,updates,scales,None,False,1,65536)
            np.testing.assert_array_equal(actual,expected)


if __name__ == '__main__':
    print('CPU-only checks: no CANN build, device precision or performance claim.')
    unittest.main(verbosity=2)
