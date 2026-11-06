#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 NPU：核对分块归约、gamma 取模与 reference.impl 一致。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))
import reference  # noqa: E402


def kernel_sim(x, gamma, beta, begin_norm_axis, epsilon, tile):
    """复刻 kernel 的行切分、room 切分和 gammaNum==1 广播。数值用 float32。"""
    x = np.asarray(x, np.float32)
    g = np.asarray(gamma, np.float32).reshape(-1)
    b = np.asarray(beta, np.float32).reshape(-1)
    rank = x.ndim
    if begin_norm_axis < 0:
        begin_norm_axis += rank
    row = int(np.prod(x.shape[:begin_norm_axis], dtype=np.int64)) if begin_norm_axis > 0 else 1
    col = int(np.prod(x.shape[begin_norm_axis:], dtype=np.int64)) if begin_norm_axis < rank else 1
    xf = x.reshape(row, col)
    y = np.empty_like(xf)
    mean = np.empty((row,), np.float32)
    rstd = np.empty((row,), np.float32)
    gnum = int(g.size)
    scalar = gnum == 1
    for r in range(row):
        total = np.float32(0)
        off = 0
        while off < col:
            valid = min(tile, col - off)
            total = np.float32(total + xf[r, off : off + valid].astype(np.float32).sum(dtype=np.float32))
            off += valid
        m = np.float32(total / np.float32(col))
        sumsq = np.float32(0)
        off = 0
        while off < col:
            valid = min(tile, col - off)
            centered = xf[r, off : off + valid] - m
            sumsq = np.float32(sumsq + np.square(centered).sum(dtype=np.float32))
            off += valid
        rs = np.float32(1.0 / np.sqrt(sumsq / np.float32(col) + np.float32(epsilon)))
        mean[r] = m
        rstd[r] = rs
        off = 0
        while off < col:
            if scalar:
                valid = min(tile, col - off)
                gg = np.float32(g[0])
                bb = np.float32(b[0])
                y[r, off : off + valid] = (xf[r, off : off + valid] - m) * rs * gg + bb
            else:
                gpos = (r * col + off) % gnum
                room = gnum - gpos
                valid = min(tile, col - off, room)
                sl = slice(off, off + valid)
                y[r, sl] = (xf[r, sl] - m) * rs * g[gpos : gpos + valid] + b[gpos : gpos + valid]
            off += valid
    mshape = [1 if i >= begin_norm_axis else x.shape[i] for i in range(rank)]
    if rank == 0:
        return y.reshape(()), mean.reshape(()), rstd.reshape(())
    return y.reshape(x.shape), mean.reshape(mshape), rstd.reshape(mshape)


def check(name, x, gamma, beta, axis, params, tile):
    y_ref, mean_ref, rstd_ref = reference.impl(
        x, gamma, beta, begin_norm_axis=axis, begin_params_axis=params, epsilon=1e-5
    )
    y, mean, rstd = kernel_sim(x, gamma, beta, axis, 1e-5, tile)
    ok = (
        np.allclose(y, y_ref.astype(np.float32), rtol=1e-4, atol=1e-4, equal_nan=True)
        and np.allclose(mean, mean_ref.astype(np.float32), rtol=1e-4, atol=1e-4, equal_nan=True)
        and np.allclose(rstd, rstd_ref.astype(np.float32), rtol=1e-4, atol=1e-4, equal_nan=True)
    )
    if not ok:
        dy = np.max(np.abs(y.astype(np.float64) - y_ref.astype(np.float64)))
        raise SystemExit(f"FAIL {name} max_abs_y={dy}")
    print(f"PASS {name} y{tuple(y.shape)} mean{tuple(mean.shape)} tile={tile}")


def main():
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, (4, 6, 8)).astype(np.float32)
    gamma = rng.uniform(0.5, 2.0, (6, 8)).astype(np.float32)
    beta = rng.uniform(-1, 1, (6, 8)).astype(np.float32)
    check("sample", x, gamma, beta, 1, 1, 48)
    check("sample-small-tile", x, gamma, beta, 1, 1, 7)

    x2 = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    check("norm2", x2, rng.normal(size=(4, 5)).astype(np.float32), rng.normal(size=(4, 5)).astype(np.float32), 2, 2, 9)
    check(
        "params-after-norm",
        x2,
        rng.normal(size=(5,)).astype(np.float32),
        rng.normal(size=(5,)).astype(np.float32),
        2,
        3,
        8,
    )
    check(
        "params-before-norm",
        x2,
        rng.normal(size=(3, 4, 5)).astype(np.float32),
        rng.normal(size=(3, 4, 5)).astype(np.float32),
        2,
        1,
        11,
    )
    check("negative-axis", x2, np.ones((5,), np.float32), np.zeros((5,), np.float32), -1, -1, 5)
    check("whole-tensor", x, np.ones((4, 6, 8), np.float32), np.zeros((4, 6, 8), np.float32), 0, 0, 13)
    check("scalar-affine", x, np.array(1.5, np.float32), np.array(-0.25, np.float32), 1, 3, 10)

    xh = rng.uniform(-2, 2, (2, 5, 8)).astype(np.float16)
    gh = rng.uniform(0.5, 1.5, (8,)).astype(np.float16)
    bh = rng.uniform(-0.5, 0.5, (8,)).astype(np.float16)
    y_ref, _, _ = reference.impl(xh, gh, bh, begin_norm_axis=-1, begin_params_axis=-1)
    y_sim, _, _ = kernel_sim(xh.astype(np.float32), gh.astype(np.float32), bh.astype(np.float32), 2, 1e-5, 6)
    y_cast = y_sim.astype(np.float16)
    if not np.allclose(y_cast, y_ref, rtol=1e-3, atol=1e-3):
        raise SystemExit("FAIL half cast")
    print("PASS half rtol=1e-3")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
