#!/usr/bin/env python3
"""Execute and check the actual PReluGradUpdate vector schedule source."""

import argparse
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL = ROOT / "op_kernel" / "p_relu_grad_update.h"
START = "Muls(negLocal, gradsFloat, weightValue, currentNum);"
END = "if constexpr (std::is_same<T, half>::value)"
CALL = re.compile(r"\b(Muls|Mul|Duplicate|Compare|Select|PipeBarrier)\s*(?:<([^>]+)>)?\s*\(([^;]*)\)\s*;", re.S)


def parse_schedule(path):
    source = path.read_text(encoding="utf-8")
    start = source.index(START)
    end = source.index(END, start)
    region = source[start:end]
    alias_match = re.search(r"LocalTensor<float>\s+zeroLocal\s*=\s*dxFloat\s*;", region)
    compute_start = source.rfind("__aicore__ inline void PReluGradUpdate<T>::Compute(", 0, start)
    independent_match = re.search(
        r"LocalTensor<float>\s+zeroLocal\s*=\s*zeroFloatBuf_\.Get<float>\(\)\s*;",
        source[compute_start:end],
    )
    assert alias_match or independent_match, "zeroLocal must be defined in the compute region"
    aliases = {"zeroLocal": "dxFloat"} if alias_match else {}
    events = []
    for match in CALL.finditer(region):
        kind, template_arg, raw_args = match.groups()
        args = [arg.strip() for arg in raw_args.split(",") if arg.strip()]
        if kind == "PipeBarrier":
            assert template_arg == "PIPE_V", f"unexpected barrier: {template_arg}"
            events.append((kind, []))
        else:
            events.append((kind, args))
    assert events, "no vector operations found"
    return aliases, events


def canonical(name, aliases):
    while name in aliases:
        name = aliases[name]
    return name


def accesses(event, aliases):
    kind, args = event
    if kind == "Muls":
        writes, reads = {args[0]}, {args[1]}
    elif kind in ("Mul",):
        writes, reads = {args[0]}, {args[1], args[2]}
    elif kind == "Duplicate":
        writes, reads = {args[0]}, set()
    elif kind == "Compare":
        writes, reads = {args[0]}, {args[1], args[2]}
    elif kind == "Select":
        writes, reads = {args[0]}, {args[1], args[2], args[3]}
    else:
        return set(), set()
    return ({canonical(x, aliases) for x in writes},
            {canonical(x, aliases) for x in reads})


def check_barrier_dependencies(aliases, events):
    op_indices = [i for i, (kind, _) in enumerate(events) if kind != "PipeBarrier"]
    for left_pos, left_i in enumerate(op_indices):
        left_writes, left_reads = accesses(events[left_i], aliases)
        for right_i in op_indices[left_pos + 1:]:
            right_writes, right_reads = accesses(events[right_i], aliases)
            hazard = (left_writes & (right_reads | right_writes)) | (left_reads & right_writes)
            if hazard:
                has_barrier = any(events[k][0] == "PipeBarrier" for k in range(left_i + 1, right_i))
                assert has_barrier, (
                    f"missing PIPE_V barrier between {events[left_i]} and {events[right_i]} "
                    f"for aliased tensors {sorted(hazard)}"
                )


def execute(aliases, events, features, grads, weight, zero_buffer=None):
    zero = (np.zeros(features.size, dtype=np.float32) if zero_buffer is None
            else zero_buffer[:features.size])
    tensors = {"featuresFloat": features, "gradsFloat": grads,
               "zeroLocal": zero, "dxFloat": zero.copy()}

    def read(name):
        return tensors[canonical(name, aliases)]

    with np.errstate(all="ignore"):
        for kind, args in events:
            if kind == "PipeBarrier":
                continue
            if kind == "Muls":
                tensors[canonical(args[0], aliases)] = read(args[1]) * np.float32(weight)
            elif kind == "Mul":
                tensors[canonical(args[0], aliases)] = read(args[1]) * read(args[2])
            elif kind == "Duplicate":
                tensors[canonical(args[0], aliases)] = np.zeros(features.size, dtype=np.float32)
            elif kind == "Compare":
                tensors[canonical(args[0], aliases)] = read(args[1]) > read(args[2])
            elif kind == "Select":
                tensors[canonical(args[0], aliases)] = np.where(
                    read(args[1]), read(args[2]), read(args[3])
                )
            else:
                raise AssertionError(f"unhandled operation {kind}")
    return tensors["dxFloat"], tensors["updateFloat"]


def verify_semantics(path, aliases, events):
    rng = np.random.default_rng(20261103)
    special = np.array(
        [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan,
         np.nextafter(np.float32(0), np.float32(1)), -1e-30],
        dtype=np.float32,
    )
    weights = [0.0, -0.0, 0.125, -0.75, np.inf, -np.inf, np.nan]
    import sys
    sys.path.insert(0, str(ROOT / "samples"))
    import reference

    zero_buffer = np.zeros(8255, dtype=np.float32)
    for size in (1, 7, 8, 15, 16, 63, 64, 65, 127, 128, 257, 8255):
        features = rng.normal(size=size).astype(np.float32)
        grads = rng.normal(size=size).astype(np.float32)
        features[:min(size, special.size)] = special[:min(size, special.size)]
        grads[:min(size, special.size)] = special[-min(size, special.size):]
        for dtype in (np.float32, np.float16):
            typed_features = features.astype(dtype)
            typed_grads = grads.astype(dtype)
            for weight in weights:
                typed_weight = np.asarray([weight], dtype=dtype)
                actual = execute(
                    aliases, events,
                    typed_features.astype(np.float32),
                    typed_grads.astype(np.float32),
                    np.float32(typed_weight.astype(np.float32)[0]),
                    zero_buffer,
                )
                with np.errstate(all="ignore"):
                    expected = reference.impl(
                        typed_grads.reshape(1, 1, -1),
                        typed_features.reshape(1, 1, -1),
                        typed_weight,
                    )
                actual = tuple(value.astype(dtype).reshape(1, 1, -1) for value in actual)
                np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
                np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=0, equal_nan=True)
                for output, golden in zip(actual, expected):
                    zeros = golden == 0
                    np.testing.assert_array_equal(
                        np.signbit(output[zeros]), np.signbit(golden[zeros])
                    )
                np.testing.assert_array_equal(zero_buffer, np.zeros_like(zero_buffer))


def check_zero_lifetime(source, events):
    init_start = source.index("__aicore__ inline void PReluGradUpdate<T>::Init(")
    init_end = source.index("__aicore__ inline void PReluGradUpdate<T>::CopyIn(", init_start)
    init = source[init_start:init_end]
    assert re.search(r"InitBuffer\(zeroFloatBuf_,\s*alignedCount\s*\*\s*sizeof\(float\)\)", init)
    assert re.search(
        r"Duplicate\(zeroLocal,\s*0\.0f,\s*static_cast<int32_t>\(alignedCount\)\);\s*PipeBarrier<PIPE_V>\(\);",
        init,
    ), "persistent zero must be initialized and synchronized before Compute"
    assert source.count("Duplicate(zeroLocal,") == 1
    assert events[-1][0] == "PipeBarrier", "outputs must be complete before conversion/copyout"
    for event in events:
        writes, _ = accesses(event, {})
        assert "zeroLocal" not in writes, "Compute must not modify the persistent zero buffer"


def check_hazard_detection(aliases, events):
    # Removing the producer/consumer barrier must fail dependency checking.
    first_barrier = next(i for i, event in enumerate(events) if event[0] == "PipeBarrier")
    broken = events[:first_barrier] + events[first_barrier + 1:]
    try:
        check_barrier_dependencies(aliases, broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("dependency checker accepted a missing producer barrier")
    # Reintroducing the P103 zero/dx alias without its WAR barrier must fail.
    try:
        check_barrier_dependencies({"zeroLocal": "dxFloat"}, events)
    except AssertionError:
        pass
    else:
        raise AssertionError("dependency checker accepted the zero/dx WAR hazard")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", nargs="?", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()

    aliases, events = parse_schedule(args.kernel)
    check_barrier_dependencies(aliases, events)
    verify_semantics(args.kernel, aliases, events)
    op_count = sum(kind != "PipeBarrier" for kind, _ in events)
    barrier_count = sum(kind == "PipeBarrier" for kind, _ in events)
    vector_ops = [event for event in events if event[0] != "PipeBarrier"]
    if "zeroLocal" in aliases:
        assert [kind for kind, _ in vector_ops] == [
            "Muls", "Mul", "Duplicate", "Compare", "Select", "Select"
        ], "P103 compute order changed"
        assert (op_count, barrier_count) == (6, 4), (op_count, barrier_count)
    else:
        assert [kind for kind, _ in vector_ops] == [
            "Muls", "Mul", "Compare", "Select", "Select"
        ], "P104 compute order changed"
        assert (op_count, barrier_count) == (5, 2), (op_count, barrier_count)
        check_zero_lifetime(args.kernel.read_text(encoding="utf-8"), events)
        check_hazard_detection(aliases, events)

    if args.baseline:
        base_aliases, base_events = parse_schedule(args.baseline)
        check_barrier_dependencies(base_aliases, base_events)
        verify_semantics(args.baseline, base_aliases, base_events)
        base_ops = sum(kind != "PipeBarrier" for kind, _ in base_events)
        base_barriers = sum(kind == "PipeBarrier" for kind, _ in base_events)
        print(f"Baseline: {base_ops} vector ops, {base_barriers} barriers; candidate: {op_count} vector ops, {barrier_count} barriers")
    else:
        print(f"{args.kernel}: {op_count} vector ops, {barrier_count} barriers")
    print("Compute schedule semantic and dependency checks: PASS")


if __name__ == "__main__":
    main()
