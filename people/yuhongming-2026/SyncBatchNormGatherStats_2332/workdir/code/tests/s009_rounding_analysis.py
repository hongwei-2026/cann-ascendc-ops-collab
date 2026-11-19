#!/usr/bin/env python3
"""Reproduce an fp32 arithmetic model for the recorded S008 large case."""
import numpy as np
import s007_matrix as matrix


def main():
    case = next(case for case in matrix.cases() if case["name"] == "large_f32_i32_c524289")
    data = matrix.inputs_for(case, 20261119 + 156)
    q = matrix.reference._q16
    count = np.float32(case["counts"][0])
    momentum = np.float32(case["momentum"])
    mean = q(data["total_sum"]).ravel() / count
    square_mean = q(data["total_square_sum"]).ravel() / count
    variance = square_mean - mean * mean
    unbias = count / (count - np.float32(1))
    model = q((np.float32(1) - momentum) * q(data["variance"]) +
              (unbias * momentum) * variance)
    reference = matrix.golden(case, data)[3]
    diff = matrix.compare(model, reference)
    print("fp32 CPU model versus reference new_variance:", diff["mismatched"], "/", diff["total"])
    indexes = np.array([188307, 201472, 215481, 301966, 368953, 377353, 384637, 454808])
    observed = np.array([7.609375, 5.921875, 5.8867188, 6.1328125,
                         7.5703125, 10.421875, 12.765625, 8.09375], dtype=np.float32)
    matches = int(np.count_nonzero(model[indexes] == observed))
    print("model matches recorded device examples:", matches, "/", len(indexes))
    print("This is a CPU model, not an independent 910B built-in comparison.")
    return 0 if matches == len(indexes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
