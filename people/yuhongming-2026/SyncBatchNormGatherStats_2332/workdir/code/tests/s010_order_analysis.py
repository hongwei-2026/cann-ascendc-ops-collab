#!/usr/bin/env python3
"""Compare CPU float32 operation orders with S010 observations."""
import numpy as np
import s007_matrix as matrix
from s010_compact import INDICES


def main():
    index, case = matrix.select_cases(matrix.cases(), "large_f32_i32_c524289")[0]
    data = matrix.inputs_for(case, 20261119 + index)
    q = matrix.reference._q16
    n, momentum = np.float32(100), np.float32(0.1)
    sums = q(data["total_sum"]).ravel()
    squares = q(data["total_square_sum"]).ravel()
    running_var = q(data["variance"])
    reference = matrix.golden(case, data)
    builtin_examples = np.array([7.6054688, 5.921875, 5.8867188, 6.1367188,
                                 7.5703125, 10.421875, 12.7578125, 8.0859375],
                                dtype=np.float32)
    unbias = np.float32(n / (n - np.float32(1)))
    for normalization in ("div", "reciprocal"):
        mean = sums / n if normalization == "div" else sums * np.float32(1 / n)
        square_mean = squares / n if normalization == "div" else squares * np.float32(1 / n)
        variance = square_mean - mean * mean
        for order in ("factor_first", "momentum_first"):
            term = ((variance * unbias) * momentum if order == "factor_first"
                    else variance * np.float32(unbias * momentum))
            result = q((np.float32(1) - momentum) * running_var + term)
            matches = np.count_nonzero(result[INDICES[7:]] == builtin_examples)
            mismatch = matrix.compare(result, reference[3])["mismatched"]
            print(f"{normalization} {order}: builtin examples {matches}/8; "
                  f"reference mismatches {mismatch}/{result.size}")
    print("CPU model only; neither platform output nor device instruction rounding is proven.")


if __name__ == "__main__":
    main()
