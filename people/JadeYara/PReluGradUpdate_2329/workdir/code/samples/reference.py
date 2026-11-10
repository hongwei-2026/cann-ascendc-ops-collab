"""CPU model of the CANN 9.3 after_v200 TBE operation chain.

This is an arithmetic model, not a substitute for the built-in NPU oracle.
Inputs/outputs use physical storage shapes; NC1HWC0 is not a layout conversion.
Logical comparisons treat signed zero as equal and compare NaNs by category.
"""
import numpy as np


def impl(grads, features, weights, data_format="ND"):
    values = (grads, features, weights)
    if any(not isinstance(v, np.ndarray) for v in values):
        raise TypeError("All inputs must be numpy arrays")
    dtype = grads.dtype
    if dtype.name not in ("float16", "float32", "bfloat16"):
        raise TypeError("Expected float16, float32 or bfloat16")
    if any(v.dtype != dtype for v in values):
        raise TypeError("Input dtypes must agree")
    if any(v.ndim > 8 for v in values):
        raise ValueError("The verified TBE rank limit is eight")
    if data_format not in ("ND", "NCHW", "NC1HWC0"):
        raise ValueError("Unsupported physical data format")
    xshape, wshape = features.shape, weights.shape
    # Keep the reference broadcast_weight_shape branch order.
    if len(wshape) == len(xshape) - 1 and len(wshape) != 1:
        mapped = (1,) + wshape
    elif len(xshape) == 4:
        if not wshape:
            raise ValueError("Scalar weight descriptor is unsupported here")
        mapped = (1, wshape[0], 1, 1)
    elif data_format == "NC1HWC0" and len(wshape) == 5:
        mapped = wshape
    elif data_format == "NC1HWC0" and len(wshape) == 1 and wshape[0] != 1:
        mapped = (1, (wshape[0] + 15) // 16, 1, 1, 16)
    else:
        if len(xshape) < 2 or not wshape:
            raise ValueError("Unsupported feature/weight descriptors")
        mapped = (1, wshape[0]) + (1,) * (len(xshape) - 2)
    output_shape = np.broadcast_shapes(grads.shape, features.shape)
    # reshape validates storage element count; broadcast_to cannot enlarge output.
    w = np.broadcast_to(weights.reshape(mapped), output_shape).astype(np.float32)
    g = np.broadcast_to(grads, output_shape).astype(np.float32)
    x = np.broadcast_to(features, output_shape).astype(np.float32)
    neg = np.where(x > 0, np.float32(0), np.float32(1))
    pos = np.where(x > 0, np.float32(1), np.float32(0))
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        dx_compute = np.multiply(np.add(pos, np.multiply(neg, w)), g)
        dx = dx_compute.astype(dtype)
        update_compute = np.multiply(np.multiply(neg, x), g)
        update_compute = np.add(update_compute,
                                np.multiply(dx.astype(np.float32), np.float32(0)))
        update = update_compute.astype(dtype)
    return dx, update
