import argparse
from pathlib import Path

import numpy as np
import reference


ROOT = Path(__file__).resolve().parent

OUTPUTS = [
    (
        'neg_log_likelihood.bin',
        [1],
        np.dtype('float32'),
        4,
        0.0001,
        0.0001,
        0.0001
    ),
    ('log_alpha.bin', [
        1,
        10,
        17
    ], np.dtype('float32'), 680, 0.0001, 0.0001, 0.0001),
]


def generate(directory):
    rng = np.random.default_rng(42)
    for name in ('input', 'output', 'golden'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    for item in OUTPUTS:
        (directory / 'output' / item[0]).unlink(missing_ok=True)
        (directory / 'golden' / item[0]).unlink(missing_ok=True)
    x0 = rng.uniform(-5, 0, [10, 1, 5]).astype(np.dtype('float32'))
    (directory / 'input' / 'log_probs.bin').write_bytes(x0.tobytes())
    x1 = rng.integers(0, 4, size=[1, 8], dtype=np.dtype('int32'))
    (directory / 'input' / 'targets.bin').write_bytes(x1.tobytes())
    x2 = np.asarray([10], dtype=np.dtype('int32')).reshape([1])
    (directory / 'input' / 'input_lengths.bin').write_bytes(x2.tobytes())
    x3 = np.asarray([8], dtype=np.dtype('int32')).reshape([1])
    (directory / 'input' / 'target_lengths.bin').write_bytes(x3.tobytes())
    result = reference.impl(x0, x1, x2, x3, **{
        'blank': 0,
        'reduction': 'mean',
        'zero_infinity': False
    })
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError('reference output group count mismatch')
    values = [result[0], result[1]]
    for item, array in zip(OUTPUTS, values):
        name, shape, dtype = item[:3]
        if np.isscalar(array):
            array = np.asarray(array).reshape(shape)
        array = np.asarray(array)
        if array.dtype != dtype or list(array.shape) != shape:
            raise ValueError('reference dtype/shape mismatch: ' + name)
        (directory / 'golden' / name).write_bytes(array.tobytes())
    print('Sample data generated.')


def verify(directory):
    passed = True
    expected = {item[0] for item in OUTPUTS}
    for folder in ('output', 'golden'):
        actual = {p.name for p in (directory / folder).iterdir()}
        if actual != expected:
            print('VERIFY ERROR: unexpected or missing files in ' + folder)
            return 1
    for item in OUTPUTS:
        name, shape, dtype, size, rtol, atol, tol = item[:7]
        try:
            raw = (directory / 'output' / name).read_bytes()
            golden = (directory / 'golden' / name).read_bytes()
            if len(raw) != size or len(golden) != size:
                raise ValueError('exact byte size mismatch')
            a, b = [np.frombuffer(v, dtype=dtype) for v in (raw, golden)]
            close = np.isclose(
                a.astype(np.float64),
                b.astype(np.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True
            )
            bad = int(np.count_nonzero(~close))
            ok = bad <= tol * a.size
            print(
                ('PASS' if ok else 'Wrong Answer')
                + ': ' + name + f', mismatched={bad}/{a.size}'
            )
            passed &= ok
        except (OSError, ValueError) as error:
            print('VERIFY ERROR: ' + name + ': ' + str(error))
            passed = False
    print('PASSED' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'verify'))
    parser.add_argument('--directory', type=Path, default=ROOT / 'build-npu')
    args = parser.parse_args()
    try:
        if args.action == 'generate':
            generate(args.directory)
        else:
            raise SystemExit(verify(args.directory))
    except Exception as error:
        print('SAMPLE ERROR: ' + str(error))
        raise SystemExit(2)
