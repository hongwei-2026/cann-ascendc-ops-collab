#!/usr/bin/env python3
"""Q005 diagnostics. Synthetic cases are NOT the official platform 50 cases.
Only the run action invokes an NPU program. Raw profiler exports remain authoritative.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parents[4]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def reference_module():
    spec = importlib.util.spec_from_file_location('qus_reference', CODE / 'samples/reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_specs():
    # Independent regimes diagnose copy, row count and width, not platform cases.
    regimes = [
        ('tiny', 1, 1, 2, 1, 32),
        ('copy_heavy', 1, 1, 8192, 1, 256),
        ('narrow_rows', 1, 16, 256, 128, 33),
        ('wide_rows', 1, 2, 32, 16, 8193),
        ('many_batches', 256, 1, 8, 2, 256),
    ]
    for name, b, m, v, u, w in regimes:
        for inplace in (False, True):
            yield dict(id=name + ('_in' if inplace else '_out'), regime=name,
                       batch=b, middle=m, var_rows=v, update_rows=u, width=w,
                       mode=0, index_bits=32, zp=True, reciprocal=False, inplace=inplace)
    for mode in (0, 1):
        for bits in (32, 64):
            for zp in (False, True):
                for reciprocal in (False, True):
                    yield dict(id=f'contract_m{mode}_i{bits}_z{int(zp)}_r{int(reciprocal)}',
                               regime='signature_contract', batch=3, middle=2,
                               var_rows=5, update_rows=2, width=65,
                               mode=mode, index_bits=bits, zp=zp,
                               reciprocal=reciprocal, inplace=bool(reciprocal))


def generate(directory):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    reference = reference_module()
    specs = list(case_specs())
    for number, case in enumerate(specs):
        folder = directory / case['id']
        for child in ('input', 'golden', 'output'):
            (folder / child).mkdir(parents=True)
        seed = 2330 + (number // 2 if number < 10 else number)
        rng = np.random.default_rng(seed)
        b, m, v, u, w = (case[k] for k in ('batch', 'middle', 'var_rows', 'update_rows', 'width'))
        if case['mode'] == 1:
            import ml_dtypes
            update_type = scale_type = zp_type = ml_dtypes.bfloat16
        else:
            update_type, scale_type, zp_type = np.float16, np.float32, np.int32
        var = rng.integers(-128, 128, (b, m, v, w), dtype=np.int8)
        indices = (np.arange(b, dtype=np.int64) % (v - u + 1)).astype(
            np.int32 if case['index_bits'] == 32 else np.int64)
        updates = (rng.integers(-2048, 2049, (b, m, u, w)) * 0.125).astype(update_type)
        scales = rng.uniform(0.2, 3.0, w).astype(scale_type).reshape(1, 1, 1, w)
        zp = ((np.arange(w) % 13) - 6).astype(zp_type).reshape(1, 1, 1, w) if case['zp'] else None
        inputs = dict(var=var, indices=indices, updates=updates, quant_scales=scales)
        if zp is not None:
            inputs['quant_zero_points'] = zp
        hashes = {}
        for key, value in inputs.items():
            raw = value.tobytes()
            (folder / 'input' / (key + '.bin')).write_bytes(raw)
            hashes[key] = hashlib.sha256(raw).hexdigest()
        expected = reference.impl(var, indices, updates, scales, zp,
                                  reciprocal_scale=case['reciprocal'])
        raw = expected.tobytes()
        (folder / 'golden/var_out.bin').write_bytes(raw)
        case.update(seed=seed, input_sha256=hashes,
                    golden_sha256=hashlib.sha256(raw).hexdigest(), output_bytes=len(raw),
                    source='synthetic_diagnostic_NOT_official_50_cases')
        write_json(folder / 'case.json', case)
    write_json(directory / 'manifest.json', dict(schema=1, cases=specs))
    return specs


def validate_case(directory, case):
    folder = directory / case['id']
    for name, expected in case['input_sha256'].items():
        path = folder / 'input' / (name + '.bin')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('changed input: ' + str(path))
    golden = (folder / 'golden/var_out.bin').read_bytes()
    if len(golden) != case['output_bytes'] or hashlib.sha256(golden).hexdigest() != case['golden_sha256']:
        raise ValueError('changed golden: ' + case['id'])
    return folder, golden


def verify(directory, case):
    folder, golden = validate_case(directory, case)
    raw = (folder / 'output/var_out.bin').read_bytes()
    if len(raw) != len(golden):
        raise ValueError('output byte count mismatch: ' + case['id'])
    bad = sum(a != b for a, b in zip(raw, golden))
    return dict(case_id=case['id'], mismatched=bad, elements=len(raw), passed=bad == 0)


def normalize(value):
    return ''.join(c for c in value.lower() if c.isalnum())


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(directory, warmup, repeat):
    # Only one identifiable kernel task per launch is safe to aggregate here.
    # Missing or ambiguous exports are incomplete, never zero-duration.
    reports = []
    for path in sorted(directory.rglob('*op_summary*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            keys = {normalize(key): key for key in headers}
            duration = next((keys[k] for k in ('taskdurationus', 'durationus') if k in keys), None)
            start = next((keys[k] for k in ('taskstarttimeus', 'starttimeus') if k in keys), None)
            names = [key for key in headers if normalize(key) in ('opname', 'optype', 'name')]
            rows = [row for row in reader if any('quantupdatescatter' in normalize(row.get(k) or '') for k in names)]
        report = dict(file=str(path), matched_tasks=len(rows), expected_tasks=warmup + repeat,
                      duration_column=duration, start_column=start, status='INCOMPLETE')
        if not rows or not duration or not start or len(rows) != warmup + repeat:
            report['reason'] = 'Need one identifiable kernel task per launch, timestamps and exact launch count.'
            reports.append(report)
            continue
        starts = [number(row.get(start)) for row in rows]
        if any(value is None for value in starts) or len(set(starts)) != len(starts):
            report['reason'] = 'Missing/duplicate timestamps; cannot identify warmup tasks safely.'
            reports.append(report)
            continue
        rows = [row for _, row in sorted(zip(starts, rows), key=lambda pair: pair[0])][warmup:]
        durations = [number(row.get(duration)) for row in rows]
        if any(value is None or value <= 0 for value in durations):
            report['reason'] = 'Nonpositive or missing duration.'
            reports.append(report)
            continue
        metrics = {}
        for key in headers:
            if any(term in normalize(key) for term in ('mte', 'vec', 'scalar', 'cycle', 'ratio')):
                values = [number(row.get(key)) for row in rows]
                if all(value is not None for value in values):
                    metrics[key] = dict(median=statistics.median(values), min=min(values), max=max(values))
        report.update(status='COMPLETE', samples=len(durations), excluded_warmup=warmup,
                      task_duration_us=dict(median=statistics.median(durations), p10=percentile(durations, .1),
                                     p90=percentile(durations, .9), min=min(durations), max=max(durations)),
                      raw_metric_columns=metrics)
        reports.append(report)
    raw_exports = []
    for path in sorted(directory.rglob('*.csv')):
        if any(term in normalize(path.name) for term in ('pipeutilization', 'opbasicinfo')):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                raw_exports.append(dict(file=str(path), columns=reader.fieldnames,
                                        rows=len(rows), preview=rows[:2]))
    complete = reports and all(r['status'] == 'COMPLETE' for r in reports)
    raw_available = any(item['rows'] > 0 for item in raw_exports)
    return dict(status='COMPLETE' if complete else ('RAW_ONLY' if raw_available else 'INCOMPLETE'),
                raw_operator_exports=raw_exports,
                exports=reports, caution='Profiler task durations only; synthetic diagnostics are not platform results. '
                'Task duration includes scheduling/execution/completion, not only vector execution. '
                'Keep raw exports/timeline. Metrics retain original column units. '
                'No pipeline overlap or performance gain is inferred automatically.')


def runner_arguments(runner, folder, case, device, warmup, repeat):
    return [str(runner), str(folder), str(device)] + [str(int(case[k])) for k in
           ('batch', 'middle', 'var_rows', 'update_rows', 'width', 'mode', 'index_bits',
            'zp', 'reciprocal', 'inplace')] + [str(warmup), str(repeat)]


def capture(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding='utf-8', errors='replace')
    return result.returncode, result.stdout


def source_identity():
    files = ['op_kernel/quant_update_scatter.h', 'op_host/quant_update_scatter_tiling.cpp',
             'op_kernel/quant_update_scatter.cpp', 'examples/profile_quant_update_scatter.cpp',
             'tools/profile_q005.py', 'examples/CMakeLists.txt']
    hashes = {name: hashlib.sha256((CODE / name).read_bytes()).hexdigest() for name in files}
    code, head = capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
    package = CODE / 'build/packages/vendors/quant_update_scatter_custom'
    artifacts = {}
    if package.is_dir():
        for path in sorted(package.rglob('*')):
            if path.is_file() and (path.suffix in ('.so', '.o', '.json') or '.so.' in path.name):
                artifacts[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = CODE / 'build/.source-fingerprint'
    return dict(git_head=head.strip() if code == 0 else None, source_sha256=hashes,
                package_sha256=artifacts,
                build_source_fingerprint=fingerprint.read_text().strip() if fingerprint.is_file() else None,
                custom_opp_path=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
                note='Source/package hashes identify local artifacts, not proof of device-loaded code. '
                     'Archive the successful build -e log and confirm custom-op path.')


def profiler_command(mode, executable, command, target, help_text, launches):
    if mode == 'op':
        result = [executable, 'op', '--output=' + str(target)]
        if '--launch-count' in help_text:
            result += ['--launch-count=' + str(launches)]
        if '--aic-metrics' in help_text:
            result += ['--aic-metrics=PipeUtilization']
        return result + command
    return [executable, '--application=' + shlex.join(command),
            '--output=' + str(target), '--aic-metrics=PipeUtilization']


def run(args, cases):
    if platform.system() != 'Linux':
        raise ValueError('run requires configured Linux NPU host; use generate/self-test locally')
    runner = args.runner.resolve()
    if not runner.is_file() or shutil.which(args.msprof) is None:
        raise ValueError('runner or msprof not found')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rc, help_text = capture([args.msprof, '--help'])
    (output / 'msprof-help.log').write_text(help_text, encoding='utf-8')
    op_rc, op_help = capture([args.msprof, 'op', '--help'])
    (output / 'msprof-op-help.log').write_text(op_help, encoding='utf-8')
    mode = args.profiler_mode
    if mode == 'auto':
        has_application = rc == 0 and all(option in help_text for option in
                                           ('--application', '--output', '--aic-metrics'))
        mode = 'application' if has_application else 'op'
    if mode == 'op' and (op_rc != 0 or '--output' not in op_help):
        raise ValueError('Unsupported msprof op CLI; inspect saved help')
    if mode == 'application' and (rc != 0 or not all(
            option in help_text for option in ('--application', '--output', '--aic-metrics'))):
        raise ValueError('Unsupported msprof application CLI; inspect saved help')
    version_rc, version = capture([args.msprof, '--version'])
    write_json(output / 'run.json', dict(schema=1, source=source_identity(), runner=str(runner), profiler_mode=mode,
               runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
               warmup=args.warmup, repeat=args.repeat, rounds=args.rounds, device=args.device,
               msprof_version_rc=version_rc, msprof_version=version,
               case_ids=[case['id'] for case in cases]))
    results = []
    for round_index in range(args.rounds):
        ordered = cases if round_index % 2 == 0 else list(reversed(cases))
        for case in ordered:
            folder, _ = validate_case(args.directory, case)
            folder = folder.resolve()
            target = output / f'round_{round_index + 1:02d}' / case['id']
            target.mkdir(parents=True)
            (folder / 'output/var_out.bin').unlink(missing_ok=True)
            command = runner_arguments(runner, folder, case, args.device, args.warmup, args.repeat)
            # Build outside application; never combine --export with --application.
            profiler = profiler_command(mode, args.msprof, command, target / 'prof',
                                        op_help if mode == 'op' else help_text,
                                        args.warmup + args.repeat)
            if mode == 'application' and '--task-time' in help_text:
                profiler += ['--task-time=l1']
            write_json(target / 'invocation.json', dict(command=profiler, case=case))
            rc, log = capture(profiler)
            (target / 'collector.log').write_text(log, encoding='utf-8')
            raw = folder / 'output/var_out.bin'
            if raw.is_file():
                shutil.copyfile(raw, target / 'var_out.bin')
            result = dict(round=round_index + 1, case_id=case['id'], collector_returncode=rc)
            try:
                result['verification'] = verify(args.directory, case)
            except (OSError, ValueError) as error:
                result['verification'] = dict(passed=False, error=str(error))
            result['profile'] = summarize(target / 'prof', args.warmup, args.repeat)
            result['execution_verified'] = rc == 0 and result['verification']['passed']
            write_json(target / 'result.json', result)
            results.append(result)
            write_json(output / 'results.json', dict(results=results, device_execution=True))
            print(case['id'], 'verified' if result['execution_verified'] else 'FAILED',
                  'profile=' + result['profile']['status'], flush=True)
            if not result['execution_verified']:
                raise ValueError('Collector/precision failure; logs preserved at ' + str(target))
    complete = all(r['profile']['status'] == 'COMPLETE' for r in results)
    collected = all(r['profile']['status'] in ('COMPLETE', 'RAW_ONLY') for r in results)
    print('Evidence:', output, 'summary complete:', complete)
    if not complete:
        print('Inspect raw PipeUtilization/OpBasicInfo and collection logs; '
              'no timing aggregate or bottleneck conclusion is inferred from RAW_ONLY exports.')
    return 0 if collected else 1


def self_test():
    with tempfile.TemporaryDirectory(prefix='q005-', dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        specs = generate(root / 'cases')
        for case in specs:
            folder, golden = validate_case(root / 'cases', case)
            (folder / 'output/var_out.bin').write_bytes(golden)
            assert verify(root / 'cases', case)['passed']
        for i in range(0, 10, 2):
            assert specs[i]['input_sha256'] == specs[i + 1]['input_sha256']
            assert specs[i]['golden_sha256'] == specs[i + 1]['golden_sha256']
        folder = root / 'cases' / specs[0]['id']
        raw = bytearray((folder / 'output/var_out.bin').read_bytes())
        raw[0] ^= 1
        (folder / 'output/var_out.bin').write_bytes(raw)
        assert verify(root / 'cases', specs[0])['mismatched'] == 1
        (folder / 'input/indices.bin').write_bytes(b'bad')
        try:
            validate_case(root / 'cases', specs[0])
            raise AssertionError('changed input accepted')
        except ValueError:
            pass
        csv_root = root / 'csv'
        csv_root.mkdir()
        report = csv_root / 'op_summary_0.csv'
        report.write_text('Op Name,Task Start Time(us),Task Duration(us),aiv_mte2_ratio\n'
                          'quant_update_scatter,30,5,0.3\nquant_update_scatter,10,100,0.1\n'
                          'quant_update_scatter,20,7,0.2\n', encoding='utf-8')
        summary = summarize(csv_root, 1, 2)
        assert summary['status'] == 'COMPLETE'
        assert summary['exports'][0]['task_duration_us']['median'] == 6
        assert summarize(csv_root, 1, 3)['status'] == 'INCOMPLETE'
        report.write_text('Op Name,Task Duration(us)\nquant_update_scatter,5\n', encoding='utf-8')
        assert summarize(csv_root, 0, 1)['status'] == 'INCOMPLETE'
        assert summarize(root / 'absent', 0, 1)['status'] == 'INCOMPLETE'
        (csv_root / 'PipeUtilization.csv').write_text('block_id,aiv_mte2_ratio\n0,0.8\n', encoding='utf-8')
        assert summarize(csv_root, 0, 1)['status'] == 'RAW_ONLY'
        (csv_root / 'PipeUtilization.csv').write_text('block_id,aiv_mte2_ratio\n', encoding='utf-8')
        assert summarize(csv_root, 0, 1)['status'] == 'INCOMPLETE'
        command = ['runner with space', '/case path', '0']
        op = profiler_command('op', 'msprof', command, root / 'op', '--launch-count', 60)
        assert op[-3:] == command and '--launch-count=60' in op
        app = profiler_command('application', 'msprof', command, root / 'app', '', 60)
        assert shlex.split(app[1].split('=', 1)[1]) == command
        assert not any(arg.startswith('--export') for arg in op + app)
        assert len(specs) == 26
        print(f'Q005 local support checks passed: {len(specs)} deterministic diagnostic cases; '
              'exact bytes, corruption/hash rejection, chronological warmup removal, '
              'missing/ambiguous export rejection.')
        print('NOT_RUN: CANN compile, NPU precision, profiler collection, pipeline timing, performance gain.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('generate')
    command.add_argument('--directory', type=Path, required=True)
    for action in ('verify', 'run'):
        command = sub.add_parser(action)
        command.add_argument('--directory', type=Path, required=True)
        command.add_argument('--case-id', action='append', default=[])
        if action == 'run':
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--runner', type=Path, required=True)
            command.add_argument('--device', type=int, default=0)
            command.add_argument('--warmup', type=int, default=10)
            command.add_argument('--repeat', type=int, default=50)
            command.add_argument('--rounds', type=int, default=3)
            command.add_argument('--msprof', default='msprof')
            command.add_argument('--profiler-mode', choices=('auto', 'op', 'application'), default='auto')
    command = sub.add_parser('summarize')
    command.add_argument('--directory', type=Path, required=True)
    command.add_argument('--warmup', type=int, default=10)
    command.add_argument('--repeat', type=int, default=50)
    sub.add_parser('self-test')
    args = parser.parse_args()
    if args.action == 'self-test':
        self_test()
        return 0
    if args.action == 'generate':
        print('Generated diagnostic cases:', len(generate(args.directory)))
        return 0
    if getattr(args, 'warmup', 0) < 0 or getattr(args, 'repeat', 1) < 1 or getattr(args, 'rounds', 1) < 1:
        parser.error('warmup >= 0, repeat >= 1, rounds >= 1 required')
    if args.action == 'run' and (args.device < 0 or args.device > 255 or args.warmup + args.repeat > 5000):
        parser.error('device must be 0..255 and warmup + repeat must be <= 5000')
    if args.action == 'summarize':
        result = summarize(args.directory, args.warmup, args.repeat)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] == 'COMPLETE' else 1
    cases = json.loads((args.directory / 'manifest.json').read_text(encoding='utf-8'))['cases']
    if args.case_id:
        unknown = set(args.case_id) - {case['id'] for case in cases}
        if unknown:
            parser.error('unknown case IDs: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case['id'] in args.case_id]
    if args.action == 'run':
        return run(args, cases)
    results = [verify(args.directory, case) for case in cases]
    print(json.dumps(results, indent=2))
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as error:
        print('Q005 ERROR:', error, file=sys.stderr)
        raise SystemExit(2)
