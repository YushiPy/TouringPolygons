"""Reproducible, sequential free-order solver comparisons with independent validation.

Example: python benchmarks/tpp.py free-order-ablation --suite FILE.bin
  --solver baseline=PATH --solver candidate=PATH --output results.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark_cases import read_encoded_cases
from unordered_runner import run_unordered_solver
from unordered_validation import validate_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, action='append', required=True)
    parser.add_argument('--solver', action='append', required=True, help='LABEL=EXECUTABLE')
    parser.add_argument('--solver-argument', action='append', default=[], help='LABEL=ARGUMENT')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--max-calls', type=int, default=10000000)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent instances; solver variants for one case remain sequential.')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse exact rows already present in --output and rerun unfinished solver/case pairs.')
    parser.add_argument('--relative-gap', type=float)
    parser.add_argument('--absolute-gap', type=float)
    parser.add_argument('--oracle-relative-gap', type=float)
    parser.add_argument('--case', type=int, action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quiet', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.stride < 1 or args.workers < 1 or args.max_calls < 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('Expected positive repeats, stride, and seconds.')
    solver_arguments = []
    for name in ('absolute_gap', 'relative_gap', 'oracle_relative_gap'):
        value = getattr(args, name)
        if value is not None:
            if not math.isfinite(value) or value < 0:
                parser.error(f'Expected a finite nonnegative {name}.')
            solver_arguments.extend(('--' + name.replace('_', '-'), str(value)))
    arguments_by_solver: dict[str, list[str]] = {}
    for spec in args.solver_argument:
        label, separator, argument = spec.partition('=')
        if not separator or not label or not argument:
            parser.error(f'Expected LABEL=ARGUMENT: {spec}')
        arguments_by_solver.setdefault(label, []).append(argument)
    solvers = []
    for spec in args.solver:
        label, separator, binary = spec.partition('=')
        if not separator or not label or not Path(binary).is_file():
            parser.error(f'Expected LABEL=EXISTING_EXECUTABLE: {spec}')
        path = Path(binary).resolve()
        solvers.append((label, path, hashlib.sha256(path.read_bytes()).hexdigest(), arguments_by_solver.pop(label, [])))
    if arguments_by_solver:
        parser.error(f'Arguments provided for unknown solvers: {", ".join(sorted(arguments_by_solver))}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'platform': platform.platform(), 'processor': platform.processor(),
        'seconds': args.seconds, 'max_calls': args.max_calls, 'repeats': args.repeats,
        'workers': args.workers,
        'solver_arguments': solver_arguments,
        'solvers': [(name, str(path), digest, arguments) for name, path, digest, arguments in solvers],
        'suites': [(str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.suite],
        'case': args.case, 'limit': args.limit, 'stride': args.stride,
        'timing': 'Solver variants for each case run sequentially; independent cases use the requested worker count. Solver seconds excludes process startup and independent validation.',
    }
    metadata_path = args.output.with_suffix('.meta.json')
    latest_rows: dict[tuple[str, int, str, int], dict] = {}
    if args.resume and args.output.exists():
        if not metadata_path.exists():
            parser.error(f'Cannot resume without campaign metadata: {metadata_path}')
        previous_metadata = json.loads(metadata_path.read_text())
        if json.dumps(previous_metadata, sort_keys=True) != json.dumps(metadata, sort_keys=True):
            parser.error('Cannot resume: campaign metadata differs from the existing output.')
        with args.output.open() as previous_output:
            for line in previous_output:
                try:
                    row = json.loads(line)
                    key = (row['suite'], int(row['case']), row['solver'], int(row['repeat']))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                latest_rows[key] = row
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    jobs = []
    for suite in args.suite:
        cases = read_encoded_cases(suite)
        cases = [c for c in cases if args.case is None or c.case_index in args.case]
        cases = cases[::args.stride][:args.limit]
        jobs.extend((suite, case) for case in cases)

    def run_case(suite, case):
        case_rows = []
        coordinates = struct.unpack_from('<dddd', case.data)
        for repeat in range(args.repeats):
            # Rotate execution order to reduce systematic temperature/order bias.
            offset = (case.case_index + repeat) % len(solvers)
            for label, binary, digest, specific_arguments in solvers[offset:] + solvers[:offset]:
                key = (str(suite), case.case_index, label, repeat)
                previous = latest_rows.get(key)
                if args.resume and previous is not None and previous.get('exact') is True:
                    continue
                row = {'suite': str(suite), 'case': case.case_index, 'sha256': case.digest,
                       'solver': label, 'binary_sha256': digest, 'repeat': repeat,
                       'polygons': case.polygon_count, 'workers': args.workers}
                began = time.perf_counter()
                try:
                    row.update(run_unordered_solver(binary, coordinates[:2], coordinates[2:],
                                                    case.polygons, args.max_calls, args.seconds,
                                                    [*solver_arguments, *specific_arguments]))
                    row['process_seconds'] = time.perf_counter() - began
                    try:
                        row['validation'] = validate_path(coordinates[:2], coordinates[2:],
                                                          case.polygons, row['path'], 1e-7)
                        row['valid'] = row['validation']['valid']
                    except ImportError as error:
                        row['validation'] = None
                        row['valid'] = None
                        row['validation_error'] = f'Independent path validation unavailable: {error}'
                    if row.get('valid') is True and abs(row['validation']['recomputed_length'] - row['upper_bound']) > 1e-7 * (1 + row['upper_bound']):
                        row['error'] = 'Reported objective disagrees with returned path.'
                    if row.get('lower_bound', -math.inf) > row.get('upper_bound', math.inf) + 1e-7:
                        row['error'] = 'Lower bound exceeds upper bound.'
                except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                    row['error'] = str(error)
                case_rows.append(row)
        return case_rows

    with args.output.open('a' if args.resume else 'w') as output, ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(jobs)))) as executor:
        futures = [executor.submit(run_case, suite, case) for suite, case in jobs]
        for future in as_completed(futures):
            for row in future.result():
                key = (row['suite'], int(row['case']), row['solver'], int(row['repeat']))
                latest_rows[key] = row
                output.write(json.dumps(row) + '\n')
                output.flush()
                if not args.quiet:
                    print(json.dumps({k: row.get(k) for k in
                                      ('solver', 'suite', 'case', 'repeat', 'seconds', 'initial_upper_bound', 'exact', 'calls', 'valid', 'error')}), flush=True)
    rows = list(latest_rows.values())
    if args.resume:
        temporary_output = args.output.with_suffix(args.output.suffix + '.tmp')
        with temporary_output.open('w') as output:
            for key in sorted(latest_rows):
                output.write(json.dumps(latest_rows[key]) + '\n')
        temporary_output.replace(args.output)
    for label, _, _, _ in solvers:
        selected = [r for r in rows if r['solver'] == label]
        times = [r['seconds'] for r in selected if 'seconds' in r]
        print(json.dumps({'solver': label, 'runs': len(selected),
                          'exact': sum(bool(r.get('exact')) for r in selected),
                          'invalid_or_error': sum(bool(r.get('error')) or r.get('valid') is False for r in selected),
                          'validation_unavailable': sum(r.get('valid') is None for r in selected),
                          'seconds': sum(times), 'median_seconds': statistics.median(times) if times else None}))
    return int(any(r.get('error') or r.get('valid') is False for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
