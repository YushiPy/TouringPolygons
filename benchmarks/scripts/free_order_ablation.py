"""Reproducible, sequential free-order solver comparisons with independent validation.

Example: python benchmarks/scripts/free_order_ablation.py --suite FILE.bin
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
    if args.repeats < 1 or args.stride < 1 or args.max_calls < 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
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
        'solver_arguments': solver_arguments,
        'solvers': [(name, str(path), digest, arguments) for name, path, digest, arguments in solvers],
        'suites': [(str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.suite],
        'case': args.case, 'limit': args.limit, 'stride': args.stride,
        'timing': 'Sequential processes; solver seconds excludes process startup and independent validation.',
    }
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    with args.output.open('w') as output:
        for suite in args.suite:
            cases = read_encoded_cases(suite)
            cases = [c for c in cases if args.case is None or c.case_index in args.case]
            cases = cases[::args.stride][:args.limit]
            for case in cases:
                coordinates = struct.unpack_from('<dddd', case.data)
                for repeat in range(args.repeats):
                    # Rotate execution order to reduce systematic temperature/order bias.
                    offset = (case.case_index + repeat) % len(solvers)
                    for label, binary, digest, specific_arguments in solvers[offset:] + solvers[:offset]:
                        row = {'suite': str(suite), 'case': case.case_index, 'sha256': case.digest,
                               'solver': label, 'binary_sha256': digest, 'repeat': repeat,
                               'polygons': case.polygon_count}
                        began = time.perf_counter()
                        try:
                            row.update(run_unordered_solver(binary, coordinates[:2], coordinates[2:],
                                                            case.polygons, args.max_calls, args.seconds,
                                                            [*solver_arguments, *specific_arguments]))
                            row['process_seconds'] = time.perf_counter() - began
                            row['validation'] = validate_path(coordinates[:2], coordinates[2:],
                                                              case.polygons, row['path'], 1e-7)
                            row['valid'] = row['validation']['valid']
                            if abs(row['validation']['recomputed_length'] - row['upper_bound']) > 1e-7 * (1 + row['upper_bound']):
                                row['error'] = 'Reported objective disagrees with returned path.'
                            if row['lower_bound'] > row['upper_bound'] + 1e-7:
                                row['error'] = 'Lower bound exceeds upper bound.'
                        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                            row['error'] = str(error)
                        rows.append(row)
                        output.write(json.dumps(row) + '\n')
                        output.flush()
                        if not args.quiet:
                            print(json.dumps({k: row.get(k) for k in
                                              ('solver', 'suite', 'case', 'repeat', 'seconds', 'exact', 'calls', 'valid', 'error')}), flush=True)
    for label, _, _, _ in solvers:
        selected = [r for r in rows if r['solver'] == label]
        times = [r['seconds'] for r in selected if 'seconds' in r]
        print(json.dumps({'solver': label, 'runs': len(selected),
                          'exact': sum(bool(r.get('exact')) for r in selected),
                          'invalid_or_error': sum(bool(r.get('error')) or not r.get('valid', False) for r in selected),
                          'seconds': sum(times), 'median_seconds': statistics.median(times) if times else None}))
    return int(any(r.get('error') or not r.get('valid', False) for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
