"""Campaign persistence and descriptive diagnostics for the native TSPN runner."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def row_key(row):
    return row['name'], row['sha256'], row['solver'], row['repeat']


def load_rows(path, expected):
    """Recover only a torn final write; reject corruption and foreign/duplicate rows."""
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, seen, offset = [], set(), 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError('invalid completed JSONL record')
            path.with_suffix('.interrupted-tail').write_bytes(line)
            with path.open('r+b') as output:
                output.truncate(offset)
            break
        key = row_key(row)
        if key not in expected or key in seen:
            raise ValueError('unexpected or duplicate input/backend/repetition in raw.jsonl')
        rows.append(row)
        seen.add(key)
        offset += len(line)
    else:
        if data and not data.endswith(b'\n'):
            with path.open('ab') as output:
                output.write(b'\n')
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(numerator, denominator):
    return numerator / denominator if finite(numerator) and finite(denominator) and denominator > 0 else None


def median(values):
    values = [v for v in values if finite(v)]
    return statistics.median(values) if values else None


def flatten(value, prefix=''):
    """Retain every scalar metric; paths, traces and arrays remain in raw JSONL."""
    out = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            out.update(flatten(item, name + '.'))
        elif not isinstance(item, (list, tuple)):
            out[name] = item
    return out


def metrics(row, config):
    profile = row.get('profile', {})
    oracle = profile.get('convex_oracle_seconds', row.get('oracle_seconds'))
    calls = row.get('oracle_profiled_calls') if 'oracle_profiled_calls' in row else row.get('calls')
    lo, up = row.get('lower_bound'), row.get('upper_bound')
    initial = row.get('incumbent_length')
    workers = config.get('ours_threads', 1) if row['solver'] == 'ours' else config.get('fekete_threads', 1)
    seconds = row.get('seconds')
    result = {
        'oracle_seconds': oracle,
        'oracle_mean_call_seconds': ratio(oracle, calls),
        # For portfolios this is summed work / summed top-level worker time,
        # not an impossible >100% share of elapsed wall time.
        'oracle_work_fraction': ratio(oracle, sum(profile.get(k, 0) for k in
            ('preprocessing_seconds', 'initial_heuristic_seconds', 'search_seconds', 'finalization_seconds'))
            if profile else seconds),
        'fallback_call_fraction': ratio(row.get('fallback_calls'), calls),
        'fallback_call_work_fraction': ratio(row.get('oracle_fallback_call_seconds'), oracle),
        'max_call_work_fraction': ratio(row.get('oracle_max_call_seconds'), oracle),
        'memo_hit_fraction': ratio(row.get('cycle_memo_hits'), row.get('cycle_memo_queries')),
        'warm_contact_accept_fraction': ratio(row.get('cycle_initial_contact_accepts'), row.get('cycle_initial_contact_checks')),
        'calls_per_node': ratio(row.get('calls'), row.get('nodes', row.get('statistics', {}).get('num_explored'))),
        'final_gap_over_ub': ratio(up - lo, up) if finite(up) and finite(lo) and up >= lo else None,
        'initial_improvement_fraction': ratio(initial - up, initial) if finite(initial) and finite(up) else None,
        'initial_to_final_feasible_ratio': ratio(initial, up),
        'initial_to_lower_bound_ratio': ratio(initial, lo),
        'worker_seconds_capacity': seconds * workers if finite(seconds) else None,
        'objective_length_difference': (row['validation']['recomputed_length'] - up)
            if finite(row.get('validation', {}).get('recomputed_length')) and finite(up) else None,
    }
    if lo == up == 0:
        result['final_gap_over_ub'] = 0
    for key in ('calls','nodes','incumbent_updates','best_updates','peak_queue',
                'insertion_branches','decomposition_branches','children_generated','bound_prunes',
                'oracle_max_call_seconds','cycle_memo_hits','portfolio_join_seconds'):
        result[key]=row.get(key)
    for key in ('preprocessing_seconds','initial_heuristic_seconds','search_seconds','finalization_seconds',
                'decomposition_seconds','visit_check_seconds','search_maintenance_seconds',
                'cycle_construction_seconds','cycle_certification_seconds','cycle_rational_recovery_seconds'):
        result[key]=profile.get(key)
    counts, times = row.get('oracle_call_histogram'), row.get('oracle_seconds_histogram')
    labels = ('le_10us', '10_100us', '100us_1ms', '1_10ms', '10_100ms', '100ms_1s', 'gt_1s')
    if counts and times:
        for label, count, elapsed in zip(labels, counts, times):
            result['oracle_bucket_calls_' + label] = count
            result['oracle_bucket_seconds_' + label] = elapsed
    return result


def termination_reason(row):
    """Return an explicit solver exit reason, keeping wrapper failures distinct."""
    if row.get('timeout'):
        return 'process_timeout'
    if row.get('error'):
        return 'error'
    if row.get('solver') == 'ours':
        native = row.get('termination')
        return {
            'optimal': 'gap_criterion',
            'time_limit': 'time_limit',
            'call_limit': 'call_limit',
            'numerical_limit': 'numerical_limit',
            'portfolio_stopped': 'portfolio_stopped',
            'memory_limit': 'memory_limit',
        }.get(native, 'unknown')
    reason = row.get('termination_reason')
    return reason if reason in {'frontier_exhausted', 'time_limit', 'gap_criterion'} else 'unknown'


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_reports(output, inputs, config, rows, status='complete'):
    repetitions = config['repetitions']
    solvers = tuple(config.get('solvers', ('ours', 'fekete')))
    enriched = [{**r, 'diagnostics': metrics(r, config)} for r in rows]
    write_csv(output / 'runs.csv', [flatten(r) for r in enriched])
    summaries = []
    for case in inputs['instances']:
        group = {b: [r for r in enriched if r['name'] == case['name'] and r['solver'] == b] for b in solvers}
        summary = {'name': case['name'], 'k': len(case['polygons']),
                   'source_class': case.get('source_class'), 'size_band': case.get('size_band')}
        for backend, runs in group.items():
            summary[backend] = {
                'runs': len(runs), 'median_seconds': median(r.get('seconds') for r in runs),
                'min_seconds': min((r['seconds'] for r in runs if finite(r.get('seconds'))), default=None),
                'max_seconds': max((r['seconds'] for r in runs if finite(r.get('seconds'))), default=None),
                'valid_runs': sum(r['validation']['valid'] for r in runs),
                'gap_closed_runs': sum(r['gap_closed'] for r in runs),
                'errors': sum('error' in r and not r.get('timeout') for r in runs),
                'timeouts': sum(bool(r.get('timeout')) for r in runs),
                'open_gap_runs': sum(not r['gap_closed'] and 'error' not in r for r in runs),
                'median_calls': median(r.get('calls') for r in runs),
                'median_diagnostics': {k: median(r['diagnostics'].get(k) for r in runs)
                    for k in dict.fromkeys(k for r in runs for k in r['diagnostics'])},
                'termination_reasons': {reason: sum(termination_reason(r) == reason for r in runs)
                    for reason in ('frontier_exhausted', 'time_limit', 'gap_criterion',
                        'call_limit', 'numerical_limit', 'portfolio_stopped', 'memory_limit', 'process_timeout', 'error', 'unknown')},
            }
        bounded = [r for runs in group.values() for r in runs if r['validation']['valid'] and
                   finite(r.get('upper_bound')) and finite(r.get('lower_bound')) and 'error' not in r]
        summary['objective_spread'] = (max(r['upper_bound'] for r in bounded) - min(r['upper_bound'] for r in bounded)) if bounded else None
        separation = max(0, max(r['lower_bound'] for r in bounded) - min(r['upper_bound'] for r in bounded)) if bounded else None
        summary['interval_separation'] = separation
        # A disjoint pair of reported intervals is evidence to investigate,
        # never a successful matched speed comparison.
        matched = (len(solvers) == 2 and separation == 0 and all(len(runs) == repetitions and all(
            r['validation']['valid'] and r['gap_closed'] and 'error' not in r and finite(r.get('seconds'))
            for r in runs) for runs in group.values()))
        summary['matched_speedup'] = ratio(summary['fekete']['median_seconds'], summary['ours']['median_seconds']) if matched else None
        summaries.append(summary)
    atomic_json(output / 'summary.json', summaries)
    write_csv(output / 'summary.csv', [flatten(r) for r in summaries])
    strata = []
    for kind, band in dict.fromkeys((r['source_class'], r['size_band']) for r in summaries):
        cases = [r for r in summaries if (r['source_class'], r['size_band']) == (kind, band)]
        ratios = [r['matched_speedup'] for r in cases if r['matched_speedup'] is not None]
        item = {'source_class': kind, 'size_band': band, 'cases': len(cases), 'matched_cases': len(ratios),
                'ours_faster': sum(v > 1 for v in ratios), 'median_speedup': median(ratios),
                'mean_speedup': statistics.mean(ratios) if ratios else None}
        for backend in solvers:
            item[backend] = {k: sum(r[backend][k] for r in cases) for k in
                             ('runs', 'valid_runs', 'gap_closed_runs', 'errors', 'timeouts', 'open_gap_runs')}
            item[backend]['termination_reasons'] = {reason: sum(
                r[backend]['termination_reasons'][reason] for r in cases)
                for reason in ('frontier_exhausted', 'time_limit', 'gap_criterion',
                    'call_limit', 'numerical_limit', 'portfolio_stopped', 'memory_limit', 'process_timeout', 'error', 'unknown')}
            # Median of case medians: no overweighting a partly repeated case.
            item[backend]['diagnostics'] = {k: median(r[backend]['median_diagnostics'].get(k) for r in cases)
                for k in dict.fromkeys(k for r in cases for k in r[backend]['median_diagnostics'])}
        strata.append(item)
    atomic_json(output / 'strata.json', strata)
    write_csv(output / 'strata.csv', [flatten(r) for r in strata])
    atomic_json(output / 'progress.json', {'status': status, 'completed_runs': len(rows),
        'planned_runs': len(inputs['instances']) * repetitions * len(solvers),
        'process_timeouts': sum(bool(r.get('timeout')) for r in rows)})
    def show(value, digits=3):
        return f'{value:.{digits}f}' if finite(value) else '—'
    lines = ['# TSPN benchmark', '', f'Status: {status}; {len(rows)} completed records. Solvers: {", ".join(solvers)}.']
    if len(solvers) == 2:
     lines += ['',
        'Speedups require all repetitions to have validated tours, close the requested gap and have overlapping reported intervals. '
        'Process timeouts are censored; numerical/time limits are not optimality proofs. Fekete bounds are numerical. '
        'See config.json for the formulation and existing B&B/validation tolerances.', '',
        '| Class | Polygons | Cases | Matched | Ours faster | Median speedup | Valid O/F | Closed O/F | Open O/F | Timeouts O/F | Errors O/F | Fekete frontier / time / gap / unknown |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
     for r in strata:
        a, b = r['ours'], r['fekete']
        lines.append(f'| {r["source_class"]} | {r["size_band"]} | {r["cases"]} | {r["matched_cases"]} | '
                     f'{r["ours_faster"]} | {show(r["median_speedup"])} | {a["valid_runs"]}/{b["valid_runs"]} | {a["gap_closed_runs"]}/{b["gap_closed_runs"]} | {a["open_gap_runs"]}/{b["open_gap_runs"]} | {a["timeouts"]}/{b["timeouts"]} | {a["errors"]}/{b["errors"]} | '
                     f'{b["termination_reasons"]["frontier_exhausted"]}/{b["termination_reasons"]["time_limit"]}/{b["termination_reasons"]["gap_criterion"]}/{b["termination_reasons"]["unknown"]} |')
    else:
     lines += ['',
        'Process timeouts are censored; numerical/time limits are not optimality proofs. '
        'See config.json for the formulation and existing B&B/validation tolerances.', '',
        '| Class | Polygons | Cases | Valid | Closed | Open | Timeouts | Errors |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
     backend = solvers[0]
     for r in strata:
        a = r[backend]
        lines.append(f'| {r["source_class"]} | {r["size_band"]} | {r["cases"]} | {a["valid_runs"]} | {a["gap_closed_runs"]} | {a["open_gap_runs"]} | {a["timeouts"]} | {a["errors"]} |')
    if 'ours' in solvers:
     lines += ['', '## Native bottlenecks by class and size', '',
        '| Class | Polygons | Oracle work % | Mean call ms | Max call ms | Fallback call work % | Cycle construction / certification / rational recovery ms | Final gap % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
     for r in strata:
        d = r['ours']['diagnostics']
        def scaled(key, scale):
            value = d.get(key)
            return show(value * scale if finite(value) else None)
        max_call=d.get('oracle_max_call_seconds')
        lines.append(f'| {r["source_class"]} | {r["size_band"]} | {scaled("oracle_work_fraction", 100)} | '
                     f'{scaled("oracle_mean_call_seconds", 1000)} | {show(max_call * 1000 if finite(max_call) else None)} | '
                     f'{scaled("fallback_call_work_fraction", 100)} | '
                     f'{show((d.get("cycle_construction_seconds") or 0) * 1000) if finite(d.get("cycle_construction_seconds")) else "—"} / '
                     f'{show((d.get("cycle_certification_seconds") or 0) * 1000) if finite(d.get("cycle_certification_seconds")) else "—"} / '
                     f'{show((d.get("cycle_rational_recovery_seconds") or 0) * 1000) if finite(d.get("cycle_rational_recovery_seconds")) else "—"} | '
                     f'{scaled("final_gap_over_ub", 100)} |')
    lines += ['', 'Bottleneck cells are medians of per-instance medians; the max-call column summarizes per-run maxima, not the global maximum.',
        f'Reported interval disagreements: {sum(bool(r["interval_separation"]) for r in summaries)}; inspect summary.csv.', '', 'runs.csv retains scalar counters, phase timings, incumbent updates, queue size, branching, and per-call histogram buckets. '
        'summary.csv and strata.csv summarize instances and strata; raw.jsonl retains paths and all original fields.', '',
        'Oracle histogram buckets are exclusive: ≤10µs, (10,100]µs, (0.1,1]ms, (1,10]ms, (10,100]ms, (0.1,1]s, >1s. '
        'They measure completed B&B oracle requests, including memo hits, excluding initial polishing and in-flight calls. '
        'Fallback-call work includes the entire call that used recovery, not recovery alone. '
        'Cycle construction, certification, and rational recovery times are the exclusive cycle phase counters when available; '
        'certification includes checks performed during recovery, while rational recovery excludes certification. Fallback-call work '
        'remains the full oracle call that used recovery. New profile times include cooperatively interrupted calls; hard-killed '
        'processes remain censored. Older raw records without these fields are unavailable, not zero. Missing external metrics are unavailable, not zero.', '',
        'Initial-to-final feasible ratio measures incumbent improvement, not an approximation guarantee when the gap is open. '
        'Initial/LB bounds the approximation factor only under the reported bound/feasibility contract. '
        'Portfolio work sums both workers and is divided by summed worker phase time; never compare its work counters as single-core elapsed time.', '',
        'Profile counters include completed calls and cooperatively interrupted calls; a hard process timeout has no recovered partial counters or tour. '
        'See process_seconds versus seconds for process startup/overrun; validation_seconds is outside solver timing.']
    (output / 'analysis.md').write_text('\n'.join(lines) + '\n')
