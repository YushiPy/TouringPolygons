from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks/_internal'))
import tspn_benchmark as benchmark
import tspn_diagnostics as diagnostics


class TspnCampaignTests(unittest.TestCase):
    def test_stratified_sampling_is_seeded_nested_and_uses_actual_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'cases.zip'
            with zipfile.ZipFile(archive, 'w') as out:
                for kind, meta in [('osm', {'geo_information': {}}), ('random', {'source': 'random'}),
                                   ('tess', {'source': 'public_instance_set'})]:
                    for i in range(10):
                        # Misleading filename n060 must not determine the band.
                        out.writestr(f'{kind}_n060_{i}.json', json.dumps({'meta': meta,
                            'polygons': ['POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'] * 5}))
            small, counts = benchmark.select_socg_inputs(archive, 1, 42)
            large, _ = benchmark.select_socg_inputs(archive, 3, 42)
            self.assertEqual(small, large[:3])
            self.assertEqual(large, benchmark.select_socg_inputs(archive, 3, 42)[0])
            self.assertNotEqual(large, benchmark.select_socg_inputs(archive, 3, 17)[0])
            self.assertEqual([r['source_class'] for r in small], ['OSM', 'random', 'tessellation'])
            self.assertTrue(all(r['size_band'] == '5-10' for r in large))
            self.assertEqual(counts['OSM:5-10'], 10)
            self.assertEqual(len({r['name'] for r in large}), 9)

    def test_resume_recovers_only_torn_final_record(self):
        row = dict(name='x', sha256='abc', solver='ours', repeat=0)
        key = diagnostics.row_key(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            complete = json.dumps(row).encode() + b'\n'
            path.write_bytes(complete + b'{"name":')
            self.assertEqual(diagnostics.load_rows(path, {key}), [row])
            self.assertEqual(path.read_bytes(), complete)
            self.assertTrue(path.with_suffix('.interrupted-tail').exists())
            path.write_bytes(complete * 2)
            with self.assertRaises(ValueError): diagnostics.load_rows(path, {key})
            path.write_bytes(b'bad\n' + complete)
            with self.assertRaises(ValueError): diagnostics.load_rows(path, {key})
            path.write_bytes(complete[:-1])
            self.assertEqual(diagnostics.load_rows(path, {key}), [row])
            self.assertEqual(path.read_bytes(), complete)
            with self.assertRaises(ValueError): diagnostics.load_rows(path, set())

    def test_diagnostics_preserve_missing_fields_and_worker_semantics(self):
        row = {'solver': 'ours', 'seconds': 2, 'lower_bound': 5, 'upper_bound': 10,
               'incumbent_length': 20, 'oracle_profiled_calls': 10,
               'profile': {'convex_oracle_seconds': 3, 'search_seconds': 4},
               'oracle_fallback_call_seconds': 1, 'fallback_calls': 2}
        data = diagnostics.metrics(row, {'ours_threads': 2})
        self.assertEqual(data['oracle_work_fraction'], .75)
        self.assertEqual(data['oracle_mean_call_seconds'], .3)
        self.assertEqual(data['initial_to_final_feasible_ratio'], 2)
        self.assertEqual(data['initial_to_lower_bound_ratio'], 4)
        self.assertEqual(data['final_gap_over_ub'], .5)
        self.assertIsNone(data['memo_hit_fraction'])
        self.assertIsNone(diagnostics.metrics({'solver': 'fekete'}, {})['fallback_call_fraction'])

    def test_reports_exclude_censored_invalid_and_incompatible_speedups(self):
        inputs = {'instances': [{'name': 'x', 'polygons': [[]], 'source_class': 'OSM', 'size_band': '5-10'}]}
        config = {'repetitions': 1, 'ours_threads': 1, 'fekete_threads': 1}
        good = dict(name='x', repeat=0, seconds=1., lower_bound=10., upper_bound=10.,
                    validation={'valid': True}, gap_closed=True)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            def summary(rows):
                diagnostics.write_reports(out, inputs, config, rows)
                return json.loads((out / 'summary.json').read_text())[0]
            rows = [{**good, 'solver': 'ours'}, {**good, 'solver': 'fekete', 'seconds': 2}]
            self.assertEqual(summary(rows)['matched_speedup'], 2)
            rows[1].update(lower_bound=11., upper_bound=11.)
            self.assertIsNone(summary(rows)['matched_speedup'])
            rows[1] = dict(name='x', repeat=0, solver='fekete', timeout=True, error='timeout',
                           gap_closed=False, validation={'valid': False})
            result = summary(rows)
            self.assertIsNone(result['matched_speedup'])
            self.assertIsNone(result['fekete']['median_seconds'])
            self.assertEqual(result['fekete']['timeouts'], 1)
            self.assertEqual(result['fekete']['errors'], 0)
            self.assertEqual(json.loads((out / 'progress.json').read_text())['completed_runs'], 2)

    def test_runner_interrupt_resume_and_configuration_guards(self):
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = dict(seconds=.01, path=[[0, 0], [0, 0]], lower_bound=0., upper_bound=0., calls=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'vendor'
            (source / 'tspn_core').mkdir(parents=True)
            (source / 'tspn_core/CMakeLists.txt').write_text('')
            inputs = root / 'inputs.json'
            inputs.write_text(json.dumps({'formulation': 'TSPN', 'instances': [{'name': 'x', 'polygons': [polygon]}]}))
            binary = root / 'solver'
            binary.write_bytes(b'original binary')
            output = root / 'campaign'
            args = ['--inputs', str(inputs), '--output', str(output), '--fekete-source', str(source),
                    '--skip-build', '--ours-binary', str(binary), '--fekete-binary', str(binary),
                    '--seconds', '1', '--repetitions', '1', '--external-timeout', '2']
            with patch.object(benchmark.platform, 'platform', return_value='test-platform'), \
                 patch.object(benchmark.subprocess, 'check_output', return_value='test-provenance\n'), \
                 patch.object(benchmark, 'run_unordered_solver', return_value=result.copy()) as native, \
                 patch.object(benchmark.subprocess, 'run', side_effect=KeyboardInterrupt) as external, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.main(args), 130)
                self.assertEqual(len((output / 'raw.jsonl').read_text().splitlines()), 1)
                native.reset_mock(); external.reset_mock()
                external.side_effect = None
                external.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(result))
                self.assertEqual(benchmark.main(args + ['--resume']), 0)
                native.assert_not_called()
                external.assert_called_once()
                native.reset_mock(); external.reset_mock()
                self.assertEqual(benchmark.main(args + ['--resume']), 0)
                native.assert_not_called(); external.assert_not_called()
                with self.assertRaises(SystemExit): benchmark.main(args + ['--resume', '--seconds', '2'])
                binary.write_bytes(b'changed binary')
                with self.assertRaises(SystemExit): benchmark.main(args + ['--resume'])
                self.assertEqual(benchmark.main(['--output', str(output), '--report-only']), 0)
                self.assertEqual(json.loads((output / 'progress.json').read_text())['status'], 'complete')

    def test_max_calls_budget_and_legacy_resume_normalization(self):
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = dict(seconds=.01, path=[[0, 0], [0, 0]], lower_bound=0., upper_bound=0., calls=7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'vendor'
            (source / 'tspn_core').mkdir(parents=True)
            (source / 'tspn_core/CMakeLists.txt').write_text('')
            inputs = root / 'inputs.json'
            inputs.write_text(json.dumps({'formulation': 'TSPN', 'instances': [{'name': 'x', 'polygons': [polygon]}]}))
            binary = root / 'solver'
            binary.write_bytes(b'fixed test binary')
            output = root / 'campaign'
            args = ['--solver', 'ours', '--inputs', str(inputs), '--output', str(output),
                    '--fekete-source', str(source), '--skip-build', '--ours-binary', str(binary),
                    '--seconds', '1', '--repetitions', '1', '--external-timeout', '2', '--max-calls', '7']
            with patch.object(benchmark.platform, 'platform', return_value='test-platform'), \
                 patch.object(benchmark.subprocess, 'check_output', return_value='test-provenance\n'), \
                 patch.object(benchmark, 'run_unordered_solver', return_value=result.copy()) as native, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.main(args), 0)
                native.assert_called_once()
                self.assertEqual(native.call_args.args[4], 7)
                config = json.loads((output / 'config.json').read_text())
                self.assertEqual(config['max_calls'], 7)
                self.assertEqual(config['run_options']['max_calls'], 7)

                invalid_cases = [(args + ['--max-calls', '-1'], '--max-calls cannot be negative')]
                for solver in ('both', 'fekete'):
                    invalid = list(args)
                    invalid[invalid.index('ours')] = solver
                    invalid_cases.append((invalid, 'a nondefault --max-calls requires --solver ours'))
                for invalid, message in invalid_cases:
                    stderr = io.StringIO()
                    with self.assertRaises(SystemExit) as error:
                        with contextlib.redirect_stderr(stderr): benchmark.main(invalid)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())

                stderr = io.StringIO()
                with self.assertRaises(SystemExit) as error:
                    with contextlib.redirect_stderr(stderr):
                        benchmark.main(args + ['--resume', '--max-calls', '8'])
                self.assertEqual(error.exception.code, 2)
                self.assertIn('--resume options differ', stderr.getvalue())

                # Simulate a pre-feature campaign: omitting run_options.max_calls
                # is equivalent to the historical default, not an option change.
                config['max_calls'] = 10**8
                config['run_options'].pop('max_calls')
                (output / 'config.json').write_text(json.dumps(config))
                native.reset_mock()
                default_args = [arg for arg in args if arg not in ('--max-calls', '7')]
                self.assertEqual(benchmark.main(default_args + ['--resume']), 0)
                native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
