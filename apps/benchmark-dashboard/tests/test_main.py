from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

MODULE_PATH = Path(__file__).resolve().parents[1] / "main.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dashboard_main", MODULE_PATH)
assert SPEC is not None
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)
import dashboard_reports  # noqa: E402


class DashboardMainTests(unittest.TestCase):
    def test_campaign_path_rejects_path_traversal(self) -> None:
        with self.assertRaises(HTTPException):
            dashboard.campaign_path("../bad")

    def test_create_osm_request_validates_ranges(self) -> None:
        with self.assertRaises(ValidationError):
            dashboard.CreateOsmRequest(
                name="city",
                pbf_path="/tmp/city.osm.pbf",
                instances=0,
            )

    def test_compare_request_requires_positive_thread_count(self) -> None:
        with self.assertRaises(ValidationError):
            dashboard.CompareSolversRequest(
                name="smoke",
                solvers=["binary"],
                threads=0,
            )

    def test_csv_cache_invalidates_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            path.write_text("name,value\nfirst,1\n")

            self.assertEqual(dashboard.read_csv_rows(path), [{"name": "first", "value": "1"}])

            path.write_text("name,value\nsecond,2\n")

            self.assertEqual(dashboard.read_csv_rows(path), [{"name": "second", "value": "2"}])

    def test_json_cache_invalidates_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text('{"value": "first"}\n')
            self.assertEqual(dashboard.read_json(path)["value"], "first")
            path.write_text('{"value": "second"}\n')
            self.assertEqual(dashboard.read_json(path)["value"], "second")

    def test_binary_offset_cache_is_bounded(self) -> None:
        dashboard._binary_offset_cache.clear()
        case = dashboard.manual_case_from_request(dashboard.ManualCaseRequest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(dashboard.FILE_CACHE_LIMIT + 5):
                path = root / f"{index}.bin"
                dashboard.write_binary_cases(path, [case])
                dashboard.read_binary_case(path, 0)
            self.assertLessEqual(len(dashboard._binary_offset_cache), dashboard.FILE_CACHE_LIMIT)

    def test_report_helpers_handle_missing_and_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(dashboard_reports.summary_result_rows(path / "missing.md"), [])
            self.assertIsNone(dashboard_reports.latest_comparison_path(path))
            self.assertEqual(dashboard_reports.parse_markdown_tables("## Summary\n"), [])
            partial = path / "partial.csv"
            partial.write_text("case_index;total_seconds\n0;1.2\n1;\n")
            self.assertEqual(dashboard_reports.read_result_rows(partial)[1]["case_index"], "1")

    def test_file_caches_are_bounded(self) -> None:
        dashboard._json_cache.clear()
        dashboard._csv_cache.clear()
        dashboard._binary_offset_cache.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(dashboard.FILE_CACHE_LIMIT + 5):
                path = root / f"{index}.json"
                path.write_text(f'{{"index": {index}}}\n')
                self.assertEqual(dashboard.read_json(path)["index"], index)

            self.assertLessEqual(len(dashboard._json_cache), dashboard.FILE_CACHE_LIMIT)
            self.assertNotIn(root / "0.json", dashboard._json_cache)
            self.assertIn(root / f"{dashboard.FILE_CACHE_LIMIT + 4}.json", dashboard._json_cache)

    def test_campaign_summary_does_not_refresh_previews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "campaign.json").write_text("""{
				"name": "summary",
				"type": "manual",
				"inputs": [],
				"previews": {"all": "previews/all.svg"},
				"instance_previews": []
			}
			""")

            with patch.object(dashboard, "refresh_stale_previews", side_effect=AssertionError):
                self.assertEqual(dashboard.campaign_summary(path)["name"], "summary")

    def test_completed_instance_count_cache_invalidates_when_result_changes(
        self,
    ) -> None:
        dashboard_reports._completed_count_cache.clear()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            results = path / "results"
            results.mkdir()
            (results / "run-index.csv").write_text("status,csv_output\ncompleted,results/run.csv\n")
            (results / "run.csv").write_text("case_index\n0\n")

            self.assertEqual(dashboard.completed_instance_count(path), 1)
            self.assertEqual(dashboard.completed_instance_count(path), 1)

            (results / "run.csv").write_text("case_index\n0\n1\n")

            self.assertEqual(dashboard.completed_instance_count(path), 2)

    def test_manual_case_round_trip_through_binary_format(self) -> None:
        case = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(0.0, 0.0),
                target=(2.0, 0.0),
                polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.bin"

            dashboard.write_binary_cases(path, [case])

            self.assertEqual(dashboard.read_binary_cases(path, limit=10), [case])

    def test_manual_binary_cache_is_rebuilt_from_editable_cases(self) -> None:
        case = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(0.0, 0.0),
                target=(2.0, 0.0),
                polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dashboard.create_manual_campaign_data(path)
            dashboard.write_manual_cases(path, [case])
            input_path = dashboard.manual_input_path(path)
            input_path.unlink()

            self.assertEqual(dashboard.ensure_manual_binary_cache(path), input_path)
            self.assertEqual(dashboard.read_binary_cases(input_path, limit=1), [case])

    def test_binary_case_reader_counts_and_respects_limit(self) -> None:
        cases = [
            dashboard.manual_case_from_request(
                dashboard.ManualCaseRequest(
                    start=(0.0, 0.0),
                    target=(1.0, 0.0),
                    polygons=[[(0.0, 1.0), (1.0, 1.0), (0.5, 2.0)]],
                )
            ),
            dashboard.manual_case_from_request(
                dashboard.ManualCaseRequest(
                    start=(2.0, 0.0),
                    target=(3.0, 0.0),
                    polygons=[[(2.0, 1.0), (3.0, 1.0), (2.5, 2.0)]],
                )
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.bin"
            dashboard.write_binary_cases(path, cases)

            self.assertEqual(dashboard.binary_case_count(path), 2)
            self.assertEqual(dashboard.read_binary_cases(path, limit=1), cases[:1])

    def test_binary_case_offset_cache_invalidates_when_file_changes(self) -> None:
        first = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(0.0, 0.0),
                target=(1.0, 0.0),
                polygons=[[(0.0, 1.0), (1.0, 1.0), (0.5, 2.0)]],
            )
        )
        second = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(2.0, 0.0),
                target=(3.0, 0.0),
                polygons=[[(2.0, 1.0), (3.0, 1.0), (2.5, 2.0)]],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.bin"
            dashboard.write_binary_cases(path, [first])
            self.assertEqual(dashboard.binary_case_count(path), 1)

            dashboard.write_binary_cases(path, [first, second])
            self.assertEqual(dashboard.binary_case_count(path), 2)
            self.assertEqual(dashboard.read_binary_case(path, 1), second)

    def test_campaign_case_lookup_handles_multiple_input_files(self) -> None:
        first = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(0.0, 0.0),
                target=(1.0, 0.0),
                polygons=[[(0.0, 1.0), (1.0, 1.0), (0.5, 2.0)]],
            )
        )
        second = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(2.0, 0.0),
                target=(3.0, 0.0),
                polygons=[[(2.0, 1.0), (3.0, 1.0), (2.5, 2.0)]],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dashboard.write_binary_cases(path / "inputs/a.bin", [first])
            dashboard.write_binary_cases(path / "inputs/b.bin", [second])
            data = {
                "inputs": [
                    {"file": "inputs/a.bin", "instances": 1},
                    {"file": "inputs/b.bin", "instances": 1},
                ]
            }

            self.assertEqual(dashboard.read_campaign_case(path, data, 0), first)
            self.assertEqual(dashboard.read_campaign_case(path, data, 1), second)
            self.assertIsNone(dashboard.read_campaign_case(path, data, 2))

    def test_instance_previews_are_generated_lazily(self) -> None:
        case = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(0.0, 0.0),
                target=(2.0, 0.0),
                polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dashboard.write_binary_cases(path / "inputs/manual.bin", [case])
            previews, instance_previews = dashboard.write_imported_previews(path, [case])
            data = {
                "inputs": [{"file": "inputs/manual.bin", "instances": 1}],
                "preview": previews["all"],
                "previews": previews,
                "instance_previews": instance_previews,
            }

            instance_path = path / instance_previews[0]
            self.assertFalse(instance_path.exists())

            self.assertEqual(dashboard.ensure_instance_preview(path, data, 0), instance_path)
            self.assertTrue(instance_path.exists())

    def test_manual_autosave_invalidates_previews_without_rebuilding_them(self) -> None:
        case = dashboard.manual_case_from_request(
            dashboard.ManualCaseRequest(
                start=(0.0, 0.0),
                target=(2.0, 0.0),
                polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dashboard.create_manual_campaign_data(path)
            dashboard.rebuild_manual_campaign(path, [case])
            preview_path = path / "previews" / "all.svg"
            self.assertTrue(preview_path.exists())

            dashboard.rebuild_manual_campaign(path, [case], rebuild_previews=False)
            self.assertFalse(preview_path.exists())
            data = dashboard.read_json(path / "campaign.json")
            self.assertEqual(data["previews"], {})
            self.assertTrue(dashboard.campaign_previews_are_stale(path, data))

    def test_manual_case_resource_limits_are_enforced(self) -> None:
        too_many_polygons = dashboard.ManualCaseRequest(
            polygons=[[] for _ in range(dashboard.MAX_POLYGONS_PER_CASE + 1)],
        )
        with self.assertRaises(HTTPException) as context:
            dashboard.validate_manual_cases([too_many_polygons])
        self.assertEqual(context.exception.status_code, 413)

    def test_job_progress_snapshot_is_compact(self) -> None:
        job = dashboard.Job(id="job", command=["run"], campaign="campaign")
        snapshot = job.progress_snapshot()
        self.assertEqual(snapshot["id"], "job")
        self.assertEqual(snapshot["campaign"], "campaign")
        self.assertNotIn("cancel_requested", snapshot)


if __name__ == "__main__":
    unittest.main()
