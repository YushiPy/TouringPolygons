from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError


MODULE_PATH = Path(__file__).resolve().parents[1] / "main.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dashboard_main", MODULE_PATH)
assert SPEC is not None
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


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

	def test_manual_case_round_trip_through_binary_format(self) -> None:
		case = dashboard.manual_case_from_request(dashboard.ManualCaseRequest(
			start=(0.0, 0.0),
			target=(2.0, 0.0),
			polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
		))
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "manual.bin"

			dashboard.write_binary_cases(path, [case])

			self.assertEqual(dashboard.read_binary_cases(path, limit=10), [case])

	def test_instance_previews_are_generated_lazily(self) -> None:
		case = dashboard.manual_case_from_request(dashboard.ManualCaseRequest(
			start=(0.0, 0.0),
			target=(2.0, 0.0),
			polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
		))
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


if __name__ == "__main__":
	unittest.main()
