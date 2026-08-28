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


if __name__ == "__main__":
	unittest.main()
