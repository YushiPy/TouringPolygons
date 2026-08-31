from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "benchmarks/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


normalizer = load_module(
    "normalize_polygon_orientation",
    REPO_ROOT / "benchmarks/scripts/normalize_polygon_orientation.py",
)
run_generated = load_module("run_generated", REPO_ROOT / "benchmarks/scripts/run_generated.py")
convert_instances = load_module("convert_instances", REPO_ROOT / "benchmarks/scripts/convert_instances.py")


class BenchmarkToolTests(unittest.TestCase):
    def test_binary_orientation_normalizer_preserves_non_polygon_payload(self) -> None:
        clockwise_square = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        ccw_triangle = [(2.0, 0.0), (3.0, 0.0), (2.5, 1.0)]
        cases = [
            normalizer.BinaryCase(
                start=(-1.0, 0.5),
                target=(4.0, 0.5),
                polygons=[clockwise_square, ccw_triangle],
                solution=[(-1.0, 0.5), (4.0, 0.5)],
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.bin"
            normalizer.write_binary_cases(path, cases)

            dry_report = normalizer.normalize_binary(path, write=False)
            self.assertEqual(dry_report.reversed_polygons, 1)

            write_report = normalizer.normalize_binary(path, write=True)
            normalized = normalizer.read_binary_cases(path)
            final_report = normalizer.normalize_binary(path, write=False)

            self.assertEqual(write_report.reversed_polygons, 1)
            self.assertEqual(final_report.reversed_polygons, 0)
            self.assertEqual(normalized[0].start, cases[0].start)
            self.assertEqual(normalized[0].target, cases[0].target)
            self.assertEqual(normalized[0].solution, cases[0].solution)
            self.assertGreater(normalizer.signed_area2(normalized[0].polygons[0]), 0.0)
            self.assertEqual(normalized[0].polygons[1], ccw_triangle)

    def test_manual_cases_orientation_normalizer_preserves_case_metadata(self) -> None:
        data = {
            "schema_version": 1,
            "cases": [
                {
                    "name": "clockwise",
                    "generated": True,
                    "start": [0.0, 0.0],
                    "target": [2.0, 0.0],
                    "polygons": [[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
                },
                {
                    "name": "counter-clockwise",
                    "start": [0.0, 0.0],
                    "target": [2.0, 0.0],
                    "polygons": [[[2.0, 0.0], [3.0, 0.0], [2.5, 1.0]]],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual-cases.json"
            path.write_text(json.dumps(data) + "\n")

            dry_report = normalizer.normalize_manual_cases(path, write=False)
            self.assertEqual(dry_report.reversed_polygons, 1)

            normalizer.normalize_manual_cases(path, write=True)
            normalized = json.loads(path.read_text())
            final_report = normalizer.normalize_manual_cases(path, write=False)

            self.assertEqual(final_report.reversed_polygons, 0)
            self.assertEqual(normalized["schema_version"], 1)
            self.assertEqual(normalized["cases"][0]["name"], "clockwise")
            self.assertTrue(normalized["cases"][0]["generated"])
            self.assertGreater(
                normalizer.signed_area2([tuple(point) for point in normalized["cases"][0]["polygons"][0]]),
                0.0,
            )
            self.assertEqual(normalized["cases"][1]["polygons"], data["cases"][1]["polygons"])

    def test_directory_scan_includes_binary_files_and_manual_case_stores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "campaign/inputs"
            nested.mkdir(parents=True)
            binary = nested / "manual.bin"
            manual = root / "campaign/manual-cases.json"
            ignored_json = root / "campaign/other.json"
            binary.write_bytes(b"")
            manual.write_text('{"cases": []}\n')
            ignored_json.write_text("{}\n")

            self.assertCountEqual(
                normalizer.candidate_files([root]),
                [manual, binary],
            )

    def test_completion_marker_stops_matching_after_input_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "manual.bin"
            binary = root / "tpp"
            marker = root / "manual.done"
            input_path.write_bytes(b"one-case")
            binary.write_bytes(b"benchmark")
            args = SimpleNamespace(
                threads=12,
                solver="binary_search_lazy",
                max_polygons="-1",
                max_instances="5",
                max_calls="1000000",
                max_branching="-1",
                max_seconds=None,
                repeat_count="1",
            )

            with patch.object(run_generated.bench, "TARGET_BINARY", binary):
                signature = run_generated.completion_signature(args, input_path)
                marker.write_text(json.dumps(signature, sort_keys=True) + "\n")
                self.assertTrue(run_generated.marker_matches(marker, signature))

                input_path.write_bytes(b"five-cases")
                next_signature = run_generated.completion_signature(args, input_path)

            self.assertFalse(run_generated.marker_matches(marker, next_signature))

    def test_normalizer_check_cli_reports_clockwise_polygons(self) -> None:
        cases = [
            normalizer.BinaryCase(
                start=(-1.0, 0.5),
                target=(2.0, 0.5),
                polygons=[[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]],
                solution=[],
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.bin"
            normalizer.write_binary_cases(path, cases)
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "benchmarks/scripts/normalize_polygon_orientation.py"), "--check", str(path)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("needs-fix:", completed.stdout)
            self.assertIn("reversed=1", completed.stdout)

    def test_normalizer_in_place_cli_is_idempotent(self) -> None:
        cases = [
            normalizer.BinaryCase(
                start=(-1.0, 0.5),
                target=(2.0, 0.5),
                polygons=[[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]],
                solution=[(-1.0, 0.5), (2.0, 0.5)],
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.bin"
            normalizer.write_binary_cases(path, cases)
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "benchmarks/scripts/normalize_polygon_orientation.py"), "--in-place", str(path)],
                check=True,
                capture_output=True,
            )
            normalized_bytes = path.read_bytes()
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "benchmarks/scripts/normalize_polygon_orientation.py"), "--in-place", str(path)],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertEqual(path.read_bytes(), normalized_bytes)
            self.assertIn("reversed=0", completed.stdout)

    def test_convert_instances_parses_polygons_as_counter_clockwise(self) -> None:
        polygon = convert_instances.parse_polygon_wkt("POLYGON ((0 1, 1 1, 1 0, 0 0, 0 1))")

        self.assertGreater(convert_instances.signed_area2(polygon), 0.0)

    def test_binary_reader_rejects_trailing_garbage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.bin"
            path.write_bytes(b"trailing")

            with self.assertRaises(ValueError):
                normalizer.read_binary_cases(path)

    def test_zero_area_polygons_are_left_unchanged(self) -> None:
        polygon = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

        normalized, changed = normalizer.normalize_polygon(polygon)

        self.assertFalse(changed)
        self.assertEqual(normalized, polygon)


if __name__ == "__main__":
    unittest.main()
