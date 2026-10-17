"""Publication trace tooling must work without dashboard event snapshots."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_trace_data as exporter


class TraceExportTests(unittest.TestCase):
    def test_the_standalone_corpus_loads_without_dashboard_data(self):
        cases = exporter.load_cases(exporter.DEFAULT_DATA)
        self.assertEqual(len(cases), 558)
        self.assertEqual(set(cases), set(range(558)))
        self.assertTrue(all(case.polygons and case.digest for case in cases.values()))

    def test_json_and_javascript_inputs_describe_the_same_geometry(self):
        data = {"rows": [{"case": 0, "sha256": "abc", "geometry": {
            "start": [0, 0], "target": [2, 0], "polygons": [[[0, 0], [1, 0], [0, 1]]],
        }}]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.js"
            raw = source.with_suffix(".json")
            raw.write_text(json.dumps(data))
            source.write_text("window.TPPEventData = " + json.dumps(data) + ";\n")
            self.assertEqual(exporter.load_cases(source), exporter.load_cases(raw))
            source.write_text("window.Unexpected = " + json.dumps(data) + ";\n")
            with self.assertRaises(ValueError):
                exporter.load_cases(source)

    def test_review_exports_are_standalone_and_never_overwrite_a_file(self):
        traces = {"1": {"case": 1, "events": [{"kind": "complete"}]}}
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("js", "json"):
                output = Path(directory) / f"reviewed.{extension}"
                with patch.object(sys, "argv", ["export_trace_data.py", "--output", str(output)]), \
                        patch.object(exporter, "load_cases", return_value={}), \
                        patch.object(exporter, "export_cases", return_value=traces):
                    exporter.main()
                    original = output.read_text()
                    payload = original
                    if extension == "js":
                        self.assertTrue(payload.startswith("window.TPPTraceData = "))
                        payload = payload.split("=", 1)[1].strip().removesuffix(";")
                    self.assertEqual(json.loads(payload), {"schema_version": 1, "cases": traces})
                    with self.assertRaises(FileExistsError):
                        exporter.main()
                    self.assertEqual(output.read_text(), original)


if __name__ == "__main__":
    unittest.main()
