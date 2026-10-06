import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import free_order_campaign
import run_spec
from case_selection import describe_selection, parse_case_selection


def three_case_suite() -> bytes:
	encoded = bytearray()
	for index in range(3):
		encoded.extend(struct.pack("<ddddQ", 0.0, 0.0, 10.0, 0.0, 1))
		encoded.extend(struct.pack("<Q", 4))
		for x, y in ((index * 10.0, 1.0), (index * 10.0 + 1.0, 1.0),
			(index * 10.0 + 1.0, 2.0), (index * 10.0, 2.0)):
			encoded.extend(struct.pack("<dd", x, y))
		encoded.extend(struct.pack("<Q", 0))
	return bytes(encoded)


class ParsingTests(unittest.TestCase):
	def test_numbers_and_ranges_are_one_based_sorted_and_unique(self):
		self.assertEqual(parse_case_selection("65, 66,130-131 66"), [64, 65, 129, 130])
		self.assertEqual(parse_case_selection("3"), [2])
		self.assertEqual(describe_selection([64, 65, 129]), "65, 66, 130")

	def test_bad_selections_say_what_is_wrong(self):
		for text, total in (("", None), ("0", None), ("5-3", None), ("a", None), ("1,,x", None), ("9", 8)):
			with self.assertRaises(ValueError, msg=text):
				parse_case_selection(text, total)
		self.assertEqual(parse_case_selection("8", 8), [7])


class FieldTests(unittest.TestCase):
	def test_the_cases_field_exists_only_for_free_order_and_follows_the_campaign_size(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "campaigns" / "sample"
			path.mkdir(parents=True)
			(path / "campaign.json").write_text(json.dumps({"inputs": [{"file": "a.bin", "instances": 10}]}))
			with patch.dict(os.environ, {"TPP_WORKSPACE": directory}):
				item = run_spec.BY_KEY["cases"]
				self.assertIsNone(item.spec("fixed-order"))
				values = run_spec.default_values("free-order")
				values.update(campaign="sample", cases="3,5-6")
				self.assertEqual(run_spec.validate(values), [])
				arguments = run_spec.to_legacy(values)[1]
				self.assertEqual(arguments[arguments.index("--cases") + 1], "3,5-6")
				self.assertTrue(run_spec.validate({**values, "cases": "11"}))
				self.assertTrue(run_spec.validate({**values, "cases": "5", "max_instances": 4}))
				self.assertEqual(run_spec.parse_value(item, "free-order", "", 10), None)
				with self.assertRaises(ValueError):
					run_spec.parse_value(item, "free-order", "11", 10)
				self.assertEqual(run_spec.from_cli(["--problem", "free-order", "--campaign", "sample", "--cases", "2"])["cases"], "2")


class CampaignTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		root = Path(self.directory.name)
		self.campaign = root / "campaign"
		self.campaign.mkdir()
		(self.campaign / "tiny.bin").write_bytes(three_case_suite())
		(self.campaign / "campaign.json").write_text(json.dumps({"inputs": [{"file": "tiny.bin"}]}))
		self.binary = root / "tpp"
		self.binary.write_bytes(b"stub")
		self.solved = []

	def ours(self, _binary, start, _target, polygons, *_rest, **_options):
		self.solved.append(polygons[0][0][0])  # the case's x offset identifies it
		return {"status": "optimal", "exact": True, "termination": "optimal", "seconds": 0.1,
			"upper_bound": 10.0, "lower_bound": 10.0, "path": [[0.0, 0.0], [10.0, 0.0]]}

	def run_main(self, *extra):
		with patch.object(free_order_campaign, "BINARY", self.binary), patch.object(
			free_order_campaign, "ensure_binary"
		), patch.object(free_order_campaign, "run_unordered_solver", side_effect=self.ours), patch.object(
			free_order_campaign, "validate_path", return_value={"valid": True}
		), patch("sys.stdout", io.StringIO()):
			return free_order_campaign.main(
				[str(self.campaign), "--solver", "tpp-ours", "--max-seconds", "60", "--no-build", *extra]
			)

	def report(self):
		return json.loads(next((self.campaign / "results").glob("*/report.json")).read_text())

	def test_only_the_chosen_cases_run_and_a_later_run_finishes_the_rest_in_the_same_report(self):
		self.assertEqual(self.run_main("--cases", "3,1"), 0)
		self.assertEqual(sorted(self.solved), [0.0, 20.0])
		report = self.report()
		self.assertEqual((report["status"], report["selected_cases"]), ("completed", [0, 2]))
		self.assertEqual(sorted(row["case"] for row in report["rows"]), [0, 2])
		text = free_order_campaign.render_comparison_summary(report, 3)
		self.assertIn("Latest attempt ran only 2 of 3 case(s) (--cases): 1, 3.", text)

		self.solved.clear()
		self.assertEqual(self.run_main(), 0)  # no selection: resume covers what is missing
		self.assertEqual(self.solved, [10.0])
		report = self.report()
		self.assertEqual(len(list((self.campaign / "results").iterdir())), 1)
		self.assertEqual(sorted(row["case"] for row in report["rows"]), [0, 1, 2])
		self.assertNotIn("selected_cases", report)

	def test_max_instances_defaults_to_all_cases_and_accepts_minus_one(self):
		self.assertEqual(self.run_main(), 0)
		self.assertEqual(len(self.report()["rows"]), 3)
		self.solved.clear()
		self.assertEqual(self.run_main("--max-instances", "-1", "--force"), 0)
		self.assertEqual(len(self.solved), 3)
		with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
			self.run_main("--max-instances", "0")

	def test_the_command_built_by_the_interface_leaves_max_instances_out_when_unlimited(self):
		values = run_spec.default_values("free-order")
		values.update(campaign="x", cases="2")
		self.assertNotIn("--max-instances", run_spec.to_legacy(values)[1])
		values["max_instances"] = 2
		arguments = run_spec.to_legacy(values)[1]
		self.assertEqual(arguments[arguments.index("--max-instances") + 1], "2")

	def test_a_selection_past_the_last_case_is_refused(self):
		with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()) as error:
			self.run_main("--cases", "4")
		self.assertIn("the campaign has 3", error.getvalue())

	def test_a_failure_outside_the_selection_does_not_make_the_run_fail(self):
		def ours(*args, **options):
			if args[3][0][0][0] == 10.0:
				raise RuntimeError("boom")
			return self.ours(*args, **options)

		with patch.object(self, "ours", side_effect=ours):
			self.run_main()  # case 2 fails
		self.assertEqual(self.report()["status"], "completed_with_errors")
		self.assertEqual(self.run_main("--cases", "1,3"), 0)  # not retried, and its error is not counted
		self.assertEqual(self.report()["status"], "completed")


if __name__ == "__main__":
	unittest.main()
