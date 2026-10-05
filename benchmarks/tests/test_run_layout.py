import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import run_generated
import run_layout

import tpp
import workspace


def touch_run(directory: Path, marker: str, age: int = 0) -> Path:
	directory.mkdir(parents=True, exist_ok=True)
	(directory / marker).write_text(
		"{}" if marker.endswith(".json") else "input_file,status\n"
	)
	stamp = 1_700_000_000 + age
	os.utime(directory / marker, (stamp, stamp))
	os.utime(directory, (stamp, stamp))
	return directory


class LayoutTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.campaign = Path(self.directory.name) / "campaign"
		self.campaign.mkdir()
		self.results = self.campaign / "results"

	def test_free_order_reports_are_found_in_both_layouts_newest_first(self):
		touch_run(self.results / "free-order/old", "report.json", age=1)
		touch_run(self.results / "new", "report.json", age=5)
		touch_run(self.results / "middle", "report.json", age=3)
		touch_run(
			self.results / "comparisons/x", "report.json", age=9
		)  # not a run: a derived artifact
		names = [
			path.parent.name for path in run_layout.free_order_reports(self.campaign)
		]
		self.assertEqual(names, ["new", "middle", "old"])
		self.assertEqual(
			run_layout.latest_free_order_report(self.campaign).parent.name, "new"
		)

	def test_the_kind_of_a_run_is_told_by_its_files(self):
		touch_run(self.results / "a", "report.json", age=1)
		touch_run(self.results / "b", "run-index.csv", age=2)
		self.assertEqual(
			[p.name for p in run_layout.fixed_order_runs(self.campaign)], ["b"]
		)
		self.assertEqual(
			[p.parent.name for p in run_layout.free_order_reports(self.campaign)], ["a"]
		)

	def test_the_latest_fixed_order_index_falls_back_to_the_flat_legacy_file(self):
		self.assertEqual(
			run_layout.latest_fixed_order_index(self.campaign),
			self.results / "run-index.csv",
		)
		touch_run(self.results / "r1", "run-index.csv", age=1)
		touch_run(self.results / "r2", "run-index.csv", age=2)
		self.assertEqual(
			run_layout.latest_fixed_order_index(self.campaign),
			self.results / "r2/run-index.csv",
		)

	def test_run_ids_sort_by_time_and_do_not_collide(self):
		ids = {run_layout.new_run_id() for _ in range(50)}
		self.assertEqual(len(ids), 50)
		self.assertRegex(next(iter(ids)), r"^\d{8}-\d{6}-[0-9a-f]{6}$")


class MigrationTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.campaign = Path(self.directory.name) / "campaign"
		self.results = self.campaign / "results"
		self.results.mkdir(parents=True)

	def flat_fixed_order(self):
		(self.results / "a.csv").write_text("x")
		(self.results / "a.md").write_text("summary")
		(self.results / "a-solutions").mkdir()
		(self.results / "a-solutions/case.svg").write_text("<svg/>")
		(self.results / "run.json").write_text('{"kind": "fixed-order-campaign"}')
		(self.results / "run-index.csv").write_text(
			f"input_file,csv_output\nin.bin,{self.results}/a.csv\n"
		)
		(self.campaign / "campaign.json").write_text(
			json.dumps({"benchmark_runs": [{"index": "results/run-index.csv"}]})
		)

	def test_free_order_runs_move_up_one_level(self):
		touch_run(self.results / "free-order/r1", "report.json")
		touch_run(self.results / "free-order/r2", "report.json")
		before = (self.results / "free-order/r1/report.json").read_text()
		actions = run_layout.migrate(self.campaign)
		self.assertEqual(len(actions), 2)
		self.assertEqual((self.results / "r1/report.json").read_text(), before)
		self.assertTrue((self.results / "r2/report.json").is_file())
		self.assertFalse((self.results / "free-order").exists())

	def test_dry_run_changes_nothing_and_a_second_run_finds_nothing_to_do(self):
		touch_run(self.results / "free-order/r1", "report.json")
		self.flat_fixed_order()
		self.assertEqual(len(run_layout.migrate(self.campaign, dry_run=True)), 2)
		self.assertTrue((self.results / "free-order/r1").exists())
		self.assertTrue((self.results / "run-index.csv").exists())
		run_layout.migrate(self.campaign, rewrite_paths=workspace.rewrite_paths)
		self.assertEqual(run_layout.migrate(self.campaign), [])

	def test_an_existing_target_is_never_overwritten(self):
		touch_run(self.results / "free-order/r1", "report.json")
		touch_run(self.results / "r1", "report.json")
		(self.results / "r1/report.json").write_text('{"keep": true}')
		actions = run_layout.migrate(self.campaign)
		self.assertTrue(actions[0].startswith("SKIP"))
		self.assertEqual(
			(self.results / "r1/report.json").read_text(), '{"keep": true}'
		)
		self.assertTrue((self.results / "free-order/r1/report.json").exists())

	def test_flat_fixed_order_files_become_one_run_with_updated_paths(self):
		self.flat_fixed_order()
		touch_run(self.results / "comparisons/c1", "comparison.csv")
		run_layout.migrate(self.campaign, rewrite_paths=workspace.rewrite_paths)
		runs = run_layout.fixed_order_runs(self.campaign)
		self.assertEqual(len(runs), 1)
		run = runs[0]
		self.assertTrue(run.name.endswith("-legacy"))
		self.assertEqual(
			{p.name for p in run.iterdir()},
			{"a.csv", "a.md", "a-solutions", "run.json", "run-index.csv"},
		)
		self.assertIn(f"{run}/a.csv", (run / "run-index.csv").read_text())
		self.assertTrue((self.results / "comparisons/c1/comparison.csv").exists())
		recorded = json.loads((self.campaign / "campaign.json").read_text())[
			"benchmark_runs"
		][0]["index"]
		self.assertEqual(recorded, f"results/{run.name}/run-index.csv")


class FixedOrderRunDirectoryTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		root = Path(self.directory.name).resolve()
		self.campaign = root / "campaign"
		(self.campaign / "inputs").mkdir(parents=True)
		(self.campaign / "inputs/cases.bin").write_bytes(b"cases")
		(self.campaign / "campaign.json").write_text('{"inputs": []}')
		self.binary = root / "tool"
		self.binary.write_bytes(b"tool")
		self.arguments = run_generated.make_parser().parse_args(
			[
				"--input",
				str(self.campaign / "inputs"),
				"--output",
				str(self.campaign / "results"),
				"--campaign-file",
				str(self.campaign / "campaign.json"),
				"--max-seconds",
				"5",
			]
		)

	def choose(self, **changes):
		namespace = SimpleNamespace(**{**vars(self.arguments), **changes})
		with patch.object(
			run_generated.bench, "ensure_target", return_value=self.binary
		):
			return run_generated.choose_run_directory(namespace)

	def finished_run(self, name="r1", age=1, **changes):
		run = touch_run(self.campaign / "results" / name, "run-index.csv", age=age)
		namespace = SimpleNamespace(**{**vars(self.arguments), **changes})
		marker = run_generated.output_paths(
			self.campaign / "inputs/cases.bin", self.campaign / "inputs", run
		)[3]
		marker.write_text(
			json.dumps(
				run_generated.completion_signature(
					namespace, self.campaign / "inputs/cases.bin", self.binary
				)
			)
		)
		return run

	def test_the_first_run_gets_a_new_directory_inside_results(self):
		chosen = self.choose()
		self.assertEqual(chosen.parent, (self.campaign / "results").resolve())
		self.assertRegex(chosen.name, r"^\d{8}-\d{6}-[0-9a-f]{6}$")
		self.assertFalse(chosen.exists())

	def test_a_run_finished_with_the_same_settings_is_continued_not_repeated_elsewhere(
		self,
	):
		run = self.finished_run()
		self.assertEqual(self.choose(), run)

	def test_changed_settings_or_force_start_a_new_run_and_keep_the_old_one(self):
		run = self.finished_run()
		self.assertNotEqual(self.choose(max_seconds=99), run)
		self.assertNotEqual(self.choose(force=True), run)
		self.assertTrue((run / "run-index.csv").exists())

	def test_a_run_without_finished_inputs_is_resumed(self):
		run = touch_run(self.campaign / "results/r1", "run-index.csv")
		self.assertEqual(self.choose(), run)

	def test_only_the_newest_run_is_considered(self):
		self.finished_run("older", age=1)
		newest = touch_run(self.campaign / "results/newest", "run-index.csv", age=9)
		(newest / "cases.done").write_text('{"stale": true}')
		self.assertNotIn(self.choose().name, {"older", "newest"})


class StatusTests(unittest.TestCase):
	def test_status_reads_the_newest_run_directory(self):
		with (
			tempfile.TemporaryDirectory() as directory,
			patch.dict(os.environ, {"TPP_WORKSPACE": directory}),
		):
			campaign = Path(directory) / "campaigns/c"
			campaign.mkdir(parents=True)
			(campaign / "campaign.json").write_text(
				json.dumps({"name": "c", "inputs": []})
			)
			run = campaign / "results/r1"
			run.mkdir(parents=True)
			with (run / "run-index.csv").open("w", newline="") as file:
				writer = csv.writer(file)
				writer.writerow(["input_file", "status", "action"])
				writer.writerow(["in.bin", "completed", "ran"])
			output = io.StringIO()
			with contextlib.redirect_stdout(output):
				self.assertEqual(tpp.command_status(["c"]), 0)
			self.assertIn("1 input files indexed", output.getvalue())


if __name__ == "__main__":
	unittest.main()
