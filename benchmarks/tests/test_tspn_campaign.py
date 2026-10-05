import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import tspn_campaign as campaign


def row(name, solver, repeat=0):
	return {"name": name, "sha256": name + "-hash", "solver": solver, "repeat": repeat}


class OptionTests(unittest.TestCase):
	def test_defaults_match_the_documented_campaign(self):
		options = campaign.parse_options([])
		self.assertEqual(
			(
				options.campaign,
				options.solver,
				options.seconds,
				options.external_timeout,
			),
			("tspn-fekete-comparison-v1", "both", "60", "75"),
		)
		self.assertEqual(options.backends, ["ours", "fekete"])
		self.assertEqual(
			options.cycle_optimizations, ("cache", "features", "root", "interval")
		)

	def test_invalid_values_are_rejected(self):
		for argv in (
			["--seconds", "0"],
			["--seconds", "x"],
			["--repetitions", "0"],
			["--campaign", "../x"],
			["--campaign", "."],
			["--workers", "-1"],
			["--relative-gap", "0"],
			["--solver", "ours"],
			["--cycle-optimizations", "nope"],
			["--portfolio", "--portfolio-no-sharing"],
			["--portfolio", "--search-strategy", "dfs-bfs"],
		):
			with self.subTest(argv=argv), self.assertRaises(SystemExit):
				campaign.parse_options(argv)

	def test_scientific_notation_gap_is_kept_verbatim(self):
		self.assertEqual(
			campaign.parse_options(["--relative-gap", "1e-4"]).relative_gap, "1e-4"
		)


class CommandTests(unittest.TestCase):
	def test_single_solver_and_portfolio_flags_reach_the_engine(self):
		options = campaign.parse_options(
			[
				"--solver",
				"tpp-ours",
				"--portfolio",
				"--capture-oracles",
				"--cycle-optimizations",
				"cache,memo",
			]
		)
		arguments = campaign.engine_arguments(options)
		self.assertEqual(arguments[arguments.index("--solver") + 1], "ours")
		self.assertIn("--portfolio", arguments)
		self.assertIn("--capture-oracles", arguments)
		self.assertEqual(
			[
				arguments[i + 1]
				for i, a in enumerate(arguments)
				if a == "--cycle-optimization"
			],
			["cache", "memo"],
		)

	def test_the_progress_interval_reaches_the_engine_and_must_not_be_negative(self):
		arguments = campaign.engine_arguments(
			campaign.parse_options(["--progress-interval", "15"])
		)
		self.assertEqual(arguments[arguments.index("--progress-interval") + 1], "15")
		self.assertEqual(campaign.parse_options([]).progress_interval, "60")
		self.assertEqual(
			campaign.parse_options(["--progress-interval", "0"]).progress_interval, "0"
		)
		with self.assertRaises(SystemExit):
			campaign.parse_options(["--progress-interval", "-1"])

	def test_the_native_binary_is_where_the_shared_build_writes_it(self):
		self.assertEqual(
			campaign.binaries(Path("/b"))["ours"], Path("/b/bin/tpp-unordered")
		)

	def test_both_solvers_omit_the_solver_flag(self):
		self.assertNotIn(
			"--solver", campaign.engine_arguments(campaign.parse_options([]))
		)

	def test_single_run_resumes_only_when_a_config_exists(self):
		options = campaign.parse_options(["--dry-run"])
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory)
			self.assertNotIn("--resume", campaign.single_arguments(options, output))
			(output / "config.json").write_text("{}")
			arguments = campaign.single_arguments(options, output)
			self.assertIn("--resume", arguments)
			self.assertIn("--dry-run", arguments)
			self.assertEqual(arguments[:2], ["--all", "--instances-zip"])

	def test_shard_arguments_freeze_binaries_and_skip_fekete_for_ours(self):
		ours = campaign.parse_options(["--solver", "tpp-ours", "--workers", "2"])
		with tempfile.TemporaryDirectory() as directory:
			arguments = campaign.shard_arguments(ours, Path(directory))
			self.assertIn("--skip-build", arguments)
			self.assertNotIn("--fekete-binary", arguments)
			both = campaign.shard_arguments(
				campaign.parse_options(["--workers", "2"]), Path(directory)
			)
			self.assertIn("--fekete-binary", both)


class ShardTests(unittest.TestCase):
	def test_merge_drops_duplicates_and_rewrites_the_plan(self):
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory)
			config = {"repetitions": 1, "solvers": ["ours"], "plan": {}}
			rows = {
				0: [row("a", "ours"), row("b", "ours")],
				1: [row("b", "ours"), row("c", "ours")],
			}
			for index in (0, 1):
				shard = output / f"shard-{index}"
				shard.mkdir()
				(shard / "raw.jsonl").write_text(
					"".join(json.dumps(r) + "\n" for r in rows[index]) + "\n"
				)
				(shard / "instances.json").write_text(
					json.dumps(
						{
							"formulation": "TSPN",
							"instances": [
								{"name": r["name"], "polygons": []} for r in rows[index]
							],
						}
					)
				)
				(shard / "config.json").write_text(json.dumps(config))
			self.assertEqual(campaign.merge_shards(output, 2), 3)
			self.assertEqual(len((output / "raw.jsonl").read_text().splitlines()), 3)
			merged = json.loads((output / "config.json").read_text())
			self.assertEqual(
				(
					merged["plan"]["cases"],
					merged["plan"]["planned_runs"],
					merged["plan"]["shards"],
				),
				(4, 4, 2),
			)

	def test_move_aside_keeps_the_previous_campaign(self):
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory) / "c"
			output.mkdir()
			(output / "raw.jsonl").write_text("x")
			backup = campaign.move_aside(output)
			self.assertFalse(output.exists())
			self.assertEqual((backup / "raw.jsonl").read_text(), "x")
			self.assertIsNone(campaign.move_aside(output))


class PreflightTests(unittest.TestCase):
	def test_suite_hash_mismatch_is_fatal(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "suite.zip"
			path.write_bytes(b"not the archive")
			with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
				campaign.verify_suite(path)
			with self.assertRaisesRegex(SystemExit, "missing"):
				campaign.verify_suite(Path(directory) / "absent.zip")

	def test_conan_packages_depend_on_the_selected_solvers(self):
		ours = {names[0] for names in campaign.conan_packages(["ours"])}
		both = {names[0] for names in campaign.conan_packages(["ours", "fekete"])}
		self.assertNotIn("cgal-config.cmake", ours)
		self.assertIn("cgal-config.cmake", both)

	def test_missing_conan_prefix_is_fatal(self):
		with (
			tempfile.TemporaryDirectory() as directory,
			patch.object(campaign, "EXTERNAL_SOURCE", Path(directory)),
		):
			with self.assertRaisesRegex(
				SystemExit, "Conan C\\+\\+ dependencies not found"
			):
				campaign.verify_conan(["ours"])


if __name__ == "__main__":
	unittest.main()
