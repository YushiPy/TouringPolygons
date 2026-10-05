import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import run_spec
import tspn_campaign

import tpp


def values_for(problem, **changes):
	values = run_spec.default_values(problem)
	values.update(changes)
	return values


class SchemaTests(unittest.TestCase):
	def test_every_field_has_a_default_for_each_problem_that_uses_it(self):
		for problem in run_spec.PROBLEMS:
			values = run_spec.default_values(problem)
			for item in run_spec.applicable(problem):
				spec = item.spec(problem)
				self.assertIn(item.key, values)
				if spec.choices and spec.default is not None:
					self.assertTrue(
						run_spec._valid_choice(spec, spec.default), (problem, item.key)
					)

	def test_defaults_follow_the_documented_protocols(self):
		tspn = run_spec.default_values("tspn")
		self.assertEqual(
			(tspn["time_limit"], tspn["campaign"], tspn["solver"]),
			(60, "tspn-fekete-comparison-v1", "both"),
		)
		self.assertEqual(
			tspn["cycle_optimizations"],
			tspn_campaign.parse_options([]).cycle_optimizations,
		)
		self.assertEqual(
			run_spec.CYCLE_OPTIMIZATIONS, tspn_campaign.CYCLE_OPTIMIZATIONS
		)
		self.assertEqual(
			run_spec.default_values("fixed-order")["oracle_calls"], 100_000_000
		)

	def test_switching_problem_keeps_shared_fields_and_drops_the_rest(self):
		values = values_for(
			"fixed-order", campaign="c", time_limit=30, threads=4, max_polygons=7
		)
		free = run_spec.switch_problem(values, "free-order")
		self.assertEqual(
			(free["campaign"], free["time_limit"], free["threads"]), ("c", 30, 4)
		)
		self.assertNotIn("max_polygons", free)
		self.assertEqual(free["solver"], "tpp-ours")

	def test_a_tspn_campaign_name_is_never_exchanged_with_an_instance_campaign(self):
		tspn = values_for("tspn", campaign="my-output")
		self.assertIsNone(run_spec.switch_problem(tspn, "fixed-order")["campaign"])
		fixed = values_for("fixed-order", campaign="instances")
		self.assertEqual(
			run_spec.switch_problem(fixed, "tspn")["campaign"],
			"tspn-fekete-comparison-v1",
		)

	def test_value_parsing_accepts_underscores_and_rejects_nonsense(self):
		calls = run_spec.BY_KEY["oracle_calls"]
		self.assertEqual(run_spec.parse_value(calls, "tspn", "1_000"), 1000)
		with self.assertRaises(ValueError):
			run_spec.parse_value(calls, "tspn", "many")
		gap = run_spec.BY_KEY["relative_gap"]
		self.assertEqual(run_spec.parse_value(gap, "tspn", "1e-4"), 1e-4)
		self.assertIsNone(run_spec.parse_value(gap, "free-order", ""))
		with self.assertRaises(ValueError):
			run_spec.parse_value(gap, "tspn", "")
		opts = run_spec.BY_KEY["cycle_optimizations"]
		self.assertEqual(
			run_spec.parse_value(opts, "tspn", "cache, memo"), ("cache", "memo")
		)
		with self.assertRaises(ValueError):
			run_spec.parse_value(opts, "tspn", "bogus")


class ExpressionTests(unittest.TestCase):
	def test_arithmetic_expressions_are_evaluated(self):
		for text, expected in (
			("10 ** -7", 1e-7),
			("1 / 100", 0.01),
			("(10 - 5) * 2 / 3", 10 / 3),
			("2 ** 10", 1024),
			("1_000", 1000),
			("-3", -3),
			("1e-4", 1e-4),
		):
			with self.subTest(text=text):
				self.assertEqual(run_spec.evaluate(text), expected)

	def test_nothing_but_arithmetic_is_ever_executed(self):
		for text in (
			"__import__('os').system('true')",
			"open",
			"x",
			"[1]",
			"1 if 1 else 2",
			"abs(-1)",
			'"a"',
			"1 < 2",
			"2 ** 1000000",
			"9 ** 9 ** 9",
			"(-8) ** 0.5",
			"1 / 0",
			"1 +",
			"",
		):
			with self.subTest(text=text), self.assertRaises(ValueError):
				run_spec.evaluate(text)

	def test_numeric_fields_take_expressions_and_enforce_their_range(self):
		gap, limit = run_spec.BY_KEY["relative_gap"], run_spec.BY_KEY["time_limit"]
		calls, workers = run_spec.BY_KEY["oracle_calls"], run_spec.BY_KEY["workers"]
		self.assertEqual(run_spec.parse_value(gap, "tspn", "10 ** -7"), 1e-7)
		self.assertEqual(run_spec.parse_value(calls, "tspn", "10 ** 8"), 100_000_000)
		self.assertEqual(run_spec.parse_value(limit, "tspn", "120 / 2"), 60)
		self.assertEqual(run_spec.parse_value(limit, "fixed-order", "-1"), -1)
		for item, problem, text in (
			(limit, "fixed-order", "-5"),
			(limit, "fixed-order", "0"),
			(limit, "tspn", "-1"),
			(calls, "tspn", "2.5"),
			(calls, "tspn", "-1"),
			(workers, "tspn", "0"),
			(gap, "tspn", "0"),
			(gap, "tspn", "1 - 1"),
		):
			with self.subTest(text=text, field=item.key), self.assertRaises(ValueError):
				run_spec.parse_value(item, problem, text)

	def test_only_numeric_fields_restrict_typing(self):
		self.assertIsNone(
			run_spec.allowed_characters(run_spec.BY_KEY["pattern"], "fixed-order")
		)
		allowed = run_spec.allowed_characters(run_spec.BY_KEY["oracle_calls"], "tspn")
		self.assertTrue(set("0123456789+-*/(). _eE") <= set(allowed))
		self.assertFalse(set("abcdfghijklmnopqrstuvwxyz;") & set(allowed))

	def test_the_command_line_rejects_out_of_range_values_up_front(self):
		for argv in (
			["--problem", "tspn", "--time-limit", "-1"],
			["--problem", "tspn", "--workers", "1 - 1"],
		):
			with self.subTest(argv=argv), self.assertRaises(SystemExit):
				run_spec.from_cli(argv)
		self.assertEqual(
			run_spec.from_cli(["--problem", "tspn", "--relative-gap", "10 ** -4"])[
				"relative_gap"
			],
			1e-4,
		)


class ValidationTests(unittest.TestCase):
	def test_fixed_and_free_order_need_an_existing_campaign(self):
		with (
			tempfile.TemporaryDirectory() as directory,
			patch.dict(os.environ, {"TPP_WORKSPACE": directory}),
		):
			self.assertTrue(run_spec.validate(values_for("fixed-order")))
			self.assertTrue(
				run_spec.validate(values_for("free-order", campaign="ghost"))
			)
			campaign = Path(directory) / "campaigns/real"
			campaign.mkdir(parents=True)
			(campaign / "campaign.json").write_text("{}")
			self.assertEqual(
				run_spec.validate(values_for("free-order", campaign="real")), []
			)
			self.assertEqual(run_spec.available_campaigns(), ["real"])

	def test_limits_must_be_positive_unless_unlimited_is_allowed(self):
		self.assertTrue(run_spec.validate(values_for("tspn", time_limit=-1)))
		self.assertTrue(run_spec.validate(values_for("tspn", time_limit=0)))
		with (
			tempfile.TemporaryDirectory() as directory,
			patch.dict(os.environ, {"TPP_WORKSPACE": directory}),
		):
			(Path(directory) / "campaigns/c").mkdir(parents=True)
			(Path(directory) / "campaigns/c/campaign.json").write_text("{}")
			self.assertEqual(
				run_spec.validate(
					values_for("fixed-order", campaign="c", time_limit=-1)
				),
				[],
			)
			self.assertTrue(
				run_spec.validate(values_for("fixed-order", campaign="c", time_limit=0))
			)

	def test_tspn_call_budget_requires_our_solver_and_strategy_excludes_portfolio(self):
		self.assertTrue(run_spec.validate(values_for("tspn", oracle_calls=5)))
		self.assertEqual(
			run_spec.validate(values_for("tspn", oracle_calls=5, solver="tpp-ours")), []
		)
		self.assertTrue(
			run_spec.validate(
				values_for("tspn", portfolio="cooperative", search_strategy="dfs-bfs")
			)
		)


class CommandTests(unittest.TestCase):
	def test_cli_omits_defaults_and_round_trips(self):
		for problem, changes in (
			(
				"tspn",
				dict(
					solver="tpp-ours",
					time_limit=10,
					oracle_calls=7,
					portfolio="cooperative",
					cycle_optimizations=("cache", "memo"),
					capture_oracles=True,
					resume=False,
					relative_gap=1e-4,
				),
			),
			(
				"free-order",
				dict(
					campaign="c",
					solver="both",
					threads=4,
					workers=2,
					relative_gap=0.01,
					rebuild=False,
				),
			),
			(
				"fixed-order",
				dict(
					campaign="c",
					solver="tan_jiang",
					max_instances=5,
					file_timeout=90,
					pattern="a*.bin",
				),
			),
		):
			with (
				self.subTest(problem=problem),
				tempfile.TemporaryDirectory() as directory,
				patch.dict(os.environ, {"TPP_WORKSPACE": directory}),
			):
				(Path(directory) / "campaigns/c").mkdir(parents=True)
				(Path(directory) / "campaigns/c/campaign.json").write_text("{}")
				values = values_for(problem, **changes)
				arguments = run_spec.to_cli(values)
				self.assertEqual(run_spec.from_cli(arguments), values)
				self.assertNotIn(
					"--workers",
					run_spec.to_cli(values_for(problem, **{**changes, "workers": 1}))
					if problem != "fixed-order"
					else [],
				)

	def test_progress_interval_defaults_on_and_zero_turns_it_off(self):
		for problem in ("free-order", "tspn"):
			values = values_for(problem, campaign="c")
			self.assertEqual(values["progress_interval"], 60)
			self.assertNotIn("--progress-interval", run_spec.to_cli(values))
			self.assertEqual(
				run_spec.validate({**values, "progress_interval": 0}),
				[] if problem == "tspn" else run_spec.validate(values),
			)
			self.assertTrue(run_spec.validate({**values, "progress_interval": -1}))
		self.assertNotIn("progress_interval", run_spec.default_values("fixed-order"))
		legacy = run_spec.to_legacy(values_for("tspn", progress_interval=0))[1]
		self.assertEqual(legacy[legacy.index("--progress-interval") + 1], "0")
		self.assertEqual(tspn_campaign.parse_options(legacy).progress_interval, "0")
		free = run_spec.to_legacy(
			values_for("free-order", campaign="c", progress_interval=15)
		)[1]
		self.assertEqual(free[free.index("--progress-interval") + 1], "15")

	def test_inapplicable_or_invalid_options_are_rejected(self):
		for argv in (
			["--problem", "tspn", "--threads", "2"],
			["--problem", "fixed-order", "--portfolio", "cooperative"],
			["--problem", "nope"],
			["--problem", "tspn", "--time-limit", "-3"],
			["--problem", "fixed-order"],
		):
			with self.subTest(argv=argv), self.assertRaises(SystemExit):
				run_spec.from_cli(argv)

	def test_legacy_translation_of_each_problem(self):
		with (
			tempfile.TemporaryDirectory() as directory,
			patch.dict(os.environ, {"TPP_WORKSPACE": directory}),
		):
			fixed = run_spec.to_legacy(
				values_for(
					"fixed-order",
					campaign="c",
					time_limit=-1,
					resume=False,
					rebuild=False,
					solver="gurobi",
					max_instances=3,
				)
			)
			self.assertEqual(fixed[0], "run")
			self.assertEqual(fixed[1][0], "c")
			self.assertNotIn("--max-seconds", fixed[1])
			self.assertEqual(fixed[1][fixed[1].index("--solver") + 1], "gurobi")
			self.assertTrue({"--force", "--no-build"} <= set(fixed[1]))
			free = run_spec.to_legacy(
				values_for("free-order", campaign="c", solver="both", time_limit=-1)
			)
			self.assertEqual(free[0], "free-order")
			self.assertEqual(
				[free[1][i + 1] for i, a in enumerate(free[1]) if a == "--solver"],
				["tpp-ours", "tpp-fekete"],
			)
			self.assertEqual(free[1][free[1].index("--max-seconds") + 1], "-1")
			name, arguments = run_spec.to_legacy(
				values_for("tspn", time_limit=10, portfolio="independent", resume=False)
			)
			self.assertEqual(name, "tspn-compare")
			self.assertEqual(arguments[arguments.index("--external-timeout") + 1], "25")
			self.assertTrue({"--portfolio-no-sharing", "--force"} <= set(arguments))

	def test_tspn_translation_is_accepted_by_the_tspn_parser(self):
		values = values_for(
			"tspn",
			solver="tpp-ours",
			oracle_calls=9,
			portfolio="cooperative",
			time_limit=0.5,
			relative_gap=1e-4,
			cycle_optimizations=("cache", "memo"),
			capture_oracles=True,
			workers=3,
			dry_run=True,
		)
		options = tspn_campaign.parse_options(run_spec.to_legacy(values)[1])
		self.assertEqual(
			(options.solver, options.max_calls, options.portfolio, options.workers),
			("tpp-ours", 9, "cooperative", 3),
		)

	def test_defaults_translate_to_a_valid_tspn_command(self):
		options = tspn_campaign.parse_options(
			run_spec.to_legacy(run_spec.default_values("tspn"))[1]
		)
		self.assertEqual(options, tspn_campaign.parse_options([]))

	def test_bench_dispatches_to_the_translated_command(self):
		seen = {}
		with patch.dict(
			tpp.COMMANDS,
			{
				"tspn-compare": tpp.Command(
					"", "", lambda argv: seen.setdefault("argv", argv) and 0
				)
			},
		):
			self.assertEqual(
				tpp.command_bench(
					["--problem", "tspn", "--solver", "tpp-ours", "--dry-run"]
				),
				0,
			)
		self.assertIn("--dry-run", seen["argv"])


class SessionTests(unittest.TestCase):
	def test_session_tracks_values_and_prints_the_command(self):
		session = run_spec.Session()
		session.set("problem", "tspn")
		session.set("time_limit", 30)
		self.assertEqual(session.errors(), [])
		self.assertIn("--time-limit 30", session.command())
		self.assertTrue(
			session.command().startswith(
				"python3 benchmarks/tpp.py bench --problem tspn"
			)
		)


if __name__ == "__main__":
	unittest.main()
