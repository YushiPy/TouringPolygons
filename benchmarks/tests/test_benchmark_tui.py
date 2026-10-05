import json
import os
import pty
import select
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))

import benchmark_tui as tui
import run_spec


def press(app, *keys):
	for key in keys:
		app.handle(key)


def type_text(app, text):
	press(app, *text)


class WorkspaceCase(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		patcher = patch.dict(os.environ, {"TPP_WORKSPACE": self.directory.name})
		patcher.start()
		self.addCleanup(patcher.stop)
		self.root = Path(self.directory.name)

	def make_campaign(self, name):
		(self.root / "campaigns" / name).mkdir(parents=True)
		(self.root / "campaigns" / name / "campaign.json").write_text("{}")


class NavigationTests(WorkspaceCase):
	def test_arrows_wrap_and_digits_jump(self):
		app = tui.App()
		press(app, "up")
		self.assertEqual(app.current(), ("action", tui.ACTIONS[-1]))
		press(app, "down")
		self.assertEqual(app.cursor, 0)
		press(app, "5")
		self.assertEqual(app.current()[1].key, "time_limit")

	def test_choice_menu_changes_problem_and_the_field_list(self):
		app = tui.App()
		press(app, "enter", "down", "down", "enter")
		self.assertEqual(app.problem, "tspn")
		self.assertIn("portfolio", [item.key for item in app.fields()])
		press(app, "right")
		self.assertEqual(app.problem, "fixed-order")
		self.assertNotIn("portfolio", [item.key for item in app.fields()])

	def test_bool_toggles_with_enter_space_and_arrows(self):
		app = tui.App()
		keys = [item.key for item in app.fields()]
		app.cursor = keys.index("resume")
		press(app, "enter")
		self.assertFalse(app.session.values["resume"])
		press(app, " ")
		self.assertTrue(app.session.values["resume"])
		press(app, "left")
		self.assertFalse(app.session.values["resume"])

	def test_campaign_menu_lists_local_campaigns_and_allows_typing(self):
		self.make_campaign("alpha")
		self.make_campaign("beta")
		app = tui.App()
		app.cursor = [item.key for item in app.fields()].index("campaign")
		press(app, "enter", "down", "enter")
		self.assertEqual(app.session.values["campaign"], "beta")
		press(app, "enter", "down", "enter")
		self.assertIsInstance(app.popup, tui.LineInput)
		press(app, *["backspace"] * 4, *"gamma", "enter")
		self.assertEqual(app.session.values["campaign"], "gamma")

	def test_multi_select_toggles_optimizations_in_schema_order(self):
		app = tui.App(run_spec.default_values("tspn"))
		app.cursor = [item.key for item in app.fields()].index("cycle_optimizations")
		press(app, "enter")
		first = app.popup.options.index(("cache", "cache"))
		app.popup.index = first
		press(app, " ")
		app.popup.index = app.popup.options.index(("memo", "memo"))
		press(app, " ", "enter")
		self.assertEqual(
			app.session.values["cycle_optimizations"],
			("features", "root", "memo", "interval"),
		)
		self.assertNotIn("cache", app.session.values["cycle_optimizations"])
		self.assertIn("memo", app.session.values["cycle_optimizations"])


class EditingTests(WorkspaceCase):
	def test_numbers_accept_underscores_and_reject_text_without_closing(self):
		app = tui.App()
		app.cursor = [item.key for item in app.fields()].index("oracle_calls")
		press(app, "enter", *["backspace"] * 12, *"5_000")
		press(app, "enter")
		self.assertEqual(app.session.values["oracle_calls"], 5000)
		press(app, "enter", *["backspace"] * 6, *"many", "enter")
		self.assertIsInstance(app.popup, tui.LineInput)
		self.assertTrue(app.popup.error)
		press(app, "esc")
		self.assertIsNone(app.popup)
		self.assertEqual(app.session.values["oracle_calls"], 5000)

	def test_cursor_editing_inside_the_text(self):
		app = tui.App()
		app.cursor = [item.key for item in app.fields()].index("pattern")
		press(app, "enter", "home", "delete", "x")
		self.assertEqual(app.popup.text, "x.bin")
		press(app, "end", "backspace", "enter")
		self.assertEqual(app.session.values["pattern"], "x.bi")

	def test_optional_number_can_be_cleared(self):
		app = tui.App(run_spec.default_values("tspn"))
		app.cursor = [item.key for item in app.fields()].index("external_timeout")
		press(app, "enter", *"90", "enter")
		self.assertEqual(app.session.values["external_timeout"], 90)
		press(app, "enter", "backspace", "backspace", "enter")
		self.assertIsNone(app.session.values["external_timeout"])


class InputGuardTests(WorkspaceCase):
	def field(self, key, problem="fixed-order"):
		app = tui.App(run_spec.default_values(problem))
		app.cursor = [item.key for item in app.fields()].index(key)
		return app

	def test_numeric_prompts_ignore_keys_that_cannot_belong_to_a_number(self):
		app = self.field("oracle_calls")
		press(app, "enter", *["backspace"] * 12, *"12abc;x34")
		self.assertEqual(app.popup.text, "1234")

	def test_expressions_are_typed_previewed_and_stored_as_numbers(self):
		app = self.field("relative_gap", "tspn")
		press(app, "enter", *["backspace"] * 6, *"10 ** -7")
		self.assertEqual(app.popup.check(app.popup.text), "= 1e-7")
		press(app, "enter")
		self.assertIsNone(app.popup)
		self.assertEqual(app.session.values["relative_gap"], 1e-7)

	def test_out_of_range_values_never_reach_the_field(self):
		app = self.field("time_limit")
		press(app, "enter", *["backspace"] * 3, *"-5", "enter")
		self.assertIsInstance(app.popup, tui.LineInput)
		self.assertIn("-1", app.popup.error)
		press(app, "esc")
		self.assertEqual(app.session.values["time_limit"], 600)
		self.assertEqual(app.session.errors()[:1], ["Campaign: choose one"])
		press(app, "enter", *["backspace"] * 3, "-", "1", "enter")
		self.assertEqual(app.session.values["time_limit"], -1)

	def test_the_preview_reports_why_a_partial_expression_is_invalid(self):
		app = self.field("time_limit")
		press(app, "enter", "end", " ", "*")
		with self.assertRaises(ValueError):
			app.popup.check(app.popup.text)

	def test_delete_and_shift_enter_clear_the_field_and_start_editing(self):
		for key in ("delete", "backspace", "shift-enter"):
			with self.subTest(key=key):
				app = self.field("pattern")
				press(app, key)
				self.assertEqual(app.popup.text, "")
				type_text(app, "a*.bin")
				press(app, "enter")
				self.assertEqual(app.session.values["pattern"], "a*.bin")

	def test_clearing_an_optional_number_restores_its_default(self):
		app = self.field("file_timeout")
		press(app, "enter", *"30", "enter")
		self.assertEqual(app.session.values["file_timeout"], 30)
		press(app, "delete", "enter")
		self.assertIsNone(app.session.values["file_timeout"])

	def test_shift_enter_on_a_choice_opens_its_menu_and_on_an_action_runs_it(self):
		app = tui.App()
		press(app, "shift-enter")
		self.assertIsInstance(app.popup, tui.Menu)
		press(app, "esc")
		app.cursor = len(app.fields()) + 5
		press(app, "enter")
		self.assertEqual(app.result, ("quit", ""))


class ExpressionMemoryTests(WorkspaceCase):
	def gap_app(self, problem="tspn"):
		app = tui.App(run_spec.default_values(problem))
		app.cursor = [item.key for item in app.fields()].index("relative_gap")
		return app

	def enter_gap(self, app, text):
		press(app, "delete")
		type_text(app, text)
		press(app, "enter")

	def test_editing_again_shows_the_expression_not_its_result(self):
		app = self.gap_app()
		self.enter_gap(app, "10 ** -7")
		self.assertEqual(app.session.values["relative_gap"], 1e-7)
		press(app, "enter")
		self.assertEqual(app.popup.text, "10 ** -7")
		press(app, "esc")

	def test_a_plain_number_or_a_changed_value_forgets_the_expression(self):
		app = self.gap_app()
		self.enter_gap(app, "1 / 100")
		self.enter_gap(app, "0.01")
		press(app, "enter")
		self.assertEqual(app.popup.text, "0.01")
		press(app, "esc")
		self.enter_gap(app, "1 / 1000")
		app.session.values["relative_gap"] = 0.5  # changed by something else
		self.assertIsNone(app.session.source_for("relative_gap"))

	def test_expressions_survive_presets_and_the_saved_last_state(self):
		state = tui.load_state()
		app = tui.App(run_spec.default_values("tspn"), state)
		app.cursor = [item.key for item in app.fields()].index("relative_gap")
		self.enter_gap(app, "10 ** -4")
		app.finish("save")
		type_text(app, "p")
		press(app, "enter")
		tui.save_state({**state, "last": app.session.snapshot()})
		reloaded = tui.load_state()
		other = tui.App(tui.initial_values(reloaded), reloaded)
		other.cursor = app.cursor
		press(other, "enter")
		self.assertEqual(other.popup.text, "10 ** -4")
		self.assertNotIn("_expressions", other.session.values)
		self.assertEqual(other.session.command(), app.session.command())

	def test_a_stale_stored_expression_is_dropped(self):
		saved = {
			**run_spec.default_values("tspn"),
			"relative_gap": 0.5,
			"_expressions": {"relative_gap": "10 ** -4"},
		}
		app = tui.App(tui.sanitize(saved))
		self.assertIsNone(app.session.source_for("relative_gap"))

	def test_the_gap_must_be_a_fraction(self):
		app = self.gap_app()
		for text in ("0", "1", "2", "1 - 1", "3 / 2"):
			press(app, "delete")
			type_text(app, text)
			press(app, "enter")
			self.assertIsInstance(app.popup, tui.LineInput, text)
			self.assertIn("less than 1", app.popup.error)
			press(app, "esc")
		self.assertEqual(app.session.values["relative_gap"], 1e-6)
		free = self.gap_app("free-order")
		self.enter_gap(free, "0")
		self.assertEqual(free.session.values["relative_gap"], 0)


class EscapeSequenceTests(unittest.TestCase):
	class Screen:
		def __init__(self, text):
			self.pending = list(text)

		def timeout(self, milliseconds):
			pass

		def get_wch(self):
			if not self.pending:
				raise tui.curses.error("no input")
			return self.pending.pop(0)

	def key(self, text):
		screen = self.Screen(text)
		return tui.read_key(screen)

	def test_known_shift_enter_sequences_and_alt_enter(self):
		for text in ("\x1b[13;2u", "\x1b[27;2;13~", "\x1b\r"):
			with self.subTest(text=text):
				self.assertEqual(self.key(text), "shift-enter")

	def test_cursor_keys_are_understood_in_every_common_encoding(self):
		cases = {
			"\x1b[A": "up",
			"\x1b[B": "down",
			"\x1b[C": "right",
			"\x1b[D": "left",
			"\x1bOA": "up",
			"\x1bOB": "down",
			"\x1bOC": "right",
			"\x1bOD": "left",
			"\x1b[1;1B": "down",
			"\x1b[1;2A": "up",
			"\x1b[H": "home",
			"\x1b[F": "end",
			"\x1b[1~": "home",
			"\x1b[4~": "end",
			"\x1b[3~": "delete",
			"\x1b[13u": "enter",
			"\x1b[127u": "backspace",
		}
		for text, expected in cases.items():
			with self.subTest(text=text):
				self.assertEqual(self.key(text), expected)

	def test_a_lone_escape_stays_escape_and_unknown_sequences_are_ignored(self):
		self.assertEqual(self.key("\x1b"), "esc")
		self.assertEqual(self.key("\x1b[27u"), "esc")
		self.assertEqual(self.key("\x1b[1;9Z"), "unknown")

	def test_ctrl_c_reported_as_a_sequence_still_interrupts(self):
		with self.assertRaises(KeyboardInterrupt):
			self.key("\x1b[99;5u")


class FinishTests(WorkspaceCase):
	def test_print_and_copy_return_the_command(self):
		app = tui.App(run_spec.default_values("tspn"))
		press(app, "p")
		self.assertEqual(app.result, ("print", app.session.command()))
		self.assertTrue(
			app.result[1].startswith("python3 benchmarks/tpp.py bench --problem tspn")
		)

	def test_run_is_blocked_while_the_command_is_invalid(self):
		app = tui.App()
		press(app, "r")
		self.assertIsNone(app.result)
		self.assertIn("Campaign", app.message)
		self.make_campaign("c")
		app.session.set("campaign", "c")
		press(app, "r")
		self.assertEqual(app.result[0], "run")

	def test_quit_produces_no_command(self):
		app = tui.App()
		press(app, "q")
		self.assertEqual(app.result, ("quit", ""))

	def test_presets_round_trip_through_the_workspace(self):
		state = tui.load_state()
		app = tui.App(run_spec.default_values("tspn"), state)
		app.session.set("time_limit", 12)
		app.finish("save")
		type_text(app, "quick")
		press(app, "enter")
		self.assertEqual(sorted(tui.load_state()["presets"]), ["quick"])
		other = tui.App(state=tui.load_state())
		other.finish("load")
		press(other, "enter")
		self.assertEqual(
			(other.problem, other.session.values["time_limit"]), ("tspn", 12)
		)

	def test_corrupt_or_stale_state_falls_back_to_defaults(self):
		tui.state_path().write_text("{not json")
		self.assertEqual(tui.load_state(), {"last": None, "presets": {}})
		tui.state_path().write_text(
			json.dumps(
				{
					"last": {
						"problem": "tspn",
						"solver": "gone",
						"time_limit": "x",
						"bogus": 1,
					},
					"presets": {"bad": {"problem": "nope"}},
				}
			)
		)
		state = tui.load_state()
		self.assertEqual(state["last"], run_spec.default_values("tspn"))
		self.assertEqual(state["presets"], {})

	def test_clipboard_tool_absence_is_reported_not_raised(self):
		with patch.object(tui, "clipboard_command", return_value=None):
			self.assertIsNone(tui.copy_to_clipboard("x"))


@unittest.skipUnless(hasattr(os, "fork"), "needs a pty")
class TerminalSmokeTest(WorkspaceCase):
	def run_keys(self, keys, wait=2.0):
		pid, descriptor = pty.fork()
		if pid == 0:
			os.environ["TERM"] = "xterm"
			os.execv(
				sys.executable, [sys.executable, str(ROOT / "benchmarks/tpp.py"), "tui"]
			)
		output = b""

		def read(seconds):
			nonlocal output
			end = time.time() + seconds
			while time.time() < end:
				ready, _, _ = select.select([descriptor], [], [], 0.1)
				if ready:
					try:
						chunk = os.read(descriptor, 65536)
					except OSError:
						return
					if not chunk:
						return
					output += chunk

		read(1.0)
		for key in keys:
			os.write(descriptor, key.encode())
			read(0.2)
		read(wait)
		os.waitpid(pid, 0)
		return output.decode(errors="replace")

	def test_real_curses_session_prints_the_composed_command(self):
		self.make_campaign("smoke")
		# Enter opens the problem menu, two Down + Enter pick TSPN, then p prints.
		text = self.run_keys(["\r", "\x1bOB", "\x1bOB", "\r", "p"])
		self.assertIn("python3 benchmarks/tpp.py bench --problem tspn", text)
		self.assertTrue(tui.state_path().exists())


if __name__ == "__main__":
	unittest.main()
