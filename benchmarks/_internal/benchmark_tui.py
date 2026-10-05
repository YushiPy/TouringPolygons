"""Terminal UI that builds a ``tpp.py bench`` command, then prints, copies or runs it.

Only the standard library (``curses``) is used, so it works before ``setup``.
The interface state lives in ``App`` and reacts to key *names* ("up", "enter",
"a"); ``draw`` and ``read_key`` are the only parts that touch curses.
"""

from __future__ import annotations

import curses
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import run_spec
from run_spec import BOOL, CAMPAIGN, CHOICE, MULTI

import workspace

ROOT = Path(__file__).resolve().parents[2]
TPP = ROOT / "benchmarks/tpp.py"
STATE_NAME = "tui-state.json"
OTHER = "\0other"

ACTIONS = (
	("run", "Run now"),
	("copy", "Copy command to clipboard"),
	("print", "Print command and exit"),
	("save", "Save preset..."),
	("load", "Load preset..."),
	("quit", "Quit without a command"),
)
HINTS = "Up/Down move  Enter edit  Del clear+edit  Left/Right change  Space toggle  r run  c copy  p print  q quit"


# --- persistence -------------------------------------------------------------


def state_path() -> Path:
	return workspace.root() / STATE_NAME


def _plain(values: dict) -> dict:
	return {
		key: list(value) if isinstance(value, tuple) else value
		for key, value in values.items()
	}


def sanitize(saved: object) -> dict | None:
	"""Saved values reduced to what the schema accepts; None when unusable."""
	if not isinstance(saved, dict) or saved.get("problem") not in run_spec.PROBLEMS:
		return None
	problem = saved["problem"]
	values = run_spec.default_values(problem)
	for item in run_spec.applicable(problem):
		if item.key == "problem" or item.key not in saved:
			continue
		spec, value, kind = (
			item.spec(problem),
			saved[item.key],
			run_spec.kind_of(item, problem),
		)
		if (
			kind == MULTI
			and isinstance(value, list)
			and all(entry in spec.choices for entry in value)
		):
			values[item.key] = tuple(value)
		elif (
			kind == BOOL
			and isinstance(value, bool)
			or kind == CHOICE
			and value in spec.choices
			or kind in (run_spec.INTEGER, run_spec.NUMBER)
			and (
				value is None
				or (isinstance(value, (int, float)) and not isinstance(value, bool))
			)
			or kind in (run_spec.TEXT, CAMPAIGN)
			and (value is None or isinstance(value, str))
		):
			values[item.key] = value
	expressions = saved.get("_expressions")
	if isinstance(expressions, dict):
		kept = {
			key: text
			for key, text in expressions.items()
			if key in values and isinstance(text, str)
		}
		if kept:
			values["_expressions"] = kept
	return values


def load_state() -> dict:
	try:
		state = json.loads(state_path().read_text())
	except (OSError, ValueError):
		return {"last": None, "presets": {}}
	presets = {
		name: clean
		for name, saved in (state.get("presets") or {}).items()
		if (clean := sanitize(saved))
	}
	return {"last": sanitize(state.get("last")), "presets": presets}


def save_state(state: dict) -> None:
	"""Best-effort: a read-only workspace must not break the tool."""
	try:
		path = state_path()
		path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"last": _plain(state["last"]) if state["last"] else None,
			"presets": {
				name: _plain(values) for name, values in state["presets"].items()
			},
		}
		path.write_text(json.dumps(payload, indent=2) + "\n")
	except OSError:
		pass


# --- actions that leave the UI ----------------------------------------------


def clipboard_command() -> list[str] | None:
	for command in (
		["pbcopy"],
		["wl-copy"],
		["xclip", "-selection", "clipboard"],
		["xsel", "--clipboard", "--input"],
		["clip.exe"],
	):
		if shutil.which(command[0]):
			return command
	return None


def copy_to_clipboard(text: str) -> str | None:
	"""Copy text; return the tool used, or None when no clipboard tool exists."""
	command = clipboard_command()
	if command is None:
		return None
	try:
		subprocess.run(command, input=text.encode(), check=True)
	except (OSError, subprocess.CalledProcessError):
		return None
	return command[0]


# --- interface state ---------------------------------------------------------


@dataclass
class Menu:
	title: str
	options: list[tuple[object, str]]
	index: int = 0
	multi: bool = False
	checked: set | None = None


@dataclass
class LineInput:
	title: str
	text: str = ""
	position: int = 0
	error: str = ""
	allowed: str | None = None  # characters that may be typed; None means any
	check: Callable[[str], str] | None = (
		None  # live preview; raises ValueError when invalid
	)


class App:
	def __init__(self, values: dict | None = None, state: dict | None = None):
		self.state = state or {"last": None, "presets": {}}
		self.session = run_spec.Session()
		self.session.restore(values or run_spec.default_values(run_spec.FIXED))
		self.cursor = 0
		self.popup: Menu | LineInput | None = None
		self.on_submit: Callable[[object], None] | None = None
		self.message = ""
		self.result: tuple[str, str] | None = None  # (action, command)

	# structure
	@property
	def problem(self) -> str:
		return self.session.values["problem"]

	def fields(self) -> list[run_spec.Field]:
		return run_spec.applicable(self.problem)

	def selectable(self) -> list[tuple[str, object]]:
		return [("field", item) for item in self.fields()] + [
			("action", entry) for entry in ACTIONS
		]

	def current(self) -> tuple[str, object]:
		return self.selectable()[self.cursor]

	def move(self, step: int) -> None:
		self.cursor = (self.cursor + step) % len(self.selectable())

	def show(self, item: run_spec.Field) -> str:
		value = self.session.values.get(item.key)
		if item.key == "problem":
			return run_spec.PROBLEMS[value]
		if item.key == "time_limit" and value == run_spec.UNLIMITED:
			return "unlimited"
		if item.key == "campaign" and not value:
			return "<choose a campaign>"
		return run_spec.format_value(value)

	def help_text(self) -> str:
		kind, target = self.current()
		if kind == "field":
			return target.spec(self.problem).help
		return {
			"run": "Runs the command now in this terminal.",
			"copy": "Copies the command, prints it and exits.",
			"print": "Prints the command and exits.",
			"save": "Stores these values under a name.",
			"load": "Replaces these values with a saved preset.",
			"quit": "Leaves without producing a command.",
		}[target[0]]

	# input
	def handle(self, key: str) -> None:
		self.message = ""
		if isinstance(self.popup, LineInput):
			self._input_key(key)
		elif isinstance(self.popup, Menu):
			self._menu_key(key)
		else:
			self._main_key(key)

	def _main_key(self, key: str) -> None:
		if key in ("up", "k"):
			self.move(-1)
		elif key in ("down", "j"):
			self.move(1)
		elif key == "home":
			self.cursor = 0
		elif key == "end":
			self.cursor = len(self.selectable()) - 1
		elif key in ("enter", " "):
			self.activate(toggle_only=key == " ")
		elif key in ("delete", "backspace", "shift-enter"):
			self.activate(clear=True)
		elif key in ("left", "right"):
			self.step_value(1 if key == "right" else -1)
		elif key in ("r", "c", "p", "q"):
			self.finish({"r": "run", "c": "copy", "p": "print", "q": "quit"}[key])
		elif len(key) == 1 and key.isdigit() and 1 <= int(key) <= len(self.fields()):
			self.cursor = int(key) - 1

	def step_value(self, step: int) -> None:
		kind, target = self.current()
		if kind != "field":
			return
		spec, value = target.spec(self.problem), self.session.values.get(target.key)
		if run_spec.kind_of(target, self.problem) == BOOL:
			self.session.set(target.key, not value)
		elif run_spec.kind_of(target, self.problem) == CHOICE:
			options = list(spec.choices)
			self.session.set(
				target.key, options[(options.index(value) + step) % len(options)]
			)

	def activate(self, toggle_only: bool = False, clear: bool = False) -> None:
		kind, target = self.current()
		if kind == "action":
			if not toggle_only:
				self.finish(target[0])
			return
		item, spec = target, target.spec(self.problem)
		item_kind = run_spec.kind_of(item, self.problem)
		if item_kind == BOOL:
			self.session.set(item.key, not self.session.values.get(item.key))
		elif toggle_only:
			return
		elif item_kind == CHOICE:
			options = [
				(choice, run_spec.PROBLEMS[choice] if item.key == "problem" else choice)
				for choice in spec.choices
			]
			value = self.session.values.get(item.key)
			self.open_menu(
				Menu(
					item.label, options, [choice for choice, _ in options].index(value)
				),
				lambda chosen: self.session.set(item.key, chosen),
			)
		elif item_kind == MULTI:
			options = [(choice, choice) for choice in spec.choices]
			self.open_menu(
				Menu(
					item.label,
					options,
					multi=True,
					checked=set(self.session.values.get(item.key) or ()),
				),
				lambda chosen: self.session.set(
					item.key,
					tuple(choice for choice in spec.choices if choice in chosen),
				),
			)
		elif item_kind == CAMPAIGN and (names := run_spec.available_campaigns()):
			options = [(name, name) for name in names] + [
				(OTHER, "Other: type a name or path...")
			]
			value = self.session.values.get(item.key)
			self.open_menu(
				Menu(item.label, options, names.index(value) if value in names else 0),
				lambda chosen: (
					self.edit_text(item)
					if chosen == OTHER
					else self.session.set(item.key, chosen)
				),
			)
		else:
			self.edit_text(item, clear=clear)

	def edit_text(self, item: run_spec.Field, clear: bool = False) -> None:
		"""Open the text prompt, empty when ``clear`` (Delete or Shift+Enter on the row)."""
		current = self.session.values.get(item.key)
		text = ""
		if not clear and current is not None:
			# Show the expression the value was typed as (10 ** -7), not its result.
			text = self.session.source_for(item.key) or run_spec.format_value(
				current, compact=True
			)
		allowed = run_spec.allowed_characters(item, self.problem)
		prompt = LineInput(
			item.label,
			text,
			len(text),
			allowed=allowed,
			check=(lambda entered: self._preview(item, entered)) if allowed else None,
		)
		self.popup, self.on_submit = (
			prompt,
			lambda entered: self._commit_text(item, entered),
		)

	def _preview(self, item: run_spec.Field, entered: str) -> str:
		"""What the typed text means, e.g. ``= 1e-7`` for ``10 ** -7``."""
		return "= " + run_spec.format_value(
			run_spec.parse_value(item, self.problem, entered)
		)

	def _commit_text(self, item: run_spec.Field, entered: str) -> bool:
		try:
			value = run_spec.parse_value(item, self.problem, entered)
		except ValueError as error:
			self.popup.error = str(error)
			return False
		typed = run_spec.allowed_characters(item, self.problem) is not None
		self.session.set(item.key, value, source=entered if typed else None)
		return True

	def open_menu(self, menu: Menu, then: Callable[[object], None]) -> None:
		self.popup, self.on_submit = menu, then

	def _menu_key(self, key: str) -> None:
		menu = self.popup
		if key in ("up", "k"):
			menu.index = (menu.index - 1) % len(menu.options)
		elif key in ("down", "j"):
			menu.index = (menu.index + 1) % len(menu.options)
		elif key == "esc" or key == "q":
			self.popup = None
		elif menu.multi and key == " ":
			choice = menu.options[menu.index][0]
			menu.checked.symmetric_difference_update({choice})
		elif key == "enter":
			chosen, then = (
				(menu.checked if menu.multi else menu.options[menu.index][0]),
				self.on_submit,
			)
			self.popup = None
			then(chosen)

	def _input_key(self, key: str) -> None:
		prompt = self.popup
		text, at = prompt.text, prompt.position
		if key == "esc":
			self.popup = None
		elif key == "enter":
			# A submit that opens a follow-up prompt replaces self.popup itself.
			if self.on_submit(prompt.text) is not False and self.popup is prompt:
				self.popup = None
		elif key == "left":
			prompt.position = max(0, at - 1)
		elif key == "right":
			prompt.position = min(len(text), at + 1)
		elif key == "home":
			prompt.position = 0
		elif key == "end":
			prompt.position = len(text)
		elif key == "backspace" and at:
			prompt.text, prompt.position = text[: at - 1] + text[at:], at - 1
		elif key == "delete":
			prompt.text = text[:at] + text[at + 1 :]
		elif (
			len(key) == 1
			and key.isprintable()
			and (prompt.allowed is None or key in prompt.allowed)
		):
			prompt.text, prompt.position = text[:at] + key + text[at:], at + 1
		if isinstance(self.popup, LineInput) and key not in ("enter", "esc"):
			self.popup.error = ""

	# leaving
	def finish(self, action: str) -> None:
		if action == "quit":
			self.result = ("quit", "")
		elif action == "save":
			self.ask_name("Save preset as", self._save_preset)
		elif action == "load":
			names = sorted(self.state["presets"])
			if not names:
				self.message = "No presets saved yet."
			else:
				self.open_menu(
					Menu("Load preset", [(name, name) for name in names]),
					self._load_preset,
				)
		else:
			errors = self.session.errors()
			if action == "run" and errors:
				self.message = "Cannot run: " + errors[0]
				return
			self.result = (action, self.session.command())

	def ask_name(self, title: str, then: Callable[[str], None]) -> None:
		def submit(entered: str) -> bool:
			name = entered.strip()
			if not name:
				self.popup.error = "a name is required"
				return False
			then(name)
			return True

		self.popup, self.on_submit = LineInput(title), submit

	def _save_preset(self, name: str) -> None:
		self.state["presets"][name] = self.session.snapshot()
		save_state(self.state)
		self.message = f"Saved preset {name!r}."

	def _load_preset(self, name: object) -> None:
		self.session.restore(self.state["presets"][name])
		self.cursor = min(self.cursor, len(self.selectable()) - 1)
		self.message = f"Loaded preset {name!r}."


# --- curses ------------------------------------------------------------------

KEYS = {
	curses.KEY_UP: "up",
	curses.KEY_DOWN: "down",
	curses.KEY_LEFT: "left",
	curses.KEY_RIGHT: "right",
	curses.KEY_ENTER: "enter",
	curses.KEY_BACKSPACE: "backspace",
	curses.KEY_DC: "delete",
	curses.KEY_HOME: "home",
	curses.KEY_END: "end",
	curses.KEY_RESIZE: "resize",
}
CHARACTERS = {
	"\n": "enter",
	"\r": "enter",
	"\x1b": "esc",
	"\x7f": "backspace",
	"\b": "backspace",
}


# Normally curses decodes keys through terminfo. Some terminals send another encoding
# (ESC [ A instead of ESC O A, modifier forms such as ESC [ 1 ; 2 A, or the kitty
# protocol's ESC [ 13 ; 2 u), which arrives here as ESC followed by characters, so
# every common variant is decoded explicitly.
DIRECTIONS = {
	"A": "up",
	"B": "down",
	"C": "right",
	"D": "left",
	"H": "home",
	"F": "end",
}
TILDE_KEYS = {"1": "home", "7": "home", "4": "end", "8": "end", "3": "delete"}
CSI_U_KEYS = {"27": "esc", "127": "backspace", "9": "unknown"}
CONTROL_SEQUENCE = re.compile(r"^\[(\d+)?(?:;(\d+))?([A-Za-z~])$")
# Opt-in: ask terminals that implement the kitty keyboard protocol (kitty, WezTerm,
# Ghostty, foot...) to report Shift+Enter distinctly. It also changes how those terminals
# encode other keys, so it is off unless TPP_TUI_KITTY_KEYS=1.
KITTY_ENABLE, KITTY_DISABLE = "\x1b[>1u", "\x1b[<u"


def decode_escape(rest: str) -> str:
	"""Name the key sent as ESC followed by ``rest`` ("" is a lone Esc key)."""
	if not rest:
		return "esc"
	if rest in ("\r", "\n"):  # Alt+Enter
		return "shift-enter"
	if len(rest) == 2 and rest[0] == "O":  # application-mode cursor keys
		return DIRECTIONS.get(rest[1], "unknown")
	if rest == "[27;2;13~":  # xterm modifyOtherKeys: Shift+Enter
		return "shift-enter"
	match = CONTROL_SEQUENCE.match(rest)
	if match is None:
		return "unknown"
	number, modifier, final = match.groups()
	if final in DIRECTIONS:
		return DIRECTIONS[final]
	if final == "~":
		return TILDE_KEYS.get(number or "", "unknown")
	if final == "u":
		if (number, modifier) == (
			"99",
			"5",
		):  # Ctrl+C when keys are reported as sequences
			raise KeyboardInterrupt
		if number == "13":
			return "shift-enter" if modifier == "2" else "enter"
		return CSI_U_KEYS.get(number or "", "unknown")
	return "unknown"


def read_escape(screen) -> str:
	"""Read the rest of a key that started with ESC and name it."""
	screen.timeout(40)
	try:
		rest = ""
		while True:
			try:
				character = screen.get_wch()
			except curses.error:
				break
			if isinstance(character, int):
				break
			rest += character
	finally:
		screen.timeout(-1)
	return decode_escape(rest)


def read_key(screen) -> str:
	key = screen.get_wch()
	if isinstance(key, int):
		return KEYS.get(key, "unknown")
	if key == "\x1b":
		return read_escape(screen)
	return CHARACTERS.get(key, key)


def put(window, row: int, column: int, text: str, attribute: int = 0) -> None:
	height, width = window.getmaxyx()
	if 0 <= row < height and column < width:
		try:
			window.addnstr(row, column, text, max(0, width - column - 1), attribute)
		except curses.error:
			pass


def draw(screen, app: App, styles: dict[str, int]) -> None:
	screen.erase()
	height, width = screen.getmaxyx()
	fields = app.fields()
	values = app.session.values
	label_width = max(len(item.label) for item in fields) + 4

	footer = [("Command:", styles["bold"])]
	footer += [
		(f"  {line}", 0)
		for line in textwrap.wrap(
			app.session.command(), max(20, width - 4), break_long_words=False
		)
	]
	for error in app.session.errors():
		footer.append((f"! {error}", styles["error"]))
	if app.message:
		footer.append((app.message, styles["ok"]))
	footer += [(f"? {app.help_text()}", styles["dim"]), (HINTS, styles["dim"])]
	body_height = max(1, height - len(footer) - 2)

	lines: list[tuple[str, int, int | None]] = []
	for number, item in enumerate(fields):
		lines.append(
			(
				f"{number + 1:>2}. {item.label + ':':<{label_width}}{app.show(item)}",
				0,
				number,
			)
		)
	lines += [
		("", 0, None),
		(
			f"    {'Instances:':<{label_width}}{run_spec.instance_source(values)}",
			styles["dim"],
			None,
		),
		(
			f"    {'Output:':<{label_width}}{run_spec.output_location(values)}",
			styles["dim"],
			None,
		),
		("", 0, None),
	]
	for number, (_, label) in enumerate(ACTIONS):
		lines.append((f"    [ {label} ]", 0, len(fields) + number))

	focused = next(index for index, line in enumerate(lines) if line[2] == app.cursor)
	top = max(0, min(focused - body_height // 2, len(lines) - body_height))
	put(screen, 0, 0, "Touring Polygons: build a benchmark command", styles["bold"])
	for row, (text, style, selectable) in enumerate(lines[top : top + body_height]):
		put(
			screen,
			row + 1,
			0,
			text,
			styles["selected"] if selectable == app.cursor else style,
		)
	for row, (text, style) in enumerate(footer):
		put(screen, height - len(footer) + row, 0, text, style)

	screen.noutrefresh()
	if app.popup is not None:
		draw_popup(screen, app, styles)
	curses.doupdate()


def draw_popup(screen, app: App, styles: dict[str, int]) -> None:
	height, width = screen.getmaxyx()
	popup = app.popup
	if isinstance(popup, Menu):
		labels = [
			(
				"[x] "
				if popup.multi and value in popup.checked
				else "[ ] "
				if popup.multi
				else ""
			)
			+ label
			for value, label in popup.options
		]
		hint = (
			"Space marks, Enter confirms, Esc cancels"
			if popup.multi
			else "Enter selects, Esc cancels"
		)
		inner = (
			max(len(popup.title) + 2, len(hint), *(len(label) for label in labels)) + 2
		)
		visible = min(len(labels), max(1, height - 6))
		first = max(0, min(popup.index - visible // 2, len(labels) - visible))
		rows = [
			(labels[i], styles["selected"] if i == popup.index else 0)
			for i in range(first, first + visible)
		]
	else:
		hint = "Enter confirms, Esc cancels"
		feedback, style = popup.error, styles["error"]
		if popup.check is not None:
			hint = "Enter confirms, Esc cancels. Math ok: 10 ** -7, 1 / 100"
			if not feedback and popup.text.strip():
				try:
					feedback, style = popup.check(popup.text), styles["ok"]
				except ValueError as error:
					feedback = str(error)
		inner = max(
			40,
			len(popup.title) + 4,
			len(feedback) + 2,
			len(hint) + 2,
			min(width - 6, len(popup.text) + 8),
		)
		rows = [(popup.text, 0), (feedback, style)]
	inner = min(inner, max(10, width - 4))
	box_height, box_width = len(rows) + 4, inner + 2
	window = curses.newwin(
		min(box_height, height),
		min(box_width, width),
		max(0, (height - box_height) // 2),
		max(0, (width - box_width) // 2),
	)
	window.erase()
	window.border("|", "|", "-", "-", "+", "+", "+", "+")
	put(window, 0, 2, f" {popup.title} ", styles["bold"])
	for number, (text, style) in enumerate(rows):
		put(window, number + 1, 2, text, style)
	put(window, len(rows) + 2, 2, hint, styles["dim"])
	if isinstance(popup, LineInput):
		try:
			curses.curs_set(1)
			window.move(1, 2 + min(popup.position, inner - 4))
		except curses.error:
			pass
	else:
		try:
			curses.curs_set(0)
		except curses.error:
			pass
	window.noutrefresh()


def make_styles() -> dict[str, int]:
	styles = {
		"bold": curses.A_BOLD,
		"dim": curses.A_DIM,
		"selected": curses.A_REVERSE,
		"error": curses.A_BOLD,
		"ok": curses.A_BOLD,
	}
	if curses.has_colors():
		curses.start_color()
		curses.use_default_colors()
		curses.init_pair(1, curses.COLOR_RED, -1)
		curses.init_pair(2, curses.COLOR_GREEN, -1)
		styles["error"], styles["ok"] = (
			curses.color_pair(1) | curses.A_BOLD,
			curses.color_pair(2),
		)
	return styles


def interact(app: App) -> None:
	def loop(screen) -> None:
		styles = make_styles()
		screen.keypad(True)
		kitty = os.environ.get("TPP_TUI_KITTY_KEYS") == "1"
		if kitty:
			os.write(sys.stdout.fileno(), KITTY_ENABLE.encode())
		try:
			while app.result is None:
				draw(screen, app, styles)
				key = read_key(screen)
				if key not in ("resize", "unknown"):
					app.handle(key)
		finally:
			if kitty:
				os.write(sys.stdout.fileno(), KITTY_DISABLE.encode())

	os.environ.setdefault("ESCDELAY", "25")
	try:
		curses.wrapper(loop)
	except KeyboardInterrupt:
		app.result = ("quit", "")


# --- entry point --------------------------------------------------------------


def debug_keys() -> int:
	"""Show what the terminal sends for each key, to diagnose navigation problems."""
	history: list[str] = []

	def loop(screen) -> None:
		screen.keypad(True)
		while True:
			screen.erase()
			put(
				screen,
				0,
				0,
				"Key diagnostics: press keys; q quits. Shows what curses received, then how the TUI reads it.",
			)
			for row, line in enumerate(history[-15:]):
				put(screen, row + 2, 0, line)
			screen.refresh()
			raw = screen.get_wch()
			if raw == "q":
				return
			if raw == "\x1b":
				screen.timeout(40)
				rest = ""
				try:
					while True:
						rest += screen.get_wch()
				except (curses.error, TypeError):
					pass
				finally:
					screen.timeout(-1)
				history.append(f"received ESC + {rest!r:<14} -> {decode_escape(rest)}")
			else:
				name = (
					KEYS.get(raw, "unknown")
					if isinstance(raw, int)
					else CHARACTERS.get(raw, raw)
				)
				history.append(f"received {raw!r:<20} -> {name}")

	os.environ.setdefault("ESCDELAY", "25")
	curses.wrapper(loop)
	print(
		f"TERM={os.environ.get('TERM')}, TERM_PROGRAM={os.environ.get('TERM_PROGRAM')}"
	)
	print("\n".join(history[-15:]))
	return 0


def initial_values(state: dict) -> dict:
	return state["last"] or run_spec.default_values(run_spec.FIXED)


def main(argv: Sequence[str] | None = None) -> int:
	if list(argv or []) in (["-h"], ["--help"]):
		print(
			"usage: scripts/benchmark.sh [--debug-keys]\n\nInteractive builder for `tpp.py bench` commands. Choose the options with the "
			"arrow keys and Enter, then print, copy or run the command.\nLast values and presets are kept in "
			f"{state_path()}."
		)
		return 0
	if list(argv or []) == ["--debug-keys"] and sys.stdin.isatty():
		return debug_keys()
	if not (sys.stdin.isatty() and sys.stdout.isatty()):
		print(
			"benchmark.sh needs an interactive terminal. Use `tpp.py bench --help` to build the command by hand.",
			file=sys.stderr,
		)
		return 2
	state = load_state()
	app = App(initial_values(state), state)
	interact(app)
	action, command = app.result
	if action == "quit":
		print("No command produced.", file=sys.stderr)
		return 1
	state["last"] = app.session.snapshot()
	save_state(state)
	print(command)
	if action == "copy":
		tool = copy_to_clipboard(command)
		print(
			f"(copied with {tool})"
			if tool
			else "(no clipboard tool found: pbcopy, wl-copy, xclip or xsel; the command is printed above)",
			file=sys.stderr,
		)
	elif action == "run":
		arguments = [
			sys.executable,
			str(TPP),
			"bench",
			*run_spec.to_cli(app.session.values),
		]
		sys.stdout.flush()
		os.execv(arguments[0], arguments)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
