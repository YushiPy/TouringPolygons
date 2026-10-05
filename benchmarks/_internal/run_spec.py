"""One declarative description of a benchmark run, shared by ``tpp.py bench`` and the TUI.

The three kinds of benchmark (fixed-order TPP, free-order TPP and TSPN) are run
by different modules with different flag names. This file lists every option
once, says which problems use it and with what default, and translates a set
of values into the command of the module that implements the problem. The TUI
only edits these values and ``bench`` only parses them, so neither can drift
from the other.

Only the standard library is imported, so the TUI works before ``setup``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import workspace

FIXED, FREE, TSPN = "fixed-order", "free-order", "tspn"
PROBLEMS = {
	FIXED: "TPP (fixed order)",
	FREE: "TPP (free order)",
	TSPN: "TSPN (closed tour)",
}
FIXED_SOLVERS = (
	"default",
	"linear_search_lazy",
	"linear_search_disjoint",
	"binary_search_lazy",
	"binary_search_disjoint",
	"binary_search_eager",
	"tan_jiang",
	"gurobi",
	"directional_maps",
)
TWO_SOLVERS = ("tpp-ours", "tpp-fekete", "both")
CYCLE_OPTIMIZATIONS = (
	"cache",
	"dual",
	"features",
	"lazy",
	"root",
	"branch",
	"one-tree",
	"learn",
	"memo",
	"bound-first",
	"dual-screen",
	"interval",
	"share-bounds",
	"proposal-bound",
	"primal-starts",
)
DEFAULT_CALLS = 100_000_000
TSPN_CAMPAIGN = "tspn-fekete-comparison-v1"
TSPN_WATCHDOG_MARGIN = 15
UNLIMITED = -1

CHOICE, NUMBER, INTEGER, TEXT, BOOL, MULTI, CAMPAIGN = (
	"choice",
	"number",
	"integer",
	"text",
	"bool",
	"multi",
	"campaign",
)


@dataclass(frozen=True)
class Spec:
	"""How one field behaves for one problem."""

	default: object = None
	choices: tuple[str, ...] | None = None
	kind: str | None = None
	help: str = ""
	unlimited: bool = False  # -1 means "no limit"
	upper: float | None = None  # exclusive upper bound
	lower_inclusive: bool = False  # 0 itself is allowed (otherwise values must be > 0)


@dataclass(frozen=True)
class Field:
	key: str
	label: str
	kind: str
	problems: Mapping[str, Spec]

	@property
	def flag(self) -> str:
		return "--" + self.key.replace("_", "-")

	def spec(self, problem: str) -> Spec | None:
		return self.problems.get(problem)


def everywhere(**kwargs) -> dict[str, Spec]:
	return {problem: Spec(**kwargs) for problem in PROBLEMS}


FIELDS: tuple[Field, ...] = (
	Field(
		"problem",
		"Problem",
		CHOICE,
		everywhere(
			default=FIXED, choices=tuple(PROBLEMS), help="Which benchmark to run."
		),
	),
	Field(
		"solver",
		"Solver",
		CHOICE,
		{
			FIXED: Spec(
				"default",
				FIXED_SOLVERS,
				help="Convex solver (TPP_BENCH_SOLVER); 'default' keeps the tool's own.",
			),
			FREE: Spec(
				"tpp-ours",
				TWO_SOLVERS,
				help="tpp-ours, the pinned Fekete solver, or both.",
			),
			TSPN: Spec("both", TWO_SOLVERS, help="tpp-ours needs no Gurobi."),
		},
	),
	Field(
		"campaign",
		"Campaign",
		CAMPAIGN,
		{
			FIXED: Spec(
				None,
				kind=CAMPAIGN,
				help="Instance set under workspace/campaigns; each run is stored in results/<run>.",
			),
			FREE: Spec(
				None,
				kind=CAMPAIGN,
				help="Instance set under workspace/campaigns; each run is stored in results/<run>.",
			),
			TSPN: Spec(
				TSPN_CAMPAIGN,
				kind=TEXT,
				help="Output directory name; the 558 Fekete instances are fixed.",
			),
		},
	),
	Field(
		"repetitions",
		"Repetitions",
		INTEGER,
		{
			FIXED: Spec(1, help="Repetitions of each instance."),
			TSPN: Spec(1, help="Repetitions of each instance per solver."),
		},
	),
	Field(
		"time_limit",
		"Time limit (per instance, s)",
		NUMBER,
		{
			FIXED: Spec(
				600,
				unlimited=True,
				help="Branch-and-bound time cap per instance; -1 means unlimited.",
			),
			FREE: Spec(
				600, unlimited=True, help="Time cap per instance; -1 means unlimited."
			),
			TSPN: Spec(60, help="Native solver time limit per instance."),
		},
	),
	Field(
		"oracle_calls",
		"Oracle calls limit (per instance)",
		INTEGER,
		{
			FIXED: Spec(DEFAULT_CALLS, help="Cap on convex-oracle calls per instance."),
			FREE: Spec(DEFAULT_CALLS, help="Cap on convex-oracle calls per instance."),
			TSPN: Spec(
				DEFAULT_CALLS,
				help="Our solver's call budget; a non-default value requires solver tpp-ours.",
			),
		},
	),
	Field(
		"threads",
		"Threads (per instance)",
		INTEGER,
		{
			FIXED: Spec(1, help="Worker threads inside each benchmark process."),
			FREE: Spec(1, help="Solver threads inside one instance."),
		},
	),
	Field(
		"workers",
		"Parallel workers",
		INTEGER,
		{
			FREE: Spec(1, help="Instances solved concurrently."),
			TSPN: Spec(
				1, help="Case shards run in parallel (each solve keeps one thread)."
			),
		},
	),
	Field(
		"progress_interval",
		"Progress report (s)",
		NUMBER,
		{
			FREE: Spec(
				60,
				lower_inclusive=True,
				help="Seconds between status lines (bounds, calls, queue) of each running instance; 0 turns them off.",
			),
			TSPN: Spec(
				60,
				lower_inclusive=True,
				help="Seconds between status lines (bounds, calls, queue) of each running instance; 0 turns them off.",
			),
		},
	),
	Field(
		"max_instances",
		"Max instances",
		INTEGER,
		{
			FIXED: Spec(
				UNLIMITED,
				unlimited=True,
				help="Only the first N instances; -1 means all.",
			),
			FREE: Spec(
				UNLIMITED,
				unlimited=True,
				help="Only the first N instances; -1 means all.",
			),
		},
	),
	Field(
		"max_polygons",
		"Max polygons",
		INTEGER,
		{
			FIXED: Spec(
				UNLIMITED,
				unlimited=True,
				help="Only the first N polygons of each instance; -1 means all.",
			),
		},
	),
	Field(
		"max_branching",
		"Max branching",
		INTEGER,
		{
			FIXED: Spec(
				UNLIMITED, unlimited=True, help="Branching cap; -1 means none."
			),
		},
	),
	Field(
		"pattern",
		"File pattern",
		TEXT,
		{
			FIXED: Spec(
				"*.bin", help="Filename pattern searched under the campaign inputs."
			),
		},
	),
	Field(
		"file_timeout",
		"File timeout (s)",
		INTEGER,
		{
			FIXED: Spec(None, help="Wall-clock cap per .bin file; empty means none."),
		},
	),
	Field(
		"relative_gap",
		"Relative gap",
		NUMBER,
		{
			FREE: Spec(
				None,
				upper=1,
				lower_inclusive=True,
				help="Target relative gap for our solver, from 0 up to (not including) 1; empty keeps its default.",
			),
			TSPN: Spec(
				1e-6, upper=1, help="Target relative gap, between 0 and 1 (exclusive)."
			),
		},
	),
	Field(
		"external_timeout",
		"Watchdog timeout (s)",
		NUMBER,
		{
			TSPN: Spec(
				None,
				help=f"Wall-clock watchdog per instance; empty means time limit + {TSPN_WATCHDOG_MARGIN}.",
			),
		},
	),
	Field(
		"cycle_optimizations",
		"Cycle optimizations",
		MULTI,
		{
			TSPN: Spec(
				("cache", "features", "root", "interval"),
				CYCLE_OPTIMIZATIONS,
				help="Native opt-ins, combined.",
			),
		},
	),
	Field(
		"portfolio",
		"Portfolio",
		CHOICE,
		{
			TSPN: Spec(
				"none",
				("none", "cooperative", "independent"),
				help="Two-search portfolio (two worker threads).",
			),
		},
	),
	Field(
		"search_strategy",
		"Search strategy",
		CHOICE,
		{
			TSPN: Spec(
				"default",
				("default", "best-bound", "dfs-bfs"),
				help="One isolated B&B strategy; not with a portfolio.",
			),
		},
	),
	Field(
		"capture_oracles",
		"Capture oracle calls",
		BOOL,
		{
			TSPN: Spec(
				False, help="Record every oracle call (diagnostic; adds overhead)."
			),
		},
	),
	Field(
		"resume",
		"Resume",
		BOOL,
		everywhere(
			default=True,
			help="Continue completed work; off restarts (rerun, or moves a TSPN campaign aside).",
		),
	),
	Field(
		"rebuild",
		"Rebuild binaries",
		BOOL,
		{
			FIXED: Spec(
				True, help="Build the native tool first; off uses the existing binary."
			),
			FREE: Spec(
				True, help="Build the native tool first; off uses the existing binary."
			),
		},
	),
	Field(
		"dry_run",
		"Dry run",
		BOOL,
		everywhere(
			default=False, help="Only list what would run; nothing is built or solved."
		),
	),
)
BY_KEY = {item.key: item for item in FIELDS}


def applicable(problem: str) -> list[Field]:
	return [item for item in FIELDS if item.spec(problem) is not None]


def default_values(problem: str) -> dict[str, object]:
	values = {item.key: item.spec(problem).default for item in applicable(problem)}
	values["problem"] = problem
	return values


def switch_problem(values: Mapping[str, object], problem: str) -> dict[str, object]:
	"""Values for another problem, carrying over what the user changed and both problems share.

	A TSPN campaign is only an output name, so it is never exchanged with the
	instance-set campaign of the other problems.
	"""
	previous = values["problem"]
	fresh = default_values(problem)
	for item in applicable(problem):
		old, new = item.spec(previous), item.spec(problem)
		if item.key == "problem" or old is None or item.key not in values:
			continue
		if item.key == "campaign" and TSPN in (previous, problem):
			continue
		if values[item.key] != old.default and _valid_choice(new, values[item.key]):
			fresh[item.key] = values[item.key]
	return fresh


def _valid_choice(spec: Spec, value: object) -> bool:
	if spec.choices is None:
		return True
	if isinstance(value, tuple):
		return all(entry in spec.choices for entry in value)
	return value in spec.choices


def kind_of(item: Field, problem: str) -> str:
	return item.spec(problem).kind or item.kind


# --- parsing and formatting of values ---------------------------------------


def range_error(spec: Spec, kind: str, value: int | float) -> str | None:
	"""Why ``value`` is outside the field's range, or None when it is allowed."""
	if spec.unlimited and value == UNLIMITED:
		return None
	above = value >= 0 if spec.lower_inclusive else value > 0
	if above and (spec.upper is None or value < spec.upper):
		return None
	if spec.upper is not None:
		start = "at least 0" if spec.lower_inclusive else "greater than 0"
		return f"must be {start} and less than {format_value(spec.upper)}"
	noun = "whole number" if kind == INTEGER else "number"
	if spec.lower_inclusive:
		return f"must be 0 or a positive {noun}"
	return f"must be a positive {noun}" + (
		" or -1 (no limit)" if spec.unlimited else ""
	)


MATH_CHARACTERS = "0123456789.eE_+-*/() "
MAX_EXPONENT = 400
MAX_INTEGER_BITS = 4096


def evaluate(text: str) -> int | float:
	"""Value of an arithmetic expression such as ``10 ** -7`` or ``(10 - 5) * 2 / 3``.

	The text is parsed, never executed: only numbers, ``+ - * / **`` and
	parentheses are accepted, and exponents are bounded so a typo like
	``9 ** 9 ** 9`` fails instead of hanging.
	"""
	try:
		tree = ast.parse(text.strip(), mode="eval")
	except SyntaxError:
		raise ValueError("not a valid expression") from None

	def number(node: ast.AST) -> int | float:
		if isinstance(node, ast.Constant) and type(node.value) in (int, float):
			return node.value
		if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
			operand = number(node.operand)
			return -operand if isinstance(node.op, ast.USub) else operand
		if isinstance(node, ast.BinOp) and isinstance(
			node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
		):
			left, right = number(node.left), number(node.right)
			try:
				if isinstance(node.op, ast.Pow):
					if abs(right) > MAX_EXPONENT or (
						isinstance(left, int)
						and left.bit_length() * abs(right) > MAX_INTEGER_BITS
					):
						raise ValueError("number too large")
					result = left**right
				elif isinstance(node.op, ast.Div):
					result = left / right
				elif isinstance(node.op, ast.Mult):
					result = left * right
				elif isinstance(node.op, ast.Add):
					result = left + right
				else:
					result = left - right
			except ZeroDivisionError:
				raise ValueError("division by zero") from None
			except OverflowError:
				raise ValueError("number too large") from None
			if isinstance(result, complex) or (
				isinstance(result, int) and result.bit_length() > MAX_INTEGER_BITS
			):
				raise ValueError("number too large or not real")
			return result
		raise ValueError("only numbers and + - * / ** ( ) are allowed")

	value = number(tree.body)
	if not math.isfinite(value):
		raise ValueError("number too large")
	return value


def allowed_characters(item: Field, problem: str) -> str | None:
	"""Characters worth typing into a field, or None when anything goes."""
	return MATH_CHARACTERS if kind_of(item, problem) in (INTEGER, NUMBER) else None


def instance_limit(values: Mapping[str, object]) -> int | None:
	"""How many instances the chosen campaign lets ``Max instances`` pick, or None when unknown.

	Free-order runs take the first N of all the campaign's instances; fixed-order runs
	apply N to each input file, so the largest file sets the limit.
	"""
	if values.get("problem") not in (FIXED, FREE) or not values.get("campaign"):
		return None
	try:
		metadata = json.loads(
			(
				workspace.campaign_path(str(values["campaign"])) / workspace.CAMPAIGN_FILE
			).read_text()
		)
		counts = [int(record["instances"]) for record in metadata["inputs"]]
	except (OSError, ValueError, KeyError, TypeError):
		return None
	if not counts:
		return None
	return sum(counts) if values["problem"] == FREE else max(counts)


def parse_value(
	item: Field, problem: str, text: str, limit: int | None = None
) -> object:
	"""Turn text typed by a user (or found on a command line) into a valid field value.

	Numeric fields accept arithmetic expressions. A value outside the field's
	range is rejected here, so no invalid number ever reaches the form.
	"""
	spec, kind = item.spec(problem), kind_of(item, problem)
	text = text.strip()
	if kind in (INTEGER, NUMBER):
		if text == "":
			if spec.default is None:
				return None
			raise ValueError("a value is required")
		value = evaluate(text)
		whole = float(value).is_integer() and abs(value) < 10**15
		if kind == INTEGER and not whole:
			raise ValueError("expected a whole number")
		if whole:
			value = int(value)
		if (problem_error := range_error(spec, kind, value)) is not None:
			raise ValueError(problem_error)
		if limit is not None and value != UNLIMITED and value > limit:
			raise ValueError(f"the campaign has only {limit} instance(s)")
		return value
	if kind == MULTI:
		names = tuple(entry.strip() for entry in text.split(",") if entry.strip())
		unknown = [entry for entry in names if entry not in spec.choices]
		if unknown:
			raise ValueError(f"unknown value(s): {', '.join(unknown)}")
		return names
	if kind == BOOL:
		if text.lower() in {"true", "yes", "on", "1"}:
			return True
		if text.lower() in {"false", "no", "off", "0"}:
			return False
		raise ValueError("expected true or false")
	if kind == CHOICE:
		if text not in spec.choices:
			raise ValueError(f"choose one of: {', '.join(spec.choices)}")
		return text
	return text or None if spec.default is None else text


def format_value(value: object, *, compact: bool = False) -> str:
	if value is None:
		return "" if compact else "(default)"
	if isinstance(value, bool):
		return "true" if value else "false"
	if isinstance(value, tuple):
		return ",".join(value)
	if isinstance(value, int) and not compact and abs(value) >= 10_000:
		return f"{value:_}"
	if isinstance(value, float):
		return re.sub(r"e([+-])0+(\d)", r"e\1\2", repr(value))
	return str(value)


def validate(values: Mapping[str, object]) -> list[str]:
	"""Reasons this set of values cannot run; empty when it can."""
	problem = values.get("problem")
	if problem not in PROBLEMS:
		return [f"unknown problem {problem!r}"]
	errors = []
	for item in applicable(problem):
		if item.key == "problem":
			continue
		spec, value = item.spec(problem), values.get(item.key)
		kind = kind_of(item, problem)
		if item.key == "campaign":
			if not value:
				errors.append("Campaign: choose one")
			elif (
				kind == CAMPAIGN
				and not (
					workspace.campaign_path(str(value)) / workspace.CAMPAIGN_FILE
				).exists()
			):
				errors.append(
					f"Campaign: no campaign.json in {workspace.campaign_path(str(value))}"
				)
		elif kind in (INTEGER, NUMBER) and value is not None:
			if (message := range_error(spec, kind, value)) is not None:
				errors.append(f"{item.label}: {message}")
			elif (
				item.key == "max_instances"
				and value != UNLIMITED
				and (limit := instance_limit(values)) is not None
				and value > limit
			):
				errors.append(f"{item.label}: the campaign has only {limit} instance(s)")
		if kind == CHOICE and value not in spec.choices:
			errors.append(f"{item.label}: invalid choice {value!r}")
	if problem == TSPN:
		if (
			values.get("oracle_calls") != DEFAULT_CALLS
			and values.get("solver") != "tpp-ours"
		):
			errors.append(
				"Oracle calls limit: a non-default value requires solver tpp-ours"
			)
		if (
			values.get("search_strategy") not in (None, "default")
			and values.get("portfolio") != "none"
		):
			errors.append("Search strategy: cannot be combined with a portfolio")
	return errors


def instance_source(values: Mapping[str, object]) -> str:
	"""Where the instances come from, for display."""
	if values["problem"] == TSPN:
		return "558 Fekete instances (third_party/tspn-socg)"
	campaign = values.get("campaign")
	return (
		str(workspace.campaign_path(str(campaign)) / "inputs")
		if campaign
		else "(choose a campaign)"
	)


def output_location(values: Mapping[str, object]) -> str:
	campaign = values.get("campaign")
	if not campaign:
		return "(choose a campaign)"
	if values["problem"] == TSPN:
		return str(workspace.campaigns_dir() / str(campaign))
	return str(workspace.campaign_path(str(campaign)) / "results" / "<run>")


# --- the canonical command line ---------------------------------------------


def to_cli(values: Mapping[str, object]) -> list[str]:
	"""``tpp.py bench`` arguments; only values that differ from the problem's defaults."""
	problem = values["problem"]
	arguments = ["--problem", problem]
	for item in applicable(problem):
		if item.key == "problem" or item.key not in values:
			continue
		spec, value = item.spec(problem), values[item.key]
		if value == spec.default and item.key != "campaign":
			continue
		if value is None:
			continue
		if kind_of(item, problem) == BOOL:
			arguments.append(item.flag if value else "--no-" + item.flag[2:])
		else:
			arguments += [item.flag, format_value(value, compact=True)]
	return arguments


def command_line(values: Mapping[str, object]) -> str:
	return shlex.join(["python3", "benchmarks/tpp.py", "bench", *to_cli(values)])


def make_parser() -> argparse.ArgumentParser:
	"""A parser generated from FIELDS; unset options stay None until resolved."""
	parser = argparse.ArgumentParser(
		prog="tpp.py bench",
		description="Run a benchmark from one set of options. Run scripts/benchmark.sh to build the command interactively.",
	)
	for item in FIELDS:
		helps = {
			problem: spec.help for problem, spec in item.problems.items() if spec.help
		}
		if len(set(helps.values())) == 1:
			text = next(iter(helps.values()))
			if len(helps) < len(PROBLEMS):
				text += f" (only: {', '.join(sorted(helps))})"
		else:
			text = "; ".join(f"{problem}: {help}" for problem, help in helps.items())
		if item.kind == BOOL:
			parser.add_argument(
				item.flag,
				action=argparse.BooleanOptionalAction,
				default=None,
				help=text,
			)
		else:
			parser.add_argument(
				item.flag, default=None, metavar=item.key.upper(), help=text
			)
	return parser


def from_cli(argv: list[str]) -> dict[str, object]:
	"""Parse ``bench`` arguments into a complete, validated set of values."""
	parser = make_parser()
	given = {
		key: value
		for key, value in vars(parser.parse_args(argv)).items()
		if value is not None
	}
	problem = given.get("problem", FIXED)
	if problem not in PROBLEMS:
		parser.error(f"--problem: choose one of {', '.join(PROBLEMS)}")
	values = default_values(problem)
	for key, text in given.items():
		item = BY_KEY[key]
		if item.spec(problem) is None:
			parser.error(f"{item.flag} does not apply to {PROBLEMS[problem]}")
		if key == "problem":
			continue
		try:
			values[key] = (
				text if isinstance(text, bool) else parse_value(item, problem, text)
			)
		except ValueError as error:
			parser.error(f"{item.flag}: {error}")
	errors = validate(values)
	if errors:
		parser.error("; ".join(errors))
	return values


# --- translation to the implementing module's command -----------------------


def _flag(arguments: list[str], flag: str, value: object) -> None:
	if value is not None:
		arguments += [flag, format_value(value, compact=True)]


def to_legacy(values: Mapping[str, object]) -> tuple[str, list[str]]:
	"""The existing ``tpp.py`` subcommand and arguments that execute these values."""
	problem, arguments = values["problem"], []
	campaign = str(values["campaign"])
	if problem == FIXED:
		arguments = [campaign]
		if values["solver"] != "default":
			arguments += ["--solver", values["solver"]]
		_flag(arguments, "--max-calls", values["oracle_calls"])
		_flag(
			arguments,
			"--max-seconds",
			None if values["time_limit"] == UNLIMITED else values["time_limit"],
		)
		_flag(arguments, "--repeat-count", values["repetitions"])
		_flag(arguments, "--threads", values["threads"])
		for key, flag in (
			("max_instances", "--max-instances"),
			("max_polygons", "--max-polygons"),
			("max_branching", "--max-branching"),
		):
			if values[key] != UNLIMITED:
				_flag(arguments, flag, values[key])
		_flag(arguments, "--pattern", values["pattern"])
		_flag(arguments, "--timeout", values["file_timeout"])
		if not values["rebuild"]:
			arguments.append("--no-build")
		if not values["resume"]:
			arguments.append("--force")
		if values["dry_run"]:
			arguments.append("--dry-run")
		return "run", arguments
	if problem == FREE:
		arguments = [campaign]
		for solver in (
			("tpp-ours", "tpp-fekete")
			if values["solver"] == "both"
			else (values["solver"],)
		):
			arguments += ["--solver", solver]
		_flag(arguments, "--max-calls", values["oracle_calls"])
		_flag(arguments, "--max-seconds", values["time_limit"])
		_flag(arguments, "--threads-per-instance", values["threads"])
		_flag(arguments, "--workers", values["workers"])
		_flag(arguments, "--progress-interval", values["progress_interval"])
		if values["max_instances"] != UNLIMITED:
			_flag(arguments, "--max-instances", values["max_instances"])
		else:
			arguments += ["--max-instances", "1000000000"]
		_flag(arguments, "--relative-gap", values["relative_gap"])
		if not values["rebuild"]:
			arguments.append("--no-build")
		if not values["resume"]:
			arguments.append("--force")
		if values["dry_run"]:
			arguments.append("--dry-run")
		return "free-order", arguments
	watchdog = values["external_timeout"]
	if watchdog is None:
		watchdog = values["time_limit"] + TSPN_WATCHDOG_MARGIN
	arguments = ["--campaign", campaign, "--solver", values["solver"]]
	_flag(arguments, "--seconds", values["time_limit"])
	_flag(arguments, "--external-timeout", watchdog)
	_flag(arguments, "--repetitions", values["repetitions"])
	_flag(arguments, "--relative-gap", values["relative_gap"])
	_flag(arguments, "--cycle-optimizations", values["cycle_optimizations"])
	if values["portfolio"] == "cooperative":
		arguments.append("--portfolio")
	elif values["portfolio"] == "independent":
		arguments.append("--portfolio-no-sharing")
	if values["search_strategy"] != "default":
		arguments += ["--search-strategy", values["search_strategy"]]
	if values["capture_oracles"]:
		arguments.append("--capture-oracles")
	if values["oracle_calls"] != DEFAULT_CALLS:
		_flag(arguments, "--max-calls", values["oracle_calls"])
	_flag(arguments, "--workers", values["workers"])
	_flag(arguments, "--progress-interval", values["progress_interval"])
	if not values["resume"]:
		arguments.append("--force")
	if values["dry_run"]:
		arguments.append("--dry-run")
	return "tspn-compare", arguments


def available_campaigns() -> list[str]:
	"""Names of the local campaigns that have generated instances."""
	directory = workspace.campaigns_dir()
	if not directory.is_dir():
		return []
	return sorted(
		path.name
		for path in directory.iterdir()
		if (path / workspace.CAMPAIGN_FILE).exists()
	)


def suggestions(item: Field, problem: str) -> Callable[[], list[str]] | None:
	"""Dynamic choices for a field, or None when the field has no fixed list."""
	if kind_of(item, problem) == CAMPAIGN:
		return available_campaigns
	return None


@dataclass
class Session:
	"""Mutable values for the TUI plus the checks it shows.

	``sources`` keeps the text a numeric value was typed as (``10 ** -7``), so
	editing it again shows the expression rather than its computed value.
	"""

	values: dict[str, object] = field(default_factory=lambda: default_values(FIXED))
	sources: dict[str, str] = field(default_factory=dict)

	def set(self, key: str, value: object, source: str | None = None) -> None:
		if key == "problem":
			self.values = switch_problem(self.values, str(value))
			return
		self.values[key] = value
		text = (source or "").strip()
		if text and text != format_value(value, compact=True):
			self.sources[key] = text
		else:
			self.sources.pop(key, None)

	def source_for(self, key: str) -> str | None:
		"""The expression ``key`` was typed as, if it still produces its current value."""
		text = self.sources.get(key)
		if text is None:
			return None
		try:
			return text if evaluate(text) == self.values.get(key) else None
		except ValueError:
			return None

	def snapshot(self) -> dict[str, object]:
		"""Values plus their expressions, as stored in presets and the last-used state."""
		saved = dict(self.values)
		expressions = {
			key: text for key in self.sources if (text := self.source_for(key))
		}
		if expressions:
			saved["_expressions"] = expressions
		return saved

	def restore(self, saved: Mapping[str, object]) -> None:
		saved = dict(saved)
		self.sources = dict(saved.pop("_expressions", None) or {})
		self.values = saved

	def errors(self) -> list[str]:
		return validate(self.values)

	def command(self) -> str:
		return command_line(self.values)
