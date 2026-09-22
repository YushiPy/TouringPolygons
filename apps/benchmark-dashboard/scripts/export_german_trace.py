#!/usr/bin/env python3
"""Export compact educational traces for the current 558-case German page."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))

from unordered_runner import run_unordered_solver  # noqa: E402

DEFAULT_DATA = ROOT / "apps/benchmark-dashboard/static/event/german-instances-exact-20260918.json"
DEFAULT_SOLVER = ROOT / ".build/unordered/tpp"


@dataclass(frozen=True)
class GermanCase:
    case_index: int
    digest: str
    start: list[float]
    target: list[float]
    polygons: list[list[list[float]]]


def compact_event(event: dict) -> dict:
    """Keep only fields consumed by the educational replay."""
    fields = (
        "kind",
        "node",
        "parent",
        "polygon",
        "piece",
        "position",
        "pass",
        "sequence",
        "order",
        "path",
        "lower_bound",
        "upper_bound",
        "length",
        "pruned",
        "source",
        "reason",
    )
    return {key: event[key] for key in fields if event.get(key) not in (None, [], "")}


def export_case(case: GermanCase, solver: Path, max_seconds: float, max_events: int) -> dict:
    result = run_unordered_solver(
        solver, case.start, case.target, case.polygons, 10_000_000, max_seconds, arguments=("--trace",)
    )
    events = [compact_event(event) for event in result.get("trace", [])]
    original_event_count = len(events)
    if max_events > 0 and len(events) > max_events:
        # Always retain the complete heuristic and the final certificate. The
        # remaining budget is spent on search events so the tree stays useful.
        heuristic = [event for event in events if event["kind"].startswith("heuristic") or event["kind"] == "incumbent"]
        search = [
            event
            for event in events
            if not event["kind"].startswith("heuristic")
            and event["kind"] != "incumbent"
            and event["kind"] != "complete"
        ]
        kept = heuristic + search[: max(0, max_events - len(heuristic) - 1)]
        kept.append(next((event for event in reversed(events) if event["kind"] == "complete"), events[-1]))
        events = kept
    result_summary = {
        key: result[key]
        for key in (
            "exact",
            "termination",
            "lower_bound",
            "upper_bound",
            "calls",
            "nodes",
            "screened_nodes",
            "insertion_branches",
            "decomposition_branches",
        )
        if key in result
    }
    return {
        "case": case.case_index,
        "sha256": case.digest,
        "optimal_order": result.get("order", []),
        "summary": result_summary,
        "event_count": original_event_count,
        "omitted_events": original_event_count - len(events),
        "events": events,
    }


def export_cases(
    cases: dict[int, GermanCase],
    indices: list[int],
    solver: Path,
    max_seconds: float,
    max_events: int,
    under_events: int | None = None,
) -> dict[str, dict]:
    """Export requested traces, optionally retaining every short tree in the corpus."""
    traces = {}
    selected = set(indices)
    for index in indices if under_events is None else cases:
        if index not in cases:
            raise SystemExit(f"Case index out of range: {index}")
        case = cases[index]
        print(f"Tracing case {index + 1} ({len(case.polygons)} polygons)...", flush=True)
        limit = max_events if under_events is None or index in selected else 0
        trace = export_case(case, solver, max_seconds, limit)
        if under_events is None or trace["event_count"] < under_events or index in selected:
            traces[str(index)] = trace
    return traces


def load_cases(data_path: Path) -> dict[int, GermanCase]:
    data = json.loads(data_path.read_text())
    return {
        row["case"]: GermanCase(
            case_index=row["case"],
            digest=row["sha256"],
            start=row["geometry"]["start"],
            target=row["geometry"]["target"],
            polygons=row["geometry"]["polygons"],
        )
        for row in data["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", help="Zero-based case index; defaults to showcase cases.")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--max-events", type=int, default=1200)
    parser.add_argument(
        "--under-events",
        type=int,
        help="Also export every case whose complete trace has fewer events than this threshold.",
    )
    args = parser.parse_args()

    cases = load_cases(args.data)
    indices = args.case if args.case is not None else [1, 3, 9]
    traces = export_cases(cases, indices, args.solver, args.seconds, args.max_events, args.under_events)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump({"schema_version": 1, "cases": traces}, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")
    print(f"Exported {len(traces)} educational traces to {args.output}")


if __name__ == "__main__":
    main()
