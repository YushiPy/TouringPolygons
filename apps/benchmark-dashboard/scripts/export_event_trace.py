"""Export compact educational traces for the archived SIICUSP event page."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/scripts"))

from benchmark_cases import read_encoded_cases  # noqa: E402
from unordered_runner import run_unordered_solver  # noqa: E402

DEFAULT_SUITE = ROOT / "benchmarks/results/unordered/siicusp34-20260906-200122/inputs/algorithm-dev-v1.bin"
DEFAULT_SOLVER = ROOT / ".build/unordered/tpp"


def compact_event(event: dict) -> dict:
    """Keep the fields used by the educational replay, omitting JSON nulls."""
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


def export_case(case, solver: Path, max_seconds: float, max_events: int) -> dict:
    start = struct.unpack_from("<dd", case.data, 0)
    target = struct.unpack_from("<dd", case.data, 16)
    result = run_unordered_solver(solver, start, target, case.polygons, 10_000_000, max_seconds, arguments=("--trace",))
    events = [compact_event(event) for event in result.get("trace", [])]
    if max_events > 0 and len(events) > max_events:
        # Preserve all heuristic steps and the end certificate, then keep the
        # first search events. Large cases remain useful without shipping a
        # multi-megabyte animation to every visitor.
        heuristic = [event for event in events if event["kind"].startswith("heuristic") or event["kind"] == "incumbent"]
        search = [event for event in events if event not in heuristic and event["kind"] != "complete"]
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
    }
    return {
        "case": case.case_index,
        "sha256": case.digest,
        "optimal_order": result.get("order", []),
        "summary": result_summary,
        "event_count": len(result.get("trace", [])),
        "omitted_events": len(result.get("trace", [])) - len(events),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", help="Zero-based case index; defaults to showcase cases.")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--max-events", type=int, default=1200)
    args = parser.parse_args()
    cases = read_encoded_cases(args.suite)
    indices = args.case if args.case is not None else [2, 9, 55]
    traces = {}
    for index in indices:
        if index < 0 or index >= len(cases):
            raise SystemExit(f"Case index out of range: {index}")
        print(f"Tracing case {index + 1} ({len(cases[index].polygons)} polygons)...", flush=True)
        traces[str(index)] = export_case(cases[index], args.solver, args.seconds, args.max_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump({"schema_version": 1, "cases": traces}, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")
    print(f"Exported {len(traces)} educational traces to {args.output}")


if __name__ == "__main__":
    main()
