"""Export the audited SIICUSP run as a portable, deterministic event dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/scripts"))

from benchmark_cases import read_encoded_cases  # noqa: E402
from summarize_unordered_siicusp import audit  # noqa: E402


def export_dataset(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    suite = run / "inputs/algorithm-dev-v1.bin"
    source = run / "canonical-confirmation/ours.jsonl"
    if hashlib.sha256(suite.read_bytes()).hexdigest() != manifest["suite_sha256"]:
        raise ValueError("Suite hash differs from the run manifest")
    cases = {case.case_index: case for case in read_encoded_cases(suite)}
    summary, _ = audit(source, cases, set(cases))
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        case = cases[row["case"]]
        coordinates = struct.unpack_from("<dddd", case.data)
        row["geometry"] = {"start": coordinates[:2], "target": coordinates[2:], "polygons": case.polygons}
        row["solver"] = "unordered"
        row["endpoint_valid"] = row["validation"]["endpoint_valid"]
    summary.pop("source")
    summary["exact_certified"] = summary["numerically_certified"]
    independent = json.loads((run / "summary/artifact-audit/summary.json").read_text())
    return {
        "schema_version": 1,
        "title": "TPP com ordem livre · SIICUSP 2026",
        "visit_order": "free",
        "status": "completed",
        "provenance": {
            "date": "2026-09-06",
            "run_id": manifest["run_id"],
            "revision": manifest["git_revision"],
            "suite": "algorithm-dev-v1",
            "suite_sha256": manifest["suite_sha256"],
            "results_sha256": summary["sha256"],
            "binary_sha256": manifest["binary_sha256"],
            "machine": "MacBook Pro · Apple M4 Pro · 24 GB · macOS 26.6.2",
        },
        "config": {
            key: manifest[key]
            for key in (
                "threads",
                "max_seconds",
                "max_calls",
                "absolute_gap",
                "relative_gap",
                "solver_visit_tolerance",
                "independent_geometry_tolerance",
            )
        },
        "summary": summary,
        "independent_validation": {
            key: independent[key]
            for key in (
                "socp_cases",
                "max_socp_objective_difference",
                "socp_nonfinite_lower_bound_cases",
                "independent_interruptions",
                "interruption_termination_counts",
            )
        },
        "notes": [
            "Recorded on 6 September 2026: development suite, fixed endpoints, free visit order.",
            "Completed cases are certified by the exact solver, which uses exact geometric predicates and rational fallback.",
            "All stored paths independently validated at 1e-7. Time-limited cases have a feasible path but no completed exact certification.",
            "This is the adopted confirmation run. The earlier interrupted campaign is excluded.",
            "The external comparison has different tolerances and is not evidence of universal speedup.",
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = export_dataset(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(data, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        file.write("\n")
    print(f"Exported {len(data['rows'])} verified cases to {args.output}")


if __name__ == "__main__":
    main()
