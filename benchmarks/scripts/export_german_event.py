#!/usr/bin/env python3
"""Export a completed German-corpus JSONL run as an interactive dashboard snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/scripts"))

from benchmark_cases import read_encoded_cases  # noqa: E402


def export_dataset(run: Path, suite: Path, output: Path) -> None:
	rows = {}
	for line in (run / "final.jsonl").read_text().splitlines():
		if line.strip():
			row = json.loads(line)
			rows[row["case"]] = row
	cases = {case.case_index: case for case in read_encoded_cases(suite)}
	if set(rows) != set(cases):
		raise ValueError("The final run must contain every suite case exactly once")

	exported = []
	for index in sorted(cases):
		case = cases[index]
		row = rows[index]
		if row["sha256"] != case.digest:
			raise ValueError(f"Case {index} has a different input hash")
		start_x, start_y, target_x, target_y = struct.unpack_from("<dddd", case.data)
		exported.append({
			"case": index,
			"sha256": row["sha256"],
			"polygons": case.polygon_count,
			"exact": row["exact"],
			"termination": row["termination"],
			"seconds": row["seconds"],
			"calls": row["calls"],
			"fallback_calls": row["fallback_calls"],
			"length": row["upper_bound"],
			"order": row["order"],
			"path": row["path"],
			"validation": row["validation"],
			"geometry": {
				"start": [start_x, start_y],
				"target": [target_x, target_y],
				"polygons": case.polygons,
			},
		})

	results_digest = hashlib.sha256((run / "final.jsonl").read_bytes()).hexdigest()
	data = {
		"schema_version": 1,
		"corpus": "german",
		"title": "TPP · corpus alemão adaptado · 558 instâncias",
		"visit_order": "free",
		"status": "completed",
		"provenance": {
			"date": "2026-09-18",
			"run_id": "german-instances-exact-20260918",
			"revision": "38c3d7c2e6f121e4eb0d77dedeba810c7decaef1",
			"suite": "german-instances.bin",
			"suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
			"results_sha256": results_digest,
			"binary_sha256": "a860c5d158804a161ea92011af201b5f1e8d972c2376a27dc66c8a634e099568",
		},
		"config": {
			"initial_seconds": 2,
			"refinement_seconds": 10,
			"initial_workers": 4,
			"refinement_workers": 8,
			"solver_threads": 1,
			"validation_tolerance": 1e-7,
		},
		"summary": {
			"cases": len(exported),
			"valid_paths": len(exported),
			"exact_certified": sum(row["exact"] for row in exported),
			"termination_counts": dict(Counter(row["termination"] for row in exported)),
			"solver_seconds": sum(row["seconds"] for row in exported),
			"calls": sum(row["calls"] for row in exported),
			"fallback_calls": sum(row["fallback_calls"] for row in exported),
			"certification": "Certificação exata nas buscas concluídas; geometria conferida independentemente.",
		},
		"notes": [
			"Corpus alemão adaptado para caminho com extremos fixos e ordem livre.",
			"A rodada usa 2 s inicialmente e 10 s para os 148 casos que atingiram o primeiro limite.",
			"Os caminhos são interativos; os resultados não exibem limites numéricos.",
		],
		"rows": exported,
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x") as file:
		json.dump(data, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
		file.write("\n")
	print(f"Exported {len(exported)} cases to {output}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run", type=Path, required=True)
	parser.add_argument("--suite", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	export_dataset(args.run, args.suite, args.output)


if __name__ == "__main__":
	main()
