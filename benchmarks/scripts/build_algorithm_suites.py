#!/usr/bin/env python3
"""Build fixed development and canonical suites from a benchmarked campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = REPO_ROOT / "benchmarks/campaigns/sao-paulo"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/suites"


@dataclass(frozen=True)
class EncodedCase:
	data: bytes
	digest: str
	polygons: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class CandidateCase:
	source_bin: Path
	case_index: int
	encoded: EncodedCase
	difficulty: str
	metrics: dict[str, str]
	generation: dict


def read_u64(data: bytes, offset: int, path: Path) -> tuple[int, int]:
	if offset + 8 > len(data):
		raise ValueError(f"Truncated size value at byte {offset} in {path}")
	return struct.unpack_from("<Q", data, offset)[0], offset + 8


def skip_bytes(data: bytes, offset: int, count: int, path: Path) -> int:
	end = offset + count
	if end > len(data):
		raise ValueError(f"Truncated case payload at byte {offset} in {path}")
	return end


def read_encoded_cases(path: Path) -> list[EncodedCase]:
	data = path.read_bytes()
	offset = 0
	cases: list[EncodedCase] = []

	while offset < len(data):
		start = offset
		offset = skip_bytes(data, offset, 32, path)  # start and target vectors
		polygon_count, offset = read_u64(data, offset, path)
		polygons: list[tuple[tuple[float, float], ...]] = []
		for _ in range(polygon_count):
			vertex_count, offset = read_u64(data, offset, path)
			vertices = tuple(
				struct.unpack_from("<dd", data, offset + 16 * vertex_index)
				for vertex_index in range(vertex_count)
			)
			polygons.append(vertices)
			offset = skip_bytes(data, offset, 16 * vertex_count, path)
		solution_count, offset = read_u64(data, offset, path)
		offset = skip_bytes(data, offset, 16 * solution_count, path)
		encoded = data[start:offset]
		cases.append(EncodedCase(encoded, hashlib.sha256(encoded).hexdigest(), tuple(polygons)))

	return cases


def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
	return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
	epsilon = 1e-12
	return (
		abs(orientation(a, b, point)) <= epsilon
		and min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
		and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
	)


def segments_intersect(
	a: tuple[float, float],
	b: tuple[float, float],
	c: tuple[float, float],
	d: tuple[float, float],
) -> bool:
	o1 = orientation(a, b, c)
	o2 = orientation(a, b, d)
	o3 = orientation(c, d, a)
	o4 = orientation(c, d, b)
	if o1 * o2 < 0.0 and o3 * o4 < 0.0:
		return True
	return (
		(abs(o1) <= 1e-12 and point_on_segment(c, a, b))
		or (abs(o2) <= 1e-12 and point_on_segment(d, a, b))
		or (abs(o3) <= 1e-12 and point_on_segment(a, c, d))
		or (abs(o4) <= 1e-12 and point_on_segment(b, c, d))
	)


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
	inside = False
	px, py = point
	for index, a in enumerate(polygon):
		b = polygon[(index + 1) % len(polygon)]
		if (a[1] > py) != (b[1] > py):
			x_crossing = (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]
			if px < x_crossing:
				inside = not inside
	return inside


def polygons_intersect(
	first: Sequence[tuple[float, float]],
	second: Sequence[tuple[float, float]],
) -> bool:
	first_bounds = (
		min(point[0] for point in first), min(point[1] for point in first),
		max(point[0] for point in first), max(point[1] for point in first),
	)
	second_bounds = (
		min(point[0] for point in second), min(point[1] for point in second),
		max(point[0] for point in second), max(point[1] for point in second),
	)
	if (
		first_bounds[2] < second_bounds[0]
		or second_bounds[2] < first_bounds[0]
		or first_bounds[3] < second_bounds[1]
		or second_bounds[3] < first_bounds[1]
	):
		return False

	for first_index, a in enumerate(first):
		b = first[(first_index + 1) % len(first)]
		for second_index, c in enumerate(second):
			d = second[(second_index + 1) % len(second)]
			if segments_intersect(a, b, c, d):
				return True
	return point_in_polygon(first[0], second) or point_in_polygon(second[0], first)


def case_has_intersections(encoded: EncodedCase) -> bool:
	return any(
		polygons_intersect(encoded.polygons[first], encoded.polygons[second])
		for first in range(len(encoded.polygons))
		for second in range(first + 1, len(encoded.polygons))
	)


def read_benchmark_rows(path: Path) -> dict[int, dict[str, str]]:
	with path.open(newline="") as file:
		rows = csv.DictReader(file, delimiter=";")
		selected = {
			int(row["case_index"]): row
			for row in rows
			if int(row["repeat_index"]) == 0
		}
	return selected


def classify(row: dict[str, str], easy_max_calls: int, medium_max_calls: int) -> str:
	calls = int(row["calls"])
	exhausted = row["exhausted"].lower() == "true"
	branch_limited = row["branch_limited"].lower() == "true"
	if not exhausted or branch_limited or calls > medium_max_calls:
		return "hard"
	if calls <= easy_max_calls:
		return "easy"
	return "medium"


def campaign_candidates(args: argparse.Namespace) -> tuple[dict, list[CandidateCase], int]:
	campaign_file = args.campaign / "campaign.json"
	if not campaign_file.exists():
		raise SystemExit(f"Missing campaign metadata: {campaign_file}")
	campaign = json.loads(campaign_file.read_text())
	candidates: list[CandidateCase] = []
	seen: set[str] = set()
	rejected_intersections = 0

	for input_record in campaign.get("inputs", []):
		input_path = args.campaign / input_record["file"]
		csv_path = args.campaign / "results" / f"{input_path.stem}.csv"
		if not input_path.exists() or not csv_path.exists():
			continue

		encoded_cases = read_encoded_cases(input_path)
		rows = read_benchmark_rows(csv_path)
		for case_index, row in rows.items():
			if case_index >= len(encoded_cases):
				raise ValueError(f"CSV case {case_index} does not exist in {input_path}")
			encoded = encoded_cases[case_index]
			if case_has_intersections(encoded):
				rejected_intersections += 1
				continue
			if encoded.digest in seen:
				continue
			seen.add(encoded.digest)
			candidates.append(CandidateCase(
				source_bin=input_path,
				case_index=case_index,
				encoded=encoded,
				difficulty=classify(row, args.easy_max_calls, args.medium_max_calls),
				metrics=row,
				generation=input_record,
			))

	return campaign, candidates, rejected_intersections


def spread_pick(candidates: Sequence[CandidateCase], count: int, rng: random.Random) -> list[CandidateCase]:
	by_source: dict[Path, list[CandidateCase]] = defaultdict(list)
	for candidate in candidates:
		by_source[candidate.source_bin].append(candidate)
	for values in by_source.values():
		rng.shuffle(values)

	sources = list(by_source)
	rng.shuffle(sources)
	selected: list[CandidateCase] = []
	while len(selected) < count and sources:
		next_sources: list[Path] = []
		for source in sources:
			values = by_source[source]
			if values and len(selected) < count:
				selected.append(values.pop())
			if values:
				next_sources.append(source)
		sources = next_sources
	return selected


def pick_hard(candidates: Sequence[CandidateCase], count: int, capped_fraction: float, rng: random.Random) -> list[CandidateCase]:
	capped = [candidate for candidate in candidates if candidate.metrics["exhausted"].lower() != "true"]
	solved = [candidate for candidate in candidates if candidate.metrics["exhausted"].lower() == "true"]
	capped_count = min(len(capped), round(count * capped_fraction))
	solved_count = min(len(solved), count - capped_count)
	selected = spread_pick(capped, capped_count, rng) + spread_pick(solved, solved_count, rng)
	if len(selected) < count:
		selected_hashes = {candidate.encoded.digest for candidate in selected}
		remaining = [candidate for candidate in candidates if candidate.encoded.digest not in selected_hashes]
		selected.extend(spread_pick(remaining, count - len(selected), rng))
	return selected


def select_suite(
	available: dict[str, list[CandidateCase]],
	per_difficulty: int,
	hard_capped_fraction: float,
	rng: random.Random,
) -> list[CandidateCase]:
	selected: list[CandidateCase] = []
	for difficulty in ("easy", "medium", "hard"):
		pool = available[difficulty]
		picked = (
			pick_hard(pool, per_difficulty, hard_capped_fraction, rng)
			if difficulty == "hard"
			else spread_pick(pool, per_difficulty, rng)
		)
		if len(picked) < per_difficulty:
			raise SystemExit(
				f"Need {per_difficulty} {difficulty} cases, but only {len(picked)} are available. "
				"Adjust suite sizes or difficulty call thresholds."
			)
		selected.extend(picked)
		picked_hashes = {candidate.encoded.digest for candidate in picked}
		available[difficulty] = [candidate for candidate in pool if candidate.encoded.digest not in picked_hashes]
	return selected


def interleave_difficulties(selected: Sequence[CandidateCase]) -> list[CandidateCase]:
	groups = {
		difficulty: [candidate for candidate in selected if candidate.difficulty == difficulty]
		for difficulty in ("easy", "medium", "hard")
	}
	interleaved: list[CandidateCase] = []
	for index in range(max(len(group) for group in groups.values())):
		for difficulty in ("easy", "medium", "hard"):
			if index < len(groups[difficulty]):
				interleaved.append(groups[difficulty][index])
	return interleaved


def relative_source(candidate: CandidateCase, campaign_dir: Path) -> str:
	try:
		return str(candidate.source_bin.relative_to(campaign_dir))
	except ValueError:
		return str(candidate.source_bin)


def write_suite(name: str, selected: Sequence[CandidateCase], output_dir: Path, campaign_dir: Path) -> None:
	bin_path = output_dir / f"{name}.bin"
	csv_path = output_dir / f"{name}.csv"
	md_path = output_dir / f"{name}.md"
	output_dir.mkdir(parents=True, exist_ok=True)

	with bin_path.open("wb") as file:
		for candidate in selected:
			file.write(candidate.encoded.data)

	fieldnames = [
		"suite_index", "difficulty", "sha256", "source_bin", "source_case_index",
		"polygons", "layout", "grid_cell_size", "convex_replacement_fraction", "seed",
		"calls", "bnb_seconds", "decomposed_pieces", "total_combinations", "exhausted",
		"branch_limited", "checksum",
	]
	with csv_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		for suite_index, candidate in enumerate(selected):
			writer.writerow({
				"suite_index": suite_index,
				"difficulty": candidate.difficulty,
				"sha256": candidate.encoded.digest,
				"source_bin": relative_source(candidate, campaign_dir),
				"source_case_index": candidate.case_index,
				"polygons": candidate.metrics["polygons"],
				"layout": candidate.generation.get("layout"),
				"grid_cell_size": candidate.generation.get("grid_cell_size"),
				"convex_replacement_fraction": candidate.generation.get("convex_replacement_fraction"),
				"seed": candidate.generation.get("seed"),
				"calls": candidate.metrics["calls"],
				"bnb_seconds": candidate.metrics["bnb_seconds"],
				"decomposed_pieces": candidate.metrics["decomposed_pieces"],
				"total_combinations": candidate.metrics["total_combinations"],
				"exhausted": candidate.metrics["exhausted"],
				"branch_limited": candidate.metrics["branch_limited"],
				"checksum": candidate.metrics["checksum"],
			})

	lines = [f"# {name}", "", f"Cases: {len(selected)}", "", "| Difficulty | Count | Calls min | Calls median | Calls mean | Calls max |", "|---|---:|---:|---:|---:|---:|"]
	for difficulty in ("easy", "medium", "hard"):
		calls = [int(candidate.metrics["calls"]) for candidate in selected if candidate.difficulty == difficulty]
		lines.append(
			f"| {difficulty} | {len(calls)} | {min(calls)} | {statistics.median(calls):.0f} | "
			f"{statistics.mean(calls):.1f} | {max(calls)} |"
		)
	lines.extend(["", f"Distinct source binaries: {len({candidate.source_bin for candidate in selected})}", ""])
	md_path.write_text("\n".join(lines))
	print(f"Wrote {len(selected)} cases to {bin_path}", flush=True)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Build fixed algorithm development and canonical benchmark suites.")
	parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--seed", type=int, default=20260811)
	parser.add_argument("--dev-per-difficulty", type=int, default=20)
	parser.add_argument("--final-per-difficulty", type=int, default=100)
	parser.add_argument("--easy-max-calls", type=int, default=500)
	parser.add_argument("--medium-max-calls", type=int, default=20000)
	parser.add_argument("--hard-capped-fraction", type=float, default=0.2)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	args.campaign = args.campaign.resolve()
	args.output = args.output.resolve()
	if args.easy_max_calls >= args.medium_max_calls:
		raise SystemExit("--easy-max-calls must be lower than --medium-max-calls")
	if not 0.0 <= args.hard_capped_fraction <= 1.0:
		raise SystemExit("--hard-capped-fraction must be between 0 and 1")

	campaign, candidates, rejected_intersections = campaign_candidates(args)
	available = {
		difficulty: [candidate for candidate in candidates if candidate.difficulty == difficulty]
		for difficulty in ("easy", "medium", "hard")
	}
	print("Available unique baseline cases:", ", ".join(f"{key}={len(value)}" for key, value in available.items()), flush=True)
	print(f"Rejected {rejected_intersections} intersecting or touching cases.", flush=True)
	rng = random.Random(args.seed)
	final_suite = interleave_difficulties(select_suite(available, args.final_per_difficulty, args.hard_capped_fraction, rng))
	dev_suite = interleave_difficulties(select_suite(available, args.dev_per_difficulty, args.hard_capped_fraction, rng))
	write_suite("canonical-v1", final_suite, args.output, args.campaign)
	write_suite("algorithm-dev-v1", dev_suite, args.output, args.campaign)

	metadata = {
		"schema_version": 1,
		"source_campaign": str(args.campaign),
		"source_campaign_git_revision": campaign.get("git_revision"),
		"source_campaign_git_dirty": campaign.get("git_dirty"),
		"baseline_benchmark": campaign.get("benchmark_runs", [])[-1] if campaign.get("benchmark_runs") else None,
		"selection_seed": args.seed,
		"rejected_intersecting_or_touching": rejected_intersections,
		"difficulty": {
			"easy": f"exhausted and calls <= {args.easy_max_calls}",
			"medium": f"exhausted and {args.easy_max_calls} < calls <= {args.medium_max_calls}",
			"hard": f"not exhausted, branch limited, or calls > {args.medium_max_calls}",
		},
		"canonical_counts": dict(Counter(candidate.difficulty for candidate in final_suite)),
		"development_counts": dict(Counter(candidate.difficulty for candidate in dev_suite)),
		"canonical_baseline_capped": sum(candidate.metrics["exhausted"].lower() != "true" for candidate in final_suite),
		"development_baseline_capped": sum(candidate.metrics["exhausted"].lower() != "true" for candidate in dev_suite),
		"hard_capped_fraction_target": args.hard_capped_fraction,
		"canonical_sha256": hashlib.sha256((args.output / "canonical-v1.bin").read_bytes()).hexdigest(),
		"development_sha256": hashlib.sha256((args.output / "algorithm-dev-v1.bin").read_bytes()).hexdigest(),
	}
	(args.output / "suite-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
