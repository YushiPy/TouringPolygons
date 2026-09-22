#!/usr/bin/env python3
"""Build fixed development and canonical suites from a benchmarked campaign."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from benchmark_cases import (
	EncodedCase,
	case_has_intersecting_hulls,
	case_has_intersections,
	read_encoded_cases,
	write_encoded_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = REPO_ROOT / "benchmarks/campaigns/sao-paulo"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/suites"


@dataclass(frozen=True)
class CandidateCase:
	source_bin: Path
	case_index: int
	encoded: EncodedCase
	difficulty: str
	metrics: dict[str, str]
	generation: dict


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
	rejected_hull_intersections = 0

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
			if case_has_intersecting_hulls(encoded):
				rejected_hull_intersections += 1
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

	return campaign, candidates, rejected_intersections + rejected_hull_intersections


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


def write_suite(name: str, selected: Sequence[CandidateCase], output_dir: Path) -> None:
	bin_path = output_dir / f"{name}.bin"
	write_encoded_cases(bin_path, [candidate.encoded for candidate in selected])
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

	_, candidates, rejected_intersections = campaign_candidates(args)
	available = {
		difficulty: [candidate for candidate in candidates if candidate.difficulty == difficulty]
		for difficulty in ("easy", "medium", "hard")
	}
	print("Available unique baseline cases:", ", ".join(f"{key}={len(value)}" for key, value in available.items()), flush=True)
	print(f"Rejected {rejected_intersections} intersecting or touching cases.", flush=True)
	rng = random.Random(args.seed)
	final_suite = interleave_difficulties(select_suite(available, args.final_per_difficulty, args.hard_capped_fraction, rng))
	dev_suite = interleave_difficulties(select_suite(available, args.dev_per_difficulty, args.hard_capped_fraction, rng))
	write_suite("canonical-v1", final_suite, args.output)
	write_suite("algorithm-dev-v1", dev_suite, args.output)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
