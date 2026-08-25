#!/usr/bin/env python3
"""Generate deterministic benchmark suites from the tracked non-convex corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from benchmark_cases import (
	EncodedCase,
	case_has_intersecting_hulls,
	read_encoded_cases,
	write_encoded_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "benchmarks/suites/nonconvex/test_cases.bin"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/suites"


def spread_select(cases: Sequence[EncodedCase], count: int) -> list[EncodedCase]:
	if count > len(cases):
		raise SystemExit(f"Requested {count} cases, but only {len(cases)} are available.")

	ordered = sorted(cases, key=lambda case: (case.polygon_count, case.vertex_count, case.digest))
	if count == len(ordered):
		return ordered

	last = len(ordered) - 1
	indices = sorted({round(index * last / (count - 1)) for index in range(count)})
	selected = [ordered[index] for index in indices]

	cursor = 0
	while len(selected) < count:
		candidate = ordered[cursor]
		if candidate not in selected:
			selected.append(candidate)
		cursor += 1

	return selected


def prefix_representative_order(cases: Sequence[EncodedCase]) -> list[EncodedCase]:
	ordered = sorted(cases, key=lambda case: (case.polygon_count, case.vertex_count, case.digest))
	bucket_count = 3
	bucket_size = (len(ordered) + bucket_count - 1) // bucket_count
	buckets = [
		ordered[index * bucket_size:(index + 1) * bucket_size]
		for index in range(bucket_count)
	]
	result: list[EncodedCase] = []

	while any(buckets):
		for bucket in reversed(buckets):
			if bucket:
				result.append(bucket.pop(0))

	return result


def write_suite(name: str, cases: Sequence[EncodedCase], output_dir: Path) -> None:
	bin_path = output_dir / f"{name}.bin"
	write_encoded_cases(bin_path, cases)
	print(f"Wrote {len(cases)} cases to {bin_path}", flush=True)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate algorithm benchmark suites from the tracked non-convex corpus.")
	parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--canonical-size", type=int, default=300)
	parser.add_argument("--dev-size", type=int, default=60)
	parser.add_argument("--require-disjoint-hulls", action="store_true")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	source = args.source.resolve()
	output = args.output.resolve()
	output.mkdir(parents=True, exist_ok=True)

	if args.dev_size > args.canonical_size:
		raise SystemExit("--dev-size must be less than or equal to --canonical-size")

	source_cases = read_encoded_cases(source)
	if args.require_disjoint_hulls:
		filtered_cases = [case for case in source_cases if not case_has_intersecting_hulls(case)]
		rejected_hull_intersections = len(source_cases) - len(filtered_cases)
		print(
			f"Available hull-disjoint cases: {len(filtered_cases)} of {len(source_cases)} "
			f"({rejected_hull_intersections} rejected).",
			flush=True,
		)
	else:
		filtered_cases = source_cases
		rejected_hull_intersections = 0

	canonical = prefix_representative_order(spread_select(filtered_cases, args.canonical_size))
	dev = prefix_representative_order(spread_select(canonical, args.dev_size))

	write_suite("canonical-v1", canonical, output)
	write_suite("algorithm-dev-v1", dev, output)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
