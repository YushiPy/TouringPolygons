#!/usr/bin/env python3
"""Generate deterministic benchmark suites from the tracked non-convex corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "benchmarks/suites/nonconvex/test_cases.bin"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/suites"


@dataclass(frozen=True)
class EncodedCase:
	data: bytes
	digest: str
	case_index: int
	polygons: int
	vertices: int


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
		offset = skip_bytes(data, offset, 32, path)
		polygon_count, offset = read_u64(data, offset, path)
		vertex_total = 0

		for _ in range(polygon_count):
			vertex_count, offset = read_u64(data, offset, path)
			vertex_total += vertex_count
			offset = skip_bytes(data, offset, 16 * vertex_count, path)

		solution_count, offset = read_u64(data, offset, path)
		offset = skip_bytes(data, offset, 16 * solution_count, path)
		encoded = data[start:offset]
		cases.append(EncodedCase(
			data=encoded,
			digest=hashlib.sha256(encoded).hexdigest(),
			case_index=len(cases),
			polygons=polygon_count,
			vertices=vertex_total,
		))

	return cases


def spread_select(cases: Sequence[EncodedCase], count: int) -> list[EncodedCase]:
	if count > len(cases):
		raise SystemExit(f"Requested {count} cases, but only {len(cases)} are available.")

	ordered = sorted(cases, key=lambda case: (case.polygons, case.vertices, case.digest))
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
	ordered = sorted(cases, key=lambda case: (case.polygons, case.vertices, case.digest))
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


def write_suite(name: str, cases: Sequence[EncodedCase], output_dir: Path, source: Path) -> None:
	bin_path = output_dir / f"{name}.bin"
	csv_path = output_dir / f"{name}.csv"
	md_path = output_dir / f"{name}.md"

	with bin_path.open("wb") as file:
		for case in cases:
			file.write(case.data)

	with csv_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=[
			"suite_index", "source_case_index", "sha256", "polygons", "vertices",
		])
		writer.writeheader()
		for suite_index, case in enumerate(cases):
			writer.writerow({
				"suite_index": suite_index,
				"source_case_index": case.case_index,
				"sha256": case.digest,
				"polygons": case.polygons,
				"vertices": case.vertices,
			})

	polygon_counts = [case.polygons for case in cases]
	vertex_counts = [case.vertices for case in cases]
	lines = [
		f"# {name}",
		"",
		f"Generated from `{source.relative_to(REPO_ROOT)}`.",
		"",
		f"Cases: {len(cases)}",
		f"Polygons: min {min(polygon_counts)}, median {statistics.median(polygon_counts):.0f}, max {max(polygon_counts)}",
		f"Vertices: min {min(vertex_counts)}, median {statistics.median(vertex_counts):.0f}, max {max(vertex_counts)}",
		"",
	]
	md_path.write_text("\n".join(lines))
	print(f"Wrote {len(cases)} cases to {bin_path}", flush=True)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate algorithm benchmark suites from the tracked non-convex corpus.")
	parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--canonical-size", type=int, default=300)
	parser.add_argument("--dev-size", type=int, default=60)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	source = args.source.resolve()
	output = args.output.resolve()
	output.mkdir(parents=True, exist_ok=True)

	if args.dev_size > args.canonical_size:
		raise SystemExit("--dev-size must be less than or equal to --canonical-size")

	source_cases = read_encoded_cases(source)
	canonical = prefix_representative_order(spread_select(source_cases, args.canonical_size))
	dev = prefix_representative_order(spread_select(canonical, args.dev_size))

	write_suite("canonical-v1", canonical, output, source)
	write_suite("algorithm-dev-v1", dev, output, source)

	metadata = {
		"schema_version": 2,
		"generator": "benchmarks/scripts/generate_algorithm_suites.py",
		"source": str(source.relative_to(REPO_ROOT)),
		"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
		"canonical_size": len(canonical),
		"development_size": len(dev),
		"selection": "even spread by polygon count, vertex count, and case digest; ordered so prefixes remain representative",
		"canonical_sha256": hashlib.sha256((output / "canonical-v1.bin").read_bytes()).hexdigest(),
		"development_sha256": hashlib.sha256((output / "algorithm-dev-v1.bin").read_bytes()).hexdigest(),
	}
	(output / "suite-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
