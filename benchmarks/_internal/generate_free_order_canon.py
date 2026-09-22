#!/usr/bin/env python3
"""Generate independent diagnostic and held-out free-order TPP suites."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SOURCE = REPO_ROOT / "packages/instance-generation/source"
DEFAULT_REGION = REPO_ROOT / "packages/instance-generation/regions/sao-paulo.osm.pbf"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/campaigns/free-order-canon-v1"

sys.path.insert(0, str(GENERATOR_SOURCE))
import gen_instances as gen


@dataclass(frozen=True)
class Region:
	name: str
	path: Path


@dataclass(frozen=True)
class Profile:
	name: str
	description: str
	layout: str = "grid"
	grid_cell_size: float = 2.0
	grid_placement: str = "random"
	order: str = "random"
	sampling: str = "local"
	convex_fraction: float = 0.0
	convex_vertices: int = 64
	convex_position: str = "random"
	endpoint_mode: str = "bbox"
	scale: float = 1.0
	reverse_endpoints: bool = False
	seed_group: str | None = None


PROFILES = (
	Profile(
		"dense-nonconvex",
		"Complex OSM footprints on a dense randomized grid.",
		grid_cell_size=1.10,
	),
	Profile(
		"low-clearance",
		"Nearly touching bounding boxes, without polygon intersections.",
		grid_cell_size=1.01,
	),
	Profile(
		"sparse-nonconvex",
		"Complex OSM footprints on a widely separated randomized grid.",
		grid_cell_size=3.0,
	),
	Profile(
		"many-vertex-convex",
		"Convex 96-vertex replacements isolate convex-oracle cost.",
		grid_cell_size=1.5,
		convex_fraction=1.0,
		convex_vertices=96,
	),
	Profile(
		"alternating-mixed",
		"Alternating nonconvex OSM and convex 64-vertex regions.",
		grid_cell_size=1.25,
		convex_fraction=0.5,
		convex_vertices=64,
		convex_position="alternating",
	),
	Profile(
		"ordered-endpoints",
		"Asymmetric endpoints derived from the sampled spatial order.",
		grid_cell_size=2.0,
		order="spatial",
		endpoint_mode="ordered",
		seed_group="endpoint-pair",
	),
	Profile(
		"reversed-endpoints",
		"The ordered-endpoint geometry with start and target exchanged.",
		grid_cell_size=2.0,
		order="spatial",
		endpoint_mode="ordered",
		reverse_endpoints=True,
		seed_group="endpoint-pair",
	),
	Profile(
		"geographic-local",
		"Actual local OSM placement, filtered to pairwise-disjoint regions.",
		layout="geographic",
		order="spatial",
		endpoint_mode="bbox",
	),
	Profile(
		"scale-small",
		"Dense geometry scaled by 1e-6 for tolerance stress.",
		grid_cell_size=1.10,
		scale=1e-6,
		seed_group="scale-pair",
	),
	Profile(
		"scale-large",
		"The scale-small geometry scaled by 1e6 instead.",
		grid_cell_size=1.10,
		scale=1e6,
		seed_group="scale-pair",
	),
)


def positive_int(text: str) -> int:
	value = int(text)
	if value < 1:
		raise argparse.ArgumentTypeError("must be positive")
	return value


def comma_ints(text: str) -> tuple[int, ...]:
	values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
	if not values or any(value < 1 for value in values):
		raise argparse.ArgumentTypeError("expected positive comma-separated integers")
	return values


def parse_region(text: str) -> Region:
	if "=" not in text:
		raise argparse.ArgumentTypeError("expected NAME=PATH")
	name, raw_path = text.split("=", 1)
	if not name.strip() or not raw_path.strip():
		raise argparse.ArgumentTypeError("expected NAME=PATH")
	return Region(name.strip(), Path(raw_path).expanduser().resolve())


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument(
		"--region",
		type=parse_region,
		action="append",
		help="OSM source as NAME=PATH; repeat for multiple regions. Defaults to São Paulo.",
	)
	parser.add_argument("--polygon-counts", type=comma_ints, default=(40, 60, 80))
	parser.add_argument("--diagnostic-per-cell", type=positive_int, default=4)
	parser.add_argument("--heldout-per-cell", type=positive_int, default=18)
	parser.add_argument("--seed", type=int, default=20260911)
	parser.add_argument("--with-preview", action="store_true")
	parser.add_argument("--overwrite", action="store_true")
	return parser


def stable_seed(base: int, *parts: object) -> int:
	payload = "\0".join((str(base), *(str(part) for part in parts))).encode()
	return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7fff_ffff


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		while chunk := file.read(1024 * 1024):
			digest.update(chunk)
	return digest.hexdigest()


def git_revision() -> str | None:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
		stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
	)
	return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
	result = subprocess.run(
		["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
		stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
	)
	return bool(result.stdout.strip()) if result.returncode == 0 else None


def generation_args(region: Region, profile: Profile, polygons: int, instances: int, seed: int) -> argparse.Namespace:
	return argparse.Namespace(
		input_pbf=region.path,
		output_bin=Path(),
		preview=Path(),
		manifest=None,
		no_preview=True,
		no_manifest=True,
		instances=instances,
		polygons_per_instance=polygons,
		seed=seed,
		simplify_tolerance=1.0,
		scale=profile.scale,
		normalization="instance",
		order=profile.order,
		sampling=profile.sampling,
		local_pool_size=(max(5_000, polygons * 20) if profile.layout == "geographic" else max(160, polygons * 2)),
		layout=profile.layout,
		grid_polygon_size=1.0,
		grid_cell_size=profile.grid_cell_size,
		grid_columns=0,
		grid_placement=profile.grid_placement,
		convex_replacement_fraction=profile.convex_fraction,
		convex_replacement_vertices=profile.convex_vertices,
		convex_replacement_scale=1.0,
		convex_replacement_position=profile.convex_position,
		endpoint_mode=profile.endpoint_mode,
		candidate_pool="nonconvex",
		nonconvex_threshold=0.98,
		min_area=10.0,
		min_vertices=4,
		max_vertices=80,
		single_preview_count=0,
		single_preview_dir=None,
		cache=None,
		no_cache=False,
	)


def case_is_supported(case: gen.TestCase) -> bool:
	polygons = [gen.Polygon(polygon) for polygon in case.polygons]
	if any(
		polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0
		for polygon in polygons
	):
		return False
	return not any(
		polygons[first].intersects(polygons[second])
		for first in range(len(polygons))
		for second in range(first + 1, len(polygons))
	)


def generate_supported_cases(
	region: Region,
	profile: Profile,
	polygons: int,
	count: int,
	seed: int,
	candidates: Sequence[gen.Candidate],
) -> tuple[list[gen.TestCase], list[int]]:
	selected: list[gen.TestCase] = []
	used_sources: set[tuple[int, ...]] = set()
	seeds: list[int] = []
	for attempt in range(24):
		attempt_seed = seed + attempt * 1_000_003
		needed = count - len(selected)
		batch_size = needed if profile.layout == "grid" else max(needed * 4, 32)
		args = generation_args(region, profile, polygons, batch_size, attempt_seed)
		for case in gen.generate_cases(args, candidates):
			key = tuple(case.source_candidate_indices)
			if key in used_sources or not case_is_supported(case):
				continue
			used_sources.add(key)
			if profile.reverse_endpoints:
				case = dataclasses.replace(case, start=case.target, target=case.start)
			selected.append(case)
			if len(selected) == count:
				break
		seeds.append(attempt_seed)
		if len(selected) == count:
			return selected, seeds
	if len(selected) < count:
		raise RuntimeError(
			f"Only generated {len(selected)}/{count} supported cases for "
			f"{region.name}/{profile.name}/p{polygons}."
		)
	return selected, seeds


def encoded_case(case: gen.TestCase) -> bytes:
	parts = [struct.pack("<ddddQ", *case.start, *case.target, len(case.polygons))]
	for polygon in case.polygons:
		parts.append(struct.pack("<Q", len(polygon)))
		parts.extend(struct.pack("<dd", *point) for point in polygon)
	parts.append(struct.pack("<Q", 0))
	return b"".join(parts)


def write_index(path: Path, rows: Sequence[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fields = (
		"case_index", "split", "region", "profile", "polygons", "vertices",
		"seed", "sha256", "source_candidate_indices",
	)
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	output = args.output.resolve()
	regions = args.region or [Region("sao-paulo", DEFAULT_REGION)]
	for region in regions:
		if not region.path.exists():
			raise SystemExit(f"Missing region: {region.path}")
	if output.exists():
		if not args.overwrite:
			raise SystemExit(f"Output already exists: {output}; pass --overwrite to replace it.")
		shutil.rmtree(output)
	output.mkdir(parents=True)

	gen.ensure_geometry_dependencies()
	diagnostic_cases: list[tuple[gen.TestCase, dict[str, object]]] = []
	heldout_cases: list[tuple[gen.TestCase, dict[str, object]]] = []
	profile_outputs: list[dict[str, object]] = []
	source_records: list[dict[str, object]] = []

	for region in regions:
		cache = region.path.with_suffix(region.path.suffix + ".buildings.pkl")
		rings = gen.load_building_rings(region.path, cache, True)
		origin = gen.projection_origin(rings)
		base_args = generation_args(region, PROFILES[0], max(args.polygon_counts), 1, args.seed)
		candidates = gen.build_candidates(base_args, rings, origin)
		if len(candidates) < max(args.polygon_counts):
			raise RuntimeError(f"Only {len(candidates)} candidates available for {region.name}.")
		source_records.append({
			"name": region.name,
			"pbf": str(region.path),
			"pbf_sha256": sha256_file(region.path),
			"cache": str(cache) if cache.exists() else None,
			"cache_sha256": sha256_file(cache) if cache.exists() else None,
			"raw_building_rings": len(rings),
			"candidate_polygons": len(candidates),
			"projection_origin_lon_lat": list(origin),
		})

		for profile in PROFILES:
			family_diagnostic: list[gen.TestCase] = []
			family_heldout: list[gen.TestCase] = []
			for polygons in args.polygon_counts:
				seed_key = profile.seed_group or profile.name
				for split, count, destination, family in (
					("diagnostic", args.diagnostic_per_cell, diagnostic_cases, family_diagnostic),
					("heldout", args.heldout_per_cell, heldout_cases, family_heldout),
				):
					seed = stable_seed(args.seed, region.name, seed_key, polygons, split)
					generated, attempt_seeds = generate_supported_cases(
						region, profile, polygons, count, seed, candidates,
					)
					family.extend(generated)
					for case in generated:
						destination.append((case, {
							"split": split,
							"region": region.name,
							"profile": profile.name,
							"polygons": len(case.polygons),
							"vertices": sum(len(polygon) for polygon in case.polygons),
							"seed": seed,
							"attempt_seeds": attempt_seeds,
							"source_candidate_indices": case.source_candidate_indices,
						}))
			family_name = f"{region.name}-{profile.name}"
			diagnostic_path = output / "diagnostics" / f"{family_name}.bin"
			heldout_path = output / "heldout" / f"{family_name}.bin"
			gen.write_binary_cases(family_diagnostic, diagnostic_path)
			gen.write_binary_cases(family_heldout, heldout_path)
			if args.with_preview:
				gen.ensure_plot_dependency()
				gen.plot_cases(family_diagnostic, output / "previews" / f"{family_name}.png", len(family_diagnostic))
			profile_outputs.append({
				"region": region.name,
				"profile": profile.name,
				"description": profile.description,
				"diagnostic_file": str(diagnostic_path.relative_to(output)),
				"diagnostic_sha256": sha256_file(diagnostic_path),
				"diagnostic_cases": len(family_diagnostic),
				"heldout_file": str(heldout_path.relative_to(output)),
				"heldout_sha256": sha256_file(heldout_path),
				"heldout_cases": len(family_heldout),
			})

	random.Random(stable_seed(args.seed, "diagnostic-shuffle")).shuffle(diagnostic_cases)
	random.Random(stable_seed(args.seed, "heldout-shuffle")).shuffle(heldout_cases)
	diagnostic_path = output / "free-order-dev-v1.bin"
	canon_path = output / "free-order-canon-v1.bin"
	gen.write_binary_cases([case for case, _ in diagnostic_cases], diagnostic_path)
	gen.write_binary_cases([case for case, _ in heldout_cases], canon_path)

	index_rows: list[dict[str, object]] = []
	for split, entries in (("diagnostic", diagnostic_cases), ("heldout", heldout_cases)):
		for index, (case, metadata) in enumerate(entries):
			row = dict(metadata)
			row["case_index"] = index
			row["sha256"] = hashlib.sha256(encoded_case(case)).hexdigest()
			row["source_candidate_indices"] = json.dumps(row["source_candidate_indices"])
			row.pop("attempt_seeds")
			index_rows.append(row)
	write_index(output / "cases.csv", index_rows)

	manifest = {
		"schema_version": 1,
		"name": output.name,
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
		"purpose": "Independent diagnostic and frozen evaluation suites for free-order endpoint TPP.",
		"policy": {
			"diagnostic": "May be inspected and used during algorithm development.",
			"heldout": "Do not inspect case-level solver outcomes while tuning; replace the canon version after it influences design.",
		},
		"generator": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
		"git_revision": git_revision(),
		"git_dirty": git_dirty(),
		"parameters": {
			"seed": args.seed,
			"polygon_counts": list(args.polygon_counts),
			"diagnostic_per_profile_count_region": args.diagnostic_per_cell,
			"heldout_per_profile_count_region": args.heldout_per_cell,
		},
		"sources": source_records,
		"profiles": [dataclasses.asdict(profile) for profile in PROFILES],
		"profile_outputs": profile_outputs,
		"combined": {
			"diagnostic_file": diagnostic_path.name,
			"diagnostic_cases": len(diagnostic_cases),
			"diagnostic_sha256": sha256_file(diagnostic_path),
			"heldout_file": canon_path.name,
			"heldout_cases": len(heldout_cases),
			"heldout_sha256": sha256_file(canon_path),
			"case_index": "cases.csv",
		},
	}
	(output / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
	print(f"Wrote {len(diagnostic_cases)} diagnostic cases to {diagnostic_path}")
	print(f"Wrote {len(heldout_cases)} held-out cases to {canon_path}")
	print(f"Wrote provenance to {output / 'campaign.json'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
