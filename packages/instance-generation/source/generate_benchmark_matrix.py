#!/usr/bin/env python3
"""
Generate a matrix of OSM-derived TPP benchmark binaries.

It intentionally writes binary test sets as the primary output and skips
previews/manifests by default. Unlike a shell wrapper, it loads the OSM cache
and builds the candidate polygon pool once, then reuses it for every output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
from pathlib import Path
from typing import Sequence

import gen_instances


DEFAULT_POLYGON_COUNTS = [1, 3, 5, 10, 20, 30, 40, 50]
DEFAULT_GRID_SPACINGS = [1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
DEFAULT_CONVEX_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_csv_numbers(text: str, cast):
	return [cast(value.strip()) for value in text.split(",") if value.strip()]


def format_number(value: float) -> str:
	text = f"{value:g}"
	return text.replace(".", "p")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate a benchmark matrix of TPP instance binaries.")
	parser.add_argument("input_pbf", type=Path, help="Input .osm.pbf file.")
	parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/campaigns/generated/inputs"), help="Directory for generated .bin files.")
	parser.add_argument("--instances", type=int, default=100, help="Instances per generated binary.")
	parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
	parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample this many jobs from the full matrix. Defaults to all jobs.")
	parser.add_argument("--polygon-counts", default=",".join(str(value) for value in DEFAULT_POLYGON_COUNTS), help="Comma-separated polygons-per-instance values.")
	parser.add_argument("--polygon-count", type=int, help="Single polygons-per-instance value. Overrides --polygon-counts.")
	parser.add_argument("--layouts", default="geographic,grid", help="Comma-separated layouts: geographic,grid.")
	parser.add_argument("--layout", choices=("geographic", "grid"), help="Single layout. Overrides --layouts.")
	parser.add_argument("--grid-spacings", default=",".join(str(value) for value in DEFAULT_GRID_SPACINGS), help="Comma-separated grid cell sizes.")
	parser.add_argument("--grid-cell-size", type=float, help="Single grid cell size. Overrides --grid-spacings.")
	parser.add_argument("--convex-fractions", default=",".join(str(value) for value in DEFAULT_CONVEX_FRACTIONS), help="Comma-separated synthetic convex replacement fractions.")
	parser.add_argument("--convex-replacement-fraction", type=float, help="Single convex replacement fraction. Overrides --convex-fractions.")
	parser.add_argument("--grid-polygon-size", type=float, default=1.0)
	parser.add_argument("--grid-columns", type=int, default=0)
	parser.add_argument("--grid-placement", choices=("row-major", "random"), default="row-major")
	parser.add_argument("--order", choices=("spatial", "left-to-right", "random", "angle"), default="spatial")
	parser.add_argument("--sampling", choices=("local", "uniform"), default="local")
	parser.add_argument("--local-pool-size", type=int, default=80)
	parser.add_argument("--simplify-tolerance", type=float, default=1.0)
	parser.add_argument("--scale", type=float, default=1.0)
	parser.add_argument("--normalization", choices=("instance", "dataset", "none"), default="instance")
	parser.add_argument("--endpoint-mode", choices=("ordered", "bbox"), default="ordered")
	parser.add_argument("--candidate-pool", choices=("nonconvex", "all"), default="nonconvex")
	parser.add_argument("--nonconvex-threshold", type=float, default=0.98)
	parser.add_argument("--min-area", type=float, default=10.0)
	parser.add_argument("--min-vertices", type=int, default=4)
	parser.add_argument("--max-vertices", type=int, default=80)
	parser.add_argument("--convex-replacement-vertices", type=int, default=64)
	parser.add_argument("--convex-replacement-scale", type=float, default=1.0)
	parser.add_argument("--convex-replacement-position", choices=("middle", "random", "alternating"), default="middle")
	parser.add_argument("--cache", type=Path, help="Raw building-ring cache. Defaults to <input>.buildings.pkl.")
	parser.add_argument("--no-cache", action="store_true", help="Do not read or write the building-ring cache.")
	parser.add_argument("--with-preview", action="store_true", help="Generate previews for every binary.")
	parser.add_argument("--with-manifest", action="store_true", help="Generate manifests for every binary.")
	parser.add_argument("--single-preview-count", type=int, default=3)
	parser.add_argument("--campaign-file", type=Path, help=argparse.SUPPRESS)
	parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
	return parser.parse_args(argv)


def matrix_job_args(
	args: argparse.Namespace,
	*,
	polygons: int,
	layout: str,
	spacing: float | None,
	convex_fraction: float,
	output_bin: Path,
	preview: Path,
	seed: int,
) -> argparse.Namespace:
	return argparse.Namespace(
		input_pbf=args.input_pbf,
		output_bin=output_bin,
		preview=preview,
		manifest=None,
		no_preview=not args.with_preview,
		no_manifest=not args.with_manifest,
		instances=args.instances,
		polygons_per_instance=polygons,
		seed=seed,
		simplify_tolerance=args.simplify_tolerance,
		scale=args.scale,
		normalization=args.normalization,
		order=args.order,
		sampling=args.sampling,
		local_pool_size=max(args.local_pool_size, polygons),
		layout=layout,
		grid_polygon_size=args.grid_polygon_size,
		grid_cell_size=spacing if spacing is not None else 0.0,
		grid_columns=args.grid_columns,
		grid_placement=args.grid_placement,
		convex_replacement_fraction=convex_fraction,
		convex_replacement_vertices=args.convex_replacement_vertices,
		convex_replacement_scale=args.convex_replacement_scale,
		convex_replacement_position=args.convex_replacement_position,
		endpoint_mode=args.endpoint_mode,
		candidate_pool=args.candidate_pool,
		nonconvex_threshold=args.nonconvex_threshold,
		min_area=args.min_area,
		min_vertices=args.min_vertices,
		max_vertices=args.max_vertices,
		single_preview_count=args.single_preview_count,
		single_preview_dir=None,
		cache=args.cache,
		no_cache=args.no_cache,
	)


def describe_job(job_args: argparse.Namespace) -> str:
	parts = [
		f"output={job_args.output_bin}",
		f"instances={job_args.instances}",
		f"polygons={job_args.polygons_per_instance}",
		f"layout={job_args.layout}",
		f"seed={job_args.seed}",
		f"convex_fraction={job_args.convex_replacement_fraction:g}",
	]

	if job_args.layout == "grid":
		parts.append(f"grid_spacing={job_args.grid_cell_size:g}")

	return " ".join(parts)


def build_jobs(args: argparse.Namespace) -> tuple[list[argparse.Namespace], int]:
	polygon_counts = [args.polygon_count] if args.polygon_count is not None else parse_csv_numbers(args.polygon_counts, int)
	layouts = [args.layout] if args.layout else [layout.strip() for layout in args.layouts.split(",") if layout.strip()]
	grid_spacings = [args.grid_cell_size] if args.grid_cell_size is not None else parse_csv_numbers(args.grid_spacings, float)
	convex_fractions = (
		[args.convex_replacement_fraction]
		if args.convex_replacement_fraction is not None
		else parse_csv_numbers(args.convex_fractions, float)
	)

	for layout in layouts:
		if layout not in {"geographic", "grid"}:
			raise SystemExit(f"Unsupported layout: {layout}")

	jobs: list[argparse.Namespace] = []

	for polygons in polygon_counts:
		for convex_fraction in convex_fractions:
			for layout in layouts:
				spacings = grid_spacings if layout == "grid" else [None]

				for spacing in spacings:
					parts = [
						f"p{polygons}",
						layout,
						f"convex{format_number(convex_fraction)}",
					]

					if spacing is not None:
						parts.append(f"spacing{format_number(spacing)}")

					basename = "_".join(parts)
					output_bin = args.output_dir / f"{basename}.bin"
					preview = args.output_dir / f"{basename}.png"
					seed = args.seed + len(jobs)
					jobs.append(matrix_job_args(
						args,
						polygons=polygons,
						layout=layout,
						spacing=spacing,
						convex_fraction=convex_fraction,
						output_bin=output_bin,
						preview=preview,
						seed=seed,
					))

	total_jobs = len(jobs)

	if args.sample_size < 0:
		raise SystemExit("--sample-size must be non-negative.")

	if 0 < args.sample_size < total_jobs:
		sample_rng = random.Random(args.seed)
		selected_indices = sorted(sample_rng.sample(range(total_jobs), args.sample_size))
		jobs = [jobs[index] for index in selected_indices]
		print(f"Sampled {len(jobs)} jobs from {total_jobs} matrix combinations with seed {args.seed}.", flush=True)

	return jobs, total_jobs


def git_revision() -> str | None:
	completed = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=Path(__file__).resolve().parents[3],
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.DEVNULL,
		check=False,
	)
	return completed.stdout.strip() if completed.returncode == 0 else None


def git_is_dirty() -> bool | None:
	completed = subprocess.run(
		["git", "status", "--porcelain"],
		cwd=Path(__file__).resolve().parents[3],
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.DEVNULL,
		check=False,
	)
	return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def job_record(job: argparse.Namespace, campaign_dir: Path) -> dict:
	try:
		output = job.output_bin.resolve().relative_to(campaign_dir.resolve())
	except ValueError:
		output = job.output_bin.resolve()

	return {
		"file": str(output),
		"instances": job.instances,
		"polygons_per_instance": job.polygons_per_instance,
		"seed": job.seed,
		"layout": job.layout,
		"order": job.order,
		"local_pool_size": job.local_pool_size,
		"grid_cell_size": job.grid_cell_size if job.layout == "grid" else None,
		"grid_polygon_size": job.grid_polygon_size if job.layout == "grid" else None,
		"grid_columns": job.grid_columns if job.layout == "grid" else None,
		"grid_placement": job.grid_placement if job.layout == "grid" else None,
		"convex_replacement_fraction": job.convex_replacement_fraction,
		"convex_replacement_vertices": job.convex_replacement_vertices,
	}


def write_campaign(
	args: argparse.Namespace,
	jobs: Sequence[argparse.Namespace],
	total_jobs: int,
	origin: tuple[float, float],
	candidate_count: int,
) -> None:
	if args.campaign_file is None:
		return

	campaign_file = args.campaign_file.resolve()
	campaign_file.parent.mkdir(parents=True, exist_ok=True)
	pbf_path = args.input_pbf.resolve()
	pbf_stat = pbf_path.stat()
	cache_path = args.cache if args.cache else args.input_pbf.with_suffix(args.input_pbf.suffix + ".buildings.pkl")
	data = {
		"schema_version": 1,
		"name": campaign_file.parent.name,
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
		"git_revision": git_revision(),
		"git_dirty": git_is_dirty(),
		"source": {
			"pbf": str(pbf_path),
			"pbf_size_bytes": pbf_stat.st_size,
			"pbf_mtime_ns": pbf_stat.st_mtime_ns,
			"building_cache": str(cache_path.resolve()) if not args.no_cache else None,
			"projection_origin_lon_lat": list(origin),
			"candidate_count": candidate_count,
		},
		"generation": {
			"instances_per_file": args.instances,
			"base_seed": args.seed,
			"sample_size": args.sample_size,
			"total_matrix_combinations": total_jobs,
			"selected_combinations": len(jobs),
			"polygon_counts": [args.polygon_count] if args.polygon_count is not None else parse_csv_numbers(args.polygon_counts, int),
			"layouts": [args.layout] if args.layout else [value.strip() for value in args.layouts.split(",") if value.strip()],
			"grid_spacings": [args.grid_cell_size] if args.grid_cell_size is not None else parse_csv_numbers(args.grid_spacings, float),
			"convex_fractions": (
				[args.convex_replacement_fraction]
				if args.convex_replacement_fraction is not None
				else parse_csv_numbers(args.convex_fractions, float)
			),
			"sampling": args.sampling,
			"local_pool_size": args.local_pool_size,
			"simplify_tolerance": args.simplify_tolerance,
			"scale": args.scale,
			"normalization": args.normalization,
			"order": args.order,
			"endpoint_mode": args.endpoint_mode,
			"grid_polygon_size": args.grid_polygon_size,
			"grid_columns": args.grid_columns,
			"grid_placement": args.grid_placement,
			"convex_replacement_vertices": args.convex_replacement_vertices,
			"convex_replacement_scale": args.convex_replacement_scale,
			"convex_replacement_position": args.convex_replacement_position,
			"candidate_pool": args.candidate_pool,
			"nonconvex_threshold": args.nonconvex_threshold,
			"min_area": args.min_area,
			"min_vertices": args.min_vertices,
			"max_vertices": args.max_vertices,
		},
		"inputs": [job_record(job, campaign_file.parent) for job in jobs],
		"benchmark_runs": [],
	}
	campaign_file.write_text(json.dumps(data, indent=2) + "\n")
	print(f"Wrote campaign metadata to {campaign_file}", flush=True)


def run_matrix(args: argparse.Namespace) -> None:
	jobs, total_jobs = build_jobs(args)

	for job in jobs:
		print("+", describe_job(job), flush=True)

	if args.dry_run:
		print(f"Prepared {len(jobs)} generation jobs from {total_jobs} matrix combinations.", flush=True)
		return

	args.output_dir.mkdir(parents=True, exist_ok=True)

	cache_path = args.cache if args.cache else args.input_pbf.with_suffix(args.input_pbf.suffix + ".buildings.pkl")
	rings = gen_instances.load_building_rings(args.input_pbf, cache_path, not args.no_cache)
	gen_instances.ensure_geometry_dependencies()
	origin = gen_instances.projection_origin(rings)
	candidate_args = jobs[0] if jobs else matrix_job_args(
		args,
		polygons=1,
		layout="geographic",
		spacing=None,
		convex_fraction=0.0,
		output_bin=args.output_dir / "unused.bin",
		preview=args.output_dir / "unused.png",
		seed=args.seed,
	)
	candidates = gen_instances.build_candidates(candidate_args, rings, origin)
	print(f"Selected {len(candidates)} candidate polygons from {len(rings)} raw building rings.", flush=True)

	if args.with_preview:
		gen_instances.ensure_plot_dependency()

	for index, job in enumerate(jobs, start=1):
		print(f"[{index}/{len(jobs)}] Generating {job.output_bin}", flush=True)
		cases = gen_instances.generate_cases(job, candidates)
		gen_instances.write_binary_cases(cases, job.output_bin)

		if args.with_preview:
			single_preview_dir = job.preview.with_name(f"{job.preview.stem}-instances")
			gen_instances.plot_cases(cases, job.preview)
			gen_instances.plot_single_case_previews(cases, single_preview_dir, job.single_preview_count)

		if args.with_manifest:
			manifest_path = job.output_bin.with_suffix(job.output_bin.suffix + ".manifest.json")
			gen_instances.write_manifest(job, cases, candidates, manifest_path, origin)

	write_campaign(args, jobs, total_jobs, origin, len(candidates))
	print(f"Generated {len(jobs)} binary files from {total_jobs} matrix combinations.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	run_matrix(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
