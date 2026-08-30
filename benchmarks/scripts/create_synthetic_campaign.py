#!/usr/bin/env python3
"""Create a synthetic benchmark campaign with exact polygon and vertex counts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class TestCase:
	start: Point
	target: Point
	polygons: list[list[Point]]


def positive_int(value: str) -> int:
	parsed = int(value)
	if parsed < 1:
		raise argparse.ArgumentTypeError("must be at least 1")
	return parsed


def parse_vertex_counts(text: str, polygons: int) -> list[int]:
	values = [int(value.strip()) for value in text.split(",") if value.strip()]
	if not values:
		raise argparse.ArgumentTypeError("must contain at least one vertex count")
	if any(value < 3 for value in values):
		raise argparse.ArgumentTypeError("each polygon must have at least 3 vertices")
	if len(values) == 1:
		return values * polygons
	if len(values) != polygons:
		raise argparse.ArgumentTypeError(
			f"expected one value or exactly {polygons} comma-separated values"
		)
	return values


def regular_polygon(center: Point, radius: float, vertices: int, rotation: float) -> list[Point]:
	return [
		(
			center[0] + math.cos(rotation + 2.0 * math.pi * index / vertices) * radius,
			center[1] + math.sin(rotation + 2.0 * math.pi * index / vertices) * radius,
		)
		for index in range(vertices)
	]


def star_polygon(center: Point, radius: float, vertices: int, rotation: float) -> list[Point]:
	if vertices < 4:
		return regular_polygon(center, radius, vertices, rotation)

	inner_radius = radius * 0.45
	return [
		(
			center[0] + math.cos(rotation + 2.0 * math.pi * index / vertices)
			* (radius if index % 2 == 0 else inner_radius),
			center[1] + math.sin(rotation + 2.0 * math.pi * index / vertices)
			* (radius if index % 2 == 0 else inner_radius),
		)
		for index in range(vertices)
	]


def make_case(
	vertex_counts: Sequence[int],
	shape: str,
	grid_spacing: float,
	polygon_radius: float,
	rng: random.Random,
) -> TestCase:
	columns = math.ceil(math.sqrt(len(vertex_counts)))
	rows = math.ceil(len(vertex_counts) / columns)
	cells = [
		((index % columns) * grid_spacing, (index // columns) * grid_spacing)
		for index in range(len(vertex_counts))
	]
	rng.shuffle(cells)

	polygons: list[list[Point]] = []
	for vertices, center in zip(vertex_counts, cells):
		radius = polygon_radius * rng.uniform(0.8, 1.0)
		rotation = rng.random() * 2.0 * math.pi
		if shape == "convex":
			polygons.append(regular_polygon(center, radius, vertices, rotation))
		else:
			polygons.append(star_polygon(center, radius, vertices, rotation))

	start = (-grid_spacing * 0.65, -grid_spacing * 0.65)
	target = ((columns - 1) * grid_spacing + grid_spacing * 0.65, (rows - 1) * grid_spacing + grid_spacing * 0.65)
	return TestCase(start=start, target=target, polygons=polygons)


def write_vector(file, point: Point) -> None:
	import struct

	file.write(struct.pack("<dd", point[0], point[1]))


def write_size(file, value: int) -> None:
	import struct

	file.write(struct.pack("<Q", value))


def write_cases(path: Path, cases: Sequence[TestCase]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as file:
		for case in cases:
			write_vector(file, case.start)
			write_vector(file, case.target)
			write_size(file, len(case.polygons))
			for polygon in case.polygons:
				write_size(file, len(polygon))
				for point in polygon:
					write_vector(file, point)
			write_size(file, 0)


def svg_points(points: Sequence[Point], offset_x: float, offset_y: float, scale: float) -> str:
	return " ".join(
		f"{offset_x + point[0] * scale:.2f},{offset_y - point[1] * scale:.2f}"
		for point in points
	)


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, opacity: float = 1.0) -> str:
	return (
		f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
		f'stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="1"/>'
	)


def preview_grid_metrics(scale: float) -> tuple[float, int]:
	decision_value = 83 / scale
	exponent = math.ceil(math.log10(decision_value)) if decision_value > 0 else 0
	multiplier = 1
	sub_grid_count = 4
	grid_scale = 10 ** exponent
	if grid_scale / 5 > decision_value:
		sub_grid_count = 3
		exponent -= 1
		multiplier = 2
	elif grid_scale / 2 > decision_value:
		exponent -= 1
		multiplier = 5
	return (10 ** exponent) * multiplier, sub_grid_count


def case_bounds(case: TestCase) -> tuple[float, float, float, float]:
	points = [case.start, case.target, *(point for polygon in case.polygons for point in polygon)]
	return (
		min(point[0] for point in points),
		min(point[1] for point in points),
		max(point[0] for point in points),
		max(point[1] for point in points),
	)


def write_preview(
	path: Path,
	cases: Sequence[TestCase],
	max_cases: int,
	*,
	cell_size: int = 240,
	columns: int = 5,
) -> None:
	preview_cases = list(cases[:max_cases])
	columns = min(columns, max(1, len(preview_cases)))
	rows = math.ceil(len(preview_cases) / columns)
	padding = 18
	width = columns * cell_size
	height = rows * cell_size
	colors = ["#38bdf8", "#a3e635", "#f97316", "#f472b6", "#c084fc"]
	elements = [
		f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
		'<rect width="100%" height="100%" fill="#121417"/>',
	]

	for case_index, case in enumerate(preview_cases):
		col = case_index % columns
		row = case_index // columns
		x0 = col * cell_size
		y0 = row * cell_size
		min_x, min_y, max_x, max_y = case_bounds(case)
		span = max(max_x - min_x, max_y - min_y, 1e-9)
		scale = (cell_size - 2 * padding) / span
		offset_x = x0 + padding - min_x * scale
		offset_y = y0 + cell_size - padding + min_y * scale
		elements.append(f'<rect x="{x0}" y="{y0}" width="{cell_size}" height="{cell_size}" fill="#121417"/>')
		grid_step, sub_grid_count = preview_grid_metrics(scale)
		first_x = math.floor(min_x / grid_step) * grid_step
		first_y = math.floor(min_y / grid_step) * grid_step
		x = first_x
		while x <= max_x + grid_step:
			screen_x = offset_x + x * scale
			if x0 <= screen_x <= x0 + cell_size:
				elements.append(svg_line(screen_x, y0, screen_x, y0 + cell_size, "#515a67", 0.62))
				for index in range(sub_grid_count):
					sub_x = screen_x + (index + 1) * grid_step * scale / (sub_grid_count + 1)
					if x0 <= sub_x <= x0 + cell_size:
						elements.append(svg_line(sub_x, y0, sub_x, y0 + cell_size, "#2a2f38", 0.74))
			x += grid_step
		y = first_y
		while y <= max_y + grid_step:
			screen_y = offset_y - y * scale
			if y0 <= screen_y <= y0 + cell_size:
				elements.append(svg_line(x0, screen_y, x0 + cell_size, screen_y, "#515a67", 0.62))
				for index in range(sub_grid_count):
					sub_y = screen_y - (index + 1) * grid_step * scale / (sub_grid_count + 1)
					if y0 <= sub_y <= y0 + cell_size:
						elements.append(svg_line(x0, sub_y, x0 + cell_size, sub_y, "#2a2f38", 0.74))
			y += grid_step
		origin_x = offset_x
		origin_y = offset_y
		if y0 <= origin_y <= y0 + cell_size:
			elements.append(svg_line(x0, origin_y, x0 + cell_size, origin_y, "#9aa3ad", 0.86))
		if x0 <= origin_x <= x0 + cell_size:
			elements.append(svg_line(origin_x, y0, origin_x, y0 + cell_size, "#9aa3ad", 0.86))

		for polygon_index, polygon in enumerate(case.polygons):
			color = colors[polygon_index % len(colors)]
			elements.append(
				f'<polygon points="{svg_points(polygon, offset_x, offset_y, scale)}" '
				f'fill="{color}" fill-opacity="0.20" stroke="{color}" stroke-width="2"/>'
			)
			for point_x, point_y in polygon:
				screen_x = offset_x + point_x * scale
				screen_y = offset_y - point_y * scale
				elements.append(f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="3.6" fill="{color}"/>')

		start_x = offset_x + case.start[0] * scale
		start_y = offset_y - case.start[1] * scale
		target_x = offset_x + case.target[0] * scale
		target_y = offset_y - case.target[1] * scale
		elements.append(f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="5" fill="#22c55e"/>')
		elements.append(f'<text x="{start_x + 10:.2f}" y="{start_y + 4:.2f}" font-size="12" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" fill="#f8fafc">s</text>')
		elements.append(f'<circle cx="{target_x:.2f}" cy="{target_y:.2f}" r="5" fill="#ef4444"/>')
		elements.append(f'<text x="{target_x + 10:.2f}" y="{target_y + 4:.2f}" font-size="12" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" fill="#f8fafc">t</text>')

	elements.append("</svg>")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(elements) + "\n")


def write_previews(output_dir: Path, cases: Sequence[TestCase], max_cases: int) -> dict[str, str]:
	previews = {
		"selected": output_dir / "selected.svg",
		"four": output_dir / "four.svg",
		"all": output_dir / "all.svg",
	}
	write_preview(previews["selected"], cases[:1], 1, cell_size=320, columns=1)
	write_preview(previews["four"], cases[:4], 4, cell_size=160, columns=2)
	write_preview(previews["all"], cases, max_cases, cell_size=120, columns=10)
	return {name: str(path.relative_to(output_dir.parent)) for name, path in previews.items()}


def write_instance_previews(output_dir: Path, cases: Sequence[TestCase]) -> list[str]:
	instance_dir = output_dir / "instances"
	instance_dir.mkdir(parents=True, exist_ok=True)
	paths: list[str] = []
	for index, case in enumerate(cases):
		path = instance_dir / f"case-{index:04}.svg"
		write_preview(path, [case], 1, cell_size=260, columns=1)
		paths.append(str(path.relative_to(output_dir.parent)))
	return paths


def git_revision(repo_root: Path) -> str | None:
	completed = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=repo_root,
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.DEVNULL,
		check=False,
	)
	return completed.stdout.strip() if completed.returncode == 0 else None


def git_is_dirty(repo_root: Path) -> bool | None:
	completed = subprocess.run(
		["git", "status", "--porcelain"],
		cwd=repo_root,
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.DEVNULL,
		check=False,
	)
	return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Create a synthetic TPP benchmark campaign.")
	parser.add_argument("--campaign", type=Path, required=True, help="Campaign directory.")
	parser.add_argument("--instances", type=positive_int, default=100, help="Number of instances to generate.")
	parser.add_argument("--polygons", type=positive_int, default=20, help="Number of polygons per instance.")
	parser.add_argument(
		"--vertices",
		default="8",
		help="Vertices per polygon. Use one value, or one comma-separated value per polygon.",
	)
	parser.add_argument("--shape", choices=("star", "convex"), default="star")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--grid-spacing", type=float, default=3.0)
	parser.add_argument("--polygon-radius", type=float, default=0.9)
	parser.add_argument("--no-preview", action="store_true")
	parser.add_argument("--preview-cases", type=positive_int, default=25)
	parser.add_argument("--overwrite", action="store_true", help="Replace an existing campaign directory.")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	campaign = args.campaign.resolve()
	campaign_file = campaign / "campaign.json"
	if campaign.exists() and args.overwrite:
		shutil.rmtree(campaign)
	elif campaign_file.exists():
		raise SystemExit(f"Campaign already exists: {campaign}")
	if args.grid_spacing <= 0.0:
		raise SystemExit("--grid-spacing must be positive")
	if args.polygon_radius <= 0.0:
		raise SystemExit("--polygon-radius must be positive")
	if args.polygon_radius * 2.0 >= args.grid_spacing:
		raise SystemExit("--grid-spacing must be greater than twice --polygon-radius")

	vertex_counts = parse_vertex_counts(args.vertices, args.polygons)
	rng = random.Random(args.seed)
	cases = [
		make_case(vertex_counts, args.shape, args.grid_spacing, args.polygon_radius, rng)
		for _ in range(args.instances)
	]

	input_path = campaign / "inputs" / "synthetic.bin"
	preview_dir = campaign / "previews"
	write_cases(input_path, cases)
	print(f"Wrote {len(cases)} cases to {input_path}", flush=True)

	previews: dict[str, str] | None = None
	instance_previews: list[str] = []
	if not args.no_preview:
		previews = write_previews(preview_dir, cases, args.preview_cases)
		instance_previews = write_instance_previews(preview_dir, cases)
		print(f"Wrote previews to {preview_dir}", flush=True)

	repo_root = Path(__file__).resolve().parents[2]
	data = {
		"schema_version": 1,
		"name": campaign.name,
		"type": "synthetic",
		"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
		"git_revision": git_revision(repo_root),
		"git_dirty": git_is_dirty(repo_root),
		"generation": {
			"instances": args.instances,
			"polygons": args.polygons,
			"vertices": vertex_counts,
			"shape": args.shape,
			"seed": args.seed,
			"grid_spacing": args.grid_spacing,
			"polygon_radius": args.polygon_radius,
		},
		"inputs": [{
			"file": "inputs/synthetic.bin",
			"instances": args.instances,
			"polygons_per_instance": args.polygons,
			"vertices_per_polygon": vertex_counts,
			"shape": args.shape,
			"seed": args.seed,
		}],
		"preview": previews["all"] if previews else None,
		"previews": previews,
		"instance_previews": instance_previews,
		"benchmark_runs": [],
	}
	campaign_file.write_text(json.dumps(data, indent=2) + "\n")
	print(f"Wrote campaign metadata to {campaign_file}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
