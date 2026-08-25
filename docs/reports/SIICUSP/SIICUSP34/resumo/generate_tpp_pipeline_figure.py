from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import hypot

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon


Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Region:
	outline: list[Point]
	pieces: list[list[Point]]
	color: str
	label: str


def rectangle(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
	return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


REGIONS = [
	Region(
		outline=[
			(0.75, 1.25),
			(1.20, 1.25),
			(1.20, 1.85),
			(2.10, 1.85),
			(2.10, 2.30),
			(0.75, 2.30),
		],
		pieces=[
			rectangle(0.75, 1.25, 1.20, 2.30),
			rectangle(1.20, 1.85, 2.10, 2.30),
		],
		color="#f2b84b",
		label=r"$P_1$",
	),
	Region(
		outline=[
			(2.85, 0.45),
			(4.15, 0.45),
			(4.15, 1.45),
			(3.70, 1.45),
			(3.70, 0.90),
			(2.85, 0.90),
		],
		pieces=[
			rectangle(2.85, 0.45, 4.15, 0.90),
			rectangle(3.70, 0.90, 4.15, 1.45),
		],
		color="#66b7b0",
		label=r"$P_2$",
	),
	Region(
		outline=[
			(5.05, 1.15),
			(5.50, 1.15),
			(5.50, 1.75),
			(6.45, 1.75),
			(6.45, 2.20),
			(5.05, 2.20),
		],
		pieces=[
			rectangle(5.05, 1.15, 5.50, 2.20),
			rectangle(5.50, 1.75, 6.45, 2.20),
		],
		color="#d9796f",
		label=r"$P_3$",
	),
]

START: Point = (0.20, 0.35)
END: Point = (7.00, 0.50)
SELECTED_PIECE_INDICES = [0, 0, 0]


def piece_box(piece: list[Point]) -> Box:
	xs = [x for x, _ in piece]
	ys = [y for _, y in piece]
	return min(xs), min(ys), max(xs), max(ys)


def clamp(value: float, low: float, high: float) -> float:
	return min(max(value, low), high)


def project(point: Point, box: Box) -> Point:
	x0, y0, x1, y1 = box
	x, y = point
	return clamp(x, x0, x1), clamp(y, y0, y1)


def segment_length(a: Point, b: Point) -> float:
	return hypot(a[0] - b[0], a[1] - b[1])


def path_length(points: list[Point]) -> float:
	return sum(segment_length(a, b) for a, b in zip(points, points[1:]))


def segment_gradient(point: Point, other: Point) -> Point:
	dx = point[0] - other[0]
	dy = point[1] - other[1]
	dist = hypot(dx, dy)
	if dist < 1e-12:
		return 0.0, 0.0
	return dx / dist, dy / dist


def optimize_path(boxes: list[Box], iterations: int = 8000) -> list[Point]:
	points = [
		((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
		for box in boxes
	]
	best = points[:]
	best_value = path_length([START, *best, END])

	for iteration in range(iterations):
		full = [START, *points, END]
		step = 0.18 / (1.0 + 0.0015 * iteration)
		next_points: list[Point] = []
		for index, (point, box) in enumerate(zip(points, boxes, strict=True), start=1):
			g0 = segment_gradient(point, full[index - 1])
			g1 = segment_gradient(point, full[index + 1])
			next_points.append(project((point[0] - step * (g0[0] + g1[0]), point[1] - step * (g0[1] + g1[1])), box))
		points = next_points
		value = path_length([START, *points, END])
		if value < best_value:
			best = points[:]
			best_value = value

	return [START, *best, END]


def selected_boxes() -> list[Box]:
	return [
		piece_box(region.pieces[selected_index])
		for region, selected_index in zip(REGIONS, SELECTED_PIECE_INDICES, strict=True)
	]


def add_marker(ax: Axes, point: Point, label: str, dx: float, dy: float, *, square: bool = False) -> None:
	x, y = point
	marker = "s" if square else "o"
	ax.plot([x], [y], marker=marker, ms=3.8, color="#202124", zorder=5)
	ax.text(x + dx, y + dy, label, fontsize=8.5, color="#202124", zorder=6)


def add_region_outline(ax: Axes, region: Region, *, alpha: float = 0.68) -> None:
	ax.add_patch(
		Polygon(
			region.outline,
			closed=True,
			facecolor=region.color,
			edgecolor="#2f3a45",
			linewidth=1.05,
			alpha=alpha,
			joinstyle="miter",
		)
	)


def add_region_pieces(ax: Axes, region: Region, *, muted: bool = False) -> None:
	face = "#dfe5ea" if muted else region.color
	alpha = 0.45 if muted else 0.72
	for piece in region.pieces:
		ax.add_patch(
			Polygon(
				piece,
				closed=True,
				facecolor=face,
				edgecolor="#ffffff",
				linewidth=1.0,
				alpha=alpha,
				joinstyle="miter",
			)
		)
	ax.add_patch(
		Polygon(
			region.outline,
			closed=True,
			facecolor="none",
			edgecolor="#2f3a45",
			linewidth=1.05,
			joinstyle="miter",
		)
	)


def setup_axis(ax: Axes, panel_label: str) -> None:
	ax.set_xlim(-0.05, 7.25)
	ax.set_ylim(0.15, 2.55)
	ax.set_aspect("equal")
	ax.axis("off")
	ax.text(0.02, 2.38, panel_label, fontsize=8.5, weight="bold", color="#2f3a45")
	add_marker(ax, START, r"$s$", 0.05, -0.17)
	add_marker(ax, END, r"$t$", 0.05, 0.05, square=True)


def draw_original(ax: Axes) -> None:
	setup_axis(ax, "(a)")
	for region in REGIONS:
		add_region_outline(ax, region)
		x0, y0 = region.outline[0]
		ax.text(x0 + 0.04, y0 + 0.12, region.label, fontsize=8.5, color="#2f3a45")


def draw_decomposition(ax: Axes) -> None:
	setup_axis(ax, "(b)")
	for region in REGIONS:
		add_region_pieces(ax, region)


def draw_path(ax: Axes) -> None:
	setup_axis(ax, "(c)")
	for region in REGIONS:
		add_region_pieces(ax, region, muted=True)
	for region, selected_index in zip(REGIONS, SELECTED_PIECE_INDICES, strict=True):
		ax.add_patch(
			Polygon(
				region.pieces[selected_index],
				closed=True,
				facecolor=region.color,
				edgecolor="#2f3a45",
				linewidth=1.05,
				alpha=0.76,
				joinstyle="miter",
			)
		)
	path = optimize_path(selected_boxes())
	xs = [x for x, _ in path]
	ys = [y for _, y in path]
	ax.plot(xs, ys, color="#1f6feb", linewidth=2.05, solid_capstyle="round", zorder=4)
	for point in path[1:-1]:
		ax.add_patch(Circle(point, 0.045, color="#1f6feb", zorder=5))


def main() -> None:
	asset_dir = Path(__file__).with_name("assets")
	asset_dir.mkdir(exist_ok=True)
	output = asset_dir / "tpp-pipeline.pdf"

	fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.05), dpi=250)
	draw_original(axes[0])
	draw_decomposition(axes[1])
	draw_path(axes[2])
	plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.06, wspace=0.08)
	fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.05)
	fig.savefig(output.with_suffix(".png"), transparent=True, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
	main()
