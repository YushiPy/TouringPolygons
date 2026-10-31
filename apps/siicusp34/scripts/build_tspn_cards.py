#!/usr/bin/env python3
"""Draw the six method cards of the English page and the poster as closed tours (TSPN).

The first-stage cards show paths from s to t (TPP). These use the challenge
drawings of ``data/challenge-data.js`` but every tour is a closed tour solved by
the C++ solver with ``--cycle``; the partial node and the refinement example are
checked geometrically. Insertion uses the m gaps of a cycle, as in
``docs/algorithms/tspn.md``. Writes SVG for the page (``en/figures/``) and,
with ``--pdf-dir``, PDF for the poster. Needs matplotlib and shapely, e.g.

	apps/benchmark-dashboard/.venv/bin/python apps/siicusp34/scripts/build_tspn_cards.py \
		--pdf-dir docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon as Patch  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))

from unordered_runner import run_unordered_solver  # noqa: E402

INK, ACCENT, TEAL, MUTED = "#142d38", "#cf5a20", "#1f7a73", "#7f9aa0"
REGION, INCIDENTAL, UNVISITED, MISSED = "#bfe3d8", "#f3d9a4", "#e6edeb", "#f2b8ae"
# Letters of the five regions of the combined challenge: the partial node is (A, B, C).
LETTERS = {0: "A", 1: "B", 2: "C", 4: "D", 3: "E"}


def read_script_json(path: Path) -> dict:
	return json.loads(path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def hull(polygon: list) -> list:
	return [list(point) for point in Polygon(polygon).convex_hull.exterior.coords[:-1]]


def closed_tour(solver: Path, polygons: list) -> dict:
	anchor = tuple(polygons[0][0])
	result = run_unordered_solver(solver, anchor, anchor, [[tuple(point) for point in polygon] for polygon in polygons], -1, 60, arguments=["--cycle"])
	if result.get("termination") != "optimal" or result["path"][0] != result["path"][-1]:
		raise ValueError("The solver did not return a closed optimal tour")
	return result


def centroid(polygon: list) -> tuple[float, float]:
	point = Polygon(polygon).centroid
	return (point.x, point.y)


def tour_length(path: list) -> float:
	return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def figure(polygons_for_bounds: list, size=(4.0, 2.6)):
	fig, ax = plt.subplots(figsize=size)
	xs = [x for polygon in polygons_for_bounds for x, _ in polygon]
	ys = [y for polygon in polygons_for_bounds for _, y in polygon]
	pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys))
	ax.set_xlim(min(xs) - pad, max(xs) + pad)
	ax.set_ylim(max(ys) + pad, min(ys) - pad)  # the drawings use screen coordinates, y down
	ax.set_aspect("equal")
	ax.axis("off")
	return fig, ax


def region(ax, polygon, fill=REGION, edge=TEAL, dashed=False, width=1.4, z=1):
	ax.add_patch(Patch(polygon, closed=True, facecolor=fill, edgecolor=edge, linewidth=width, linestyle=(0, (3, 2)) if dashed else "solid", zorder=z))


def label(ax, polygon, text):
	x, y = centroid(polygon)
	ax.text(x, y, text, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=6,
		path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground="white")])


def route(ax, path, color=ACCENT, dashed=False, width=2.4, z=4):
	ax.plot([x for x, _ in path], [y for _, y in path], color=color, linewidth=width, linestyle=(0, (4, 2.5)) if dashed else "solid",
		solid_joinstyle="round", solid_capstyle="round", zorder=z)


def save(fig, name: str, args: argparse.Namespace) -> None:
	args.svg_dir.mkdir(parents=True, exist_ok=True)
	# No dates in the metadata, so rebuilding gives the same files.
	fig.savefig(args.svg_dir / f"{name}.svg", transparent=True, bbox_inches="tight", pad_inches=0.04, metadata={"Date": None})
	if args.pdf_dir:
		args.pdf_dir.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.pdf_dir / f"{name}.pdf", transparent=True, bbox_inches="tight", pad_inches=0.04, metadata={"CreationDate": None})
	plt.close(fig)


def build(args: argparse.Namespace) -> dict:
	import matplotlib.patheffects  # noqa: F401  (used through matplotlib.patheffects in label)

	plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "siicusp34"})
	data = read_script_json(args.challenges)
	report = {}

	# 01: a greedy tour through the centres (nearest neighbour from region A) against the optimum.
	polygons = data["geometry"]["polygons"]
	centres = [centroid(polygon) for polygon in polygons]
	order, left = [0], set(range(1, len(polygons)))
	while left:
		nearest = min(left, key=lambda index: math.dist(centres[order[-1]], centres[index]))
		order.append(nearest)
		left.remove(nearest)
	greedy = [centres[index] for index in order] + [centres[0]]
	optimum = closed_tour(args.solver, polygons)
	if tour_length(greedy) <= optimum["upper_bound"]:
		raise ValueError("The greedy tour should be longer than the optimum")
	fig, ax = figure(polygons)
	for polygon in polygons:
		region(ax, polygon)
	route(ax, optimum["path"], color=INK, dashed=True, width=1.8)
	route(ax, greedy)
	save(fig, "method-01", args)
	report["01"] = {"greedy": tour_length(greedy), "optimum": optimum["upper_bound"]}

	combined = data["combined_challenge"]
	polygons = combined["geometry"]["polygons"]
	hulls = [hull(polygon) for polygon in polygons]
	shapes = [Polygon(polygon) for polygon in polygons]

	# 02: the node (A, B, C) as hulls; its tour touches D by chance and misses E.
	node = [0, 1, 2]
	partial = closed_tour(args.solver, [hulls[index] for index in node])
	line = LineString(partial["path"])
	if not line.intersects(shapes[4]) or line.intersects(shapes[3]):
		raise ValueError("The partial node should touch D and miss E")
	fig, ax = figure(polygons)
	for index, polygon in enumerate(polygons):
		if index in node:
			region(ax, polygon)
		elif index == 4:
			region(ax, polygon, fill=INCIDENTAL, edge="#a87b22")
		else:
			region(ax, polygon, fill=UNVISITED, edge=MUTED, dashed=True)
	route(ax, partial["path"])
	for index, polygon in enumerate(polygons):
		label(ax, polygon, LETTERS[index])
	save(fig, "method-02", args)
	report["02"] = {"length": partial["upper_bound"]}

	# 04: the optimal tour through the five regions themselves, a feasible tour.
	full = closed_tour(args.solver, polygons)
	if not all(LineString(full["path"]).intersects(shape) for shape in shapes):
		raise ValueError("The full tour must visit every region")
	fig, ax = figure(polygons)
	for polygon in polygons:
		region(ax, polygon)
	route(ax, full["path"])
	for index, polygon in enumerate(polygons):
		label(ax, polygon, LETTERS[index])
	save(fig, "method-04", args)
	report["04"] = {"length": full["upper_bound"]}

	# 06: the tour through the hulls misses some regions; a missed region splits into its pieces.
	relaxed = closed_tour(args.solver, hulls)
	line = LineString(relaxed["path"])
	missed = [index for index, shape in enumerate(shapes) if not line.intersects(shape)]
	if not missed:
		raise ValueError("The hull tour should miss a region")
	refined = missed[0]
	fig, ax = figure(polygons)
	for index, polygon in enumerate(polygons):
		region(ax, hulls[index], fill="none", edge=MUTED, dashed=True, width=1.1, z=0)
		if index == refined:
			for piece in combined["pieces"][index]:
				region(ax, piece, fill=MISSED, edge="#b0473c", width=1.3, z=2)
		else:
			region(ax, polygon, fill=MISSED if index in missed else REGION, edge="#b0473c" if index in missed else TEAL)
	route(ax, relaxed["path"])
	for index, polygon in enumerate(polygons):
		label(ax, polygon, LETTERS[index])
	save(fig, "method-06", args)
	report["06"] = {"missed": [LETTERS[index] for index in missed], "refined": LETTERS[refined], "pieces": len(combined["pieces"][refined])}

	# 03: pruning, L reaching U.
	fig, ax = plt.subplots(figsize=(4.0, 1.6))
	ax.set_xlim(0, 10)
	ax.set_ylim(0, 4)
	ax.axis("off")
	ax.annotate("", xy=(9.6, 2), xytext=(0.4, 2), arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.5})
	for x, color, text in ((3.8, TEAL, "U"), (6.4, ACCENT, "L")):
		ax.plot([x, x], [1.3, 2.7], color=color, linewidth=3)
		ax.text(x, 3.3, text, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
	ax.text(5.1, 0.55, "L ≥ U − ε", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
	save(fig, "method-03", args)

	# 05: inserting P into each of the m = 3 gaps of the cycle (A B C).
	fig, ax = plt.subplots(figsize=(4.0, 1.9))
	ax.set_xlim(0, 12)
	ax.set_ylim(0, 6)
	ax.axis("off")
	def box(x, y, text, edge):
		ax.add_patch(FancyBboxPatch((x - 1.55, y - 0.55), 3.1, 1.1, boxstyle="round,pad=0.05,rounding_size=0.25", facecolor="#eef4f1", edgecolor=edge, linewidth=1.5))
		ax.text(x, y, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
	for x in (2.0, 6.0, 10.0):
		ax.plot([6.0, x], [4.4, 2.0], color=ACCENT, linewidth=1.6)
	box(6.0, 4.8, "(A B C)", TEAL)
	for x, text in ((2.0, "(A P B C)"), (6.0, "(A B P C)"), (10.0, "(A B C P)")):
		box(x, 1.4, text, ACCENT)
	save(fig, "method-05", args)
	return report


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--challenges", type=Path, default=ROOT / "apps/siicusp34/data/challenge-data.js")
	parser.add_argument("--solver", type=Path, default=ROOT / ".build/tools/bin/tpp-unordered")
	parser.add_argument("--svg-dir", type=Path, default=ROOT / "apps/siicusp34/en/figures")
	parser.add_argument("--pdf-dir", type=Path)
	args = parser.parse_args()
	print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
	main()
