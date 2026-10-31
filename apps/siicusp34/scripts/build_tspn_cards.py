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
from shapely.geometry import LineString, Point, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))

from unordered_runner import run_unordered_solver  # noqa: E402

INK, ACCENT, TEAL, MUTED = "#142d38", "#cf5a20", "#1f7a73", "#7f9aa0"
REGION, INCIDENTAL, UNVISITED, MISSED = "#bfe3d8", "#f3d9a4", "#e6edeb", "#f2b8ae"
# Letters of the five regions of the combined challenge: the partial node is (A, B, C).
LETTERS = {0: "A", 1: "B", 2: "C", 4: "D", 3: "E"}
# Shared label positions keep names clear of the tours and the partition diagonal.
LABELS = {0: (308, 183), 1: (115, 106), 2: (280, 57), 3: (146, 205), 4: (198, 102)}
GEOMETRY_TOLERANCE = 1e-7


def read_script_json(path: Path) -> dict:
	return json.loads(path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def hull(polygon: list) -> list:
	return [list(point) for point in Polygon(polygon).convex_hull.exterior.coords[:-1]]


def validate_tour(path: list, polygons: list) -> LineString:
	if len(path) < 3 or path[0] != path[-1]:
		raise ValueError("The tour is not closed")
	line = LineString(path)
	if any(line.distance(Polygon(polygon)) > GEOMETRY_TOLERANCE for polygon in polygons):
		raise ValueError("The tour misses a target of its own subproblem")
	return line


def closed_tour(solver: Path, polygons: list, *, trace=False) -> dict:
	anchor = tuple(polygons[0][0])
	result = run_unordered_solver(solver, anchor, anchor, [[tuple(point) for point in polygon] for polygon in polygons], -1, 60,
		arguments=["--cycle", *(["--trace"] if trace else [])])
	if result.get("termination") != "optimal" or result["path"][0] != result["path"][-1]:
		raise ValueError("The solver did not return a closed optimal tour")
	if result["upper_bound"] - result["lower_bound"] > 1e-9 * max(1, abs(result["upper_bound"])):
		raise ValueError("The illustrated optimum has an unresolved gap")
	validate_tour(result["path"], polygons)
	return result


def initial_heuristic(result: dict, polygons: list) -> list:
	"""Read the actual initial incumbent; never reconstruct the heuristic in Python."""
	candidates = [event for event in result.get("trace", [])
		if event.get("kind") == "incumbent" and event.get("source", "").startswith("heuristic")
		and math.isclose(event.get("length", math.inf), result["initial_length"], rel_tol=1e-12, abs_tol=1e-9)]
	if not candidates:
		raise ValueError("No native initial incumbent matches initial_length")
	path = candidates[-1]["path"]
	validate_tour(path, polygons)
	# A straight passage may contain a redundant interior point; a strict bend may not.
	for index, point in enumerate(path[:-1]):
		a, b = path[index - 1] if index else path[-2], path[index + 1]
		cross = (point[0] - a[0]) * (b[1] - point[1]) - (point[1] - a[1]) * (b[0] - point[0])
		if abs(cross) > GEOMETRY_TOLERANCE and all(Polygon(p).boundary.distance(Point(point)) > GEOMETRY_TOLERANCE
			for p in polygons):
			raise ValueError("The initial heuristic bends strictly inside a region")
	return path


def centroid(polygon: list) -> tuple[float, float]:
	point = Polygon(polygon).centroid
	return (point.x, point.y)


def tour_length(path: list) -> float:
	return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def figure(polygons_for_bounds: list, size=(4.0, 2.8), *, branches=False):
	fig, ax = plt.subplots(figsize=size)
	if branches:
		ax.set_position([0.03, 0.27, 0.94, 0.70])
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


def label(ax, polygon, text, position=None, *, fontsize=12):
	x, y = position or centroid(polygon)
	ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=INK, zorder=6,
		path_effects=[matplotlib.patheffects.withStroke(linewidth=1.6, foreground="white")])


def region_labels(ax, polygons, *, skip=(), fontsize=12):
	for index, polygon in enumerate(polygons):
		if index not in skip:
			label(ax, polygon, LETTERS[index], LABELS[index], fontsize=fontsize)


def route(ax, path, color=ACCENT, dashed=False, width=2.4, z=4):
	ax.plot([x for x, _ in path], [y for _, y in path], color=color, linewidth=width, linestyle=(0, (4, 2.5)) if dashed else "solid",
		solid_joinstyle="round", solid_capstyle="round", zorder=z)


def save(fig, name: str, args: argparse.Namespace) -> None:
	args.svg_dir.mkdir(parents=True, exist_ok=True)
	# No dates in the metadata, so rebuilding gives the same files.
	svg_path = args.svg_dir / f"{name}.svg"
	fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.04, metadata={"Date": None})
	svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
	if args.pdf_dir:
		args.pdf_dir.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.pdf_dir / f"{name}.pdf", transparent=True, bbox_inches="tight", pad_inches=0.04, metadata={"CreationDate": None})
	plt.close(fig)


def build(args: argparse.Namespace) -> dict:
	import matplotlib.patheffects  # noqa: F401  (used through matplotlib.patheffects in label)

	plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "siicusp34",
		"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
		"mathtext.cal": "Arial"})
	data = read_script_json(args.challenges)
	report = {}

	# Every geometric card uses this same five-region instance and the same axes.
	combined = data["combined_challenge"]
	polygons = combined["geometry"]["polygons"]
	hulls = [hull(polygon) for polygon in polygons]
	shapes = [Polygon(polygon) for polygon in polygons]
	full = closed_tour(args.solver, polygons, trace=True)
	greedy = initial_heuristic(full, polygons)
	if tour_length(greedy) <= full["upper_bound"] + GEOMETRY_TOLERANCE:
		raise ValueError("This illustration needs a native initial tour longer than the final tour")
	fig, ax = figure(polygons)
	for polygon in polygons:
		region(ax, polygon)
	route(ax, full["path"], color=INK, dashed=True, width=1.8)
	route(ax, greedy)
	region_labels(ax, polygons)
	save(fig, "method-01", args)
	report["01"] = {"source": "native incumbent/heuristic trace", "greedy": tour_length(greedy), "optimum": full["upper_bound"]}

	# 02: the node (A, B, C) as hulls; its tour touches D by chance and misses E.
	node = [0, 1, 2]
	partial = closed_tour(args.solver, [hulls[index] for index in node])
	line = LineString(partial["path"])
	if line.distance(shapes[4]) > GEOMETRY_TOLERANCE or line.distance(shapes[3]) <= GEOMETRY_TOLERANCE:
		raise ValueError("The partial node should touch D and miss E")
	fig, ax = figure(polygons)
	for index in node:
		region(ax, hulls[index], fill="#edf1f4", edge="#526c79", dashed=True, width=1.6, z=0)
	for index, polygon in enumerate(polygons):
		if index in node:
			region(ax, polygon)
		elif index == 4:
			region(ax, polygon, fill=INCIDENTAL, edge="#a87b22")
		else:
			region(ax, polygon, fill=UNVISITED, edge=MUTED, dashed=True)
	route(ax, partial["path"])
	# Mark hull contacts explicitly, including the contacts outside the original regions.
	ax.scatter(*zip(*partial["path"][:-1]), s=14, color=ACCENT, edgecolors="white", linewidths=0.45, zorder=5)
	region_labels(ax, polygons)
	save(fig, "method-02", args)
	report["02"] = {"length": partial["upper_bound"], "targets": "convex hulls of A, B, C",
		"original_distances": {LETTERS[i]: line.distance(shape) for i, shape in enumerate(shapes)}}

	# 04: the optimal tour through the five regions themselves, a feasible tour.
	validate_tour(full["path"], polygons)
	fig, ax = figure(polygons)
	for polygon in polygons:
		region(ax, polygon)
	route(ax, full["path"])
	region_labels(ax, polygons)
	save(fig, "method-04", args)
	report["04"] = {"length": full["upper_bound"]}

	# 06: the tour through the hulls misses some regions; a missed region splits into its pieces.
	relaxed = closed_tour(args.solver, hulls)
	line = LineString(relaxed["path"])
	missed = [index for index, shape in enumerate(shapes) if line.distance(shape) > GEOMETRY_TOLERANCE]
	if [LETTERS[index] for index in missed] != ["B", "C", "E"] or len(combined["pieces"][1]) != 2:
		raise ValueError("The refinement illustration needs missed regions B, C, E and two pieces of B")
	refined = missed[0]
	fig, ax = figure(polygons, size=(4.0, 3.4), branches=True)
	for index, polygon in enumerate(polygons):
		region(ax, hulls[index], fill="none", edge=MUTED, dashed=True, width=1.1, z=0)
		if index == refined:
			for piece_index, piece in enumerate(combined["pieces"][index]):
				region(ax, piece, fill=("#f8c89e", "#ffe7bf")[piece_index % 2], edge="#8b4d28", width=1.6, z=2)
		else:
			region(ax, polygon, fill=MISSED if index in missed else REGION, edge="#b0473c" if index in missed else TEAL)
	route(ax, relaxed["path"])
	region_labels(ax, polygons, skip=(refined,), fontsize=14)
	for piece_index, piece in enumerate(combined["pieces"][refined]):
		label(ax, piece, rf"$\mathrm{{{LETTERS[refined]}}}_{{{piece_index + 1}}}$", fontsize=14)
	branch = fig.add_axes([0.13, 0.02, 0.74, 0.22])
	branch.set(xlim=(0, 10), ylim=(0, 3.4))
	branch.axis("off")
	for x in (2.3, 7.7):
		branch.annotate("", xy=(x, 0.9), xytext=(5, 2.5), arrowprops={"arrowstyle": "-|>", "color": ACCENT, "linewidth": 1.5})
	branch.text(5, 2.8, "Hull of B", ha="center", va="center", fontsize=13.5, color=INK)
	for x, text in ((2.3, r"$\mathrm{B}_1$"), (7.7, r"$\mathrm{B}_2$")):
		branch.text(x, 0.55, text, ha="center", va="center", fontsize=16, color=INK,
			bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff1df", "edgecolor": "#8b4d28"})
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
		ax.add_patch(FancyBboxPatch((x - 1.55, y - 0.55), 3.1, 1.1, boxstyle="round,pad=0.05,rounding_size=0.25", facecolor="#eef4f1", edgecolor=edge, linewidth=1.5, zorder=2))
		ax.text(x, y, text, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
	for x in (2.0, 6.0, 10.0):
		# Meet the padded borders: parent bottom 4.8 - .55 - .05; child top 1.4 + .55 + .05.
		ax.plot([6.0, x], [4.2, 2.0], color=ACCENT, linewidth=1.6, solid_capstyle="butt", zorder=1)
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
