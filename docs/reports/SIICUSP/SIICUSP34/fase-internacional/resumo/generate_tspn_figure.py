"""Figure 1 of the international abstract: instance, decomposition, relaxation, tour.

The convex pieces come from our port of Greene's algorithm
(packages/optimal-convex-partition), so every cut is a diagonal between
polygon vertices. Both tours come from tpp-unordered --cycle: the optimal tour
of the instance and the optimal tour of the convex hulls, a lower bound that
misses a polygon. Run with a Python that has matplotlib and shapely, e.g.

	apps/benchmark-dashboard/.venv/bin/python generate_tspn_figure.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as Patch
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT / "benchmarks"))

from _internal.native_build import ensure_tool  # noqa: E402

LIBRARY = ROOT / "packages/optimal-convex-partition/cpp"
PARTITION_SOURCE = ROOT / "apps/benchmark-dashboard/scripts/partition_cli.cpp"
PARTITION = ROOT / ".build/dashboard-partition/partition"

Point = tuple[float, float]

# Eight thin non-convex regions (counterclockwise) around a central T, plus a
# thin L that the tour crosses. The optimal tour detours to the T, so it is not
# convex and its order is not the angular one; the wide U on top opens toward
# the tour, so the hull relaxation touches the hull across openings and misses
# polygons. Crossing the L's interior counts as visiting it.
POLYGONS: list[list[Point]] = [
	[(0.92, 10.17), (0.92, 7.37), (1.42, 7.37), (1.42, 9.67), (3.72, 9.67), (3.72, 10.17)],
	[(8.6, 11.99), (4.4, 11.99), (4.4, 9.39), (4.9, 9.39), (4.9, 11.49), (8.1, 11.49), (8.1, 9.39), (8.6, 9.39)],
	[(10.09, 8.15), (11.69, 8.15), (11.69, 9.15), (12.69, 9.15), (12.69, 9.65), (11.19, 9.65), (11.19, 8.65), (10.09, 8.65)],
	[(12.88, 4.31), (12.88, 7.11), (12.38, 7.11), (12.38, 4.81), (10.08, 4.81), (10.08, 4.31)],
	[(7.5, 0.48), (10.5, 0.48), (10.5, 3.08), (10, 3.08), (10, 0.98), (8, 0.98), (8, 3.08), (7.5, 3.08)],
	[(1.75, 2), (2.75, 2), (2.75, 0), (3.25, 0), (3.25, 2), (4.25, 2), (4.25, 2.5), (1.75, 2.5)],
	[(1.54, 4.38), (1.54, 5.98), (0.54, 5.98), (0.54, 6.98), (0.04, 6.98), (0.04, 5.48), (1.04, 5.48), (1.04, 4.38)],
	[(4.75, 6), (5.75, 6), (5.75, 4), (6.25, 4), (6.25, 6), (7.25, 6), (7.25, 6.5), (4.75, 6.5)],
	[(5.3, 7.7), (6.6, 7.7), (6.6, 8.1), (5.7, 8.1), (5.7, 9.2), (5.3, 9.2)],
]

FILL = "#f2b84b"
FILL_LIGHT = "#f8d999"
MUTED = "#e6e9ec"
INK = "#2f3a45"
MISS = "#d9534f"
TOUR = "#1f6feb"


def counterclockwise(points: list[Point]) -> list[Point]:
	return points if Polygon(points).exterior.is_ccw else points[::-1]


def decompose(polygon: list[Point]) -> list[list[Point]]:
	sources = [PARTITION_SOURCE, LIBRARY / "src/optimal_convex_partition.cpp"]
	if not PARTITION.exists() or PARTITION.stat().st_mtime < max(p.stat().st_mtime for p in sources):
		PARTITION.parent.mkdir(parents=True, exist_ok=True)
		subprocess.run(
			["c++", "-std=c++20", "-O3", "-I", str(LIBRARY / "include"), *map(str, sources), "-o", str(PARTITION)],
			check=True,
		)
	text = f"{len(polygon)} " + " ".join(f"{x} {y}" for x, y in polygon) + "\n"
	output = subprocess.run([str(PARTITION)], input=text, capture_output=True, text=True, check=True).stdout
	return [[(x, y) for x, y in piece] for piece in json.loads(output)]


def solve_cycle(polygons: list[list[Point]]) -> dict:
	text = f"0 0 0 0 {len(polygons)} 100000000 60\n" + "".join(
		f"{len(p)} " + " ".join(f"{x:.17g} {y:.17g}" for x, y in p) + "\n" for p in polygons
	)
	result = json.loads(
		subprocess.run([str(ensure_tool("tpp-unordered")), "--cycle"], input=text, capture_output=True, text=True, check=True).stdout
	)
	if result["termination"] != "optimal":
		raise RuntimeError(f"tpp-unordered stopped with {result['termination']}")
	return result


# A miss must be visible at print size, about 3% of the panel width.
VISIBLE_MISS = 0.3


def distances(path: list[Point], polygons: list[list[Point]]) -> list[float]:
	line = LineString(path)
	return [line.distance(Polygon(polygon)) for polygon in polygons]


def missed(path: list[Point], polygons: list[list[Point]]) -> list[int]:
	return [i for i, distance in enumerate(distances(path, polygons)) if distance > 1e-7]


def crossed(path: list[Point], polygons: list[list[Point]]) -> list[int]:
	line = LineString(path)
	return [i for i, polygon in enumerate(polygons) if line.intersection(Polygon(polygon).buffer(-1e-6)).length > 1e-3]


def setup(ax: Axes, label: str) -> None:
	ax.set_xlim(-0.3, 13.2)
	ax.set_ylim(-0.3, 12.3)
	ax.set_aspect("equal")
	ax.axis("off")
	ax.text(-0.3, 12.35, label, fontsize=9, weight="bold", color=INK, va="bottom")


def outline(ax: Axes, polygon: list[Point], **style) -> None:
	ax.add_patch(Patch(polygon, closed=True, joinstyle="miter", **({"facecolor": "none", "edgecolor": INK, "linewidth": 0.8} | style)))


def tour(ax: Axes, path: list[Point], **style) -> None:
	xs, ys = zip(*path)
	ax.plot(xs, ys, **({"color": TOUR, "linewidth": 1.6, "solid_capstyle": "round", "zorder": 4} | style))


def main() -> None:
	polygons = [counterclockwise(p) for p in POLYGONS]
	pieces = [decompose(p) for p in polygons]
	hulls = [counterclockwise(list(Polygon(p).convex_hull.exterior.coords)[:-1]) for p in polygons]
	optimal = solve_cycle(polygons)
	relaxed = solve_cycle(hulls)
	relaxed_misses = missed(relaxed["path"], polygons)
	gaps = [distances(relaxed["path"], polygons)[i] for i in relaxed_misses]
	crossings = crossed(optimal["path"], polygons)
	if not relaxed_misses or min(gaps) < VISIBLE_MISS or missed(optimal["path"], polygons) or not crossings:
		raise RuntimeError(f"the instance no longer illustrates the relaxation (misses by {gaps}, crossings {crossings})")
	print(
		f"optimal {optimal['upper_bound']:.4f} (relative gap {optimal['final_relative_gap']:.1e}) order {optimal['order']}; "
		f"hull bound {relaxed['upper_bound']:.4f} misses {relaxed_misses} by {[round(g, 3) for g in gaps]}; crosses {crossings}; "
		f"pieces {[len(p) for p in pieces]}"
	)

	plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
	fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.7), dpi=250)

	setup(axes[0], "(a)")
	for polygon in polygons:
		outline(axes[0], polygon, facecolor=FILL, alpha=0.85)

	setup(axes[1], "(b)")
	for polygon, parts in zip(polygons, pieces, strict=True):
		for index, part in enumerate(parts):
			outline(axes[1], part, facecolor=FILL if index % 2 == 0 else FILL_LIGHT, edgecolor=INK, linewidth=0.4)
		outline(axes[1], polygon)

	setup(axes[2], "(c)")
	for index, (polygon, hull) in enumerate(zip(polygons, hulls, strict=True)):
		outline(axes[2], hull, facecolor=MUTED, edgecolor="#b5bec6", linewidth=0.6)
		color = MISS if index in relaxed_misses else FILL
		outline(axes[2], polygon, facecolor=color, alpha=0.85)
	tour(axes[2], relaxed["path"])

	setup(axes[3], "(d)")
	line = LineString(optimal["path"])
	for polygon, parts in zip(polygons, pieces, strict=True):
		for part in parts:
			touched = line.distance(Polygon(part)) <= 1e-7
			outline(axes[3], part, facecolor=FILL if touched else MUTED, edgecolor=INK, linewidth=0.4)
		outline(axes[3], polygon)
	tour(axes[3], optimal["path"])

	plt.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.06)
	output = HERE / "assets/tspn-figure.pdf"
	output.parent.mkdir(exist_ok=True)
	fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
	main()
