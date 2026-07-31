"""
Usage:
	python gen_instances.py <osm.pbf> [n] [seed]

	On first run, parses the .pbf and saves a cache next to it.
	Subsequent runs load from cache instantly.

Download SP extract:
	curl -L -o sp.osm.pbf https://download.geofabrik.de/south-america/brazil/sudeste-latest.osm.pbf

Crop to a smaller area (requires osmium-tool: brew install osmium-tool):
	osmium extract --bbox=-46.826,-24.008,-46.365,-23.357 sudeste-latest.osm.pbf -o sp-city.osm.pbf
"""
import json
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import osmium
from shapely.geometry import Polygon


class BuildingHandler(osmium.SimpleHandler):
	def __init__(self) -> None:
		super().__init__()
		self.polygons: list[Polygon] = []

	def area(self, a: osmium.osm.Area) -> None:
		if "building" not in a.tags:
			return
		try:
			outer = next(iter(a.outer_rings()))
			coords = [(n.lon, n.lat) for n in outer]
			if len(coords) < 4:
				return
			poly = Polygon(coords)
			if poly.is_valid and not poly.is_empty:
				self.polygons.append(poly)
		except Exception:
			pass


def load_polygons(pbf_path: str) -> list[Polygon]:
	cache = Path(pbf_path).with_suffix(".cache.pkl")
	if cache.exists():
		print(f"Loading from cache {cache}...", flush=True)
		with open(cache, "rb") as f:
			polygons = pickle.load(f)
		print(f"Loaded {len(polygons)} buildings.", flush=True)
		return polygons

	print(f"Parsing {pbf_path}...", flush=True)
	handler = BuildingHandler()
	handler.apply_file(pbf_path, locations=True)
	polygons = handler.polygons
	print(f"Found {len(polygons)} buildings. Saving cache...", flush=True)
	with open(cache, "wb") as f:
		pickle.dump(polygons, f)
	print(f"Cache saved to {cache}", flush=True)
	return polygons


def is_nonconvex(poly: Polygon, threshold: float = 0.95) -> bool:
	return (poly.area / poly.convex_hull.area) < threshold


def sample_polygons(polygons: list[Polygon], n: int, seed: int = 42) -> list[Polygon]:
	nonconvex = [p for p in polygons if is_nonconvex(p)]
	convex = [p for p in polygons if not is_nonconvex(p)]
	pct = 100 * len(nonconvex) / len(polygons) if polygons else 0
	print(f"{len(nonconvex)} non-convex ({pct:.1f}%), {len(convex)} convex.", flush=True)

	if len(polygons) < n:
		raise ValueError(f"Only {len(polygons)} polygons available, requested {n}.")

	rng = random.Random(seed)
	sampled = rng.sample(nonconvex, min(n, len(nonconvex)))
	if len(sampled) < n:
		sampled += rng.sample(convex, n - len(sampled))
	return sampled


def save_json(polygons: list[Polygon], path: str) -> None:
	data = [{"coordinates": list(p.exterior.coords)} for p in polygons]
	with open(path, "w") as f:
		json.dump(data, f, indent=2)
	print(f"Saved to {path}", flush=True)


def plot_polygons(polygons: list[Polygon], output_path: str) -> None:
	cols = 5
	rows = -(-len(polygons) // cols)
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
	axes = axes.flatten()

	for i, poly in enumerate(polygons):
		x, y = poly.exterior.xy
		axes[i].fill(x, y, alpha=0.5, fc="steelblue", ec="black")
		axes[i].set_aspect("equal")
		axes[i].axis("off")

	for j in range(len(polygons), len(axes)):
		axes[j].axis("off")

	fig.suptitle(f"{len(polygons)} building polygons", fontsize=14)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	print(f"Plot saved to {output_path}", flush=True)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(__doc__)
		sys.exit(1)

	pbf_path = sys.argv[1]
	n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
	seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

	polygons = load_polygons(pbf_path)
	sampled = sample_polygons(polygons, n, seed)
	save_json(sampled, "instances.json")
	plot_polygons(sampled, "instances.png")