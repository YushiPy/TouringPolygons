"""Paper instances with every region replaced by its convex hull plus a small disk.

The many-vertices comparison (results-saved ``fekete-smoothed-mac-2026-10-21`` and
``tspn-smoothed-vertex-scaling-2026-10-21``). ``tpp.py smoothed-instances`` writes one archive in
the upstream format: ``NAME-VARIANT.json`` with WKT polygons, ``meta`` and Fekete's own order
annotations. ``tpp.py fekete-paper|fekete-single-thread|ours-single-thread --instances ARCHIVE``
then runs it in the reference protocol.

Each region of a source instance becomes the convex hull of points on the boundary of
hull + disk. Coordinates are rounded to the grid 2^-30 and the hull is exact on that grid. The
disk's radius is 1% of the instance's median region diameter. Variant ``m`` spreads m points
over the arcs at the hull's vertices, in proportion to their angles; variant ``hull`` keeps the
hull itself. The annotation runs in the Python of the upstream build (``tpp.py fekete-setup``),
as ``smoothed_instances_child.py``. ``--no-annotations`` skips it, for archives only our solver
reads.

The annotated archive holds the polygons as Fekete's model writes them back (Shapely WKT with 16
significant digits, so a few coordinates near zero move by about one ulp), and both solvers read
them from there. The m = 65,536 archive of 2026-10-22 was written with ``--no-annotations``: only
our solver ran it, since Fekete's already leaves most m = 16,384 instances without a tour in 900 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import fekete_reference as reference
import workspace
from convert_instances import convert_json_case

SOURCE_INSTANCES = ("bangalore_n060_seed4413", "random_mixed_square_n060_seed60018", "vienna_n050_seed2410",
	"istanbul_n060_seed10845", "warsaw_n060_seed403", "tokyo_n060_seed53")
DEFAULT_VARIANTS = ("hull", "256", "2048", "16384")
GRID = 2.0 ** 30
RADIUS_FRACTION = 0.01
CHILD = Path(__file__).with_name("smoothed_instances_child.py")


def hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
	"""Exact monotone chain on integers: counterclockwise, no collinear vertices."""
	points = sorted(set(points))
	if len(points) <= 2:
		return points
	def cross(o, a, b):
		return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
	lower, upper = [], []
	for p in points:
		while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
			lower.pop()
		lower.append(p)
	for p in reversed(points):
		while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
			upper.pop()
		upper.append(p)
	return lower[:-1] + upper[:-1]


def grid_hull(points) -> list[tuple[int, int]]:
	return hull([(round(x * GRID), round(y * GRID)) for x, y in points])


def smoothed(region, radius: float, vertices: int | None) -> list[tuple[float, float]]:
	"""The region's hull (vertices None) or hull + disk with about this many vertices, on the grid."""
	base = grid_hull(region)
	if vertices is None:
		return [(x / GRID, y / GRID) for x, y in base]
	corners = [(x / GRID, y / GRID) for x, y in base]
	def normal(a, b):
		return math.atan2(-(b[0] - a[0]), b[1] - a[1])
	arcs = []
	for i, corner in enumerate(corners):
		start = normal(corners[i - 1], corner)
		sweep = (normal(corner, corners[(i + 1) % len(corners)]) - start) % (2 * math.pi)
		arcs.append((corner, start, sweep))
	total = sum(sweep for _, _, sweep in arcs)
	points = []
	for (cx, cy), start, sweep in arcs:
		count = max(2, round(vertices * sweep / total))
		for j in range(count):
			angle = start + sweep * j / (count - 1)
			points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
	return [(x / GRID, y / GRID) for x, y in grid_hull(points)]


def source_instances(names) -> dict[str, tuple[list, dict]]:
	"""Name -> (polygons, meta) from the paper's archive."""
	archive = reference.SOURCE / reference.INSTANCE_ARCHIVE
	if not archive.is_file():
		reference.fail(f"{archive} não existe; rode antes tpp.py fekete-setup")
	found = {}
	with zipfile.ZipFile(archive) as file:
		for member in file.namelist():
			name = Path(member).stem
			if member.endswith(".json") and name in names:
				payload = file.read(member)
				found[name] = (convert_json_case(member, payload).polygons, json.loads(payload).get("meta", {}))
	missing = sorted(set(names) - set(found))
	if missing:
		reference.fail(f"instâncias ausentes do arquivo do artigo: {', '.join(missing)}")
	return found


def wkt(polygon) -> str:
	ring = list(polygon) + [polygon[0]]
	return "POLYGON((" + ", ".join(f"{x!r} {y!r}" for x, y in ring) + "))"


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py smoothed-instances", description=(
		"Gera as instâncias suavizadas (fecho convexo de cada região mais um disco, com m vértices) num "
		"arquivo no formato do Fekete, anotado pelo código dele; depois, rode as configurações com --instances."))
	parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
		help="hull e/ou números de vértices por região, separados por vírgula (padrão: %(default)s)")
	parser.add_argument("--instances", default=",".join(SOURCE_INSTANCES),
		help="instâncias do artigo de origem, separadas por vírgula (padrão: as seis da comparação no Mac)")
	parser.add_argument("--output", type=Path,
		help="arquivo .zip de saída (padrão: experiments/smoothed-instances/smoothed.zip no workspace)")
	parser.add_argument("--no-annotations", action="store_true",
		help="não anota a ordem no fecho (o arquivo serve só para o nosso solver)")
	args = parser.parse_args(argv)
	variants = [v.strip() for v in args.variants.split(",") if v.strip()]
	if not variants or any(v != "hull" and not (v.isdigit() and int(v) >= 8) for v in variants):
		parser.error("--variants: 'hull' ou números de vértices (>= 8)")
	names = [n.strip() for n in args.instances.split(",") if n.strip()]
	output = (args.output or workspace.experiments_dir() / "smoothed-instances" / "smoothed.zip").resolve()
	if output.exists():
		reference.say(f"{output} já existe; nada a fazer (apague-o para gerar de novo).")
		return 0
	output.parent.mkdir(parents=True, exist_ok=True)
	sources = source_instances(names)
	manifest = {"source_archive_sha256": reference.INSTANCE_ARCHIVE_SHA256, "grid": "2^-30",
		"radius": f"{RADIUS_FRACTION} x median region diameter", "variants": variants, "instances": {},
		"annotated": not args.no_annotations}
	temporary = output.with_name(output.name + ".tmp")
	with tempfile.TemporaryDirectory(prefix="smoothed-") as scratch:
		lines = Path(scratch) / "instances.jsonl"
		with lines.open("w") as file:
			for name in names:
				polygons, meta = sources[name]
				diameters = sorted(max(math.dist(p, q) for p in region for q in region) for region in polygons)
				radius = RADIUS_FRACTION * diameters[len(diameters) // 2]
				for variant in variants:
					regions = [smoothed(region, radius, None if variant == "hull" else int(variant)) for region in polygons]
					label = f"{name}-{'hull' if variant == 'hull' else 'm' + variant}"
					meta_out = {**meta, "smoothing": {"variant": variant, "source_instance": name, "radius": radius,
						"regions": "convex hull plus a disk", "grid": "2^-30"}}
					file.write(json.dumps({"name": label, "polygons": [wkt(r) for r in regions], "meta": meta_out}) + "\n")
					manifest["instances"][label] = {"regions": len(regions), "vertices": sum(len(r) for r in regions)}
					reference.say(f"{label}: {len(regions)} regiões, {manifest['instances'][label]['vertices']} vértices")
		if args.no_annotations:
			with lines.open() as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as out:
				for line in source:
					item = json.loads(line)
					out.writestr(item["name"] + ".json", json.dumps({"polygons": item["polygons"], "meta": item["meta"],
						"annotations": {}}))
		else:
			python = reference.VENV / "bin/python"
			if not python.is_file():
				reference.fail("o solver de Fekete não foi preparado; rode antes tpp.py fekete-setup")
			reference.say("Anotando com add_order_annotations (código do Fekete)...")
			done = subprocess.run([str(python), str(CHILD), str(lines), str(temporary)],
				env=reference.child_environment(None), cwd=scratch, check=False)
			if done.returncode:
				temporary.unlink(missing_ok=True)
				reference.fail("a anotação falhou")
	temporary.replace(output)
	manifest["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
	output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
	reference.say(f"Arquivo: {output}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
