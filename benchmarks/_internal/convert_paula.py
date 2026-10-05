"""Import the 235 Paula instances as endpoint TPP with a bbox-center depot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from convert_instances import ConvertedCase, bounding_box, write_binary_cases

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
	ROOT / "third_party/paula-tspn/tspn-instances/tspn-instances/npol-le-100"
)


def parse_dat(text: str) -> list[list[tuple[float, float]]]:
	tokens = iter(text.split())
	try:
		count, maximum = int(next(tokens)), int(next(tokens))
		if count <= 0 or maximum <= 0:
			raise ValueError("Invalid region count or maximum vertex count")
		regions = []
		for _ in range(count):
			size = int(next(tokens))
			if size < 1 or size > maximum:
				raise ValueError("Invalid vertex count")
			region = [(float(next(tokens)), float(next(tokens))) for _ in range(size)]
			if not all(math.isfinite(v) for p in region for v in p):
				raise ValueError("Nonfinite coordinate")
			regions.append(region)
		if next(tokens, None) is not None:
			raise ValueError("Unexpected trailing data")
		return regions
	except StopIteration as error:
		raise ValueError("Truncated Paula instance") from error


def convert_file(path: Path, source: Path) -> ConvertedCase:
	raw = path.read_bytes()
	polygons = parse_dat(raw.decode("utf-8"))
	xmin, ymin, xmax, ymax = bounding_box(polygons)
	depot = (xmin / 2 + xmax / 2, ymin / 2 + ymax / 2)
	return ConvertedCase(
		path.stem,
		depot,
		depot,
		polygons,
		{
			"source": str(path.relative_to(source)),
			"source_sha256": hashlib.sha256(raw).hexdigest(),
			"bbox": [xmin, ymin, xmax, ymax],
			"points": sum(len(p) == 1 for p in polygons),
			"segments": sum(len(p) == 2 for p in polygons),
		},
	)


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("campaign", nargs="?", default="paula-center")
	parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
	args = parser.parse_args(argv)
	from tpp import resolve_campaign

	campaign = resolve_campaign(args.campaign)
	source = args.source.resolve()
	paths = sorted(source.rglob("*.dat"))
	if len(paths) != 235:
		raise SystemExit(
			f"Expected exactly 235 npol-le-100 instances; found {len(paths)} in {source}"
		)
	cases = [convert_file(p, source) for p in paths]
	if any(len(c.polygons) > 100 for c in cases) or len({c.name for c in cases}) != 235:
		raise SystemExit("Unexpected corpus: duplicate names or more than 100 regions")
	suite = campaign / "inputs/paula-center.bin"
	manifest_path = campaign / "paula-manifest.json"
	if suite.exists() or manifest_path.exists():
		raise SystemExit(f"Import already exists in {campaign}; choose a new campaign")
	write_binary_cases(cases, suite)
	manifest = {
		"schema": "paula-center-v1",
		"formulation": "free-order endpoint TPP; start=target=global bbox center",
		"source_root": str(source),
		"binary": str(suite),
		"redistribution": "local third-party material; keep this campaign ignored",
		"count": len(cases),
		"cases": [
			{
				"name": c.name,
				"start": c.start,
				"target": c.target,
				"polygons": c.polygons,
				"meta": c.meta,
			}
			for c in cases
		],
	}
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
	print(f"Imported {len(cases)} instances into {suite}\nManifest: {manifest_path}")
	return 0
