from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "static/event/siicusp34.json"


@lru_cache(maxsize=1)
def event_data() -> dict:
	return json.loads(DATA_PATH.read_text())


def event_context() -> dict:
	data = event_data()
	row = next(row for row in data["rows"] if row["case"] == 2)
	points = [point for polygon in row["geometry"]["polygons"] for point in polygon] + row["path"]
	xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
	ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
	scale = min(760 / max(xmax - xmin, 1e-9), 400 / max(ymax - ymin, 1e-9))
	ox, oy = (840 - (xmax - xmin) * scale) / 2, (480 - (ymax - ymin) * scale) / 2

	def project(point: list[float]) -> tuple[float, float]:
		return ox + (point[0] - xmin) * scale, 480 - oy - (point[1] - ymin) * scale

	return {
		"data": data,
		"initial": row,
		"polygons": [" ".join(f"{x},{y}" for x, y in map(project, polygon)) for polygon in row["geometry"]["polygons"]],
		"path": " ".join(f"{x},{y}" for x, y in map(project, row["path"])),
		"start": project(row["geometry"]["start"]),
		"target": project(row["geometry"]["target"]),
	}


def inline_event_assets(html: str) -> str:
	static = DATA_PATH.parents[1]
	css = (static / "event.css").read_text()
	modules = [(static / name).read_text() for name in ("dom.js", "event-geometry.js", "event.js")]
	javascript = "\n".join(re.sub(r"^export ", "", re.sub(r"^import .*;\n", "", source, flags=re.M), flags=re.M) for source in modules)
	html = html.replace('<link rel="stylesheet" href="/static/event.css">', f"<style>{css}</style>")
	html = html.replace('<script type="module" src="/static/event.js"></script>', f'<script type="module">{javascript}</script>')
	encoded = base64.b64encode(DATA_PATH.read_bytes()).decode("ascii")
	return html.replace('href="/static/event/siicusp34.json"', f'href="data:application/json;base64,{encoded}"')
