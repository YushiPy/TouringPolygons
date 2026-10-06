"""Prototype: the paper title as a TPP instance (one region per glyph component).

Glyph outlines come from Arial Bold; holes (a, e, o, g, P...) are joined to the
outside by a thin slit so each region is a simple polygon, as the solver requires.
Needs fontTools and shapely (any venv). Writes title-instance.json, title-art.tex.

    python3 make_title_instance.py --solver ../../../../../.build/unordered/tpp
"""
import argparse, json, math, subprocess
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import affinity

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
LINES = ["Algoritmos Exatos", "Para o Problema", "de Visita", "de Polígonos"]
HERE = Path(__file__).parent
SIMPLIFY = 0.003   # em; smaller = smoother curves, more vertices for the solver


class Flatten(BasePen):
    def __init__(self, glyphset, steps=16):
        super().__init__(glyphset)
        self.contours, self.cur, self.steps = [], [], steps

    def _moveTo(self, p): self.cur = [p]
    def _lineTo(self, p): self.cur.append(p)
    def _curveToOne(self, a, b, c):
        p0 = self.cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps; u = 1 - t
            self.cur.append((u**3*p0[0] + 3*u*u*t*a[0] + 3*u*t*t*b[0] + t**3*c[0],
                             u**3*p0[1] + 3*u*u*t*a[1] + 3*u*t*t*b[1] + t**3*c[1]))
    def _qCurveToOne(self, a, b):
        p0 = self.cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps; u = 1 - t
            self.cur.append((u*u*p0[0] + 2*u*t*a[0] + t*t*b[0], u*u*p0[1] + 2*u*t*a[1] + t*t*b[1]))
    def _closePath(self):
        if len(self.cur) > 2: self.contours.append(self.cur)
        self.cur = []
    _endPath = _closePath


def glyph_shape(font, name):
    gs = font.getGlyphSet(); pen = Flatten(gs); gs[name].draw(pen)
    shape = None
    for c in pen.contours:                      # even-odd fill: holes by xor
        p = Polygon(c).buffer(0)
        shape = p if shape is None else shape.symmetric_difference(p)
    return shape


def slit_open(poly, width):
    """Join every hole to the outside with a thin straight slit -> simple polygon."""
    for hole in list(poly.interiors):
        hp = Polygon(hole)
        c = hp.representative_point()
        best = None
        minx, miny, maxx, maxy = poly.bounds
        far = max(maxx - minx, maxy - miny) * 2
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            line = box(0, -width / 2, far, width / 2)
            if dx == -1: line = affinity.rotate(line, 180, origin=(0, 0))
            if dy == 1: line = affinity.rotate(line, 90, origin=(0, 0))
            if dy == -1: line = affinity.rotate(line, -90, origin=(0, 0))
            line = affinity.translate(line, c.x, c.y)
            cut = poly.difference(line)
            if isinstance(cut, Polygon) and not cut.interiors:
                loss = poly.area - cut.area
                if best is None or loss < best[0]: best = (loss, cut)
        if best is None: raise ValueError("no slit works")
        poly = best[1]
    return poly


def build():
    font = TTFont(FONT); cmap = font.getBestCmap(); hmtx = font["hmtx"]
    upm = font["head"].unitsPerEm
    comps = []   # (line, char, polygon in font units shifted)
    y = 0
    widths = [sum(hmtx[cmap[ord(c)]][0] for c in t) for t in LINES]
    for li, text in enumerate(LINES):
        x = (max(widths) - widths[li]) / 2
        for ch in text:
            name = cmap[ord(ch)]
            if ch != " ":
                shape = glyph_shape(font, name)
                if shape is not None and not shape.is_empty:
                    shape = affinity.translate(shape, x, y)
                    parts = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
                    for part in parts:
                        if part.area < 1: continue
                        comps.append((li, ch, part))
            x += hmtx[name][0]
        y -= upm * 1.18
    polys = []
    for li, ch, part in comps:
        if part.interiors: part = slit_open(part, upm * 0.022)
        part = part.simplify(upm * SIMPLIFY, preserve_topology=True)
        if not part.is_valid or part.interiors: raise ValueError(f"bad glyph {ch}")
        polys.append((li, ch, orient(part, 1.0)))
    return polys, upm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--solver", required=True)
    ap.add_argument("--seconds", type=int, default=120)
    a = ap.parse_args()
    polys, upm = build()
    s = 1 / upm * 1.0                       # 1 em = 1 unit
    pts = [[(x * s, y * s) for x, y in list(p.exterior.coords)[:-1]] for _, _, p in polys]
    minx = min(x for p in pts for x, _ in p); maxx = max(x for p in pts for x, _ in p)
    miny = min(y for p in pts for _, y in p); maxy = max(y for p in pts for _, y in p)
    start = (minx - 0.45, (miny + maxy) / 2)
    lines = [f"{start[0]:.9f} {start[1]:.9f} {start[0]:.9f} {start[1]:.9f} {len(pts)} 20000000 {a.seconds}"]
    for p in pts: lines.append(f"{len(p)} " + " ".join(f"{x:.9f} {y:.9f}" for x, y in p))
    r = subprocess.run([a.solver], input=("\n".join(lines) + "\n").encode(), capture_output=True, check=True)
    res = json.loads(r.stdout)
    print({k: res.get(k) for k in ("termination", "exact", "lower_bound", "upper_bound", "solve_seconds", "calls")},
          "regions", len(pts), "vertices", sum(map(len, pts)))
    json.dump({"start": start, "polygons": pts, "chars": [c for _, c, _ in polys],
               "path": res["path"], "order": res["order"], "termination": res["termination"],
               "lower_bound": res["lower_bound"], "upper_bound": res["upper_bound"],
               "bounds": [minx, miny, maxx, maxy]}, open(HERE / "title-instance.json", "w"))
    write_tex(pts, res["path"], start, (minx, miny, maxx, maxy), res["order"])


LETTER_RAMP = ["2B5FA8", "1F7A73", "5C9E31"]          # first -> last visited region
ROUTE_RAMP = ["E8B100", "E26A12", "B3261E"]           # start -> end of the route


def ramp(stops, t):
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2); f = t - i
    a, b = (tuple(int(h[k:k + 2], 16) for k in (0, 2, 4)) for h in stops[i:i + 2])
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def rgb(c):
    return "{rgb,255:red,%d;green,%d;blue,%d}" % c


def write_tex(pts, path, start, bounds, order):
    """TikZ scope in em units: letters coloured by visit order, route by progress."""
    minx, miny, maxx, maxy = bounds
    rank = {region: k for k, region in enumerate(order)}
    out = [f"% generated by make_title_instance.py; extent {maxx - start[0]:.4f} x {maxy - miny:.4f} em",
           f"\\def\\artwidth{{{maxx - start[0] + 0.04:.4f}}}\\def\\artheight{{{maxy - miny:.4f}}}",
           "\\begin{scope}[shift={(%.4f,%.4f)}]" % (-start[0] + 0.02, -miny)]
    for i, p in enumerate(pts):
        c = ramp(LETTER_RAMP, rank[i] / max(len(order) - 1, 1))
        out.append(f"\\fill[color={rgb(c)}] " + " -- ".join(f"({x:.4f},{y:.4f})" for x, y in p) + " -- cycle;")
    # route: cut into short pieces, each coloured by the length travelled so far
    pieces, total = [], 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(1, math.ceil(math.dist((x0, y0), (x1, y1)) / 0.08))
        for k in range(n):
            a = (x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
            b = (x0 + (x1 - x0) * (k + 1) / n, y0 + (y1 - y0) * (k + 1) / n)
            pieces.append((a, b, total)); total += math.dist(a, b)
    out.append("\\draw[white,line width=3.4pt,line cap=round,line join=round] " + " ".join(
        f"({a[0]:.4f},{a[1]:.4f})--({b[0]:.4f},{b[1]:.4f})" for a, b, _ in pieces) + ";")
    for a, b, t0 in pieces:
        c = ramp(ROUTE_RAMP, (t0 + math.dist(a, b) / 2) / total)
        out.append(f"\\draw[color={rgb(c)},line width=1.9pt,line cap=round] ({a[0]:.4f},{a[1]:.4f})--({b[0]:.4f},{b[1]:.4f});")
    out.append(f"\\fill[white,draw=Ink,line width=1.2pt] ({start[0]:.4f},{start[1]:.4f}) circle (0.045);")
    out.append("\\end{scope}")
    (HERE / "title-art.tex").write_text("\n".join(out) + "\n")
    (HERE / "title-art-body.tex").write_text("\n".join(out[2:]) + "\n")


if __name__ == "__main__":
    main()
