"""TikZ drawing of the USP demonstration (apps/siicusp34/data/usp-demo.js, read-only),
re-coloured with the deck palette. Writes usp-art-body.tex (unit: 1 = full height)."""
import json, math
from pathlib import Path
from palette import LETTER_RAMP, ROUTE_RAMP, ramp, rgb

HERE = Path(__file__).parent
SRC = HERE.parents[4] / "apps/siicusp34/data/usp-demo.js"
text = SRC.read_text()
demo = json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))
polys = demo["geometry"]["polygons"]; path = demo["path"]; start = demo["geometry"]["start"]
xs = [x for p in polys for x, _ in p] + [x for x, _ in path]
ys = [y for p in polys for _, y in p] + [y for _, y in path]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
k = 1 / (y1 - y0)                                # height = 1 unit; width = aspect
rank = {r: i for i, r in enumerate(demo["order"])}
f = lambda x, y: f"({(x - x0) * k:.4f},{(y - y0) * k:.4f})"
out = [f"\\def\\uspaspect{{{(x1 - x0) * k:.4f}}}\\def\\uspsx{{{(start[0] - x0) * k:.4f}}}\\def\\uspsy{{{(start[1] - y0) * k:.4f}}}"]
IME = {i for i, b in enumerate(demo["buildings"]) if b["qgis_id"].startswith("ime")}   # blocos A, B, C
GOLD_FILL, GOLD_LINE = (0xE9, 0xB4, 0x4C), (0x9A, 0x6B, 0x12)
ink = (0x14, 0x2D, 0x38)
for i, p in enumerate(polys):
    if i in IME:
        c, edge = GOLD_FILL, GOLD_LINE
    else:
        c = ramp(LETTER_RAMP, rank[i] / (len(polys) - 1))
        edge = tuple(round(v * 0.55 + k * 0.45) for v, k in zip(c, ink))   # same hue mixed with ink
    out.append(f"\\filldraw[fill={rgb(c)},draw={rgb(edge)},line width=0.7pt,line join=round] "
               + " -- ".join(f(x, y) for x, y in p) + " -- cycle;")
pieces, total = [], 0.0
for (ax, ay), (bx, by) in zip(path, path[1:]):
    n = max(1, math.ceil(math.hypot(bx - ax, by - ay) * k / 0.03))
    for j in range(n):
        a = (ax + (bx - ax) * j / n, ay + (by - ay) * j / n)
        b = (ax + (bx - ax) * (j + 1) / n, ay + (by - ay) * (j + 1) / n)
        pieces.append((a, b, total)); total += math.dist(a, b)
out.append("\\begin{scope}[transparency group,opacity=0.7]")
for a, b, t0 in pieces:
    c = ramp(ROUTE_RAMP, (t0 + math.dist(a, b) / 2) / total)
    out.append(f"\\draw[color={rgb(c)},line width=1.8pt] {f(*a)}--{f(*b)};")
out.append("\\end{scope}")
out.append(f"\\fill[white,draw=Ink,line width=1pt] {f(*start)} circle (0.012);")
(HERE / "usp-art-body.tex").write_text("\n".join(out) + "\n")
print("aspect", (x1 - x0) * k, "regions", len(polys), "length", demo["length"])
