"""Gera scatter-data.tex (coordenadas do gráfico do slide 4) a partir da comparação
na dantzig, benchmarks/results-saved/free-order-dantzig-2026-10-06/per-case.csv.
Só entram os casos que ambos os solvers fecharam (550 de 558)."""
import csv
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[5]
src = root / "benchmarks/results-saved/free-order-dantzig-2026-10-06/per-case.csv"
rows = [r for r in csv.DictReader(src.open()) if r["fekete_status"] == "optimal"]
ratios = [float(r["fekete_over_ours"]) for r in rows]
out = []
for fam in ("OSM", "random", "tessellation"):
    sel = [r for r in rows if r["source"] == fam]
    out.append(f"% {fam}: {len(sel)} casos")
    out.append(r"\def\data" + fam.replace("tessellation", "voronoi") + "{%")
    for r in sel:
        out.append(f"({float(r['fekete_seconds']):.6g},{max(float(r['ours_seconds']), 1e-4):.6g})%")
    out.append("}")
print(len(rows), "mediana", statistics.median(ratios),
      "geométrica", math.exp(statistics.mean(map(math.log, ratios))),
      "mais rápidos", sum(x > 1 for x in ratios))
(Path(__file__).parent / "scatter-data.tex").write_text("\n".join(out) + "\n")
