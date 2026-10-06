"""Gera scatter-data.tex (coordenadas do gráfico do slide 3) a partir da
campanha preservada em benchmarks/results-saved/fekete-comparison."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[5]
src = root / "benchmarks/results-saved/fekete-comparison/analysis/instance-classification.csv"
rows = [r for r in csv.DictReader(src.open()) if r["fekete_completed"] == "True"]
out = []
wins = {}
for fam in ("OSM", "random", "tessellation"):
    sel = [r for r in rows if r["source_type"] == fam]
    w = sum(float(r["ours_seconds"]) < float(r["fekete_seconds"]) for r in sel)
    wins[fam] = (w, len(sel))
    out.append(f"% {fam}: {w}/{len(sel)}")
    out.append(r"\def\data" + fam.replace("tessellation", "voronoi") + "{%")
    for r in sel:
        out.append(f"({float(r['fekete_seconds']):.6g},{float(r['ours_seconds']):.6g})%")
    out.append("}")
print(wins, len(rows))
(Path(__file__).parent / "scatter-data.tex").write_text("\n".join(out) + "\n")
