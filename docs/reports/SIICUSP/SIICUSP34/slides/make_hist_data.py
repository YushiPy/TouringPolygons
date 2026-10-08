"""Gera hist-data.tex (barras do slide 4) a partir da comparação na dantzig,
benchmarks/results-saved/free-order-dantzig-2026-10-06/per-case.csv.
Histograma do speedup Fekete/nosso em escala log10, empilhado por tipo de instância.
Só entram os casos que ambos os solvers fecharam (550 de 558)."""
import csv
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[5]
src = root / "benchmarks/results-saved/free-order-dantzig-2026-10-06/per-case.csv"
rows = [r for r in csv.DictReader(src.open()) if r["fekete_status"] == "optimal"]
LO, STEP, N = 0.5, 0.25, 12            # bins em log10(speedup): 10^0.5 .. 10^3.5
out = []
for fam in ("OSM", "random", "tessellation"):
    counts = [0] * N
    for r in rows:
        if r["source"] == fam:
            b = int((math.log10(float(r["fekete_over_ours"])) - LO) // STEP)
            counts[min(max(b, 0), N - 1)] += 1
    out.append(f"% {fam}: {sum(counts)} casos")
    out.append(r"\def\hist" + fam.replace("tessellation", "voronoi") + "{%")
    out += [f"({LO + STEP * (i + 0.5):.3f},{c})%" for i, c in enumerate(counts)]
    out.append("}")
ratios = [float(r["fekete_over_ours"]) for r in rows]
med = statistics.median(ratios)
out.append(f"\\def\\histmedian{{{math.log10(med):.4f}}}")
out.append(f"\\def\\histmediantext{{{med:.1f}}}".replace(".", "{,}"))
mean = statistics.mean(ratios)
out.append(f"\\def\\histmean{{{math.log10(mean):.4f}}}  % média aritmética {mean:.1f}x")
out.append(f"\\def\\histmeantext{{{mean:.1f}}}\\def\\histcount{{{len(rows)}}}".replace(".", "{,}"))
out.append("\\def\\histmintext{" + f"{min(ratios):.1f}".replace(".", "{,}") + "}\\def\\histmaxtext{" + f"{max(ratios):,.0f}".replace(",", ".") + "}")
for fam, tag in (('OSM', 'OSM'), ('random', 'RANDOM'), ('tessellation', 'VORONOI')):
    out.append(f"\\def\\histn{tag}{{{sum(r['source'] == fam for r in rows)}}}")
for fam in ("OSM", "random", "tessellation"):
    g = [float(r["fekete_over_ours"]) for r in rows if r["source"] == fam]
    out.append(f"% mediana {fam}: {statistics.median(g):.1f}x")
print(len(rows), "mediana", med, "mínimo", min(ratios), "máximo", max(ratios))
(Path(__file__).parent / "hist-data.tex").write_text("\n".join(out) + "\n")
