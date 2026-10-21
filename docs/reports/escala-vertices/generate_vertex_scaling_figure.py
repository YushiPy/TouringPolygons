"""Time against vertices per region: our TSPN solver and Fekete et al.'s, on the smoothed corpus instances.

Reads the compact per-run CSV of benchmarks/results-saved/fekete-smoothed-mac-2026-10-21 and draws
escala-vertices.png: one line per instance and solver (Fekete: the better of the CHR and LEFP roots),
hollow markers for runs that did not close the 0.1% gap within the limit.

    .build/fekete-upstream/venv/bin/python docs/reports/escala-vertices/generate_vertex_scaling_figure.py
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "benchmarks/results-saved/fekete-smoothed-mac-2026-10-21/per-run.csv"
LIMIT = 300
OURS, FEKETE, MUTED = "#2563eb", "#ea580c", "#6b7280"
VARIANTS = ["hull", "m256", "m2048", "m16384"]

rows = list(csv.DictReader(open(DATA)))
by = defaultdict(dict)
for row in rows:
    by[(row["instance"], row["variant"])][row["solver"]] = row

def per_region(instance, variant):
    row = by[(instance, variant)]["ours"]
    return int(row["vertices"]) / int(row["regions"])

def fekete_best(instance, variant):
    """The better root: solved and faster, else the smaller final gap."""
    runs = [by[(instance, variant)].get(s) for s in ("fekete-CHR", "fekete-LEFP")]
    runs = [r for r in runs if r]
    solved = [r for r in runs if r["solved"] == "True"]
    if solved: return min(solved, key=lambda r: float(r["seconds"]))
    return min(runs, key=lambda r: float(r["gap"]) if r["gap"] else float("inf"))

figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=200)
instances = sorted({instance for instance, _ in by})
for instance in instances:
    xs = [per_region(instance, v) for v in VARIANTS]
    for solver, color, pick in (("ours", OURS, lambda i, v: by[(i, v)]["ours"]), ("fekete", FEKETE, fekete_best)):
        runs = [pick(instance, v) for v in VARIANTS]
        ys = [min(float(r["seconds"]), LIMIT) for r in runs]
        axis.plot(xs, ys, color=color, linewidth=1, alpha=0.55, zorder=2)
        for x, y, r in zip(xs, ys, runs):
            closed = r["solved"] == "True"
            axis.plot([x], [y], marker="o", markersize=4.5, color=color, zorder=3,
                      markerfacecolor=color if closed else "white", markeredgewidth=1.2)
axis.axhline(LIMIT, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
# Where Fekete stopped at the limit: the final gap range, or how many runs had no tour.
for variant in VARIANTS[1:]:
    runs = [fekete_best(instance, variant) for instance in instances]
    open_runs = [r for r in runs if r["solved"] != "True"]
    gaps = [100 * float(r["gap"]) for r in open_runs if r["gap"]]
    untoured = sum(1 for r in open_runs if not r["gap"])
    parts = [f"gap {min(gaps):.0f}–{max(gaps):.0f}%".replace(".", ",")] if gaps else []
    if untoured: parts.append(f"{untoured} de {len(open_runs)} sem tour")
    x = statistics.median(per_region(instance, variant) for instance in instances)
    axis.text(x, LIMIT * 0.62, "\n".join(parts), color=FEKETE, fontsize=7, ha="center", va="top")
axis.text(1.0, LIMIT * 1.12, "limite de 300 s", color=MUTED, fontsize=8, ha="right", transform=axis.get_yaxis_transform())
axis.set_xscale("log"); axis.set_yscale("log")
axis.set_xlabel("vértices por região (média da instância)")
axis.set_ylabel("tempo até fechar o gap de 0,1% (s)")
axis.set_title("TSPN em seis instâncias do corpus com regiões suavizadas (50 a 60 regiões, 1 thread)", fontsize=9)
handles = [plt.Line2D([], [], color=OURS, marker="o", label="nosso"),
           plt.Line2D([], [], color=FEKETE, marker="o", label="Fekete et al. (melhor raiz)"),
           plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", label="não fechou no limite")]
axis.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
axis.grid(True, which="major", color="#e5e7eb", linewidth=0.6)
for side in ("top", "right"): axis.spines[side].set_visible(False)
figure.tight_layout()
figure.savefig(HERE / "escala-vertices.png")
print("wrote", HERE / "escala-vertices.png")
