"""Time against vertices per region: our TSPN solver and Fekete et al.'s, on the smoothed corpus instances.

Draws two figures from the compact CSVs in benchmarks/results-saved:
- escala-vertices.png, from fekete-smoothed-mac-2026-10-21 (Mac): ours and Fekete with 1 thread (the
  better of the CHR and LEFP roots), limit 300 s;
- escala-vertices-dantzig.png, from fekete-smoothed-dantzig-2026-10-22 (the reference machine and
  protocol): ours (C), Fekete with 16 threads (A) and with 1 thread (B, better root), limit 900 s.
One line per instance and solver; hollow markers did not close the 0.1% gap within the limit.

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
SAVED = ROOT / "benchmarks/results-saved"
OURS, FEKETE, FEKETE_ONE, MUTED = "#2563eb", "#ea580c", "#a21caf", "#6b7280"
VARIANTS = ["hull", "m256", "m2048", "m16384"]
INF = float("inf")


def mac_series():
    """(instance, variant) -> {series: (seconds, closed, gap)} and vertices per region, from the Mac CSV."""
    by = defaultdict(dict)
    for row in csv.DictReader(open(SAVED / "fekete-smoothed-mac-2026-10-21/per-run.csv")):
        by[(row["instance"], row["variant"])][row["solver"]] = row
    outcome = lambda r: (float(r["seconds"]), r["solved"] == "True", float(r["gap"]) if r["gap"] else None)
    series, per_region = {}, {}
    for key, runs in by.items():
        fekete = [outcome(runs[s]) for s in ("fekete-CHR", "fekete-LEFP") if s in runs]
        series[key] = {"ours": outcome(runs["ours"]), "fekete": best(fekete)}
        per_region[key] = int(runs["ours"]["vertices"]) / int(runs["ours"]["regions"])
    return series, per_region


def dantzig_series():
    """The same from the dantzig CSV: a Fekete run counts at 300 s if it closed there, else at 900 s."""
    by = defaultdict(dict)
    for row in csv.DictReader(open(SAVED / "fekete-smoothed-dantzig-2026-10-22/per-run.csv")):
        by[(row["instance"], row["variant"])][(row["configuration"], row["root"], int(row["time_limit"]))] = row
    def outcome(runs, configuration, root):
        short, long = runs.get((configuration, root, 300)), runs.get((configuration, root, 900))
        closed = lambda r: r is not None and int(r["solved"]) == int(r["repetitions"])
        final = short if closed(short) or long is None else long
        seconds = float(final["seconds_median"]) if final["seconds_median"] else float(final["time_limit"])
        return seconds, closed(final), float(final["gap"]) if final["gap"] else None
    series, per_region = {}, {}
    for key, runs in by.items():
        series[key] = {"ours": outcome(runs, "C", "ours"), "fekete-A": outcome(runs, "A", "CHR"),
			"fekete-B": best([outcome(runs, "B", "CHR"), outcome(runs, "B", "LEFP")])}
        ours = runs[("C", "ours", 300)]
        per_region[key] = int(ours["vertices"]) / int(ours["polygons"])
    return series, per_region


def best(outcomes):
    """The better root: closed and faster, else the smaller final gap (no tour counts as worst)."""
    closed = [o for o in outcomes if o[1]]
    if closed:
        return min(closed, key=lambda o: o[0])
    return min(outcomes, key=lambda o: o[2] if o[2] is not None else INF)


def notes_text(series, instances, variant, name, prefix):
    """Where a Fekete series stopped at the limit (the hollow markers): runs without a tour, final gaps."""
    open_runs = [series[(i, variant)][name] for i in instances if not series[(i, variant)][name][1]]
    if not open_runs:
        return ""
    gaps = [round(100 * gap) for _, _, gap in open_runs if gap is not None]
    parts = []
    untoured = sum(1 for _, _, gap in open_runs if gap is None)
    if untoured:
        parts.append(f"{untoured}/{len(instances)} sem tour")
    if gaps:
        parts.append(f"gap {min(gaps)}%" if min(gaps) == max(gaps) else f"gap {min(gaps)}–{max(gaps)}%")
    return prefix + ", ".join(parts)


def draw(series, per_region, styles, limit, title, output, notes=()):
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    instances = sorted({instance for instance, _ in series})
    for instance in instances:
        xs = [per_region[(instance, v)] for v in VARIANTS]
        for name, (color, dashes, _) in styles.items():
            runs = [series[(instance, v)][name] for v in VARIANTS]
            ys = [min(seconds, limit) for seconds, _, _ in runs]
            axis.plot(xs, ys, color=color, linewidth=1, alpha=0.55, zorder=2, linestyle=dashes)
            for x, y, (_, closed, _) in zip(xs, ys, runs):
                axis.plot([x], [y], marker="o", markersize=4.5, color=color, zorder=3,
					markerfacecolor=color if closed else "white", markeredgewidth=1.2)
    axis.axhline(limit, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    for variant in VARIANTS[1:]:
        x = statistics.median(per_region[(i, variant)] for i in instances)
        last = variant == VARIANTS[-1]
        for row, (name, prefix) in enumerate(notes):
            text = notes_text(series, instances, variant, name, prefix)
            if text:
                axis.text(x * (1.25 if last else 1), limit * 0.72 / (1.45 ** row), text, color=styles[name][0],
					fontsize=6.5, ha="right" if last else "center", va="top")
    axis.text(1.0, limit * 1.12, f"limite de {limit} s", color=MUTED, fontsize=8, ha="right",
		transform=axis.get_yaxis_transform())
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("vértices por região (média da instância)")
    axis.set_ylabel("tempo até fechar o gap de 0,1% (s)")
    axis.set_title(title, fontsize=9)
    handles = [plt.Line2D([], [], color=color, marker="o", linestyle=dashes, label=label)
		for color, dashes, label in styles.values()]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none",
		label="não fechou no limite"))
    axis.legend(handles=handles, fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.7))
    axis.grid(True, which="major", color="#e5e7eb", linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(HERE / output)
    print("wrote", HERE / output)


series, per_region = mac_series()
draw(series, per_region, {"ours": (OURS, "-", "nosso"), "fekete": (FEKETE, "-", "Fekete et al. (melhor raiz)")}, 300,
	"TSPN em seis instâncias do corpus com regiões suavizadas (50 a 60 regiões, 1 thread, Mac)", "escala-vertices.png",
	notes=[("fekete", "")])
series, per_region = dantzig_series()
draw(series, per_region, {"ours": (OURS, "-", "nosso (1 thread)"),
	"fekete-A": (FEKETE, "-", "Fekete et al., 16 threads (A)"),
	"fekete-B": (FEKETE_ONE, "--", "Fekete et al., 1 thread (B, melhor raiz)")}, 900,
	"TSPN em seis instâncias do corpus com regiões suavizadas (50 a 60 regiões, dantzig)", "escala-vertices-dantzig.png",
	notes=[("fekete-A", "A: "), ("fekete-B", "B: ")])
