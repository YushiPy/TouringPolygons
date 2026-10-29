"""Figure 2 of the international abstract: per-instance speedup over Fekete et al.

Reads benchmarks/results-saved/fekete-reference-armijo-2026-10-17/per-case.csv
and compares configuration B (original code of Fekete et al., CHR root, one
thread) with C (ours, one thread) on the instances both solve. The time of an
instance is the median of the three 300 s runs when all three close, otherwise
the 900 s run, as in the summary of that campaign. Histogram in log scale,
stacked by instance source, with the arithmetic mean and the median, like
slide 4 of the first stage. Run with a Python that has matplotlib, e.g.

	apps/benchmark-dashboard/.venv/bin/python generate_speedup_figure.py
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SOURCE = ROOT / "benchmarks/results-saved/fekete-reference-armijo-2026-10-17/per-case.csv"

INK = "#2f3a45"
MUTED = "#8a96a3"
RULE = "#dfe3e7"
MEAN = "#d9534f"
FAMILIES = [
	("OSM", "City maps", "#1f6feb"),
	("random", "Random", "#f2b84b"),
	("tessellation", "Tessellations", INK),
]
LOW, STEP, BINS = 0.25, 0.25, 14  # bins in log10(speedup): 10^0.25 .. 10^3.75


def solved_times(rows: list[dict], configuration: str, root: str) -> dict[str, tuple[float, str]]:
	runs: dict[str, dict[str, dict]] = defaultdict(dict)
	for row in rows:
		if row["configuration"] == configuration and row["root"] == root:
			runs[row["case"]][row["time_limit"]] = row
	times = {}
	for case, by_limit in runs.items():
		short, long = by_limit["300"], by_limit.get("900")
		if short["solved"] == "3":
			times[case] = (float(short["seconds_median"]), short["source"])
		elif long is not None and long["solved"] == "1":
			times[case] = (float(long["seconds_median"]), short["source"])
	return times


def main() -> None:
	rows = list(csv.DictReader(SOURCE.open()))
	fekete = solved_times(rows, "B", "CHR")
	ours = solved_times(rows, "C", "ours")
	if set(fekete) - set(ours):
		raise RuntimeError("an instance is solved only by Fekete et al.")
	common = sorted(fekete)
	speedups = {case: fekete[case][0] / ours[case][0] for case in common}
	values = list(speedups.values())
	mean, median = statistics.mean(values), statistics.median(values)
	geometric = math.exp(statistics.mean(map(math.log, values)))
	if min(values) < 10**LOW or max(values) >= 10 ** (LOW + STEP * BINS):
		raise RuntimeError(f"speedups from {min(values):.2f} to {max(values):.0f} fall outside the bins")
	print(
		f"{len(common)} common instances; mean {mean:.1f}x, median {median:.1f}x, geometric {geometric:.1f}x, "
		f"min {min(values):.2f}x, max {max(values):.0f}x; only ours {len(set(ours) - set(fekete))}"
	)

	plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
	fig, ax = plt.subplots(figsize=(3.0, 1.7), dpi=300)
	centers = [LOW + STEP * (i + 0.5) for i in range(BINS)]
	bottom = [0] * BINS
	for source, label, color in FAMILIES:
		counts = [0] * BINS
		for case in common:
			if fekete[case][1] == source:
				counts[int((math.log10(speedups[case]) - LOW) // STEP)] += 1
		n = sum(counts)
		ax.bar(centers, counts, width=STEP, bottom=bottom, color=color, edgecolor="white", linewidth=0.4, label=f"{label} ({n})", zorder=2)
		bottom = [b + c for b, c in zip(bottom, counts)]

	top = max(bottom) * 1.22
	for value, color, text, align in ((median, INK, f"median {median:.1f}×", "right"), (mean, MEAN, f"mean {mean:.0f}×", "left")):
		x = math.log10(value)
		ax.plot([x, x], [0, top * 0.86], color=color, linewidth=1.1, linestyle=(0, (3, 1.5)) if color == INK else "-", zorder=3)
		ax.text(x + (0.04 if align == "left" else -0.04), top * 0.87, text, color=color, ha=align, va="bottom", fontsize=7, weight="bold")

	ax.set_xlim(0, 4)
	ax.set_ylim(0, top)
	ax.set_xticks([0, 1, 2, 3, 4], ["1×", "10×", "100×", "1000×", "10000×"])
	ax.set_xlabel("Speedup = time of [4] / our time (one thread each)", color=INK)
	ax.set_ylabel("Instances", color=INK)
	ax.yaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
	ax.spines[["top", "right"]].set_visible(False)
	ax.tick_params(length=2, width=0.6)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(
		handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
		fontsize=6.5, handlelength=1, handleheight=0.8, columnspacing=1.0, borderaxespad=0.2,
	)

	output = HERE / "assets/speedup-figure.pdf"
	output.parent.mkdir(exist_ok=True)
	fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
	main()
