"""Figure 2 of the international abstract: speedup over Fekete et al. by instance size.

Reads benchmarks/results-saved/fekete-reference-armijo-2026-10-30/per-case.csv
and compares configuration B (original code of Fekete et al., CHR root, one
thread) with F (ours, one thread, same stopping rule) on the instances both solve. The time of an
instance is the median of the three 300 s runs when all three close, otherwise
the 900 s run, as in the summary of that campaign. One point per instance
against its number of polygons (log scale), with the median and the
arithmetic mean over the instances of each size, rounded to a multiple of 5;
the lines skip the sizes with fewer than MIN_GROUP instances.
Run with a Python that has matplotlib, e.g.

	apps/benchmark-dashboard/.venv/bin/python generate_speedup_figure.py
"""

from __future__ import annotations

import csv
import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SOURCE = ROOT / "benchmarks/results-saved/fekete-reference-armijo-2026-10-30/per-case.csv"

INK = "#2f3a45"
POINT = "#8a96a3"
RULE = "#dfe3e7"
MEAN = "#d9534f"
LABELED = (5, 30, 60)  # sizes whose median is written on the plot
MIN_GROUP = 20  # sizes with fewer instances (a few tessellations) get no line point


def solved_times(rows: list[dict], configuration: str, root: str) -> dict[str, tuple[float, int]]:
	runs: dict[str, dict[str, dict]] = defaultdict(dict)
	for row in rows:
		if row["configuration"] == configuration and row["root"] == root:
			runs[row["case"]][row["time_limit"]] = row
	times = {}
	for case, by_limit in runs.items():
		short, long = by_limit["300"], by_limit.get("900")
		if short["solved"] == "3":
			times[case] = (float(short["seconds_median"]), int(short["polygons"]))
		elif long is not None and long["solved"] == "1":
			times[case] = (float(long["seconds_median"]), int(short["polygons"]))
	return times


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path, default=HERE / "assets/speedup-figure.pdf")
	parser.add_argument("--print", dest="print_palette", action="store_true", help="higher-contrast points for the printed poster")
	args = parser.parse_args()
	rows = list(csv.DictReader(SOURCE.open()))
	fekete = solved_times(rows, "B", "CHR")
	ours = solved_times(rows, "F", "ours")
	if set(fekete) - set(ours):
		raise RuntimeError("an instance is solved only by Fekete et al.")
	common = sorted(fekete)
	speedups = {case: fekete[case][0] / ours[case][0] for case in common}
	values = list(speedups.values())
	mean, median = statistics.mean(values), statistics.median(values)
	geometric = math.exp(statistics.mean(map(math.log, values)))
	groups: dict[int, list[float]] = defaultdict(list)
	for case in common:
		groups[5 * max(1, round(fekete[case][1] / 5))].append(speedups[case])
	sizes = [n for n in sorted(groups) if len(groups[n]) >= MIN_GROUP]
	medians = [statistics.median(groups[n]) for n in sizes]
	means = [statistics.mean(groups[n]) for n in sizes]
	print(
		f"{len(common)} common instances; mean {mean:.1f}x, median {median:.1f}x, geometric {geometric:.1f}x, "
		f"min {min(values):.2f}x, max {max(values):.0f}x; only ours {len(set(ours) - set(fekete))}"
	)
	for n, m, a in zip(sizes, medians, means):
		print(f"  {n:2d} polygons: {len(groups[n]):3d} instances, median {m:.1f}x, mean {a:.1f}x")
	print(f"  off the lines: {sum(len(v) for n, v in groups.items() if n not in sizes)} instances")

	plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 7, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
	fig, ax = plt.subplots(figsize=(3.0, 1.38), dpi=300)
	ax.scatter(
		[fekete[case][1] for case in common], [speedups[case] for case in common],
		s=2.5, color="#657587" if args.print_palette else POINT, alpha=0.85 if args.print_palette else 0.45,
		linewidths=0, zorder=2, label="instance",
	)
	ax.plot(sizes, means, color=MEAN, linewidth=1.2, linestyle=(0, (3, 1.5)), marker="o", markersize=2.2, zorder=3, label="arithmetic mean")
	ax.plot(sizes, medians, color=INK, linewidth=1.3, marker="o", markersize=2.2, zorder=4, label="median")
	for n, m in zip(sizes, medians):
		if n in LABELED:
			ax.text(n, m / 2.2, f"{m:.0f}×", ha="center", va="top", fontsize=6.5, color=INK, weight="bold", zorder=5)

	ax.set_yscale("log")
	ax.set_xlim(0, 63)
	ax.set_ylim(1, 10000)
	ax.set_xticks(range(0, 61, 10))
	ax.set_yticks([1, 10, 100, 1000, 10000], ["1×", "10×", "100×", "1000×", "10000×"])
	ax.minorticks_off()
	ax.set_xlabel("Polygons in the instance", color=INK)
	ax.set_ylabel("Speedup", color=INK)
	ax.yaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
	ax.spines[["top", "right"]].set_visible(False)
	ax.tick_params(length=2, width=0.6)
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1, frameon=False, fontsize=6.5, ncol=3, markerscale=1.5, handlelength=1.6, columnspacing=1.0)

	output = args.output
	output.parent.mkdir(exist_ok=True)
	fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
	main()
