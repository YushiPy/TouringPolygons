"""Anchor function F(x) of the fixed-order convex cycle on four squares turned 45 degrees.

F(x) is the shortest closed route that starts and ends at x, on the boundary of the
first region, and visits the other regions in order. This script samples F at 1000
boundary points with our fixed-source TPP solver and, when gurobipy is importable,
with an independent SOCP, then draws funcao-ancoragem.png. The same two cases are
checked by rotated_squares() in packages/convex-tpp/cpp/src/main-cycle_tests.cpp.

    python3 benchmarks/tpp.py build tpp-visualizer-solve
    .build/fekete-upstream/venv/bin/python docs/reports/funcao-ancoragem/generate_anchor_figure.py
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOLVER = ROOT / ".build/tools/bin/tpp-visualizer-solve"
SAMPLES = 1000

OURS, GUROBI, REGION, OTHER, MUTED = "#2563eb", "#ea580c", "#dbeafe", "#9ca3af", "#6b7280"


def diamond(x, y, r):
	return [(x + r, y), (x, y + r), (x - r, y), (x, y - r)]


# Optima and optimal cycles certified exactly by rotated_squares() in the C++ tests.
CASES = {
	"A": {
		"title": "A — simétrico (sugestão de 16/10)",
		"regions": [diamond(5, 5, 2), diamond(-5, 5, 2), diamond(-5, -5, 2), diamond(5, -5, 2)],
		"optimum": 32.0,
		"cycle": [(4, 4), (-4, 4), (-4, -4), (4, -4)],
		"local": (6.0, 6.0),
		"extra": [(5, 3), (-5, 3), (-5, -3), (5, -3)],
	},
	"B": {
		"title": "B — assimétrico",
		"regions": [diamond(4, -3, 4), diamond(0, 2, 2), diamond(-6, -3, 4), diamond(-4, 6, 3)],
		"optimum": math.sqrt(65) + math.sqrt(85),
		"cycle": [(18 / 13, -21 / 13), (0, 0), (-40 / 9, -5 / 9), (-4, 3)],
		"local": (5.31, -5.69),
		"extra": None,
	},
}


def anchored_path(x, regions):
	"""Our fixed-source TPP with s = t = x through `regions` in order."""
	lines = [f"{x[0]!r} {x[1]!r}", f"{x[0]!r} {x[1]!r}", f"{len(regions)} 1000000 30"]
	lines += [f"{len(p)} " + " ".join(f"{a!r} {b!r}" for a, b in p) for p in regions]
	words = subprocess.run([str(SOLVER)], input="\n".join(lines) + "\n", capture_output=True, text=True,
		check=True).stdout.split()
	if words[0] != "OK":
		raise RuntimeError(" ".join(words))
	count = int(words[4])
	return [(float(words[5 + 2 * i]), float(words[6 + 2 * i])) for i in range(count)]


def length(path):
	return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def gurobi_anchored():
	"""An independent SOCP for F, or None without gurobipy."""
	try:
		import gurobipy as gp
	except ImportError:
		return None
	env = gp.Env(params={"OutputFlag": 0})

	def solve(x, regions):
		model = gp.Model(env=env)
		model.Params.Threads = 1
		model.Params.BarQCPConvTol = 1e-12
		model.Params.FeasibilityTol = 1e-9
		points = [(x[0], x[1])]
		for polygon in regions:
			q = model.addVars(2, lb=-gp.GRB.INFINITY)
			for i, (x0, y0) in enumerate(polygon):
				x1, y1 = polygon[(i + 1) % len(polygon)]
				model.addConstr((y1 - y0) * q[0] + (x0 - x1) * q[1] <= (y1 - y0) * x0 + (x0 - x1) * y0)
			points.append((q[0], q[1]))
		points.append((x[0], x[1]))
		links = []
		for a, b in zip(points, points[1:]):
			d = model.addVars(2, lb=-gp.GRB.INFINITY)
			t = model.addVar()
			model.addConstr(d[0] == b[0] - a[0])
			model.addConstr(d[1] == b[1] - a[1])
			model.addQConstr(d[0] * d[0] + d[1] * d[1] <= t * t)
			links.append(t)
		model.setObjective(gp.quicksum(links))
		model.optimize()
		return model.ObjVal

	return solve


def boundary_samples(polygon, samples):
	"""Points at equal arc length along the closed boundary, from the first vertex."""
	edges = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
	perimeter = sum(math.dist(a, b) for a, b in edges)
	out = []
	for j in range(samples):
		s = rest = perimeter * j / samples
		for a, b in edges:
			if rest <= math.dist(a, b):
				t = rest / math.dist(a, b)
				out.append((s, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))))
				break
			rest -= math.dist(a, b)
	return out, perimeter


def main():
	reference = gurobi_anchored()
	figure, axes = plt.subplots(2, 2, figsize=(11, 8.8), gridspec_kw={"width_ratios": [1, 1.45]})
	for row, (key, case) in enumerate(CASES.items()):
		regions = case["regions"]
		rest = regions[1:]
		samples, perimeter = boundary_samples(regions[0], SAMPLES)
		values = [length(anchored_path(x, rest)) for _, x in samples]
		optimum = case["optimum"]
		cycle = case["cycle"] + case["cycle"][:1]
		assert min(values) >= optimum - 1e-9, "a sampled anchor beats the certified optimum"
		if reference is not None:
			checked = [reference(x, rest) for _, x in samples]
			difference = max(abs(a - b) for a, b in zip(values, checked))
			print(f"{key}: optimum {optimum:.9f}, sampled minimum {min(values):.9f}; ours vs Gurobi at "
				f"{len(samples)} anchors: max |diff| {difference:.2e}")
		else:
			checked = None
			print(f"{key}: optimum {optimum:.9f} (gurobipy unavailable: no independent check)")

		geometry = axes[row][0]
		for i, polygon in enumerate(regions):
			closed = polygon + polygon[:1]
			geometry.fill(*zip(*closed), color=REGION if i == 0 else "#f3f4f6", zorder=1)
			geometry.plot(*zip(*closed), color=OURS if i == 0 else OTHER, lw=1.2, zorder=2)
			cx = sum(p[0] for p in polygon) / len(polygon)
			cy = sum(p[1] for p in polygon) / len(polygon)
			geometry.text(cx, cy, str(i), ha="center", va="center", fontsize=9, color=MUTED, zorder=3)
		if case["extra"]:
			rectangle = case["extra"] + case["extra"][:1]
			geometry.plot(*zip(*rectangle), color=MUTED, lw=1, ls="--", zorder=3, label="também ótimo (vértices)")
		geometry.plot(*zip(*cycle), color=OURS, lw=2, zorder=4, label="ciclo ótimo")
		if case["local"]:
			local_path = anchored_path(case["local"], rest)
			geometry.plot(*zip(*local_path), color=GUROBI, lw=1.2, ls=":", zorder=4, label="mínimo local não global")
			geometry.plot(*case["local"], "o", color=GUROBI, ms=5, zorder=5)
		cx = sum(p[0] for p in regions[0]) / len(regions[0])
		cy = sum(p[1] for p in regions[0]) / len(regions[0])
		for k, (vx, vy) in enumerate(regions[0]):
			geometry.annotate(f"v{k}", (vx, vy), xytext=(6 * (vx - cx) / math.dist((vx, vy), (cx, cy)),
				6 * (vy - cy) / math.dist((vx, vy), (cx, cy))), textcoords="offset points", ha="center",
				va="center", fontsize=8, color=OURS)
		geometry.set_aspect("equal")
		geometry.set_title(case["title"], fontsize=11, loc="left")
		geometry.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
		geometry.tick_params(labelsize=8)

		plot = axes[row][1]
		s = [a / perimeter for a, _ in samples]
		plot.plot(s, [v - optimum for v in values], color=OURS, lw=2, label="nosso TPP ancorado")
		if checked:
			step = 10
			plot.plot(s[::step], [v - optimum for v in checked[::step]], "o", color=GUROBI, ms=2.5,
				label="Gurobi (SOCP), 1 em cada 10")
		corner = 0.0
		for k, (a, b) in enumerate(zip(regions[0], regions[0][1:] + regions[0][:1])):
			edge = corner / perimeter
			corner += math.dist(a, b)
			plot.axvline(edge, color=OTHER, lw=0.8, zorder=0)
			plot.text(edge, 1.0, f"v{k}", transform=plot.get_xaxis_transform(), ha="center", va="bottom",
				fontsize=8, color=MUTED)
		plot.set_xlim(0, 1)
		plot.set_ylim(bottom=-0.05 * max(v - optimum for v in values))
		plot.set_xlabel("posição da âncora x na borda da região 0 (fração do perímetro, a partir de v0)", fontsize=9)
		plot.set_ylabel("F(x) − ótimo", fontsize=9)
		plot.grid(axis="y", color="#e5e7eb", lw=0.6)
		plot.tick_params(labelsize=8)
		if row == len(CASES) - 1:
			plot.legend(fontsize=8, frameon=False, loc="upper center")
		for side in ("top", "right"):
			plot.spines[side].set_visible(False)
	figure.tight_layout()
	output = HERE / "funcao-ancoragem.png"
	figure.savefig(output, dpi=160)
	print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
	main()
