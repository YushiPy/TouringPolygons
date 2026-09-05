#!/usr/bin/env python3
"""Independent exhaustive order/piece SOCP baseline for unions of two rectangles.

Requires gurobipy (licensed) and shapely. The production solver requires neither.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
from pathlib import Path

import gurobipy as gp
from shapely.geometry import LineString, Polygon

Point = tuple[float, float]


def baseline(start: Point, target: Point, rectangles: list[list[tuple[float, float, float, float]]], env: gp.Env) -> tuple[float, float]:
	lower = upper = math.inf
	for order in itertools.permutations(range(len(rectangles))):
		for selection in itertools.product(*(rectangles[i] for i in order)):
			with gp.Model(env=env) as model:
				model.Params.Threads = 1
				model.Params.BarQCPConvTol = 1e-8
				model.Params.FeasibilityTol = 1e-9
				x = [start[0]]
				y = [start[1]]
				for xmin, ymin, xmax, ymax in selection:
					x.append(model.addVar(lb=xmin, ub=xmax))
					y.append(model.addVar(lb=ymin, ub=ymax))
				x.append(target[0])
				y.append(target[1])
				cost = []
				for i in range(len(x) - 1):
					dx = model.addVar(lb=-gp.GRB.INFINITY)
					dy = model.addVar(lb=-gp.GRB.INFINITY)
					d = model.addVar(lb=0)
					model.addConstr(dx == x[i + 1] - x[i])
					model.addConstr(dy == y[i + 1] - y[i])
					model.addQConstr(dx * dx + dy * dy <= d * d)
					cost.append(d)
				model.setObjective(gp.quicksum(cost))
				model.optimize()
				if model.Status != gp.GRB.OPTIMAL:
					model.Params.BarQCPConvTol = 1e-6
					model.Params.NumericFocus = 3
					model.reset()
					model.optimize()
				if model.Status != gp.GRB.OPTIMAL:
					raise RuntimeError(f'Baseline status {model.Status}')
				upper = min(upper, model.ObjVal)
				lower = min(lower, model.ObjBound)
	return lower, upper


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--solver', type=Path, default=Path('.build/unordered/tpp'))
	parser.add_argument('--cases', type=int, default=24)
	args = parser.parse_args()
	rng = random.Random(342026)
	with gp.Env(empty=True) as env:
		env.setParam('OutputFlag', 0)
		env.start()
		decompositions = 0
		for case in range(args.cases):
			polygons = []
			rectangles = []
			for i in range(1 if case < 8 else 1 + case % 4):
				x = (1.2 if case % 2 else 4) * i + rng.uniform(-1, 1)
				y = rng.uniform(-2, 3)
				p = [(x, y), (x + 2, y), (x + 2, y + .6), (x + .6, y + .6), (x + .6, y + 2), (x, y + 2)]
				polygons.append(p[::-1] if case % 3 else p)
				rectangles.append([(x, y, x + 2, y + .6), (x, y, x + .6, y + 2)])
			start = (-3., -2.)
			target = start if case % 5 == 0 else (13., 8.)
			if case < 8:
				x, y = rectangles[0][0][:2]
				start = (x + 1, y + 1.2)
				target = start if case % 2 else (x + 1.8, y + 1.3)
			data = ' '.join(map(str, (*start, *target, len(polygons), 1000000, 30))) + '\n'
			for p in polygons:
				data += str(len(p)) + ' ' + ' '.join(str(v) for point in p for v in point) + '\n'
			process = subprocess.run([str(args.solver.resolve())], input=data, text=True, capture_output=True, check=True, timeout=60)
			result = json.loads(process.stdout)
			lo, hi = baseline(start, target, rectangles, env)
			line = LineString(result['path'])
			assert all(line.distance(Polygon(p)) <= 1e-7 for p in polygons), (case, 'coverage')
			assert result['path'][0] == list(start) and result['path'][-1] == list(target)
			assert result['exact'], (case, result)
			assert abs(result['upper_bound'] - hi) <= 1e-5 * (1 + hi), (case, result, lo, hi)
			assert result['lower_bound'] <= hi + 1e-6 * (1 + hi)
			decompositions += result['decomposition_branches']
			print(json.dumps({'case': case, 'ours': result['upper_bound'], 'gurobi': hi, 'decomposition_branches': result['decomposition_branches']}), flush=True)
		assert decompositions > 0, 'Expected explicit nonconvex decomposition branching.'
		print(f'Passed {args.cases} independent SOCP comparisons; {decompositions} decomposition branches.')


if __name__ == '__main__':
	main()
