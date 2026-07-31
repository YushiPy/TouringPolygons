import gurobipy as gp
from gurobipy import GRB
import numpy as np

def vertices_to_halfplanes(verts: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
	n = len(verts)
	A = np.zeros((n, 2))
	b = np.zeros(n)
	for i in range(n):
		x0, y0 = verts[i]
		x1, y1 = verts[(i + 1) % n]
		dx, dy = x1 - x0, y1 - y0
		A[i] = [dy, -dx]
		b[i] = dy * x0 - dx * y0
	return A, b

def shortest_path(
	start: tuple[float, float],
	end: tuple[float, float],
	polygons: list[list[tuple[float, float]]],
) -> list[np.ndarray]:
	"""
	Returns the list of k points (one per polygon) of the shortest path
	start -> P_1 -> ... -> P_k -> end.
	"""
	k = len(polygons)
	halfplanes = [vertices_to_halfplanes(poly) for poly in polygons]

	s = np.array(start)
	t = np.array(end)

	m = gp.Model()
	m.Params.OutputFlag = 0

	p = [m.addMVar(2, lb=-GRB.INFINITY, name=f"p_{i}") for i in range(k)]
	t_vars = [m.addVar(lb=0, name=f"t_{i}") for i in range(k + 1)]

	for i, (A, b) in enumerate(halfplanes):
		m.addConstr(A @ p[i] <= b)

	# start -> p_0
	d = m.addMVar(2, lb=-GRB.INFINITY, name="d_start")
	m.addConstr(d == p[0] - s)
	m.addGenConstrNorm(t_vars[0], d, which=2)

	# p_i -> p_{i+1}
	for i in range(k - 1):
		d = m.addMVar(2, lb=-GRB.INFINITY, name=f"d_{i}")
		m.addConstr(d == p[i + 1] - p[i])
		m.addGenConstrNorm(t_vars[i + 1], d, which=2)

	# p_{k-1} -> end
	d = m.addMVar(2, lb=-GRB.INFINITY, name="d_end")
	m.addConstr(d == t - p[k - 1])
	m.addGenConstrNorm(t_vars[k], d, which=2)

	m.setObjective(gp.quicksum(t_vars), GRB.MINIMIZE)
	m.optimize()

	if m.Status != GRB.OPTIMAL:
		raise RuntimeError(f"Model status: {m.Status}")

	return [p[i].X for i in range(k)]

if __name__ == "__main__":

	start = (-1, -1)
	end = (-1, 0)
	polygons = [
		[(0, 0), (1, 0), (1, 1), (0, 1)],
		[(2, -1), (3, -1), (3, 0), (2, 0)],
		[(2, 1), (3, 1), (3, 2), (2, 2)],
		[(4, -1), (5, -1), (5, 0), (4, 0)],
		[(4, 1), (5, 1), (5, 2), (4, 2)]
	]

	path = shortest_path(start, end, polygons)
	path = [start] + path + [end]

	import matplotlib.pyplot as plt

	plt.plot(*zip(*path), marker="o")

	for poly in polygons:
		plt.fill(*zip(*poly), alpha=0.5)

	plt.gca().set_aspect("equal", adjustable="box")