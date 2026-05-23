
from collections.abc import Callable, Sequence

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]
type Matrix = NDArray[np.float64]


def quadratic_model(n: Vector, k: Vector) -> Matrix:
	return np.column_stack((np.ones_like(n), n + k, (n + k) ** 2))

def polylog_model(n: Vector, k: Vector) -> Matrix:
	return np.column_stack((np.ones_like(n), n + k, n * k, n * k * np.log(n / k)))

def linear_model(n: Vector, k: Vector) -> Matrix:
	return np.column_stack((np.ones_like(n), n + k, n * k))



ALGORITHMS = {
	"Linear Search": ("#003f5c", quadratic_model),
	"Binary Search": ("#7a4f99", polylog_model),
	"Binary Search (lazy)": ("#ef527a", polylog_model),
	"TAMC": ("#ffa600", linear_model),
}

def regression(x: Vector, y: Vector, model: Matrix) -> Vector:
	"""
	Finds the optimal values for the parameters such that the mean squared error between the predicted values and the actual values is minimized.
	The 

	---
	Parameters:
	- `x`: A vector of real numbers representing the input data.
	- `y`: A vector of real numbers representing the target values corresponding to each input in `x`.
	- `model`: A matrix where each row corresponds to a vector of features derived from the input `x` according to a specific model (e.g., linear, polynomial, logarithmic). The number of columns in the model matrix should match the number of parameters to be estimated.
	---
	Returns:
	- A vector of optimal parameter values that minimize the mean squared error between the predicted values and the actual values.
	"""

	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)

	x_matrix = np.array(model)
	y_vector = np.array(y, dtype=np.float64)

	# Solve the normal equations: (X^T X) w = X^T y
	X_transpose = x_matrix.T
	normal_matrix = X_transpose @ x_matrix
	normal_vector = X_transpose @ y_vector
	optimal_parameters = np.linalg.solve(normal_matrix, normal_vector).astype(np.float64)

	return optimal_parameters

def plot_vs_k(df: pd.DataFrame, ax: Axes | None = None) -> None:

	algs = [alg for alg in ALGORITHMS if alg in df.columns]

	k = np.array(df["k"], dtype=np.float64)
	n = np.array(df["n"], dtype=np.float64)
	timings = [np.array(df[alg], dtype=np.float64) for alg in algs]

	min_len = len(k)

	for x in [k, n] + timings:
		if np.isnan(x).any():
			min_len = min(min_len, np.where(np.isnan(x))[0][0])
	
	k = k[:min_len]
	n = n[:min_len]
	timings = [timing[:min_len] for timing in timings]

	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(10, 6))
		fig.tight_layout()

	for i in range(len(algs)):

		alg_name = algs[i]
		timing = timings[i]
		color = ALGORITHMS[alg_name][0]
		base_model = ALGORITHMS[alg_name][1]
		model = base_model(n, k)

		try:
	
			fit = regression(k, timing, model)

			with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
				predicted = fit.T @ model.T
	
		except np.linalg.LinAlgError:
			predicted = timing
		
		ax.plot(k, timing, label=alg_name, color=color)
		
		ax.plot(k, predicted, color="black", alpha=0.5)
		ax.plot(k, predicted, linestyle="dashed", color=color)

	m = round(np.mean(n / k))

	ax.set_xlabel("k")
	ax.set_ylabel("Time (seconds)")
	ax.set_title(f"Execution Time vs Number of Polygons (k) for m={m}")
	ax.legend()
	ax.grid(True)

def plot_vs_m(df: pd.DataFrame, ax: Axes | None = None) -> None:

	algs = [alg for alg in ALGORITHMS if alg in df.columns]

	k = np.array(df["k"], dtype=np.float64)
	n = np.array(df["n"], dtype=np.float64)
	m = n / k
	
	timings = [np.array(df[alg], dtype=np.float64) for alg in algs]

	min_len = len(k)

	for x in [k, n] + timings:
		if np.isnan(x).any():
			min_len = min(min_len, np.where(np.isnan(x))[0][0])

	k = k[:min_len]
	n = n[:min_len]
	m = m[:min_len]
	timings = [timing[:min_len] for timing in timings]

	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(10, 6))
		fig.tight_layout()

	for i in range(len(algs)):

		alg_name = algs[i]
		timing = timings[i]
		color = ALGORITHMS[alg_name][0]
		base_model = ALGORITHMS[alg_name][1]
		model = base_model(n, k)

		try:

			fit = regression(m, timing, model)

			with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
				predicted = fit.T @ model.T

		except np.linalg.LinAlgError:
			predicted = timing

		
		ax.plot(m, timing, label=alg_name, color=color)
		
		ax.plot(m, predicted, color="black", alpha=0.5)
		ax.plot(m, predicted, linestyle="dashed", color=color)

	average_k = round(np.mean(k))

	ax.set_xlabel("m")
	ax.set_ylabel("Time (seconds)")
	ax.set_title(f"Execution Time vs Vertices per Polygon (m) for k={average_k}")
	ax.legend()
	ax.grid(True)


df = pd.read_csv("benchmark_results_k_100_m_1_to_40000.csv")
# df = pd.read_csv("benchmark_results_k_1_to_3000_m_100.csv")

plot_vs_m(df)
