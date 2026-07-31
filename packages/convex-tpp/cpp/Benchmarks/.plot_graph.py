
from collections.abc import Callable, Sequence
import re

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
	"Binary Search": ("#7a4f99", polylog_model, "Busca Binária"),
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

	m = round(np.mean(n / k))

	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(4, 4))
		fig.tight_layout()

	for i in range(len(algs)):

		alg_name = algs[i]
		timing = timings[i]
		color = ALGORITHMS[alg_name][0]
		
		try:
	
			model = quadratic_model(n, k)
			fit = regression(n, timing, model)

			print(f"Fit for {alg_name} (k={m}): {fit}")

			with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
				predicted = fit.T @ model.T
	
		except np.linalg.LinAlgError:
			predicted = timing
		
		error = np.mean((timing - predicted) ** 2)
		label = f"Modelagem Polylog (MSE: {error:.2e})"

		ax.plot(k, timing, label="Tempo Real", color="blue", alpha=0.5)
		ax.plot(k, predicted, color="black", alpha=0.8, label=label, linestyle="dashed", linewidth=2)

		

	m = round(np.mean(n / k))

	ax.set_xlabel("Número de polígonos (k)")
	ax.set_ylabel("Tempo de execução (segundos)")
	ax.set_title(f"Tempo de execução vs k para m={m}")
	ax.legend(fontsize='x-large')
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

	k_value = round(np.mean(k))

	for i in range(len(algs)):

		alg_name = algs[i]
		timing = timings[i]

		try:

			_linear_model = linear_model(n, k)
			_polylog_model = polylog_model(n, k)

			fit_linear = regression(n, timing, _linear_model)
			fit_polylog = regression(n, timing, _polylog_model)
			print(f"Fit for {alg_name} (k={k_value}): Linear: {fit_linear}, Polylog: {fit_polylog}")

			with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
				predicted_linear = fit_linear.T @ _linear_model.T
				predicted_polylog = fit_polylog.T @ _polylog_model.T

		except np.linalg.LinAlgError:
			predicted_linear = timing
			predicted_polylog = timing

		error_linear = np.mean((timing - predicted_linear) ** 2)
		error_polylog = np.mean((timing - predicted_polylog) ** 2)

		label_linear = f"Modelagem Linear (MSE: {error_linear:.2e})"
		label_polylog = f"Modelagem Polylog (MSE: {error_polylog:.2e})"

		ax.plot(m, timing, label="Tempo Real", color="blue", alpha=0.5)
		# ax.plot(m, predicted_linear, color="black", alpha=0.8, label=label_linear, linestyle="--", linewidth=2)
		ax.plot(m, predicted_polylog, linestyle="dashed", alpha=0.8, color="red", label=label_polylog, linewidth=2)

	average_k = round(np.mean(k))

	ax.set_xlabel("Número de vértices de cada polígono (m)")
	ax.set_ylabel("Tempo de execução (segundos)")
	ax.set_title(f"Tempo de execução vs m para k={average_k}")
	ax.legend(fontsize='x-large')
	ax.grid(True)

if __name__ == "__main__":

	# fig, ax = plt.subplots(3, 2, figsize=(6, 6))
	# flat = ax.flatten()

	import os

	m_files = [f for f in os.listdir() if f.startswith("benchmark_results_k_1_to")]
	m_files.sort(key=lambda x: int(x.split("_")[-1].strip(".csv")))

	m_files = [
		"benchmark_results_k_1_to_3000_m_5.csv",
		"benchmark_results_k_1_to_3000_m_50.csv",
	]

	dfs = [pd.read_csv(f) for f in m_files]

	figsize = (6, 6)

	for df in dfs:

		fig, ax = plt.subplots(1, 1, figsize=figsize)
		plot_vs_k(df, ax)

		m = round(np.mean(df["n"] / df["k"]))
		filename = f"plt_vs_k_m_{m}.png"
		fig.tight_layout()
		plt.savefig(filename, dpi=250)

	k_files = [f for f in os.listdir() if re.search(r"benchmark\_results\_k\_\d+\_m\_1\_to\_\d+\.csv", f)]
	k_files.sort(key=lambda x: int(re.search(r"k_(\d+)_m", x).group(1)))

	k_files = [
		"benchmark_results_k_5_m_1_to_40000.csv",
		"benchmark_results_k_20_m_1_to_40000.csv",
	]

	dfs = [pd.read_csv(f) for f in k_files]

	for df in dfs:

		fig, ax = plt.subplots(1, 1, figsize=figsize)
		plot_vs_m(df, ax)

		k = round(np.mean(df["k"]))
		filename = f"plt_vs_m_k_{k}.png"
		fig.tight_layout()
		plt.savefig(filename, dpi=250)

