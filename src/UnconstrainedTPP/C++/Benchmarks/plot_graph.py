
from collections.abc import Callable, Sequence

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]

def regression(x: Vector, y: Vector, parameters: Callable[[float], Vector]) -> Vector:
	"""
	Finds the optimal values for the parameters such that the mean squared error between the predicted values and the actual values is minimized.
	The prediction for a real number `x_i` is given by `<parameters(x_i).T @ w>`, where `w` is the vector of parameters to be optimized.

	The parameters function could be for example:
	- `parameters(x) = np.array([1, x])` for a linear regression.
	- `parameters(x) = np.array([1, x, x**2])` for a quadratic regression.
	- `parameters(x) = np.array([1, x, x log(x)])` for a regression with a logarithmic term.

	---
	Parameters:
	- `x`: A vector of real numbers representing the input data.
	- `y`: A vector of real numbers representing the target values corresponding to each input in `x`.
	- `parameters`: A function that takes a real number and returns a vector of features corresponding to that number. The length of the returned vector determines the number of parameters to be optimized.
	---
	Returns:
	- A vector of optimal parameter values that minimize the mean squared error between the predicted values and the actual values.
	"""

	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)

	x_matrix = np.array([parameters(x_i) for x_i in x], dtype=np.float64)
	y_vector = np.array(y, dtype=np.float64)

	# Solve the normal equations: (X^T X) w = X^T y
	X_transpose = x_matrix.T
	normal_matrix = X_transpose @ x_matrix
	normal_vector = X_transpose @ y_vector
	optimal_parameters = np.linalg.solve(normal_matrix, normal_vector).astype(np.float64)

	return optimal_parameters

COLORS = ["#003f5c", "#008c54", "#ffa600"]
ALGORITHMS = ["Linear Search", "Binary Search", "TAMC"]

df = pd.read_csv("benchmark_results_k_1_to_10000_m_3.csv")

k = np.array(df["k"], dtype=np.float64)
n = np.array(df["n"], dtype=np.float64)
timings = [np.array(df[alg], dtype=np.float64) for alg in ALGORITHMS]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for i in range(len(ALGORITHMS)):

	alg_name = ALGORITHMS[i]
	timing = timings[i]
	color = COLORS[i]

	model = lambda x: np.array([1, x, x * np.log(x)])
	fit = regression(k, timing, model)
	predicted = np.array([fit.T @ model(k_i) for k_i in k], dtype=np.float64)
	
	ax.plot(k, timing, label=alg_name, color=color)
	
	ax.plot(k, predicted, color="black", alpha=0.5)
	ax.plot(k, predicted, linestyle="dashed", color=color)

ax.set_xlabel("k")
ax.set_ylabel("Time (seconds)")
ax.set_title("Algorithm Performance as a Function of k")
ax.legend()
ax.grid(True)
