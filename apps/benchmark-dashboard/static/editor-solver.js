/* global DOMException, Worker */

import { convexDecomposition, polygonIsConvex } from "./editor-geometry.js";

export const editorSolverState = {
	module: null,
	load: null,
	failed: false,
	geometry: null,
};

export const WORKER_SOLVE_VERTEX_THRESHOLD = 120;
const WASM_SOLVER_VERSION = "intersections-2026-09-01-length";
let idleSolverWorker = null;

function solveVertexCount(caseData, pieceGroups) {
	if (pieceGroups) {
		return pieceGroups.flat(2).length;
	}
	return caseData.polygons.reduce((sum, polygon) => sum + polygon.length, 0);
}

export function solveEditorWasmAsync(caseData, pieceGroups = null, signal = null) {
	if (signal?.aborted) {
		return Promise.reject(new DOMException("The solve was cancelled.", "AbortError"));
	}
	if (editorSolverState.module && solveVertexCount(caseData, pieceGroups) <= WORKER_SOLVE_VERTEX_THRESHOLD) {
		return Promise.resolve(pieceGroups ? solveEditorWasmGroups(caseData, pieceGroups) : solveEditorWasm(caseData));
	}
	const worker = idleSolverWorker || new Worker(new URL(`./editor-solver-worker.js?v=${WASM_SOLVER_VERSION}`, import.meta.url), { type: "module" });
	idleSolverWorker = null;
	let settled = false;
	let cancel = null;

	const promise = new Promise((resolve, reject) => {
		const finish = (callback, value, reusable = false) => {
			if (settled) {
				return;
			}
			settled = true;
			if (signal && cancel) {
				signal.removeEventListener("abort", cancel);
			}
			worker.onmessage = null;
			worker.onerror = null;
			if (reusable) {
				if (idleSolverWorker && idleSolverWorker !== worker) {
					idleSolverWorker.terminate();
				}
				idleSolverWorker = worker;
			} else {
				worker.terminate();
				if (idleSolverWorker === worker) {
					idleSolverWorker = null;
				}
			}
			callback(value);
		};
		cancel = () => finish(reject, new DOMException("The solve was cancelled.", "AbortError"));

		worker.onmessage = (event) => {
			if (event.data.error) {
				finish(reject, new Error(event.data.error));
				return;
			}
			finish(resolve, event.data.result, true);
		};
		worker.onerror = (event) => finish(reject, new Error(event.message || "WASM solver worker failed."));
		if (signal) {
			signal.addEventListener("abort", cancel, { once: true });
			if (signal.aborted) {
				cancel();
				return;
			}
		}
		worker.postMessage({ caseData, pieceGroups });
	});

	return promise;
}

export function loadEditorWasm() {
	if (editorSolverState.load) {
		return editorSolverState.load;
	}
	editorSolverState.load = import(`/visualizer-static/wasm/tpp_convex_wasm.js?v=${WASM_SOLVER_VERSION}`)
		.then((module) => module.default({
			locateFile: (path) => path.endsWith(".wasm") ? `/visualizer-static/wasm/${path}?v=${WASM_SOLVER_VERSION}` : path,
		}))
		.then((module) => {
			editorSolverState.module = module;
			return module;
		})
		.catch(() => {
			editorSolverState.failed = true;
			editorSolverState.module = null;
			return null;
		});
	return editorSolverState.load;
}

export async function loadEditorGeometry() {
	if (!globalThis.polygonClipping) {
		return null;
	}
	if (editorSolverState.geometry) {
		return editorSolverState.geometry;
	}
	try {
		editorSolverState.geometry = {
			vector: await import("/visualizer-static/js/vector2.js"),
			partition: await import("/visualizer-static/js/convex-partition.js"),
		};
	} catch {
		editorSolverState.geometry = null;
	}
	return editorSolverState.geometry;
}

function signedArea2(polygon) {
	let area = 0;
	for (let index = 0; index < polygon.length; index += 1) {
		const point = polygon[index];
		const next = polygon[(index + 1) % polygon.length];
		area += point[0] * next[1] - next[0] * point[1];
	}
	return area;
}

function counterClockwisePolygon(polygon) {
	const points = polygon.map((point) => [...point]);
	return points.length >= 3 && signedArea2(points) < 0 ? points.reverse() : points;
}

function counterClockwiseCase(caseData) {
	return {
		...caseData,
		start: [...caseData.start],
		target: [...caseData.target],
		polygons: caseData.polygons.map(counterClockwisePolygon),
	};
}

function pathLength(path) {
	let length = 0;
	for (let index = 1; index < path.length; index += 1) {
		length += Math.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]);
	}
	return length;
}

export function solveEditorWasm(caseData, maxCalls = 200000, maxSeconds = 3) {
	const module = editorSolverState.module;
	if (!module) {
		return null;
	}
	const normalizedCase = counterClockwiseCase(caseData);
	const polygons = normalizedCase.polygons;
	const totalVertices = polygons.reduce((sum, polygon) => sum + polygon.length, 0);
	const pointsPtr = module._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
	const sizesPtr = module._malloc(polygons.length * Int32Array.BYTES_PER_ELEMENT);
	try {
		const points = new Float64Array(module.HEAPF64.buffer, pointsPtr, totalVertices * 2);
		const sizes = new Int32Array(module.HEAP32.buffer, sizesPtr, polygons.length);
		let pointIndex = 0;
		polygons.forEach((polygon, polygonIndex) => {
			sizes[polygonIndex] = polygon.length;
			polygon.forEach((point) => {
				points[2 * pointIndex] = point[0];
				points[2 * pointIndex + 1] = point[1];
				pointIndex += 1;
			});
		});
		const pathSize = module._tpp_solve(
			normalizedCase.start[0],
			normalizedCase.start[1],
			normalizedCase.target[0],
			normalizedCase.target[1],
			pointsPtr,
			sizesPtr,
			polygons.length,
			maxCalls,
			maxSeconds,
		);
		if (pathSize < 0) {
			return null;
		}
		const outputPtr = module._tpp_get_path_points();
		const output = new Float64Array(module.HEAPF64.buffer, outputPtr, pathSize * 2);
		const path = [];
		for (let index = 0; index < pathSize; index += 1) {
			path.push([output[2 * index], output[2 * index + 1]]);
		}
		return {
			path,
			length: pathLength(path),
			exact: module._tpp_solution_exact() === 1,
			calls: module._tpp_solution_calls(),
			seconds: module._tpp_solution_seconds(),
			source: "wasm",
		};
	} finally {
		module._free(pointsPtr);
		module._free(sizesPtr);
	}
}

export function solveEditorWasmGroups(caseData, pieceGroups, maxCalls = 200000, maxSeconds = 3) {
	const module = editorSolverState.module;
	if (!module) {
		return null;
	}
	const normalizedCase = counterClockwiseCase(caseData);
	const normalizedGroups = pieceGroups.map((group) => group.map(counterClockwisePolygon));
	const pieces = normalizedGroups.flat();
	const totalVertices = pieces.reduce((sum, piece) => sum + piece.length, 0);
	const pointsPtr = module._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
	const pieceSizesPtr = module._malloc(pieces.length * Int32Array.BYTES_PER_ELEMENT);
	const groupSizesPtr = module._malloc(normalizedGroups.length * Int32Array.BYTES_PER_ELEMENT);
	try {
		const points = new Float64Array(module.HEAPF64.buffer, pointsPtr, totalVertices * 2);
		const pieceSizes = new Int32Array(module.HEAP32.buffer, pieceSizesPtr, pieces.length);
		const groupSizes = new Int32Array(module.HEAP32.buffer, groupSizesPtr, normalizedGroups.length);
		let pointIndex = 0;
		let pieceIndex = 0;
		normalizedGroups.forEach((group, groupIndex) => {
			groupSizes[groupIndex] = group.length;
			group.forEach((piece) => {
				pieceSizes[pieceIndex] = piece.length;
				pieceIndex += 1;
				piece.forEach((point) => {
					points[2 * pointIndex] = point[0];
					points[2 * pointIndex + 1] = point[1];
					pointIndex += 1;
				});
			});
		});
		const pathSize = module._tpp_solve_piece_groups(
			normalizedCase.start[0],
			normalizedCase.start[1],
			normalizedCase.target[0],
			normalizedCase.target[1],
			pointsPtr,
			pieceSizesPtr,
			groupSizesPtr,
			normalizedGroups.length,
			maxCalls,
			maxSeconds,
		);
		if (pathSize < 0) {
			return null;
		}
		const outputPtr = module._tpp_get_path_points();
		const output = new Float64Array(module.HEAPF64.buffer, outputPtr, pathSize * 2);
		const path = [];
		for (let index = 0; index < pathSize; index += 1) {
			path.push([output[2 * index], output[2 * index + 1]]);
		}
		return {
			path,
			length: pathLength(path),
			exact: module._tpp_solution_exact() === 1,
			calls: module._tpp_solution_calls(),
			seconds: module._tpp_solution_seconds(),
			source: "wasm",
		};
	} finally {
		module._free(pointsPtr);
		module._free(pieceSizesPtr);
		module._free(groupSizesPtr);
	}
}

export function solveCaseWithEditorWasm(caseData) {
	let wasmResult = null;
	const normalizedCase = counterClockwiseCase(caseData);
	if (normalizedCase.polygons.every(polygonIsConvex)) {
		wasmResult = solveEditorWasm(normalizedCase);
	} else {
		wasmResult = solveEditorWasmGroups(normalizedCase, normalizedCase.polygons.map(convexDecomposition));
	}
	return wasmResult;
}
