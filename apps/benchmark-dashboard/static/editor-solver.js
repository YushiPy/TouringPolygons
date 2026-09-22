/* global DOMException, Worker */

import { convexDecomposition, polygonIsConvex, polygonsPairwiseDisjoint } from "./editor-geometry.js";

export const editorSolverState = {
	module: null,
	load: null,
	failed: false,
};

const WASM_SOLVER_VERSION = "editor-align-2026-09-21a";
let idleSolverWorker = null;

export function solveEditorWasmAsync(caseData, pieceGroups = null, signal = null) {
	if (signal?.aborted) {
		return Promise.reject(new DOMException("The solve was cancelled.", "AbortError"));
	}
	if (globalThis.__editorWasmAvailable === false) return Promise.resolve(null);
	const smallConvex = !pieceGroups && caseData.polygons.length <= 4
		&& caseData.polygons.reduce((count, polygon) => count + polygon.length, 0) <= 24
		&& caseData.polygons.every(polygonIsConvex);
	if (smallConvex) {
		return loadEditorWasm().then(() => {
			if (signal?.aborted) throw new DOMException("The solve was cancelled.", "AbortError");
			return solveEditorWasm(caseData);
		});
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
		try {
			worker.postMessage({ caseData: { start: caseData.start, target: caseData.target, polygons: caseData.polygons }, pieceGroups });
		} catch (error) {
			finish(reject, error);
		}
	});

	return promise;
}

export function loadEditorWasm() {
	if (globalThis.__editorWasmAvailable === false) {
		editorSolverState.failed = true;
		return Promise.resolve(null);
	}
	if (editorSolverState.load) {
		return editorSolverState.load;
	}
	editorSolverState.load = import(`/static/wasm/tpp_convex_wasm.js?v=${WASM_SOLVER_VERSION}`)
		.then((module) => module.default({
			locateFile: (path) => path.endsWith(".wasm") ? `/static/wasm/${path}?v=${WASM_SOLVER_VERSION}` : path,
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
		const solveStarted = performance.now();
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
		const solveSeconds = (performance.now() - solveStarted) / 1000;
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
			seconds: solveSeconds,
			source: "wasm",
		};
	} finally {
		module._free(pointsPtr);
		module._free(sizesPtr);
	}
}

export async function solveEditorWasmMaps(caseData) {
	const polygons = caseData?.polygons || [];
	if (!polygonsPairwiseDisjoint(polygons)) {
		return {
			eligible: false,
			reason: "Last-step maps require convex, pairwise-disjoint polygons.",
		};
	}
	const module = await loadEditorWasm();
	if (!module || typeof module._tpp_solve_convex_maps !== "function") {
		return null;
	}
	return solveEditorWasmMapsSync(caseData, module);
}

function solveEditorWasmMapsSync(caseData, module) {
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
		const polygonCount = module._tpp_solve_convex_maps(
			normalizedCase.start[0],
			normalizedCase.start[1],
			normalizedCase.target[0],
			normalizedCase.target[1],
			pointsPtr,
			sizesPtr,
			polygons.length,
		);
		if (polygonCount < 0 || polygonCount !== polygons.length) {
			return null;
		}
		const offsetsPtr = module._tpp_get_map_offsets();
		const raysPtr = module._tpp_get_map_rays();
		const firstContactPtr = module._tpp_get_map_first_contact();
		if (!offsetsPtr || !raysPtr || !firstContactPtr) return null;
		const offsets = new Int32Array(module.HEAP32.buffer, offsetsPtr, polygonCount + 1);
		const rays = new Float64Array(module.HEAPF64.buffer, raysPtr, totalVertices * 4);
		const firstContact = new Uint8Array(module.HEAPU8.buffer, firstContactPtr, totalVertices);
		const mapPolygons = polygons.map((polygon, polygonIndex) => {
			const start = offsets[polygonIndex];
			const end = offsets[polygonIndex + 1];
			if (start < 0 || end - start !== polygon.length) return null;
			return {
				polygon: polygon.map((point) => [...point]),
				cones: polygon.map((_, vertexIndex) => {
					const offset = (start + vertexIndex) * 4;
					return [
						[rays[offset], rays[offset + 1]],
						[rays[offset + 2], rays[offset + 3]],
					];
				}),
				firstContact: Array.from(firstContact.slice(start, end), Boolean),
			};
		});
		if (mapPolygons.some((map) => !map || map.cones.some((cone) => cone.flat().some((value) => !Number.isFinite(value))))) {
			return null;
		}
		return {
			eligible: true,
			source: "wasm",
			polygons: mapPolygons,
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
		const solveStarted = performance.now();
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
		const solveSeconds = (performance.now() - solveStarted) / 1000;
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
			seconds: solveSeconds,
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
