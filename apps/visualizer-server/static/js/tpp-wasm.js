import { Vector2 } from "./vector2.js";

let wasmModule = null;
let wasmLoadStarted = false;
let wasmLoadFailed = false;
const WASM_SOLVER_VERSION = "intersections-2026-09-01-length";

export function loadTppWasm() {
	if (wasmLoadStarted) {
		return;
	}

	wasmLoadStarted = true;
	wasmLoadFailed = false;

	fetch(`/static/wasm/tpp_convex_wasm.js?v=${WASM_SOLVER_VERSION}`, { method: "HEAD" })
		.then(response => response.ok ? import(`/static/wasm/tpp_convex_wasm.js?v=${WASM_SOLVER_VERSION}`) : null)
		.then(module => {
			if (module === null) {
				return null;
			}

			return module.default({
				locateFile: path => path.endsWith(".wasm") ? `/static/wasm/${path}?v=${WASM_SOLVER_VERSION}` : path,
			});
		})
		.then(module => {
			wasmModule = module;
		})
		.catch(() => {
			wasmModule = null;
			wasmLoadFailed = true;
		});
}

export function isTppWasmReady() {
	return wasmModule !== null;
}

export function tppWasmStatus() {
	if (wasmModule !== null) {
		return "ready";
	}

	return wasmLoadFailed ? "failed" : "loading";
}

export function solveTppWasm(start, target, polygons, maxCalls = 200000, maxSeconds = 3) {
	if (wasmModule === null) {
		return null;
	}

	const totalVertices = polygons.reduce((sum, polygon) => sum + polygon.length, 0);
	const pointsPtr = wasmModule._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
	const sizesPtr = wasmModule._malloc(polygons.length * Int32Array.BYTES_PER_ELEMENT);

	try {
		const points = new Float64Array(wasmModule.HEAPF64.buffer, pointsPtr, totalVertices * 2);
		const sizes = new Int32Array(wasmModule.HEAP32.buffer, sizesPtr, polygons.length);

		let pointIndex = 0;
		for (let i = 0; i < polygons.length; i++) {
			sizes[i] = polygons[i].length;

			for (const point of polygons[i]) {
				points[2 * pointIndex] = point[0];
				points[2 * pointIndex + 1] = point[1];
				pointIndex++;
			}
		}

		const pathSize = wasmModule._tpp_solve(
			start[0],
			start[1],
			target[0],
			target[1],
			pointsPtr,
			sizesPtr,
			polygons.length,
			maxCalls,
			maxSeconds,
		);

		return readSolution(pathSize);
	} finally {
		wasmModule._free(pointsPtr);
		wasmModule._free(sizesPtr);
	}
}

export function solveTppWasmGroups(start, target, pieceGroups, maxCalls = 200000, maxSeconds = 3) {
	if (wasmModule === null) {
		return null;
	}

	const pieces = pieceGroups.flat();
	const totalVertices = pieces.reduce((sum, piece) => sum + piece.length, 0);
	const pointsPtr = wasmModule._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
	const pieceSizesPtr = wasmModule._malloc(pieces.length * Int32Array.BYTES_PER_ELEMENT);
	const groupSizesPtr = wasmModule._malloc(pieceGroups.length * Int32Array.BYTES_PER_ELEMENT);

	try {
		const points = new Float64Array(wasmModule.HEAPF64.buffer, pointsPtr, totalVertices * 2);
		const pieceSizes = new Int32Array(wasmModule.HEAP32.buffer, pieceSizesPtr, pieces.length);
		const groupSizes = new Int32Array(wasmModule.HEAP32.buffer, groupSizesPtr, pieceGroups.length);

		let pointIndex = 0;
		let pieceIndex = 0;
		for (let i = 0; i < pieceGroups.length; i++) {
			groupSizes[i] = pieceGroups[i].length;

			for (const piece of pieceGroups[i]) {
				pieceSizes[pieceIndex] = piece.length;
				pieceIndex++;

				for (const point of piece) {
					points[2 * pointIndex] = point[0];
					points[2 * pointIndex + 1] = point[1];
					pointIndex++;
				}
			}
		}

		const pathSize = wasmModule._tpp_solve_piece_groups(
			start[0],
			start[1],
			target[0],
			target[1],
			pointsPtr,
			pieceSizesPtr,
			groupSizesPtr,
			pieceGroups.length,
			maxCalls,
			maxSeconds,
		);

		return readSolution(pathSize);
	} finally {
		wasmModule._free(pointsPtr);
		wasmModule._free(pieceSizesPtr);
		wasmModule._free(groupSizesPtr);
	}
}

export function solveConvexTppWasm(start, target, polygons) {
	const result = solveTppWasm(start, target, polygons);
	return result === null ? null : result.path;
}

function readSolution(pathSize) {
	if (pathSize < 0) {
		return null;
	}

	const outputPtr = wasmModule._tpp_get_path_points();
	const output = new Float64Array(wasmModule.HEAPF64.buffer, outputPtr, pathSize * 2);
	const path = [];

	for (let i = 0; i < pathSize; i++) {
		path.push(new Vector2(output[2 * i], output[2 * i + 1]));
	}

	return {
		path,
		exact: wasmModule._tpp_solution_exact() === 1,
		calls: wasmModule._tpp_solution_calls(),
		seconds: wasmModule._tpp_solution_seconds(),
	};
}
