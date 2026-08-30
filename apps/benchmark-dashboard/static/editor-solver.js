import { convexDecomposition, polygonIsConvex } from "./editor-geometry.js";

export const editorSolverState = {
	module: null,
	load: null,
	failed: false,
	geometry: null,
};

export function loadEditorWasm() {
	if (editorSolverState.load) {
		return editorSolverState.load;
	}
	editorSolverState.load = import("/visualizer-static/wasm/tpp_convex_wasm.js")
		.then((module) => module.default({
			locateFile: (path) => path.endsWith(".wasm") ? `/visualizer-static/wasm/${path}` : path,
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

export function solveEditorWasm(caseData, maxCalls = 200000, maxSeconds = 3) {
	const module = editorSolverState.module;
	if (!module) {
		return null;
	}
	const polygons = caseData.polygons;
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
			caseData.start[0],
			caseData.start[1],
			caseData.target[0],
			caseData.target[1],
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
	const pieces = pieceGroups.flat();
	const totalVertices = pieces.reduce((sum, piece) => sum + piece.length, 0);
	const pointsPtr = module._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
	const pieceSizesPtr = module._malloc(pieces.length * Int32Array.BYTES_PER_ELEMENT);
	const groupSizesPtr = module._malloc(pieceGroups.length * Int32Array.BYTES_PER_ELEMENT);
	try {
		const points = new Float64Array(module.HEAPF64.buffer, pointsPtr, totalVertices * 2);
		const pieceSizes = new Int32Array(module.HEAP32.buffer, pieceSizesPtr, pieces.length);
		const groupSizes = new Int32Array(module.HEAP32.buffer, groupSizesPtr, pieceGroups.length);
		let pointIndex = 0;
		let pieceIndex = 0;
		pieceGroups.forEach((group, groupIndex) => {
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
			caseData.start[0],
			caseData.start[1],
			caseData.target[0],
			caseData.target[1],
			pointsPtr,
			pieceSizesPtr,
			groupSizesPtr,
			pieceGroups.length,
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
	if (caseData.polygons.every(polygonIsConvex)) {
		wasmResult = solveEditorWasm(caseData);
	} else {
		wasmResult = solveEditorWasmGroups(caseData, caseData.polygons.map(convexDecomposition));
	}
	return wasmResult;
}
