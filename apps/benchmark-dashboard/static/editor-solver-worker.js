/* global self */

import { loadEditorWasm, solveEditorWasm, solveEditorWasmGroups } from "./editor-solver.js?v=intersections-2026-09-01-length";

self.onmessage = async ({ data }) => {
	try {
		const module = await loadEditorWasm();
		if (!module) {
			throw new Error("WASM solver unavailable.");
		}
		const result = data.pieceGroups
			? solveEditorWasmGroups(data.caseData, data.pieceGroups)
			: solveEditorWasm(data.caseData);
		self.postMessage({ result });
	} catch (error) {
		self.postMessage({ error: error.message || "WASM solver worker failed." });
	}
};
