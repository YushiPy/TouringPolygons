/* global self */

import { loadEditorWasm, solveCaseWithEditorWasm, solveEditorWasmGroups } from "./editor-solver.js?v=editor-align-2026-09-09d";

self.onmessage = async ({ data }) => {
	try {
		const module = await loadEditorWasm();
		if (!module) {
			throw new Error("WASM solver unavailable.");
		}
		const result = data.pieceGroups
			? solveEditorWasmGroups(data.caseData, data.pieceGroups)
			: solveCaseWithEditorWasm(data.caseData);
		self.postMessage({ result });
	} catch (error) {
		self.postMessage({ error: error.message || "WASM solver worker failed." });
	}
};
