import { createManualEditor } from "./manual-editor.js?v=offline-editor-20260921";
import { casePayload, cloneCaseData, emptyCaseData } from "./case-data.js";
import { convexDecomposition } from "./editor-geometry.js";
import { solveEditorWasmAsync, solveEditorWasmMaps } from "./editor-solver.js?v=editor-align-2026-09-21a";
import { formatLength, formatSeconds } from "./format.js";
import { $ } from "./dom.js";

const STORAGE_KEY = "tpp-offline-editor-library-v2";
const LEGACY_STORAGE_KEY = "tpp-offline-editor-case-v1";

function demoCase() {
	return {
		name: "offline-instance",
		generated: false,
		start: [-8, 0],
		target: [8, 0],
		polygons: [
			[[-5, -2], [-3.5, -2.5], [-2.5, 0], [-3.5, 2.5], [-5, 2]],
			[[0, -2.5], [2.5, -2], [3, 0], [2, 2.4], [0, 2.1], [-0.8, 0]],
		],
		background: null,
		map_view: null,
	};
}

function offlineKeybinds() {
	const bindings = {
		closePolygon: ["Enter", "C"],
		deleteSelection: ["X"],
		clearSelection: ["Z"],
		toggleSnap: ["S"],
		fitInstance: ["F"],
		toggleGrid: ["G"],
		togglePath: ["P"],
		toggleDecomposition: ["D"],
		toggleLabels: ["L"],
	};
	return {
		bindings,
		capturePending: () => false,
		close: () => undefined,
		open: () => undefined,
		updateUI: () => undefined,
		matches(event, action) {
			const modifiers = [event.ctrlKey && "Ctrl", event.altKey && "Alt", event.shiftKey && "Shift", event.metaKey && "Meta"].filter(Boolean);
			const key = event.key === " " ? "Space" : event.key.length === 1 ? event.key.toUpperCase() : event.key;
			return bindings[action]?.includes([...modifiers, key].join("+"));
		},
	};
}

function localPartition(polygon) {
	return convexDecomposition(polygon);
}

function noServerFallback() {
	throw new Error("The offline editor could not load the local WASM solver. Build apps/benchmark-dashboard/wasm first.");
}

function normaliseCaseData(data, fallbackName = "offline-instance") {
	const source = data?.case && typeof data.case === "object" ? data.case : data;
	if (!source || !Array.isArray(source.polygons)) {
		throw new Error("JSON must contain start, target, and polygons.");
	}
	return cloneCaseData({
		...source,
		name: source.name || source.drawingName || fallbackName,
		start: source.start || source.startPoint || [0, 0],
		target: source.target || source.targetPoint || [1, 0],
	});
}

function readStoredCases() {
	try {
		const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
		if (Array.isArray(stored)) {
			const cases = stored.map((item, index) => normaliseCaseData(item, `offline-instance-${index + 1}`));
			return cases.length ? cases : [demoCase()];
		}
		if (Array.isArray(stored?.cases)) {
			const cases = stored.cases.map((item, index) => normaliseCaseData(item, `offline-instance-${index + 1}`));
			return cases.length ? cases : [demoCase()];
		}
		const legacy = JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY) || "null");
		if (legacy) {
			return [normaliseCaseData(legacy)];
		}
		return [demoCase()];
	} catch {
		return [demoCase()];
	}
}

const state = {
	manualCases: readStoredCases(),
	manualCaseIndex: 0,
};

let saveTimer = null;
function persistCase() {
	const cases = state.manualCases.map((current) => casePayload(current));
	localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 2, cases }));
	$("#manual-save-status").textContent = "Saved in this browser";
}

function scheduleManualAutosave(options = {}) {
	clearTimeout(saveTimer);
	if (options.immediate) {
		persistCase();
		return;
	}
	saveTimer = setTimeout(persistCase, 180);
}

function updateSummary() {
	const current = state.manualCases[state.manualCaseIndex];
	$("#offline-summary").textContent = `${current.polygons.filter((polygon) => polygon.length >= 3).length} polygon(s), ${current.polygons.reduce((total, polygon) => total + polygon.length, 0)} vertices`;
}

function renderCaseList() {
	const list = $("#offline-case-list");
	list.replaceChildren();
	state.manualCases.forEach((current, index) => {
		const button = document.createElement("button");
		button.type = "button";
		button.className = `offline-case-item${index === state.manualCaseIndex ? " is-active" : ""}`;
		button.setAttribute("role", "option");
		button.setAttribute("aria-selected", String(index === state.manualCaseIndex));
		const title = document.createElement("span");
		title.textContent = current.name || `offline-instance-${index + 1}`;
		const details = document.createElement("small");
		details.textContent = `${current.polygons.length} polygon(s)`;
		button.replaceChildren(title, details);
		button.addEventListener("click", () => selectCase(index));
		list.append(button);
	});
}

function cssVar(name) {
	return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const manualEditor = createManualEditor({
	$,
	state,
	keybinds: offlineKeybinds(),
	formatLength,
	formatSeconds,
	cssVar,
	scheduleManualAutosave,
	updateManualCaseListMetadata: updateSummary,
	partitionProvider: localPartition,
	visitOrderProvider: () => "fixed",
	solveWasmProvider: solveEditorWasmAsync,
	lastStepMapProvider: solveEditorWasmMaps,
	solveFixedOrderProvider: noServerFallback,
});

function loadCase(data) {
	state.manualCases[state.manualCaseIndex] = normaliseCaseData(data, "imported-instance");
	manualEditor.cancelPendingSolution();
	manualEditor.solutionPath = null;
	manualEditor.activePolygon = null;
	manualEditor.clearSelection();
	manualEditor.frameCurrentCase();
	manualEditor.changed();
	$("#offline-name").value = state.manualCases[state.manualCaseIndex].name;
	updateSummary();
	renderCaseList();
}

function selectCase(index) {
	if (index < 0 || index >= state.manualCases.length || index === state.manualCaseIndex) return;
	state.manualCaseIndex = index;
	manualEditor.cancelPendingSolution();
	manualEditor.solutionPath = null;
	manualEditor.activePolygon = null;
	manualEditor.clearSelection();
	$("#offline-name").value = state.manualCases[index].name;
	manualEditor.frameCurrentCase();
	manualEditor.changed();
	updateSummary();
	renderCaseList();
}

function addCase() {
	state.manualCases.push({ ...emptyCaseData(), name: `offline-instance-${state.manualCases.length + 1}` });
	state.manualCaseIndex = state.manualCases.length - 1;
	loadCase(state.manualCases[state.manualCaseIndex]);
}

function deleteCase() {
	if (state.manualCases.length === 1) {
		loadCase({ ...emptyCaseData(), name: "offline-instance" });
		return;
	}
	state.manualCases.splice(state.manualCaseIndex, 1);
	state.manualCaseIndex = Math.min(state.manualCaseIndex, state.manualCases.length - 1);
	const nextIndex = state.manualCaseIndex;
	state.manualCaseIndex = -1;
	selectCase(nextIndex);
}

function downloadCase() {
	const current = casePayload(state.manualCases[state.manualCaseIndex]);
	const name = (current.name || "offline-instance").replace(/[^A-Za-z0-9._-]+/g, "-");
	const blob = new Blob([JSON.stringify(current, null, 2) + "\n"], { type: "application/json" });
	const url = URL.createObjectURL(blob);
	const link = document.createElement("a");
	link.href = url;
	link.download = `${name || "offline-instance"}.tpp.json`;
	link.click();
	URL.revokeObjectURL(url);
}

function downloadLibrary() {
	const payload = {
		version: 2,
		cases: state.manualCases.map((current) => casePayload(current)),
	};
	const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json" });
	const url = URL.createObjectURL(blob);
	const link = document.createElement("a");
	link.href = url;
	link.download = "tpp-offline-library.json";
	link.click();
	URL.revokeObjectURL(url);
}

$("#offline-name").addEventListener("input", (event) => {
	state.manualCases[state.manualCaseIndex].name = event.currentTarget.value.trim() || "offline-instance";
	scheduleManualAutosave();
	renderCaseList();
});
$("#offline-new").addEventListener("click", addCase);
$("#offline-add-case").addEventListener("click", addCase);
$("#offline-delete-case").addEventListener("click", deleteCase);
$("#offline-export").addEventListener("click", downloadCase);
$("#offline-export-all").addEventListener("click", downloadLibrary);
$("#offline-import").addEventListener("click", () => $("#offline-import-input").click());
$("#offline-import-input").addEventListener("change", async (event) => {
	const file = event.currentTarget.files?.[0];
	event.currentTarget.value = "";
	if (!file) return;
	try {
		const imported = JSON.parse(await file.text());
		const importedCases = Array.isArray(imported) ? imported : imported?.cases;
		if (Array.isArray(importedCases)) {
			state.manualCases = importedCases.map((item, index) => normaliseCaseData(item, `imported-instance-${index + 1}`));
			state.manualCaseIndex = 0;
			loadCase(state.manualCases[0]);
		} else {
			loadCase(imported);
		}
		$("#offline-local-status").textContent = `Imported ${file.name}`;
	} catch (error) {
		$("#offline-local-status").textContent = error.message || "Invalid JSON file.";
	}
});

$("#close-manual-polygon").addEventListener("click", () => manualEditor.closePolygon());
$("#clear-manual-selection").addEventListener("click", () => manualEditor.clearSelection());
$("#delete-manual-selection").addEventListener("click", () => manualEditor.deleteSelection());
$("#toggle-manual-snapping").addEventListener("click", () => manualEditor.toggleSnapping());
$("#manual-zoom-out").addEventListener("click", () => manualEditor.zoomBy(1 / 1.2));
$("#manual-zoom-in").addEventListener("click", () => manualEditor.zoomBy(1.2));
$("#manual-fit-instance").addEventListener("click", () => manualEditor.frameCurrentCase());
document.querySelectorAll("[data-manual-mode] [data-mode]").forEach((button) => {
	button.addEventListener("click", () => manualEditor.setMode(button.dataset.mode));
});
document.querySelectorAll("[data-editor-layer]").forEach((button) => {
	button.addEventListener("click", () => manualEditor.toggleLayer(button.dataset.editorLayer));
});

manualEditor.init();
manualEditor.frameCurrentCase();
$("#offline-name").value = state.manualCases[0].name;
updateSummary();
renderCaseList();
$("#offline-local-status").textContent = "No dashboard API requests; data stays in this browser.";
