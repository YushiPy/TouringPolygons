import { $ } from "./dom.js";

const KEYBIND_STORAGE_KEY = "benchmarkDashboardManualEditorKeybinds";

const defaultKeybinds = {
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

function normalizeBindings(value, fallback) {
	if (Array.isArray(value)) {
		return value.filter(Boolean);
	}
	return value ? [value] : [...fallback];
}

function loadEditorKeybinds() {
	try {
		const loaded = JSON.parse(localStorage.getItem(KEYBIND_STORAGE_KEY) || "{}");
		return {
			closePolygon: normalizeBindings(loaded.closePolygon, defaultKeybinds.closePolygon),
			deleteSelection: normalizeBindings(loaded.deleteSelection, defaultKeybinds.deleteSelection),
			clearSelection: normalizeBindings(loaded.clearSelection, defaultKeybinds.clearSelection),
			toggleSnap: normalizeBindings(loaded.toggleSnap, defaultKeybinds.toggleSnap),
			fitInstance: normalizeBindings(loaded.fitInstance, defaultKeybinds.fitInstance),
			toggleGrid: normalizeBindings(loaded.toggleGrid, defaultKeybinds.toggleGrid),
			togglePath: normalizeBindings(loaded.togglePath, defaultKeybinds.togglePath),
			toggleDecomposition: normalizeBindings(loaded.toggleDecomposition, defaultKeybinds.toggleDecomposition),
			toggleLabels: normalizeBindings(loaded.toggleLabels, defaultKeybinds.toggleLabels),
		};
	} catch {
		return {
			closePolygon: [...defaultKeybinds.closePolygon],
			deleteSelection: [...defaultKeybinds.deleteSelection],
			clearSelection: [...defaultKeybinds.clearSelection],
			toggleSnap: [...defaultKeybinds.toggleSnap],
			fitInstance: [...defaultKeybinds.fitInstance],
			toggleGrid: [...defaultKeybinds.toggleGrid],
			togglePath: [...defaultKeybinds.togglePath],
			toggleDecomposition: [...defaultKeybinds.toggleDecomposition],
			toggleLabels: [...defaultKeybinds.toggleLabels],
		};
	}
}

function keyEventToBinding(event) {
	const parts = [];
	if (event.ctrlKey) {
		parts.push("Ctrl");
	}
	if (event.altKey) {
		parts.push("Alt");
	}
	if (event.shiftKey) {
		parts.push("Shift");
	}
	if (event.metaKey) {
		parts.push("Meta");
	}
	if (["Control", "Alt", "Shift", "Meta"].includes(event.key)) {
		return "";
	}
	const key = event.key === " " ? "Space" : event.key.length === 1 ? event.key.toUpperCase() : event.key;
	parts.push(key);
	return parts.join("+");
}

export function createKeybindManager({ setCloseIcon }) {
	const bindings = loadEditorKeybinds();
	let pendingAction = null;

	function save() {
		localStorage.setItem(KEYBIND_STORAGE_KEY, JSON.stringify(bindings));
	}

	function renderControl(action, selector) {
		const root = $(selector);
		if (!root) return;
		root.innerHTML = "";
		bindings[action].forEach((binding, index) => {
			const wrapper = document.createElement("span");
			wrapper.className = "keybind-chip";
			const button = document.createElement("button");
			button.type = "button";
			button.className = "secondary keybind-input";
			button.textContent = pendingAction?.action === action && pendingAction.index === index ? "Press keys..." : binding;
			button.addEventListener("click", () => {
				pendingAction = { action, index };
				updateUI();
			});
			const remove = document.createElement("button");
			remove.type = "button";
			remove.className = "keybind-remove";
			setCloseIcon(remove);
			remove.setAttribute("aria-label", `Remove ${binding}`);
			remove.addEventListener("click", () => {
				bindings[action].splice(index, 1);
				save();
				updateUI();
			});
			wrapper.append(button, remove);
			root.appendChild(wrapper);
		});
		const add = document.createElement("button");
		add.type = "button";
		add.className = "secondary keybind-input";
		add.textContent = pendingAction?.action === action && pendingAction.index === bindings[action].length ? "Press keys..." : "...";
		add.addEventListener("click", () => {
			pendingAction = { action, index: bindings[action].length };
			updateUI();
		});
		root.appendChild(add);
	}

	function updateUI() {
		renderControl("closePolygon", "#close-polygon-keybinds");
		renderControl("deleteSelection", "#delete-selection-keybinds");
		renderControl("clearSelection", "#clear-selection-keybinds");
		renderControl("toggleSnap", "#toggle-snap-keybinds");
		renderControl("fitInstance", "#fit-instance-keybinds");
		renderControl("toggleGrid", "#toggle-grid-keybinds");
		renderControl("togglePath", "#toggle-path-keybinds");
		renderControl("toggleDecomposition", "#toggle-decomposition-keybinds");
		renderControl("toggleLabels", "#toggle-labels-keybinds");
	}

	function open() {
		pendingAction = null;
		updateUI();
		$("#keybind-modal")?.classList.add("is-top-modal");
		$("#keybind-modal")?.classList.remove("is-hidden");
	}

	function close() {
		if (pendingAction) {
			bindings[pendingAction.action].splice(pendingAction.index, 1);
			bindings[pendingAction.action] = bindings[pendingAction.action].filter(Boolean);
			save();
		}
		pendingAction = null;
		updateUI();
		$("#keybind-modal")?.classList.remove("is-top-modal");
		$("#keybind-modal")?.classList.add("is-hidden");
	}

	function capturePending(event) {
		if (!pendingAction) {
			return false;
		}
		event.preventDefault();
		const binding = keyEventToBinding(event);
		if (event.key === "Escape") {
			pendingAction = null;
			updateUI();
		} else if (binding) {
			const actionBindings = bindings[pendingAction.action];
			if (!actionBindings.includes(binding)) {
				actionBindings[pendingAction.index] = binding;
			}
			bindings[pendingAction.action] = actionBindings.filter(Boolean);
			save();
			pendingAction = null;
			updateUI();
		}
		return true;
	}

	function matches(event, action) {
		return bindings[action].includes(keyEventToBinding(event));
	}

	return {
		bindings,
		capturePending,
		close,
		matches,
		open,
		updateUI,
	};
}
