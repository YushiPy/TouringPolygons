import { drawCanvasScene } from "./canvas-renderer.js";
import { cloneCaseData } from "./case-data.js";
import { convexDecomposition, polygonIsConvex, solutionDirectionAt } from "./editor-geometry.js";
import { editorSolverState, loadEditorGeometry, loadEditorWasm, solveEditorWasmAsync } from "./editor-solver.js";
import { CAMERA_STORAGE_KEY, canvasToWorld as cameraCanvasToWorld, caseBounds as cameraCaseBounds, worldToCanvas as cameraWorldToCanvas, zoomLimits } from "./manual-editor-camera.js";
import { mergeSelections, pointsInRect as selectionPointsInRect, samePointSelection as selectionSamePoint } from "./manual-editor-selection.js";

export function createManualEditor({
	$,
	state,
	keybinds,
	formatSeconds,
	cssVar,
	scheduleManualAutosave,
	updateManualCaseListMetadata,
}) {
	const manualEditor = {
		canvas: null,
		ctx: null,
		mode: "move",
		scale: 70,
		minScale: 0.1,
		maxScale: 50000,
		offsetX: 0,
		offsetY: 0,
		activePoint: null,
		selectedPoint: null,
		selectedPoints: [],
		selectionBase: [],
		dragPolygon: null,
		dragSelection: null,
		panDrag: null,
		pointerStart: null,
		activePointers: new Map(),
		pinchGesture: null,
		selectionRect: null,
		mouseCanvas: { x: 0, y: 0 },
		selectionSpinStarted: performance.now(),
		selectionAnimationRunning: false,
		snapping: false,
		activePolygon: null,
		solutionPath: null,
		solutionStale: false,
		solutionTimer: null,
		solutionFrame: null,
		solutionAbort: null,
		solutionRevision: 0,
		layers: {
			grid: true,
			solution: true,
			decomposition: true,
			labels: true,
		},
		labelDirections: {
			start: [1, 0],
			target: [-1, 0],
		},
		labelAnimation: null,
		expanded: false,

		init() {
			this.canvas = $("#manual-case-canvas");
			if (!this.canvas) {
				return;
			}
			this.ctx = this.canvas.getContext("2d");
			this.canvas.tabIndex = 0;
			this.loadCamera();
			this.resize();
			new ResizeObserver(() => {
				this.resize();
				this.draw();
			}).observe(this.canvas);
			this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
			this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
			$("#cases-panel")?.addEventListener("click", (event) => {
				if (this.expanded && event.target === event.currentTarget) {
					this.toggleExpanded(false);
				}
			});
			const finishPointer = (event) => {
				this.activePointers.delete(event.pointerId);
				if (this.pinchGesture && this.activePointers.size < 2) {
					this.pinchGesture = null;
				}
				this.activePoint = null;
				this.dragSelection = null;
				this.dragPolygon = null;
				const pointerStart = this.pointerStart;
				this.panDrag = null;
				this.pointerStart = null;
				if (this.selectionRect) {
					this.finishSelection();
				} else if (pointerStart?.kind === "pan") {
					const moved = Math.hypot(event.clientX - pointerStart.clientX, event.clientY - pointerStart.clientY) > 4;
					if (!moved) {
						this.clearSelection();
					}
				}
				if (event.pointerId !== undefined && this.canvas.hasPointerCapture?.(event.pointerId)) {
					this.canvas.releasePointerCapture(event.pointerId);
				}
				this.updateCursor();
			};
			window.addEventListener("pointerup", finishPointer);
			window.addEventListener("pointercancel", finishPointer);
			this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
			document.addEventListener("keydown", (event) => this.onKeyDown(event));
			loadEditorWasm();
			loadEditorGeometry();
			this.syncCloseButton();
			this.toggleSnapping(false);
			this.draw();
		},

		resize() {
			const dpr = window.devicePixelRatio || 1;
			this.canvas.width = Math.max(1, Math.round(this.canvas.offsetWidth * dpr));
			this.canvas.height = Math.max(1, Math.round(this.canvas.offsetHeight * dpr));
			this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			if (this.offsetX === 0 && this.offsetY === 0) {
				this.offsetX = this.canvas.offsetWidth / 2;
				this.offsetY = this.canvas.offsetHeight / 2;
			}
		},

		loadCamera() {
			try {
				const camera = JSON.parse(localStorage.getItem(CAMERA_STORAGE_KEY) || "null");
				if (!camera) {
					return;
				}
				this.scale = Number(camera.scale) || this.scale;
				this.offsetX = Number(camera.offsetX) || this.offsetX;
				this.offsetY = Number(camera.offsetY) || this.offsetY;
			} catch {
				// Ignore stale local camera data.
			}
		},

		saveCamera() {
			localStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify({
				scale: this.scale,
				offsetX: this.offsetX,
				offsetY: this.offsetY,
			}));
		},

		currentCase() {
			return state.manualCases[state.manualCaseIndex] || null;
		},

		caseBounds(caseData = this.currentCase()) {
			return cameraCaseBounds(caseData);
		},

		updateZoomLimits(bounds = this.caseBounds()) {
			const limits = zoomLimits(bounds, this.canvas?.offsetWidth || 0, this.canvas?.offsetHeight || 0, this.scale);
			this.minScale = limits.minScale;
			this.maxScale = limits.maxScale;
			this.scale = limits.scale;
		},

		frameCurrentCase() {
			const bounds = this.caseBounds();
			if (!bounds || !this.canvas) {
				return;
			}
			this.updateZoomLimits(bounds);
			const width = this.canvas.offsetWidth;
			const height = this.canvas.offsetHeight;
			const pad = 42;
			const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
			const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
			this.scale = Math.max(this.minScale, Math.min(this.maxScale, Math.min(
				(width - 2 * pad) / spanX,
				(height - 2 * pad) / spanY,
			)));
			const centerX = (bounds.minX + bounds.maxX) / 2;
			const centerY = (bounds.minY + bounds.maxY) / 2;
			this.offsetX = width / 2 - centerX * this.scale;
			this.offsetY = height / 2 + centerY * this.scale;
			this.saveCamera();
			this.draw();
		},

		toggleLayer(layer, force = null) {
			if (!(layer in this.layers)) {
				return;
			}
			this.layers[layer] = force === null ? !this.layers[layer] : force;
			const button = document.querySelector(`[data-editor-layer="${layer}"]`);
			button?.classList.toggle("is-active", this.layers[layer]);
			button?.setAttribute("aria-pressed", this.layers[layer] ? "true" : "false");
			this.draw();
		},

		pointRadius(kind = "vertex") {
			const base = Math.sqrt(Math.max(0.1, this.scale)) * 0.62;
			const radius = Math.max(1.4, Math.min(kind === "endpoint" ? 6.5 : 5.2, base));
			return radius;
		},

		worldToCanvas(point) {
			return cameraWorldToCanvas(point, this);
		},

		canvasToWorld(x, y) {
			return cameraCanvasToWorld(x, y, this);
		},

		snapStep() {
			return this.gridMetrics().subGridSpacing;
		},

		gridMetrics() {
			const decisionValue = 83 / this.scale;
			let exponent = Math.ceil(Math.log10(decisionValue)) || 0;
			let multiplier = 1;
			let subGridCount = 4;
			const gridScale = 10 ** exponent;
			if (gridScale / 5 > decisionValue) {
				subGridCount = 3;
				exponent -= 1;
				multiplier = 2;
			} else if (gridScale / 2 > decisionValue) {
				exponent -= 1;
				multiplier = 5;
			}
			const gridSpacing = (10 ** exponent) * multiplier;
			return {
				gridSpacing,
				subGridCount,
				subGridSpacing: gridSpacing / (subGridCount + 1),
			};
		},

		snap(point) {
			if (!this.snapping) {
				return point;
			}
			const step = this.snapStep();
			return point.map((value) => Math.round(value / step) * step);
		},

		setMode(mode) {
			this.mode = mode;
			document.querySelectorAll("[data-manual-mode] .segment").forEach((button) => {
				button.classList.toggle("is-active", button.dataset.mode === mode);
			});
			this.updateCursor();
			this.syncCloseButton();
			this.draw();
		},

		toggleSnapping(force = null) {
			this.snapping = force === null ? !this.snapping : force;
			const button = $("#toggle-manual-snapping");
			button?.classList.toggle("is-active", this.snapping);
			button?.setAttribute("aria-pressed", this.snapping ? "true" : "false");
			this.draw();
		},

		zoomBy(factor) {
			const centerX = this.canvas.offsetWidth / 2;
			const centerY = this.canvas.offsetHeight / 2;
			const before = this.canvasToWorld(centerX, centerY);
			this.updateZoomLimits();
			this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
			const after = this.canvasToWorld(centerX, centerY);
			this.offsetX += (after[0] - before[0]) * this.scale;
			this.offsetY -= (after[1] - before[1]) * this.scale;
			this.saveCamera();
			this.draw();
		},

		toggleExpanded(force = null) {
			const centerWorld = this.canvasToWorld(this.canvas.offsetWidth / 2, this.canvas.offsetHeight / 2);
			this.expanded = force === null ? !this.expanded : force;
			const panel = $("#cases-panel");
			const button = $("#manual-editor-expand");
			panel?.classList.toggle("editor-expanded", this.expanded);
			document.body.classList.toggle("manual-editor-is-expanded", this.expanded);
			button?.setAttribute("aria-pressed", this.expanded ? "true" : "false");
			if (button) {
				button.textContent = this.expanded ? "Collapse" : "Expand";
			}
			requestAnimationFrame(() => {
				this.resize();
				this.offsetX = this.canvas.offsetWidth / 2 - centerWorld[0] * this.scale;
				this.offsetY = this.canvas.offsetHeight / 2 + centerWorld[1] * this.scale;
				this.saveCamera();
				this.draw();
			});
		},

		setSolveStatus(message) {
			const status = $("#manual-solve-status");
			if (status) {
				status.textContent = message || "";
			}
		},

		setSaveStatus(message) {
			const status = $("#manual-save-status");
			if (status) {
				status.textContent = message || "";
			}
		},

		setStatus(message) {
			this.setSolveStatus(message);
		},

		syncCloseButton() {
			const button = $("#close-manual-polygon");
			if (!button) {
				return;
			}
			const current = this.currentCase();
			const polygon = current && this.activePolygon !== null ? current.polygons[this.activePolygon] : null;
			const canClose = Boolean(polygon && polygon.length >= 3);
			button.classList.toggle("is-disabled", !canClose);
			button.disabled = !canClose;
			button.setAttribute("aria-disabled", canClose ? "false" : "true");
		},

		showCloseHint() {
			const button = $("#close-manual-polygon");
			if (!button) {
				return;
			}
			button.classList.add("show-tooltip");
			clearTimeout(button._tooltipTimer);
			button._tooltipTimer = setTimeout(() => button.classList.remove("show-tooltip"), 1800);
		},

		updateLabelDirections(animate = false) {
			const targets = {
				start: solutionDirectionAt(this.solutionPath, 0),
				target: solutionDirectionAt(this.solutionPath, 1),
			};
			if (!animate) {
				this.labelDirections = targets;
				return;
			}
			if (this.labelAnimation) {
				cancelAnimationFrame(this.labelAnimation);
			}
			let frame = 0;
			const step = () => {
				frame += 1;
				for (const key of ["start", "target"]) {
					this.labelDirections[key] = [
						this.labelDirections[key][0] + (targets[key][0] - this.labelDirections[key][0]) * 0.24,
						this.labelDirections[key][1] + (targets[key][1] - this.labelDirections[key][1]) * 0.24,
					];
				}
				this.draw();
				if (frame < 14) {
					this.labelAnimation = requestAnimationFrame(step);
				} else {
					this.labelDirections = targets;
					this.labelAnimation = null;
					this.draw();
				}
			};
			step();
		},

		selectNearestPoint(x, y) {
			const current = this.currentCase();
			if (!current) {
				return null;
			}
			const candidates = [
				{ kind: "start", point: current.start },
				{ kind: "target", point: current.target },
			];
			current.polygons.forEach((polygon, polygonIndex) => {
				polygon.forEach((point, vertexIndex) => {
					candidates.push({ kind: "vertex", point, polygonIndex, vertexIndex });
				});
			});
			let best = null;
			let bestDistance = 20;
			for (const candidate of candidates) {
				const canvasPoint = this.worldToCanvas(candidate.point);
				const distance = Math.hypot(canvasPoint.x - x, canvasPoint.y - y);
				if (distance <= bestDistance) {
					best = candidate;
					bestDistance = distance;
				}
			}
			return best;
		},

		isClosingPolygon(x, y) {
			const current = this.currentCase();
			if (!current || this.mode !== "polygon" || this.activePolygon === null) {
				return false;
			}
			const polygon = current.polygons[this.activePolygon];
			if (!polygon || polygon.length < 3) {
				return false;
			}
			const first = this.worldToCanvas(polygon[0]);
			return Math.hypot(first.x - x, first.y - y) <= 16;
		},

		pointInPolygon(world, polygon) {
			let inside = false;
			for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
				const xi = polygon[i][0];
				const yi = polygon[i][1];
				const xj = polygon[j][0];
				const yj = polygon[j][1];
				if (((yi > world[1]) !== (yj > world[1]))
					&& world[0] < ((xj - xi) * (world[1] - yi)) / (yj - yi) + xi) {
					inside = !inside;
				}
			}
			return inside;
		},

		selectPolygon(x, y) {
			const current = this.currentCase();
			if (!current) {
				return -1;
			}
			const world = this.canvasToWorld(x, y);
			for (let index = current.polygons.length - 1; index >= 0; index -= 1) {
				const polygon = current.polygons[index];
				if (polygon.length >= 3 && this.pointInPolygon(world, polygon)) {
					return index;
				}
			}
			return -1;
		},

		pointsInRect(rect) {
			return selectionPointsInRect(this.currentCase(), rect, (point) => this.worldToCanvas(point));
		},

		samePointSelection(left, right) {
			return selectionSamePoint(left, right);
		},

		mergeSelections(base, selected) {
			return mergeSelections(base, selected);
		},

		activeTouchPoints() {
			return [...this.activePointers.values()].filter((pointer) => pointer.pointerType === "touch");
		},

		startPinchGesture() {
			const touches = this.activeTouchPoints().slice(0, 2);
			if (touches.length < 2) {
				return false;
			}
			const rect = this.canvas.getBoundingClientRect();
			const midpoint = {
				x: (touches[0].clientX + touches[1].clientX) / 2 - rect.left,
				y: (touches[0].clientY + touches[1].clientY) / 2 - rect.top,
			};
			this.pinchGesture = {
				distance: Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY),
				midpoint,
			};
			this.activePoint = null;
			this.dragSelection = null;
			this.dragPolygon = null;
			this.panDrag = null;
			this.selectionRect = null;
			this.pointerStart = { kind: "pinch", clientX: midpoint.x + rect.left, clientY: midpoint.y + rect.top };
			return true;
		},

		updatePinchGesture() {
			if (!this.pinchGesture && !this.startPinchGesture()) {
				return false;
			}
			const gesture = this.pinchGesture;
			if (!gesture || gesture.distance <= 0) {
				return false;
			}
			const touches = this.activeTouchPoints().slice(0, 2);
			const distance = Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
			if (distance <= 0) {
				return false;
			}
			const rect = this.canvas.getBoundingClientRect();
			const midpoint = {
				x: (touches[0].clientX + touches[1].clientX) / 2 - rect.left,
				y: (touches[0].clientY + touches[1].clientY) / 2 - rect.top,
			};
			const before = this.canvasToWorld(midpoint.x, midpoint.y);
			this.updateZoomLimits();
			this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * (distance / gesture.distance)));
			const after = this.canvasToWorld(midpoint.x, midpoint.y);
			this.offsetX += (after[0] - before[0]) * this.scale;
			this.offsetY -= (after[1] - before[1]) * this.scale;
			this.pinchGesture = { distance, midpoint };
			this.saveCamera();
			this.draw();
			return true;
		},

		onPointerDown(event) {
			event.preventDefault();
			const current = this.currentCase();
			if (!current) {
				return;
			}
			const rect = this.canvas.getBoundingClientRect();
			const x = event.clientX - rect.left;
			const y = event.clientY - rect.top;
			this.mouseCanvas = { x, y };
			this.activePointers.set(event.pointerId, {
				pointerType: event.pointerType,
				clientX: event.clientX,
				clientY: event.clientY,
			});
			if (event.pointerType === "touch" && this.activeTouchPoints().length >= 2) {
				this.canvas.setPointerCapture(event.pointerId);
				this.startPinchGesture();
				return;
			}
			const world = this.snap(this.canvasToWorld(x, y));
			if (this.mode === "select" || (event.shiftKey && this.mode === "move")) {
				this.selectionRect = { start: { x, y }, end: { x, y } };
				this.selectionBase = [...this.selectedPoints];
				this.selectedPoint = null;
				this.draw();
				return;
			}
			if (this.mode === "polygon") {
				if (this.isClosingPolygon(x, y)) {
					this.closePolygon();
					return;
				}
				if (this.activePolygon === null) {
					current.polygons.push([]);
					this.activePolygon = current.polygons.length - 1;
				}
				current.polygons[this.activePolygon].push(world);
				this.changed();
				return;
			}
			this.activePoint = this.selectNearestPoint(x, y);
			this.selectedPoint = this.activePoint;
			if (this.activePoint) {
				if (this.selectedPoints.some((point) => this.samePointSelection(point, this.activePoint))) {
					this.dragSelection = {
						lastWorld: this.canvasToWorld(x, y),
					};
				}
				this.pointerStart = { kind: "point", clientX: event.clientX, clientY: event.clientY };
				this.canvas.setPointerCapture(event.pointerId);
				this.canvas.style.cursor = "grabbing";
				this.animateSelection();
				this.draw();
				return;
			}
			const polygonIndex = this.selectPolygon(x, y);
			if (polygonIndex !== -1) {
				this.dragPolygon = {
					index: polygonIndex,
					lastWorld: this.canvasToWorld(x, y),
				};
				this.pointerStart = { kind: "polygon", clientX: event.clientX, clientY: event.clientY };
				this.canvas.setPointerCapture(event.pointerId);
				this.canvas.style.cursor = "grabbing";
				this.draw();
				return;
			}
			this.selectedPoint = null;
			this.panDrag = { x, y };
			this.pointerStart = { kind: "pan", clientX: event.clientX, clientY: event.clientY };
			this.canvas.setPointerCapture(event.pointerId);
		},

		onPointerMove(event) {
			event.preventDefault();
			if (this.activePointers.has(event.pointerId)) {
				this.activePointers.set(event.pointerId, {
					pointerType: event.pointerType,
					clientX: event.clientX,
					clientY: event.clientY,
				});
			}
			const current = this.currentCase();
			const rect = this.canvas.getBoundingClientRect();
			const x = event.clientX - rect.left;
			const y = event.clientY - rect.top;
			this.mouseCanvas = { x, y };
			if (!current) {
				return;
			}
			if (this.pinchGesture || (event.pointerType === "touch" && this.activeTouchPoints().length >= 2)) {
				if (this.updatePinchGesture()) {
					return;
				}
			}
			this.updateCursor();
			if (this.selectionRect) {
				this.selectionRect.end = { x, y };
				this.selectedPoints = this.mergeSelections(this.selectionBase, this.pointsInRect(this.selectionRect));
				this.draw();
				return;
			}
			if (this.panDrag) {
				this.offsetX += x - this.panDrag.x;
				this.offsetY += y - this.panDrag.y;
				this.panDrag = { x, y };
				this.saveCamera();
				this.draw();
				return;
			}
			if (this.dragSelection) {
				const world = this.snap(this.canvasToWorld(x, y));
				const dx = world[0] - this.dragSelection.lastWorld[0];
				const dy = world[1] - this.dragSelection.lastWorld[1];
				for (const selected of this.selectedPoints) {
					const next = this.snap([selected.point[0] + dx, selected.point[1] + dy]);
					if (selected.kind === "start") {
						current.start = next;
					} else if (selected.kind === "target") {
						current.target = next;
					} else if (selected.kind === "vertex") {
						current.polygons[selected.polygonIndex][selected.vertexIndex] = next;
					}
					selected.point = next;
				}
				this.dragSelection.lastWorld = world;
				this.changed();
				return;
			}
			if (this.dragPolygon) {
				const world = this.snap(this.canvasToWorld(x, y));
				const dx = world[0] - this.dragPolygon.lastWorld[0];
				const dy = world[1] - this.dragPolygon.lastWorld[1];
				current.polygons[this.dragPolygon.index] = current.polygons[this.dragPolygon.index]
					.map(([px, py]) => this.snap([px + dx, py + dy]));
				for (const selected of this.selectedPoints) {
					if (selected.kind === "vertex" && selected.polygonIndex === this.dragPolygon.index) {
						selected.point = current.polygons[selected.polygonIndex][selected.vertexIndex];
					}
				}
				this.dragPolygon.lastWorld = world;
				this.changed();
				return;
			}
			if (!this.activePoint) {
				if (this.mode === "polygon") {
					this.draw();
				}
				return;
			}
			const world = this.snap(this.canvasToWorld(x, y));
			if (this.activePoint.kind === "start") {
				current.start = world;
			} else if (this.activePoint.kind === "target") {
				current.target = world;
			} else if (this.activePoint.kind === "vertex") {
				current.polygons[this.activePoint.polygonIndex][this.activePoint.vertexIndex] = world;
				this.activePoint.point = world;
			}
			this.changed();
		},

		onWheel(event) {
			event.preventDefault();
			const rect = this.canvas.getBoundingClientRect();
			const before = this.canvasToWorld(event.clientX - rect.left, event.clientY - rect.top);
			const delta = Math.max(-80, Math.min(80, event.deltaY));
			const factor = Math.exp(-delta * (event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? 0.0035 : 0.1));
			this.updateZoomLimits();
			this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
			const after = this.canvasToWorld(event.clientX - rect.left, event.clientY - rect.top);
			this.offsetX += (after[0] - before[0]) * this.scale;
			this.offsetY -= (after[1] - before[1]) * this.scale;
			this.saveCamera();
			this.draw();
		},

		updateCursor() {
			if (!this.canvas) {
				return;
			}
			if (this.mode === "polygon") {
				this.canvas.style.cursor = this.isClosingPolygon(this.mouseCanvas.x, this.mouseCanvas.y) ? "copy" : "crosshair";
				return;
			}
			if (this.mode === "select") {
				this.canvas.style.cursor = "crosshair";
				return;
			}
			if (this.selectNearestPoint(this.mouseCanvas.x, this.mouseCanvas.y)) {
				this.canvas.style.cursor = "grab";
				return;
			}
			if (this.selectPolygon(this.mouseCanvas.x, this.mouseCanvas.y) !== -1) {
				this.canvas.style.cursor = "grab";
				return;
			}
			this.canvas.style.cursor = "all-scroll";
		},

		onKeyDown(event) {
			if (keybinds.capturePending(event)) {
				return;
			}
			if (!$("#keybind-modal")?.classList.contains("is-hidden")) {
				if (event.key === "Escape") {
					event.preventDefault();
					keybinds.close();
				}
				return;
			}
			if (event.target?.closest?.("input, textarea, select, [contenteditable='true']")) {
				return;
			}
			if (!$("#cases-panel")?.classList.contains("is-active")) {
				return;
			}
			if (keybinds.matches(event, "toggleSnap")) {
				event.preventDefault();
				this.toggleSnapping();
			} else if (keybinds.matches(event, "deleteSelection") || event.key === "Backspace" || event.key === "Delete") {
				event.preventDefault();
				this.deleteSelection();
			} else if (keybinds.matches(event, "clearSelection")) {
				event.preventDefault();
				this.clearSelection();
			} else if (keybinds.matches(event, "fitInstance")) {
				event.preventDefault();
				this.frameCurrentCase();
			} else if (keybinds.matches(event, "toggleGrid")) {
				event.preventDefault();
				this.toggleLayer("grid");
			} else if (keybinds.matches(event, "togglePath")) {
				event.preventDefault();
				this.toggleLayer("solution");
			} else if (keybinds.matches(event, "toggleDecomposition")) {
				event.preventDefault();
				this.toggleLayer("decomposition");
			} else if (keybinds.matches(event, "toggleLabels")) {
				event.preventDefault();
				this.toggleLayer("labels");
			} else if (keybinds.matches(event, "closePolygon")) {
				event.preventDefault();
				this.closePolygon();
			} else if (event.key === "Escape") {
				if (this.expanded) {
					event.preventDefault();
					this.toggleExpanded(false);
					return;
				}
				this.activePolygon = null;
				this.selectedPoint = null;
				this.selectedPoints = [];
				this.selectionRect = null;
				this.setMode("move");
			} else if (event.key === "1") {
				this.setMode("move");
			} else if (event.key === "2") {
				this.setMode("polygon");
			} else if (event.key === "3") {
				this.setMode("select");
			}
		},

		finishSelection() {
			const selected = this.pointsInRect(this.selectionRect);
			const moved = Math.hypot(
				this.selectionRect.end.x - this.selectionRect.start.x,
				this.selectionRect.end.y - this.selectionRect.start.y,
			) > 4;
			if (!moved && selected.length === 0) {
				this.selectedPoints = [];
				this.selectedPoint = null;
				this.selectionRect = null;
				this.selectionBase = [];
				this.draw();
				return;
			}
			this.selectedPoints = this.mergeSelections(this.selectionBase, selected);
			this.selectedPoint = this.selectedPoints[0] || null;
			this.selectionRect = null;
			this.selectionBase = [];
			this.animateSelection();
			this.draw();
		},

		closePolygon() {
			const current = this.currentCase();
			if (!current || this.activePolygon === null) {
				this.showCloseHint();
				return;
			}
			if ((current.polygons[this.activePolygon] || []).length < 3) {
				this.showCloseHint();
				return;
			}
			this.activePolygon = null;
			this.syncCloseButton();
			this.changed();
			this.canvas?.focus({ preventScroll: true });
		},

		deleteSelection() {
			const current = this.currentCase();
			const selected = this.selectedPoints;
			if (!current || selected.length === 0) {
				return;
			}
			const byPolygon = new Map();
			for (const point of selected) {
				if (point.kind !== "vertex") {
					continue;
				}
				const vertices = byPolygon.get(point.polygonIndex) || [];
				vertices.push(point.vertexIndex);
				byPolygon.set(point.polygonIndex, vertices);
			}
			[...byPolygon.entries()]
				.sort(([left], [right]) => right - left)
				.forEach(([polygonIndex, vertexIndices]) => {
					const polygon = current.polygons[polygonIndex];
					[...new Set(vertexIndices)].sort((left, right) => right - left).forEach((vertexIndex) => {
						polygon.splice(vertexIndex, 1);
					});
					if (polygon.length < 3) {
						current.polygons.splice(polygonIndex, 1);
					}
				});
			if (this.activePolygon !== null && this.activePolygon >= current.polygons.length) {
				this.activePolygon = null;
			}
			this.activePoint = null;
			this.selectedPoint = null;
			this.selectedPoints = [];
			this.changed();
		},

		clearSelection() {
			this.selectedPoint = null;
			this.selectedPoints = [];
			this.selectionBase = [];
			this.selectionRect = null;
			this.draw();
		},

		cancelPendingSolution() {
			this.solutionRevision += 1;
			this.solutionAbort?.abort();
			this.solutionAbort = null;
			if (this.solutionTimer) {
				clearTimeout(this.solutionTimer);
				this.solutionTimer = null;
			}
			if (this.solutionFrame) {
				cancelAnimationFrame(this.solutionFrame);
				this.solutionFrame = null;
			}
		},

		changed() {
			this.cancelPendingSolution();
			this.solutionStale = Boolean(this.solutionPath);
			this.updateLabelDirections(false);
			this.syncCloseButton();
			updateManualCaseListMetadata();
			this.draw();
			this.scheduleSolve();
			scheduleManualAutosave();
		},

		scheduleSolve() {
			const current = this.currentCase();
			if (!current) {
				return;
			}
			this.draw();
			if (this.solutionStale) {
				this.setStatus("Updating solution...");
			}
			if (this.solutionTimer) {
				clearTimeout(this.solutionTimer);
				this.solutionTimer = null;
			}
			if (this.solutionFrame) {
				cancelAnimationFrame(this.solutionFrame);
				this.solutionFrame = null;
			}
			this.solutionTimer = setTimeout(() => {
				this.solutionTimer = null;
				this.fetchSolution(cloneCaseData(this.currentCase()), this.solutionRevision);
			}, 0);
		},

		async fetchSolution(caseData, revision = this.solutionRevision) {
			if (!caseData) {
				return;
			}
			if (this.solutionAbort) {
				this.solutionAbort.abort();
			}
			this.solutionAbort = new AbortController();
			const signal = this.solutionAbort.signal;
			if (caseData.polygons.length === 0) {
				if (revision !== this.solutionRevision) {
					return;
				}
				this.solutionPath = [caseData.start, caseData.target];
				this.solutionStale = false;
				this.updateLabelDirections(true);
				this.setStatus("Solution: exact, 0 calls");
				this.draw();
				return;
			}
			this.setStatus(editorSolverState.module ? "Solving..." : editorSolverState.failed ? "WASM solver unavailable." : "Loading solver...");
			try {
				await loadEditorWasm();
				if (signal.aborted || revision !== this.solutionRevision) {
					return;
				}
				if (!editorSolverState.module) {
					if (revision !== this.solutionRevision) {
						return;
					}
					this.solutionPath = null;
					this.solutionStale = false;
					this.updateLabelDirections(true);
					this.setStatus("WASM solver unavailable.");
					this.draw();
					return;
				}
				let pieceGroups = null;
				const solveStarted = performance.now();
				if (!caseData.polygons.every(polygonIsConvex)) {
					const geometry = await loadEditorGeometry();
					if (signal.aborted || revision !== this.solutionRevision) {
						return;
					}
					pieceGroups = geometry
						? caseData.polygons.map((polygon) => (
						geometry.partition.convexPartition(polygon.map(([x, y]) => new geometry.vector.Vector2(x, y)))
							.filter((piece) => piece.length >= 3)
							.map((piece) => piece.map((point) => [point.x, point.y]))
						))
						: caseData.polygons.map(convexDecomposition);
				}
				const wasmResult = await solveEditorWasmAsync(caseData, pieceGroups, signal);
				const solveWallSeconds = (performance.now() - solveStarted) / 1000;
				if (wasmResult) {
					if (revision !== this.solutionRevision) {
						return;
					}
					this.solutionPath = wasmResult.path;
					this.solutionStale = false;
					this.updateLabelDirections(true);
					this.setStatus(`Solution: ${wasmResult.exact ? "exact" : "approximate"}, ${wasmResult.calls} calls, ${formatSeconds(Math.max(wasmResult.seconds || 0, solveWallSeconds))} via WASM`);
					this.draw();
					return;
				}
				if (revision !== this.solutionRevision) {
					return;
				}
				this.solutionPath = null;
				this.solutionStale = false;
				this.updateLabelDirections(true);
				this.setStatus("WASM solver could not solve this case.");
				this.draw();
			} catch (error) {
				if (error.name !== "AbortError") {
					this.setStatus(error.message);
				}
			}
		},

		drawGrid() {
			const width = this.canvas.offsetWidth;
			const height = this.canvas.offsetHeight;
			const ctx = this.ctx;
			ctx.fillStyle = cssVar("--editor-bg") || "#121417";
			ctx.fillRect(0, 0, width, height);
			if (!this.layers.grid) {
				return;
			}
			const { gridSpacing, subGridCount } = this.gridMetrics();
			const left = (0 - this.offsetX) / this.scale;
			const right = (width - this.offsetX) / this.scale;
			const top = (this.offsetY - 0) / this.scale;
			const bottom = (this.offsetY - height) / this.scale;
			const drawWorldLine = (a, b, color) => {
				const start = this.worldToCanvas(a);
				const end = this.worldToCanvas(b);
				ctx.strokeStyle = color;
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(start.x, start.y);
				ctx.lineTo(end.x, end.y);
				ctx.stroke();
			};
			const firstX = Math.floor(left / gridSpacing) * gridSpacing;
			const firstY = Math.floor(bottom / gridSpacing) * gridSpacing;
			for (let x = firstX; x <= right + gridSpacing; x += gridSpacing) {
				drawWorldLine([x, bottom], [x, top], cssVar("--editor-grid-major") || "#515a67");
				for (let index = 0; index < subGridCount; index += 1) {
					const subX = x + (gridSpacing * (index + 1)) / (subGridCount + 1);
					drawWorldLine([subX, bottom], [subX, top], cssVar("--editor-grid-minor") || "#2a2f38");
				}
			}
			for (let y = firstY; y <= top + gridSpacing; y += gridSpacing) {
				drawWorldLine([left, y], [right, y], cssVar("--editor-grid-major") || "#515a67");
				for (let index = 0; index < subGridCount; index += 1) {
					const subY = y + (gridSpacing * (index + 1)) / (subGridCount + 1);
					drawWorldLine([left, subY], [right, subY], cssVar("--editor-grid-minor") || "#2a2f38");
				}
			}
			const origin = this.worldToCanvas([0, 0]);
			ctx.strokeStyle = cssVar("--editor-axis") || "#9aa3ad";
			ctx.beginPath();
			ctx.moveTo(0, origin.y);
			ctx.lineTo(width, origin.y);
			ctx.moveTo(origin.x, 0);
			ctx.lineTo(origin.x, height);
			ctx.stroke();
		},

		drawPoint(point, color, label = "", labelDirection = [1, -1], options = {}) {
			const ctx = this.ctx;
			const canvasPoint = this.worldToCanvas(point);
			const radius = options.radius ?? this.pointRadius(label ? "endpoint" : "vertex");
			ctx.fillStyle = color;
			ctx.beginPath();
			ctx.arc(canvasPoint.x, canvasPoint.y, radius, 0, Math.PI * 2);
			ctx.fill();
			if (label) {
				const length = Math.hypot(labelDirection[0], labelDirection[1]) || 1;
				const offset = Math.max(13, radius + 9);
				const offsetX = (labelDirection[0] / length) * offset;
				const offsetY = -(labelDirection[1] / length) * offset;
				ctx.fillStyle = "#f8fafc";
				ctx.font = "12px system-ui";
				ctx.textAlign = "center";
				ctx.textBaseline = "middle";
				ctx.fillText(label, canvasPoint.x + offsetX, canvasPoint.y + offsetY);
				ctx.textAlign = "start";
				ctx.textBaseline = "alphabetic";
			}
		},

		drawSelectedRing(point) {
			const ctx = this.ctx;
			const canvasPoint = this.worldToCanvas(point);
			const radius = this.pointRadius("vertex") + 5;
			const rotation = ((performance.now() - this.selectionSpinStarted) / 700) * Math.PI * 2;
			ctx.save();
			ctx.translate(canvasPoint.x, canvasPoint.y);
			ctx.rotate(rotation);
			ctx.strokeStyle = "#f8fafc";
			ctx.lineWidth = 1.5;
			ctx.setLineDash([5, 4]);
			ctx.beginPath();
			ctx.arc(0, 0, radius, 0, Math.PI * 2);
			ctx.stroke();
			ctx.restore();
		},

		animateSelection() {
			if (this.selectionAnimationRunning) {
				return;
			}
			this.selectionAnimationRunning = true;
			const tick = () => {
				if (this.selectedPoints.length === 0) {
					this.selectionAnimationRunning = false;
					return;
				}
				this.draw();
				requestAnimationFrame(tick);
			};
			requestAnimationFrame(tick);
		},

		drawDecomposition(polygon, color) {
			const pieces = convexDecomposition(polygon);
			if (pieces.length <= 1) {
				return;
			}
			const ctx = this.ctx;
			ctx.save();
			ctx.strokeStyle = color;
			ctx.lineWidth = 1;
			ctx.globalAlpha = 0.82;
			ctx.setLineDash([5, 5]);
			for (const piece of pieces) {
				ctx.beginPath();
				piece.forEach((point, pointIndex) => {
					const canvasPoint = this.worldToCanvas(point);
					if (pointIndex === 0) {
						ctx.moveTo(canvasPoint.x, canvasPoint.y);
					} else {
						ctx.lineTo(canvasPoint.x, canvasPoint.y);
					}
				});
				ctx.closePath();
				ctx.stroke();
			}
			ctx.restore();
		},

		draw() {
			drawCanvasScene(this);
		},
	};

	return manualEditor;
}
