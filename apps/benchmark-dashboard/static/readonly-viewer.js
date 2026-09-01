import { drawCanvasScene } from "./canvas-renderer.js";
import { cloneCaseData } from "./case-data.js";
import { escapeHTML } from "./dom.js";
import { convexDecomposition, polygonIsConvex } from "./editor-geometry.js";
import { editorSolverState, loadEditorGeometry, loadEditorWasm, solveEditorWasmAsync } from "./editor-solver.js?v=intersections-2026-09-01-length";
import { formatLength, formatSeconds } from "./format.js";
import { bindTapZoom } from "./ui-utils.js";

export function renderCanvasPlaceholder(root, text = "Loading instances...") {
	root.innerHTML = `<div class="missing-preview">${escapeHTML(text)}</div>`;
	root.classList.remove("is-hidden");
}

export function readonlyInstanceDetail(title) {
	return `
		<figure class="instance-detail readonly-instance-detail">
			<div class="case-toolbar readonly-toolbar">
				<div class="readonly-toolbar-main">
					<button class="secondary tap-zoom-button" type="button" data-readonly-zoom-out>-</button>
					<button class="secondary tap-zoom-button" type="button" data-readonly-zoom-in>+</button>
					<button class="secondary" type="button" data-readonly-fit>Fit</button>
					<div class="editor-layer-toggles" aria-label="Viewer layers">
						<button class="secondary is-active" data-readonly-layer="grid" type="button" aria-pressed="true">Grid</button>
						<button class="secondary is-active" data-readonly-layer="solution" type="button" aria-pressed="true">Path</button>
						<button class="secondary is-active" data-readonly-layer="decomposition" type="button" aria-pressed="true">Decomp</button>
						<button class="secondary is-active" data-readonly-layer="labels" type="button" aria-pressed="true">Labels</button>
					</div>
				</div>
				<button class="secondary readonly-edit-button" type="button" data-edit-instance>Edit Instance</button>
			</div>
			<canvas class="readonly-instance-canvas" width="960" height="620" aria-label="${escapeHTML(title)}"></canvas>
			<div class="editor-status-row"><span data-readonly-status></span></div>
		</figure>
	`;
}

export function createReadonlyInstanceViewer(canvas, caseData, options = {}) {
	const manualEditor = options.manualEditor;
	const viewer = {
		canvas,
		ctx: canvas.getContext("2d"),
		caseData: cloneCaseData(caseData),
		scale: 70,
		minScale: 0.1,
		maxScale: 50000,
		offsetX: 0,
		offsetY: 0,
		activePolygon: null,
		selectedPoints: [],
		selectionRect: null,
		mouseCanvas: { x: 0, y: 0 },
		solutionPath: null,
		solutionStale: false,
		solutionRevision: 0,
		solutionAbort: null,
		labelDirections: {
			start: [1, 0],
			target: [-1, 0],
		},
		layers: {
			grid: options.grid ?? true,
			solution: options.solution ?? true,
			decomposition: options.decomposition ?? true,
			labels: options.labels ?? true,
		},
		currentCase() {
			return this.caseData;
		},
		saveCamera() { },
		setStatus(message) {
			if (options.status) {
				options.status.textContent = message || "";
			}
		},
		resize: manualEditor.resize,
		caseBounds: manualEditor.caseBounds,
		updateZoomLimits: manualEditor.updateZoomLimits,
		frameCurrentCase: manualEditor.frameCurrentCase,
		pointRadius: manualEditor.pointRadius,
		worldToCanvas: manualEditor.worldToCanvas,
		canvasToWorld: manualEditor.canvasToWorld,
		gridMetrics: manualEditor.gridMetrics,
		drawGrid: manualEditor.drawGrid,
		drawPoint: manualEditor.drawPoint,
		drawDecomposition: manualEditor.drawDecomposition,
		updateLabelDirections: manualEditor.updateLabelDirections,
		draw() {
			drawCanvasScene(this);
		},
		zoomAtCanvasPoint(factor, x, y) {
			const before = this.canvasToWorld(x, y);
			this.updateZoomLimits();
			this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
			const after = this.canvasToWorld(x, y);
			this.offsetX += (after[0] - before[0]) * this.scale;
			this.offsetY -= (after[1] - before[1]) * this.scale;
			this.draw();
		},
		zoomBy(factor) {
			const centerX = this.canvas.offsetWidth / 2;
			const centerY = this.canvas.offsetHeight / 2;
			this.zoomAtCanvasPoint(factor, centerX, centerY);
		},
		cancelPendingSolution() {
			this.solutionRevision += 1;
			this.solutionAbort?.abort();
			this.solutionAbort = null;
		},
		destroy() {
			this.cancelPendingSolution();
			if (this.labelAnimation) {
				cancelAnimationFrame(this.labelAnimation);
				this.labelAnimation = null;
			}
		},
		async fetchSolution(caseData, revision = this.solutionRevision) {
			if (!caseData) {
				return;
			}
			if (caseData.polygons.length === 0) {
				if (revision !== this.solutionRevision) {
					return;
				}
				this.solutionPath = [caseData.start, caseData.target];
				const length = Math.hypot(caseData.target[0] - caseData.start[0], caseData.target[1] - caseData.start[1]);
				this.solutionStale = false;
				this.updateLabelDirections(true);
				this.setStatus(`Solution: exact, length ${formatLength(length)}, 0 calls`);
				this.draw();
				return;
			}
			if (this.solutionAbort) {
				this.solutionAbort.abort();
			}
			this.solutionAbort = new AbortController();
			const signal = this.solutionAbort.signal;
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
				if (revision !== this.solutionRevision) {
					return;
				}
				this.solutionPath = wasmResult?.path || null;
				this.solutionStale = false;
				this.updateLabelDirections(true);
				this.setStatus(wasmResult
					? `Solution: ${wasmResult.exact ? "exact" : "approximate"}, length ${formatLength(wasmResult.length)}, ${wasmResult.calls} calls, ${formatSeconds(Math.max(wasmResult.seconds || 0, solveWallSeconds))} via WASM`
					: "WASM solver could not solve this case.");
				this.draw();
			} catch (error) {
				if (error.name !== "AbortError") {
					this.setStatus(error.message);
				}
			}
		},
	};

	viewer.resize();
	viewer.frameCurrentCase();
	const resizeObserver = new ResizeObserver(() => {
		viewer.resize();
		viewer.frameCurrentCase();
	});
	resizeObserver.observe(canvas);
	const wheelHandler = (event) => {
		if (!options.interactive) {
			return;
		}
		event.preventDefault();
		const rect = canvas.getBoundingClientRect();
		const delta = Math.max(-80, Math.min(80, event.deltaY));
		const factor = Math.exp(-delta * (event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? 0.0035 : 0.1));
		viewer.zoomAtCanvasPoint(factor, event.clientX - rect.left, event.clientY - rect.top);
	};
	canvas.addEventListener("wheel", wheelHandler, { passive: false });
	const cleanup = [
		() => resizeObserver.disconnect(),
		() => canvas.removeEventListener("wheel", wheelHandler),
	];
	if (options.interactive) {
		let pan = null;
		const activePointers = new Map();
		let pinch = null;
		const touchPointers = () => [...activePointers.values()].filter((pointer) => pointer.pointerType === "touch");
		const startPinch = () => {
			const touches = touchPointers().slice(0, 2);
			if (touches.length < 2) {
				pinch = null;
				return false;
			}
			const rect = canvas.getBoundingClientRect();
			pinch = {
				distance: Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY),
				x: (touches[0].clientX + touches[1].clientX) / 2 - rect.left,
				y: (touches[0].clientY + touches[1].clientY) / 2 - rect.top,
			};
			pan = null;
			return true;
		};
		const updatePinch = () => {
			const touches = touchPointers().slice(0, 2);
			if (touches.length < 2 || !pinch) {
				return false;
			}
			const distance = Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
			if (distance <= 0 || pinch.distance <= 0) {
				return false;
			}
			const rect = canvas.getBoundingClientRect();
			const x = (touches[0].clientX + touches[1].clientX) / 2 - rect.left;
			const y = (touches[0].clientY + touches[1].clientY) / 2 - rect.top;
			viewer.zoomAtCanvasPoint(distance / pinch.distance, x, y);
			pinch = { distance, x, y };
			return true;
		};
		const pointerDownHandler = (event) => {
			event.preventDefault();
			activePointers.set(event.pointerId, {
				pointerType: event.pointerType,
				clientX: event.clientX,
				clientY: event.clientY,
			});
			if (event.pointerType === "touch" && touchPointers().length >= 2 && startPinch()) {
				canvas.setPointerCapture(event.pointerId);
				return;
			}
			pan = { x: event.clientX, y: event.clientY };
			canvas.setPointerCapture(event.pointerId);
			canvas.style.cursor = "grabbing";
		};
		const pointerMoveHandler = (event) => {
			if (activePointers.has(event.pointerId)) {
				activePointers.set(event.pointerId, {
					pointerType: event.pointerType,
					clientX: event.clientX,
					clientY: event.clientY,
				});
			}
			if (pinch || (event.pointerType === "touch" && touchPointers().length >= 2)) {
				if (!pinch) {
					startPinch();
				}
				updatePinch();
				return;
			}
			if (!pan) {
				return;
			}
			viewer.offsetX += event.clientX - pan.x;
			viewer.offsetY += event.clientY - pan.y;
			pan = { x: event.clientX, y: event.clientY };
			viewer.draw();
		};
		const finishPan = (event) => {
			activePointers.delete(event.pointerId);
			if (pinch && touchPointers().length < 2) {
				pinch = null;
			}
			pan = null;
			canvas.style.cursor = "grab";
		};
		canvas.addEventListener("pointerdown", pointerDownHandler);
		canvas.addEventListener("pointermove", pointerMoveHandler);
		canvas.addEventListener("pointerup", finishPan);
		canvas.addEventListener("pointercancel", finishPan);
		cleanup.push(
			() => canvas.removeEventListener("pointerdown", pointerDownHandler),
			() => canvas.removeEventListener("pointermove", pointerMoveHandler),
			() => canvas.removeEventListener("pointerup", finishPan),
			() => canvas.removeEventListener("pointercancel", finishPan),
		);
	}
	const destroy = viewer.destroy;
	viewer.destroy = () => {
		destroy.call(viewer);
		while (cleanup.length > 0) {
			cleanup.pop()();
		}
	};
	if (options.solve) {
		viewer.fetchSolution(viewer.caseData);
	}
	return viewer;
}

export function setupReadonlyInstanceDetail(root, caseData, manualEditor) {
	const canvas = root.querySelector(".readonly-instance-canvas");
	if (!canvas) {
		return null;
	}
	const viewer = createReadonlyInstanceViewer(canvas, caseData, {
		interactive: true,
		manualEditor,
		solve: true,
		status: root.querySelector("[data-readonly-status]"),
	});
	const zoomOut = root.querySelector("[data-readonly-zoom-out]");
	const zoomIn = root.querySelector("[data-readonly-zoom-in]");
	bindTapZoom(zoomOut, () => viewer.zoomBy(1 / 1.2));
	bindTapZoom(zoomIn, () => viewer.zoomBy(1.2));
	root.querySelector("[data-readonly-fit]")?.addEventListener("click", () => viewer.frameCurrentCase());
	root.querySelectorAll("[data-readonly-layer]").forEach((button) => {
		button.addEventListener("click", () => {
			const layer = button.dataset.readonlyLayer;
			viewer.layers[layer] = !viewer.layers[layer];
			button.classList.toggle("is-active", viewer.layers[layer]);
			button.setAttribute("aria-pressed", viewer.layers[layer] ? "true" : "false");
			viewer.draw();
		});
	});
	return viewer;
}
