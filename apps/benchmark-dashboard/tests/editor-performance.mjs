import { chromium, webkit } from "@playwright/test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const baseUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8017";
const browser = await (process.env.TEST_BROWSER === "webkit" ? webkit : chromium).launch({ headless: true });
try {
	const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
	const errors = [];
	page.on("pageerror", error => errors.push(error.message));
	const modals = await readFile(new URL("../templates/partials/modals.html", import.meta.url), "utf8");
	await page.route(`${baseUrl}/performance-harness`, route => route.fulfill({ contentType: "text/html", body: `<link rel="stylesheet" href="/static/base.css"><link rel="stylesheet" href="/static/editor.css"><link rel="stylesheet" href="/static/overlays.css"><canvas id="editor" style="width:600px;height:400px"></canvas><button id="manual-satellite-button"></button><button id="manual-refit-background"></button>${modals}` }));
	await page.route("https://services.arcgisonline.com/**", route => route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="green"/></svg>' }));
	await page.goto(`${baseUrl}/performance-harness`);
	const pixel = await page.evaluate(async () => {
		const { createManualEditor } = await import("/static/manual-editor.js");
		const current = { start: [-1, 0], target: [1, 0], polygons: [] };
		const editor = createManualEditor({ $: selector => document.querySelector(selector), state: { manualCases: [current], manualCaseIndex: 0 }, cssVar: () => "#000000", scheduleManualAutosave() {} });
		editor.canvas = document.querySelector("#editor");
		editor.ctx = editor.canvas.getContext("2d");
		editor.layers.grid = false;
		editor.resize();
		const reference = document.createElement("canvas");
		reference.width = reference.height = 10;
		reference.getContext("2d").fillStyle = "#ff0000";
		reference.getContext("2d").fillRect(0, 0, 10, 10);
		editor.setBackground(reference.toDataURL(), 10, 10);
		editor.draw();
		await editor.backgroundImage.decode();
		editor.draw();
		const pixel = [...editor.ctx.getImageData(600, 300, 1, 1).data];
		const originalFetchSolution = editor.fetchSolution;
		current.polygons = Array.from({ length: 48 }, () => [[0, 0], [1, 0], [0, 1]]);
		await new Promise(resolve => {
			editor.fetchSolution = resolve;
			editor.scheduleSolve();
		});
		editor.fetchSolution = originalFetchSolution;
		current.polygons = [];
		const { createReadonlyInstanceViewer } = await import("/static/readonly-viewer.js");
		const viewerCanvas = document.createElement("canvas");
		viewerCanvas.style.cssText = "width:600px;height:400px";
		document.body.append(viewerCanvas);
		const viewer = createReadonlyInstanceViewer(viewerCanvas, current, { manualEditor: editor, solve: false });
		viewer.destroy?.();
		window.editor = editor;
		window.current = current;
		editor.changed = () => editor.draw();
		const { createSatelliteMap } = await import("/static/satellite-map.js");
		window.map = createSatelliteMap({ $: selector => document.querySelector(selector), manualEditor: editor, scheduleManualAutosave() {} });
		window.map.init();
		window.tileImages = [];
		const OriginalImage = window.Image;
		window.Image = class extends OriginalImage {
			constructor(...args) { super(...args); window.tileImages.push(this); }
		};
		window.map.open();
		return pixel;
	});
	assert.ok(pixel[0] > 100 && pixel[1] === 0, `Background was erased: ${pixel}`);
	await page.waitForFunction(() => document.querySelector("#satellite-map").width > 100);
	for (let index = 0; index < 16; index += 1) {
		await page.locator("#satellite-longitude").fill(String(-46 + index));
		await page.locator("#satellite-go").click();
		await page.evaluate(() => new Promise(requestAnimationFrame));
	}
	const tiles = await page.evaluate(() => ({ created: window.tileImages.length, retained: window.tileImages.filter(image => image.onload).length }));
	assert.ok(tiles.created > 128, JSON.stringify(tiles));
	assert.ok(tiles.retained <= 128, JSON.stringify(tiles));
	const anchoring = await page.evaluate(() => {
		const canvas = document.querySelector("#satellite-map"), rect = canvas.getBoundingClientRect();
		const pixels = (latitude, longitude, z) => {
			const size = 256 * 2 ** z;
			return [(longitude + 180) / 360 * size, (1 - Math.asinh(Math.tan(latitude * Math.PI / 180)) / Math.PI) / 2 * size];
		};
		const read = z => pixels(Number(document.querySelector("#satellite-latitude").value), Number(document.querySelector("#satellite-longitude").value), z);
		const before = read(19), dx = 80, dy = -40;
		canvas.dispatchEvent(new WheelEvent("wheel", { clientX: rect.left + rect.width / 2 + dx, clientY: rect.top + rect.height / 2 + dy, deltaY: -1, bubbles: true, cancelable: true }));
		const after = read(20);
		return [after[0] - 2 * before[0] - dx, after[1] - 2 * before[1] - dy];
	});
	assert.ok(anchoring.every(value => Math.abs(value) < 0.3), `Map anchor drift: ${anchoring}`);
	for (let polygon = 0; polygon < 2; polygon += 1) {
		for (const [x, y] of [[100, 100], [200, 100], [150, 200]]) {
			await page.locator("#satellite-map").click({ position: { x: x + polygon * 200, y } });
		}
		await page.locator("#satellite-finish").click();
	}
	assert.equal(await page.evaluate(() => window.current.polygons.length), 2);
	assert.equal(await page.locator("#satellite-modal").isVisible(), true);
	const geometryBefore = await page.evaluate(() => JSON.stringify(window.current.polygons));
	await page.locator("#satellite-instance-in").click();
	await page.locator("#satellite-zoom-out").click();
	assert.equal(await page.evaluate(() => JSON.stringify(window.current.polygons)), geometryBefore);
	await page.locator("#satellite-tool").selectOption("start");
	await page.locator("#satellite-map").click({ position: { x: 100, y: 300 } });
	assert.notDeepEqual(await page.evaluate(() => window.current.start), [-1, 0]);
	await page.waitForFunction(() => window.tileImages.filter(image => image.onload).every(image => image.complete));
	await page.locator("#satellite-use-background").click();
	assert.equal(await page.locator("#satellite-modal").isVisible(), false);
	assert.match(await page.evaluate(() => window.current.background.data_url), /^data:image\/jpeg/);
	const imageMoved = await page.evaluate(() => {
		const editor = window.editor;
		const before = [...window.current.background.bounds];
		editor.toggleBackgroundEditing();
		editor.canvas.setPointerCapture = () => {};
		editor.onPointerDown({ preventDefault() {}, clientX: 100, clientY: 100, pointerId: 1 });
		editor.onPointerMove({ preventDefault() {}, clientX: 120, clientY: 110, pointerId: 1 });
		editor.backgroundDrag = null;
		editor.resizeBackground(1.2, 120, 110);
		editor.toggleBackgroundEditing();
		return { before, after: window.current.background.bounds };
	});
	assert.notDeepEqual(imageMoved.before, imageMoved.after);
	await page.evaluate(() => window.map.close());
	assert.equal(await page.evaluate(() => window.tileImages.filter(image => image.onload).length), 0);
	assert.equal(await page.locator("#satellite-map").getAttribute("width"), "1");
	const solver = await page.evaluate(async () => {
		const { solveEditorWasmAsync, editorSolverState } = await import("/static/editor-solver.js?v=editor-align-2026-09-09d");
		const polygons = Array.from({ length: 48 }, (_, i) => [[i + 1, -1], [i + 1.5, -1], [i + 1.5, 1], [i + 1, 1]]);
		const result = await solveEditorWasmAsync({ start: [0, 0], target: [50, 0], polygons, background: { data_url: "x".repeat(1000000), mustNotBeCloned() {} } });
		const concave = await solveEditorWasmAsync({ start: [-1, 0], target: [4, 0], polygons: [[[0, -1], [3, -1], [3, 2], [1, 1], [0, 2]]] });
		const mainThreadModuleBeforeTiny = Boolean(editorSolverState.module);
		const timings = [];
		const tiny = { start: [0, 0], target: [4, 0], polygons: [[[1, -1], [2, -1], [1, 1]], [[3, -1], [3, 1], [2, 1]]] };
		for (let i = 0; i < 10; i += 1) {
			const started = performance.now();
			const solved = await solveEditorWasmAsync(tiny);
			timings.push({ solverMs: solved.seconds * 1000, updateMs: performance.now() - started });
		}
		const controller = new AbortController();
		const pending = solveEditorWasmAsync({ start: [0, 0], target: [50, 0], polygons }, null, controller.signal);
		controller.abort();
		let cancelled = false;
		try { await pending; } catch (error) { cancelled = error.name === "AbortError"; }
		return { result, concave, timings, cancelled, mainThreadModuleBeforeTiny, mainThreadModule: Boolean(editorSolverState.module) };
	});
	assert.equal(solver.result.exact, true);
	assert.ok(Math.abs(solver.result.length - 50) < 1e-6);
	assert.equal(solver.concave.exact, true);
	assert.equal(solver.cancelled, true);
	assert.equal(solver.mainThreadModuleBeforeTiny, false);
	assert.equal(solver.mainThreadModule, true);
	assert.deepEqual(errors, []);
	console.log(JSON.stringify({ backgroundPixel: pixel, tiles, solver }));
} finally {
	await browser.close();
}
