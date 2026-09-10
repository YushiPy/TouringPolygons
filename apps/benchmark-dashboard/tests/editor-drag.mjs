import { chromium, webkit } from "@playwright/test";
import assert from "node:assert/strict";

const browser = await (process.env.TEST_BROWSER === "webkit" ? webkit : chromium).launch({ headless: true });
try {
	const page = await browser.newPage();
	await page.route("**/drag-harness", route => route.fulfill({ contentType: "text/html", body: '<canvas style="width:800px;height:500px"></canvas>' }));
	await page.goto("http://127.0.0.1:8017/drag-harness");
	const results = await page.evaluate(async () => {
		const { createManualEditor } = await import("/static/manual-editor.js");
		const { formatLength, formatSeconds } = await import("/static/format.js");
		let workers = 0, terminated = 0;
		const OriginalWorker = window.Worker;
		window.Worker = class extends OriginalWorker {
			constructor(...args) { super(...args); workers += 1; }
			terminate() { terminated += 1; super.terminate(); }
		};
		const results = [];
		for (const count of [2, 5]) {
			const current = { start: [0, 0], target: [12, 0], polygons: Array.from({ length: count }, (_, i) => [[i * 2 + 1, 1], [i * 2 + 2, 1], [i * 2 + 1.5, 2]]) };
			const editor = createManualEditor({ $: selector => document.querySelector(selector), state: { manualCases: [current], manualCaseIndex: 0 }, cssVar: () => "", formatLength, formatSeconds, scheduleManualAutosave() {}, updateManualCaseListMetadata() {} });
			editor.canvas = document.querySelector("canvas");
			editor.ctx = editor.canvas.getContext("2d");
			editor.resize();
			editor.frameCurrentCase();
			const idle = async () => {
				const until = performance.now() + 5000;
				while (editor.solutionRun || editor.solutionQueued) {
					if (performance.now() > until) throw new Error("Drag solution did not settle");
					await new Promise(resolve => setTimeout(resolve, 1));
				}
				await new Promise(requestAnimationFrame);
			};
			editor.scheduleSolve();
			await idle();
			editor.updateLabelDirections(false);
			await new Promise(requestAnimationFrame);
				let renders = 0, solves = 0;
				const draw = editor.draw.bind(editor), fetchSolution = editor.fetchSolution.bind(editor);
				editor.draw = () => { renders += 1; draw(); };
			editor.fetchSolution = (...args) => { solves += 1; return fetchSolution(...args); };
			const rect = editor.canvas.getBoundingClientRect();
			editor.dragPolygon = { index: 0, lastWorld: editor.canvasToWorld(200, 200) };
			const beforeWorkers = workers, beforeTerminated = terminated;
			const started = performance.now();
			for (let i = 1; i <= 200; i += 1) {
				editor.onPointerMove({ preventDefault() {}, pointerId: 1, clientX: rect.left + 200 + i / 10, clientY: rect.top + 200 });
			}
			const synchronousRenders = renders;
			await idle();
			results.push({ count, events: 200, synchronousRenders, renders, solves, workersCreated: workers - beforeWorkers, workersTerminated: terminated - beforeTerminated,
				settleMs: performance.now() - started, stale: editor.solutionStale, hasPath: Boolean(editor.solutionPath), labelAnimation: editor.labelAnimation });
			editor.cancelPendingSolution();
		}
		return results;
	});
	for (const result of results) {
		assert.equal(result.synchronousRenders, 0);
		assert.ok(result.renders <= 3, JSON.stringify(result));
		assert.equal(result.solves, 2);
		assert.equal(result.workersCreated, 0);
		assert.equal(result.workersTerminated, 0);
		assert.equal(result.stale, false);
		assert.equal(result.hasPath, true);
		assert.equal(result.labelAnimation, null);
	}
	console.log(JSON.stringify(results));
} finally {
	await browser.close();
}
