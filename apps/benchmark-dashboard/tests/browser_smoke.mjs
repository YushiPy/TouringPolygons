import { chromium } from "@playwright/test";
import assert from "node:assert/strict";

const baseUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8017";
const browser = await chromium.launch({ headless: true });
const runtimeErrors = [];
const failedRequests = [];

function monitor(page) {
	page.on("pageerror", (error) => runtimeErrors.push(`pageerror: ${error.message}`));
	page.on("console", (message) => {
		if (message.type() === "error") {
			runtimeErrors.push(`console: ${message.text()}`);
		}
	});
	page.on("requestfailed", (request) => {
		failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || "unknown failure"}`);
	});
}

try {
	const dashboardPage = await browser.newPage();
	monitor(dashboardPage);
	const dashboardResponse = await dashboardPage.goto(`${baseUrl}/`, { waitUntil: "load" });
	assert.equal(dashboardResponse?.ok(), true, "Dashboard route did not load successfully.");
	assert.equal(await dashboardPage.title(), "TPP Benchmark Dashboard", "Dashboard title did not load.");
	await dashboardPage.waitForFunction(() => window.__benchmarkDashboardReady === true);
	await dashboardPage.locator("#campaign-list").waitFor({ state: "attached" });
	await dashboardPage.locator("#manual-case-canvas").waitFor({ state: "attached" });
	const fixedFallback = await dashboardPage.evaluate(async () => {
		const { createManualEditor } = await import("/static/manual-editor.js?v=editor-align-2026-09-12a");
		const caseData = {
			start: [0, 0],
			target: [4, 0],
			polygons: [[[1, -1], [2, -1], [2, 1], [1, 1]]],
		};
		const editor = createManualEditor({
			$: selector => document.querySelector(selector),
			state: { manualCases: [caseData], manualCaseIndex: 0 },
			formatLength: value => String(value),
			formatSeconds: value => String(value),
			cssVar: () => "",
			scheduleManualAutosave() {},
			updateManualCaseListMetadata() {},
		});
		editor.requestDraw = () => {};
		editor.updateLabelDirections = () => {};
		await editor.fetchSolution(caseData, 0, performance.now());
		return { path: editor.solutionPath, status: document.querySelector("#manual-solve-status").textContent };
	});
	assert.deepEqual(fixedFallback.path, [[0, 0], [4, 0]]);
	assert.match(fixedFallback.status, /Runtime\s*server/);
	const reference = await (await dashboardPage.request.get(`${baseUrl}/api/free-order/reference`)).json();
	if (reference.rows.length) {
		assert.ok(reference.rows.every((row) => !("geometry" in row) && !("path" in row)));
		const detail = await (await dashboardPage.request.get(`${baseUrl}${reference.visualization_endpoint}/0`)).json();
		assert.ok(detail.rows.some((row) => row.geometry && row.path));
	}
	const virtualWindow = await dashboardPage.evaluate(async () => {
		const { renderFreeOrderReport } = await import("/static/free-order-report.js?v=2026-09-10e");
		const root = document.createElement("div");
		document.body.append(root);
		renderFreeOrderReport(root, { title: "Virtual", path: "virtual", rows: Array.from({ length: 250 }, (_, caseIndex) => ({ case: caseIndex, solver: "unordered", exact: true })) });
		return { rendered: root.querySelectorAll(".free-result-row").length, label: root.querySelector("[data-free-window-label]").textContent };
	});
	assert.deepEqual(virtualWindow, { rendered: 100, label: "1–100 of 250" });
	await dashboardPage.locator('[data-panel="benchmark-panel"]').click();
	const threads = dashboardPage.locator("#threads-slider");
	if (Number(await threads.getAttribute("max")) > 1) {
		await threads.focus();
		await threads.press("Home");
		await threads.press("ArrowRight");
		assert.equal(await dashboardPage.locator("#threads-input").inputValue(), "2");
		await dashboardPage.locator('#run-form [data-visit-order-picker] [data-value="free"]').click();
		assert.equal(await threads.isEnabled(), true, "Free-order campaign parallelism disabled the thread control.");
		assert.equal(await dashboardPage.locator("#threads-input").inputValue(), "2");
	}

	const eventPage = await browser.newPage();
	monitor(eventPage);
	const eventResponse = await eventPage.goto(`${baseUrl}/evento`, { waitUntil: "load" });
	assert.equal(eventResponse?.ok(), true, "Event route did not load successfully.");
	assert.equal(await eventPage.title(), "Visitar regiões, escolher caminhos · 34º SIICUSP", "Event title did not load.");
	await eventPage.locator("#map-content").waitFor({ state: "attached" });
	await eventPage.locator("#challenge-dialog").waitFor({ state: "attached" });
	await eventPage.locator("#references-dialog").waitFor({ state: "attached" });

	if (failedRequests.length || runtimeErrors.length) {
		throw new Error([
			"Browser smoke detected runtime or network errors.",
			...runtimeErrors,
			...failedRequests.map((request) => `requestfailed: ${request}`),
		].join("\n"));
	}
	console.log("browser smoke test passed");
} finally {
	await browser.close();
}
