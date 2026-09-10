import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

const baseUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8137";
const browser = await chromium.launch({ headless: true });
const name = `free-order-smoke-${Date.now()}`;
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
let created = false;
try {
	await page.goto(baseUrl);
	await page.waitForFunction(() => window.__benchmarkDashboardReady === true);
	await page.locator('[data-panel="comparison-panel"]').click();
	await page.locator("#show-free-reference").click();
	await page.locator("#free-reference-report .free-solved").first().waitFor();
	assert.equal(await page.locator("#compare-visit-order").inputValue(), "free");
	assert.equal(await page.locator("#run-visit-order").inputValue(), "free");
	assert.match(await page.locator("#free-reference-report").innerText(), /41 \/ 60/);
	assert.equal(await page.locator('#free-reference-report [data-free-case="0"] span').textContent(), "1");
	await page.locator('#free-reference-report [data-free-case="0"]').click();
	await page.locator('#free-reference-report [data-free-detail="0"] svg').waitFor({ state: "visible" });
	assert.equal(await page.locator('#free-reference-report [data-free-detail="0"] .free-contact').count(), 40);
	assert.ok(await page.locator('#free-reference-report [data-free-detail="0"] marker').count());
	await page.locator('#free-reference-report [data-free-detail="0"] [data-free-progress]').evaluate((slider) => {
		slider.value = "370";
		slider.dispatchEvent(new slider.ownerDocument.defaultView.Event("input", { bubbles: true }));
	});
	await page.locator("#show-free-reference").click();
	await page.locator('#free-reference-report [data-free-detail="0"] svg').waitFor({ state: "visible" });
	assert.equal(await page.locator('#free-reference-report [data-free-detail="0"] [data-free-progress]').inputValue(), "370");
	assert.equal(await page.locator('#free-reference-report [data-free-case="0"]').getAttribute("aria-expanded"), "true");
	await page.locator('#free-reference-report [data-free-detail="0"] [data-free-speed="1"]').click();
	assert.equal(await page.locator('#free-reference-report [data-free-detail="0"] [data-free-speed-value]').textContent(), "1.5×");
	await page.locator('#free-reference-report [data-free-sort-key="polygons"]').click();
	assert.equal(await page.locator('#free-reference-report [data-sort-column="polygons"]').getAttribute("aria-sort"), "ascending");
	assert.equal(await page.locator('#free-reference-report [data-free-detail="0"] [data-free-progress]').inputValue(), "370");
	await mkdir("../../benchmarks/results/unordered/dashboard", { recursive: true });
	await page.screenshot({ path: "../../benchmarks/results/unordered/dashboard/reference.png", fullPage: true });
	await page.locator("#free-reference-report [data-free-results] > summary").click();
	await page.locator("#show-free-reference").click();
	assert.equal(await page.locator("#free-reference-report [data-free-results]").getAttribute("open"), null);
	await page.locator('#compare-form [data-visit-order-picker] [data-value="fixed"]').click();
	assert.equal(await page.locator('#compare-form [data-order-only="fixed"]').isVisible(), true);
	assert.equal(await page.locator('#compare-form [data-order-only="free"]').isVisible(), false);
	const create = await page.request.post(`${baseUrl}/api/campaigns/manual`, { data: { name } });
	assert.equal(create.status(), 200); created = true;
	const polygons = [
		[[9, 3], [11, 3], [11, 3.6], [9.6, 3.6], [9.6, 5], [9, 5]],
		[[1, 3], [3, 3], [3, 3.6], [1.6, 3.6], [1.6, 5], [1, 5]],
		[[5, -5], [7, -5], [7, -4.4], [5.6, -4.4], [5.6, -3], [5, -3]],
	];
	const geometry = { start: [0, 0], target: [12, 0], polygons };
	assert.equal((await page.request.put(`${baseUrl}/api/campaigns/${name}/cases?refresh_previews=false`, { data: { cases: [geometry] } })).status(), 200);
	const live = await page.request.post(`${baseUrl}/api/editor/solve`, { data: { ...geometry, visit_order: "free" }, timeout: 60000 });
	assert.equal(live.status(), 200, await live.text());
	assert.deepEqual((await live.json()).order, [1, 2, 0]);
	await page.reload();
	await page.waitForFunction(() => window.__benchmarkDashboardReady === true);
	await page.locator('[data-panel="benchmark-panel"]').click();
	await page.locator('#run-form [data-visit-order-picker] [data-value="free"]').click();
	await page.locator(`#run-campaign-grid [data-value="${name}"]`).click();
	await page.locator('#run-form [name="max_seconds"]').fill("2");
	const response = page.waitForResponse((r) => r.url().endsWith("/api/runs") && r.request().method() === "POST");
	await page.locator("#run-submit-button").click();
	const submitted = await (await response).json();
	assert.ok(submitted.command.includes("free-order"));
	for (let i = 0; i < 120; i++) {
		const job = await (await page.request.get(`${baseUrl}/api/jobs/${submitted.job}`)).json();
		if (job.status === "completed") break;
		assert.notEqual(job.status, "failed", job.output);
		await page.waitForTimeout(500);
	}
	await page.waitForFunction(() => document.querySelector("#benchmark-report")?.textContent.includes("1 / 1"), null, { timeout: 15000 });
	await page.locator('#benchmark-report [data-free-case="0"]').click();
	assert.match(await page.locator("#benchmark-report").innerText(), /1 → 2 → 0/);
	assert.equal((await (await page.request.get(`${baseUrl}/api/campaigns/${name}/summaries`)).json()).files.length, 0);
	await page.locator('[data-panel="comparison-panel"]').click();
	await page.locator(`#compare-campaign-grid [data-value="${name}"]`).click();
	await page.locator('#compare-form [name="max_seconds"]').fill("2");
	const compareResponse = page.waitForResponse((r) => r.url().endsWith("/api/comparisons") && r.request().method() === "POST");
	await page.locator('#compare-form button[type="submit"]').click();
	const comparison = await (await compareResponse).json();
	assert.ok(comparison.command.includes("free-order"));
	for (let i = 0; i < 180; i++) {
		const job = await (await page.request.get(`${baseUrl}/api/jobs/${comparison.job}`)).json();
		if (job.status === "completed") break;
		assert.notEqual(job.status, "failed", job.output);
		await page.waitForTimeout(500);
	}
	await page.waitForFunction(() => document.querySelectorAll("#comparison-report .free-solved").length === 2, null, { timeout: 15000 });
	const report = await (await page.request.get(`${baseUrl}/api/campaigns/${name}/free-results`)).json();
	assert.equal(report.status, "completed");
	assert.equal(report.rows.length, 2);
	assert.equal(report.rows[0].sha256, report.rows[1].sha256);
	assert.ok(report.rows.every((row) => row.exact && !row.error));
	assert.deepEqual(errors, []);
	console.log("Free-order browser smoke passed: recorded report, mode switch, live solver, campaign run, external comparison, saved path, result isolation.");
} finally {
	if (created) await page.request.delete(`${baseUrl}/api/campaigns/${name}`);
	await browser.close();
}
