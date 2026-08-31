import { chromium } from "@playwright/test";
import assert from "node:assert/strict";

const baseUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8017";
const browser = await chromium.launch({ headless: true });
try {
	const page = await browser.newPage();
	const runtimeErrors = [];
	const failedRequests = [];
	page.on("pageerror", (error) => runtimeErrors.push(`pageerror: ${error.message}`));
	page.on("console", (message) => {
		if (message.type() === "error") {
			runtimeErrors.push(`console: ${message.text()}`);
		}
	});
	page.on("requestfailed", (request) => {
		failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || "unknown failure"}`);
	});
	await page.goto(`${baseUrl}/`, { waitUntil: "networkidle" });
	assert.equal(await page.title(), "TPP Benchmark Dashboard", "Dashboard title did not load.");
	await page.waitForFunction(() => window.__benchmarkDashboardReady === true);
	for (const selector of [
		"#campaign-list", "#job-list", "#create-form", "#cases-panel",
		"#inspect-panel", "#benchmark-panel", "#comparison-panel", "#manual-case-canvas",
	]) {
		await page.locator(selector).waitFor({ state: "attached" });
	}

	const campaignsResponse = await page.request.get(`${baseUrl}/api/campaigns`);
	assert.equal(campaignsResponse.status(), 200, "Campaign API did not respond successfully.");
	assert.ok(Array.isArray((await campaignsResponse.json()).campaigns), "Campaign API returned an invalid payload.");

	await page.getByRole("button", { name: "Inspect" }).click();
	await assertActivePanel(page, "inspect-panel");
	await page.getByRole("button", { name: "Benchmark" }).click();
	await assertActivePanel(page, "benchmark-panel");
	await page.getByRole("button", { name: "Comparison" }).click();
	await assertActivePanel(page, "comparison-panel");
	await page.getByRole("button", { name: "Cases" }).click();
	await assertActivePanel(page, "cases-panel");
	assert.equal(await page.locator("#manual-case-canvas").count(), 1, "Manual editor canvas is missing.");
	assert.equal(await page.locator("#manual-case-canvas").getAttribute("tabindex"), "0", "Editor canvas is not keyboard focusable.");
	assert.equal(await page.locator("#campaign-modal").getAttribute("aria-modal"), "true");
	assert.equal(await page.locator("#confirm-modal").getAttribute("aria-modal"), "true");

	await page.getByRole("button", { name: "Inspect" }).click();
	const campaignCard = page.locator(".campaign-card").first();
	if (await campaignCard.count() > 0) {
		await campaignCard.click();
		await page.locator("#campaign-modal").waitFor({ state: "visible" });
		assert.equal(await page.locator("#campaign-modal").getAttribute("role"), "dialog");
		await page.locator("[data-close-modal]").last().click();
		await page.locator("#campaign-modal").waitFor({ state: "hidden" });
	}

	const initialTheme = await page.locator("html").getAttribute("data-theme");
	await page.locator("#theme-toggle").click();
	const toggledTheme = await page.locator("html").getAttribute("data-theme");
	assert.notEqual(toggledTheme, initialTheme, "Theme toggle did not change the document theme.");
	assert.equal(
		await page.evaluate(() => localStorage.getItem("benchmarkDashboardTheme")),
		toggledTheme,
		"Theme choice was not persisted.",
	);
	await page.locator("#theme-toggle").click();
	assert.equal(await page.locator("html").getAttribute("data-theme"), initialTheme, "Theme did not toggle back.");

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

async function assertActivePanel(page, panelId) {
	assert.equal(
		await page.locator(`#${panelId}`).evaluate((panel) => panel.classList.contains("is-active")),
		true,
		`${panelId} did not become active.`,
	);
}
