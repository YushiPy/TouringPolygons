import { chromium } from "@playwright/test";

const baseUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8017";
const browser = await chromium.launch({ headless: true });
try {
	const page = await browser.newPage();
	await page.goto(`${baseUrl}/`, { waitUntil: "networkidle" });
	if (await page.title() !== "TPP Benchmark Dashboard") {
		throw new Error("Dashboard title did not load.");
	}
	await page.locator("#campaign-list").waitFor({ state: "attached" });
	await page.locator("#job-list").waitFor({ state: "attached" });
	const moduleLoaded = await page.evaluate(() => Boolean(window.__benchmarkDashboardReady));
	if (!moduleLoaded) {
		throw new Error("Dashboard module did not finish initialization.");
	}
	console.log("browser smoke test passed");
} finally {
	await browser.close();
}
