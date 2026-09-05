import { escapeHTML } from "./dom.js";
import { downloadCSV } from "./format.js";

export function relativeGap(row) {
	if (!row || !Number.isFinite(row.upper_bound) || !Number.isFinite(row.lower_bound)) return null;
	return row.upper_bound === 0 ? 0 : Math.max(0, (row.upper_bound - row.lower_bound) / row.upper_bound);
}

export function freeOrderSummary(rows, solver) {
	const selected = rows.filter((row) => row.solver === solver);
	const gaps = selected.map(relativeGap).filter((gap) => gap !== null);
	return { count: selected.length, solved: selected.filter((r) => r.exact && !r.error).length,
		seconds: selected.reduce((sum, r) => sum + (r.seconds || 0), 0),
		gap: gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : null,
		errors: selected.filter((r) => r.error).length };
}

const number = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "n/a";
const percent = (value) => value === null ? "n/a" : `${(100 * value).toFixed(3)}%`;

function pathSVG(row) {
	if (!row?.geometry || !row.path) return "<p>Stored path geometry is unavailable for this result.</p>";
	const { polygons, start, target } = row.geometry;
	const points = [...polygons.flat(), ...row.path, start, target];
	const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
	const minX = Math.min(...xs), minY = Math.min(...ys);
	const width = Math.max(...xs) - minX, height = Math.max(...ys) - minY;
	const scale = Math.min(680 / Math.max(width, 1e-9), 310 / Math.max(height, 1e-9));
	const point = ([x, y]) => [20 + (x - minX) * scale, 330 - (y - minY) * scale];
	const coords = (p) => p.map((v) => point(v).join(",")).join(" ");
	return `<svg viewBox="0 0 720 350" class="free-path" role="img" aria-label="Stored free-order solution for case ${row.case}">
		${polygons.map((p, i) => `<polygon points="${coords(p)}"/><text x="${point(p[0])[0]}" y="${point(p[0])[1] - 5}">${i}</text>`).join("")}
		<polyline points="${coords(row.path)}"/>
		${[start, target].map((p, i) => `<circle cx="${point(p)[0]}" cy="${point(p)[1]}" r="4"/><text x="${point(p)[0] + 7}" y="${point(p)[1]}">${i ? "t" : "s"}</text>`).join("")}
	</svg><p>First-visit order (zero-based): <strong>${escapeHTML(row.order?.join(" → ") || "No regions")}</strong></p>`;
}

export function renderFreeOrderReport(root, report) {
	root.classList.remove("is-hidden");
	const rows = report?.rows || [];
	const solvers = ["unordered", "tspn"].filter((solver) => rows.some((r) => r.solver === solver));
	const labels = { unordered: "Our TPP B&B", tspn: "External TSPN" };
	const groups = new Map();
	for (const row of rows) {
		if (!groups.has(row.case)) groups.set(row.case, {});
		groups.get(row.case)[row.solver] = row;
	}
	const failures = rows.filter((r) => r.solver === "tspn" && r.endpoint_valid === false).length;
	root.innerHTML = `<header class="report-header"><div><h3>${escapeHTML(report?.title || "Free-order results")}</h3>
		<p>Free visit order · fixed endpoints · ${escapeHTML(report?.status || "No run")}</p></div>
		<button type="button" class="secondary" data-free-export>Export CSV</button></header>
		<div class="free-summary">${solvers.map((solver) => {
			const summary = freeOrderSummary(rows, solver);
			return `<section class="report-panel"><h4>${labels[solver]}</h4><strong class="free-solved">${summary.solved} / ${summary.count}</strong><p>Optimal within configured tolerance</p>
				<p>Total solve time <b>${number(summary.seconds, 2)} s</b> · Mean gap <b>${percent(summary.gap)}</b></p>
				<p>${summary.errors} errors${solver === "unordered" ? ` · ${rows.filter((r) => r.solver === solver && r.valid === true).length} independently validated paths` : ` · ${failures} endpoint check failures`}</p></section>`;
		}).join("")}</div>
		<details class="free-notes" open><summary>Method and numerical tolerances</summary>${(report?.notes || []).map((note) => `<p>${escapeHTML(note)}</p>`).join("")}</details>
		${report?.error ? `<p role="alert">${escapeHTML(report.error)}</p>` : ""}
		${rows.length ? `<label class="free-filter">Show <select data-free-filter><option value="all">All instances</option><option value="open">Open gap / errors</option><option value="solved">Our solved instances</option></select></label>
		<div class="free-table-wrap"><table class="free-table"><thead><tr><th>Case</th><th>Polygons</th><th>Our time (s)</th><th>External time (s)</th><th>Our gap</th><th>External gap</th><th>Our status</th><th>External status</th></tr></thead><tbody></tbody></table></div>` : "<p>No free-order results yet. Run the selected campaign or open the recorded comparison.</p>"}`;
	const drawRows = () => {
		const filter = root.querySelector("[data-free-filter]")?.value || "all";
		const tbody = root.querySelector("tbody");
		if (!tbody) return;
		tbody.innerHTML = [...groups.entries()].sort((a, b) => a[0] - b[0]).filter(([, pair]) => filter === "all" || (filter === "solved" ? pair.unordered?.exact : Object.values(pair).some((row) => !row.exact || row.error))).map(([index, pair]) => {
			const a = pair.unordered, b = pair.tspn;
			return `<tr><td><button type="button" class="secondary" data-free-case="${index}">${index} · View path</button></td><td>${a?.polygons ?? b?.polygons ?? ""}</td>
				<td>${number(a?.seconds)}</td><td>${number(b?.seconds)}</td><td>${percent(relativeGap(a))}</td><td>${percent(relativeGap(b))}</td>
				<td>${escapeHTML(a?.error || a?.termination || "n/a")}</td><td>${escapeHTML(b?.error || b?.termination || "n/a")}${b?.endpoint_valid === false ? " · endpoint warning" : ""}</td></tr>
				<tr data-free-detail="${index}" class="is-hidden"><td colspan="8">${pathSVG(a)}<p>Our bounds: ${number(a?.lower_bound, 8)} ≤ optimum ≤ ${number(a?.upper_bound, 8)} · ${a?.calls ?? "n/a"} convex calls · ${a?.fallback_calls ?? "n/a"} fallback calls</p></td></tr>`;
		}).join("");
		tbody.querySelectorAll("[data-free-case]").forEach((button) => button.addEventListener("click", () => tbody.querySelector(`[data-free-detail="${button.dataset.freeCase}"]`).classList.toggle("is-hidden")));
	};
	root.querySelector("[data-free-filter]")?.addEventListener("change", drawRows);
	root.querySelector("[data-free-export]").addEventListener("click", () => downloadCSV("free-order-results.csv", rows.map((r) => ({
		visit_order: "free", solver: r.solver, case: r.case, sha256: r.sha256, lower_bound: r.lower_bound, upper_bound: r.upper_bound,
		seconds: r.seconds, gap: relativeGap(r), exact: r.exact, termination: r.termination, valid: r.valid, endpoint_valid: r.endpoint_valid,
		calls: r.calls, fallback_calls: r.fallback_calls, order: r.order?.join(" "), error: r.error,
	}))));
	drawRows();
}
