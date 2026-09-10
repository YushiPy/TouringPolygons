import { escapeHTML } from "./dom.js";
import { downloadCSV } from "./format.js";
import { convexHull, pathPrefix, playbackDuration, projectedCase, regionColors } from "./event-geometry.js";
import { displayPartition } from "./native-partition.js?v=editor-align-2026-09-09d";

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

const coordinates = (points) => points.map((point) => point.join(",")).join(" ");

function pathViewerHTML(row) {
	if (!row?.geometry || !row.path) return "<p>Stored path geometry is unavailable for this result.</p>";
	return `<figure class="free-path-viewer" data-free-path-viewer>
		<div class="free-path-toolbar">
			<button class="secondary free-play" type="button" data-free-play aria-pressed="false">▶ Play path</button>
			<label>Progress <input type="range" min="0" max="1000" value="1000" data-free-progress><output data-free-progress-value>100%</output></label>
			<div class="editor-layer-toggles" aria-label="Path layers">
				<button class="secondary is-active" type="button" data-free-layer="contacts" aria-pressed="true">Visitation points</button>
				<button class="secondary" type="button" data-free-layer="decomposition" aria-pressed="false">Convex decomposition</button>
				<button class="secondary" type="button" data-free-layer="hulls" aria-pressed="false">Convex hull</button>
				<button class="secondary is-active" type="button" data-free-layer="labels" aria-pressed="true">Labels</button>
			</div>
		</div>
		<svg viewBox="0 0 840 480" class="free-path" role="img" aria-label="Stored free-order solution for case ${row.case}" data-free-svg></svg>
		<figcaption><span>First-visit order</span><strong>${escapeHTML(row.order?.join(" → ") || "No regions")}</strong></figcaption>
	</figure>`;
}

function setupPathViewer(root, row) {
	const projected = projectedCase(row);
	const svg = root.querySelector("[data-free-svg]");
	const progress = root.querySelector("[data-free-progress]");
	const progressValue = root.querySelector("[data-free-progress-value]");
	const play = root.querySelector("[data-free-play]");
	const layers = { contacts: true, decomposition: false, hulls: false, labels: true };
	const order = row.order || row.geometry.polygons.map((_, index) => index);
	const segmentLengths = row.path.slice(1).map((point, index) => Math.hypot(point[0] - row.path[index][0], point[1] - row.path[index][1]));
	const totalLength = segmentLengths.reduce((sum, value) => sum + value, 0) || 1;
	const fractions = [];
	let traversed = 0;
	for (let index = 0; index < order.length; index += 1) {
		traversed += segmentLengths[index] || 0;
		fractions[order[index]] = traversed / totalLength;
	}
	let fraction = 1;
	let frame = null;

	const stop = () => {
		if (frame !== null) cancelAnimationFrame(frame);
		frame = null;
		play.textContent = "▶ Play path";
		play.setAttribute("aria-pressed", "false");
	};
	const draw = () => {
		const labels = layers.labels ? projected.polygons.map((polygon, index) => `<text class="free-region-label" x="${polygon[0][0]}" y="${polygon[0][1] - 7}">${index}</text>`).join("") : "";
		const hulls = layers.hulls ? row.geometry.polygons.map((polygon) => `<polygon class="free-hull" points="${coordinates(convexHull(polygon).map(projected.project))}"/>`).join("") : "";
		const decomposition = layers.decomposition ? row.geometry.polygons.flatMap((polygon) => (displayPartition(polygon, draw, () => {}) || []).map((piece) => `<polygon class="free-convex-piece" points="${coordinates(piece.map(projected.project))}"/>`)).join("") : "";
		const polygons = projected.polygons.map((polygon, index) => {
			const visited = fraction + 1e-9 >= (fractions[index] ?? Infinity);
			const colors = regionColors(order.indexOf(index), order.length, visited);
			return `<polygon class="free-region" points="${coordinates(polygon)}" style="fill:${colors.fill};stroke:${colors.stroke}"/>`;
		}).join("");
		const contacts = layers.contacts ? order.map((region, index) => {
			const point = projected.path[index + 1];
			if (!point) return "";
			return `<circle class="free-contact ${fraction + 1e-9 >= (fractions[region] ?? Infinity) ? "is-visited" : ""}" cx="${point[0]}" cy="${point[1]}" r="4"/>`;
		}).join("") : "";
		const prefix = pathPrefix(projected.path, fraction);
		const start = projected.project(row.geometry.start);
		const target = projected.project(row.geometry.target);
		svg.innerHTML = `${hulls}${polygons}${decomposition}<polyline class="free-route" points="${coordinates(prefix)}"/>${contacts}${labels}<circle class="free-endpoint free-start" cx="${start[0]}" cy="${start[1]}" r="5"/><circle class="free-endpoint free-target" cx="${target[0]}" cy="${target[1]}" r="5"/>${layers.labels ? `<text class="free-endpoint-label" x="${start[0] + 9}" y="${start[1] + 4}">s</text><text class="free-endpoint-label" x="${target[0] + 9}" y="${target[1] + 4}">t</text>` : ""}`;
		progress.value = String(Math.round(fraction * 1000));
		progress.style.setProperty("--progress", `${fraction * 100}%`);
		progressValue.textContent = `${Math.round(fraction * 100)}%`;
	};
	progress.addEventListener("input", () => { stop(); fraction = Number(progress.value) / 1000; draw(); });
	play.addEventListener("click", () => {
		if (frame !== null) { stop(); return; }
		if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { fraction = 1; draw(); return; }
		if (fraction >= 1) fraction = 0;
		play.textContent = "Ⅱ Pause";
		play.setAttribute("aria-pressed", "true");
		let previous = performance.now();
		const tick = (now) => {
			fraction = Math.min(1, fraction + (now - previous) / playbackDuration(order.length));
			previous = now;
			draw();
			if (fraction < 1) frame = requestAnimationFrame(tick); else stop();
		};
		frame = requestAnimationFrame(tick);
	});
	root.querySelectorAll("[data-free-layer]").forEach((button) => button.addEventListener("click", () => {
		const layer = button.dataset.freeLayer;
		layers[layer] = !layers[layer];
		button.classList.toggle("is-active", layers[layer]);
		button.setAttribute("aria-pressed", String(layers[layer]));
		draw();
	}));
	draw();
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
	let filter = "all";
	let sortKey = "case";
	let descending = false;
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
		${rows.length ? `<div class="free-table-controls">
			<div class="field"><span class="field-label">Show</span><div class="segmented" data-free-filter>
				<button class="segment is-active" data-value="all" type="button" aria-pressed="true">All instances</button>
				<button class="segment" data-value="open" type="button" aria-pressed="false">Open gap / errors</button>
				<button class="segment" data-value="solved" type="button" aria-pressed="false">Our solved</button>
			</div></div>
			<div class="field"><span class="field-label">Sort by</span><div class="segmented free-sort" data-free-sort>
				<button class="segment is-active" data-value="case" type="button" aria-pressed="true">Case</button>
				<button class="segment" data-value="polygons" type="button" aria-pressed="false">Polygons</button>
				<button class="segment" data-value="time" type="button" aria-pressed="false">Solve time</button>
				<button class="segment" data-value="gap" type="button" aria-pressed="false">Gap</button>
				<button class="segment" data-value="result" type="button" aria-pressed="false">Result</button>
				<button class="segment icon-segment" data-free-reverse type="button" aria-label="Reverse order" title="Reverse order" aria-pressed="false">↕</button>
			</div></div>
		</div>
		<div class="free-table-wrap"><table class="free-table"><thead><tr><th>Case</th><th>Polygons</th><th>Our time (s)</th><th>External time (s)</th><th>Our gap</th><th>External gap</th><th>Our status</th><th>External status</th></tr></thead><tbody></tbody></table></div>` : "<p>No free-order results yet. Run the selected campaign or open the recorded comparison.</p>"}`;
	const drawRows = () => {
		const tbody = root.querySelector("tbody");
		if (!tbody) return;
		const value = ([index, pair]) => {
			const ours = pair.unordered;
			if (sortKey === "polygons") return ours?.polygons ?? pair.tspn?.polygons ?? 0;
			if (sortKey === "time") return ours?.seconds ?? Infinity;
			if (sortKey === "gap") return relativeGap(ours) ?? Infinity;
			if (sortKey === "result") return ours?.exact && !ours?.error ? 0 : 1;
			return index;
		};
		const ordered = [...groups.entries()].filter(([, pair]) => filter === "all" || (filter === "solved" ? pair.unordered?.exact && !pair.unordered?.error : Object.values(pair).some((item) => !item.exact || item.error)))
			.sort((left, right) => (descending ? -1 : 1) * (value(left) - value(right)) || left[0] - right[0]);
		tbody.innerHTML = ordered.map(([index, pair]) => {
			const a = pair.unordered, b = pair.tspn;
			return `<tr><td><button type="button" class="secondary" data-free-case="${index}">${index} · View path</button></td><td>${a?.polygons ?? b?.polygons ?? ""}</td>
				<td>${number(a?.seconds)}</td><td>${number(b?.seconds)}</td><td>${percent(relativeGap(a))}</td><td>${percent(relativeGap(b))}</td>
				<td>${escapeHTML(a?.error || a?.termination || "n/a")}</td><td>${escapeHTML(b?.error || b?.termination || "n/a")}${b?.endpoint_valid === false ? " · endpoint warning" : ""}</td></tr>
				<tr data-free-detail="${index}" class="free-detail-row is-hidden"><td colspan="8"><div class="free-detail-content">${pathViewerHTML(a || b)}<p class="free-detail-stats">Our bounds: ${number(a?.lower_bound, 8)} ≤ optimum ≤ ${number(a?.upper_bound, 8)} · ${a?.calls ?? "n/a"} convex calls · ${a?.fallback_calls ?? "n/a"} fallback calls</p></div></td></tr>`;
		}).join("");
		tbody.querySelectorAll("[data-free-case]").forEach((button) => button.addEventListener("click", () => {
			const detail = tbody.querySelector(`[data-free-detail="${button.dataset.freeCase}"]`);
			const opening = detail.classList.contains("is-hidden");
			detail.classList.toggle("is-hidden");
			if (opening && !detail.dataset.ready) {
				detail.dataset.ready = "true";
				const pair = groups.get(Number(button.dataset.freeCase)) || groups.get(button.dataset.freeCase);
				const viewer = detail.querySelector("[data-free-path-viewer]");
				if (viewer) setupPathViewer(viewer, pair.unordered || pair.tspn);
			}
		}));
	};
	root.querySelectorAll("[data-free-filter] [data-value]").forEach((button) => button.addEventListener("click", () => {
		filter = button.dataset.value;
		root.querySelectorAll("[data-free-filter] [data-value]").forEach((item) => {
			item.classList.toggle("is-active", item === button);
			item.setAttribute("aria-pressed", String(item === button));
		});
		drawRows();
	}));
	root.querySelectorAll("[data-free-sort] [data-value]").forEach((button) => button.addEventListener("click", () => {
		if (sortKey === button.dataset.value) descending = !descending; else { sortKey = button.dataset.value; descending = false; }
		root.querySelectorAll("[data-free-sort] [data-value]").forEach((item) => {
			item.classList.toggle("is-active", item === button);
			item.setAttribute("aria-pressed", String(item === button));
		});
		root.querySelector("[data-free-reverse]").setAttribute("aria-pressed", String(descending));
		drawRows();
	}));
	root.querySelector("[data-free-reverse]")?.addEventListener("click", (event) => {
		descending = !descending;
		event.currentTarget.classList.toggle("is-active", descending);
		event.currentTarget.setAttribute("aria-pressed", String(descending));
		drawRows();
	});
	root.querySelector("[data-free-export]").addEventListener("click", () => downloadCSV("free-order-results.csv", rows.map((r) => ({
		visit_order: "free", solver: r.solver, case: r.case, sha256: r.sha256, lower_bound: r.lower_bound, upper_bound: r.upper_bound,
		seconds: r.seconds, gap: relativeGap(r), exact: r.exact, termination: r.termination, valid: r.valid, endpoint_valid: r.endpoint_valid,
		calls: r.calls, fallback_calls: r.fallback_calls, order: r.order?.join(" "), error: r.error,
	}))));
	drawRows();
}
