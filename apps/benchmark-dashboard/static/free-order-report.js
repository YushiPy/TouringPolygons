import { escapeHTML } from "./dom.js";
import { downloadCSV } from "./format.js";
import { requestJSON } from "./api.js";
import { convexHull, pathPolygonContacts, pathPrefix, playbackDuration, projectedCase, regionColors } from "./event-geometry.js?v=2026-09-10d";
import { displayPartition } from "./native-partition.js?v=editor-align-2026-09-12a";

const reportStates = new WeakMap();
const speeds = [.25, .5, 1, 1.5, 2, 3, 4];
let markerSequence = 0;

export function relativeGap(row) {
	if (!row || !Number.isFinite(row.upper_bound) || !Number.isFinite(row.lower_bound)) return null;
	return row.upper_bound === 0 ? 0 : Math.max(0, (row.upper_bound - row.lower_bound) / row.upper_bound);
}

export function freeOrderSummary(rows, solver) {
	const selected = rows.filter((row) => row.solver === solver);
	const gaps = selected.map(relativeGap).filter((gap) => gap !== null);
	return { count: selected.length, solved: selected.filter((row) => row.exact && !row.error).length,
		seconds: selected.reduce((sum, row) => sum + (row.seconds || 0), 0),
		gap: gaps.length ? gaps.reduce((left, right) => left + right, 0) / gaps.length : null,
		errors: selected.filter((row) => row.error).length };
}

const number = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "n/a";
const percent = (value) => value === null ? "n/a" : `${(100 * value).toFixed(3)}%`;
const coordinates = (points) => points.map((point) => point.join(",")).join(" ");

function initialViewerState() {
	return { fraction: 1, speedIndex: 2, playing: false, layers: { contacts: true, decomposition: false, hulls: false, labels: true } };
}

function pathViewerHTML(row, viewerState) {
	if (!row?.geometry || !row.path) return "<p>Stored path geometry is unavailable for this result.</p>";
	return `<figure class="free-path-viewer" data-free-path-viewer>
		<div class="free-path-toolbar">
			<div class="free-player-controls">
				<button class="free-play" type="button" data-free-play aria-pressed="false"><span aria-hidden="true">▶</span><span>Play</span></button>
				<label class="free-progress-control"><span class="visually-hidden">Playback progress</span><input type="range" min="0" max="1000" value="${Math.round(viewerState.fraction * 1000)}" data-free-progress><output data-free-progress-value>${Math.round(viewerState.fraction * 100)}%</output></label>
				<div class="free-speed-control" aria-label="Playback speed"><button class="secondary" type="button" data-free-speed="-1" aria-label="Slower">−</button><output data-free-speed-value>${speeds[viewerState.speedIndex]}×</output><button class="secondary" type="button" data-free-speed="1" aria-label="Faster">+</button></div>
			</div>
			<div class="editor-layer-toggles" aria-label="Path layers">
				<button class="secondary ${viewerState.layers.contacts ? "is-active" : ""}" type="button" data-free-layer="contacts" aria-pressed="${viewerState.layers.contacts}">Visit points</button>
				<button class="secondary ${viewerState.layers.decomposition ? "is-active" : ""}" type="button" data-free-layer="decomposition" aria-pressed="${viewerState.layers.decomposition}">Decomposition</button>
				<button class="secondary ${viewerState.layers.hulls ? "is-active" : ""}" type="button" data-free-layer="hulls" aria-pressed="${viewerState.layers.hulls}">Hull</button>
				<button class="secondary ${viewerState.layers.labels ? "is-active" : ""}" type="button" data-free-layer="labels" aria-pressed="${viewerState.layers.labels}">Labels</button>
			</div>
		</div>
		<svg viewBox="0 0 840 480" class="free-path" role="img" aria-label="Stored free-order solution for case ${Number(row.case) + 1}" data-free-svg></svg>
		<figcaption><span>Region sequence</span><strong>${escapeHTML(row.order?.join(" → ") || "No regions")}</strong></figcaption>
	</figure>`;
}

function setupPathViewer(root, row, viewerState) {
	const projected = projectedCase(row);
	const svg = root.querySelector("[data-free-svg]");
	const progress = root.querySelector("[data-free-progress]");
	const progressValue = root.querySelector("[data-free-progress-value]");
	const speedValue = root.querySelector("[data-free-speed-value]");
	const play = root.querySelector("[data-free-play]");
	const order = (row.order || row.geometry.polygons.map((_, index) => index)).map(Number);
	const ranks = new Map(order.map((region, rank) => [region, rank]));
	const contacts = pathPolygonContacts(row.path, row.geometry.polygons);
	const projectedContacts = contacts.map((contact) => contact ? { ...contact, point: projected.project(contact.point) } : null);
	const markerId = `free-route-arrow-${markerSequence++}`;
	let frame = null;
	let decompositionCache = null;

	const stop = () => {
		if (frame !== null) cancelAnimationFrame(frame);
		frame = null;
		viewerState.playing = false;
		play.innerHTML = '<span aria-hidden="true">▶</span><span>Play</span>';
		play.setAttribute("aria-pressed", "false");
	};
	const draw = () => {
		const fraction = viewerState.fraction;
		const labels = viewerState.layers.labels ? projected.polygons.map((polygon, index) => {
			const rank = ranks.get(index);
			return `<text class="free-region-label" x="${polygon[0][0]}" y="${polygon[0][1] - 7}">${rank === undefined ? "·" : rank + 1}</text>`;
		}).join("") : "";
		const hulls = viewerState.layers.hulls ? row.geometry.polygons.map((polygon) => `<polygon class="free-hull" points="${coordinates(convexHull(polygon).map(projected.project))}"/>`).join("") : "";
		if (viewerState.layers.decomposition && decompositionCache === null) {
			decompositionCache = row.geometry.polygons.flatMap((polygon) => (displayPartition(polygon, draw, () => {}) || [])
				.map((piece) => `<polygon class="free-convex-piece" points="${coordinates(piece.map(projected.project))}"/>`)).join("");
		}
		const decomposition = viewerState.layers.decomposition ? decompositionCache || "" : "";
		const polygons = projected.polygons.map((polygon, index) => {
			const contact = contacts[index];
			const visited = contact !== null && fraction + 1e-9 >= contact.fraction;
			const rank = ranks.get(index) ?? index;
			const colors = regionColors(rank, Math.max(order.length, row.geometry.polygons.length), visited);
			return `<polygon class="free-region" points="${coordinates(polygon)}" style="fill:${colors.fill};stroke:${colors.stroke}"/>`;
		}).join("");
		const visitPoints = viewerState.layers.contacts ? projectedContacts.map((contact) => contact
			? `<circle class="free-contact ${fraction + 1e-9 >= contact.fraction ? "is-visited" : ""}" cx="${contact.point[0]}" cy="${contact.point[1]}" r="4"/>` : "").join("") : "";
		const prefix = pathPrefix(projected.path, fraction);
		const start = projected.project(row.geometry.start);
		const target = projected.project(row.geometry.target);
		svg.innerHTML = `<defs><marker id="${markerId}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs>${hulls}${polygons}${decomposition}<polyline class="free-route" points="${coordinates(prefix)}" marker-end="url(#${markerId})"/>${visitPoints}${labels}<circle class="free-endpoint free-start" cx="${start[0]}" cy="${start[1]}" r="5"/><circle class="free-endpoint free-target" cx="${target[0]}" cy="${target[1]}" r="5"/>${viewerState.layers.labels ? `<text class="free-endpoint-label" x="${start[0] + 9}" y="${start[1] + 4}">s</text><text class="free-endpoint-label" x="${target[0] + 9}" y="${target[1] + 4}">t</text>` : ""}`;
		progress.value = String(Math.round(fraction * 1000));
		progress.style.setProperty("--progress", `${fraction * 100}%`);
		progressValue.textContent = `${Math.round(fraction * 100)}%`;
		speedValue.textContent = `${speeds[viewerState.speedIndex]}×`;
	};
	progress.addEventListener("input", () => { stop(); viewerState.fraction = Number(progress.value) / 1000; draw(); });
	play.addEventListener("click", () => {
		if (frame !== null) { stop(); return; }
		if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { viewerState.fraction = 1; draw(); return; }
		if (viewerState.fraction >= 1) viewerState.fraction = 0;
		viewerState.playing = true;
		play.innerHTML = '<span aria-hidden="true">Ⅱ</span><span>Pause</span>';
		play.setAttribute("aria-pressed", "true");
		let previous = performance.now();
		const tick = (now) => {
			viewerState.fraction = Math.min(1, viewerState.fraction + (now - previous) * speeds[viewerState.speedIndex] / playbackDuration(order.length));
			previous = now;
			draw();
			if (viewerState.fraction < 1) frame = requestAnimationFrame(tick); else stop();
		};
		frame = requestAnimationFrame(tick);
	});
	root.querySelectorAll("[data-free-speed]").forEach((button) => button.addEventListener("click", () => {
		viewerState.speedIndex = Math.max(0, Math.min(speeds.length - 1, viewerState.speedIndex + Number(button.dataset.freeSpeed)));
		draw();
	}));
	root.querySelectorAll("[data-free-layer]").forEach((button) => button.addEventListener("click", () => {
		const layer = button.dataset.freeLayer;
		viewerState.layers[layer] = !viewerState.layers[layer];
		button.classList.toggle("is-active", viewerState.layers[layer]);
		button.setAttribute("aria-pressed", String(viewerState.layers[layer]));
		draw();
	}));
	draw();
	return stop;
}

function resultPill(row) {
	if (!row) return '<span class="table-status">n/a</span>';
	if (row.error) return `<span class="table-status is-error" title="${escapeHTML(row.error)}">Error</span>`;
	if (row.exact) return '<span class="table-status is-solved">Solved</span>';
	return `<span class="table-status is-open">${escapeHTML(row.termination || "Open")}</span>`;
}

export function renderFreeOrderReport(root, report) {
	root.classList.remove("is-hidden");
	const rows = report?.rows || [];
	const solvers = ["unordered", "tspn"].filter((solver) => rows.some((row) => row.solver === solver));
	const labels = { unordered: "Our TPP B&B", tspn: "External TSPN" };
	const groups = new Map();
	for (const row of rows) {
		if (!groups.has(row.case)) groups.set(row.case, {});
		groups.get(row.case)[row.solver] = row;
	}
	const failures = rows.filter((row) => row.solver === "tspn" && row.endpoint_valid === false).length;
	const reportKey = report?.path || report?.title || "empty";
	let state = reportStates.get(root);
	if (!state || state.reportKey !== reportKey) state = { reportKey, sortKey: "case", descending: false, tableOpen: true, openCases: new Set(), viewers: new Map(), stops: [], windowStart: 0, windowSize: 100 };
	state.stops.forEach((stop) => stop());
	state.stops = [];
	reportStates.set(root, state);
	root.innerHTML = `<header class="report-header"><div><h3>${escapeHTML(report?.title || "Free-order results")}</h3>
		<p>Free visit order · fixed endpoints · ${escapeHTML(report?.status || "No run")}</p></div>
		<button type="button" class="secondary" data-free-export>Export CSV</button></header>
		<div class="free-summary">${solvers.map((solver) => {
			const summary = freeOrderSummary(rows, solver);
			return `<section class="report-panel"><h4>${labels[solver]}</h4><strong class="free-solved">${summary.solved} / ${summary.count}</strong><p>Optimal within configured tolerance</p>
				<p>Total solve time <b>${number(summary.seconds, 2)} s</b> · Mean gap <b>${percent(summary.gap)}</b></p>
				<p>${summary.errors} errors${solver === "unordered" ? ` · ${rows.filter((row) => row.solver === solver && row.valid === true).length} independently validated paths` : ` · ${failures} endpoint check failures`}</p></section>`;
		}).join("")}</div>
		<details class="free-notes" open><summary>Method and numerical tolerances</summary>${(report?.notes || []).map((note) => `<p>${escapeHTML(note)}</p>`).join("")}</details>
		${report?.error ? `<p role="alert">${escapeHTML(report.error)}</p>` : ""}
		${rows.length ? `<details class="free-results" data-free-results ${state.tableOpen ? "open" : ""}>
			<summary><span>Instance results</span><span>${groups.size} cases · click to ${state.tableOpen ? "fold" : "expand"}</span></summary>
			<div class="free-table-wrap"><table class="free-table"><thead><tr>
				<th data-sort-column="case"><button type="button" data-free-sort-key="case">Case <span></span></button></th>
				<th data-sort-column="polygons"><button type="button" data-free-sort-key="polygons">Polygons <span></span></button></th>
				<th data-sort-column="time"><button type="button" data-free-sort-key="time">Our time <span></span></button></th>
				<th>External time</th>
				<th data-sort-column="gap"><button type="button" data-free-sort-key="gap">Our gap <span></span></button></th>
				<th>External gap</th>
				<th data-sort-column="result"><button type="button" data-free-sort-key="result">Our result <span></span></button></th>
				<th>External result</th>
			</tr></thead><tbody></tbody></table></div>
			<nav class="table-window-controls is-hidden" data-free-window aria-label="Result table window"><button class="secondary" type="button" data-free-window-step="-1">Previous</button><span data-free-window-label></span><button class="secondary" type="button" data-free-window-step="1">Next</button></nav></details>` : "<p>No free-order results yet. Run the selected campaign or open the recorded comparison.</p>"}`;

	const drawRows = () => {
		state.stops.forEach((stop) => stop());
		state.stops = [];
		const tbody = root.querySelector("tbody");
		if (!tbody) return;
		const value = ([index, pair]) => {
			const ours = pair.unordered;
			if (state.sortKey === "polygons") return ours?.polygons ?? pair.tspn?.polygons ?? 0;
			if (state.sortKey === "time") return ours?.seconds ?? Infinity;
			if (state.sortKey === "gap") return relativeGap(ours) ?? Infinity;
			if (state.sortKey === "result") return ours?.exact && !ours?.error ? 0 : 1;
			return Number(index);
		};
		const ordered = [...groups.entries()]
			.sort((left, right) => (state.descending ? -1 : 1) * (value(left) - value(right)) || Number(left[0]) - Number(right[0]));
		const maximumStart = Math.max(0, Math.floor((ordered.length - 1) / state.windowSize) * state.windowSize);
		state.windowStart = Math.min(state.windowStart, maximumStart);
		const visible = ordered.slice(state.windowStart, state.windowStart + state.windowSize);
		const windowControls = root.querySelector("[data-free-window]");
		if (windowControls) {
			windowControls.classList.toggle("is-hidden", ordered.length <= state.windowSize);
			windowControls.querySelector("[data-free-window-label]").textContent = `${state.windowStart + 1}–${Math.min(ordered.length, state.windowStart + state.windowSize)} of ${ordered.length}`;
			windowControls.querySelector('[data-free-window-step="-1"]').disabled = state.windowStart === 0;
			windowControls.querySelector('[data-free-window-step="1"]').disabled = state.windowStart + state.windowSize >= ordered.length;
		}
		tbody.innerHTML = visible.map(([index, pair]) => {
			const ours = pair.unordered, external = pair.tspn, key = String(index), isOpen = state.openCases.has(key);
			const viewerState = state.viewers.get(key) || initialViewerState();
			state.viewers.set(key, viewerState);
			return `<tr class="free-result-row ${isOpen ? "is-open" : ""}"><td><button type="button" class="free-view-path" data-free-case="${index}" aria-label="${isOpen ? "Hide" : "View"} path for case ${Number(index) + 1}" aria-expanded="${isOpen}"><span>${Number(index) + 1}</span><small>${isOpen ? "Hide" : "View"} path</small></button></td><td>${ours?.polygons ?? external?.polygons ?? ""}</td>
				<td>${number(ours?.seconds)}</td><td>${number(external?.seconds)}</td><td>${percent(relativeGap(ours))}</td><td>${percent(relativeGap(external))}</td>
				<td>${resultPill(ours)}</td><td>${resultPill(external)}${external?.endpoint_valid === false ? '<span class="endpoint-warning" title="Endpoint validation failed">!</span>' : ""}</td></tr>
				<tr data-free-detail="${index}" class="free-detail-row ${isOpen ? "" : "is-hidden"}"><td colspan="8"><div class="free-detail-content">${pair.visualizationError ? `<p role="alert">${escapeHTML(pair.visualizationError)}</p>` : pathViewerHTML(ours || external, viewerState)}<p class="free-detail-stats">Our bounds: ${number(ours?.lower_bound, 8)} ≤ optimum ≤ ${number(ours?.upper_bound, 8)} · ${ours?.calls ?? "n/a"} convex calls · ${ours?.fallback_calls ?? "n/a"} fallback calls</p></div></td></tr>`;
		}).join("");
		root.querySelectorAll("[data-sort-column]").forEach((header) => {
			const active = header.dataset.sortColumn === state.sortKey;
			header.setAttribute("aria-sort", active ? (state.descending ? "descending" : "ascending") : "none");
			header.querySelector("span").textContent = active ? (state.descending ? "↓" : "↑") : "↕";
		});
		tbody.querySelectorAll("[data-free-case]").forEach((button) => button.addEventListener("click", async () => {
			const key = String(button.dataset.freeCase);
			if (state.openCases.has(key)) {
				state.openCases.delete(key);
			} else {
				const pair = groups.get(Number(key)) || groups.get(key);
				const candidate = pair?.unordered || pair?.tspn;
				if (candidate && (!candidate.geometry || !candidate.path) && candidate.visualization_available && report.visualization_endpoint) {
					button.disabled = true;
					button.querySelector("small").textContent = "Loading…";
					try {
						const detail = await requestJSON(`${report.visualization_endpoint}/${encodeURIComponent(key)}`);
						for (const loaded of detail.rows || []) {
							const target = rows.find((row) => String(row.case) === key && row.solver === loaded.solver);
							if (target) Object.assign(target, loaded);
							if (pair) pair[loaded.solver] = target || loaded;
						}
					} catch (error) {
						if (pair) pair.visualizationError = error.message;
					}
				}
				state.openCases.add(key);
			}
			drawRows();
		}));
		for (const key of state.openCases) {
			const detail = tbody.querySelector(`[data-free-detail="${CSS.escape(key)}"]`);
			if (!detail) continue;
			const pair = groups.get(Number(key)) || groups.get(key);
			const viewer = detail.querySelector("[data-free-path-viewer]");
			if (viewer && pair) state.stops.push(setupPathViewer(viewer, pair.unordered || pair.tspn, state.viewers.get(key)));
		}
	};

	root.querySelector("[data-free-results]")?.addEventListener("toggle", (event) => { state.tableOpen = event.currentTarget.open; });
	root.querySelectorAll("[data-free-sort-key]").forEach((button) => button.addEventListener("click", () => {
		if (state.sortKey === button.dataset.freeSortKey) state.descending = !state.descending;
		else { state.sortKey = button.dataset.freeSortKey; state.descending = false; }
		state.windowStart = 0;
		drawRows();
	}));
	root.querySelectorAll("[data-free-window-step]").forEach((button) => button.addEventListener("click", () => {
		state.windowStart = Math.max(0, state.windowStart + Number(button.dataset.freeWindowStep) * state.windowSize);
		drawRows();
		root.querySelector(".free-table-wrap")?.scrollIntoView({ block: "nearest" });
	}));
	root.querySelector("[data-free-export]").addEventListener("click", () => downloadCSV("free-order-results.csv", rows.map((row) => ({
		visit_order: "free", solver: row.solver, case: row.case, sha256: row.sha256, lower_bound: row.lower_bound, upper_bound: row.upper_bound,
		seconds: row.seconds, gap: relativeGap(row), exact: row.exact, termination: row.termination, valid: row.valid, endpoint_valid: row.endpoint_valid,
		calls: row.calls, fallback_calls: row.fallback_calls, order: row.order?.join(" "), error: row.error,
	}))));
	drawRows();
}
