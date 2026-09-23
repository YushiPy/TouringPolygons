import { tppSolveConvex } from "./tpp-solver.js";

const $ = (selector) => document.querySelector(selector);

function setOutput(target, text) {
	if (!target) {
		return;
	}
	const next = text || "";
	if (target.textContent === next) return;
	const selection = window.getSelection?.();
	if (selection && !selection.isCollapsed && selection.rangeCount > 0 && target.contains(selection.anchorNode)) {
		target.dataset.pendingOutput = next;
		return;
	}
	target.textContent = target.dataset.pendingOutput || next;
	delete target.dataset.pendingOutput;
}

function escapeHTML(value) {
	return String(value ?? "").replace(/[&<>"']/g, (character) => ({
		"&": "&amp;",
		"<": "&lt;",
		">": "&gt;",
		'"': "&quot;",
		"'": "&#39;",
	})[character]);
}

function highlightPseudocode(source) {
	const pattern = /(?<![\p{L}_])(para|enquanto|senão|se|continuar|retornar)(?![\p{L}_])|([\p{L}_]+)(?=\()|([←≥≠∉∅])/gu;
	let result = "", cursor = 0;
	for (const match of source.matchAll(pattern)) {
		result += escapeHTML(source.slice(cursor, match.index));
		const kind = match[1] ? "code-keyword" : match[2] ? "code-function" : "code-operator";
		result += `<span class="${kind}">${escapeHTML(match[0])}</span>`;
		cursor = match.index + match[0].length;
	}
	return result + escapeHTML(source.slice(cursor));
}

function formatDuration(value) {
	const seconds = Number(value);
	if (!Number.isFinite(seconds)) return "—";
	if (seconds < 0.001) return "< 0,001 s";
	if (seconds < 60) return `${number(seconds, 3)} s`;
	const minutes = seconds / 60;
	if (minutes < 60) return `${number(minutes, 2)} min`;
	return `${number(minutes / 60, 2)} horas`;
}

function formatDistance(value) {
	const distance = Number(value);
	if (!Number.isFinite(distance)) return "—";
	if (distance === 0) return "0";
	if (Math.abs(distance) >= 0.01) return number(distance, 3);
	return distance.toExponential(2).replace(".", ",");
}

function convexHull(points) {
	const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
	const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
	const half = (values) => {
		const hull = [];
		for (const point of values) {
			while (hull.length > 1 && cross(hull.at(-2), hull.at(-1), point) <= 0) hull.pop();
			hull.push(point);
		}
		return hull.slice(0, -1);
	};
	return [...half(sorted), ...half([...sorted].reverse())];
}

function polygonCentroid(polygon) {
	let twiceArea = 0;
	let centroid = [0, 0];
	for (let index = 0; index < polygon.length; index += 1) {
		const current = polygon[index], next = polygon[(index + 1) % polygon.length];
		const cross = current[0] * next[1] - next[0] * current[1];
		twiceArea += cross;
		centroid[0] += (current[0] + next[0]) * cross;
		centroid[1] += (current[1] + next[1]) * cross;
	}
	if (Math.abs(twiceArea) > 1e-12) return [centroid[0] / (3 * twiceArea), centroid[1] / (3 * twiceArea)];
	return polygon.reduce((sum, point) => [sum[0] + point[0] / polygon.length, sum[1] + point[1] / polygon.length], [0, 0]);
}

function projectedCase(row, width = 840, height = 480) {
	const points = [...row.geometry.polygons.flat(), ...row.path, row.geometry.start, row.geometry.target];
	const xs = points.map((point) => point[0]), ys = points.map((point) => point[1]);
	const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
	const scale = Math.min((width - 80) / Math.max(maxX - minX, 1e-9), (height - 80) / Math.max(maxY - minY, 1e-9));
	const offsetX = (width - (maxX - minX) * scale) / 2, offsetY = (height - (maxY - minY) * scale) / 2;
	const project = ([x, y]) => [offsetX + (x - minX) * scale, height - offsetY - (y - minY) * scale];
	return { project, path: row.path.map(project), polygons: row.geometry.polygons.map((polygon) => polygon.map(project)) };
}

function pathPrefix(path, fraction) {
	if (!path.length) return [];
	const distances = path.slice(1).map((point, index) => Math.hypot(point[0] - path[index][0], point[1] - path[index][1]));
	let remaining = Math.max(0, Math.min(1, fraction)) * distances.reduce((sum, value) => sum + value, 0);
	const prefix = [path[0]];
	for (let index = 0; index < distances.length; index += 1) {
		if (remaining >= distances[index]) {
			prefix.push(path[index + 1]);
			remaining -= distances[index];
		} else {
			const ratio = remaining / distances[index];
			prefix.push(path[index].map((value, axis) => value + ratio * (path[index + 1][axis] - value)));
			break;
		}
	}
	return prefix;
}

function closestPointOnSegment(point, start, end) {
	const direction = [end[0] - start[0], end[1] - start[1]];
	const lengthSquared = direction[0] ** 2 + direction[1] ** 2;
	const fraction = lengthSquared > 1e-18
		? Math.max(0, Math.min(1, ((point[0] - start[0]) * direction[0] + (point[1] - start[1]) * direction[1]) / lengthSquared))
		: 0;
	return [start[0] + fraction * direction[0], start[1] + fraction * direction[1]];
}

function squaredDistance(left, right) {
	return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2;
}

function closestSegmentPair(firstStart, firstEnd, secondStart, secondEnd) {
	const firstDirection = [firstEnd[0] - firstStart[0], firstEnd[1] - firstStart[1]];
	const secondDirection = [secondEnd[0] - secondStart[0], secondEnd[1] - secondStart[1]];
	const offset = [secondStart[0] - firstStart[0], secondStart[1] - firstStart[1]];
	const cross = (left, right) => left[0] * right[1] - left[1] * right[0];
	const denominator = cross(firstDirection, secondDirection);
	if (Math.abs(denominator) > 1e-12) {
		const firstFraction = cross(offset, secondDirection) / denominator;
		const secondFraction = cross(offset, firstDirection) / denominator;
		if (firstFraction >= 0 && firstFraction <= 1 && secondFraction >= 0 && secondFraction <= 1) {
			const point = [firstStart[0] + firstFraction * firstDirection[0], firstStart[1] + firstFraction * firstDirection[1]];
			return { first: point, second: point, distance: 0 };
		}
	}
	const candidates = [
		{ first: firstStart, second: closestPointOnSegment(firstStart, secondStart, secondEnd) },
		{ first: firstEnd, second: closestPointOnSegment(firstEnd, secondStart, secondEnd) },
		{ first: closestPointOnSegment(secondStart, firstStart, firstEnd), second: secondStart },
		{ first: closestPointOnSegment(secondEnd, firstStart, firstEnd), second: secondEnd },
	].map((candidate) => ({ ...candidate, distance: Math.sqrt(squaredDistance(candidate.first, candidate.second)) }));
	return candidates.reduce((best, candidate) => candidate.distance < best.distance ? candidate : best);
}

function closestPathPolygonConnection(path, polygon) {
	if (!Array.isArray(path) || path.length === 0 || !Array.isArray(polygon) || polygon.length === 0) return null;
	const pathSegments = path.length > 1 ? path.slice(1).map((point, index) => [path[index], point]) : [[path[0], path[0]]];
	const polygonSegments = polygon.length > 1
		? polygon.map((point, index) => [point, polygon[(index + 1) % polygon.length]])
		: [[polygon[0], polygon[0]]];
	let best = null;
	for (const [firstStart, firstEnd] of pathSegments) {
		for (const [secondStart, secondEnd] of polygonSegments) {
			const candidate = closestSegmentPair(firstStart, firstEnd, secondStart, secondEnd);
			if (!best || candidate.distance < best.distance) best = candidate;
		}
	}
	return best;
}

function pointOnSegment(point, start, end, epsilon = 1e-9) {
	const cross = (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0]);
	if (Math.abs(cross) > epsilon * Math.max(1, Math.hypot(end[0] - start[0], end[1] - start[1]))) return false;
	return point[0] >= Math.min(start[0], end[0]) - epsilon && point[0] <= Math.max(start[0], end[0]) + epsilon
		&& point[1] >= Math.min(start[1], end[1]) - epsilon && point[1] <= Math.max(start[1], end[1]) + epsilon;
}

function decompositionEdges(pieces, boundary) {
	const edges = [];
	const seen = new Set();
	const pointKey = (point) => point.map((value) => Number(value).toFixed(8)).join(",");
	const edgeKey = (start, end) => [pointKey(start), pointKey(end)].sort().join("|");
	const isBoundaryEdge = (start, end) => boundary.some((point, index) => pointOnSegment(start, point, boundary[(index + 1) % boundary.length], 1e-8)
		&& pointOnSegment(end, point, boundary[(index + 1) % boundary.length], 1e-8));
	for (const piece of pieces || []) {
		if (!Array.isArray(piece) || piece.length < 2) continue;
		for (let index = 0; index < piece.length; index += 1) {
			const start = piece[index], end = piece[(index + 1) % piece.length];
			if (isBoundaryEdge(start, end)) continue;
			const key = edgeKey(start, end);
			if (seen.has(key)) continue;
			seen.add(key);
			edges.push([start, end]);
		}
	}
	return edges;
}

function decompositionLinesMarkup(decomposition, polygons, project) {
	return (decomposition || []).flatMap((pieces, index) => decompositionEdges(pieces, polygons?.[index] || []).map(([start, end]) => {
		const first = project(start), second = project(end);
		return `<line class="convex-piece" x1="${first[0]}" y1="${first[1]}" x2="${second[0]}" y2="${second[1]}"/>`;
	})).join("");
}

function pointInPolygon(point, polygon) {
	let inside = false;
	for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
		const a = polygon[previous], b = polygon[index];
		if (pointOnSegment(point, a, b)) return true;
		if ((a[1] > point[1]) !== (b[1] > point[1])
			&& point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
	}
	return inside;
}

function segmentIntersectionFraction(start, end, a, b, epsilon = 1e-9) {
	const direction = [end[0] - start[0], end[1] - start[1]];
	const edge = [b[0] - a[0], b[1] - a[1]];
	const offset = [a[0] - start[0], a[1] - start[1]];
	const cross = (left, right) => left[0] * right[1] - left[1] * right[0];
	const denominator = cross(direction, edge);
	if (Math.abs(denominator) <= epsilon) {
		if (Math.abs(cross(offset, direction)) > epsilon) return null;
		const lengthSquared = direction[0] ** 2 + direction[1] ** 2;
		if (lengthSquared <= epsilon ** 2) return pointOnSegment(start, a, b, epsilon) ? 0 : null;
		const first = ((a[0] - start[0]) * direction[0] + (a[1] - start[1]) * direction[1]) / lengthSquared;
		const second = ((b[0] - start[0]) * direction[0] + (b[1] - start[1]) * direction[1]) / lengthSquared;
		const entry = Math.max(0, Math.min(first, second));
		return entry <= Math.min(1, Math.max(first, second)) + epsilon ? entry : null;
	}
	const alongPath = cross(offset, edge) / denominator;
	const alongEdge = cross(offset, direction) / denominator;
	return alongPath >= -epsilon && alongPath <= 1 + epsilon && alongEdge >= -epsilon && alongEdge <= 1 + epsilon
		? Math.max(0, Math.min(1, alongPath)) : null;
}

function pathPolygonContacts(path, polygons) {
	const lengths = path.slice(1).map((point, index) => Math.hypot(point[0] - path[index][0], point[1] - path[index][1]));
	const total = lengths.reduce((sum, length) => sum + length, 0);
	return polygons.map((polygon) => {
		let traversed = 0;
		for (let index = 0; index < path.length - 1; index += 1) {
			const start = path[index], end = path[index + 1], length = lengths[index];
			let entry = pointInPolygon(start, polygon) ? 0 : null;
			for (let edge = 0; edge < polygon.length; edge += 1) {
				const candidate = segmentIntersectionFraction(start, end, polygon[edge], polygon[(edge + 1) % polygon.length]);
				if (candidate !== null && (entry === null || candidate < entry)) entry = candidate;
			}
			if (entry !== null) {
				return {
					point: [start[0] + entry * (end[0] - start[0]), start[1] + entry * (end[1] - start[1])],
					fraction: total > 0 ? (traversed + entry * length) / total : 0,
				};
			}
			traversed += length;
		}
		if (path.length && pointInPolygon(path.at(-1), polygon)) return { point: [...path.at(-1)], fraction: 1 };
		return null;
	});
}

function filterRows(rows, status, search) {
	const term = search.trim();
	return rows.filter((row) => (status === "all" || row.termination === status)
		&& (!term || String(row.case).padStart(2, "0").includes(term)));
}

function endpointOffset(path, target, distance) {
	const points = target ? [...path].reverse() : path;
	const a = points[0];
	for (const b of points.slice(1)) {
		const length = Math.hypot(a[0] - b[0], a[1] - b[1]);
		if (length > 1e-10) return [a[0] + distance * (a[0] - b[0]) / length, a[1] + distance * (a[1] - b[1]) / length];
	}
	return [a[0] + (target ? distance : -distance), a[1]];
}

function regionColors(rank, count, visited) {
	const position = count > 1 ? rank / (count - 1) : 0;
	return visited
		? { fill: `hsl(${105 + position * 70} 60% ${62 - position * 32}% / .62)`, stroke: `hsl(${105 + position * 70} 65% 68%)` }
		: { fill: `hsl(${195 + position * 65} 60% ${65 - position * 30}% / .4)`, stroke: `hsl(${195 + position * 65} 55% 70%)` };
}

function sortRows(rows, key = "case", descending = false) {
	const value = (row) => row[key];
	return [...rows].sort((a, b) => (descending ? -1 : 1) * (value(a) - value(b)) || a.case - b.case);
}

function playbackDuration(polygons) {
	return 1800 + 15 * Math.min(80, Math.max(0, polygons));
}

function toggleOrderRegion(order, index) {
	return order.includes(index) ? order.filter(value => value !== index) : [...order, index];
}

function toggleChoice(current, next) {
	return current === next ? null : next;
}


const element = (id) => document.getElementById(id);
const number = (value, digits = 4) => value.toLocaleString("pt-BR", { maximumFractionDigits: digits });
const coordinates = (points) => points.map((point) => point.join(",")).join(" ");
const caseLabel = (id) => String(id + 1).padStart(id + 1 >= 100 ? 3 : 2, "0");
const visitorRows = (rows, query) => {
	const term = query.trim();
	if (!term) return rows;
	const requestedCase = /^\d+$/.test(term) ? Number(term) - 1 : null;
	return rows.filter((row) => (requestedCase !== null && row.case === requestedCase) || caseLabel(row.case).includes(term));
};
const MAX_ZOOM = 8;

function showModalWithTransition(dialog) {
	if (!dialog || dialog.open) return;
	dialog.classList.remove("is-entering");
	dialog.showModal();
	if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
	dialog.classList.add("is-entering");
	requestAnimationFrame(() => dialog.classList.remove("is-entering"));
}

function traceNumber(value, digits = 4) {
	return value === null || value === undefined || value === "" ? "—" : Number.isFinite(Number(value)) ? number(Number(value), digits) : "—";
}

function pathLength(path) {
	return Array.isArray(path) ? path.slice(1).reduce((total, point, index) => total + Math.hypot(point[0] - path[index][0], point[1] - path[index][1]), 0) : null;
}

function solveChallengeRoute(start, target, polygons) {
	const vectorPath = tppSolveConvex(start, target, polygons, true);
	const path = vectorPath.map(({ x, y }) => [x, y]);
	return { path, length: pathLength(path) };
}

const TRACE_PATH_EPSILON = 1e-5;

function tracePathsDiffer(left, right, epsilon = TRACE_PATH_EPSILON) {
	if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return true;
	return left.some((point, index) => Math.hypot(point[0] - right[index][0], point[1] - right[index][1]) > epsilon);
}

function filterTraceEvents(events) {
	let previousPath = null;
	return events.filter((event) => {
		const keep = event.kind !== "heuristic_contact_pass" || !previousPath || tracePathsDiffer(previousPath, event.path);
		if (Array.isArray(event.path) && event.path.length) previousPath = event.path;
		return keep;
	});
}

function traceLabel(order, original) {
	const rank = Array.isArray(order) ? order.indexOf(Number(original)) : -1;
	return rank >= 0 ? String(rank + 1) : String(Number(original) + 1);
}

function traceLabels(order, sequence) {
	return Array.isArray(sequence) && sequence.length ? sequence.map((index) => traceLabel(order, index)).join(" → ") : "∅";
}

function traceEscape(value) {
	return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function animateChallengeRoute(map) {
	const route = map.querySelector(".challenge-solution-route");
	if (!route || !route.animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
	const length = route.getTotalLength();
	route.style.strokeDasharray = `${length} ${length}`;
	route.style.strokeDashoffset = String(length);
	const animation = route.animate([{ strokeDashoffset: String(length) }, { strokeDashoffset: "0" }], {
		duration: 2000,
		easing: "cubic-bezier(.3,.65,.25,1)",
		fill: "forwards",
	});
	animation.finished.then(() => {
		route.style.strokeDasharray = "none";
		route.style.strokeDashoffset = "0";
		animation.cancel();
	}).catch(() => {});
}

function challengeComparison(chosenLabel, chosenLength, bestLabel, bestLength, choiceLabel, chosenChoice, bestChoice) {
	return `<table class="challenge-comparison"><thead><tr><td></td><th><i class="comparison-line chosen" aria-hidden="true"></i>${chosenLabel}</th><th><i class="comparison-line reference" aria-hidden="true"></i>${bestLabel}</th></tr></thead><tbody><tr><th scope="row">Comprimento</th><td>${number(chosenLength, 2)}</td><td>${number(bestLength, 2)}</td></tr><tr><th scope="row">${choiceLabel}</th><td>${chosenChoice}</td><td>${bestChoice}</td></tr></tbody></table><p class="comparison-note">Comprimentos em unidades deste exemplo.</p>`;
}

function challengeVerdict(chosenLength, bestLength, name) {
	const percent = 100 * (chosenLength / bestLength - 1);
	if (percent < 1e-5) return `Você encontrou uma das melhores ${name}!`;
	if (percent < 1) return "Seu caminho ficou a menos de 1% do menor encontrado.";
	return `Seu caminho ficou ${number(percent, 1)}% mais longo.`;
}

function challengeGreedyNote(solution, best, description, sequence) {
	return `<p class="challenge-local-note">Escolha local: ${description} daria ${sequence} e um caminho ${number(100 * (solution.length / best.length - 1), 1)}% mais longo que o melhor.</p>`;
}

async function initialize() {
	const data = window.TPPEventData;
	const uspDemo = window.TPPUspDemo;
	const tracePayload = window.TPPTraceData || { schema_version: 1, cases: {} };
	const traces = tracePayload.schema_version === 1 ? (tracePayload.cases || {}) : {};
	const shortTraceCases = Object.values(traces)
		.filter((trace) => trace.omitted_events === 0 && trace.event_count < 200)
		.sort((a, b) => a.case - b.case);
	if (data.schema_version !== 1 || !data.rows.length || uspDemo?.schema_version !== 1 || uspDemo.case !== "usp") throw new Error("Dados da demonstração indisponíveis.");
	const defaultCase = "usp";
	let row = uspDemo;
	let traceRow = data.rows.find((item) => item.case === 3)
		|| data.rows.find((item) => item.case === shortTraceCases[0]?.case)
		|| data.rows[0];
	let projected, fraction = 1, zoom = 1, frame = 0, playing = false;
	let routeWidth = 840, routeHeight = 480;
	let traceEvents = [], traceIndex = 0, traceFrame = null, traceTransition = null, tracePlaying = false;
	let pan = [0, 0], drag = null, traceZoom = 1, tracePan = [0, 0], traceDrag = null;
	const speeds = [.25, .5, 1, 1.5, 2, 3, 4];
	let speedIndex = 2;
	let showAllResults = false;
	const sorting = { picker: { key: "case", descending: false }, result: { key: "case", descending: false } };
	const enabled = (id) => element(id).getAttribute("aria-pressed") === "true";
	const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mapContent = element("map-content");
	const map = element("route-map");
	const traceMapContent = element("trace-map-content");
	const traceMap = element("trace-map");
	const hasTraceUI = Boolean(traceMapContent);
	const traceCode = [
		[0, "U ← heurística(regiões, s, t)"],
		[0, "fila ← {raiz(∅)}"],
		[0, "enquanto fila ≠ ∅:"],
		[1, "n ← melhor(fila)"],
		[1, "(L, r) ← convexo(n)"],
		[1, "se L ≥ U − ε: continuar"],
		[1, "F ← não_visitados(r)"],
		[1, "se F = ∅:"],
		[2, "U ← min(U, comprimento(r))"],
		[2, "continuar"],
		[1, "P ← mais_distante(r, F)"],
		[1, "se P ∉ n.sequência:"],
		[2, "para i em posições(n):"],
		[3, "filho ← inserir(n, fecho(P), i)"],
		[3, "se promissor: enfileirar(filho)"],
		[1, "senão:"],
		[2, "para C em decompor(P):"],
		[3, "filho ← refinar(n, P, C)"],
		[3, "se promissor: enfileirar(filho)"],
		[0, "retornar (U, L_global, status)"],
	];
	if (hasTraceUI) element("trace-code-lines").innerHTML = traceCode.map(([indent, line], index) => `<li data-code-line="${index}" style="--indent:${indent * .85}em"><code>${highlightPseudocode(line)}</code></li>`).join("");
	let activeCodeLine = -1;

	function updateTraceCode(event) {
		const kind = event?.kind;
		const line = kind?.startsWith("heuristic_") || (kind === "incumbent" && event.source === "heuristic") ? 0
			: kind === "root" ? 1
			: kind === "expand" ? 3
			: kind === "oracle" ? 4
			: kind === "prune" ? 5
			: kind === "incumbent" ? 8
			: kind === "branch" ? (event.reason === "decomposition" ? 16 : 10)
			: kind === "child" ? (event.piece === undefined ? 14 : 18)
			: kind === "complete" ? 19 : -1;
		if (line === activeCodeLine) return;
		activeCodeLine = line;
		element("trace-code-lines").querySelectorAll("li").forEach((item, index) => {
			item.classList.toggle("active", index === line);
			if (index === line) item.setAttribute("aria-current", "step");
			else item.removeAttribute("aria-current");
		});
		if (line >= 0) {
			const item = element("trace-code-lines").children[line];
			const viewport = element("trace-code-scroll");
			const itemBox = item.getBoundingClientRect(), viewportBox = viewport.getBoundingClientRect();
			if (itemBox.top < viewportBox.top + 8) viewport.scrollTop += itemBox.top - viewportBox.top - 8;
			else if (itemBox.bottom > viewportBox.bottom - 8) viewport.scrollTop += itemBox.bottom - viewportBox.bottom + 8;
		}
	}

	function tracePickerLabel(trace) {
		const candidate = data.rows.find((item) => item.case === trace.case);
		const displayedSteps = Array.isArray(trace.events) ? filterTraceEvents(trace.events).length : trace.event_count;
		return `Caso ${caseLabel(trace.case)} · ${displayedSteps} passos · ${candidate?.polygons ?? "—"} regiões`;
	}

	function renderTracePicker() {
		const options = element("trace-picker-options");
		if (!options) return;
		const query = element("trace-picker-search")?.value.trim() || "";
		const rows = shortTraceCases.filter((trace) => !query || caseLabel(trace.case).includes(query));
		element("trace-picker-count").textContent = `${rows.length} simulações disponíveis`;
		options.innerHTML = rows.length ? rows.map((trace) => `<button type="button" data-trace-case="${trace.case}" aria-pressed="${trace.case === traceRow.case}"><span><strong>${tracePickerLabel(trace)}</strong><small>Árvore registrada para acompanhar passo a passo</small></span></button>`).join("") : "<p>Nenhuma simulação encontrada. Experimente outro número.</p>";
	}

	function populateCaseSelect() {
		const select = element("case-select");
		if (!select) return;
		select.innerHTML = data.rows.map((item) => `<option value="${item.case}">Caso ${caseLabel(item.case)} · ${item.polygons} regiões</option>`).join("");
	}

	function updateResultSummary() {
		const summary = data.summary || {};
		const comparison = data.comparison || {};
		const cases = Number(summary.cases || data.rows.length);
		const certified = Number(summary.exact_certified || 0);
		const underTen = Number(summary.resolved_under_10_seconds || 0);
		const medianSpeedup = Number(comparison.median_speedup_fekete_over_ours);
		const common = Number(comparison.common_completed || 0);
		const comparisonTime = comparison.time || {};
		const comparisonPrecision = comparison.precision || {};
		const certifiedElement = element("result-certified");
		const underTenElement = element("result-under-10");
		const speedupElement = element("result-speedup");
		const fasterElement = element("result-faster");
		if (certifiedElement) certifiedElement.innerHTML = `${certified}<span>/ ${cases}</span>`;
		if (underTenElement) underTenElement.innerHTML = `${underTen}<span>/ ${cases}</span>`;
		if (speedupElement && Number.isFinite(medianSpeedup)) speedupElement.textContent = `${number(medianSpeedup, 2)}×`;
		if (fasterElement) fasterElement.innerHTML = `${Number(comparison.ours_faster_count || 0)}<span>/ ${common}</span>`;
		const note = element("result-benchmark-note");
		if (note) note.textContent = `Corpus de instâncias usado por Fekete et al. no artigo. Nosso solver concluiu ${certified}/${cases}; no conjunto comum concluído, o speedup mediano Fekete/nosso foi ${number(medianSpeedup, 2)}×. O solver de Fekete et al. não concluiu ${comparison.fekete_unresolved ?? "—"} instâncias no limite de 6 horas.`;
		const setText = (id, value) => { const target = element(id); if (target) target.textContent = value; };
		const duration = (value) => formatDuration(value);
		const hours = (value) => Number.isFinite(Number(value)) ? `${number(Number(value), 2)} h` : "—";
		const fraction = (value, total) => `${Number(value ?? 0)}/${Number(total ?? 0)}`;
		setText("comparison-time-ours-median", duration(comparisonTime.ours_median_seconds));
		setText("comparison-time-fekete-median", duration(comparisonTime.fekete_median_seconds));
		setText("comparison-time-ours-mean", duration(comparisonTime.ours_mean_seconds));
		setText("comparison-time-fekete-mean", duration(comparisonTime.fekete_mean_seconds));
		setText("comparison-time-ours-total", hours(comparisonTime.ours_total_hours));
		setText("comparison-time-fekete-total", hours(comparisonTime.fekete_total_hours));
		setText("comparison-time-common-ours", duration(comparisonTime.common_ours_median_seconds));
		setText("comparison-time-common-fekete", duration(comparisonTime.common_fekete_median_seconds));
		setText("comparison-speedup-median", Number.isFinite(medianSpeedup) ? `${number(medianSpeedup, 2)}×` : "—");
		setText("comparison-speedup-geometric", Number.isFinite(Number(comparison.geometric_mean_speedup)) ? `${number(comparison.geometric_mean_speedup, 2)}×` : "—");
		setText("comparison-speedup-ours-faster", fraction(comparison.ours_faster_count, common));
		setText("comparison-speedup-fekete-faster", fraction(comparison.fekete_faster_count, common));
		setText("comparison-precision-ours-median", formatDistance(comparisonPrecision.ours_max_per_instance?.median));
		setText("comparison-precision-fekete-median", formatDistance(comparisonPrecision.fekete_raw_max_per_instance?.median));
		setText("comparison-precision-ours-p95", formatDistance(comparisonPrecision.ours_max_per_instance?.p95));
		setText("comparison-precision-fekete-p95", formatDistance(comparisonPrecision.fekete_raw_max_per_instance?.p95));
		const oursPrecisionCount = Number(comparisonPrecision.ours_max_per_instance?.n || cases);
		const feketePrecisionCount = Number(comparisonPrecision.fekete_raw_max_per_instance?.n || 0);
		setText("comparison-precision-ours-threshold", fraction(oursPrecisionCount - Number(comparisonPrecision["ours_instances_at_1e-7"] || 0), oursPrecisionCount));
		setText("comparison-precision-fekete-threshold", fraction(feketePrecisionCount - Number(comparisonPrecision["fekete_raw_instances_at_1e-7"] || 0), feketePrecisionCount));
		setText("comparison-precision-fekete-n", comparisonPrecision.fekete_raw_max_per_instance?.n ?? "—");
	}

	function populateTracePicker() {
		const picker = element("trace-case-select");
		if (picker) picker.innerHTML = `<option value="">Escolha uma simulação curta</option>${shortTraceCases.map((trace) => `<option value="${trace.case}">${tracePickerLabel(trace)}</option>`).join("")}`;
		const customPicker = element("trace-case-picker");
		if (!customPicker) return;
		customPicker.hidden = false;
		renderTracePicker();
	}

	function cancelTraceFrame() {
		if (traceFrame !== null) window.cancelAnimationFrame(traceFrame);
		traceFrame = null;
	}

	function stopTrace() {
		if (!hasTraceUI) return;
		tracePlaying = false;
		cancelTraceFrame();
		traceTransition = null;
		element("trace-play").textContent = "▶ Reproduzir";
		element("trace-play").setAttribute("aria-pressed", "false");
	}

	function traceCurrentPath(eventIndex = traceIndex) {
		const event = traceEvents[eventIndex];
		if (Array.isArray(event?.path) && event.path.length) return event.path;
		const synthetic = traceSyntheticChildPath(event, eventIndex);
		if (synthetic) return synthetic;
		for (let index = eventIndex - 1; index >= 0; index -= 1) {
			if (Array.isArray(traceEvents[index]?.path) && traceEvents[index].path.length) return traceEvents[index].path;
		}
		return traceRow.path;
	}

	function traceIncumbent(index) {
		for (let eventIndex = index; eventIndex >= 0; eventIndex -= 1) {
			const event = traceEvents[eventIndex];
			const value = Number(event?.upper_bound);
			if (Number.isFinite(value)) return value;
			if (event?.kind === "incumbent" && Number.isFinite(Number(event.length))) return Number(event.length);
		}
		return null;
	}

	function traceIncumbentPath(index) {
		for (let eventIndex = index; eventIndex >= 0; eventIndex -= 1) {
			const event = traceEvents[eventIndex];
			if (event?.kind === "incumbent" && Array.isArray(event.path) && event.path.length) return event.path;
		}
		return null;
	}

	function tracePathForSequence(sequence, endIndex) {
		const key = sequence.join(",");
		for (let eventIndex = endIndex; eventIndex >= 0; eventIndex -= 1) {
			const event = traceEvents[eventIndex];
			if (traceSequence(event).join(",") === key && Array.isArray(event.path) && event.path.length) return event.path;
		}
		return null;
	}

	function traceSyntheticChildPath(event, eventIndex) {
		if (event?.kind !== "child" || event.reason !== "bound" || !Array.isArray(event.sequence)) return null;
		const polygonIndex = Number(event.polygon), position = Number(event.position);
		if (!Number.isInteger(polygonIndex) || !Number.isInteger(position) || !traceRow.geometry.polygons[polygonIndex]) return null;
		const parentSequence = [...event.sequence];
		parentSequence.splice(position, 1);
		const parentPath = tracePathForSequence(parentSequence, eventIndex - 1);
		if (!parentPath || parentPath.length < 2) return null;
		const segmentIndex = Math.max(0, Math.min(parentPath.length - 2, position));
		const connection = closestPathPolygonConnection([parentPath[segmentIndex], parentPath[segmentIndex + 1]], traceRow.geometry.polygons[polygonIndex]);
		if (!connection) return null;
		const path = parentPath.map((point) => [...point]);
		path.splice(segmentIndex + 1, 0, connection.second);
		return path;
	}

	function traceProjectedPath(index) {
		const projection = projectedCase(traceRow);
		return traceCurrentPath(index).map(projection.project);
	}

	function traceBranchGeometry(event, path, polygons, originalPath, reached) {
		if (event?.kind !== "branch" || event.reason !== "insertion") return null;
		const connections = polygons.map((polygon, polygonIndex) => {
			if (reached[polygonIndex]) return null;
			const connection = closestPathPolygonConnection(path, polygon);
			const originalConnection = closestPathPolygonConnection(originalPath, traceRow.geometry.polygons[polygonIndex]);
			return connection && originalConnection ? { ...connection, distance: originalConnection.distance, polygonIndex } : null;
		}).filter(Boolean);
		const longest = connections.reduce((best, connection) => !best || connection.distance > best.distance ? connection : best, null);
		return { connections, longest, reached };
	}

	function prepareTraceTransition() {
		while (traceIndex < traceEvents.length - 1) {
			const fromPath = traceProjectedPath(traceIndex);
			const toPath = traceProjectedPath(traceIndex + 1);
			const from = fromPath.at(-1) || [0, 0];
			const to = toPath.at(-1) || from;
			const length = Math.hypot(to[0] - from[0], to[1] - from[1]);
			if (length <= 1e-7) {
				traceIndex += 1;
				continue;
			}
			traceTransition = {
				fromIndex: traceIndex,
				toIndex: traceIndex + 1,
				from,
				to,
				length,
				phase: reducedMotion.matches ? "travel" : "preview",
				elapsed: 0,
				previewDuration: reducedMotion.matches ? 0 : 240,
				travelDuration: reducedMotion.matches ? 0 : Math.min(1100, Math.max(520, 280 + length * 1.7)),
				progress: 0,
			};
			drawTrace();
			return true;
		}
		traceTransition = null;
		drawTrace();
		return false;
	}

	function updateTraceAnimationVisuals() {
		const transition = traceTransition;
		const route = traceMapContent.querySelector("#trace-active-route");
		if (!transition || !route || typeof route.getTotalLength !== "function") return;
		let length;
		try { length = route.getTotalLength(); } catch { return; }
		if (!Number.isFinite(length) || length <= 0) return;
		const progress = Math.max(0, Math.min(1, transition.progress));
		route.style.strokeDasharray = `${length} ${length}`;
		route.style.strokeDashoffset = String(length * (1 - progress));
	}

	function traceAnimationTick(now) {
		if (!tracePlaying || !traceTransition) return;
		const transition = traceTransition;
		if (transition.lastTime === undefined) transition.lastTime = now;
		transition.elapsed += now - transition.lastTime;
		transition.lastTime = now;
		if (transition.phase === "preview" && transition.elapsed >= transition.previewDuration) {
			transition.phase = "travel";
			transition.elapsed -= transition.previewDuration;
		}
		transition.progress = transition.phase === "travel" ? Math.min(1, transition.elapsed / Math.max(1, transition.travelDuration)) : 0;
		drawTrace();
		if (transition.progress >= 1) {
			traceIndex = transition.toIndex;
			traceTransition = null;
			if (!prepareTraceTransition()) {
				tracePlaying = false;
				element("trace-play").textContent = "▶ Reproduzir";
				element("trace-play").setAttribute("aria-pressed", "false");
			}
		}
		if (tracePlaying) traceFrame = window.requestAnimationFrame(traceAnimationTick);
	}

	function startTracePlayback() {
		if (!traceEvents.length) return;
		if (traceIndex >= traceEvents.length - 1) {
			traceIndex = 0;
			traceTransition = null;
			drawTrace();
		}
		if (!traceTransition && !prepareTraceTransition()) return;
		tracePlaying = true;
		element("trace-play").textContent = "Ⅱ Pausar";
		element("trace-play").setAttribute("aria-pressed", "true");
		traceTransition.lastTime = performance.now();
		cancelTraceFrame();
		traceFrame = window.requestAnimationFrame(traceAnimationTick);
	}

	function traceSequence(event) {
		if (Array.isArray(event?.order) && event.order.length) return event.order;
		return Array.isArray(event?.sequence) ? event.sequence : [];
	}

	function traceChangedContactPolygons(event, eventIndex) {
		if (event?.kind !== "heuristic_contact_pass" || !Array.isArray(event.path)) return [];
		const previous = traceEvents.slice(0, eventIndex).reverse().find((candidate) => Array.isArray(candidate.path) && candidate.path.length);
		if (!previous || previous.path.length !== event.path.length) return [];
		const order = traceSequence(event);
		return event.path.slice(1, -1).flatMap((point, index) => {
			const previousPoint = previous.path[index + 1];
			if (!previousPoint || Math.hypot(point[0] - previousPoint[0], point[1] - previousPoint[1]) <= TRACE_PATH_EPSILON) return [];
			const polygon = order[index];
			return polygon === undefined ? [] : [traceLabel(order, polygon)];
		});
	}

	function traceEventCopy(event, trace, branchGeometry = null) {
		const order = trace.optimal_order || traceRow.order || [];
		const selected = traceLabels(order, traceSequence(event));
		const polygon = event.polygon === undefined ? null : traceLabel(order, event.polygon);
		const kind = event.kind;
		if (kind === "heuristic_start") return ["A heurística começa", `O caminho inicial parte de S e procura uma ordem promissora de forma gulosa (${event.source === "reverse" ? "sentido reverso" : "sentido direto"}).`];
		if (kind === "heuristic_greedy_step") return ["Escolha gulosa", `A heurística acrescenta a região ${polygon}; a sequência parcial agora é ${selected}.`];
		if (kind === "heuristic_greedy_complete") return ["Ordem gulosa completa", `Todas as regiões foram inseridas. A ordem candidata é ${selected}.`];
		if (kind === "heuristic_2opt") return ["Refino 2-opt", event.reason === "changed" ? `A heurística trocou trechos da ordem e obteve ${selected}.` : "Nenhuma troca 2-opt melhorou a ordem; o refino estabilizou."];
		if (kind === "heuristic_contact_pass") {
			const changedPolygons = traceChangedContactPolygons(event, traceIndex);
			const changedText = changedPolygons.length === 1
				? `O ponto de contato da região ${changedPolygons[0]} mudou.`
				: changedPolygons.length > 1
					? `Mudaram os pontos de contato das regiões ${changedPolygons.join(", ")}.`
					: "Os pontos de contato foram refinados.";
			return ["Refino geométrico", `${changedText} A ordem ${selected} foi refinada (passagem ${Number(event.pass) + 1}).`];
		}
		if (kind === "incumbent") return ["Novo incumbente", `Foi encontrado um caminho viável de comprimento ${traceNumber(event.length, 4)} para a ordem ${selected}.`];
		if (kind === "root") return ["Relaxação na raiz", "O solver ignora temporariamente os buracos e as regiões ainda não escolhidas para obter um limite inferior global."];
		if (kind === "expand") return ["Expande um nó", `A busca examina a ordem parcial ${selected} e calcula como ela pode ser completada.`];
		if (kind === "oracle") return ["Cálculo geométrico", `A relaxação convexa para ${selected} produz limite inferior ${traceNumber(event.lower_bound, 4)}${event.source === "refinement" ? "; a decomposição também foi refinada" : "."}`];
		if (kind === "branch") {
			const branchConnection = branchGeometry?.connections.find((connection) => connection.polygonIndex === Number(event.polygon));
			if (branchConnection && branchGeometry.longest) {
				const reachedLabels = branchGeometry.reached.map((isReached, index) => isReached ? traceLabel(order, index) : null).filter(Boolean);
				const reachedText = reachedLabels.length ? `O caminho já atingiu as regiões ${reachedLabels.join(", ")}.` : "O caminho ainda não atingiu nenhum polígono.";
				return ["Branching", `${reachedText} Para cada região restante, uma linha tracejada mostra a menor distância até o caminho atual. A região ${polygon} é a mais distante (${traceNumber(branchGeometry.longest.distance, 2)}) e por isso é escolhida para gerar alternativas de inserção.`];
			}
			return ["Branching", `A árvore escolhe a região ${polygon} e cria alternativas de inserção${event.reason === "decomposition" ? " ou de peça convexa" : " na ordem"}.`];
		}
		if (kind === "child") return [event.pruned ? "Filho podado" : "Filho enfileirado", event.piece === undefined ? (event.pruned ? `A sequência ${selected} tem limite ${traceNumber(event.lower_bound, 4)}, que já não pode melhorar o incumbente.` : `A sequência ${selected} permanece candidata e entra na fila de busca.`) : `A região ${polygon} foi refinada na peça convexa ${Number(event.piece) + 1}; ${event.pruned ? "o filho é podado" : "o filho entra na fila"}.`];
		if (kind === "prune") return ["Nó podado", event.reason === "incumbent" ? "O caminho encontrado já é tão bom quanto o incumbente." : "O limite inferior excede o melhor caminho conhecido."];
		if (kind === "complete") return ["Busca concluída", `A melhor ordem registrada é ${traceLabels(order, trace.optimal_order)}; comprimento ${traceNumber(event.length ?? trace.summary?.upper_bound, 4)}.`];
		return ["Passo da busca", "O solver atualizou o estado da busca."];
	}

	function renderTraceTree(trace) {
		const visibleKinds = new Set(["root", "expand", "branch", "child", "prune", "complete"]);
		const events = traceEvents.slice(0, traceIndex + 1).filter((event) => visibleKinds.has(event.kind));
		const visible = events.length > 42 ? events.slice(-42) : events;
		element("trace-tree").innerHTML = visible.length ? visible.map((event) => {
			const sequence = traceLabels(trace.optimal_order || traceRow.order, event.sequence || []);
			const label = event.kind === "root" ? "Raiz" : event.kind === "expand" ? `Expande ${sequence}` : event.kind === "branch" ? `Branching em ${traceLabel(trace.optimal_order || traceRow.order, event.polygon)}` : event.kind === "child" ? `${event.pruned ? "Poda" : "Fila"}: ${sequence}` : event.kind === "prune" ? `Poda: ${sequence}` : "Busca concluída";
			const detail = event.kind === "child" && Number.isFinite(Number(event.lower_bound)) ? `LB ${traceNumber(event.lower_bound, 3)}` : event.kind === "complete" ? `UB ${traceNumber(event.upper_bound ?? trace.summary?.upper_bound, 3)}` : "";
			const indent = Math.min(Array.isArray(event.sequence) ? event.sequence.length : 0, 8);
			return `<div class="trace-tree-item ${event.pruned ? "is-pruned" : ""} ${event.kind === "complete" ? "is-complete" : ""}" style="--trace-depth:${indent}"><span>${traceEscape(label)}</span><small>${traceEscape(detail)}</small></div>`;
		}).join("") : '<p class="trace-tree-empty">A árvore aparecerá quando a busca começar.</p>';
	}

	function traceCamera() {
		if (!hasTraceUI) return;
		tracePan = [Math.max(-420 * (traceZoom - 1), Math.min(420 * (traceZoom - 1), tracePan[0])), Math.max(-240 * (traceZoom - 1), Math.min(240 * (traceZoom - 1), tracePan[1]))];
		traceMapContent.setAttribute("transform", `translate(${420 + tracePan[0]} ${240 + tracePan[1]}) scale(${traceZoom}) translate(-420 -240)`);
		traceMap.classList.toggle("is-zoomed", traceZoom > 1);
		traceMap.style.touchAction = traceZoom > 1 ? "none" : "pan-y";
		element("trace-zoom-out").disabled = traceZoom <= 1;
		element("trace-zoom-in").disabled = traceZoom >= MAX_ZOOM;
		element("trace-fit-view").disabled = traceZoom <= 1;
	}

	function drawTrace() {
		if (!hasTraceUI) return;
		const trace = traces[String(traceRow.case)];
		if (!trace || !traceEvents.length) {
			updateTraceCode(null);
			traceMapContent.innerHTML = "";
			element("trace-progress").textContent = "Nenhuma simulação carregada para este caso.";
			element("trace-step-title").textContent = "Escolha uma instância didática";
			element("trace-step-text").textContent = traceRow.case === "usp"
				? "A rota da USP é uma demonstração própria. Escolha um caso do corpus acima para acompanhar uma execução registrada da busca."
				: "Os destaques mostram os Casos 02, 04 e 10. O seletor acima inclui qualquer árvore com menos de 200 passos.";
			element("trace-kind").textContent = "PASSO ATUAL";
			element("trace-sequence").textContent = "—";
			element("trace-lower-bound").textContent = "—";
			element("trace-upper-bound").textContent = "—";
			element("trace-lower-bound").closest(".trace-metrics")?.removeAttribute("data-bound-symbol");
			element("trace-incumbent").textContent = "—";
			element("trace-current-length").textContent = "—";
			element("trace-previous").disabled = true;
			element("trace-next").disabled = true;
			element("trace-play").disabled = true;
			element("trace-tree").innerHTML = '<p class="trace-tree-empty">Selecione um caso com trace disponível.</p>';
			traceCamera();
			return;
		}
		const event = traceEvents[traceIndex];
		updateTraceCode(event);
		const order = trace.optimal_order || traceRow.order || [];
		const currentSequence = traceSequence(event);
		const selected = new Set(currentSequence.map(Number));
		const projection = projectedCase(traceRow);
		const start = projection.project(traceRow.geometry.start), target = projection.project(traceRow.geometry.target);
		const originalPath = traceCurrentPath(traceIndex);
		const currentPath = originalPath.map(projection.project);
		const reached = pathPolygonContacts(originalPath, traceRow.geometry.polygons).map(Boolean);
		const incumbentPath = (traceIncumbentPath(traceIndex) || []).map(projection.project);
		const isHeuristic = event.kind.startsWith("heuristic_") || (event.kind === "incumbent" && event.source === "heuristic");
		const showIncumbentRoute = enabled("trace-show-incumbent") && !isHeuristic && incumbentPath.length > 1;
		const transition = traceTransition?.fromIndex === traceIndex ? traceTransition : null;
		const movingSegment = transition ? [transition.from, transition.to] : [];
		const showLabels = enabled("trace-show-labels") && (traceRow.polygons <= 15 || selected.size > 0);
		const branchGeometry = traceBranchGeometry(event, currentPath, projection.polygons, originalPath, reached);
		const visibleBranchGeometry = enabled("trace-show-branching") ? branchGeometry : null;
		const decompositionMarkup = enabled("trace-show-decomposition")
			? decompositionLinesMarkup(traceRow.visualization?.decomposition, traceRow.geometry.polygons, projection.project)
			: "";
		const branchConnectionLines = visibleBranchGeometry?.connections.map((connection) => {
			const longest = branchGeometry.longest?.polygonIndex === connection.polygonIndex;
			const lineClass = longest ? "trace-branch-connection trace-branch-connection-longest" : "trace-branch-connection";
			const pointClass = longest ? "trace-branch-point trace-branch-point-longest" : "trace-branch-point";
			return `<line class="${lineClass}" x1="${connection.first[0]}" y1="${connection.first[1]}" x2="${connection.second[0]}" y2="${connection.second[1]}"/><circle class="${pointClass}" cx="${connection.first[0]}" cy="${connection.first[1]}" r="3.5"/><circle class="${pointClass}" cx="${connection.second[0]}" cy="${connection.second[1]}" r="3.5"/>`;
		}).join("") || "";
		const progress = transition?.progress ?? 0;
		const traveler = transition ? [
			transition.from[0] + (transition.to[0] - transition.from[0]) * progress,
			transition.from[1] + (transition.to[1] - transition.from[1]) * progress,
		] : currentPath.at(-1) || start;
		traceMapContent.innerHTML = `${traceRow.geometry.polygons.map((polygon, index) => {
			const branchSelected = branchGeometry?.longest?.polygonIndex === index;
			const title = `Região ${traceLabel(order, index)}${branchSelected ? "; escolhida para o branching por ser a mais distante" : ""}`;
			return `<polygon class="trace-region ${selected.has(index) ? "trace-selected" : ""} ${reached[index] ? "trace-reached" : ""} ${branchSelected ? "trace-branch-selected" : ""}" points="${coordinates(polygon.map(projection.project))}"><title>${title}</title></polygon>`;
		}).join("")}
			${enabled("trace-show-hulls") ? traceRow.geometry.polygons.map((polygon) => `<polygon class="trace-hull" points="${coordinates(convexHull(polygon).map(projection.project))}"/>`).join("") : ""}
			${decompositionMarkup}
			${branchConnectionLines}
			${showIncumbentRoute ? `<polyline class="trace-incumbent-route" points="${coordinates(incumbentPath)}"/>` : ""}
			${currentPath.length > 1 ? `<polyline class="trace-route-completed" points="${coordinates(currentPath)}"/>` : ""}
			${movingSegment.length > 1 ? `<polyline class="trace-route-preview" points="${coordinates(movingSegment)}"/><polyline id="trace-active-route" class="trace-current-route" points="${coordinates(movingSegment)}"/>` : ""}
			${showLabels ? projection.polygons.map((polygon, index) => { const center = polygonCentroid(polygon); return `<text class="trace-region-label ${selected.has(index) ? "trace-label-selected" : ""}" x="${center[0]}" y="${center[1]}" text-anchor="middle" dominant-baseline="central">${traceLabel(order, index)}</text>`; }).join("") : ""}
			<circle class="trace-traveler" cx="${traveler[0]}" cy="${traveler[1]}" r="5"/><circle class="trace-endpoint" cx="${start[0]}" cy="${start[1]}" r="6"/><text class="trace-endpoint-label" x="${start[0] + 13}" y="${start[1] + 4}">S</text><circle class="trace-endpoint trace-target" cx="${target[0]}" cy="${target[1]}" r="6"/><text class="trace-endpoint-label" x="${target[0] + 13}" y="${target[1] + 4}">T</text>`;
		updateTraceAnimationVisuals();
		const [title, text] = traceEventCopy(event, trace, branchGeometry);
		element("trace-kind").textContent = event.kind.replaceAll("_", " ").toUpperCase();
		element("trace-step-title").textContent = title;
		element("trace-step-text").textContent = text;
		element("trace-sequence").textContent = traceLabels(order, currentSequence);
		const incumbent = traceIncumbent(traceIndex);
		const lowerBound = Number(event.lower_bound);
		const boundSymbol = Number.isFinite(lowerBound) && Number.isFinite(incumbent) ? (lowerBound >= incumbent ? "≥" : "<") : "";
		element("trace-lower-bound").textContent = traceNumber(event.lower_bound, 4);
		element("trace-upper-bound").textContent = traceNumber(event.upper_bound, 4);
		element("trace-lower-bound").closest(".trace-metrics")?.setAttribute("data-bound-symbol", boundSymbol);
		element("trace-incumbent").textContent = traceNumber(incumbent, 2);
		element("trace-current-length").textContent = traceNumber(event.length ?? pathLength(originalPath), 2);
		element("trace-progress").textContent = `Passo ${traceIndex + 1} de ${traceEvents.length}${trace.omitted_events ? ` · ${trace.omitted_events.toLocaleString("pt-BR")} eventos omitidos` : ""}`;
		element("trace-previous").disabled = traceIndex === 0;
		element("trace-next").disabled = traceIndex === traceEvents.length - 1;
		element("trace-play").disabled = traceEvents.length < 2;
		const branchCaption = branchGeometry ? " As linhas tracejadas mostram as menores distâncias até as regiões não atingidas; a linha vermelha é a maior e destaca a região escolhida." : "";
		element("trace-map-caption").textContent = isHeuristic
			? `Regiões destacadas pertencem à sequência parcial ${traceLabels(order, currentSequence)}. A linha laranja mostra somente o caminho construído até este passo; o trecho mais recente é animado.${branchCaption}`
			: `Regiões destacadas pertencem à sequência parcial ${traceLabels(order, currentSequence)}. A linha tracejada mostra o incumbente; a laranja sólida mostra o caminho calculado neste passo.${branchCaption}`;
		traceCamera();
		renderTraceTree(trace);
		updateLayerNotes();
	}

	function selectTrace(caseIndex = traceRow.case) {
		if (!hasTraceUI) return;
		const selected = data.rows.find((item) => item.case === caseIndex);
		if (!selected || !traces[String(caseIndex)]) return;
		stopTrace();
		traceRow = selected;
		const trace = traces[String(traceRow.case)];
		traceEvents = filterTraceEvents(trace?.events || []);
		traceIndex = 0;
		element("trace-case-select").value = shortTraceCases.some((item) => item.case === caseIndex) ? String(caseIndex) : "";
		element("trace-case-picker-value").textContent = tracePickerLabel(trace);
		document.querySelectorAll(".trace-showcases .example").forEach((button) => {
			const active = button.dataset.case === String(caseIndex);
			button.classList.toggle("active", active);
			button.setAttribute("aria-pressed", String(active));
		});
		drawTrace();
	}

	function advanceTrace(delta) {
		if (!traceEvents.length) return;
		stopTrace();
		traceIndex = Math.max(0, Math.min(traceEvents.length - 1, traceIndex + delta));
		drawTrace();
	}

	function camera() {
		const centerX = routeWidth / 2, centerY = routeHeight / 2;
		pan = [Math.max(-centerX * (zoom - 1), Math.min(centerX * (zoom - 1), pan[0])), Math.max(-centerY * (zoom - 1), Math.min(centerY * (zoom - 1), pan[1]))];
		mapContent.setAttribute("transform", `translate(${centerX + pan[0]} ${centerY + pan[1]}) scale(${zoom}) translate(${-centerX} ${-centerY})`);
		map.classList.toggle("is-zoomed", zoom > 1);
		map.style.touchAction = zoom > 1 ? "none" : "pan-y";
		element("zoom-out").disabled = zoom <= 1;
		element("zoom-in").disabled = zoom >= MAX_ZOOM;
		element("fit-view").disabled = zoom <= 1;
		element("map-navigation").textContent = zoom > 1 ? `Zoom ${number(zoom, 1)}×. Arraste para mover ou toque em Redefinir para voltar à rolagem da página.` : "Amplie para mover o desenho. Na visualização ajustada, arraste para continuar rolando a página.";
	}

	function stop() {
		cancelAnimationFrame(frame);
		playing = false;
		element("play-route").textContent = "▶ Veja o caminho";
		element("play-route").setAttribute("aria-pressed", "false");
	}

	function drawRoute() {
		const prefix = pathPrefix(projected.path, fraction);
		element("animated-route").setAttribute("points", coordinates(prefix));
		const point = prefix.at(-1);
		element("traveler").setAttribute("cx", point[0]);
		element("traveler").setAttribute("cy", point[1]);
		element("traveler").style.display = fraction >= 1 ? "none" : "";
		element("route-progress").value = Math.round(fraction * 1000);
		element("progress-value").textContent = `${Math.round(fraction * 100)}%`;
		element("route-progress").style.setProperty("--progress", `${fraction * 100}%`);
		mapContent.querySelectorAll(".region").forEach((polygon) => {
			const index = Number(polygon.dataset.region);
			const contact = row.visualization.contacts[index];
			const visited = Boolean(contact) && fraction + 1e-12 >= contact.fraction;
			const colors = row.case === "usp" && row.buildings[index].id === "ime"
				? { fill: visited ? "#e6ac63aa" : "#ffcf8b55", stroke: "#ffdcaa" }
				: regionColors(row.order.indexOf(index), row.polygons, visited);
			polygon.style.fill = colors.fill;
			polygon.style.stroke = colors.stroke;
			polygon.classList.toggle("visited", visited);
		});
		mapContent.querySelectorAll(".visit-contact").forEach((point) => {
			const contact = row.visualization.contacts[Number(point.dataset.region)];
			point.classList.toggle("reached", Boolean(contact) && fraction + 1e-12 >= contact.fraction);
		});
		if (row.case === "usp") {
			const next = row.order.find((index) => row.visualization.contacts[index].fraction > fraction + 1e-12);
			setOutput(element("usp-current-stop"), next === undefined
				? `Percurso concluído: ${row.polygons} edifícios visitados e retorno ao IME.`
				: `Próximo edifício: ${row.buildings[next].label}.`);
		}
	}

	function updateLayerNotes() {
		document.querySelectorAll("[data-layer-note]").forEach((item) => item.classList.toggle("active", enabled(item.dataset.layerNote)));
	}

	function draw() {
		const mobileRoute = row.case === "usp" && window.matchMedia("(max-width: 560px)").matches;
		map.classList.toggle("usp-route", row.case === "usp");
		routeWidth = mobileRoute ? 420 : 840;
		routeHeight = mobileRoute ? 440 : 480;
		map.setAttribute("viewBox", `0 0 ${routeWidth} ${routeHeight}`);
		projected = projectedCase(row, routeWidth, routeHeight);
		const start = projected.project(row.geometry.start), target = projected.project(row.geometry.target);
		const labels = enabled("show-labels");
		const hulls = enabled("show-hulls");
		const decomposition = enabled("show-decomposition");
		const rect = map.getBoundingClientRect();
		const textScale = 1 / Math.max(.1, Math.min(rect.width / routeWidth, rect.height / routeHeight)) / zoom;
		mapContent.innerHTML = `${hulls ? row.geometry.polygons.map((polygon) => `<polygon class="hull" points="${coordinates(convexHull(polygon).map(projected.project))}"/>`).join("") : ""}
			${projected.polygons.map((polygon, index) => {
				const visited = fraction + 1e-12 >= (row.visualization.contacts[index]?.fraction ?? Infinity);
				const colors = row.case === "usp" && row.buildings[index].id === "ime"
					? { fill: visited ? "#e6ac63aa" : "#ffcf8b55", stroke: "#ffdcaa" }
					: regionColors(row.order.indexOf(index), row.polygons, visited);
				return `<polygon class="region" data-region="${index}" points="${coordinates(polygon)}" style="fill:${colors.fill};stroke:${colors.stroke}"><title>${row.case === "usp" ? escapeHTML(row.buildings[index].label) : `Região ${index + 1}`}; ${row.order.indexOf(index) + 1}ª visita</title></polygon>`;
			}).join("")}
			${decomposition ? decompositionLinesMarkup(row.visualization.decomposition, row.geometry.polygons, projected.project) : ""}
			<polyline class="route-ghost" points="${coordinates(projected.path)}"/><polyline id="animated-route" class="route-line"/>
			${labels ? projected.polygons.map((polygon, index) => {
				const center = polygonCentroid(polygon);
				return `<text class="region-label" x="${center[0]}" y="${center[1]}" text-anchor="middle" dominant-baseline="central">${row.order.indexOf(index) + 1}</text>`;
			}).join("") : ""}
			<circle class="endpoint" cx="${start[0]}" cy="${start[1]}" r="7"/><text class="endpoint-label" x="${start[0] + 14}" y="${start[1] + 5}">${row.depot ? "S = T" : "S"}</text>
			${row.depot ? "" : `<circle class="endpoint target" cx="${target[0]}" cy="${target[1]}" r="7"/><text class="endpoint-label" x="${target[0] + 14}" y="${target[1] + 5}">T</text>`}<circle id="traveler" class="traveler" r="6"/>
			${enabled("show-contacts") ? row.visualization.contacts.map((contact, index) => {
				if (!contact) return "";
				const point = projected.project(contact.point);
				return `<circle class="visit-contact" data-region="${index}" cx="${point[0]}" cy="${point[1]}" r="4"><title>${row.order.indexOf(index) + 1}ª visita · região original ${index + 1}</title></circle>`;
			}).join("") : ""}`;
		mapContent.querySelectorAll(".region-label").forEach((label) => { label.style.fontSize = `${13 * textScale}px`; });
		mapContent.querySelectorAll(".endpoint-label").forEach((label, index) => {
			const point = row.depot ? [start[0] + 15 * textScale, start[1] + 17 * textScale] : endpointOffset(projected.path, Boolean(index), 22 * textScale);
			label.setAttribute("text-anchor", row.depot ? "start" : "middle");
			label.setAttribute("dominant-baseline", "central");
			label.style.fontSize = `${16 * textScale}px`;
			label.setAttribute("x", point[0]);
			label.setAttribute("y", point[1]);
		});
		mapContent.querySelectorAll(".endpoint, .traveler").forEach((point) => point.setAttribute("r", 6 * textScale));
		mapContent.querySelectorAll(".visit-contact").forEach((point) => point.setAttribute("r", 4 * textScale));
		camera();
		updateLayerNotes();
		element("map-title").textContent = row.case === "usp"
			? `Rota fechada do IME por ${row.polygons} edifícios da USP, com gap numérico fechado.`
			: `Caso ${caseLabel(row.case)}: busca concluída para ${row.polygons} regiões.`;
		drawRoute();
	}

	function selectCase(index, updateURL = true) {
		const selected = index === "usp" ? uspDemo : data.rows.find((item) => item.case === index);
		if (!selected) return false;
		stop();
		stopTrace();
		row = selected;
		fraction = 1;
		zoom = 1;
		pan = [0, 0];
		traceZoom = 1;
		tracePan = [0, 0];
		element("show-labels").setAttribute("aria-pressed", String(row.polygons <= 15));
		element("case-select").value = row.case;
		element("case-picker-value").textContent = row.case === "usp" ? "Escolha um dos 558 casos do corpus" : `Caso ${caseLabel(row.case)} · ${row.polygons} regiões`;
		document.querySelectorAll(".explorer .example").forEach((button) => {
			const active = button.dataset.case === String(row.case);
			button.classList.toggle("active", active);
			button.setAttribute("aria-pressed", String(active));
			if (active) button.setAttribute("aria-current", "true");
			else button.removeAttribute("aria-current");
		});
		const isUsp = row.case === "usp";
		element("route-endpoint-legend").textContent = isUsp ? "IME dourado · S = T na entrada" : "S partida · T chegada";
		element("usp-demo-details").hidden = !isUsp;
		element("usp-current-stop").hidden = !isUsp;
		element("show-decomposition").disabled = isUsp;
		if (isUsp) {
			const itinerary = [
				`S = T · ${escapeHTML(row.depot.label)} (3 m fora do contorno)`,
				...row.order.map((building, position) => `${position + 1} · ${escapeHTML(row.buildings[building].label)}`),
				`Retorno · ${escapeHTML(row.depot.label)}`,
			];
			element("usp-itinerary").innerHTML = itinerary.map((entry) => `<li>${entry}</li>`).join("");
		}
		element("speed-value").title = `A 1×, este caso leva ${number(playbackDuration(row.polygons) / 1000, 1)} s para percorrer o caminho.`;
		element("outcome-badge").className = "status certified";
		element("outcome-badge").textContent = isUsp ? "✓ Gap numérico fechado" : "✓ Busca concluída";
		element("outcome-title").textContent = isUsp ? "Rota da USP" : "Caminho mínimo";
		element("outcome-explanation").textContent = isUsp
			? "O solver C++ fechou os limites numéricos nesta instância; ela não integra as estatísticas do corpus."
			: "A busca foi concluída pelo solver.";
		element("case-length").textContent = number(row.length ?? row.validation?.recomputed_length, 2);
		element("case-unit").textContent = isUsp ? "metros no modelo plano" : "unidades da instância";
		element("case-time").textContent = formatDuration(row.seconds);
		element("case-time-note").textContent = isUsp ? "uma execução local, uma thread" : "uma thread";
		if (updateURL && window.location.protocol !== "file:") {
			const url = new URL(window.location.href);
			url.searchParams.set("caso", row.case);
			window.history.replaceState(null, "", url);
		}
		draw();
		selectTrace(traces[String(row.case)] ? row.case : traceRow.case);
		return true;
	}

	function renderTable() {
		const rows = sortRows(data.rows, sorting.result.key, sorting.result.descending);
		const visible = showAllResults ? rows : rows.slice(0, 8);
		element("result-count").textContent = `${rows.length} instâncias concluídas.`;
		const toggle = rows.length > 8 ? `<tr class="result-toggle-row"><td colspan="4"><button type="button" data-toggle-results aria-expanded="${showAllResults}">${showAllResults ? "Mostrar somente os 8 destaques ↑" : `Ver todos os ${rows.length} resultados ↓`}</button></td></tr>` : "";
		element("result-rows").innerHTML = visible.map((item) => `<tr><th scope="row"><span class="mobile-case-label">Caso </span>${caseLabel(item.case)}</th><td data-label="Regiões">${item.polygons}</td><td data-label="Tempo">${formatDuration(item.seconds)}</td><td class="result-action"><button type="button" data-open-case="${item.case}" aria-label="Ver caminho do caso ${item.case + 1}">Ver caminho →</button></td></tr>`).join("") + toggle;
	}

	document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach((control) => { control.disabled = false; });
	populateCaseSelect();
	updateResultSummary();
	populateTracePicker();
	function showMap() {
		element("route-map").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
		element("play-route").focus({ preventScroll: true });
	}
	document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => {
		const key = button.dataset.case === "usp" ? "usp" : Number(button.dataset.case);
		if (button.closest(".trace-showcases")) {
			if (key === traceRow.case) return;
			selectTrace(key);
			element("trace-map").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
			element("trace-play").focus({ preventScroll: true });
		} else {
			if (key === row.case) return;
			selectCase(key);
			showMap();
		}
	}));
	element("case-select").addEventListener("change", (event) => { selectCase(Number(event.target.value)); showMap(); });
	element("show-labels").addEventListener("click", () => { element("show-labels").setAttribute("aria-pressed", String(!enabled("show-labels"))); draw(); });
	element("show-hulls").addEventListener("click", () => { element("show-hulls").setAttribute("aria-pressed", String(!enabled("show-hulls"))); draw(); });
	element("show-contacts").addEventListener("click", () => { element("show-contacts").setAttribute("aria-pressed", String(!enabled("show-contacts"))); draw(); });
	element("show-decomposition").addEventListener("click", () => { element("show-decomposition").setAttribute("aria-pressed", String(!enabled("show-decomposition"))); draw(); });
	["trace-show-labels", "trace-show-hulls", "trace-show-decomposition", "trace-show-branching", "trace-show-incumbent"].forEach((id) => element(id)?.addEventListener("click", () => {
		const control = element(id);
		control.setAttribute("aria-pressed", String(!enabled(id)));
		drawTrace();
	}));
	element("fit-view").addEventListener("click", () => { zoom = 1; pan = [0, 0]; draw(); });
	element("zoom-out").addEventListener("click", () => zoomAt(zoom / 1.5, [0, 0]));
	element("zoom-in").addEventListener("click", () => zoomAt(zoom * 1.5, [0, 0]));
	const pointers = new Map();
	let gesture = null;
	function localPoint(event) {
		const box = map.getBoundingClientRect();
		const scale = Math.min(box.width / routeWidth, box.height / routeHeight);
		return [(event.clientX - box.left - box.width / 2) / scale, (event.clientY - box.top - box.height / 2) / scale];
	}
	function zoomAt(next, point) {
		const previous = zoom;
		zoom = Math.max(1, Math.min(MAX_ZOOM, next));
		pan = point.map((value, axis) => value - (value - pan[axis]) * zoom / previous);
		draw();
	}
	let trackpadGesture = null;
	map.addEventListener("wheel", (event) => {
		if (trackpadGesture) return;
		const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 480 : 1;
		if (!event.ctrlKey && zoom <= 1 && event.deltaY * units > 0) return;
		event.preventDefault();
		zoomAt(zoom * Math.exp(-event.deltaY * units * (event.ctrlKey ? .01 : .002)), localPoint(event));
	}, { passive: false });
	map.addEventListener("gesturestart", (event) => { event.preventDefault(); trackpadGesture = { zoom, scale: 1 }; }, { passive: false });
	map.addEventListener("gesturechange", (event) => {
		event.preventDefault();
		if (!trackpadGesture || !Number.isFinite(event.scale)) return;
		zoomAt(trackpadGesture.zoom * event.scale, localPoint(event));
	}, { passive: false });
	map.addEventListener("gestureend", (event) => { event.preventDefault(); trackpadGesture = null; }, { passive: false });
	function resetGesture() {
		const points = [...pointers.values()];
		if (points.length >= 2) {
			const center = points[0].map((v, i) => (v + points[1][i]) / 2);
			gesture = { distance: Math.hypot(...points[0].map((v, i) => v - points[1][i])), zoom, center, pan: [...pan] };
		} else { gesture = null; drag = points.length ? [...points[0], ...pan] : null; }
	}
	map.addEventListener("pointerdown", (event) => {
		if (event.button !== 0) return;
		if (event.pointerType === "touch" && zoom <= 1 && event.isPrimary) return;
		map.focus({ preventScroll: true });
		pointers.set(event.pointerId, localPoint(event));
		map.setPointerCapture(event.pointerId);
		resetGesture();
	});
	map.addEventListener("pointermove", (event) => {
		if (!pointers.has(event.pointerId)) return;
		const point = localPoint(event);
		pointers.set(event.pointerId, point);
		if (gesture && pointers.size >= 2) {
			const points = [...pointers.values()];
			const distance = Math.hypot(...points[0].map((v, i) => v - points[1][i]));
			zoom = Math.max(1, Math.min(MAX_ZOOM, gesture.zoom * distance / Math.max(gesture.distance, 1e-6)));
			pan = points[0].map((v, i) => (v + points[1][i]) / 2 - (gesture.center[i] - gesture.pan[i]) * zoom / gesture.zoom);
			draw();
		} else if (drag) { pan = point.map((v, i) => drag[i + 2] + v - drag[i]); camera(); }
	});
	for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) map.addEventListener(name, (event) => {
		pointers.delete(event.pointerId); resetGesture();
	});
	map.addEventListener("keydown", (event) => {
		const moves = { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] };
		if (moves[event.key]) pan = pan.map((value, axis) => value + moves[event.key][axis]);
		else if (event.key === "+" || event.key === "=") zoom = Math.min(MAX_ZOOM, zoom + .5);
		else if (event.key === "-" || event.key === "−") zoom = Math.max(1, zoom - .5);
		else if (event.key === "Home") { zoom = 1; pan = [0, 0]; }
		else return;
		event.preventDefault();
		draw();
	});
	new ResizeObserver(draw).observe(map);
	if (hasTraceUI) {
		const tracePointers = new Map();
		let traceGesture = null;
		let traceTrackpadGesture = null;
		function traceLocalPoint(event) {
			const box = traceMap.getBoundingClientRect();
			const scale = Math.min(box.width / 840, box.height / 480);
			return [(event.clientX - box.left - box.width / 2) / scale, (event.clientY - box.top - box.height / 2) / scale];
		}
		function traceZoomAt(next, point) {
			const previous = traceZoom;
			traceZoom = Math.max(1, Math.min(MAX_ZOOM, next));
			tracePan = point.map((value, axis) => value - (value - tracePan[axis]) * traceZoom / previous);
			traceCamera();
		}
		element("trace-fit-view").addEventListener("click", () => {
			traceZoom = 1;
			tracePan = [0, 0];
			traceCamera();
		});
		element("trace-zoom-out").addEventListener("click", () => traceZoomAt(traceZoom / 1.5, [0, 0]));
		element("trace-zoom-in").addEventListener("click", () => traceZoomAt(traceZoom * 1.5, [0, 0]));
		function resetTraceGesture() {
			const points = [...tracePointers.values()];
			if (points.length >= 2) {
				const center = points[0].map((value, axis) => (value + points[1][axis]) / 2);
				traceGesture = { distance: Math.hypot(...points[0].map((value, axis) => value - points[1][axis])), zoom: traceZoom, center, pan: [...tracePan] };
			} else {
				traceGesture = null;
				traceDrag = points.length ? [...points[0], ...tracePan] : null;
			}
		}
		traceMap.addEventListener("wheel", (event) => {
			if (traceTrackpadGesture) return;
			const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 480 : 1;
			if (!event.ctrlKey && traceZoom <= 1 && event.deltaY * units > 0) return;
			event.preventDefault();
			traceZoomAt(traceZoom * Math.exp(-event.deltaY * units * (event.ctrlKey ? .01 : .002)), traceLocalPoint(event));
		}, { passive: false });
		traceMap.addEventListener("gesturestart", (event) => { event.preventDefault(); traceTrackpadGesture = { zoom: traceZoom }; }, { passive: false });
		traceMap.addEventListener("gesturechange", (event) => {
			event.preventDefault();
			if (!traceTrackpadGesture || !Number.isFinite(event.scale)) return;
			traceZoomAt(traceTrackpadGesture.zoom * event.scale, traceLocalPoint(event));
		}, { passive: false });
		traceMap.addEventListener("gestureend", (event) => { event.preventDefault(); traceTrackpadGesture = null; }, { passive: false });
		traceMap.addEventListener("pointerdown", (event) => {
			if (event.button !== 0) return;
			if (event.pointerType === "touch" && traceZoom <= 1 && event.isPrimary) return;
			traceMap.focus({ preventScroll: true });
			tracePointers.set(event.pointerId, traceLocalPoint(event));
			traceMap.setPointerCapture(event.pointerId);
			resetTraceGesture();
		});
		traceMap.addEventListener("pointermove", (event) => {
			if (!tracePointers.has(event.pointerId)) return;
			const point = traceLocalPoint(event);
			tracePointers.set(event.pointerId, point);
			if (traceGesture && tracePointers.size >= 2) {
				const points = [...tracePointers.values()];
				const distance = Math.hypot(...points[0].map((value, axis) => value - points[1][axis]));
				traceZoom = Math.max(1, Math.min(MAX_ZOOM, traceGesture.zoom * distance / Math.max(traceGesture.distance, 1e-6)));
				tracePan = points[0].map((value, axis) => (value + points[1][axis]) / 2 - (traceGesture.center[axis] - traceGesture.pan[axis]) * traceZoom / traceGesture.zoom);
				traceCamera();
			} else if (traceDrag) {
				tracePan = point.map((value, axis) => traceDrag[axis + 2] + value - traceDrag[axis]);
				traceCamera();
			}
			traceMap.classList.toggle("is-dragging", Boolean(traceDrag || traceGesture));
		});
		for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) traceMap.addEventListener(name, (event) => {
			tracePointers.delete(event.pointerId);
			resetTraceGesture();
			traceMap.classList.toggle("is-dragging", Boolean(traceDrag || traceGesture));
		});
		traceMap.addEventListener("keydown", (event) => {
			const moves = { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] };
			if (moves[event.key]) tracePan = tracePan.map((value, axis) => value + moves[event.key][axis]);
			else if (event.key === "+" || event.key === "=") traceZoom = Math.min(MAX_ZOOM, traceZoom + .5);
			else if (event.key === "-" || event.key === "−") traceZoom = Math.max(1, traceZoom - .5);
			else if (event.key === "Home") { traceZoom = 1; tracePan = [0, 0]; }
			else return;
			event.preventDefault();
			traceCamera();
		});
		new ResizeObserver(traceCamera).observe(traceMap);
	}
	element("route-progress").addEventListener("input", (event) => { stop(); fraction = Number(event.target.value) / 1000; drawRoute(); });
	element("play-route").addEventListener("click", () => {
		if (playing) { stop(); return; }
		if (reducedMotion.matches) { fraction = 1; drawRoute(); return; }
		if (fraction >= 1) fraction = 0;
		playing = true;
		element("play-route").textContent = "Ⅱ Pausar percurso";
		element("play-route").setAttribute("aria-pressed", "true");
		let previous = performance.now();
		const tick = (now) => {
			fraction = Math.min(1, fraction + (now - previous) * speeds[speedIndex] / playbackDuration(row.polygons));
			previous = now;
			drawRoute();
			if (fraction >= 1) stop();
			else frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
	});
	if (hasTraceUI) {
		element("trace-previous").addEventListener("click", () => advanceTrace(-1));
		element("trace-next").addEventListener("click", () => advanceTrace(1));
		element("trace-play").addEventListener("click", () => {
			if (!traceEvents.length) return;
			if (tracePlaying) {
				tracePlaying = false;
				cancelTraceFrame();
				element("trace-play").textContent = "▶ Reproduzir";
				element("trace-play").setAttribute("aria-pressed", "false");
				return;
			}
			startTracePlayback();
		});
		element("trace-case-select")?.addEventListener("change", (event) => {
			if (event.target.value === "") return;
			selectTrace(Number(event.target.value));
			element("trace-map").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
			element("trace-play").focus({ preventScroll: true });
		});
		const tracePicker = element("trace-case-picker");
		const tracePickerDialog = element("trace-case-picker-dialog");
		if (tracePicker && tracePickerDialog) {
			element("trace-case-picker-button").addEventListener("click", () => {
				element("trace-picker-search").value = "";
				renderTracePicker();
				showModalWithTransition(tracePickerDialog);
				element("trace-case-picker-button").setAttribute("aria-expanded", "true");
				element("trace-picker-search").focus();
			});
			element("trace-picker-close").addEventListener("click", () => tracePickerDialog.close());
			tracePickerDialog.addEventListener("close", () => {
				element("trace-case-picker-button").setAttribute("aria-expanded", "false");
				element("trace-case-picker-button").focus({ preventScroll: true });
			});
			tracePickerDialog.addEventListener("click", (event) => {
				if (event.target === tracePickerDialog) {
					const box = tracePickerDialog.getBoundingClientRect();
					if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) tracePickerDialog.close();
				}
			});
			element("trace-picker-search").addEventListener("input", renderTracePicker);
			element("trace-picker-options").addEventListener("click", (event) => {
				const button = event.target.closest("[data-trace-case]");
				if (!button) return;
				selectTrace(Number(button.dataset.traceCase));
				tracePickerDialog.close();
				element("trace-map").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
			});
		}
	}
	document.addEventListener("visibilitychange", () => { if (document.hidden) { stop(); stopTrace(); } });
	reducedMotion.addEventListener("change", stop);
	let returnContext = null;
	element("result-rows").addEventListener("click", (event) => {
		const toggle = event.target.closest("[data-toggle-results]");
		if (toggle) {
			showAllResults = !showAllResults;
			renderTable();
			element("result-rows").querySelector("[data-toggle-results]")?.focus({ preventScroll: true });
			return;
		}
		const button = event.target.closest("[data-open-case]");
		if (!button) return;
		returnContext = { top: window.scrollY, control: button };
		selectCase(Number(button.dataset.openCase));
		element("context-return").hidden = false;
		element("section-dialog")?.closeGuideSection?.();
		showMap();
	});
	element("context-return").addEventListener("click", () => {
		if (!returnContext) return;
		const destination = returnContext;
		returnContext = null;
		element("context-return").hidden = true;
		const start = window.scrollY;
		const finish = () => {
			const sectionDialog = element("section-dialog");
			if (sectionDialog?.openGuideSection) sectionDialog.openGuideSection("resultados", destination.control);
			else destination.control.focus({ preventScroll: true });
		};
		if (reducedMotion.matches || Math.abs(start - destination.top) < 2) {
			window.scrollTo({ top: destination.top, behavior: "instant" });
			finish();
			return;
		}
		window.scrollTo({ top: start, behavior: "instant" });
		const started = performance.now();
		const animate = (now) => {
			const progress = Math.min(1, (now - started) / 260);
			const eased = 1 - (1 - progress) ** 3;
			window.scrollTo(0, start + (destination.top - start) * eased);
			if (progress < 1) requestAnimationFrame(animate);
			else finish();
		};
		requestAnimationFrame(animate);
	});
	function setSpeed(index) {
		speedIndex = Math.max(0, Math.min(speeds.length - 1, index));
		element("speed-value").textContent = `${number(speeds[speedIndex], 2)}×`;
		element("speed-down").disabled = speedIndex === 0;
		element("speed-up").disabled = speedIndex === speeds.length - 1;
	}
	setSpeed(speedIndex);
	element("speed-down").addEventListener("click", () => setSpeed(speedIndex - 1));
	element("speed-up").addEventListener("click", () => setSpeed(speedIndex + 1));
	const picker = element("case-picker-dialog");
	function renderPicker() {
		const rows = sortRows(visitorRows(data.rows, element("picker-search").value), sorting.picker.key, sorting.picker.descending);
		element("picker-count").textContent = `${rows.length} casos disponíveis`;
		element("picker-options").innerHTML = rows.length ? rows.map((item) => `<button type="button" data-pick-case="${item.case}" aria-pressed="${item.case === row.case}"><span><strong>Caso ${caseLabel(item.case)}</strong><small>${item.polygons} regiões · ${formatDuration(item.seconds)}</small></span></button>`).join("") : "<p>Nenhum caso encontrado. Experimente outro número.</p>";
	}
	element("case-picker-button").addEventListener("click", () => {
		element("picker-search").value = "";
		renderPicker();
		showModalWithTransition(picker);
		element("case-picker-button").setAttribute("aria-expanded", "true");
		element("picker-close").focus();
	});
	element("picker-close").addEventListener("click", () => picker.close());
	picker.addEventListener("close", () => {
		element("case-picker-button").setAttribute("aria-expanded", "false");
		element("case-picker-button").focus({ preventScroll: true });
	});
	picker.addEventListener("click", (event) => { if (event.target === picker) {
		const box = picker.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) picker.close();
	} });
	element("picker-search").addEventListener("input", renderPicker);
	element("picker-options").addEventListener("click", (event) => {
		const button = event.target.closest("[data-pick-case]");
		if (button) { selectCase(Number(button.dataset.pickCase)); picker.close(); showMap(); }
	});
	element("picker-options").addEventListener("keydown", (event) => {
		const buttons = [...element("picker-options").querySelectorAll("button")];
		const index = buttons.indexOf(document.activeElement);
		if (event.key === "ArrowDown" || event.key === "ArrowUp") {
			event.preventDefault();
			buttons[(index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length]?.focus();
		}
	});
	for (const prefix of ["picker"]) {
		const direction = element(`${prefix}-direction`);
		const group = direction.parentElement;
		const render = prefix === "picker" ? renderPicker : renderTable;
		group.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
			sorting[prefix].key = button.dataset.sort;
			group.querySelectorAll("[data-sort]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
			render();
		}));
		direction.addEventListener("click", () => {
			sorting[prefix].descending = !sorting[prefix].descending;
			direction.setAttribute("aria-pressed", String(sorting[prefix].descending));
			direction.textContent = sorting[prefix].descending ? "↓ Decrescente" : "↑ Crescente";
			render();
		});
	}
	function updateResultSorting() {
		document.querySelectorAll("[data-result-column]").forEach((header) => {
			const key = header.dataset.resultColumn;
			const active = key === sorting.result.key;
			const descending = sorting.result.descending;
			const primary = active;
			header.setAttribute("aria-sort", active && primary ? (descending ? "descending" : "ascending") : "none");
			header.querySelector(".sort-arrow").textContent = active ? (descending ? " ↓" : " ↑") : "";
			header.classList.toggle("sort-active", active);
		});
		renderTable();
	}
	document.querySelectorAll("[data-result-sort]").forEach((button) => button.addEventListener("click", () => {
		const key = button.dataset.resultSort;
		sorting.result.descending = sorting.result.key === key ? !sorting.result.descending : false;
		sorting.result.key = key;
		updateResultSorting();
	}));
	updateResultSorting();
	element("case-picker").hidden = false;
	element("case-select").hidden = true;
	document.querySelector('label[for="case-select"]').setAttribute("for", "case-picker-button");
	const requested = new URLSearchParams(window.location.search).get("caso");
	selectCase(requested !== null && /^\d+$/.test(requested) ? Number(requested) : defaultCase, false) || selectCase(defaultCase, false);
	renderTable();
	if (window.TPPChallengeData) {
		initializeChallenge();
		initializePieceChallenge();
		initializePieceChallenge(true);
		initializeChallengeFlow();
	}
	if (element("references-dialog")) initializeReferences();
	initializeDisclosures();
	initializeGuide();
	if (element("toc-handle")) initializeContents();
}

function orderSketch(id, geometry, polygons) {
	if (!polygons.length) return "";
	const centers = polygons.map((polygon) => polygon.reduce((sum, point) => sum.map((value, axis) => value + point[axis] / polygon.length), [0, 0]));
	const points = [geometry.start, ...centers, geometry.target];
	return `<g class="order-sketch" aria-hidden="true"><defs><marker id="sketch-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1 1L8 5L1 9" fill="none" stroke="#c3d9e5" stroke-width="1.8"/></marker></defs>${points.slice(1).map((point, index) => `<path d="M${points[index].join(" ")}L${point.join(" ")}" marker-end="url(#sketch-${id})"/>`).join("")}</g>`;
}

function initializeChallenge() {
	const challenge = window.TPPChallengeData;
	const letters = ["A", "B", "C", "D"];
	const best = challenge.solutions[challenge.reference];
	const greedy = challenge.solutions.find((solution) => solution.order.every((region, index) => region === challenge.greedy.order[index]));
	let order = [], compared = false;
	const map = element("challenge-map");
	map.setAttribute("role", "group");
	function render() {
		const chosen = order.length === challenge.geometry.polygons.length
			? solveChallengeRoute(challenge.geometry.start, challenge.geometry.target, order.map((index) => challenge.geometry.polygons[index]))
			: null;
		map.innerHTML = challenge.geometry.polygons.map((polygon, index) => {
			const [x, y] = polygonCentroid(polygon);
			const rank = order.indexOf(index);
			return `<g role="button" tabindex="0" data-challenge-region="${index}" aria-label="Região ${letters[index]}${rank >= 0 ? `, escolha ${rank + 1}` : ""}" aria-pressed="${rank >= 0}"><polygon points="${coordinates(polygon)}" class="challenge-region ${rank >= 0 ? "chosen" : ""}"/><text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${letters[index]}${rank >= 0 ? ` · ${rank + 1}` : ""}</text></g>`;
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line challenge-solution-route" points="${coordinates(chosen.path)}"/>` : orderSketch("order", challenge.geometry, order.map((index) => challenge.geometry.polygons[index]))) + `<circle cx="${challenge.geometry.start[0]}" cy="${challenge.geometry.start[1]}" r="5" fill="white"/><text x="${challenge.geometry.start[0] - 8}" y="${challenge.geometry.start[1] + 23}">S</text><circle cx="${challenge.geometry.target[0]}" cy="${challenge.geometry.target[1]}" r="5" fill="#ffad66"/><text x="${challenge.geometry.target[0] - 8}" y="${challenge.geometry.target[1] + 23}">T</text>`;
		element("challenge-choices").innerHTML = letters.map((letter, index) => `<button type="button" data-challenge-region="${index}" aria-pressed="${order.includes(index)}">${letter}</button>`).join("");
		element("challenge-order").textContent = `Sua ordem: S → ${order.length ? order.map((index) => letters[index]).join(" → ") + " → " : ""}${order.length < 4 ? "… → " : ""}T`;
		element("challenge-guidance").textContent = order.length === 4 ? "Sequência completa. Compare agora os caminhos." : `Escolha mais ${4 - order.length} ${4 - order.length === 1 ? "região" : "regiões"} para comparar.`;
		element("challenge-compare").disabled = order.length !== 4 || compared;
		element("challenge-undo").disabled = !order.length;
		element("challenge-reset").disabled = !order.length;
		element("challenge-feedback").innerHTML = compared ? `<strong>${challengeVerdict(chosen.length, best.length, "ordens")}</strong>${challengeComparison("Sua escolha", chosen.length, "Melhor das 24", best.length, "Ordem", order.map((index) => letters[index]).join(" → "), best.order.map((index) => letters[index]).join(" → "))}${challengeGreedyNote(greedy, best, "visitar sempre a região não visitada mais próxima da anterior, começando em S,", challenge.greedy.order.map((index) => letters[index]).join(" → "))}` : "";
		if (compared) animateChallengeRoute(map);
	}
	function choose(event) {
		const control = event.target.closest("[data-challenge-region]");
		if (!control) return;
		const index = Number(control.dataset.challengeRegion);
		order = toggleOrderRegion(order, index);
		compared = false;
		element("challenge-next-one").hidden = true;
		const fromMap = map.contains(control);
		render();
		if (event.type === "keydown" || event.detail === 0) {
			const parent = fromMap ? map : element("challenge-choices");
			parent.querySelector(`[data-challenge-region="${index}"]`).focus({ preventScroll: true });
		}
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	element("challenge-choices").addEventListener("click", choose);
	element("challenge-compare").addEventListener("click", () => { compared = true; render(); element("challenge-next-one").hidden = false; });
	element("challenge-undo").addEventListener("click", () => { order.pop(); compared = false; element("challenge-next-one").hidden = true; render(); });
	element("challenge-reset").addEventListener("click", () => { order = []; compared = false; element("challenge-next-one").hidden = true; render(); map.querySelector("[data-challenge-region]").focus({ preventScroll: true }); });
	render();
}

function initializePieceChallenge(combined = false) {
	const all = window.TPPChallengeData;
	const data = combined ? all.combined_challenge : all.piece_challenge;
	const prefix = combined ? "combined" : "piece";
	const ui = (suffix) => element(`${prefix}-${suffix}`);
	let order = [];
	const best = data.solutions[data.reference];
	const letters = combined ? ["A", "B", "C", "D"] : ["A", "B", "C"];
	const greedy = data.solutions.find((solution) => (combined ? solution.order.every((region, index) => region === data.greedy.order[index]) : true)
		&& solution.choices.every((piece, region) => piece === data.greedy.choices[region]));
	let choices = letters.map(() => null), compared = false;
	const map = ui("map");
	function render() {
		const complete = choices.every((piece) => piece !== null) && (!combined || order.length === letters.length);
		const selected = complete
			? solveChallengeRoute(
				data.geometry.start,
				data.geometry.target,
				(combined ? order : letters.map((_, index) => index)).map((region) => data.pieces[region][choices[region]]),
			)
			: null;
		map.innerHTML = data.pieces.map((pieces, region) => {
			const rank = order.indexOf(region);
			const bounds = data.geometry.polygons[region].reduce((box, point) => ({
				minX: Math.min(box.minX, point[0]), maxX: Math.max(box.maxX, point[0]), minY: Math.min(box.minY, point[1]),
			}), { minX: Infinity, maxX: -Infinity, minY: Infinity });
			const regionName = combined && rank >= 0 ? `${letters[region]} · ${rank + 1}ª` : letters[region];
			const pieceMarkup = pieces.map((piece, index) => {
				const [x, y] = polygonCentroid(piece);
				return `<g role="button" tabindex="0" data-piece="${index}" data-piece-region="${region}" aria-label="Região ${letters[region]}, peça ${index + 1}${rank >= 0 ? `, visita ${rank + 1}` : ""}" aria-pressed="${choices[region] === index}"><polygon class="challenge-region ${choices[region] === index ? "chosen" : ""}" points="${coordinates(piece)}"/><text x="${x}" y="${y}" dominant-baseline="central" text-anchor="middle">${index + 1}</text></g>`;
			}).join("");
			return `${pieceMarkup}<text class="challenge-region-name" x="${(bounds.minX + bounds.maxX) / 2}" y="${Math.max(18, bounds.minY - 10)}" text-anchor="middle">${regionName}</text>`;
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line challenge-solution-route" points="${coordinates(selected.path)}"/>` : orderSketch(prefix, data.geometry, combined ? order.map(index => data.pieces[index][choices[index]]) : choices.slice(0, choices.includes(null) ? choices.indexOf(null) : choices.length).map((piece, region) => data.pieces[region][piece]))) + `<circle cx="${data.geometry.start[0]}" cy="${data.geometry.start[1]}" r="5" fill="white"/><text x="${data.geometry.start[0] - 6}" y="${data.geometry.start[1] + 22}">S</text><circle cx="${data.geometry.target[0]}" cy="${data.geometry.target[1]}" r="5" fill="#ffad66"/><text x="${data.geometry.target[0] - 6}" y="${data.geometry.target[1] + 22}">T</text>`;
		ui("choices").innerHTML = letters.map((letter, region) => `<fieldset><legend>Região ${letter}</legend>${[0, 1, 2].map((piece) => `<button type="button" data-piece="${piece}" data-piece-region="${region}" aria-pressed="${choices[region] === piece}" aria-label="${letter}: peça ${piece + 1}">${piece + 1}</button>`).join("")}</fieldset>`).join("");
		ui("selection").textContent = choices.map((piece, region) => `${letters[region]}: ${piece === null ? "?" : `peça ${piece + 1}`}`).join(" · ");
		if (combined) ui("selection").textContent = `Sua sequência: S → ${order.map(index => `${letters[index]}${choices[index] + 1}`).join(" → ")}${order.length < letters.length ? " → …" : ""} → T`;
		const remaining = choices.filter((piece) => piece === null).length;
		ui("guidance").textContent = remaining ? `Faltam ${remaining} ${remaining === 1 ? "região" : "regiões"}.` : "Escolha completa. Compare agora os caminhos.";
		ui("compare").disabled = choices.includes(null) || compared;
		ui("reset").disabled = choices.every((piece) => piece === null);
		const chosenSequence = (combined ? order : letters.map((_, index) => index)).map(region => `${letters[region]}${choices[region] + 1}`).join(" → ");
		const bestSequence = (combined ? best.order : letters.map((_, index) => index)).map(region => `${letters[region]}${best.choices[region] + 1}`).join(" → ");
		const greedySequence = data.greedy.order.map(region => `${letters[region]}${data.greedy.choices[region] + 1}`).join(" → ");
		const greedyDescription = combined ? "visitar a próxima região pela peça mais próxima da peça anterior, a partir de S," : "escolher, em cada região da ordem fixa, a peça mais próxima da anterior, a partir de S,";
		const impact = combined ? '<aside class="challenge-impact"><strong>Achou difícil?</strong> O caso 49 do corpus tem 60 regiões: uma enumeração ingênua teria 60! ≈ 8,3 × 10<sup>81</sup> ordens de visita. Nosso solver concluiu esse caso em 2,47 s, numa execução com uma thread, sem percorrer todas as ordens. Essa contagem bruta não mede sozinha a dificuldade; limites geométricos e podas reduzem a busca.</aside>' : "";
		ui("feedback").innerHTML = compared ? `<strong>${challengeVerdict(selected.length, best.length, "combinações")}</strong>${challengeComparison("Sua escolha", selected.length, `Melhor das ${data.solutions.length.toLocaleString("pt-BR")}`, best.length, combined ? "Solução" : "Peças", chosenSequence, bestSequence)}${challengeGreedyNote(greedy, best, greedyDescription, greedySequence)}${impact}` : "";
		if (compared) animateChallengeRoute(map);
	}
	function choose(event) {
		const control = event.target.closest("[data-piece]");
		if (!control) return;
		const region = Number(control.dataset.pieceRegion), piece = Number(control.dataset.piece);
		const parent = map.contains(control) ? map : ui("choices");
		if (combined) {
			if (choices[region] === piece) { choices[region] = null; order = order.filter(index => index !== region); }
			else { if (choices[region] === null) order.push(region); choices[region] = piece; }
		} else choices[region] = toggleChoice(choices[region], piece);
		compared = false;
		if (combined) element("finish-challenge").hidden = true;
		else element("challenge-next-two").hidden = true;
		render();
		if (event.type === "keydown" || event.detail === 0) parent.querySelector(`[data-piece-region="${region}"][data-piece="${piece}"]`).focus({ preventScroll: true });
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	ui("choices").addEventListener("click", choose);
	ui("compare").addEventListener("click", () => {
		compared = true;
		render();
		if (combined) element("finish-challenge").hidden = false;
		else element("challenge-next-two").hidden = false;
	});
	ui("reset").addEventListener("click", () => {
		choices = letters.map(() => null);
		order = [];
		compared = false;
		if (combined) element("finish-challenge").hidden = true;
		else element("challenge-next-two").hidden = true;
		render();
		map.querySelector("[data-piece]").focus({ preventScroll: true });
	});
	render();
}

function initializeChallengeFlow() {
	const dialog = element("challenge-dialog");
	if (!dialog) return;
	if (dialog.parentElement !== document.body) document.body.append(dialog);
	const steps = [...dialog.querySelectorAll("[data-challenge-step]")];
	const titles = ["Escolha a ordem das visitas", "Escolha uma peça em cada região", "Escolha a ordem e as peças"];
	let current = 0;
	let opener = element("open-challenge");
	let historyEntry = false;
	let sectionNavigation = null;
	const previousSectionButton = element("challenge-previous-section");
	const nextSectionButton = element("challenge-next-section");
	function show(index) {
		current = Math.max(0, Math.min(steps.length - 1, index));
		steps.forEach((step, position) => { step.hidden = position !== current; });
		element("challenge-progress").textContent = `DESAFIO ${current + 1} DE ${steps.length}`;
		element("challenge-dialog-title").textContent = titles[current];
		dialog.querySelectorAll("[data-challenge-tab]").forEach((button, position) => {
			button.classList.toggle("visited", position <= current);
			button.classList.toggle("active", position === current);
			if (position === current) button.setAttribute("aria-current", "step");
			else button.removeAttribute("aria-current");
		});
		dialog.scrollTo({ top: 0, behavior: "instant" });
		const heading = steps[current].querySelector("h3");
		heading.setAttribute("tabindex", "-1");
		heading.focus({ preventScroll: true });
	}
	function openChallenge(source = element("open-challenge"), fromHistory = false, historyEntryOverride = null) {
		opener = source || element("open-challenge");
		if (!dialog.open) {
			historyEntry = historyEntryOverride ?? (!fromHistory && window.location.hash !== "#desafio");
			if (historyEntryOverride === null && historyEntry) window.history.pushState({ tppOverlay: "desafio" }, "", "#desafio");
			showModalWithTransition(dialog);
		}
		show(0);
	}
	function requestClose(fromHistory = false) {
		if (!dialog.open) return;
		if (!fromHistory && historyEntry && window.location.hash === "#desafio") {
			window.history.back();
			return;
		}
		if (!fromHistory && !historyEntry && window.location.hash === "#desafio") {
			window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
		}
		historyEntry = false;
		dialog.close();
	}
	function setSectionNavigationState(state = {}) {
		if (previousSectionButton) previousSectionButton.disabled = !state.previous;
		if (nextSectionButton) nextSectionButton.disabled = !state.next;
	}
	dialog.openChallenge = openChallenge;
	dialog.setSectionNavigation = (callback) => { sectionNavigation = callback; };
	dialog.setSectionNavigationState = setSectionNavigationState;
	dialog.getSectionNavigationState = () => ({ opener, historyEntry });
	dialog.closeForSectionNavigation = () => { if (dialog.open) dialog.close(); };
	element("open-challenge").addEventListener("click", (event) => openChallenge(event.currentTarget));
	element("close-challenge").addEventListener("click", () => requestClose());
	previousSectionButton?.addEventListener("click", () => sectionNavigation?.(-1));
	nextSectionButton?.addEventListener("click", () => sectionNavigation?.(1));
	element("challenge-next-one").addEventListener("click", () => show(1));
	element("challenge-next-two").addEventListener("click", () => show(2));
	dialog.querySelectorAll("[data-challenge-tab]").forEach((button) => button.addEventListener("click", () => show(Number(button.dataset.challengeTab))));
	dialog.querySelectorAll("[data-challenge-back]").forEach((button) => button.addEventListener("click", () => show(current - 1)));
	element("finish-challenge").addEventListener("click", () => requestClose());
	dialog.addEventListener("click", (event) => {
		if (event.target !== dialog) return;
		const box = dialog.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) requestClose();
	});
	dialog.addEventListener("cancel", (event) => { event.preventDefault(); requestClose(); });
	dialog.addEventListener("close", () => opener?.focus({ preventScroll: true }));
	setSectionNavigationState({ previous: false, next: true });
	window.addEventListener("popstate", () => {
		if (dialog.open && window.location.hash !== "#desafio") requestClose(true);
		else if (!dialog.open && window.location.hash === "#desafio") openChallenge(null, true);
	});
}

function initializeReferences() {
	const dialog = element("references-dialog");
	let opener = null;
	let highlightTimer = null;
	document.addEventListener("click", (event) => {
		const trigger = event.target.closest('[data-open-references], a[href^="#ref-"]');
		if (!trigger || dialog.contains(trigger)) return;
		event.preventDefault();
		opener = trigger;
		if (element("mobile-toc")?.open) element("mobile-toc").close();
		if (!dialog.open) showModalWithTransition(dialog);
		const selector = trigger.getAttribute("href");
		const target = selector?.startsWith("#ref-") ? dialog.querySelector(selector) : null;
		dialog.querySelectorAll("li.current-reference").forEach((item) => item.classList.remove("current-reference"));
		if (highlightTimer) window.clearTimeout(highlightTimer);
		target?.classList.add("current-reference");
		if (target) {
			highlightTimer = window.setTimeout(() => {
				target.classList.remove("current-reference");
				highlightTimer = null;
			}, 1800);
		}
		requestAnimationFrame(() => target?.scrollIntoView({ behavior: "smooth", block: "center" }));
	});
	element("references-close").addEventListener("click", () => dialog.close());
	dialog.addEventListener("click", (event) => {
		if (event.target !== dialog) return;
		const box = dialog.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
	});
	dialog.addEventListener("close", () => {
		if (highlightTimer) window.clearTimeout(highlightTimer);
		dialog.querySelectorAll("li.current-reference").forEach((item) => item.classList.remove("current-reference"));
		highlightTimer = null;
		opener?.focus({ preventScroll: true });
	});
}

function initializeDisclosures() {
	const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
	document.querySelectorAll("details").forEach((details) => {
		const summary = details.querySelector(":scope > summary");
		if (!summary) return;
		const body = document.createElement("div");
		body.className = "disclosure-body";
		while (summary.nextSibling) body.append(summary.nextSibling);
		details.append(body);
		if (details.classList.contains("guided-section")) return;
		let expanded = details.open, animation = null;
		const finish = () => {
			details.open = expanded;
			body.style.height = "";
			body.style.overflow = "";
			body.inert = false;
			animation = null;
		};
		summary.addEventListener("click", (event) => {
			event.preventDefault();
			const from = details.open ? body.getBoundingClientRect().height : 0;
			animation?.cancel();
			expanded = !expanded;
			if (reduced.matches) { finish(); return; }
			details.open = true;
			body.inert = !expanded;
			body.style.overflow = "hidden";
			const to = expanded ? body.scrollHeight : 0;
			animation = body.animate([{ height: `${from}px`, opacity: from ? 1 : 0 }, { height: `${to}px`, opacity: expanded ? 1 : 0 }], { duration: 220, easing: "cubic-bezier(.2,.7,.2,1)", fill: "forwards" });
			animation.onfinish = () => { animation.cancel(); finish(); };
		});
		reduced.addEventListener("change", () => { animation?.cancel(); finish(); });
	});
}

function initializeGuide() {
	const copy = {
		desafio: ["2", "Tente você mesmo!", "Escolha a ordem de visita e compare com o solver."],
		metodo: ["3", "Como o algoritmo resolve", "Rota inicial, ramificações, limites e uma execução passo a passo."],
		historia: ["4", "Trabalhos anteriores", "Uma linha do tempo do TPP até este solver autocontido."],
		resultados: ["5", "O que melhoramos", "Compare os resultados deste solver com os anteriores."],
		contato: ["6", "Fale com o autor", "Comentários, dúvidas ou uma conversa sobre a pesquisa."],
	};
	const ids = ["desafio", "metodo", "historia", "resultados", "contato"];
	const nodes = new Map();
	for (const id of ids) {
		let node = element(id);
		if (!node) continue;
		if (!node.id) node.id = id;
		node.dataset.guideSection = id;
		if (node.matches("section")) {
			node.classList.add("guided-section");
			const [number, title, subtitle] = copy[id] || ["", id, ""];
			const toggle = document.createElement("button");
			toggle.type = "button";
			toggle.className = "guided-summary guided-toggle";
			toggle.dataset.guideToggle = id;
			toggle.innerHTML = `<span><b>${number}</b><strong>${title}</strong><small>${subtitle}</small></span><em data-guide-status="${id}">Ainda não visitada</em>`;
			const panel = document.createElement("div");
			panel.className = "guided-panel";
			while (node.firstChild) panel.append(node.firstChild);
			node.append(toggle, panel);
			panel.hidden = true;
			panel.inert = true;
			toggle.setAttribute("aria-expanded", "false");
		}
		node.hidden = true;
		node.inert = true;
		nodes.set(id, node);
	}
	if (!nodes.size) return;
	const dialog = element("section-dialog");
	if (!dialog) return;
	const content = element("section-dialog-content");
	const closeButton = element("close-section");
	const previousSectionButton = element("previous-section");
	const nextSectionButton = element("next-section");
	const challengeDialog = element("challenge-dialog");
	const title = element("section-dialog-title");
	const eyebrow = element("section-dialog-eyebrow");
	const subtitle = element("section-dialog-subtitle");
	const resetDialogScroll = () => {
		dialog.scrollTop = 0;
		content.scrollTop = 0;
	};
	const storageKey = "tpp-siicusp34-visited-sections";
	let visited = new Set();
	try {
		visited = new Set(JSON.parse(localStorage.getItem(storageKey) || "[]").filter((id) => nodes.has(id)));
	} catch {
		visited = new Set();
	}
	let active = null;
	const statusText = (id) => visited.has(id) ? "Visitada" : "Ainda não visitada";
	function update(id) {
		const node = nodes.get(id);
		if (!node) return;
		const isOpen = active?.id === id;
		node.classList.toggle("is-open", isOpen);
		node.classList.toggle("is-visited", visited.has(id));
		document.querySelectorAll(`[data-guide-status="${id}"]`).forEach((status) => { status.textContent = statusText(id); });
		document.querySelectorAll(`[data-guide-target="${id}"]`).forEach((link) => link.classList.toggle("is-visited", visited.has(id)));
		const toggle = node.querySelector(":scope > .guided-toggle");
		if (toggle) toggle.setAttribute("aria-expanded", String(isOpen));
		const summary = node.matches("details") ? node.querySelector(":scope > .guided-summary") : null;
		if (summary) summary.setAttribute("aria-expanded", String(isOpen));
	}
	function markVisited(id) {
		if (!nodes.has(id)) return;
		visited.add(id);
		try { localStorage.setItem(storageKey, JSON.stringify([...visited])); } catch { /* Private browsing can reject storage. */ }
		update(id);
		const count = element("guide-title")?.closest(".guide-nav")?.querySelector("[data-guide-count]");
		if (count) count.textContent = `${visited.size}/${nodes.size} etapas visitadas`;
	}
	function sourceFor(node) {
		return node.matches("details")
			? node.querySelector(":scope > .disclosure-body")
			: node.querySelector(":scope > .guided-panel");
	}
	function restoreSource(entry) {
		if (!entry || entry.source.parentElement !== content) return;
		entry.source.hidden = entry.node.matches("section");
		entry.source.inert = entry.node.matches("section");
		entry.node.append(entry.source);
	}
	function updateNavigation() {
		const index = active ? ids.indexOf(active.id) : -1;
		if (previousSectionButton) previousSectionButton.disabled = index <= 0;
		if (nextSectionButton) nextSectionButton.disabled = index < 0 || index >= ids.length - 1;
		challengeDialog?.setSectionNavigationState?.({ previous: false, next: true });
	}
	function clearHash() {
		window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
	}
	function close(fromHistory = false) {
		if (!active) return;
		if (!fromHistory && active.historyEntry && window.location.hash === `#${active.id}`) {
			window.history.back();
			return;
		}
		if (!fromHistory && !active.historyEntry && window.location.hash === `#${active.id}`) clearHash();
		dialog.close();
	}
	function open(id, opener = null, fromHistory = false, historyEntryOverride = null) {
		const node = nodes.get(id);
		if (!node) return;
		markVisited(id);
		if (id === "desafio") {
			challengeDialog?.openChallenge?.(opener, fromHistory, historyEntryOverride);
			return;
		}
		const source = sourceFor(node);
		if (!source) return;
		if (active && active.id !== id) close(true);
		const shouldPushHistory = historyEntryOverride === null && !fromHistory && window.location.hash !== `#${id}`;
		active = { id, node, source, opener: opener || node.querySelector(":scope > .guided-summary, :scope > .guided-toggle"), historyEntry: historyEntryOverride ?? shouldPushHistory };
		if (shouldPushHistory) window.history.pushState({ tppOverlay: id }, "", `#${id}`);
		source.hidden = false;
		source.inert = false;
		content.append(source);
		resetDialogScroll();
		dialog.dataset.section = id;
		eyebrow.textContent = `ETAPA ${copy[id][0]} de 6`;
		title.textContent = copy[id][1];
		subtitle.textContent = copy[id][2];
		update(id);
		updateNavigation();
		if (!dialog.open) showModalWithTransition(dialog);
		closeButton.focus({ preventScroll: true });
	}
	dialog.openGuideSection = (id, opener = null) => open(id, opener);
	dialog.closeGuideSection = () => {
		if (!active) return;
		if (window.location.hash === `#${active.id}`) clearHash();
		dialog.close();
	};
	function navigateFrom(currentId, delta, navigationState = null) {
		const currentIndex = ids.indexOf(currentId);
		const nextId = ids[currentIndex + delta];
		if (currentIndex < 0 || !nextId) return;
		if (nextId === "desafio") {
			if (!active || !dialog.open) return;
			const current = active;
			restoreSource(current);
			active = null;
			dialog.removeAttribute("data-section");
			update(current.id);
			updateNavigation();
			dialog.close();
			window.history.replaceState(window.history.state, "", "#desafio");
			challengeDialog?.openChallenge?.(current.opener, true, current.historyEntry);
			return;
		}
		if (currentId === "desafio") {
			const state = navigationState || challengeDialog?.getSectionNavigationState?.();
			challengeDialog?.closeForSectionNavigation?.();
			window.history.replaceState(window.history.state, "", `#${nextId}`);
			open(nextId, state?.opener || null, true, state?.historyEntry ?? false);
			return;
		}
		if (!active || active.id !== currentId) return;
		const node = nodes.get(nextId), source = sourceFor(node);
		if (!node || !source) return;
		const previous = active;
		restoreSource(previous);
		active = { id: nextId, node, source, opener: previous.opener, historyEntry: previous.historyEntry };
		markVisited(nextId);
		source.hidden = false;
		source.inert = false;
		content.append(source);
		resetDialogScroll();
		dialog.dataset.section = nextId;
		eyebrow.textContent = `ETAPA ${copy[nextId][0]} de 6`;
		title.textContent = copy[nextId][1];
		subtitle.textContent = copy[nextId][2];
		update(previous.id);
		update(nextId);
		updateNavigation();
		if (window.location.hash !== `#${nextId}`) window.history.replaceState(window.history.state, "", `#${nextId}`);
	}
	dialog.addEventListener("close", () => {
		const closing = active;
		if (!closing) return;
		restoreSource(closing);
		active = null;
		dialog.removeAttribute("data-section");
		update(closing.id);
		updateNavigation();
		closing.opener?.focus({ preventScroll: true });
	});
	dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
	dialog.addEventListener("click", (event) => {
		if (event.target !== dialog) return;
		const box = dialog.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) close();
	});
	closeButton.addEventListener("click", () => close());
	previousSectionButton?.addEventListener("click", () => navigateFrom(active?.id, -1));
	nextSectionButton?.addEventListener("click", () => navigateFrom(active?.id, 1));
	challengeDialog?.setSectionNavigation?.((delta) => navigateFrom("desafio", delta));
	window.addEventListener("popstate", () => {
		if (active && window.location.hash !== `#${active.id}`) close(true);
		else if (!active) {
			const id = window.location.hash.slice(1);
			if (nodes.has(id)) open(id, null, true);
		}
	});
	for (const [id, node] of nodes) {
		const trigger = node.matches("details")
			? node.querySelector(":scope > .guided-summary")
			: node.querySelector(":scope > .guided-toggle");
		trigger?.addEventListener("click", (event) => {
			event.preventDefault();
			open(id, event.currentTarget);
		});
		update(id);
	}
	document.querySelectorAll("[data-guide-target]").forEach((link) => link.addEventListener("click", (event) => {
			const id = link.dataset.guideTarget;
			if (!nodes.has(id)) return;
			event.preventDefault();
			open(id, link);
		}));
	document.querySelectorAll("[data-open-comparison]").forEach((link) => link.addEventListener("click", (event) => {
		event.preventDefault();
		if (active?.id === "historia" && nextSectionButton && !nextSectionButton.disabled) {
			nextSectionButton.click();
			return;
		}
		document.querySelector('.guide-links [data-guide-target="resultados"]')?.click();
	}));
	const count = element("guide-title")?.closest(".guide-nav")?.querySelector("[data-guide-count]");
	if (count) count.textContent = `${visited.size}/${nodes.size} etapas visitadas`;
	updateNavigation();
	const initial = window.location.hash.slice(1);
	if (nodes.has(initial)) open(initial, null, true);
}

function initializeContents() {
	const handle = element("toc-handle"), dialog = element("mobile-toc");
	const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mobile = window.matchMedia("(max-width: 850px)");
	let animation = null;
	function open() {
		if (dialog.open) return;
		animation?.cancel();
		dialog.showModal();
		handle.setAttribute("aria-expanded", "true");
		if (!reduced.matches) animation = dialog.animate([{ transform: "translateX(100%)" }, { transform: "translateX(0)" }], { duration: 200, easing: "ease-out" });
		element("toc-close").focus();
	}
	function close(after) {
		animation?.cancel();
		const finish = () => { dialog.close(); handle.setAttribute("aria-expanded", "false"); after?.(); };
		if (reduced.matches) { finish(); return; }
		animation = dialog.animate([{ transform: "translateX(0)" }, { transform: "translateX(100%)" }], { duration: 180, easing: "ease-in" });
		animation.onfinish = finish;
	}
	handle.hidden = false;
	handle.addEventListener("click", open);
	element("toc-close").addEventListener("click", () => close());
	dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
	dialog.addEventListener("close", () => handle.setAttribute("aria-expanded", "false"));
	dialog.addEventListener("click", (event) => {
		const link = event.target.closest('a[href^="#"]');
		if (link) {
			event.preventDefault();
			const target = document.getElementById(link.hash.slice(1));
			close(() => {
				if (target.matches("details") && !target.open) target.querySelector("summary").click();
				target.scrollIntoView({ behavior: reduced.matches ? "instant" : "smooth" });
				const focusTarget = target.matches("details") ? target.querySelector("summary") : target;
				if (!focusTarget.hasAttribute("tabindex")) focusTarget.setAttribute("tabindex", "-1");
				focusTarget.focus({ preventScroll: true });
				dialog.querySelectorAll("a").forEach(item => item.removeAttribute("aria-current"));
				link.setAttribute("aria-current", "location");
			});
		} else if (event.target === dialog) {
			const box = dialog.getBoundingClientRect();
			if (event.clientX < box.left) close();
		}
	});
	for (const surface of [handle, dialog]) {
		let start = null;
		surface.addEventListener("pointerdown", event => { if (event.isPrimary) { start = [event.clientX, event.clientY]; if (surface === handle) surface.setPointerCapture(event.pointerId); } });
		surface.addEventListener("pointerup", event => {
			if (!start) return;
			const dx = event.clientX - start[0], dy = event.clientY - start[1];
			start = null;
			if (Math.abs(dx) < 35 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
			if (surface === handle && dx < 0) open();
			if (surface === dialog && dx > 0) close();
		});
		surface.addEventListener("pointercancel", () => { start = null; });
	}
	mobile.addEventListener("change", () => { if (!mobile.matches && dialog.open) { animation?.cancel(); dialog.close(); } });
}

initialize().catch((error) => {
	element("event-error").hidden = false;
	element("event-error").textContent = "Não foi possível iniciar os controles interativos. Os resultados e a visualização estática continuam disponíveis. Recarregue a página para tentar novamente.";
	console.error(error);
});
