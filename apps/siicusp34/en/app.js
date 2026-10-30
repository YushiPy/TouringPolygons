import { tppSolveConvex } from "../tpp-solver.js";

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


function formatDuration(value) {
	const seconds = Number(value);
	if (!Number.isFinite(seconds)) return "—";
	if (seconds < 0.001) return "< 0.001 s";
	if (seconds < 1) return `${number(seconds, 3)} s`;
	if (seconds < 60) return `${number(seconds, 2)} s`;
	const minutes = seconds / 60;
	if (minutes < 60) return `${number(minutes, 2)} min`;
	return `${number(minutes / 60, 2)} hours`;
}

function formatDistance(value) {
	const distance = Number(value);
	if (!Number.isFinite(distance)) return "—";
	if (distance === 0) return "0";
	if (Math.abs(distance) >= 0.01) return number(distance, 3);
	return distance.toExponential(2);
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

function playbackDuration(selected) {
	return 2400 + 50 * Math.min(80, Math.max(0, selected.polygons));
}

function toggleOrderRegion(order, index) {
	return order.includes(index) ? order.filter(value => value !== index) : [...order, index];
}

function toggleChoice(current, next) {
	return current === next ? null : next;
}


const element = (id) => document.getElementById(id);
const number = (value, digits = 4) => value.toLocaleString("en-US", { maximumFractionDigits: digits });
const coordinates = (points) => points.map((point) => point.join(",")).join(" ");
const caseLabel = (id) => String(id + 1).padStart(id + 1 >= 100 ? 3 : 2, "0");
const visitorRows = (rows, query) => {
	const term = query.trim();
	if (!term) return rows;
	const requestedCase = /^\d+$/.test(term) ? Number(term) - 1 : null;
	return rows.filter((row) => (requestedCase !== null && row.case === requestedCase) || caseLabel(row.case).includes(term));
};
const MAX_ZOOM = 8;

function closeOnBackdropClick(dialog) {
	dialog.addEventListener("click", (event) => {
		if (event.target !== dialog) return;
		const box = dialog.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
	});
}

function showModalWithTransition(dialog) {
	if (!dialog || dialog.open) return;
	dialog.classList.remove("is-entering");
	dialog.showModal();
	if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
	dialog.classList.add("is-entering");
	requestAnimationFrame(() => dialog.classList.remove("is-entering"));
}


function pathLength(path) {
	return Array.isArray(path) ? path.slice(1).reduce((total, point, index) => total + Math.hypot(point[0] - path[index][0], point[1] - path[index][1]), 0) : null;
}

function solveChallengeRoute(start, target, polygons) {
	const vectorPath = tppSolveConvex(start, target, polygons, true);
	const path = vectorPath.map(({ x, y }) => [x, y]);
	return { path, length: pathLength(path) };
}

function methodDiagramProjector(geometry, polygonIndices, routePaths = [], width = 320, height = 160) {
	const polygons = polygonIndices.flatMap((index) => geometry.polygons[index] || []);
	const points = [...polygons, geometry.start, geometry.target, ...routePaths.flat()];
	const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
	const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
	const padding = 24;
	const scale = Math.min((width - padding * 2) / Math.max(maxX - minX, 1e-9), (height - padding * 2) / Math.max(maxY - minY, 1e-9));
	const offsetX = (width - (maxX - minX) * scale) / 2;
	const offsetY = (height - (maxY - minY) * scale) / 2;
	return ([x, y]) => [+(offsetX + (x - minX) * scale).toFixed(2), +(offsetY + (y - minY) * scale).toFixed(2)];
}

function methodPolygonMarkup(polygon, project, className = "diagram-region") {
	return `<polygon class="${className}" points="${coordinates(polygon.map(project))}"/>`;
}

function methodEndpointMarkup(start, target, project) {
	const [sx, sy] = project(start), [tx, ty] = project(target);
	return `<circle class="diagram-endpoint" cx="${sx}" cy="${sy}" r="3.5"/><text class="diagram-point-label" x="${sx - 6}" y="${sy + 12}" text-anchor="middle">s</text><circle class="diagram-endpoint" cx="${tx}" cy="${ty}" r="3.5"/><text class="diagram-point-label" x="${tx + 7}" y="${ty - 4}">t</text>`;
}

function methodRegionLabelMarkup(polygon, label, project) {
	const [x, y] = project(polygonCentroid(polygon));
	return `<text class="diagram-region-label" x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${label}</text>`;
}

function methodFixedOrderRoute(geometry, decomposition, order) {
	let best = null;
	function choose(position, selectedPieces) {
		if (position === order.length) {
			const route = solveChallengeRoute(geometry.start, geometry.target, selectedPieces.map((piece) => piece.polygon));
			if (!best || route.length < best.length) best = { ...route, choices: selectedPieces.map((piece) => piece.choice), order: [...order] };
			return;
		}
		const region = order[position];
		for (let choice = 0; choice < decomposition[region].length; choice += 1) {
			choose(position + 1, [...selectedPieces, { choice, polygon: decomposition[region][choice] }]);
		}
	}
	choose(0, []);
	return best;
}

function drawAlgorithmFigures() {
	const data = window.TPPChallengeData;
	if (!data?.geometry || !data.piece_challenge || !data.combined_challenge) return;

	const heuristicFigure = document.getElementById("method-heuristic-diagram");
	const greedyPath = solveChallengeRoute(data.geometry.start, data.geometry.target,
		data.greedy.order.map((index) => data.geometry.polygons[index])).path;
	const optimum = data.solutions[data.reference];
	const firstProjection = methodDiagramProjector(data.geometry, data.geometry.polygons.map((_, index) => index), [greedyPath, optimum.path]);
	heuristicFigure.innerHTML = `${data.geometry.polygons.map((polygon) => methodPolygonMarkup(polygon, firstProjection)).join("")}
		<polyline class="diagram-optimal-route" points="${coordinates(optimum.path.map(firstProjection))}"/>
		<polyline class="diagram-heuristic-route" points="${coordinates(greedyPath.map(firstProjection))}"/>
		${methodEndpointMarkup(data.geometry.start, data.geometry.target, firstProjection)}`;
	const greedyLength = pathLength(greedyPath) * data.meters_per_display_unit;
	document.getElementById("method-heuristic-length").textContent = `${number(greedyLength, 2)} m`;
	document.getElementById("method-optimal-length").textContent = `${number(optimum.length, 2)} m`;

	const combined = data.combined_challenge;
	const geometry = combined.geometry;
	// Challenge 3 is the full author-drawn five-region instance. This partial node fixes A then B;
	// its shortest route is recomputed over the convex pieces by the bundled exact solver.
	const regions = { A: 2, B: 1, C: 4, D: 0, E: 3 };
	const partialOrder = [regions.A, regions.B];
	const partial = methodFixedOrderRoute(geometry, combined.pieces, partialOrder);
	const partialContacts = pathPolygonContacts(partial.path, geometry.polygons);
	if (!partialContacts[regions.A] || !partialContacts[regions.B] || !partialContacts[regions.C] || partialContacts[regions.D] || partialContacts[regions.E]) {
		throw new Error("The selected partial-order example no longer visits C without visiting D.");
	}
	const partialIndices = geometry.polygons.map((_, index) => index);
	const partialProjection = methodDiagramProjector(geometry, partialIndices, [partial.path]);
	const roleByRegion = new Map(Object.entries(regions).map(([label, index]) => [index, label]));
	const partialClasses = new Map([
		[regions.A, "diagram-region"], [regions.B, "diagram-region"],
		[regions.C, "diagram-region diagram-region-incidental"],
		[regions.D, "diagram-region diagram-region-unvisited"], [regions.E, "diagram-region diagram-region-unvisited"],
	]);
	document.getElementById("method-partial-diagram").innerHTML = `${partialIndices.map((index) => methodPolygonMarkup(geometry.polygons[index], partialProjection, partialClasses.get(index))).join("")}
		<polyline class="diagram-heuristic-route" points="${coordinates(partial.path.map(partialProjection))}"/>
		${partialIndices.map((index) => methodRegionLabelMarkup(geometry.polygons[index], roleByRegion.get(index), partialProjection)).join("")}
		${methodEndpointMarkup(geometry.start, geometry.target, partialProjection)}`;

	const selectedRegions = geometry.polygons.map((_, index) => index);
	const complete = combined.solutions[combined.reference];
	const completeContacts = pathPolygonContacts(complete.path, geometry.polygons);
	if (selectedRegions.some((index) => !completeContacts[index])) throw new Error("The full-instance reference route must touch every target region.");
	const completeProjection = methodDiagramProjector(geometry, selectedRegions, [complete.path]);
	document.getElementById("method-complete-diagram").innerHTML = `${selectedRegions.map((index) => methodPolygonMarkup(geometry.polygons[index], completeProjection)).join("")}
		<polyline class="diagram-heuristic-route" points="${coordinates(complete.path.map(completeProjection))}"/>
		${selectedRegions.map((index) => methodRegionLabelMarkup(geometry.polygons[index], roleByRegion.get(index), completeProjection)).join("")}
		${methodEndpointMarkup(geometry.start, geometry.target, completeProjection)}`;

	const pieceChallenge = data.piece_challenge;
	const pieceSolution = pieceChallenge.solutions[pieceChallenge.reference];
	const pieceGeometry = pieceChallenge.geometry;
	const pieceIndices = pieceGeometry.polygons.map((_, index) => index);
	const pieceProjection = methodDiagramProjector(pieceGeometry, pieceIndices, [pieceSolution.path]);
	const pieceMarkup = pieceChallenge.pieces.flatMap((pieces, regionIndex) => pieces.map((piece, pieceIndex) =>
		methodPolygonMarkup(piece, pieceProjection, `diagram-piece-fill${pieceIndex === pieceSolution.choices[regionIndex] ? " diagram-piece-chosen" : ""}`))).join("");
	const cutsMarkup = pieceChallenge.pieces.flatMap((pieces, regionIndex) => decompositionEdges(pieces, pieceGeometry.polygons[regionIndex]).map(([start, end]) => {
		const [x1, y1] = pieceProjection(start), [x2, y2] = pieceProjection(end);
		return `<line class="diagram-piece-cut" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
	})).join("");
	document.getElementById("method-decomposition-diagram").innerHTML = `${pieceMarkup}
		${pieceGeometry.polygons.map((polygon) => methodPolygonMarkup(polygon, pieceProjection, "diagram-region-outline")).join("")}
		${cutsMarkup}<polyline class="diagram-heuristic-route" points="${coordinates(pieceSolution.path.map(pieceProjection))}"/>
		${pieceIndices.map((index) => methodRegionLabelMarkup(pieceGeometry.polygons[index], String.fromCharCode(65 + index), pieceProjection)).join("")}
		${methodEndpointMarkup(pieceGeometry.start, pieceGeometry.target, pieceProjection)}`;
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
	return `<table class="challenge-comparison"><thead><tr><td></td><th><i class="comparison-line chosen" aria-hidden="true"></i>${chosenLabel}</th><th><i class="comparison-line reference" aria-hidden="true"></i>${bestLabel}</th></tr></thead><tbody><tr><th scope="row">Length</th><td>${number(chosenLength, 2)} m</td><td>${number(bestLength, 2)} m</td></tr><tr><th scope="row">${choiceLabel}</th><td>${chosenChoice}</td><td>${bestChoice}</td></tr></tbody></table>`;
}

function challengeVerdict(chosenLength, bestLength, name) {
	const percent = 100 * (chosenLength / bestLength - 1);
	if (percent < 1e-5) return `You found one of the best ${name}!`;
	if (percent < 1) return "Your path is within 1% of the shortest one found.";
	return `Your path is ${number(percent, 1)}% longer.`;
}

async function initialize() {
	const data = window.TPPEventData;
	const uspDemo = window.TPPUspDemo;
	const spBairrosDemo = window.TPPSpBairrosDemo;
	const brEstadosDemo = window.TPPBrEstadosDemo;
	if (data.schema_version !== 1 || !data.rows.length || uspDemo?.schema_version !== 1 || uspDemo.case !== "usp" || spBairrosDemo?.schema_version !== 1 || spBairrosDemo.case !== "sp-bairros" || brEstadosDemo?.schema_version !== 1 || brEstadosDemo.case !== "br-estados") throw new Error("Demonstration data unavailable.");
	drawAlgorithmFigures();
	const defaultCase = "br-estados";
	let row = brEstadosDemo;
	let projected, fraction = 1, zoom = 1, frame = 0, playing = false;
	let routeWidth = 840, routeHeight = 480;
	let pan = [0, 0], drag = null;
	const speeds = [.25, .5, 1, 1.5, 2, 3, 4];
	let speedIndex = 2;
	let showAllResults = false;
	const sorting = { picker: { key: "case", descending: false }, result: { key: "case", descending: false } };
	const enabled = (id) => element(id).getAttribute("aria-pressed") === "true";
	const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mapContent = element("map-content");
	const map = element("route-map");
	function populateCaseSelect() {
		const select = element("case-select");
		if (!select) return;
		select.innerHTML = data.rows.map((item) => `<option value="${item.case}">Case ${caseLabel(item.case)} · ${item.polygons} regions</option>`).join("");
	}

	// Figure 2 of the international abstract: one point per instance both solve, against its number of
	// polygons, with the median and the arithmetic mean over the sizes rounded to a multiple of 5.
	function renderSpeedupChart() {
		const panel = element("speedup-panel");
		const host = element("speedup-chart");
		if (!panel || !host) return;
		const points = data.rows.filter((item) => Number.isFinite(item.fekete_seconds) && item.seconds > 0)
			.map((item) => ({ case: item.case, polygons: item.polygons, speedup: item.fekete_seconds / item.seconds }));
		if (!points.length) return;
		panel.hidden = false;
		const measured = Math.round(host.clientWidth);
		if (!host.dataset.observed && "ResizeObserver" in window) {
			host.dataset.observed = "true";
			new ResizeObserver(() => { if (Math.abs(Math.round(host.clientWidth) - Number(host.dataset.width || 0)) > 8) renderSpeedupChart(); }).observe(host);
		}
		host.dataset.width = String(measured);
		const groups = new Map();
		for (const point of points) {
			const size = 5 * Math.max(1, Math.round(point.polygons / 5));
			if (!groups.has(size)) groups.set(size, []);
			groups.get(size).push(point.speedup);
		}
		// Sizes with fewer than 20 instances (a few tessellations) get no line point, as in the abstract.
		const sizes = [...groups.keys()].filter((size) => groups.get(size).length >= 20).sort((a, b) => a - b);
		const median = (values) => { const sorted = [...values].sort((a, b) => a - b), middle = sorted.length / 2; return sorted.length % 2 ? sorted[Math.floor(middle)] : (sorted[middle - 1] + sorted[middle]) / 2; };
		const medians = sizes.map((size) => median(groups.get(size)));
		const means = sizes.map((size) => groups.get(size).reduce((sum, value) => sum + value, 0) / groups.get(size).length);
		const width = Math.max(300, measured || 900), narrow = width < 560, height = narrow ? 320 : 380, left = narrow ? 64 : 82, right = 14, top = 20, bottom = narrow ? 58 : 56;
		const x = (polygons) => left + (polygons / 63) * (width - left - right);
		const y = (speedup) => top + (1 - Math.log10(Math.min(10000, Math.max(1, speedup))) / 4) * (height - top - bottom);
		let svg = `<svg viewBox="0 0 ${width} ${height}" focusable="false">`;
		for (let power = 0; power <= 4; power += 1) svg += `<line class="grid" x1="${left}" x2="${width - right}" y1="${y(10 ** power)}" y2="${y(10 ** power)}"/><text class="tick" x="${left - 10}" y="${y(10 ** power) + 4}" text-anchor="end">${number(10 ** power, 0)}×</text>`;
		for (let polygons = 0; polygons <= 60; polygons += 10) svg += `<line class="tick-mark" x1="${x(polygons)}" x2="${x(polygons)}" y1="${height - bottom}" y2="${height - bottom + 6}"/><text class="tick" x="${x(polygons)}" y="${height - bottom + 24}" text-anchor="middle">${polygons}</text>`;
		svg += `<line class="axis" x1="${left}" x2="${width - right}" y1="${height - bottom}" y2="${height - bottom}"/>`;
		svg += points.map((point) => `<circle class="speedup-point" data-open-speedup-case="${point.case}" cx="${x(point.polygons).toFixed(1)}" cy="${y(point.speedup).toFixed(1)}" r="${narrow ? 2.4 : 3}"><title>Case ${caseLabel(point.case)} · ${point.polygons} polygons · ${number(point.speedup, 1)}×</title></circle>`).join("");
		const line = (values, className) => `<polyline class="${className}" points="${sizes.map((size, index) => `${x(size)},${y(values[index])}`).join(" ")}"/>` + sizes.map((size, index) => `<circle class="${className}-dot" cx="${x(size)}" cy="${y(values[index])}" r="3.5"/>`).join("");
		svg += line(means, "speedup-mean-line") + line(medians, "speedup-median-line");
		svg += sizes.map((size, index) => [5, 30, 60].includes(size) ? `<text class="speedup-size-label" x="${x(size)}" y="${y(medians[index] / 2.4)}" text-anchor="middle">${number(medians[index], 0)}×</text>` : "").join("");
		svg += `<text class="axis-title" x="${(left + width - right) / 2}" y="${height - 10}" text-anchor="middle">Polygons in the instance</text>`;
		// On a phone the tick labels (1× to 10,000×) already name the axis; a rotated title would overlap them.
		svg += `${narrow ? "" : `<text class="axis-title" transform="translate(16 ${(top + height - bottom) / 2}) rotate(-90)" text-anchor="middle">Speedup (Fekete et al. ÷ ours), log scale</text>`}</svg>`;
		host.innerHTML = svg + `<div class="speedup-legend"><span><i class="speedup-legend-point"></i>Instance</span><span><i class="speedup-legend-median"></i>Median</span><span><i class="speedup-legend-mean"></i>Arithmetic mean</span></div><p class="speedup-hint">Click (or tap) a point to open that instance on the map.</p>`;
		host.querySelectorAll("[data-open-speedup-case]").forEach((point) => point.addEventListener("click", () => openCaseOnMap(Number(point.dataset.openSpeedupCase), host)));
		element("speedup-chart-summary").textContent = `Speedup over Fekete et al., one thread each, on the ${points.length} instances both solve, against the number of polygons. The median grows from ${number(medians[0], 0)}× with about ${sizes[0]} polygons to ${number(medians.at(-1), 0)}× with about ${sizes.at(-1)}.`;
	}

	function updateResultSummary() {
		const comparison = data.comparison || {};
		const common = Number(comparison.common_completed || 0);
		const solved = comparison.solved || {};
		const comparisonTime = comparison.time || {};
		const precision = comparison.precision || {};
		const cases = data.rows.length;
		renderSpeedupChart();
		const setText = (id, value) => { const target = element(id); if (target) target.textContent = value; };
		const fraction = (value, total) => `${Number(value ?? 0)}/${Number(total ?? 0)}`;
		const speedup = (value) => Number.isFinite(Number(value)) ? `${number(Number(value), 0)}×` : "—";
		setText("solved-fekete-paper-300", fraction(solved.fekete_paper_300, cases));
		setText("solved-fekete-paper-900", fraction(solved.fekete_paper_900, cases));
		setText("solved-fekete-single-300", fraction(solved.fekete_single_300, cases));
		setText("solved-fekete-single-900", fraction(solved.fekete_single_900, cases));
		setText("solved-ours-300", fraction(solved.ours_300, cases));
		setText("solved-ours-900", fraction(solved.ours_300, cases));
		if (comparison.hardest) {
			setText("solved-hardest", `The last instance, case ${caseLabel(comparison.hardest.case)}, is open for every configuration of Fekete et al.; our solver, with one thread, closes it in ${number(comparison.hardest.seconds / 60, 0)} minutes. In 300 s, all three runs of a configuration must close the instance.`);
		}
		const left = cases - common;
		setText("speedup-panel-text", `Speedup over Fekete et al., one thread each, on the ${common} instances both solve; median and arithmetic mean by size, rounded to a multiple of 5. Every point lies above 1×. The ${left} instances that Fekete et al. do not solve with one thread are left out, since no speedup is measured; ours solves all of them.`);
		setText("comparison-time-scope", `One thread each, on the ${common} instances both solve.`);
		setText("comparison-time-ours-median", formatDuration(comparisonTime.common_ours_median_seconds));
		setText("comparison-time-fekete-median", formatDuration(comparisonTime.common_fekete_median_seconds));
		setText("comparison-time-ours-mean", formatDuration(comparisonTime.common_ours_mean_seconds));
		setText("comparison-time-fekete-mean", formatDuration(comparisonTime.common_fekete_mean_seconds));
		setText("comparison-time-ours-total", formatDuration(comparisonTime.common_ours_total_seconds));
		setText("comparison-time-fekete-total", formatDuration(comparisonTime.common_fekete_total_seconds));
		setText("comparison-speedup-median", speedup(comparison.median_speedup_fekete_over_ours));
		setText("comparison-speedup-mean", speedup(comparison.mean_speedup_fekete_over_ours));
		setText("comparison-speedup-small", speedup(comparison.by_size?.["5"]?.median));
		setText("comparison-speedup-large", speedup(Math.round(Number(comparison.by_size?.["60"]?.median) / 10) * 10));
		setText("comparison-speedup-ours-faster", fraction(comparison.ours_faster_count, common));
		setText("comparison-precision-ours-median", formatDistance(precision.ours?.median));
		setText("comparison-precision-fekete-median", formatDistance(precision.fekete?.median));
		setText("comparison-precision-ours-p95", formatDistance(precision.ours?.p95));
		setText("comparison-precision-fekete-p95", formatDistance(precision.fekete?.p95));
		setText("comparison-precision-ours-threshold", fraction(precision["ours_above_1e-7"], precision.ours?.n));
		setText("comparison-precision-fekete-threshold", fraction(precision["fekete_above_1e-7"], precision.fekete?.n));
	}

	function camera() {
		const centerX = routeWidth / 2, centerY = routeHeight / 2;
		pan = [Math.max(-centerX * (zoom - 1), Math.min(centerX * (zoom - 1), pan[0])), Math.max(-centerY * (zoom - 1), Math.min(centerY * (zoom - 1), pan[1]))];
		mapContent.setAttribute("transform", `translate(${centerX + pan[0]} ${centerY + pan[1]}) scale(${zoom}) translate(${-centerX} ${-centerY})`);
		map.classList.toggle("is-zoomed", zoom > 1);
		map.style.touchAction = zoom > 1 ? "none" : "pan-y";
		const label = zoom === 1 ? "1×" : `${number(zoom, 1)}×`;
		element("zoom-slider").value = String(Math.round(zoom * 10));
		element("zoom-slider").setAttribute("aria-valuetext", label);
		element("zoom-level").textContent = label;
		element("map-navigation").textContent = zoom > 1 ? `Zoom ${label}. The wheel or a pinch adjusts the zoom; return to 1× to scroll the page.` : "At 1×, the wheel scrolls the page. Use the slider or pinch on the screen or trackpad to zoom in; then the wheel adjusts the zoom.";
	}

	function stop() {
		cancelAnimationFrame(frame);
		playing = false;
		element("play-route").textContent = "▶ Watch the tour";
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
			const colors = row.case === "usp" && ["ime-a", "ime-b", "ime-c"].includes(row.buildings[index].qgis_id)
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
				? ""
				: `Next target: ${row.buildings[next].label}.`);
		}
	}

	function updateLayerNotes() {
		document.querySelectorAll("[data-layer-note]").forEach((item) => item.classList.toggle("active", enabled(item.dataset.layerNote)));
	}

	function draw() {
		const mobilePortraitRoute = ["usp", "sp-bairros", "br-estados"].includes(row.case) && window.matchMedia("(max-width: 560px)").matches;
		map.classList.toggle("usp-route", row.case === "usp");
		map.classList.toggle("portrait-route", mobilePortraitRoute);
		routeWidth = mobilePortraitRoute ? 420 : 840;
		routeHeight = mobilePortraitRoute ? 440 : 480;
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
			const colors = row.case === "usp" && ["ime-a", "ime-b", "ime-c"].includes(row.buildings[index].qgis_id)
				? { fill: visited ? "#e6ac63aa" : "#ffcf8b55", stroke: "#ffdcaa" }
				: regionColors(row.order.indexOf(index), row.polygons, visited);
			const regionTitle = row.case === "usp" ? row.buildings[index].label
				: row.case === "sp-bairros" ? `Subprefeitura ${row.region_names[index]}`
					: row.case === "br-estados" ? `${row.region_codes[index]} · ${row.region_names[index]}`
					: `Region ${index + 1}`;
			return `<polygon class="region" data-region="${index}" points="${coordinates(polygon)}" style="fill:${colors.fill};stroke:${colors.stroke}"><title>${escapeHTML(regionTitle)} · visit ${row.order.indexOf(index) + 1}</title></polygon>`;
		}).join("")}
			${decomposition ? decompositionLinesMarkup(row.visualization.decomposition, row.geometry.polygons, projected.project) : ""}
			<polyline class="route-ghost" points="${coordinates(projected.path)}"/><polyline id="animated-route" class="route-line"/>
			${labels ? projected.polygons.map((polygon, index) => {
				const center = polygonCentroid(polygon);
				return `<text class="region-label" x="${center[0]}" y="${center[1]}" text-anchor="middle" dominant-baseline="central">${row.order.indexOf(index) + 1}</text>`;
			}).join("") : ""}
			${row.cycle ? "" : `<circle class="endpoint" cx="${start[0]}" cy="${start[1]}" r="7"/><text class="endpoint-label" x="${start[0] + 14}" y="${start[1] + 5}">${row.depot ? "s = t" : "s"}</text>`}
			${row.depot || row.cycle ? "" : `<circle class="endpoint target" cx="${target[0]}" cy="${target[1]}" r="7"/><text class="endpoint-label" x="${target[0] + 14}" y="${target[1] + 5}">t</text>`}<circle id="traveler" class="traveler" r="6"/>
			${enabled("show-contacts") ? row.visualization.contacts.map((contact, index) => {
				if (!contact) return "";
				const point = projected.project(contact.point);
				const regionName = row.case === "sp-bairros" ? `Subprefeitura ${row.region_names[index]}`
					: row.case === "br-estados" ? `${row.region_codes[index]} · ${row.region_names[index]}`
						: `original region ${index + 1}`;
				return `<circle class="visit-contact" data-region="${index}" cx="${point[0]}" cy="${point[1]}" r="4"><title>Visit ${row.order.indexOf(index) + 1} · ${escapeHTML(regionName)}</title></circle>`;
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
			? "Shortest closed tour that visits more than 50 regions of the USP campus and returns to IME."
			: row.case === "sp-bairros"
				? "Shortest closed tour that visits the 32 subprefectures of São Paulo and returns to Sé."
				: row.case === "br-estados"
					? "Shortest closed tour that crosses Brazil's 27 federative units and returns to Brasília."
					: `Case ${caseLabel(row.case)}: closed tour through ${row.polygons} regions.`;
		element("map-description").textContent = row.case === "usp"
			? "Closed tour starting and ending near IME; more than 50 USP outlines and the computed tour in orange."
			: row.case === "sp-bairros"
				? "Closed tour in orange; crossing a polygon counts as visiting the subprefecture. The outlines were generalized for this demonstration."
				: row.case === "br-estados"
					? "Closed tour in orange; crossing the main outline of a federative unit counts as visiting it. Other disconnected parts were left out and the borders were simplified for this demonstration."
					: `Computed tour in orange, visiting ${row.polygons} regions and returning to where it started.`;
		drawRoute();
	}

	const seenTabsKey = "tpp-siicusp34-en-seen-cases";
	let seenTabs = new Set();
	try { seenTabs = new Set(JSON.parse(localStorage.getItem(seenTabsKey) || "[]")); } catch { /* Storage can be unavailable. */ }
	function markTabSeen(key) {
		if (key !== "sp-bairros" && key !== "usp") return;
		seenTabs.add(key);
		try { localStorage.setItem(seenTabsKey, JSON.stringify([...seenTabs])); } catch { /* Private browsing can reject storage. */ }
	}
	function updateNewTabs() {
		document.querySelectorAll('.case-tabs .example[data-case="sp-bairros"], .case-tabs .example[data-case="usp"]').forEach((button) => {
			const unseen = !seenTabs.has(button.dataset.case);
			button.classList.toggle("is-new", unseen);
			const label = button.dataset.case === "sp-bairros" ? "São Paulo" : "USP campus";
			button.setAttribute("aria-label", unseen ? `${label} (not seen yet)` : label);
		});
	}
	function selectCase(index, updateURL = true) {
		const selected = index === "usp" ? uspDemo
			: index === "sp-bairros" ? spBairrosDemo
				: index === "br-estados" ? brEstadosDemo
					: data.rows.find((item) => item.case === index);
		if (!selected) return false;
		markTabSeen(index);
		updateNewTabs();
		stop();
		row = selected;
		fraction = 1;
		zoom = 1;
		pan = [0, 0];
		element("show-labels").setAttribute("aria-pressed", String(row.polygons <= 15));
		element("case-select").value = Number.isInteger(row.case) ? row.case : "";
		element("case-picker-value").textContent = "Explore";
		const featured = ["usp", "sp-bairros", "br-estados"].includes(row.case);
		element("case-picker-active").hidden = featured;
		element("case-picker-active").textContent = featured ? "" : `· ${caseLabel(row.case)}`;
		element("case-picker-button").setAttribute("aria-pressed", String(!featured));
		element("case-picker-button").setAttribute("aria-label", featured
			? "Explore the 558 cases"
			: `Explore the 558 cases; case ${caseLabel(row.case)} selected`);
		document.querySelectorAll("#rota-usp .example").forEach((button) => {
			const active = button.dataset.case === String(row.case);
			button.classList.toggle("active", active);
			button.setAttribute("aria-pressed", String(active));
			if (active) button.setAttribute("aria-current", "true");
			else button.removeAttribute("aria-current");
		});
		const isUsp = row.case === "usp";
		const isSpBairros = row.case === "sp-bairros";
		const isBrEstados = row.case === "br-estados";
			element("intro-story").textContent = isUsp
			? "What route should a drone fly to visit more than 50 regions of the University of São Paulo campus and return to the entrance of IME? Below is an optimal tour for this geometric model."
			: isSpBairros
				? "What closed route should a drone fly to visit the 32 subprefectures of São Paulo, starting and ending at Sé? Crossing a region counts as visiting it. Below is the optimal tour for these generalized outlines."
				: isBrEstados
					? "What route should a drone take to leave Brasília, cross the main territory of each of Brazil's 27 federative units and return to the capital, flying the shortest distance? Below is the optimal tour for this simplified version of the problem."
					: `In case ${caseLabel(row.case)}, from the benchmark of Fekete et al., a closed tour visits ${row.polygons} regions and returns to where it started; no point is fixed. Below is the tour our solver found, proved to be within 0.1% of the shortest possible.`;
		element("route-endpoint-legend").textContent = isUsp
			? "s = t · entrance of IME"
			: isSpBairros ? "s = t · centre of the Sé subprefecture"
				: isBrEstados ? "s = t · Brasília" : "Closed tour · no fixed point";
		element("usp-current-stop").hidden = !isUsp;
		element("show-decomposition").setAttribute("aria-pressed", "false");
		element("show-decomposition").disabled = !row.visualization.decomposition?.some((pieces) => pieces.length > 1);
		element("speed-value").title = `At 1×, the tour of this case takes ${number(playbackDuration(row) / 1000, 1)} s to play.`;
		element("case-time").textContent = formatDuration(row.seconds);
		element("case-time-detail").textContent = !row.cycle ? "· one thread"
			: Number.isFinite(row.fekete_seconds) ? `· one thread · Fekete et al., one thread: ${formatDuration(row.fekete_seconds)}`
				: "· one thread · Fekete et al., one thread: not solved in 900 s";
		element("case-dialog-title").textContent = typeof row.case === "number" ? `Case ${caseLabel(row.case)} · ${row.polygons} regions` : row.title;
		if (updateURL && window.location.protocol !== "file:") {
			const url = new URL(window.location.href);
			url.searchParams.set("case", row.case);
			window.history.replaceState(null, "", url);
		}
		draw();
		return true;
	}

	function renderTable() {
		const rows = sortRows(data.rows, sorting.result.key, sorting.result.descending);
		const visible = showAllResults ? rows : rows.slice(0, 8);
		element("result-count").textContent = `${rows.length} instances.`;
		const toggle = rows.length > 8 ? `<tr class="result-toggle-row"><td colspan="4"><button type="button" data-toggle-results aria-expanded="${showAllResults}">${showAllResults ? "Show only the first 8 ↑" : `See all ${rows.length} results ↓`}</button></td></tr>` : "";
		element("result-rows").innerHTML = visible.map((item) => `<tr><th scope="row"><span class="mobile-case-label">Case </span>${caseLabel(item.case)}</th><td data-label="Regions">${item.polygons}</td><td data-label="Time">${formatDuration(item.seconds)}</td><td class="result-action"><button type="button" data-open-case="${item.case}" aria-label="See the tour of case ${item.case + 1}"><span class="picker-action-full">See tour →</span><span class="picker-action-short">Open →</span></button></td></tr>`).join("") + toggle;
	}

	document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach((control) => { control.disabled = false; });
	populateCaseSelect();
	updateResultSummary();
	function showMap(focusPlayback = false) {
		if (element("rota-usp").closest("dialog")) return;
		document.querySelector(".drawing-panel").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "start" });
		if (focusPlayback) element("play-route").focus({ preventScroll: true });
	}
	document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => {
		const rawKey = button.dataset.case;
		const key = ["usp", "sp-bairros", "br-estados"].includes(rawKey) ? rawKey : Number(rawKey);
		if (key === row.case) return;
		selectCase(key);
		showMap();
	}));
	element("case-select").addEventListener("change", (event) => { selectCase(Number(event.target.value)); showMap(); });
	element("show-labels").addEventListener("click", () => { element("show-labels").setAttribute("aria-pressed", String(!enabled("show-labels"))); draw(); });
	element("show-hulls").addEventListener("click", () => { element("show-hulls").setAttribute("aria-pressed", String(!enabled("show-hulls"))); draw(); });
	element("show-contacts").addEventListener("click", () => { element("show-contacts").setAttribute("aria-pressed", String(!enabled("show-contacts"))); draw(); });
	element("show-decomposition").addEventListener("click", () => { element("show-decomposition").setAttribute("aria-pressed", String(!enabled("show-decomposition"))); draw(); });
	element("zoom-slider").addEventListener("input", (event) => {
		const requested = Number(event.currentTarget.value) / 10;
		zoomAt(requested < zoom && requested <= 1.15 ? 1 : requested, [0, 0]);
	});
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
	// Capture at the SVG root so child polygon targets cannot swallow pinch events.
	map.addEventListener("wheel", (event) => {
		const trackpadPinch = event.ctrlKey;
		if (trackpadGesture || (!trackpadPinch && zoom <= 1) || event.deltaY === 0) return;
		const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 480 : 1;
		const requested = zoom * Math.exp(-event.deltaY * units * (trackpadPinch ? .01 : .002));
		event.preventDefault();
		if (requested <= 1) {
			zoom = 1;
			pan = [0, 0];
			draw();
			return;
		}
		zoomAt(requested, localPoint(event));
	}, { passive: false, capture: true });
	map.addEventListener("gesturestart", (event) => { event.preventDefault(); trackpadGesture = { zoom, scale: 1 }; }, { passive: false, capture: true });
	map.addEventListener("gesturechange", (event) => {
		event.preventDefault();
		if (!trackpadGesture || !Number.isFinite(event.scale)) return;
		zoomAt(trackpadGesture.zoom * event.scale, localPoint(event));
	}, { passive: false, capture: true });
	map.addEventListener("gestureend", (event) => { event.preventDefault(); trackpadGesture = null; }, { passive: false, capture: true });
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
	element("route-progress").addEventListener("input", (event) => { stop(); fraction = Number(event.target.value) / 1000; drawRoute(); });
	element("play-route").addEventListener("click", () => {
		if (playing) { stop(); return; }
		if (window.matchMedia("(max-width: 560px)").matches && element("route-map").getBoundingClientRect().top > 0) showMap();
		if (reducedMotion.matches) { fraction = 1; drawRoute(); return; }
		if (fraction >= 1) fraction = 0;
		playing = true;
		element("play-route").textContent = "Ⅱ Pause the tour";
		element("play-route").setAttribute("aria-pressed", "true");
		let previous = performance.now();
		const tick = (now) => {
			fraction = Math.min(1, fraction + (now - previous) * speeds[speedIndex] / playbackDuration(row));
			previous = now;
			drawRoute();
			if (fraction >= 1) stop();
			else frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
	});
	document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
	reducedMotion.addEventListener("change", stop);
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
		openCaseOnMap(Number(button.dataset.openCase), button);
	});
	function openCaseOnMap(caseIndex, control) {
		const dialog = element("case-dialog");
		if (!dialog.open) {
			// The explorer card itself moves into the overlay, so every control and the renderer keep working.
			const panel = element("rota-usp");
			const home = { parent: panel.parentNode, next: panel.nextSibling, previous: row.case };
			element("case-dialog-content").append(panel);
			dialog.addEventListener("close", () => {
				stop();
				home.parent.insertBefore(panel, home.next);
				selectCase(home.previous, false);
				control?.focus({ preventScroll: true });
			}, { once: true });
			showModalWithTransition(dialog);
		}
		selectCase(caseIndex, false);
		element("play-route").focus({ preventScroll: true });
	}
	element("close-case").addEventListener("click", () => element("case-dialog").close());
	closeOnBackdropClick(element("case-dialog"));
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
		element("picker-count").textContent = `${rows.length} cases available`;
		element("picker-options").innerHTML = rows.length ? rows.map((item) => `<tr><th scope="row"><span class="mobile-case-label">Case </span>${caseLabel(item.case)}</th><td data-label="Regions">${item.polygons}</td><td data-label="Time">${formatDuration(item.seconds)}</td><td class="result-action"><button type="button" data-pick-case="${item.case}" aria-label="See the tour of case ${item.case + 1}" aria-current="${item.case === row.case}"><span class="picker-action-full">See tour →</span><span class="picker-action-short">Open →</span></button></td></tr>`).join("") : '<tr><td colspan="4">No case found. Try another number.</td></tr>';
		document.querySelectorAll("[data-picker-column]").forEach((header) => {
			const active = header.dataset.pickerColumn === sorting.picker.key;
			header.setAttribute("aria-sort", active ? (sorting.picker.descending ? "descending" : "ascending") : "none");
			header.querySelector(".sort-arrow").textContent = active ? (sorting.picker.descending ? " ↓" : " ↑") : "";
		});
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
	closeOnBackdropClick(picker);
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
	document.querySelectorAll("[data-picker-sort]").forEach((button) => button.addEventListener("click", () => {
		const key = button.dataset.pickerSort;
		sorting.picker.descending = sorting.picker.key === key ? !sorting.picker.descending : false;
		sorting.picker.key = key;
		renderPicker();
	}));
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
	const requested = new URLSearchParams(window.location.search).get("case");
	const requestedCase = ["usp", "sp-bairros"].includes(requested) ? requested
		: requested !== null && /^\d+$/.test(requested) ? Number(requested) : defaultCase;
	selectCase(requestedCase, false) || selectCase(defaultCase, false);
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
	const letters = challenge.geometry.polygons.map((_, index) => String.fromCharCode(65 + index));
	const best = challenge.solutions[challenge.reference];
	let order = [], compared = false;
	const map = element("challenge-map");
	map.setAttribute("role", "group");
	function render() {
		const chosen = order.length === challenge.geometry.polygons.length
			? solveChallengeRoute(challenge.geometry.start, challenge.geometry.target, order.map((index) => challenge.geometry.polygons[index]))
			: null;
		if (chosen) chosen.length *= challenge.meters_per_display_unit;
		map.innerHTML = challenge.geometry.polygons.map((polygon, index) => {
			const [x, y] = polygonCentroid(polygon);
			const rank = order.indexOf(index);
			return `<g role="button" tabindex="0" data-challenge-region="${index}" aria-label="Region ${letters[index]}${rank >= 0 ? `, choice ${rank + 1}` : ""}" aria-pressed="${rank >= 0}"><polygon points="${coordinates(polygon)}" class="challenge-region ${rank >= 0 ? "chosen" : ""}"/><text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${letters[index]}${rank >= 0 ? ` · ${rank + 1}` : ""}</text></g>`;
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line challenge-solution-route" points="${coordinates(chosen.path)}"/>` : orderSketch("order", challenge.geometry, order.map((index) => challenge.geometry.polygons[index]))) + `<circle cx="${challenge.geometry.start[0]}" cy="${challenge.geometry.start[1]}" r="5" fill="white"/><text x="${challenge.geometry.start[0] - 8}" y="${challenge.geometry.start[1] + 23}">s</text><circle cx="${challenge.geometry.target[0]}" cy="${challenge.geometry.target[1]}" r="5" fill="#ffad66"/><text x="${challenge.geometry.target[0] - 8}" y="${challenge.geometry.target[1] + 23}">t</text>`;
		element("challenge-guidance").textContent = order.length === letters.length ? "Sequence complete. Now compare the paths." : `Choose ${letters.length - order.length} more ${letters.length - order.length === 1 ? "region" : "regions"} to compare.`;
		element("challenge-guidance").hidden = compared;
		element("challenge-compare").disabled = order.length !== letters.length || compared;
		element("challenge-compare").hidden = compared;
		element("challenge-undo").disabled = !order.length;
		element("challenge-reset").disabled = !order.length;
		element("challenge-feedback").innerHTML = compared ? `<strong>${challengeVerdict(chosen.length, best.length, "orders")}</strong>${challengeComparison("Your choice", chosen.length, `Best of ${challenge.solutions.length.toLocaleString("en-US")}`, best.length, "Order", order.map((index) => letters[index]).join(" → "), best.order.map((index) => letters[index]).join(" → "))}` : "";
		if (compared) animateChallengeRoute(map);
	}
	function choose(event) {
		const control = event.target.closest("[data-challenge-region]");
		if (!control) return;
		const index = Number(control.dataset.challengeRegion);
		order = toggleOrderRegion(order, index);
		compared = false;
		element("challenge-next-one").hidden = true;
		render();
		if (event.type === "keydown" || event.detail === 0) {
			map.querySelector(`[data-challenge-region="${index}"]`).focus({ preventScroll: true });
		}
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	element("challenge-compare").addEventListener("click", () => { compared = true; render(); element("challenge-next-one").hidden = false; scrollToChallengeDrawing(map); });
	element("challenge-undo").addEventListener("click", () => { order.pop(); compared = false; element("challenge-next-one").hidden = true; render(); });
	element("challenge-reset").addEventListener("click", () => { order = []; compared = false; element("challenge-next-one").hidden = true; render(); map.querySelector("[data-challenge-region]").focus({ preventScroll: true }); });
	render();
}

function scrollToChallengeDrawing(map) {
	requestAnimationFrame(() => map.closest(".challenge-drawing")?.scrollIntoView({
		behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
		block: "start",
	}));
}

function initializePieceChallenge(combined = false) {
	const all = window.TPPChallengeData;
	// Case 49 of the corpus (index 48) is the formal-count example of the final challenge.
	const impactSeconds = formatDuration(window.TPPEventData?.rows.find((row) => row.case === 48)?.seconds);
	const data = combined ? all.combined_challenge : all.piece_challenge;
	const prefix = combined ? "combined" : "piece";
	const ui = (suffix) => element(`${prefix}-${suffix}`);
	let order = [];
	const best = data.solutions[data.reference];
	const letters = data.geometry.polygons.map((_, index) => String.fromCharCode(65 + index));
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
		if (selected) selected.length *= data.meters_per_display_unit;
		map.innerHTML = data.pieces.map((pieces, region) => {
			const rank = order.indexOf(region);
			const bounds = data.geometry.polygons[region].reduce((box, point) => ({
				minX: Math.min(box.minX, point[0]), maxX: Math.max(box.maxX, point[0]), minY: Math.min(box.minY, point[1]), maxY: Math.max(box.maxY, point[1]),
			}), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
			const regionName = combined && rank >= 0 ? `${letters[region]} · #${rank + 1}` : letters[region];
			const pieceMarkup = pieces.map((piece, index) => {
				const [x, y] = polygonCentroid(piece);
				return `<g role="button" tabindex="0" data-piece="${index}" data-piece-region="${region}" aria-label="Region ${letters[region]}, piece ${index + 1}${rank >= 0 ? `, visit ${rank + 1}` : ""}" aria-pressed="${choices[region] === index}"><polygon class="challenge-region ${choices[region] === index ? "chosen" : ""}" points="${coordinates(piece)}"/><text x="${x}" y="${y}" dominant-baseline="central" text-anchor="middle">${index + 1}</text></g>`;
			}).join("");
			const labelY = (combined && region === 3) || (!combined && region === 2) ? Math.min(248, bounds.maxY + 18) : Math.max(18, bounds.minY - 10);
			const labelX = combined && region === 0 ? Math.min(408, bounds.maxX + 14) : (bounds.minX + bounds.maxX) / 2;
			return `${pieceMarkup}<text class="challenge-region-name" x="${labelX}" y="${labelY}" text-anchor="${combined && region === 0 ? "start" : "middle"}">${regionName}</text>`;
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line challenge-solution-route" points="${coordinates(selected.path)}"/>` : orderSketch(prefix, data.geometry, combined ? order.map(index => data.pieces[index][choices[index]]) : choices.slice(0, choices.includes(null) ? choices.indexOf(null) : choices.length).map((piece, region) => data.pieces[region][piece]))) + `<circle cx="${data.geometry.start[0]}" cy="${data.geometry.start[1]}" r="5" fill="white"/><text x="${data.geometry.start[0] - 6}" y="${data.geometry.start[1] + 22}">s</text><circle cx="${data.geometry.target[0]}" cy="${data.geometry.target[1]}" r="5" fill="#ffad66"/><text x="${combined ? data.geometry.target[0] - 6 : data.geometry.target[0] + 8}" y="${combined ? data.geometry.target[1] + 24 : data.geometry.target[1] - 18}">t</text>`;
		const remaining = choices.filter((piece) => piece === null).length;
		ui("guidance").textContent = remaining ? `${remaining} ${remaining === 1 ? "region" : "regions"} left to choose.` : "Choice complete. Now compare the paths.";
		ui("guidance").hidden = compared;
		ui("compare").disabled = choices.includes(null) || compared;
		ui("compare").hidden = compared;
		ui("reset").disabled = choices.every((piece) => piece === null);
		const chosenSequence = (combined ? order : letters.map((_, index) => index)).map(region => `${letters[region]}${choices[region] + 1}`).join(" → ");
		const bestSequence = (combined ? best.order : letters.map((_, index) => index)).map(region => `${letters[region]}${best.choices[region] + 1}`).join(" → ");
		const impact = combined ? `<aside class="challenge-impact"><strong>Found it hard?</strong> In case 49, the 60! orders and the choices of convex pieces give about 1.6 × 10<sup>117</sup> formal combinations. Our solver closed that case in ${impactSeconds} with one thread. Imagine <em>every atom on Earth</em> as a computer <a href="#ref-atoms">[7]</a>, testing one combination per nanosecond: listing them all would still take more than 10<sup>49</sup> years. That is more than 10<sup>39</sup> times the roughly 5 billion years until the Sun becomes a red giant and possibly swallows the Earth <a href="#ref-sun">[6]</a>. The solver did not list these combinations; geometric bounds and pruning cut the search down.</aside>` : "";
		ui("feedback").innerHTML = compared ? `<strong>${challengeVerdict(selected.length, best.length, "combinations")}</strong>${challengeComparison("Your choice", selected.length, `Best of ${data.solutions.length.toLocaleString("en-US")}`, best.length, combined ? "Solution" : "Pieces", chosenSequence, bestSequence)}${impact}` : "";
		if (compared) animateChallengeRoute(map);
	}
	function choose(event) {
		const control = event.target.closest("[data-piece]");
		if (!control) return;
		const region = Number(control.dataset.pieceRegion), piece = Number(control.dataset.piece);
		if (combined) {
			if (choices[region] === piece) { choices[region] = null; order = order.filter(index => index !== region); }
			else { if (choices[region] === null) order.push(region); choices[region] = piece; }
		} else choices[region] = toggleChoice(choices[region], piece);
		compared = false;
		if (combined) element("finish-challenge").hidden = true;
		else element("challenge-next-two").hidden = true;
		render();
		if (event.type === "keydown" || event.detail === 0) map.querySelector(`[data-piece-region="${region}"][data-piece="${piece}"]`).focus({ preventScroll: true });
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	ui("compare").addEventListener("click", () => {
		compared = true;
		render();
		if (combined) element("finish-challenge").hidden = false;
		else element("challenge-next-two").hidden = false;
		scrollToChallengeDrawing(map);
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
	const titles = ["Choose the visiting order", "Choose one piece of each region", "Choose the order and the pieces"];
	let current = 0;
	let opener = element("open-challenge");
	let historyEntry = false;
	let sectionNavigation = null;
	const previousSectionButton = element("challenge-previous-section");
	const nextSectionButton = element("challenge-next-section");
	function show(index) {
		current = Math.max(0, Math.min(steps.length - 1, index));
		steps.forEach((step, position) => { step.hidden = position !== current; });
		element("challenge-progress").textContent = `CHALLENGE ${current + 1} OF ${steps.length}`;
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
		desafio: ["1", "Try it yourself!", "Choose the visiting order and compare with the solver."],
		metodo: ["2", "How the algorithm works", "Initial route, branching, bounds and pruning."],
		historia: ["3", "Prior work", "A timeline from the TPP to this self-contained solver."],
		resultados: ["4", "What we improved", "Compare this solver with the state of the art."],
		contato: ["5", "Talk to the author", "Comments, questions or a conversation about the research."],
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
			toggle.innerHTML = `<span><b>${number}</b><strong>${title}</strong><small>${subtitle}</small></span><em data-guide-status="${id}">Not visited yet</em>`;
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
	const storageKey = "tpp-siicusp34-en-visited-sections";
	let visited = new Set();
	try {
		visited = new Set(JSON.parse(localStorage.getItem(storageKey) || "[]").filter((id) => nodes.has(id)));
	} catch {
		visited = new Set();
	}
	let active = null;
	const statusText = (id) => visited.has(id) ? "Visited" : "Not visited yet";
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
		if (count) count.textContent = `${visited.size}/${nodes.size} steps visited`;
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
		eyebrow.textContent = `STEP ${copy[id][0]} OF ${ids.length}`;
		title.textContent = copy[id][1];
		subtitle.textContent = copy[id][2];
		update(id);
		updateNavigation();
		if (!dialog.open) showModalWithTransition(dialog);
		requestAnimationFrame(resetDialogScroll);
		closeButton.focus({ preventScroll: true });
	}
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
		eyebrow.textContent = `STEP ${copy[nextId][0]} OF ${ids.length}`;
		title.textContent = copy[nextId][1];
		subtitle.textContent = copy[nextId][2];
		update(previous.id);
		update(nextId);
		updateNavigation();
		if (window.location.hash !== `#${nextId}`) window.history.replaceState(window.history.state, "", `#${nextId}`);
		requestAnimationFrame(resetDialogScroll);
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
	if (count) count.textContent = `${visited.size}/${nodes.size} steps visited`;
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
	element("event-error").textContent = "The interactive controls could not start. Reload the page to try again.";
	console.error(error);
});
