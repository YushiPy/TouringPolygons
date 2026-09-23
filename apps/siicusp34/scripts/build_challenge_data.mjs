#!/usr/bin/env node
// Rebuild the three small, synthetic event challenges with the event-local
// fixed-order convex solver. They are separate from the 558 research cases.
import { writeFile } from "node:fs/promises";
import { tppSolveConvex } from "../tpp-solver.js";

const output = new URL("../data/challenge-data.js", import.meta.url);
const rectangle = (x, y, width, height) => [[x, y], [x + width, y], [x + width, y + height], [x, y + height]];

function lRegion(x, y, width, height, arm, split, flipX = false, flipY = false) {
	const polygon = [[x, y], [x + width, y], [x + width, y + height], [x + width - arm, y + height], [x + width - arm, y + arm], [x, y + arm]];
	const pieces = [rectangle(x, y, split, arm), rectangle(x + split, y, width - split, arm), rectangle(x + width - arm, y + arm, arm, height - arm)];
	const transform = (points) => points.map(([px, py]) => [flipX ? 2 * x + width - px : px, flipY ? 2 * y + height - py : py]);
	return { polygon: transform(polygon), pieces: pieces.map(transform) };
}

const fixedRegions = [
	lRegion(55, 85, 90, 120, 18, 48),
	lRegion(170, 95, 80, 90, 18, 48, true, true),
	lRegion(285, 25, 90, 105, 22, 28),
];
const combinedRegions = [
	lRegion(60, 25, 70, 70, 24, 35, true),
	lRegion(220, 145, 80, 70, 16, 25),
	lRegion(60, 145, 70, 90, 16, 35, false, true),
	lRegion(270, 20, 70, 70, 16, 25, false, true),
];
const cases = {
	free: { geometry: { start: [25, 130], target: [395, 130], polygons: [rectangle(235, 150, 40, 40), rectangle(235, 60, 40, 40), rectangle(145, 180, 40, 40), rectangle(115, 30, 40, 40)] } },
	piece_challenge: { geometry: { start: [25, 150], target: [395, 170], polygons: fixedRegions.map((region) => region.polygon) }, pieces: fixedRegions.map((region) => region.pieces) },
	combined_challenge: { geometry: { start: [25, 170], target: [395, 150], polygons: combinedRegions.map((region) => region.polygon) }, pieces: combinedRegions.map((region) => region.pieces) },
};

function* permutations(items) {
	if (!items.length) { yield []; return; }
	for (const item of items) for (const rest of permutations(items.filter((other) => other !== item))) yield [item, ...rest];
}

function* combinations(count) {
	if (!count) { yield []; return; }
	for (let piece = 0; piece < 3; piece++) for (const rest of combinations(count - 1)) yield [piece, ...rest];
}

function route(geometry, polygons) {
	const path = tppSolveConvex(geometry.start, geometry.target, polygons, true).map(({ x, y }) => [x, y]);
	const length = path.slice(1).reduce((total, point, index) => total + Math.hypot(point[0] - path[index][0], point[1] - path[index][1]), 0);
	if (!Number.isFinite(length)) throw new Error("Non-finite route");
	return { path, length };
}

function bounds(polygon) {
	return [Math.min(...polygon.map(([x]) => x)), Math.min(...polygon.map(([, y]) => y)), Math.max(...polygon.map(([x]) => x)), Math.max(...polygon.map(([, y]) => y))];
}

function distance(left, right) {
	const a = Array.isArray(left[0]) ? bounds(left) : [left[0], left[1], left[0], left[1]];
	const b = Array.isArray(right[0]) ? bounds(right) : [right[0], right[1], right[0], right[1]];
	return Math.hypot(Math.max(a[0] - b[2], b[0] - a[2], 0), Math.max(a[1] - b[3], b[1] - a[3], 0));
}

// In these examples every selectable region/piece is an axis-aligned rectangle,
// so the bounding-box distance is the exact set-to-set Euclidean distance.
function greedy(data, mode) {
	const count = data.geometry.polygons.length;
	const order = [], choices = Array(count).fill(null), margins = [];
	let previous = data.geometry.start;
	for (let step = 0; step < count; step++) {
		const candidateRegions = mode === "piece" ? [step] : Array.from({ length: count }, (_, index) => index).filter((index) => !order.includes(index));
		const ranked = candidateRegions.flatMap((region) => (mode === "free" ? [null] : [0, 1, 2]).map((piece) => ({
			region, piece, polygon: piece === null ? data.geometry.polygons[region] : data.pieces[region][piece],
		}))).map((candidate) => ({ ...candidate, distance: distance(previous, candidate.polygon) }))
			.sort((a, b) => a.distance - b.distance || a.region - b.region || (a.piece ?? 0) - (b.piece ?? 0));
		margins.push(ranked.length > 1 ? ranked[1].distance - ranked[0].distance : null);
		const choice = ranked[0];
		order.push(choice.region);
		if (choice.piece !== null) choices[choice.region] = choice.piece;
		previous = choice.polygon;
	}
	return { order, ...(mode === "free" ? {} : { choices }), margins };
}

function build(data, mode) {
	const count = data.geometry.polygons.length;
	const orders = mode === "piece" ? [Array.from({ length: count }, (_, index) => index)] : [...permutations(Array.from({ length: count }, (_, index) => index))];
	const solutions = [];
	for (const order of orders) for (const choices of mode === "free" ? [[]] : combinations(count)) {
		const polygons = order.map((region) => mode === "free" ? data.geometry.polygons[region] : data.pieces[region][choices[region]]);
		solutions.push({ ...(mode === "piece" ? {} : { order }), ...(mode === "free" ? {} : { choices }), ...route(data.geometry, polygons) });
	}
	const reference = solutions.reduce((best, solution, index) => solution.length < solutions[best].length ? index : best, 0);
	const greedyChoice = greedy(data, mode);
	const greedySolution = solutions.find((solution) => (mode === "piece" || solution.order.every((region, index) => region === greedyChoice.order[index])) && (mode === "free" || solution.choices.every((piece, region) => piece === greedyChoice.choices[region])));
	const best = solutions[reference];
	const gap = greedySolution.length / best.length - 1;
	const near = solutions.filter((solution) => solution.length / best.length >= 1.015 && solution.length / best.length <= 1.12).length;
	if (gap < 0.10 || near < 3 || greedyChoice.margins.some((margin) => margin !== null && margin < 8)) throw new Error(`${mode}: weak local-choice example (gap ${(100 * gap).toFixed(1)}%, near ${near}, margins ${greedyChoice.margins})`);
	console.log(`${mode}: ${solutions.length} routes, best ${best.length.toFixed(2)}, greedy ${greedySolution.length.toFixed(2)} (+${(100 * gap).toFixed(1)}%), ${near} near-optimal`);
	return { ...data, reference, solutions, greedy: greedyChoice };
}

const free = build(cases.free, "free");
const piece = build(cases.piece_challenge, "piece");
const combined = build(cases.combined_challenge, "combined");
const payload = { ...free, provenance: "Synthetic teaching examples; all small alternatives recomputed with the event-local fixed-order convex solver. Independent of the 558-case research corpus.", piece_challenge: piece, combined_challenge: combined };
await writeFile(output, `window.TPPChallengeData = ${JSON.stringify(payload)};\n`);
