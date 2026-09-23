#!/usr/bin/env node
// Rebuild the three author-drawn SIICUSP challenges independently of the corpus.
// Convex pieces come from the same C++ partition package used by the solver.
import { execFileSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { tppSolveConvex } from "../tpp-solver.js";

const output = new URL("../data/challenge-data.js", import.meta.url);
const source = JSON.parse(execFileSync("python3", [fileURLToPath(new URL("./export_challenge_geometry.py", import.meta.url))], { encoding: "utf8" }));
const [freeCase, pieceCase, combinedCase] = source.cases;

function* permutations(items) {
	if (!items.length) { yield []; return; }
	for (const item of items) for (const rest of permutations(items.filter((other) => other !== item))) yield [item, ...rest];
}

function* combinations(groups, index = 0, choice = []) {
	if (index === groups.length) { yield [...choice]; return; }
	for (let piece = 0; piece < groups[index].length; piece++) {
		choice.push(piece);
		yield* combinations(groups, index + 1, choice);
		choice.pop();
	}
}

function route(geometry, polygons) {
	const path = tppSolveConvex(geometry.start, geometry.target, polygons, true).map(({ x, y }) => [x, y]);
	const length = path.slice(1).reduce((total, point, index) => total + Math.hypot(point[0] - path[index][0], point[1] - path[index][1]), 0);
	if (!Number.isFinite(length)) throw new Error("Non-finite route");
	return { path, length };
}

const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
function onSegment(point, a, b) {
	return Math.abs(cross(a, b, point)) < 1e-9 && point.every((v, axis) => v >= Math.min(a[axis], b[axis]) - 1e-9 && v <= Math.max(a[axis], b[axis]) + 1e-9);
}
function inPolygon(point, polygon) {
	let inside = false;
	for (let index = 0; index < polygon.length; index++) {
		const a = polygon[index], b = polygon[(index + 1) % polygon.length];
		if (onSegment(point, a, b)) return true;
		if ((a[1] > point[1]) !== (b[1] > point[1]) && point[0] < a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])) inside = !inside;
	}
	return inside;
}
function segments(polygon) {
	return polygon.map((point, index) => [point, polygon[(index + 1) % polygon.length]]);
}
function pointSegmentDistance(point, a, b) {
	const dx = b[0] - a[0], dy = b[1] - a[1];
	const lengthSquared = dx * dx + dy * dy;
	const fraction = lengthSquared ? Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSquared)) : 0;
	return Math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy);
}
function segmentDistance(a, b, c, d) {
	const abC = cross(a, b, c), abD = cross(a, b, d), cdA = cross(c, d, a), cdB = cross(c, d, b);
	if (((abC > 0 && abD < 0) || (abC < 0 && abD > 0))
		&& ((cdA > 0 && cdB < 0) || (cdA < 0 && cdB > 0))) return 0;
	if (onSegment(a, c, d) || onSegment(b, c, d) || onSegment(c, a, b) || onSegment(d, a, b)) return 0;
	return Math.min(pointSegmentDistance(a, c, d), pointSegmentDistance(b, c, d), pointSegmentDistance(c, a, b), pointSegmentDistance(d, a, b));
}
function distance(left, right) {
	if (!Array.isArray(left[0])) return inPolygon(left, right) ? 0 : Math.min(...segments(right).map(([a, b]) => pointSegmentDistance(left, a, b)));
	if (left.some((point) => inPolygon(point, right)) || right.some((point) => inPolygon(point, left))) return 0;
	return Math.min(...segments(left).flatMap(([a, b]) => segments(right).map(([c, d]) => segmentDistance(a, b, c, d))));
}

function greedy(data, mode) {
	const count = data.geometry.polygons.length;
	const order = [], choices = Array(count).fill(null), margins = [];
	let previous = data.geometry.start;
	for (let step = 0; step < count; step++) {
		const regions = mode === "piece" ? [step] : Array.from({ length: count }, (_, index) => index).filter((index) => !order.includes(index));
		const ranked = regions.flatMap((region) => (mode === "free" ? [null] : data.pieces[region].map((_, index) => index)).map((piece) => ({
			region, piece, polygon: piece === null ? data.geometry.polygons[region] : data.pieces[region][piece],
		}))).map((candidate) => ({ ...candidate, distance: distance(previous, candidate.polygon) }))
			.sort((a, b) => a.distance - b.distance || a.region - b.region || (a.piece ?? 0) - (b.piece ?? 0));
		margins.push(ranked.length > 1 ? ranked[1].distance - ranked[0].distance : null);
		const chosen = ranked[0];
		order.push(chosen.region);
		if (chosen.piece !== null) choices[chosen.region] = chosen.piece;
		previous = chosen.polygon;
	}
	return { order, ...(mode === "free" ? {} : { choices }), margins };
}

function build(data, mode) {
	const count = data.geometry.polygons.length;
	const orders = mode === "piece" ? [Array.from({ length: count }, (_, index) => index)] : [...permutations(Array.from({ length: count }, (_, index) => index))];
	const pieceChoices = mode === "free" ? [[]] : [...combinations(data.pieces)];
	const solutions = [];
	for (const order of orders) for (const choices of pieceChoices) {
		const polygons = order.map((region) => mode === "free" ? data.geometry.polygons[region] : data.pieces[region][choices[region]]);
		solutions.push({ ...(mode === "piece" ? {} : { order }), ...(mode === "free" ? {} : { choices }), ...route(data.geometry, polygons) });
	}
	const reference = solutions.reduce((best, solution, index) => solution.length < solutions[best].length ? index : best, 0);
	const greedyChoice = greedy(data, mode);
	const greedySolution = solutions.find((solution) => (mode === "piece" || solution.order.every((region, index) => region === greedyChoice.order[index]))
		&& (mode === "free" || solution.choices.every((piece, region) => piece === greedyChoice.choices[region])));
	if (!greedySolution) throw new Error(`${mode}: no route for greedy choices`);
	const best = solutions[reference];
	const greedyGap = greedySolution.length / best.length - 1;
	const closeAlternatives = solutions.filter((solution) => solution.length / best.length >= 1.015 && solution.length / best.length <= 1.12).length;
	const metersPerDisplayUnit = 100 / best.length;
	for (const solution of solutions) solution.length *= metersPerDisplayUnit;
	for (let index = 0; index < solutions.length; index++) if (index !== reference) delete solutions[index].path;
	console.log(`${mode}: ${solutions.length} routes, best ${best.length.toFixed(3)} m, greedy +${(100 * greedyGap).toFixed(1)}%, ${closeAlternatives} close alternatives`);
	return { ...data, meters_per_display_unit: metersPerDisplayUnit, reference, solutions, greedy: greedyChoice, analysis: { greedy_gap_ratio: greedyGap, close_alternatives: closeAlternatives } };
}

const free = build(freeCase, "free");
const piece = build(pieceCase, "piece");
const combined = build(combinedCase, "combined");
const payload = {
	...free,
	provenance: "Author-drawn SIICUSP34 manual cases; convex pieces from the C++ optimal partition; all small alternatives solved by the event-local fixed-order solver. Separate from the 558-case corpus.",
	source_file: source.source,
	source_sha256: source.source_sha256,
	piece_challenge: piece,
	combined_challenge: combined,
};
await writeFile(output, `window.TPPChallengeData = ${JSON.stringify(payload)};\n`);
