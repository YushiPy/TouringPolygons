import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { tppSolveConvex } from "./tpp-solver.js";

const dataSource = await readFile(new URL("./data/challenge-data.js", import.meta.url), "utf8");
const data = JSON.parse(dataSource.replace(/^window\.TPPChallengeData\s*=\s*/, "").replace(/;\s*$/, ""));

function length(path) {
	return path.slice(1).reduce((total, point, index) => total + Math.hypot(point.x - path[index].x, point.y - path[index].y), 0);
}

function solve(start, target, polygons) {
	return tppSolveConvex(start, target, polygons, true);
}

function assertRoute(path, expectedStart, expectedTarget, expectedLength) {
	assert.deepEqual([path[0].x, path[0].y], expectedStart);
	assert.deepEqual([path.at(-1).x, path.at(-1).y], expectedTarget);
	assert.ok(path.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)));
	assert.ok(Math.abs(length(path) - expectedLength) < 1e-7, `expected ${expectedLength}, got ${length(path)}`);
}

test("event-local solver handles the empty and separated fixed-order cases", () => {
	const empty = solve([0, 0], [10, 10], []);
	assertRoute(empty, [0, 0], [10, 10], Math.hypot(10, 10));

	const separated = solve(
		[0, 0],
		[10, 10],
		[
			[[2, -1], [4, -1], [4, 1], [2, 1]],
			[[6, 4], [8, 4], [8, 6], [6, 6]],
		],
	);
	assertRoute(separated, [0, 0], [10, 10], length(separated));
});

test("event-local solver reproduces every convex challenge solution", () => {
	const challenge = data;
	for (const solution of challenge.solutions) {
		const polygons = solution.order.map((region) => challenge.geometry.polygons[region]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length);
	}
});

test("event-local solver reproduces every fixed-order piece choice", () => {
	const challenge = data.piece_challenge;
	for (const solution of challenge.solutions) {
		const polygons = solution.choices.map((piece, region) => challenge.pieces[region][piece]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length);
	}
});

test("event-local solver reproduces every combined order and piece choice", () => {
	const challenge = data.combined_challenge;
	for (const solution of challenge.solutions) {
		const polygons = solution.order.map((region) => challenge.pieces[region][solution.choices[region]]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length);
	}
});
