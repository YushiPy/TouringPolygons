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

function visitsRectangle(path, polygon) {
	const bounds = [0, 1].map((axis) => [Math.min(...polygon.map((point) => point[axis])) - 1e-7, Math.max(...polygon.map((point) => point[axis])) + 1e-7]);
	return path.slice(1).some((end, index) => {
		const start = path[index];
		let entry = 0, exit = 1;
		for (const [axis, key] of ["x", "y"].entries()) {
			const delta = end[key] - start[key];
			if (Math.abs(delta) < 1e-12) {
				if (start[key] < bounds[axis][0] || start[key] > bounds[axis][1]) return false;
			} else {
				const lower = (bounds[axis][0] - start[key]) / delta;
				const upper = (bounds[axis][1] - start[key]) / delta;
				entry = Math.max(entry, Math.min(lower, upper));
				exit = Math.min(exit, Math.max(lower, upper));
			}
		}
		return entry <= exit + 1e-9;
	});
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
		assert.ok(polygons.every((polygon) => visitsRectangle(path, polygon)), "free-order path misses a region");
	}
});

test("event-local solver reproduces every fixed-order piece choice", () => {
	const challenge = data.piece_challenge;
	for (const solution of challenge.solutions) {
		const polygons = solution.choices.map((piece, region) => challenge.pieces[region][piece]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length);
		assert.ok(polygons.every((polygon) => visitsRectangle(path, polygon)), "fixed-order path misses a selected piece");
	}
});

test("event-local solver reproduces every combined order and piece choice", () => {
	const challenge = data.combined_challenge;
	for (const solution of challenge.solutions) {
		const polygons = solution.order.map((region) => challenge.pieces[region][solution.choices[region]]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length);
		assert.ok(polygons.every((polygon) => visitsRectangle(path, polygon)), "combined path misses a selected piece");
	}
});

test("the three challenges have a clear local-choice trap and close alternatives", () => {
	for (const [name, challenge] of [["free", data], ["piece", data.piece_challenge], ["combined", data.combined_challenge]]) {
		const best = challenge.solutions[challenge.reference];
		assert.ok(challenge.solutions.every((solution) => solution.length >= best.length - 1e-7), `${name}: invalid reference`);
		const greedy = challenge.solutions.find((solution) => (name === "piece" || solution.order.every((region, index) => region === challenge.greedy.order[index]))
			&& (name === "free" || solution.choices.every((piece, region) => piece === challenge.greedy.choices[region])));
		assert.ok(greedy, `${name}: missing greedy route`);
		assert.ok(greedy.length / best.length >= 1.10, `${name}: local choice is too close to the best route`);
		assert.ok(challenge.greedy.margins.every((margin) => margin === null || margin >= 8), `${name}: local choice depends on a tiny distance difference`);
		assert.ok(challenge.solutions.filter((solution) => solution.length / best.length >= 1.015 && solution.length / best.length <= 1.12).length >= 3, `${name}: no close alternatives`);
	}
});
