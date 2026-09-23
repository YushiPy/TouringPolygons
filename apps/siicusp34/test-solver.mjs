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

function assertRoute(path, expectedStart, expectedTarget, expectedLength, scale = 1) {
	assert.deepEqual([path[0].x, path[0].y], expectedStart);
	assert.deepEqual([path.at(-1).x, path.at(-1).y], expectedTarget);
	assert.ok(path.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)));
	assert.ok(Math.abs(length(path) * scale - expectedLength) < 1e-7, `expected ${expectedLength}, got ${length(path) * scale}`);
}

function visitsPolygon(path, polygon) {
	const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
	const onEdge = (point, a, b) => Math.abs(cross(a, b, point)) < 1e-6
		&& point.every((value, axis) => value >= Math.min(a[axis], b[axis]) - 1e-6 && value <= Math.max(a[axis], b[axis]) + 1e-6);
	const inside = (point) => {
		let parity = false;
		for (let index = 0; index < polygon.length; index++) {
			const a = polygon[index], b = polygon[(index + 1) % polygon.length];
			if (onEdge(point, a, b)) return true;
			if ((a[1] > point[1]) !== (b[1] > point[1])
				&& point[0] < a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])) parity = !parity;
		}
		return parity;
	};
	for (let index = 0; index < path.length - 1; index++) {
		const a = [path[index].x, path[index].y], b = [path[index + 1].x, path[index + 1].y];
		if (inside(a) || inside(b)) return true;
		for (let edge = 0; edge < polygon.length; edge++) {
			const c = polygon[edge], d = polygon[(edge + 1) % polygon.length];
			if ((cross(a, b, c) * cross(a, b, d) < 0 && cross(c, d, a) * cross(c, d, b) < 0)
				|| onEdge(c, a, b) || onEdge(d, a, b)) return true;
		}
	}
	return false;
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
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length, challenge.meters_per_display_unit);
		assert.ok(polygons.every((polygon) => visitsPolygon(path, polygon)), "free-order path misses a region");
	}
});

test("event-local solver reproduces every fixed-order piece choice", () => {
	const challenge = data.piece_challenge;
	for (const solution of challenge.solutions) {
		const polygons = solution.choices.map((piece, region) => challenge.pieces[region][piece]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length, challenge.meters_per_display_unit);
		assert.ok(polygons.every((polygon) => visitsPolygon(path, polygon)), "fixed-order path misses a selected piece");
	}
});

test("event-local solver reproduces every combined order and piece choice", () => {
	const challenge = data.combined_challenge;
	for (const solution of challenge.solutions) {
		const polygons = solution.order.map((region) => challenge.pieces[region][solution.choices[region]]);
		const path = solve(challenge.geometry.start, challenge.geometry.target, polygons);
		assertRoute(path, challenge.geometry.start, challenge.geometry.target, solution.length, challenge.meters_per_display_unit);
		assert.ok(polygons.every((polygon) => visitsPolygon(path, polygon)), "combined path misses a selected piece");
	}
});

test("the three challenges have a clear local-choice trap and close alternatives", () => {
	for (const [name, challenge] of [["free", data], ["piece", data.piece_challenge], ["combined", data.combined_challenge]]) {
		const best = challenge.solutions[challenge.reference];
		assert.ok(Math.abs(best.length - 100) < 1e-9, `${name}: best route should be 100 m`);
		assert.ok(challenge.solutions.every((solution) => solution.length >= best.length - 1e-7), `${name}: invalid reference`);
		const greedy = challenge.solutions.find((solution) => (name === "piece" || solution.order.every((region, index) => region === challenge.greedy.order[index]))
			&& (name === "free" || solution.choices.every((piece, region) => piece === challenge.greedy.choices[region])));
		assert.ok(greedy, `${name}: missing greedy route`);
		assert.ok(greedy.length / best.length >= 1.02, `${name}: local choice is too close to the best route`);
		assert.ok(challenge.greedy.margins.some((margin) => margin !== null && margin >= 3), `${name}: local choices are indistinguishable`);
		assert.ok(challenge.solutions.filter((solution) => solution.length / best.length >= 1.015 && solution.length / best.length <= 1.12).length >= 3, `${name}: no close alternatives`);
	}
});
