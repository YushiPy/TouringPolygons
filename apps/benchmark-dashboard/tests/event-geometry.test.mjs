import assert from "node:assert/strict";
import test from "node:test";
import { convexHull, filterRows, gapRatio, pathPrefix, projectedCase } from "../static/event-geometry.js";

test("playback follows arc length and clamps progress", () => {
	const path = [[0, 0], [3, 0], [3, 4]];
	assert.deepEqual(pathPrefix(path, .5), [[0, 0], [3, 0], [3, .5]]);
	assert.deepEqual(pathPrefix(path, 2), path);
	assert.deepEqual(pathPrefix(path, -1).at(-1), path[0]);
});

test("playback tolerates repeated vertices and closed zero-length paths", () => {
	assert.deepEqual(pathPrefix([[1, 2], [1, 2]], .5), [[1, 2], [1, 2]]);
	assert.deepEqual(pathPrefix([[0, 0], [0, 0], [2, 0]], .5).at(-1), [1, 0]);
	assert.deepEqual(pathPrefix([], .5), []);
});

test("convex hull keeps extreme points without changing original geometry", () => {
	const polygon = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]];
	const original = polygon.map((point) => [...point]);
	assert.deepEqual(convexHull(polygon), [[0, 0], [2, 0], [2, 1], [1, 2], [0, 2]]);
	assert.deepEqual(polygon, original);
});

test("projection fits extreme coordinates and preserves shape", () => {
	const row = { geometry: { start: [1e9, 0], target: [1e9 + 100, 100], polygons: [[[1e9, 0], [1e9 + 100, 0], [1e9, 100]]] }, path: [[1e9, 0], [1e9 + 100, 100]] };
	const result = projectedCase(row);
	for (const [x, y] of [...result.path, ...result.polygons.flat()]) assert.ok(x >= 0 && x <= 840 && y >= 0 && y <= 480);
	assert.equal(result.path[1][0] - result.path[0][0], result.path[0][1] - result.path[1][1]);
});

test("filters distinguish time limits from closed gaps and allow zero-padded IDs", () => {
	const rows = [{ case: 2, termination: "optimal" }, { case: 55, termination: "time_limit" }];
	assert.deepEqual(filterRows(rows, "all", "02"), [rows[0]]);
	assert.deepEqual(filterRows(rows, "time_limit", ""), [rows[1]]);
	assert.deepEqual(filterRows(rows, "optimal", "55"), []);
	assert.equal(gapRatio({ lower_bound: 90, upper_bound: 100 }), .1);
	assert.equal(gapRatio({ lower_bound: 0, upper_bound: 0 }), 0);
});
