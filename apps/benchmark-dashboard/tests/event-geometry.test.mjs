import assert from "node:assert/strict";
import test from "node:test";
import { convexHull, endpointOffset, filterRows, gapRatio, pathPrefix, projectedCase, qualityLabel, regionColors } from "../static/event-geometry.js";

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

test("endpoint labels point away from the first and last nonzero rays", () => {
	const path = [[0, 0], [0, 0], [3, 0], [3, 4], [3, 4]];
	assert.deepEqual(endpointOffset(path, false, 10), [-10, 0]);
	assert.deepEqual(endpointOffset(path, true, 10), [3, 14]);
	assert.deepEqual(endpointOffset([[1, 1], [1, 1]], false, 10), [-9, 1]);
	assert.deepEqual(endpointOffset([[1, 1], [1, 1]], true, 10), [11, 1]);
});

test("quality is conservatively rounded and never declares an open gap optimal", () => {
	assert.equal(qualityLabel({ exact: false, lower_bound: 99.99999, upper_bound: 100 }), "Pelo menos 99,99% de qualidade numérica");
	assert.match(qualityLabel({ exact: true, lower_bound: 0, upper_bound: 0 }), /≈ 100%/);
	assert.match(qualityLabel({ exact: false, lower_bound: 90.126, upper_bound: 100 }), /90,12%/);
});

test("subtle region palettes vary by visit rank and by reached state", () => {
	assert.notDeepEqual(regionColors(0, 40, false), regionColors(39, 40, false));
	assert.notDeepEqual(regionColors(0, 40, false), regionColors(0, 40, true));
	assert.ok(!regionColors(0, 1, true).fill.includes("NaN"));
});

test("case sorting is numeric, stable and reversible without mutating evidence", async () => {
	const { sortRows } = await import("../static/event-geometry.js");
	const rows = [
		{ case: 2, polygons: 4, seconds: 10, upper_bound: 10, lower_bound: 9, exact: false },
		{ case: 1, polygons: 10, seconds: 2, upper_bound: 10, lower_bound: 10, exact: true },
		{ case: 3, polygons: 4, seconds: 3, upper_bound: 10, lower_bound: 8, exact: false },
	];
	assert.deepEqual(sortRows(rows, "polygons").map(r => r.case), [2, 3, 1]);
	assert.deepEqual(sortRows(rows, "seconds", true).map(r => r.case), [2, 3, 1]);
	assert.deepEqual(sortRows(rows, "gap").map(r => r.case), [1, 2, 3]);
	assert.deepEqual(sortRows(rows, "result").map(r => r.case), [1, 2, 3]);
	assert.deepEqual(rows.map(r => r.case), [2, 1, 3]);
});

test("playback duration scales with region count and remains bounded", async () => {
	const { playbackDuration } = await import("../static/event-geometry.js");
	assert.equal(playbackDuration(4), 7600);
	assert.equal(playbackDuration(40), 22000);
	assert.equal(playbackDuration(60), 30000);
	assert.equal(playbackDuration(1000), 30000);
	assert.ok(playbackDuration(10) < playbackDuration(20));
});
