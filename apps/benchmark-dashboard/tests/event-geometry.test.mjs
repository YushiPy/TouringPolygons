import assert from "node:assert/strict";
import test from "node:test";
import { convexHull, endpointOffset, filterRows, pathPolygonContacts, pathPrefix, projectedCase, regionColors } from "../static/event-geometry.js";

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

test("contacts are based on polygon intersections rather than matching path indices", () => {
	const path = [[-2, 0], [8, 0]];
	const polygons = [
		[[-1, -1], [1, -1], [1, 1], [-1, 1]],
		[[3, -1], [4, -1], [4, 1], [3, 1]],
		[[6, -1], [7, -1], [7, 1], [6, 1]],
	];
	const contacts = pathPolygonContacts(path, polygons);
	assert.deepEqual(contacts.map((contact) => contact.point), [[-1, 0], [3, 0], [6, 0]]);
	assert.deepEqual(contacts.map((contact) => contact.fraction), [.1, .5, .8]);
});

test("contacts include paths starting inside regions and collinear boundary visits", () => {
	const polygons = [
		[[-1, -1], [1, -1], [1, 1], [-1, 1]],
		[[2, 0], [4, 0], [4, 2], [2, 2]],
	];
	const contacts = pathPolygonContacts([[0, 0], [3, 0], [5, 0]], polygons);
	assert.deepEqual(contacts[0], { point: [0, 0], fraction: 0 });
	assert.deepEqual(contacts[1], { point: [2, 0], fraction: .4 });
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

test("filters distinguish time limits and allow zero-padded IDs", () => {
	const rows = [{ case: 2, termination: "optimal" }, { case: 55, termination: "time_limit" }];
	assert.deepEqual(filterRows(rows, "all", "02"), [rows[0]]);
	assert.deepEqual(filterRows(rows, "time_limit", ""), [rows[1]]);
	assert.deepEqual(filterRows(rows, "optimal", "55"), []);
});

test("endpoint labels point away from the first and last nonzero rays", () => {
	const path = [[0, 0], [0, 0], [3, 0], [3, 4], [3, 4]];
	assert.deepEqual(endpointOffset(path, false, 10), [-10, 0]);
	assert.deepEqual(endpointOffset(path, true, 10), [3, 14]);
	assert.deepEqual(endpointOffset([[1, 1], [1, 1]], false, 10), [-9, 1]);
	assert.deepEqual(endpointOffset([[1, 1], [1, 1]], true, 10), [11, 1]);
});

test("subtle region palettes vary by visit rank and by reached state", () => {
	assert.notDeepEqual(regionColors(0, 40, false), regionColors(39, 40, false));
	assert.notDeepEqual(regionColors(0, 40, false), regionColors(0, 40, true));
	assert.ok(!regionColors(0, 1, true).fill.includes("NaN"));
});

test("case sorting is numeric, stable and reversible without mutating evidence", async () => {
	const { sortRows } = await import("../static/event-geometry.js");
	const rows = [
		{ case: 2, polygons: 4, seconds: 10, length: 10, exact: false },
		{ case: 1, polygons: 10, seconds: 2, length: 10, exact: true },
		{ case: 3, polygons: 4, seconds: 3, length: 8, exact: false },
	];
	assert.deepEqual(sortRows(rows, "polygons").map(r => r.case), [2, 3, 1]);
	assert.deepEqual(sortRows(rows, "seconds", true).map(r => r.case), [2, 3, 1]);
	assert.deepEqual(sortRows(rows, "result").map(r => r.case), [1, 2, 3]);
	assert.deepEqual(rows.map(r => r.case), [2, 1, 3]);
});

test("playback duration scales with region count and remains bounded", async () => {
	const { playbackDuration } = await import("../static/event-geometry.js");
	assert.equal(playbackDuration(4), 1860);
	assert.equal(playbackDuration(40), 2400);
	assert.equal(playbackDuration(60), 2700);
	assert.equal(playbackDuration(1000), 3000);
	assert.ok(playbackDuration(10) < playbackDuration(20));
});

test("result grouping stays primary while numeric sort orders each group", async () => {
	const { sortGroupedRows } = await import("../static/event-geometry.js");
	const rows = [
		{ case: 0, exact: false, seconds: 1 },
		{ case: 1, exact: true, seconds: 9 },
		{ case: 2, exact: false, seconds: 4 },
		{ case: 3, exact: true, seconds: 2 },
	];
	assert.deepEqual(sortGroupedRows(rows, "seconds", false, false).map(row => row.case), [3, 1, 0, 2]);
	assert.deepEqual(sortGroupedRows(rows, "seconds", true, true).map(row => row.case), [2, 0, 1, 3]);
	assert.deepEqual(sortGroupedRows(rows, "seconds", false, null).map(row => row.case), [0, 3, 2, 1]);
	assert.deepEqual(rows.map(row => row.case), [0, 1, 2, 3]);
});

test("deselecting any region preserves the remaining order and allows reselecting", async () => {
	const { toggleOrderRegion } = await import("../static/event-geometry.js");
	const order = [2, 0, 3, 1];
	assert.deepEqual(toggleOrderRegion(order, 0), [2, 3, 1]);
	assert.deepEqual(toggleOrderRegion(toggleOrderRegion(order, 0), 0), [2, 3, 1, 0]);
	assert.deepEqual(toggleOrderRegion([2], 2), []);
	assert.deepEqual(order, [2, 0, 3, 1]);
});

test("tapping the selected convex piece deselects it", async () => {
	const { toggleChoice } = await import("../static/event-geometry.js");
	assert.equal(toggleChoice(null, 2), 2);
	assert.equal(toggleChoice(2, 2), null);
	assert.equal(toggleChoice(2, 1), 1);
});
