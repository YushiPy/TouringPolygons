import assert from "node:assert/strict";
import test from "node:test";
import { convexHull, pathPolygonContacts, pathPrefix, playbackDuration, projectedCase, regionColors } from "../static/path-geometry.js";

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

test("subtle region palettes vary by visit rank and by reached state", () => {
	assert.notDeepEqual(regionColors(0, 40, false), regionColors(39, 40, false));
	assert.notDeepEqual(regionColors(0, 40, false), regionColors(0, 40, true));
	assert.ok(!regionColors(0, 1, true).fill.includes("NaN"));
});

test("playback duration scales with region count and remains bounded", async () => {
	assert.equal(playbackDuration(4), 1860);
	assert.equal(playbackDuration(40), 2400);
	assert.equal(playbackDuration(60), 2700);
	assert.equal(playbackDuration(1000), 3000);
	assert.ok(playbackDuration(10) < playbackDuration(20));
});
