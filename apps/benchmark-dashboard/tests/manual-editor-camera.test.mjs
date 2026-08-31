import assert from "node:assert/strict";
import test from "node:test";
import {
	CAMERA_STORAGE_KEY,
	caseBounds,
	canvasToWorld,
	worldToCanvas,
	zoomLimits,
} from "../static/manual-editor-camera.js";

test("camera helpers calculate bounds and reversible coordinates", () => {
	const data = { start: [0, 1], target: [4, -2], polygons: [[[1, 2], [3, 0]]] };
	assert.deepEqual(caseBounds(data), { minX: 0, minY: -2, maxX: 4, maxY: 2 });
	const camera = { scale: 10, offsetX: 20, offsetY: 30 };
	assert.deepEqual(canvasToWorld(30, 20, camera), [1, 1]);
	assert.deepEqual(worldToCanvas([1, 1], camera), { x: 30, y: 20 });
	assert.equal(CAMERA_STORAGE_KEY, "benchmarkDashboardManualEditorCamera");
});

test("zoom limits clamp the current scale", () => {
	const limits = zoomLimits({ minX: 0, minY: 0, maxX: 10, maxY: 10 }, 100, 100, 0.01);
	assert.equal(limits.minScale, 0.25);
	assert.equal(limits.maxScale, 1800);
	assert.equal(limits.scale, limits.minScale);
});
