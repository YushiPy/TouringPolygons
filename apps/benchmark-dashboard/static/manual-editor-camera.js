export { CAMERA_STORAGE_KEY } from "./storage.js";

export function caseBounds(caseData) {
	if (!caseData) {
		return null;
	}
	const points = [caseData.start, caseData.target, ...caseData.polygons.flat()];
	if (points.length === 0) {
		return null;
	}
	const xs = points.map((point) => point[0]);
	const ys = points.map((point) => point[1]);
	return {
		minX: Math.min(...xs),
		minY: Math.min(...ys),
		maxX: Math.max(...xs),
		maxY: Math.max(...ys),
	};
}

export function zoomLimits(bounds, width, height, scale) {
	if (!bounds) {
		return { minScale: 0.1, maxScale: 50000, scale };
	}
	const safeWidth = Math.max(width, 1);
	const safeHeight = Math.max(height, 1);
	const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
	const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
	const fitScale = Math.min(safeWidth / spanX, safeHeight / spanY);
	const minScale = Math.max(0.00001, fitScale * 0.025);
	const maxScale = Math.min(2000000, Math.max(fitScale * 180, minScale * 10));
	return { minScale, maxScale, scale: Math.max(minScale, Math.min(maxScale, scale)) };
}

export function worldToCanvas(point, camera) {
	return {
		x: camera.offsetX + point[0] * camera.scale,
		y: camera.offsetY - point[1] * camera.scale,
	};
}

export function canvasToWorld(x, y, camera) {
	return [
		(x - camera.offsetX) / camera.scale,
		-(y - camera.offsetY) / camera.scale,
	];
}
