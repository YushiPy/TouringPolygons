const MAP_COLORS = ["#38bdf8", "#a3e635", "#f97316", "#f472b6", "#c084fc"];

function viewBounds(renderer) {
	const width = Math.max(renderer.canvas.offsetWidth || renderer.canvas.width || 1, 1);
	const height = Math.max(renderer.canvas.offsetHeight || renderer.canvas.height || 1, 1);
	const center = renderer.canvasToWorld(width / 2, height / 2);
	const halfWidth = width / (2 * renderer.scale);
	const halfHeight = height / (2 * renderer.scale);
	const padding = Math.max(1 / renderer.scale, 1e-6);
	return [
		center[0] - halfWidth - padding,
		center[1] - halfHeight - padding,
		center[0] + halfWidth + padding,
		center[1] + halfHeight + padding,
	];
}

function rayInterval(origin, direction, bounds) {
	if (!origin?.every(Number.isFinite) || !direction?.every(Number.isFinite)) {
		return null;
	}
	const directionLength = Math.hypot(direction[0], direction[1]);
	if (directionLength <= 1e-12) {
		return null;
	}
	let lower = 0;
	let upper = Number.POSITIVE_INFINITY;
	for (const [position, delta, minimum, maximum] of [
		[origin[0], direction[0], bounds[0], bounds[2]],
		[origin[1], direction[1], bounds[1], bounds[3]],
	]) {
		if (Math.abs(delta) <= 1e-12) {
			if (position < minimum || position > maximum) return null;
			continue;
		}
		const first = (minimum - position) / delta;
		const second = (maximum - position) / delta;
		lower = Math.max(lower, Math.min(first, second));
		upper = Math.min(upper, Math.max(first, second));
		if (lower > upper) return null;
	}
	if (!Number.isFinite(upper) || upper < lower || upper < 0) return null;
	return {
		start: Math.max(lower, 0),
		end: upper,
	};
}

export function clipRayToViewport(origin, direction, bounds) {
	const interval = rayInterval(origin, direction, bounds);
	if (!interval || interval.end - interval.start <= 1e-12) return null;
	return {
		start: [origin[0] + direction[0] * interval.start, origin[1] + direction[1] * interval.start],
		end: [origin[0] + direction[0] * interval.end, origin[1] + direction[1] * interval.end],
	};
}

function boundaryPoint(origin, direction, bounds) {
	const clipped = clipRayToViewport(origin, direction, bounds);
	return clipped?.end || null;
}

function wallIndex(point, bounds) {
	if (!point) return -1;
	const tolerance = Math.max(1e-7, (bounds[2] - bounds[0] + bounds[3] - bounds[1]) * 1e-8);
	if (Math.abs(point[1] - bounds[1]) <= tolerance) return 0;
	if (Math.abs(point[0] - bounds[2]) <= tolerance) return 1;
	if (Math.abs(point[1] - bounds[3]) <= tolerance) return 2;
	if (Math.abs(point[0] - bounds[0]) <= tolerance) return 3;
	return -1;
}

function boundaryCorners(bounds) {
	return [
		[bounds[2], bounds[1]],
		[bounds[2], bounds[3]],
		[bounds[0], bounds[3]],
		[bounds[0], bounds[1]],
	];
}

function locateEdge(firstOrigin, firstDirection, secondOrigin, secondDirection, bounds) {
	const first = boundaryPoint(firstOrigin, firstDirection, bounds);
	const second = boundaryPoint(secondOrigin, secondDirection, bounds);
	if (!first || !second) return null;
	const firstWall = wallIndex(first, bounds);
	const secondWall = wallIndex(second, bounds);
	if (firstWall < 0 || secondWall < 0) return null;
	const corners = boundaryCorners(bounds);
	const cross = firstDirection[0] * secondDirection[1] - firstDirection[1] * secondDirection[0];
	const points = [firstOrigin, first];
	let wall = firstWall;
	if (wall === secondWall && cross < 0) {
		for (let index = wall; index < wall + 4; index += 1) points.push(corners[index % 4]);
	} else {
		while (wall !== secondWall) {
			points.push(corners[wall]);
			wall = (wall + 1) % 4;
		}
	}
	points.push(second, secondOrigin);
	return points;
}

function drawPolygon(ctx, renderer, points, fill, stroke = null, lineWidth = 1) {
	if (!points || points.length < 3) return;
	ctx.save();
	ctx.beginPath();
	points.forEach((point, index) => {
		const canvasPoint = renderer.worldToCanvas(point);
		if (index === 0) ctx.moveTo(canvasPoint.x, canvasPoint.y);
		else ctx.lineTo(canvasPoint.x, canvasPoint.y);
	});
	ctx.closePath();
	if (fill) {
		ctx.fillStyle = fill;
		ctx.fill();
	}
	if (stroke) {
		ctx.strokeStyle = stroke;
		ctx.lineWidth = lineWidth;
		ctx.stroke();
	}
	ctx.restore();
}

function drawRay(ctx, renderer, origin, direction, bounds, color, width = 1.5) {
	const clipped = clipRayToViewport(origin, direction, bounds);
	if (!clipped) return;
	const start = renderer.worldToCanvas(clipped.start);
	const end = renderer.worldToCanvas(clipped.end);
	ctx.save();
	ctx.strokeStyle = color;
	ctx.lineWidth = width;
	ctx.beginPath();
	ctx.moveTo(start.x, start.y);
	ctx.lineTo(end.x, end.y);
	ctx.stroke();
	ctx.restore();
}

function conePoints(origin, firstDirection, secondDirection, bounds) {
	return locateEdge(origin, firstDirection, origin, secondDirection, bounds);
}

/** Render the C++/WASM cone maps without recreating the solver in JavaScript. */
export function drawLastStepMap(renderer, mapData) {
	if (!mapData?.polygons?.length) return;
	const bounds = viewBounds(renderer);
	const ctx = renderer.ctx;

	for (const [polygonIndex, map] of mapData.polygons.entries()) {
		const polygon = map.polygon;
		const cones = map.cones;
		const firstContact = map.firstContact;
		if (!polygon || !cones || polygon.length !== cones.length || firstContact.length !== polygon.length) continue;
		const baseColor = MAP_COLORS[polygonIndex % MAP_COLORS.length];

		for (let index = 0; index < polygon.length; index += 1) {
			const next = (index + 1) % polygon.length;
			const firstRay = cones[index]?.[1];
			const secondRay = cones[next]?.[0];
			const points = locateEdge(polygon[index], firstRay, polygon[next], secondRay, bounds);
			if (points) {
				drawPolygon(ctx, renderer, points,
					firstContact[index] ? "rgba(34,197,94,0.10)" : "rgba(14,165,233,0.055)");
			}
		}

		for (let index = 0; index < polygon.length; index += 1) {
			const previous = (index + polygon.length - 1) % polygon.length;
			const cone = cones[index];
			if (!cone) continue;
			const hasContact = Boolean(firstContact[index] || firstContact[previous]);
			if (!hasContact) {
				drawRay(ctx, renderer, polygon[index], cone[0], bounds, `${baseColor}cc`, 1.7);
				continue;
			}
			const points = conePoints(polygon[index], cone[0], cone[1], bounds);
			if (points) {
				drawPolygon(ctx, renderer, points, `${baseColor}22`, `${baseColor}dd`, 1.5);
			} else {
				drawRay(ctx, renderer, polygon[index], cone[0], bounds, `${baseColor}dd`, 1.5);
				drawRay(ctx, renderer, polygon[index], cone[1], bounds, `${baseColor}dd`, 1.5);
			}
		}
	}
}
