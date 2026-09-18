export function convexHull(points) {
	const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
	const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
	const half = (values) => {
		const hull = [];
		for (const point of values) {
			while (hull.length > 1 && cross(hull.at(-2), hull.at(-1), point) <= 0) hull.pop();
			hull.push(point);
		}
		return hull.slice(0, -1);
	};
	return [...half(sorted), ...half([...sorted].reverse())];
}

export function projectedCase(row, width = 840, height = 480) {
	const points = [...row.geometry.polygons.flat(), ...row.path, row.geometry.start, row.geometry.target];
	const xs = points.map((point) => point[0]), ys = points.map((point) => point[1]);
	const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
	const scale = Math.min((width - 80) / Math.max(maxX - minX, 1e-9), (height - 80) / Math.max(maxY - minY, 1e-9));
	const offsetX = (width - (maxX - minX) * scale) / 2, offsetY = (height - (maxY - minY) * scale) / 2;
	const project = ([x, y]) => [offsetX + (x - minX) * scale, height - offsetY - (y - minY) * scale];
	return { project, path: row.path.map(project), polygons: row.geometry.polygons.map((polygon) => polygon.map(project)) };
}

export function pathPrefix(path, fraction) {
	if (!path.length) return [];
	const distances = path.slice(1).map((point, index) => Math.hypot(point[0] - path[index][0], point[1] - path[index][1]));
	let remaining = Math.max(0, Math.min(1, fraction)) * distances.reduce((sum, value) => sum + value, 0);
	const prefix = [path[0]];
	for (let index = 0; index < distances.length; index += 1) {
		if (remaining >= distances[index]) {
			prefix.push(path[index + 1]);
			remaining -= distances[index];
		} else {
			const ratio = remaining / distances[index];
			prefix.push(path[index].map((value, axis) => value + ratio * (path[index + 1][axis] - value)));
			break;
		}
	}
	return prefix;
}

function pointOnSegment(point, start, end, epsilon = 1e-9) {
	const cross = (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0]);
	if (Math.abs(cross) > epsilon * Math.max(1, Math.hypot(end[0] - start[0], end[1] - start[1]))) return false;
	return point[0] >= Math.min(start[0], end[0]) - epsilon && point[0] <= Math.max(start[0], end[0]) + epsilon
		&& point[1] >= Math.min(start[1], end[1]) - epsilon && point[1] <= Math.max(start[1], end[1]) + epsilon;
}

function pointInPolygon(point, polygon) {
	let inside = false;
	for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
		const a = polygon[previous], b = polygon[index];
		if (pointOnSegment(point, a, b)) return true;
		if ((a[1] > point[1]) !== (b[1] > point[1])
			&& point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
	}
	return inside;
}

function segmentIntersectionFraction(start, end, a, b, epsilon = 1e-9) {
	const direction = [end[0] - start[0], end[1] - start[1]];
	const edge = [b[0] - a[0], b[1] - a[1]];
	const offset = [a[0] - start[0], a[1] - start[1]];
	const cross = (left, right) => left[0] * right[1] - left[1] * right[0];
	const denominator = cross(direction, edge);
	if (Math.abs(denominator) <= epsilon) {
		if (Math.abs(cross(offset, direction)) > epsilon) return null;
		const lengthSquared = direction[0] ** 2 + direction[1] ** 2;
		if (lengthSquared <= epsilon ** 2) return pointOnSegment(start, a, b, epsilon) ? 0 : null;
		const first = ((a[0] - start[0]) * direction[0] + (a[1] - start[1]) * direction[1]) / lengthSquared;
		const second = ((b[0] - start[0]) * direction[0] + (b[1] - start[1]) * direction[1]) / lengthSquared;
		const entry = Math.max(0, Math.min(first, second));
		return entry <= Math.min(1, Math.max(first, second)) + epsilon ? entry : null;
	}
	const alongPath = cross(offset, edge) / denominator;
	const alongEdge = cross(offset, direction) / denominator;
	return alongPath >= -epsilon && alongPath <= 1 + epsilon && alongEdge >= -epsilon && alongEdge <= 1 + epsilon
		? Math.max(0, Math.min(1, alongPath)) : null;
}

export function pathPolygonContacts(path, polygons) {
	const lengths = path.slice(1).map((point, index) => Math.hypot(point[0] - path[index][0], point[1] - path[index][1]));
	const total = lengths.reduce((sum, length) => sum + length, 0);
	return polygons.map((polygon) => {
		let traversed = 0;
		for (let index = 0; index < path.length - 1; index += 1) {
			const start = path[index], end = path[index + 1], length = lengths[index];
			let entry = pointInPolygon(start, polygon) ? 0 : null;
			for (let edge = 0; edge < polygon.length; edge += 1) {
				const candidate = segmentIntersectionFraction(start, end, polygon[edge], polygon[(edge + 1) % polygon.length]);
				if (candidate !== null && (entry === null || candidate < entry)) entry = candidate;
			}
			if (entry !== null) {
				return {
					point: [start[0] + entry * (end[0] - start[0]), start[1] + entry * (end[1] - start[1])],
					fraction: total > 0 ? (traversed + entry * length) / total : 0,
				};
			}
			traversed += length;
		}
		if (path.length && pointInPolygon(path.at(-1), polygon)) return { point: [...path.at(-1)], fraction: 1 };
		return null;
	});
}

export function filterRows(rows, status, search) {
	const term = search.trim();
	return rows.filter((row) => (status === "all" || row.termination === status)
		&& (!term || String(row.case).padStart(2, "0").includes(term)));
}

export function endpointOffset(path, target, distance) {
	const points = target ? [...path].reverse() : path;
	const a = points[0];
	for (const b of points.slice(1)) {
		const length = Math.hypot(a[0] - b[0], a[1] - b[1]);
		if (length > 1e-10) return [a[0] + distance * (a[0] - b[0]) / length, a[1] + distance * (a[1] - b[1]) / length];
	}
	return [a[0] + (target ? distance : -distance), a[1]];
}

export function regionColors(rank, count, visited) {
	const position = count > 1 ? rank / (count - 1) : 0;
	return visited
		? { fill: `hsl(${105 + position * 70} 60% ${62 - position * 32}% / .62)`, stroke: `hsl(${105 + position * 70} 65% 68%)` }
		: { fill: `hsl(${195 + position * 65} 60% ${65 - position * 30}% / .4)`, stroke: `hsl(${195 + position * 65} 55% 70%)` };
}

export function sortRows(rows, key = "case", descending = false) {
	const value = (row) => key === "result" ? Number(!row.exact) : row[key];
	return [...rows].sort((a, b) => (descending ? -1 : 1) * (value(a) - value(b)) || a.case - b.case);
}

export function playbackDuration(polygons) {
	return 1800 + 15 * Math.min(80, Math.max(0, polygons));
}

export function sortGroupedRows(rows, key, descending, resultGroup = null) {
	const ordered = sortRows(rows, key, descending);
	if (resultGroup === null) return ordered;
	const first = resultGroup ? false : true;
	return [...ordered.filter(row => row.exact === first), ...ordered.filter(row => row.exact !== first)];
}

export function toggleOrderRegion(order, index) {
	return order.includes(index) ? order.filter(value => value !== index) : [...order, index];
}

export function toggleChoice(current, next) {
	return current === next ? null : next;
}
