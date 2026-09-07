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

export function gapRatio(row) {
	return row.upper_bound === 0 ? 0 : Math.max(0, (row.upper_bound - row.lower_bound) / Math.abs(row.upper_bound));
}

export function filterRows(rows, status, search) {
	const term = search.trim();
	return rows.filter((row) => (status === "all" || row.termination === status)
		&& (!term || String(row.case).padStart(2, "0").includes(term)));
}
