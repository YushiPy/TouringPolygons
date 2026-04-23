
import { Vector2 } from "./vector2.js";
import { convexPartition } from "./convex-partition.js";

const BINARY_SEACH_THRESHOLD = 15;


// Vector operations

function segmentSegmentIntersection(start1, end1, start2, end2) {

	const direction1 = end1.sub(start1);
	const direction2 = end2.sub(start2);

	const cross = direction1.cross(direction2);

	if (cross === 0) {
		return null;
	}

	const sdiff = start2.sub(start1);
	const rate1 = sdiff.cross(direction2) / cross;
	const rate2 = sdiff.cross(direction1) / cross;

	if (rate1 >= 0 && rate1 <= 1 && rate2 >= 0 && rate2 <= 1) {
		return start1.add(direction1.mul(rate1));
	}

	return null;
}

// Point location functions

function pointInCone(point, vertex, ray1, ray2) {

	if (ray1.cross(ray2) >= 0) {
		return ray1.cross(point.sub(vertex)) >= 0 && ray2.cross(point.sub(vertex)) <= 0;
	} else {
		return ray1.cross(point.sub(vertex)) >= 0 || ray2.cross(point.sub(vertex)) <= 0;
	}
}

function pointInEdge(point, vertex1, ray1, vertex2, ray2) {
	return ray1.cross(point.sub(vertex1)) > 0 && ray2.cross(point.sub(vertex2)) < 0 && vertex2.sub(vertex1).cross(point.sub(vertex1)) <= 0;
}

function pointInCone2(point, vertex, ray1, ray2) {

	if (ray1.isSameDirection(ray2)) {
		return point.sub(vertex).isSameDirection(ray1);
	}

	if (ray1.cross(ray2) < 0) {
		return ray1.cross(point.sub(vertex)) >= 0 || ray2.cross(point.sub(vertex)) <= 0;
	} else {
		return ray1.cross(point.sub(vertex)) >= 0 && ray2.cross(point.sub(vertex)) <= 0;
	}
}

function pointInEdge2(point, vertex1, vertex2, ray1, ray2) {

	if (vertex1.x === vertex2.x && vertex1.y === vertex2.y) {
		return pointInCone2(point, vertex1, ray1, ray2);
	}

	const dv = vertex2.sub(vertex1);

	if (ray1.isSameDirection(dv) || ray2.mul(-1).isSameDirection(dv)) {
		return false;
	}

	const p1 = point.sub(vertex1);
	const p2 = point.sub(vertex2);

	const rp1 = ray1.cross(p1) >= 0;
	const rp2 = ray2.cross(p2) <= 0;
	const dp = dv.cross(p1) <= 0;

	if (dv.cross(ray1) < 0) {
		if (dv.cross(ray2) < 0) {
			return rp1 && rp2 && dp;
		} else {
			return rp1 ? dp : rp2;
		}
	} else {
		if (dv.cross(ray2) < 0) {
			return rp2 ? dp : rp1;
		} else {
			return rp1 || rp2 || dp;
		}
	}
}

// Cleanup functions

function removeCollinearPoints(points) {

	const cleaned = [points[0], points[1]];

	for (let i = 2; i < points.length; i++) {

		const a = cleaned[cleaned.length - 2];
		const b = cleaned[cleaned.length - 1];
		const candidate = points[i];

		const v1 = b.sub(a);
		const v2 = candidate.sub(b);

		if (v1.isSameDirection(v2)) {
			cleaned[cleaned.length - 1] = candidate;
		} else {
			cleaned.push(candidate);
		}
	}

	return cleaned;
}

function cleanPolygon(polygon) {

	const cleaned = removeCollinearPoints(polygon);

	if (cleaned[1].sub(cleaned[0]).cross(cleaned[cleaned.length - 1].sub(cleaned[0])) < 0) {
		cleaned.reverse();
	}

	return cleaned;
}


export function tppSolveConvex(start, target, polygons, simplify = false) {

	start = new Vector2(start);
	target = new Vector2(target);
	polygons = polygons.map(polygon => polygon.map(vertex => new Vector2(vertex)));

	if (simplify) {
		polygons = polygons.map(cleanPolygon);
	}

	const cones = polygons.map(polygon => Array(polygon.length).fill(null));
	const firstContact = polygons.map(polygon => Array(polygon.length).fill(false));

	function getCone(i, j) {

		if (cones[i][j] === null) {

			const before = polygons[i][(j - 1 + polygons[i].length) % polygons[i].length];
			const vertex = polygons[i][j];
			const after = polygons[i][(j + 1) % polygons[i].length];

			const last = query(vertex, i);
			const diff = vertex.sub(last);

			let ray1 = diff.reflect(vertex.sub(before).perpendicular());
			let ray2 = diff.reflect(after.sub(vertex).perpendicular());

			firstContact[i][(j - 1 + polygons[i].length) % polygons[i].length] = diff.cross(vertex.sub(before)) < 0;
			firstContact[i][j] = diff.cross(after.sub(vertex)) < 0;

			if (!firstContact[i][(j - 1 + polygons[i].length) % polygons[i].length]) {
				ray1 = diff;
			}

			if (!firstContact[i][j]) {
				ray2 = diff;
			}

			cones[i][j] = [ray1, ray2];
		}

		return cones[i][j];
	}

	function locatePointLinearSearch(point, i) {

		const polygon = polygons[i];

		for (let j = 0; j < polygon.length; j++) {

			const v = polygon[j];
			const [ray1, ray2] = getCone(i, j);

			if (!firstContact[i][j] && !firstContact[i][(j - 1 + polygon.length) % polygon.length]) {
				continue;
			}

			if (pointInCone(point, v, ray1, ray2)) {
				return 2 * j;
			}
		}

		for (let j = 0; j < polygon.length; j++) {

			const v1 = polygon[j];
			const v2 = polygon[(j + 1) % polygon.length];

			const ray1 = getCone(i, j)[1];
			const ray2 = getCone(i, (j + 1) % polygon.length)[0];

			if (pointInEdge(point, v1, ray1, v2, ray2)) {
				return firstContact[i][j] ? 2 * j + 1 : -1;
			}
		}

		return -1;
	}

	function locatePointBinarySearch(point, i) {

		function checkVertex(j) {
			return pointInCone2(point, polygon[j], ...getCone(i, j));
		}

		function checkEdge(l, r) {

			const rIndex = (r + 1) % polygon.length;

			const v1 = polygon[l];
			const v2 = polygon[rIndex];
			const ray1 = getCone(i, l)[1];
			const ray2 = getCone(i, rIndex)[0];

			return pointInEdge2(point, v1, v2, ray1, ray2);
		}

		const polygon = polygons[i];

		let left = 0;
		let right = polygon.length - 1;

		if (checkVertex(0)) {
			return 0;
		}

		while (left !== right) {

			const mid = Math.floor((left + right) / 2);

			if (checkVertex(mid + 1)) {
				return 2 * (mid + 1);
			}

			if (checkEdge(left, mid)) {
				right = mid;
			} else {
				left = mid + 1;
			}
		}

		if (!checkEdge(left, right)) {
			throw new Error("Point is not located in any cone or edge.");
		}

		return 2 * left + 1;
	}

	function locatePoint(point, i) {

		if (polygons[i].length < BINARY_SEACH_THRESHOLD) {
			return locatePointLinearSearch(point, i);
		}

		const location = locatePointBinarySearch(point, i);
		const visible = firstContact[i];

		if (visible[Math.floor(location / 2)] || visible[Math.floor((location - 1) / 2)]) {
			return location;
		} else {
			return -1;
		}
	}

	function queryFull(point, i) {

		if (i === 0) {
			return [start, point];
		}

		const polygon = polygons[i - 1];
		const location = locatePoint(point, i - 1);

		if (location === -1) {
			return queryFull(point, i - 1);
		}

		const pos = Math.floor(location / 2);

		if (location % 2 === 0) {
			return queryFull(polygon[pos], i - 1).concat([point]);
		}

		const v1 = polygon[pos];
		const v2 = polygon[(pos + 1) % polygon.length];

		const reflected = point.reflectSegment(v1, v2);

		const path = queryFull(reflected, i - 1);
		const last = path[path.length - 2];

		const intersection = segmentSegmentIntersection(last, reflected, v1, v2);

		if (intersection === null) {
			throw new Error(`Intersection not found for point ${point} in polygon ${i} at edge ${pos}`);
		}

		return path.slice(0, -1).concat([intersection, point]);
	}

	function query(point, i) {

		if (i === 0) {
			return start;
		}

		const polygon = polygons[i - 1];
		const location = locatePoint(point, i - 1);

		if (location === -1) {
			return query(point, i - 1);
		}

		const pos = Math.floor(location / 2);

		if (location % 2 === 0) {
			return polygon[pos];
		}

		const v1 = polygon[pos];
		const v2 = polygon[(pos + 1) % polygon.length];

		const reflected = point.reflectSegment(v1, v2);
		const last = query(reflected, i - 1);

		const intersection = segmentSegmentIntersection(last, reflected, v1, v2);

		if (intersection === null) {
			throw new Error(`Intersection not found for point ${point} in polygon ${i} at edge ${pos}`);
		}

		return intersection;
	}

	return removeCollinearPoints(queryFull(target, polygons.length));
}

function convexHull(points) {

	const pts = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
	const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

	const lower = [];

	for (const p of pts) {
		while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), p) <= 0)
			lower.pop();
		lower.push(p);
	}

	const upper = [];

	for (const p of pts.toReversed()) {
		while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), p) <= 0)
			upper.pop();
		upper.push(p);
	}

	upper.pop();
	lower.pop();

	return [...lower, ...upper]; // CCW
}

function pathLength(path) {
	
	let length = 0;

	for (let i = 1; i < path.length; i++) {
		length += path[i].distanceTo(path[i - 1]);
	}

	return length;
}

export function tppSolveBNB(start, target, polygons, simplify = false) {
	
	start = new Vector2(start);
	target = new Vector2(target);
	polygons = polygons.map(polygon => polygon.map(vertex => new Vector2(vertex)));

	if (simplify) {
		polygons = polygons.map(cleanPolygon);
	}

	const convexHulls = polygons.map(convexHull);
	const convexPartitions = polygons.map(polygon => convexPartition(polygon));

	function lowerBound(instance) {

		const selectedPolygons = instance.map((index, i) => convexPartitions[i][index]);
		const remainingHulls = convexHulls.slice(instance.length, polygons.length);
		const inputPolygons = selectedPolygons.concat(remainingHulls);

		try {
			const path = tppSolveConvex(start, target, inputPolygons);
			return pathLength(path);
		} catch (e) {
			console.log("Error in lower bound calculation:", e);
			return Infinity;
		}
	}

	let minimalPath = null;
	let minimalPathLength = Infinity;

	let queue = [[]];

	while (queue.length > 0) {

		const current = queue.pop();

		if (current.length == polygons.length) {

			const instance = current.map((index, i) => convexPartitions[i][index]);

			try {

				const path = tppSolveConvex(start, target, instance);
				const length = path.reduce((sum, point, i) => i > 0 ? sum + point.distanceTo(path[i - 1]) : sum, 0);

				if (length < minimalPathLength) {
					minimalPath = path;
					minimalPathLength = length;
				}
			} catch (e) {
				// No path found for this combination, ignore
			}
		} else {
			
			const i = current.length;

			for (let j = 0; j < convexPartitions[i].length; j++) {
				
				const next = current.concat(j);
				const bound = lowerBound(next);

				if (bound < minimalPathLength) {
					queue.push(next);
				}
			}
		}
	}
	
	if (minimalPath === null) {
		throw new Error("No path found.");
	}

	return minimalPath;
}

export function tppSolve(start, target, polygons, simplify = false) {
	return tppSolveBNB(start, target, polygons, simplify);
}
