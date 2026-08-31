function signedArea(points) {
	let area = 0;
	for (let index = 0; index < points.length; index += 1) {
		const point = points[index];
		const next = points[(index + 1) % points.length];
		area += point[0] * next[1] - next[0] * point[1];
	}
	return area / 2;
}

function pointInTriangle(point, a, b, c) {
	const area = (u, v, w) => (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
	const ab = area(a, b, point);
	const bc = area(b, c, point);
	const ca = area(c, a, point);
	return (ab >= -1e-9 && bc >= -1e-9 && ca >= -1e-9) || (ab <= 1e-9 && bc <= 1e-9 && ca <= 1e-9);
}

function cross(a, b, c) {
	return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function segmentsAreAdjacent(first, second, length) {
	return first === second || (first + 1) % length === second || (second + 1) % length === first;
}

function segmentIntersection(a, b, c, d) {
	const denominator = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0]);
	if (Math.abs(denominator) <= 1e-9) {
		return null;
	}
	const t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / denominator;
	const u = ((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / denominator;
	if (t <= 1e-9 || t >= 1 - 1e-9 || u <= 1e-9 || u >= 1 - 1e-9) {
		return null;
	}
	return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
}

function selfIntersections(polygon) {
	const intersections = [];
	for (let first = 0; first < polygon.length; first += 1) {
		const a = polygon[first];
		const b = polygon[(first + 1) % polygon.length];
		for (let second = first + 1; second < polygon.length; second += 1) {
			if (segmentsAreAdjacent(first, second, polygon.length)) {
				continue;
			}
			const point = segmentIntersection(a, b, polygon[second], polygon[(second + 1) % polygon.length]);
			if (point) {
				intersections.push({ first, second, point });
			}
		}
	}
	return intersections;
}

function polygonIsSimple(polygon) {
	return selfIntersections(polygon).length === 0;
}

export function polygonIsConvex(polygon) {
	if (!polygonIsSimple(polygon)) {
		return false;
	}
	let gotNegative = false;
	let gotPositive = false;
	for (let index = 0; index < polygon.length; index += 1) {
		const a = polygon[index];
		const b = polygon[(index + 1) % polygon.length];
		const c = polygon[(index + 2) % polygon.length];
		const turn = cross(a, b, c);
		if (turn < 0) {
			gotNegative = true;
		} else if (turn > 0) {
			gotPositive = true;
		}
		if (gotNegative && gotPositive) {
			return false;
		}
	}
	return true;
}

function splitSingleSelfIntersection(polygon) {
	const intersections = selfIntersections(polygon);
	if (intersections.length !== 1) {
		return null;
	}
	const { first, second, point } = intersections[0];
	const firstLoop = [point];
	for (let index = (first + 1) % polygon.length; index !== (second + 1) % polygon.length; index = (index + 1) % polygon.length) {
		firstLoop.push([...polygon[index]]);
	}
	const secondLoop = [point];
	for (let index = (second + 1) % polygon.length; index !== (first + 1) % polygon.length; index = (index + 1) % polygon.length) {
		secondLoop.push([...polygon[index]]);
	}
	return [firstLoop, secondLoop].filter((piece) => Math.abs(signedArea(piece)) > 1e-9);
}

function earClipDecomposition(polygon) {
	if (polygon.length < 3) {
		return [];
	}
	const splitPieces = splitSingleSelfIntersection(polygon);
	if (splitPieces) {
		return splitPieces.flatMap(earClipDecomposition);
	}
	if (polygonIsConvex(polygon)) {
		return [polygon];
	}
	const oriented = signedArea(polygon) >= 0 ? polygon.map((point) => [...point]) : [...polygon].reverse().map((point) => [...point]);
	const remaining = oriented.map((_, index) => index);
	const pieces = [];
	let guard = 0;
	while (remaining.length > 3 && guard < oriented.length * oriented.length) {
		let clipped = false;
		for (let index = 0; index < remaining.length; index += 1) {
			const previousIndex = remaining[(index + remaining.length - 1) % remaining.length];
			const currentIndex = remaining[index];
			const nextIndex = remaining[(index + 1) % remaining.length];
			const previous = oriented[previousIndex];
			const current = oriented[currentIndex];
			const next = oriented[nextIndex];
			const cross = (current[0] - previous[0]) * (next[1] - current[1]) - (current[1] - previous[1]) * (next[0] - current[0]);
			if (cross <= 1e-9) {
				continue;
			}
			const containsOther = remaining.some((candidateIndex) => (
				candidateIndex !== previousIndex
				&& candidateIndex !== currentIndex
				&& candidateIndex !== nextIndex
				&& pointInTriangle(oriented[candidateIndex], previous, current, next)
			));
			if (containsOther) {
				continue;
			}
			pieces.push([previous, current, next].map((point) => [...point]));
			remaining.splice(index, 1);
			clipped = true;
			break;
		}
		if (!clipped) {
			return [polygon];
		}
		guard += 1;
	}
	if (remaining.length === 3) {
		pieces.push(remaining.map((index) => [...oriented[index]]));
	}
	return pieces;
}

const decompositionCache = new WeakMap();

export function convexDecomposition(polygon) {
	const signature = polygon.map((point) => `${point[0]},${point[1]}`).join(";");
	const cached = decompositionCache.get(polygon);
	if (cached?.signature === signature) {
		return cached.pieces;
	}
	const pieces = earClipDecomposition(polygon).filter((piece) => piece.length >= 3);
	decompositionCache.set(polygon, { signature, pieces });
	return pieces;
}

export function solutionDirectionAt(path, index) {
	if (!path || path.length < 2) {
		return index === 0 ? [1, 0] : [-1, 0];
	}
	if (index === 0) {
		return [path[0][0] - path[1][0], path[0][1] - path[1][1]];
	}
	const last = path.length - 1;
	return [path[last][0] - path[last - 1][0], path[last][1] - path[last - 1][1]];
}
