const cache = new WeakMap();
const shapeCache = new Map();
const MAX_SHAPE_CACHE_SIZE = 128;

function rounded(value) {
	return Math.round(value * 1e9) / 1e9;
}

function partitionShape(polygon) {
	const origin = polygon[0] || [0, 0];
	const key = JSON.stringify(polygon.map(([x, y]) => [rounded(x - origin[0]), rounded(y - origin[1])]));
	return { key, origin: [...origin] };
}

function translatePieces(pieces, origin) {
	return pieces.map((piece) => piece.map(([x, y]) => [x + origin[0], y + origin[1]]));
}

function rememberShape(key, pieces) {
	shapeCache.delete(key);
	shapeCache.set(key, pieces);
	while (shapeCache.size > MAX_SHAPE_CACHE_SIZE) {
		shapeCache.delete(shapeCache.keys().next().value);
	}
}

export function displayPartition(polygon, ready, failed) {
	const { key, origin } = partitionShape(polygon);
	const current = cache.get(polygon);
	if (current?.key === key) {
		if (current.origin[0] !== origin[0] || current.origin[1] !== origin[1]) {
			current.origin = origin;
			if (current.relativePieces) {
				current.pieces = translatePieces(current.relativePieces, origin);
			}
		}
		if (!current.settled) current.listeners.set(ready, failed);
		return current.pieces;
	}
	if (current?.key !== key) {
		if (current) {
			current.listeners.clear();
			current.controller.abort();
		}
		const relativePieces = shapeCache.get(key);
		if (relativePieces) {
			rememberShape(key, relativePieces);
			const entry = {
				key,
				origin,
				relativePieces,
				pieces: translatePieces(relativePieces, origin),
				settled: true,
				listeners: new Map(),
				controller: new AbortController(),
			};
			cache.set(polygon, entry);
			return entry.pieces;
		}
		const entry = { key, origin, requestOrigin: [...origin], relativePieces: null, pieces: null, settled: false, listeners: new Map(), controller: new AbortController() };
		cache.set(polygon, entry);
		fetch("/api/geometry/partition", {
			signal: entry.controller.signal,
			method: "POST", headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ polygons: [polygon] }),
		}).then(async (response) => {
			if (!response.ok) throw new Error("Decomposição nativa indisponível para esta região.");
			const pieces = (await response.json()).pieces[0];
			const relativePieces = pieces.map((piece) => piece.map(([x, y]) => [x - entry.requestOrigin[0], y - entry.requestOrigin[1]]));
			rememberShape(key, relativePieces);
			entry.relativePieces = relativePieces;
			entry.pieces = translatePieces(relativePieces, entry.origin);
			entry.settled = true;
			const callbacks = [...entry.listeners.keys()];
			entry.listeners.clear();
			for (const callback of callbacks) callback();
		}).catch((error) => {
			entry.settled = true;
			const callbacks = [...entry.listeners.values()];
			entry.listeners.clear();
			for (const callback of callbacks) callback(error.message);
		});
	}
	const entry = cache.get(polygon);
	if (!entry.settled) entry.listeners.set(ready, failed);
	return entry.pieces;
}
