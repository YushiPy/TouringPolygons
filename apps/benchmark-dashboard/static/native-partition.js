const cache = new WeakMap();

export function displayPartition(polygon, ready, failed) {
	const key = JSON.stringify(polygon);
	if (cache.get(polygon)?.key !== key) {
		const previous = cache.get(polygon);
		previous?.listeners.clear();
		previous?.controller.abort();
		const entry = { key, pieces: null, settled: false, listeners: new Map(), controller: new AbortController() };
		cache.set(polygon, entry);
		fetch("/api/geometry/partition", {
			signal: entry.controller.signal,
			method: "POST", headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ polygons: [polygon] }),
		}).then(async (response) => {
			if (!response.ok) throw new Error("Decomposição nativa indisponível para esta região.");
			entry.pieces = (await response.json()).pieces[0];
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
