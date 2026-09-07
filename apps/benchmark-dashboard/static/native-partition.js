const cache = new Map();

export function displayPartition(polygon, ready, failed) {
	const key = JSON.stringify(polygon);
	if (!cache.has(key)) {
		const entry = { pieces: null, settled: false, listeners: new Map() };
		cache.set(key, entry);
		fetch("/api/geometry/partition", {
			method: "POST", headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ polygons: [polygon] }),
		}).then(async (response) => {
			if (!response.ok) throw new Error("Decomposição nativa indisponível para esta região.");
			entry.pieces = (await response.json()).pieces[0];
			for (const callback of entry.listeners.keys()) callback();
		}).catch((error) => {
			for (const callback of entry.listeners.values()) callback(error.message);
		}).finally(() => { entry.settled = true; entry.listeners.clear(); });
	}
	const entry = cache.get(key);
	if (!entry.settled) entry.listeners.set(ready, failed);
	return entry.pieces;
}
