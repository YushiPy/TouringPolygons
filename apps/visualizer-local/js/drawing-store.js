/**
 * Persist drawings in IndexedDB (browser-local, no server).
 */

const DB_NAME = "tpp-visualizer-local";
const VERSION = 1;
const STORE_NAME = "drawings";

function openDb() {
	return new Promise((resolve, reject) => {
		const req = indexedDB.open(DB_NAME, VERSION);
		req.onerror = () => reject(req.error);
		req.onsuccess = () => resolve(req.result);
		req.onupgradeneeded = (event) => {
			const db = event.target.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
			}
		};
	});
}

/**
 * @param {string} jsonString — output of Scene.saveData()
 * @returns {Promise<number>} new drawing id
 */
export async function createDrawing(jsonString) {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, "readwrite");
		const store = tx.objectStore(STORE_NAME);
		const now = Date.now();
		const record = {
			payload: jsonString,
			createdAt: now,
			modifiedAt: now,
		};
		const req = store.add(record);
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => reject(req.error);
	});
}

/**
 * @param {number} id
 * @param {string} jsonString
 */
export async function updateDrawing(id, jsonString) {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, "readwrite");
		const store = tx.objectStore(STORE_NAME);
		const getReq = store.get(id);
		getReq.onsuccess = () => {
			const row = getReq.result;
			if (!row) {
				reject(new Error("Drawing not found"));
				return;
			}
			row.payload = jsonString;
			row.modifiedAt = Date.now();
			const putReq = store.put(row);
			putReq.onsuccess = () => resolve();
			putReq.onerror = () => reject(putReq.error);
		};
		getReq.onerror = () => reject(getReq.error);
	});
}

export async function deleteDrawing(id) {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, "readwrite");
		const store = tx.objectStore(STORE_NAME);
		const req = store.delete(id);
		req.onsuccess = () => resolve();
		req.onerror = () => reject(req.error);
	});
}

/**
 * @returns {Promise<Array<{ id: number, drawing_name: string, modified_at: string, created_at: string, dataURL: string }>>}
 */
export async function listDrawingsMeta() {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, "readonly");
		const store = tx.objectStore(STORE_NAME);
		const req = store.getAll();
		req.onsuccess = () => {
			const rows = req.result.map((r) => {
				let meta = {};
				try {
					meta = JSON.parse(r.payload);
				} catch {
					meta = {};
				}
				return {
					id: r.id,
					drawing_name: meta.drawingName ?? "Untitled",
					modified_at: new Date(r.modifiedAt).toISOString(),
					created_at: new Date(r.createdAt).toISOString(),
					dataURL: meta.dataURL ?? "",
				};
			});
			rows.sort((a, b) => new Date(b.modified_at) - new Date(a.modified_at));
			resolve(rows);
		};
		req.onerror = () => reject(req.error);
	});
}

/**
 * @param {number} id
 * @returns {Promise<object>} parsed drawing JSON for Scene.loadFromData
 */
export async function getDrawing(id) {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, "readonly");
		const store = tx.objectStore(STORE_NAME);
		const req = store.get(id);
		req.onsuccess = () => {
			const row = req.result;
			if (!row) {
				reject(new Error("Drawing not found"));
				return;
			}
			try {
				resolve(JSON.parse(row.payload));
			} catch (e) {
				reject(e);
			}
		};
		req.onerror = () => reject(req.error);
	});
}
