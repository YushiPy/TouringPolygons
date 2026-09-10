/* global Image */

const TILE_SIZE = 256;
const MAX_TILES = 128;
const EARTH_RADIUS = 6378137;
const MAX_LATITUDE = 85.051129;

function clamp(value, minimum, maximum) {
	return Math.max(minimum, Math.min(maximum, value));
}

function worldPixel(latitude, longitude, zoom) {
	const size = TILE_SIZE * 2 ** zoom;
	const lat = clamp(latitude, -MAX_LATITUDE, MAX_LATITUDE) * Math.PI / 180;
	return {
		x: (longitude + 180) / 360 * size,
		y: (1 - Math.asinh(Math.tan(lat)) / Math.PI) / 2 * size,
	};
}

function pixelLocation(x, y, zoom) {
	const size = TILE_SIZE * 2 ** zoom;
	return {
		latitude: Math.atan(Math.sinh(Math.PI * (1 - 2 * y / size))) * 180 / Math.PI,
		longitude: x / size * 360 - 180,
	};
}

export function createSatelliteMap({ $, manualEditor, scheduleManualAutosave }) {
	const state = {
		canvas: null,
		ctx: null,
		latitude: -23.5614,
		longitude: -46.7308,
		zoom: 19,
		instanceScale: 1,
		offsetX: 0,
		offsetY: 0,
		unitsPerPixel: 1,
		mode: "polygon",
		points: [],
		drag: null,
		moved: false,
		tiles: new Map(),
		active: false,
		width: 0,
		height: 0,
		frame: null,
	};

	function resize() {
		if (!state.active) return;
		const container = state.canvas.parentElement;
		state.width = Math.min(2048, Math.max(1, container.clientWidth));
		state.height = Math.min(1024, Math.max(1, container.clientHeight));
		const ratio = Math.min(window.devicePixelRatio || 1, 2);
		const width = Math.max(1, Math.round(state.width * ratio));
		const height = Math.max(1, Math.round(state.height * ratio));
		if (state.canvas.width !== width) state.canvas.width = width;
		if (state.canvas.height !== height) state.canvas.height = height;
		state.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		draw();
	}

	function centerPixel() {
		return worldPixel(state.latitude, state.longitude, state.zoom);
	}

	function screenToLocation(x, y) {
		const center = centerPixel();
		return pixelLocation(center.x + (x - state.width / 2 - state.offsetX) / state.instanceScale,
			center.y + (y - state.height / 2 - state.offsetY) / state.instanceScale, state.zoom);
	}

	function screenToWorld(x, y) {
		return [(x - state.width / 2 - state.offsetX) * state.unitsPerPixel / state.instanceScale,
			-(y - state.height / 2 - state.offsetY) * state.unitsPerPixel / state.instanceScale];
	}

	function worldToScreen([x, y]) {
		return { x: state.width / 2 + state.offsetX + x / state.unitsPerPixel * state.instanceScale,
			y: state.height / 2 + state.offsetY - y / state.unitsPerPixel * state.instanceScale };
	}

	function requestTile(zoom, x, y) {
		const count = 2 ** zoom;
		const wrappedX = ((x % count) + count) % count;
		if (y < 0 || y >= count) return null;
		const key = `${zoom}/${wrappedX}/${y}`;
		if (!state.tiles.has(key)) {
			const image = new Image();
			image.crossOrigin = "anonymous";
			image.onload = draw;
			image.onerror = () => {
				if (state.active) $("#satellite-status").textContent = "Some satellite tiles could not load. Check your connection or try another area.";
			};
			image.src = `https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${y}/${wrappedX}`;
			state.tiles.set(key, image);
		}
		const image = state.tiles.get(key);
		state.tiles.delete(key);
		state.tiles.set(key, image);
		while (state.tiles.size > MAX_TILES) {
			const oldest = state.tiles.keys().next().value;
			const discarded = state.tiles.get(oldest);
			discarded.onload = discarded.onerror = null;
			if (!discarded.complete) discarded.src = "";
			state.tiles.delete(oldest);
		}
		return image;
	}

	function drawTiles() {
		const center = centerPixel();
		if (![center.x, center.y, state.width, state.height].every(Number.isFinite)) return;
		const left = center.x - (state.width / 2 + state.offsetX) / state.instanceScale;
		const top = center.y - (state.height / 2 + state.offsetY) / state.instanceScale;
		const minX = Math.floor(left / TILE_SIZE);
		const minY = Math.floor(top / TILE_SIZE);
		state.visibleTiles = [];
		const columns = Math.min(11, Math.ceil(state.width / state.instanceScale / TILE_SIZE) + 1);
		const rows = Math.min(8, Math.ceil(state.height / state.instanceScale / TILE_SIZE) + 1);
		for (let row = 0; row < rows; row += 1) {
			for (let column = 0; column < columns; column += 1) {
				const x = minX + column, y = minY + row;
				const image = requestTile(state.zoom, x, y);
				if (image) state.visibleTiles.push(image);
				if (image?.complete && image.naturalWidth) {
					state.ctx.drawImage(image, (x * TILE_SIZE - left) * state.instanceScale,
						(y * TILE_SIZE - top) * state.instanceScale, TILE_SIZE * state.instanceScale, TILE_SIZE * state.instanceScale);
				}
			}
		}
	}

	function drawExistingPolygons() {
		const current = manualEditor.currentCase();
		for (const polygon of current?.polygons || []) {
			state.ctx.beginPath();
			polygon.forEach((point, index) => {
				const screen = worldToScreen(point);
				if (index === 0) state.ctx.moveTo(screen.x, screen.y);
				else state.ctx.lineTo(screen.x, screen.y);
			});
			state.ctx.closePath();
			state.ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
			state.ctx.strokeStyle = "#38bdf8";
			state.ctx.lineWidth = 2;
			state.ctx.fill();
			state.ctx.stroke();
		}
		for (const [key, color] of [["start", "#22c55e"], ["target", "#ef4444"]]) {
			if (!current?.[key]) continue;
			const screen = worldToScreen(current[key]);
			state.ctx.fillStyle = color;
			state.ctx.beginPath();
			state.ctx.arc(screen.x, screen.y, 5, 0, 2 * Math.PI);
			state.ctx.fill();
			state.ctx.font = "14px system-ui";
			state.ctx.fillText(key === "start" ? "s" : "t", screen.x + 8, screen.y - 8);
		}
	}

	function drawTrace() {
		if (!state.points.length) return;
		state.ctx.beginPath();
		state.points.forEach((point, index) => {
			const screen = worldToScreen(point);
			if (index === 0) state.ctx.moveTo(screen.x, screen.y);
			else state.ctx.lineTo(screen.x, screen.y);
		});
		state.ctx.strokeStyle = "#facc15";
		state.ctx.lineWidth = 3;
		state.ctx.stroke();
		for (const point of state.points) {
			const screen = worldToScreen(point);
			state.ctx.beginPath();
			state.ctx.arc(screen.x, screen.y, 5, 0, Math.PI * 2);
			state.ctx.fillStyle = "#facc15";
			state.ctx.fill();
		}
	}

	function draw() {
		if (!state.active || state.frame !== null) return;
		state.frame = requestAnimationFrame(() => {
			state.frame = null;
			if (state.active) render();
		});
	}

	function render() {
		if (!state.ctx) return;
		state.ctx.clearRect(0, 0, state.width, state.height);
		state.ctx.fillStyle = "#17202a";
		state.ctx.fillRect(0, 0, state.width, state.height);
		drawTiles();
		drawExistingPolygons();
		drawTrace();
	}

	function syncInputs() {
		$("#satellite-latitude").value = state.latitude.toFixed(7);
		$("#satellite-longitude").value = state.longitude.toFixed(7);
	}

	function setCenter(latitude, longitude) {
		if (!Number.isFinite(Number(latitude)) || !Number.isFinite(Number(longitude))) return;
		state.latitude = clamp(Number(latitude), -85, 85);
		state.longitude = clamp(Number(longitude), -180, 180);
		syncInputs();
		draw();
	}

	function zoom(delta, x = state.width / 2, y = state.height / 2) {
		const anchor = screenToLocation(x, y);
		state.zoom = clamp(state.zoom + delta, 1, 20);
		const pixel = worldPixel(anchor.latitude, anchor.longitude, state.zoom);
		const center = pixelLocation(pixel.x - (x - state.width / 2 - state.offsetX) / state.instanceScale,
			pixel.y - (y - state.height / 2 - state.offsetY) / state.instanceScale, state.zoom);
		state.latitude = center.latitude;
		state.longitude = center.longitude;
		syncInputs();
		draw();
	}

	function zoomInstance(factor, x = state.width / 2, y = state.height / 2) {
		const oldScale = state.instanceScale;
		// Keep at most 96 visible tiles, including partially visible edge tiles.
		const minimum = Math.max(state.width / (10 * TILE_SIZE), state.height / (7 * TILE_SIZE));
		state.instanceScale = clamp(oldScale * factor, minimum, 16);
		const ratio = state.instanceScale / oldScale;
		state.offsetX = x - state.width / 2 - (x - state.width / 2 - state.offsetX) * ratio;
		state.offsetY = y - state.height / 2 - (y - state.height / 2 - state.offsetY) * ratio;
		draw();
	}

	function saveView() {
		const current = manualEditor.currentCase();
		current.map_view = { latitude: clamp(state.latitude, -85, 85),
			longitude: ((state.longitude + 180) % 360 + 360) % 360 - 180,
			zoom: state.zoom, units_per_pixel: state.unitsPerPixel };
	}

	function open() {
		if (!manualEditor.currentCase()) {
			manualEditor.setSaveStatus("Create or select an instance first.");
			return;
		}
		const view = manualEditor.currentCase().map_view;
		if (view) {
			state.latitude = clamp(view.latitude, -85, 85);
			state.longitude = clamp(view.longitude, -180, 180);
			state.zoom = clamp(Math.round(view.zoom), 1, 20);
		}
		state.unitsPerPixel = view?.units_per_pixel || (view ? 2 * Math.PI * EARTH_RADIUS / (TILE_SIZE * 2 ** view.zoom) : 0);
		if (!state.unitsPerPixel) {
			const bounds = manualEditor.caseBounds();
			state.unitsPerPixel = Math.max(bounds ? Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) / 400 : 1, 1e-6);
		}
		state.instanceScale = 1;
		state.offsetX = state.offsetY = 0;
		state.active = true;
		state.points = [];
		syncInputs();
		$("#satellite-modal").classList.remove("is-hidden");
		requestAnimationFrame(() => {
			resize();
			const bounds = manualEditor.caseBounds();
			if (bounds) {
				state.offsetX = -(bounds.minX + bounds.maxX) / 2 / state.unitsPerPixel;
				state.offsetY = (bounds.minY + bounds.maxY) / 2 / state.unitsPerPixel;
			}
			draw();
		});
	}

	function close() {
		if (state.active) { saveView(); scheduleManualAutosave(); }
		state.active = false;
		state.drag = null;
		if (state.frame !== null) cancelAnimationFrame(state.frame);
		state.frame = null;
		for (const image of state.tiles.values()) {
			image.onload = image.onerror = null;
			if (!image.complete) image.src = "";
		}
		state.tiles.clear();
		state.canvas.width = state.canvas.height = 1;
		$("#satellite-modal").classList.add("is-hidden");
	}

	function finish() {
		if (state.points.length < 3) {
			$("#satellite-status").textContent = "Choose at least three corners before finishing.";
			return;
		}
		const current = manualEditor.currentCase();
		saveView();
		current.polygons.push(state.points.map(point => [...point]));
		state.points = [];
		manualEditor.changed();
		scheduleManualAutosave({ immediate: true });
		$("#satellite-status").textContent = `Polygon added. ${current.polygons.length} polygons in this instance. Click to draw the next one.`;
		draw();
	}

	function useBackground() {
		if (!state.visibleTiles?.length || state.visibleTiles.some(image => !image.complete || !image.naturalWidth)) {
			$("#satellite-status").textContent = "Wait for the visible imagery to load before applying the background.";
			return;
		}
		state.ctx.fillStyle = "#17202a";
		state.ctx.fillRect(0, 0, state.width, state.height);
		drawTiles();
		try {
			const current = manualEditor.currentCase();
			const lower = screenToWorld(0, state.height), upper = screenToWorld(state.width, 0);
			current.background = { data_url: state.canvas.toDataURL("image/jpeg", 0.9), opacity: 0.65,
				bounds: [lower[0], lower[1], upper[0], upper[1]] };
			saveView();
			manualEditor.backgroundSource = "";
			manualEditor.frameBounds(manualEditor.backgroundBounds());
			window.dispatchEvent(new window.Event("manual-case-changed"));
			scheduleManualAutosave({ immediate: true });
			close();
		} catch {
			$("#satellite-status").textContent = "Could not export the imagery. Check the tile connection and try again.";
			draw();
		}
	}

	function init() {
		state.canvas = $("#satellite-map");
		state.ctx = state.canvas.getContext("2d");
		$("#manual-satellite-button").addEventListener("click", open);
		document.querySelectorAll("[data-close-satellite]").forEach((button) => button.addEventListener("click", close));
		$("#satellite-go").addEventListener("click", () => setCenter($("#satellite-latitude").value, $("#satellite-longitude").value));
		$("#satellite-zoom-in").addEventListener("click", () => zoom(1));
		$("#satellite-zoom-out").addEventListener("click", () => zoom(-1));
		$("#satellite-undo").addEventListener("click", () => { state.points.pop(); draw(); });
		$("#satellite-finish").addEventListener("click", finish);
		$("#satellite-use-background").addEventListener("click", useBackground);
		$("#satellite-instance-in").addEventListener("click", () => zoomInstance(1.25));
		$("#satellite-instance-out").addEventListener("click", () => zoomInstance(1 / 1.25));
		$("#satellite-tool").addEventListener("change", event => { state.mode = event.target.value; });
		state.canvas.addEventListener("pointerdown", (event) => {
			state.drag = { x: event.clientX, y: event.clientY, center: centerPixel(), offsetX: state.offsetX, offsetY: state.offsetY };
			state.moved = false;
			state.canvas.setPointerCapture(event.pointerId);
		});
		state.canvas.addEventListener("pointermove", (event) => {
			if (!state.drag) return;
			const dx = event.clientX - state.drag.x;
			const dy = event.clientY - state.drag.y;
			if (Math.hypot(dx, dy) > 3) state.moved = true;
			if (state.mode === "map") {
				const location = pixelLocation(state.drag.center.x - dx / state.instanceScale, state.drag.center.y - dy / state.instanceScale, state.zoom);
				state.latitude = clamp(location.latitude, -85, 85);
				state.longitude = ((location.longitude + 180) % 360 + 360) % 360 - 180;
				syncInputs();
			} else {
				state.offsetX = state.drag.offsetX + dx;
				state.offsetY = state.drag.offsetY + dy;
			}
			draw();
		});
		state.canvas.addEventListener("pointercancel", () => { state.drag = null; });
		state.canvas.addEventListener("pointerup", (event) => {
			if (!state.drag) return;
			if (!state.moved) {
				const rect = state.canvas.getBoundingClientRect();
				const point = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
				if (state.mode === "polygon") state.points.push(point);
				else if (state.mode === "start" || state.mode === "target") {
					manualEditor.currentCase()[state.mode] = point;
					manualEditor.changed();
				}
				draw();
			}
			state.drag = null;
		});
		state.canvas.addEventListener("wheel", (event) => {
			event.preventDefault();
			const rect = state.canvas.getBoundingClientRect();
			const x = event.clientX - rect.left, y = event.clientY - rect.top;
			if ($("#satellite-wheel-mode").value === "instance") zoomInstance(Math.exp(-clamp(event.deltaY, -100, 100) * 0.005), x, y);
			else zoom(event.deltaY < 0 ? 1 : -1, x, y);
		}, { passive: false });
		new ResizeObserver(() => {
			if (state.active) requestAnimationFrame(resize);
		}).observe(state.canvas.parentElement);
	}

	return { init, open, close };
}
