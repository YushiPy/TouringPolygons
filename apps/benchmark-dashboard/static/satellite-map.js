const TILE_SIZE = 256;
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

function mercatorMeters(latitude, longitude) {
	const lat = clamp(latitude, -MAX_LATITUDE, MAX_LATITUDE) * Math.PI / 180;
	return [
		EARTH_RADIUS * longitude * Math.PI / 180,
		EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + lat / 2)),
	];
}

export function createSatelliteMap({ $, manualEditor, scheduleManualAutosave }) {
	const state = {
		canvas: null,
		ctx: null,
		latitude: -23.5614,
		longitude: -46.7308,
		zoom: 19,
		points: [],
		drag: null,
		moved: false,
		tiles: new Map(),
	};

	function resize() {
		const ratio = window.devicePixelRatio || 1;
		state.canvas.width = Math.max(1, Math.round(state.canvas.clientWidth * ratio));
		state.canvas.height = Math.max(1, Math.round(state.canvas.clientHeight * ratio));
		state.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		draw();
	}

	function centerPixel() {
		return worldPixel(state.latitude, state.longitude, state.zoom);
	}

	function screenToLocation(x, y) {
		const center = centerPixel();
		return pixelLocation(
			center.x + x - state.canvas.clientWidth / 2,
			center.y + y - state.canvas.clientHeight / 2,
			state.zoom,
		);
	}

	function locationToScreen(location) {
		const center = centerPixel();
		const point = worldPixel(location.latitude, location.longitude, state.zoom);
		return {
			x: point.x - center.x + state.canvas.clientWidth / 2,
			y: point.y - center.y + state.canvas.clientHeight / 2,
		};
	}

	function requestTile(zoom, x, y) {
		const count = 2 ** zoom;
		const wrappedX = ((x % count) + count) % count;
		if (y < 0 || y >= count) return null;
		const key = `${zoom}/${wrappedX}/${y}`;
		if (!state.tiles.has(key)) {
			const image = new Image();
			image.crossOrigin = "anonymous";
			image.addEventListener("load", draw, { once: true });
			image.src = `https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${y}/${wrappedX}`;
			state.tiles.set(key, image);
		}
		return state.tiles.get(key);
	}

	function drawTiles() {
		const center = centerPixel();
		const left = center.x - state.canvas.clientWidth / 2;
		const top = center.y - state.canvas.clientHeight / 2;
		const minX = Math.floor(left / TILE_SIZE);
		const minY = Math.floor(top / TILE_SIZE);
		const maxX = Math.floor((left + state.canvas.clientWidth) / TILE_SIZE);
		const maxY = Math.floor((top + state.canvas.clientHeight) / TILE_SIZE);
		for (let y = minY; y <= maxY; y += 1) {
			for (let x = minX; x <= maxX; x += 1) {
				const image = requestTile(state.zoom, x, y);
				if (image?.complete && image.naturalWidth) {
					state.ctx.drawImage(image, x * TILE_SIZE - left, y * TILE_SIZE - top, TILE_SIZE, TILE_SIZE);
				}
			}
		}
	}

	function drawExistingPolygons() {
		const view = manualEditor.currentCase()?.map_view;
		if (!view) return;
		const origin = mercatorMeters(view.latitude, view.longitude);
		for (const polygon of manualEditor.currentCase()?.polygons || []) {
			state.ctx.beginPath();
			polygon.forEach(([x, y], index) => {
				const longitude = (origin[0] + x) / EARTH_RADIUS * 180 / Math.PI;
				const latitude = (2 * Math.atan(Math.exp((origin[1] + y) / EARTH_RADIUS)) - Math.PI / 2) * 180 / Math.PI;
				const screen = locationToScreen({ latitude, longitude });
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
	}

	function drawTrace() {
		if (!state.points.length) return;
		state.ctx.beginPath();
		state.points.forEach((point, index) => {
			const screen = locationToScreen(point);
			if (index === 0) state.ctx.moveTo(screen.x, screen.y);
			else state.ctx.lineTo(screen.x, screen.y);
		});
		state.ctx.strokeStyle = "#facc15";
		state.ctx.lineWidth = 3;
		state.ctx.stroke();
		for (const point of state.points) {
			const screen = locationToScreen(point);
			state.ctx.beginPath();
			state.ctx.arc(screen.x, screen.y, 5, 0, Math.PI * 2);
			state.ctx.fillStyle = "#facc15";
			state.ctx.fill();
		}
	}

	function draw() {
		if (!state.ctx) return;
		state.ctx.clearRect(0, 0, state.canvas.clientWidth, state.canvas.clientHeight);
		state.ctx.fillStyle = "#17202a";
		state.ctx.fillRect(0, 0, state.canvas.clientWidth, state.canvas.clientHeight);
		drawTiles();
		drawExistingPolygons();
		drawTrace();
	}

	function syncInputs() {
		$("#satellite-latitude").value = state.latitude.toFixed(7);
		$("#satellite-longitude").value = state.longitude.toFixed(7);
	}

	function setCenter(latitude, longitude) {
		state.latitude = clamp(Number(latitude), -85, 85);
		state.longitude = clamp(Number(longitude), -180, 180);
		syncInputs();
		draw();
	}

	function zoom(delta) {
		state.zoom = clamp(state.zoom + delta, 1, 20);
		draw();
	}

	function open() {
		if (!manualEditor.currentCase()) {
			manualEditor.setSaveStatus("Create or select an instance first.");
			return;
		}
		const view = manualEditor.currentCase().map_view;
		if (view) {
			state.latitude = view.latitude;
			state.longitude = view.longitude;
			state.zoom = view.zoom;
		}
		state.points = [];
		syncInputs();
		$("#satellite-modal").classList.remove("is-hidden");
		requestAnimationFrame(resize);
	}

	function close() {
		$("#satellite-modal").classList.add("is-hidden");
	}

	function finish() {
		if (state.points.length < 3) {
			$("#satellite-status").textContent = "Choose at least three corners before finishing.";
			return;
		}
		const current = manualEditor.currentCase();
		const view = current.map_view || { latitude: state.latitude, longitude: state.longitude, zoom: state.zoom };
		const origin = mercatorMeters(view.latitude, view.longitude);
		current.map_view = { ...view, zoom: state.zoom };
		current.polygons.push(state.points.map((point) => {
			const meters = mercatorMeters(point.latitude, point.longitude);
			return [meters[0] - origin[0], meters[1] - origin[1]];
		}));
		state.points = [];
		manualEditor.changed();
		scheduleManualAutosave({ immediate: true });
		close();
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
		state.canvas.addEventListener("pointerdown", (event) => {
			state.drag = { x: event.clientX, y: event.clientY, center: centerPixel() };
			state.moved = false;
			state.canvas.setPointerCapture(event.pointerId);
		});
		state.canvas.addEventListener("pointermove", (event) => {
			if (!state.drag) return;
			const dx = event.clientX - state.drag.x;
			const dy = event.clientY - state.drag.y;
			if (Math.hypot(dx, dy) > 3) state.moved = true;
			const location = pixelLocation(state.drag.center.x - dx, state.drag.center.y - dy, state.zoom);
			state.latitude = location.latitude;
			state.longitude = location.longitude;
			syncInputs();
			draw();
		});
		state.canvas.addEventListener("pointerup", (event) => {
			if (!state.moved) {
				const rect = state.canvas.getBoundingClientRect();
				state.points.push(screenToLocation(event.clientX - rect.left, event.clientY - rect.top));
				draw();
			}
			state.drag = null;
		});
		state.canvas.addEventListener("wheel", (event) => {
			event.preventDefault();
			zoom(event.deltaY < 0 ? 1 : -1);
		}, { passive: false });
		new ResizeObserver(resize).observe(state.canvas);
	}

	return { init, open, close };
}
