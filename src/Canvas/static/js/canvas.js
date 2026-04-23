import { Vector2 } from "./vector2.js";
import { tppSolve } from "./tpp.js";
import { convexPartition } from "./convex-partition.js";
import * as settings from "./settings.js";

const floatToString = (integerPart, exponent) => {
	if (Math.abs(exponent) >= 5) {
		return `${integerPart}e${exponent}`;
	}
	return (integerPart * Math.pow(10, exponent))
		.toFixed(6)
		.replace(/\.?0+$/, "");
};

class Canvas {

	constructor(cameraPos, unitsToPixels) {

		if (!cameraPos) cameraPos = settings.INITIAL_CAMERA_POSITION;
		if (!unitsToPixels) unitsToPixels = settings.INITIAL_UNITS_TO_PIXELS;

		this.camera = new Camera(cameraPos, unitsToPixels);

		const canvas = document.getElementById(settings.CANVAS_ELEMENT_ID);
		const ctx = canvas.getContext("2d");

		const resizeObserver = new ResizeObserver(() => {
			const dpr = window.devicePixelRatio || 1;
			canvas.width = canvas.offsetWidth * dpr;
			canvas.height = canvas.offsetHeight * dpr;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			this.canvasCenter = new Vector2(canvas.offsetWidth / 2, canvas.offsetHeight / 2);
		});

		resizeObserver.observe(canvas);

		this.canvas = canvas;
		this.ctx = ctx;
		this.canvasCenter = new Vector2(canvas.offsetWidth / 2, canvas.offsetHeight / 2);
	}

	canvasToWorld(x, y) {
		if (x instanceof Vector2) { y = x.y; x = x.x; }
		return new Vector2(
			(x - this.canvasCenter.x) * this.camera.pixelsToUnits + this.camera.x,
			-(y - this.canvasCenter.y) * this.camera.pixelsToUnits + this.camera.y
		);
	}

	worldToCanvas(x, y = null) {

		if (x instanceof Object) { y = x.y; x = x.x; }

		return new Vector2(
			(x - this.camera.x) * this.camera.unitsToPixels + this.canvasCenter.x,
			-(y - this.camera.y) * this.camera.unitsToPixels + this.canvasCenter.y
		);
	}

	_drawLine(startCanvas, endCanvas, color, lineWidth = 1) {
		const ctx = this.ctx;
		ctx.strokeStyle = color;
		ctx.lineWidth = lineWidth;
		ctx.beginPath();
		ctx.moveTo(startCanvas.x, startCanvas.y);
		ctx.lineTo(endCanvas.x, endCanvas.y);
		ctx.stroke();
	}

	_drawLineGlow(startCanvas, endCanvas, color, lineWidth = 1) {
		const ctx = this.ctx;
		ctx.save();
		ctx.shadowColor = color;
		ctx.shadowBlur = lineWidth * 20;
		for (let i = 0; i < 3; i++) {
			this._drawLine(startCanvas, endCanvas, color, lineWidth);
		}
		ctx.restore();
	}

	_drawLineDashed(startCanvas, endCanvas, color, lineWidth = 1, dashLength = 5, glow = false) {
		const ctx = this.ctx;
		if (glow) {
			ctx.save();
			ctx.shadowColor = color;
			ctx.shadowBlur = lineWidth * 10;
			for (let i = 0; i < 3; i++) {
				this._drawLineDashed(startCanvas, endCanvas, color, lineWidth, dashLength, false);
			}
			ctx.restore();
		} else {
			ctx.save();
			ctx.setLineDash([dashLength, dashLength]);
			this._drawLine(startCanvas, endCanvas, color, lineWidth);
			ctx.restore();
		}
	}

	drawLine(startCanvas, endCanvas, color, lineWidth = 1, dashLength = 0, glow = false) {
		if (dashLength > 0) {
			this._drawLineDashed(startCanvas, endCanvas, color, lineWidth, dashLength, glow);
		} else if (glow) {
			this._drawLineGlow(startCanvas, endCanvas, color, lineWidth);
		} else {
			this._drawLine(startCanvas, endCanvas, color, lineWidth);
		}
	}

	drawLineWorld(startWorld, endWorld, color, lineWidth = 1, dashLength = 0, glow = false) {
		this.drawLine(
			this.worldToCanvas(startWorld),
			this.worldToCanvas(endWorld),
			color, lineWidth, dashLength, glow
		);
	}

	drawPoint(canvasPos, color, radius = 5, glow = false) {
		const ctx = this.ctx;

		ctx.save();
		ctx.fillStyle = color;

		if (glow) {
			ctx.shadowColor = color;
			ctx.shadowBlur = radius * 5;
		}

		ctx.beginPath();
		ctx.arc(canvasPos.x, canvasPos.y, radius, 0, 2 * Math.PI);
		ctx.fill();

		ctx.restore();
	}

	drawPointWorld(worldPos, color, radius, glow = false) {
		this.drawPoint(this.worldToCanvas(worldPos), color, radius, glow);
	}

	drawPolygon(pointsWorld, color, lineWidth = 1, glow = false, dashLength = 0, alpha = 0.3) {

		const points = pointsWorld.map(p => this.worldToCanvas(p));

		if (points.length < 2) return;

		const ctx = this.ctx;

		// Split polygon into convex parts:
		const partition = convexPartition(points);

		ctx.save();
		ctx.fillStyle = color;
		
		// Draw partition edges dashed
		for (const part of partition) {
			for (let i = 0; i < part.length; i++) {
				this.drawLine(part[i], part[(i + 1) % part.length], color, lineWidth, dashLength + 10, glow);
			}
		}

		// Draw original polygon edges solid
		for (let i = 0; i < points.length; i++) {
			this.drawLine(points[i], points[(i + 1) % points.length], color, lineWidth, dashLength, glow);
		}

		ctx.globalAlpha = alpha;

		// Fill in
		for (const part of partition) {

			ctx.beginPath();
			ctx.moveTo(part[0].x, part[0].y);

			for (let i = 1; i < part.length; i++) {
				ctx.lineTo(part[i].x, part[i].y);
			}

			ctx.closePath();
			ctx.fill();
		}

		ctx.restore();
	}

	clear() {
		const { canvas, ctx } = this;
		ctx.fillStyle = settings.BACKGROUND_COLOR;
		ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
	}

	get width() { return this.canvas.offsetWidth; }
	get height() { return this.canvas.offsetHeight; }
	get center() { return this.canvasCenter; }
}

class Polygon {

	constructor(points, color) {
		this.points = points.map(p => new Vector2(p));
		this.color = color;
	}

	*[Symbol.iterator]() {
		for (const point of this.points) yield point;
	}

	isConvex() {

		let gotNegative = false;
		let gotPositive = false;
		const n = this.points.length;

		for (let i = 0; i < n; i++) {

			const p0 = this.points[i];
			const p1 = this.points[(i + 1) % n];
			const p2 = this.points[(i + 2) % n];

			const cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);

			if (cross < 0) gotNegative = true;
			else if (cross > 0) gotPositive = true;

			if (gotNegative && gotPositive) return false;
		}

		return true;
	}

	containsPoint(point) {
		let inside = false;
		const pts = this.points;
		for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
			const xi = pts[i].x, yi = pts[i].y;
			const xj = pts[j].x, yj = pts[j].y;
			const intersect = ((yi > point.y) !== (yj > point.y)) &&
				(point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
			if (intersect) inside = !inside;
		}
		return inside;
	}
}

class Camera {

	constructor(position = { x: 0, y: 0 }, unitsToPixels = 1) {
		this.position = new Vector2(position);
		this.unitsToPixels = unitsToPixels;
	}

	get pixelsToUnits() { return 1 / this.unitsToPixels; }
	set pixelsToUnits(value) { this.unitsToPixels = 1 / value; }

	get x() { return this.position.x; }
	set x(value) { this.position.x = value; }

	get y() { return this.position.y; }
	set y(value) { this.position.y = value; }
}

class Scene {

	constructor(data = null) {
		this.loadFromData(data);
	}

	// --- Coordinate helpers ---

	clampToCanvas(point) {
		return new Vector2(
			Math.min(Math.max(point.x, 0), this.canvas.width),
			Math.min(Math.max(point.y, 0), this.canvas.height)
		);
	}

	snapPoint(point) {
		if (!this.snapping) return point;
		const s = this.getSubgridSpacing();
		return new Vector2(
			Math.round(point.x / s) * s,
			Math.round(point.y / s) * s
		);
	}

	movePoint(point, movement) {
		const snapped = this.snapPoint(new Vector2(point.x + movement.x, point.y + movement.y));
		point.x = snapped.x;
		point.y = snapped.y;
	}

	getSubgridSpacing() {
		const decisionValue = settings.MINIMUM_GRID_SPACING / this.canvas.camera.unitsToPixels;
		let exponent = Math.ceil(Math.log10(decisionValue)) | 0;
		let multiplier = 1;
		let subGridCount = 4;
		const gridScale = Math.pow(10, exponent);
		if (gridScale / 5 > decisionValue) { subGridCount = 3; exponent--; multiplier = 2; }
		else if (gridScale / 2 > decisionValue) { exponent--; multiplier = 5; }
		return Math.pow(10, exponent) * multiplier / (subGridCount + 1);
	}

	changeZoom(scale, fixedCanvasPoint) {
		const camera = this.canvas.camera;
		const fixedWorld = this.canvas.canvasToWorld(fixedCanvasPoint);
		camera.position.x = fixedWorld.x - (fixedWorld.x - camera.x) / scale;
		camera.position.y = fixedWorld.y - (fixedWorld.y - camera.y) / scale;
		camera.unitsToPixels *= scale;
	}

	// --- Hit testing ---

	findDraggablePoint(canvasX, canvasY, candidates = null) {
		if (candidates === null) {
			candidates = [this.startPoint, this.targetPoint];
			for (const poly of this.polygons) candidates.push(...poly.points);
		}
		for (const candidate of candidates) {
			const cp = this.canvas.worldToCanvas(candidate);
			if (Math.hypot(cp.x - canvasX, cp.y - canvasY) <= settings.HIT_RADIUS) return candidate;
		}
		return null;
	}

	findDraggablePolygon(canvasX, canvasY) {
		const world = this.canvas.canvasToWorld(canvasX, canvasY);
		for (let i = 0; i < this.polygons.length; i++) {
			if (this.polygons[i].containsPoint(world)) return i;
		}
		return -1;
	}

	dragObjects(mousePosition) {
		if (!this.dragging) return;
		const { referencePoint, pointsDragged } = this.dragging;
		const clamped = this.clampToCanvas(mousePosition);
		const world = this.canvas.canvasToWorld(clamped);
		const movement = new Vector2(world.x - referencePoint.x, world.y - referencePoint.y);
		this.movePoint(referencePoint, movement);
		for (const point of pointsDragged) {
			if (point !== referencePoint) this.movePoint(point, movement);
		}
	}

	// --- Selection ---

	updateSelectionRect(start, end) {
		if (!this.selectionRect) {
			this.selectionRect = { start: start ?? new Vector2(0, 0), end: end ?? new Vector2(0, 0) };
		} else {
			if (start) this.selectionRect.start = start.clone();
			if (end) this.selectionRect.end = end.clone();
		}
		this._findSelectedPoints();
	}

	unselectRect() {
		for (const p of this.selectedPoints) this.selectedPointsTotal.add(p);
		this.selectionRect = null;
		this.selectedPoints = [];
	}

	_findSelectedPoints() {
		this.selectedPoints = [];
		const candidates = [this.startPoint, this.targetPoint];
		for (const poly of this.polygons) candidates.push(...poly.points);

		const { start, end } = this.selectionRect;
		const left = Math.min(start.x, end.x);
		const right = Math.max(start.x, end.x);
		const top = Math.min(start.y, end.y);
		const bottom = Math.max(start.y, end.y);

		for (const candidate of candidates) {
			const cp = this.canvas.worldToCanvas(candidate);
			if (cp.x >= left && cp.x <= right && cp.y >= top && cp.y <= bottom) {
				this.selectedPoints.push(candidate);
			}
		}
	}

	// --- Drawing ---

	drawGrid() {
		const { canvas, camera, ctx } = this.canvas;
		const { position: cameraCenter, unitsToPixels } = camera;

		const decisionValue = settings.MINIMUM_GRID_SPACING / unitsToPixels;
		let exponent = Math.ceil(Math.log10(decisionValue)) | 0;
		let multiplier = 1;
		let subGridCount = 4;

		const gridScale = Math.pow(10, exponent);
		if (gridScale / 5 > decisionValue) { subGridCount = 3; exponent--; multiplier = 2; }
		else if (gridScale / 2 > decisionValue) { exponent--; multiplier = 5; }

		const gridSpacing = Math.pow(10, exponent) * multiplier;
		const halfWidth = canvas.offsetWidth / 2 / unitsToPixels;
		const halfHeight = canvas.offsetHeight / 2 / unitsToPixels;

		const bounds = {
			left: cameraCenter.x - halfWidth,
			right: cameraCenter.x + halfWidth,
			bottom: cameraCenter.y - halfHeight,
			top: cameraCenter.y + halfHeight,
		};

		const origin = this.canvas.worldToCanvas(0, 0);
		const { GRID_NUMBER_FONT: font, GRID_NUMBER_COLOR: color, GRID_NUMBER_LIGHT_COLOR: lightColor } = settings;

		const clampTextAnchor = (raw, lo, hi) => {
			if (raw < lo) return { pos: lo, dimmed: true };
			if (raw > hi) return { pos: hi, dimmed: true };
			return { pos: raw, dimmed: false };
		};

		const xAnchor = clampTextAnchor(origin.x - 8, -1, canvas.offsetWidth - 8);
		const yAnchor = clampTextAnchor(origin.y + 3, 0, canvas.offsetHeight - 20);
		const pow = (e) => Math.pow(10, e);

		const drawAxis = (horizontal) => {
			const [rangeStart, count, lo, hi] = horizontal
				? [Math.floor((cameraCenter.y - halfHeight) / gridSpacing) * multiplier, Math.ceil(halfHeight * 2 / gridSpacing), bounds.left, bounds.right]
				: [Math.floor((cameraCenter.x - halfWidth) / gridSpacing) * multiplier, Math.ceil(halfWidth * 2 / gridSpacing), bounds.bottom, bounds.top];

			for (let i = 0; i <= count; i++) {
				const integerPart = rangeStart + i * multiplier;
				const world = integerPart * pow(exponent);

				const a = horizontal ? new Vector2(lo, world) : new Vector2(world, lo);
				const b = horizontal ? new Vector2(hi, world) : new Vector2(world, hi);
				this.canvas.drawLineWorld(a, b, settings.GRID_COLOR, settings.GRID_WIDTH);

				for (let j = 0; j < subGridCount; j++) {
					const sub = world + gridSpacing * (j + 1) / (subGridCount + 1);
					const sa = horizontal ? new Vector2(lo, sub) : new Vector2(sub, lo);
					const sb = horizontal ? new Vector2(hi, sub) : new Vector2(sub, hi);
					this.canvas.drawLineWorld(sa, sb, settings.SUB_GRID_COLOR, settings.SUB_GRID_WIDTH);
				}

				if (Math.abs(world) <= 1e-12) continue;

				const anchor = horizontal ? xAnchor : yAnchor;
				const screenPos = horizontal
					? this.canvas.worldToCanvas(0, world).y
					: this.canvas.worldToCanvas(world, 0).x;

				ctx.font = font;
				ctx.fillStyle = anchor.dimmed ? lightColor : color;

				if (horizontal) {
					ctx.textAlign = anchor.pos === -1 ? "left" : "right";
					ctx.textBaseline = "middle";
					ctx.fillText(floatToString(integerPart, exponent), anchor.pos === -1 ? 10 : anchor.pos, screenPos);
				} else {
					ctx.textAlign = "center";
					ctx.textBaseline = "top";
					ctx.fillText(floatToString(integerPart, exponent), screenPos, anchor.pos);
				}
			}
		};

		drawAxis(false);
		drawAxis(true);

		ctx.fillStyle = color;
		ctx.font = font;
		ctx.textAlign = "right";
		ctx.textBaseline = "top";
		ctx.fillText("0", origin.x - 8, origin.y + 3);

		this.canvas.drawLineWorld(new Vector2(bounds.left, 0), new Vector2(bounds.right, 0), settings.MAIN_AXIS_COLOR, settings.MAIN_AXIS_WIDTH);
		this.canvas.drawLineWorld(new Vector2(0, bounds.bottom), new Vector2(0, bounds.top), settings.MAIN_AXIS_COLOR, settings.MAIN_AXIS_WIDTH);
	}

	drawSolution() {

		const start = [this.startPoint.x, this.startPoint.y];
		const target = [this.targetPoint.x, this.targetPoint.y];
		const polys = this.polygons.map(poly => poly.points.map(v => [v.x, v.y]));

		let path;

		try {
			path = tppSolve(start, target, polys, true);
		} catch (e) {
			return;
		}

		for (let i = 0; i < path.length - 1; i++) {
			const p1 = new Vector2(path[i].x, path[i].y);
			const p2 = new Vector2(path[i + 1].x, path[i + 1].y);
			this.canvas.drawLineWorld(p1, p2, settings.SOLUTION_COLOR, 3);
			this.canvas.drawPointWorld(p1, settings.SOLUTION_COLOR, 6);
		}
	}

	drawPolygons() {

		for (let i = 0; i < this.polygons.length; i++) {

			const poly = this.polygons[i];
			const color = poly.color || settings.POLYGON_COLORS[i % settings.POLYGON_COLORS.length];
			const isSelected = this.currentPolygon % this.polygons.length === i;

			for (const vertex of poly.points) {
				this.canvas.drawPointWorld(vertex, color, settings.POINT_RADIUS * 0.6, isSelected);
			}

			const alpha = settings.POLYGON_INSIDE_ALPHA;
			this.canvas.drawPolygon(poly.points, color, 2, isSelected, 0, alpha);
		}
	}

	drawVertexLine() {

		if (!this.showVertexLine || this.polygons.length === 0) return;

		const poly = this.polygons[this.currentPolygon % this.polygons.length];
		const n = poly.points.length;
		const v1 = poly.points[((this.currentPolygonVertex - 1) % n + n) % n];
		const v2 = poly.points[this.currentPolygonVertex % n];
		const mouseWorld = this.canvas.canvasToWorld(this.mouseLocation);
		const color = poly.color || settings.POLYGON_COLORS[this.currentPolygon % settings.POLYGON_COLORS.length];

		this.canvas.drawLineWorld(new Vector2(v1.x, v1.y), mouseWorld, color, 2, 0, true);
		this.canvas.drawLineWorld(new Vector2(v2.x, v2.y), mouseWorld, color, 2, 0, true);
	}

	drawSelectionRect() {
		if (!this.selectionRect) return;
		const ctx = this.canvas.ctx;
		const { start, end } = this.selectionRect;
		const left = Math.min(start.x, end.x);
		const top = Math.min(start.y, end.y);
		const width = Math.abs(start.x - end.x);
		const height = Math.abs(start.y - end.y);

		ctx.strokeStyle = settings.MAIN_AXIS_COLOR;
		ctx.lineWidth = 1.5;
		ctx.setLineDash([5, 3]);
		ctx.strokeRect(left, top, width, height);
		ctx.fillStyle = settings.MAIN_AXIS_COLOR + "66";
		ctx.fillRect(left, top, width, height);
		ctx.setLineDash([]);
	}

	drawSelectedPoints() {
		const ctx = this.canvas.ctx;
		const r = settings.POINT_RADIUS * 1.5;
		const speed = 0.002;
		const angle = performance.now() * speed;

		for (const point of [...this.selectedPoints, ...this.selectedPointsTotal]) {
			const cp = this.canvas.worldToCanvas(point);
			ctx.beginPath();
			ctx.arc(cp.x, cp.y, r, angle, angle + Math.PI * 2);
			ctx.strokeStyle = settings.MAIN_AXIS_COLOR;
			ctx.lineWidth = 2;
			ctx.setLineDash([5, 3]);
			ctx.stroke();
			ctx.setLineDash([]);
		}
	}

	draw() {

		// Fill background
		this.canvas.ctx.fillStyle = settings.BACKGROUND_COLOR;
		this.canvas.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

		this.canvas.clear();
		this.drawGrid();
		this.drawSolution();
		this.drawPolygons();
		this.drawVertexLine();
		this.drawSelectionRect();
		this.drawSelectedPoints();
		this.canvas.drawPointWorld(this.startPoint, settings.START_POINT_COLOR, settings.POINT_RADIUS, true);
		this.canvas.drawPointWorld(this.targetPoint, settings.TARGET_POINT_COLOR, settings.POINT_RADIUS, true);

		if (this.selectionRect) this.updateSelectionRect(null, this.mouseLocation);
	}

	// --- Input ---
	_isOverlayUp() {

		for (const id of settings.OVERLAY_ELEMENTS_ID) {

			const overlay = document.getElementById(id);

			if (!overlay) continue;

			const opacity = overlay.style.opacity;

			if (opacity === 0 || opacity === "0") continue;
			if (overlay.innerHTML.trim() !== "") return true;
		}

		return false;
	}

	_isBlockClickActive() {

		for (const id of settings.BLOCK_CLICK_IDS) {
			const element = document.getElementById(id);

			if (element && element.style.opacity != 0 && element.innerHTML.trim() !== "") {
				return true;
			}
		}

		return false;
	}

	_initInput() {

		// Remove old listeners first
		this._removeInput();

		const canvas = this.canvas.canvas;

		this._boundMouseDown = (e) => this._onMouseDown(e);
		this._boundMouseUp = (e) => this._onMouseUp(e);
		this._boundMouseMove = (e) => this._onMouseMove(e);
		this._boundKeyDown = (e) => this._onKeyDown(e);
		this._boundKeyUp = (e) => this._onKeyUp(e);
		this._boundWheel = (e) => {
			e.preventDefault();
			this.changeZoom(1 - e.deltaY * this.scrollSensitivity, this.mouseLocation);
		};

		window.addEventListener("blur", () => { this.mouseHeld = false; this.dragging = null; });
		document.addEventListener("mouseleave", () => { this.mouseHeld = false; this.dragging = null; });
		canvas.addEventListener("mousedown", this._boundMouseDown);
		document.addEventListener("mouseup", this._boundMouseUp);
		document.addEventListener("mousemove", this._boundMouseMove);
		document.addEventListener("keydown", this._boundKeyDown);
		document.addEventListener("keyup", this._boundKeyUp);
		canvas.addEventListener("wheel", this._boundWheel, { passive: false });

		this._boundOnSnapping = () => this._updateSnapping(!this.snapping);
		this._boundOnTriangle = () => this._onTriangle();
		this._boundOnVertexLine = () => this._updateVertexLine(!this.showVertexLine);

		this.snapButton.addEventListener("click", this._boundOnSnapping);
		this.triangleButton.addEventListener("click", this._boundOnTriangle);
		this.vertexLineButton.addEventListener("click", this._boundOnVertexLine);
	}

	_removeInput() {
		if (!this._boundMouseDown) return;

		document.removeEventListener("mousedown", this._boundMouseDown);
		document.removeEventListener("mouseup", this._boundMouseUp);
		document.removeEventListener("mousemove", this._boundMouseMove);
		document.removeEventListener("keydown", this._boundKeyDown);
		document.removeEventListener("keyup", this._boundKeyUp);
		this.canvas.canvas.removeEventListener("wheel", this._boundWheel);

		this.snapButton.removeEventListener("click", this._boundOnSnapping);
		this.triangleButton.removeEventListener("click", this._boundOnTriangle);
		this.vertexLineButton.removeEventListener("click", this._boundOnVertexLine);
	}

	_updateSnapping(value) {

		this.snapping = value;

		if (this.snapping) {
			this.snapButton.classList.add("active");
		} else {
			this.snapButton.classList.remove("active");
		}
	}

	_onTriangle() {

		const radius = this.canvas.camera.pixelsToUnits * 100;
		const center = this.canvas.canvasToWorld(this.canvas.width / 2, this.canvas.height / 2);

		const points = [0, 1, 2].map(i => {
			const angle = i * 2 * Math.PI / 3;
			return new Vector2(Math.cos(angle), Math.sin(angle)).mul(radius).add(center);
		});

		const color = settings.POLYGON_COLORS[this.polygons.length % settings.POLYGON_COLORS.length];

		this.polygons.push(new Polygon(points, color));
		this.currentPolygon = this.polygons.length - 1;
	}

	_updateVertexLine(value) {

		this.showVertexLine = value;

		if (this.showVertexLine) {
			this.vertexLineButton.classList.add("active");
		} else {
			this.vertexLineButton.classList.remove("active");
		}
	}

	_getCanvasPos(e) {
		const bounds = this.canvas.canvas.getBoundingClientRect();
		return new Vector2(e.clientX - bounds.left, e.clientY - bounds.top);
	}

	_isInCanvas(pos) {
		return pos.x >= 0 && pos.x <= this.canvas.width && pos.y >= 0 && pos.y <= this.canvas.height;
	}

	_onMouseDown(e) {

		if (this._isOverlayUp()) return;

		this.changedPolygon = false;

		const pos = this._getCanvasPos(e);

		this.lastClickTime = performance.now();
		this.lastClickPosition = pos;

		const selectionPoint = this.findDraggablePoint(pos.x, pos.y, [...this.selectedPointsTotal]);
		if (selectionPoint) {
			this.dragging = { referencePoint: selectionPoint.clone(), pointsDragged: [...this.selectedPointsTotal] };
			return;
		}

		const point = this.findDraggablePoint(pos.x, pos.y);
		const polyIndex = this.findDraggablePolygon(pos.x, pos.y);

		if (point) {
			this.dragging = { referencePoint: point.clone(), pointsDragged: [point] };
			this.changedPolygon = true;
			return;
		}

		if (polyIndex !== -1) {
			this.dragging = {
				referencePoint: this.canvas.canvasToWorld(pos),
				pointsDragged: this.polygons[polyIndex].points,
			};
			this.changedPolygon = polyIndex !== this.currentPolygon % this.polygons.length;
			this.currentPolygon = polyIndex;
			return;
		}

		if (this._isInCanvas(pos)) this.isDraggingCanvas = true;
	}

	_onMouseUp(e) {

		if (this._isOverlayUp()) return;

		this.isDraggingCanvas = false;

		const pos = this._getCanvasPos(e);
		if (!this._isInCanvas(pos)) return;

		const isRecent = performance.now() - this.lastClickTime < 300;
		const isClose = this.mouseLocation.distanceTo(this.lastClickPosition) < settings.HIT_RADIUS;
		const isSamePolygon = !this.changedPolygon;
		
		if (isRecent && isClose && isSamePolygon && this.polygons.length > 0 && !this._isBlockClickActive()) {
			const clamped = this.clampToCanvas(this.mouseLocation);
			const world = this.snapPoint(this.canvas.canvasToWorld(clamped));
			const poly = this.polygons[this.currentPolygon % this.polygons.length];
			poly.points.push(world);
		}

		if (this.dragging) {
			this.dragging = null;
			return;
		}
	}

	_onMouseMove(e) {

		if (this._isOverlayUp()) return;

		const bounds = this.canvas.canvas.getBoundingClientRect();
		this.mouseLocation = new Vector2(e.clientX - bounds.left, e.clientY - bounds.top);

		if (e.shiftKey && !this.selectionRect) {
			this.updateSelectionRect(this.mouseLocation, this.mouseLocation);
		} else if (!e.shiftKey && this.selectionRect) {
			this.unselectRect();
		}

		if (this.selectionRect) this.updateSelectionRect(null, this.mouseLocation);

		if (this.dragging) {
			this.dragObjects(this.mouseLocation);
			return;
		}

		const hovering = this.findDraggablePoint(this.mouseLocation.x, this.mouseLocation.y) ||
			this.findDraggablePolygon(this.mouseLocation.x, this.mouseLocation.y) !== -1;
		this.canvas.canvas.style.cursor = hovering ? "move" : "default";

		if (this.isDraggingCanvas) {
			this.canvas.camera.position.x -= e.movementX / this.canvas.camera.unitsToPixels;
			this.canvas.camera.position.y += e.movementY / this.canvas.camera.unitsToPixels;
		}
	}

	_onMouseWheel(e) {
		if (this._isOverlayUp()) return;

		e.preventDefault();
		this.changeZoom(1 - e.deltaY * this.scrollSensitivity, this.mouseLocation);
	}

	_onKeyDown(e) {

		if (this._isOverlayUp()) return;

		if (e.key === "Shift") {
			this.updateSelectionRect(this.mouseLocation, this.mouseLocation);
		}

		if (e.key === "ArrowUp") this.currentPolygon++;
		if (e.key === "ArrowDown") this.currentPolygon--;

		if (e.key === "Backspace" || e.key === "Delete" || e.key === "x") {
			this._deleteSelected();
		}

		if (e.key === "1") { this.snapButton.click(); }
		if (e.key === "2") { this.triangleButton.click(); }
		if (e.key === "3") { this.vertexLineButton.click(); }
	}

	_onKeyUp(e) {

		if (this._isOverlayUp()) return;

		if (e.key === "Shift") {

			this.unselectRect();

			const isRecent = this.lastShiftPressTime && (performance.now() - this.lastShiftPressTime < settings.DOUBLE_CLICK_TIME);
			const isClose = this.mouseLocation.distanceTo(this.lastShiftPosition) < settings.HIT_RADIUS;

			this.lastShiftPressTime = performance.now();
			this.lastShiftPosition = this.mouseLocation.clone();

			if (isRecent && isClose) {
				this.selectedPointsTotal = new Set();
				this.selectedPoints = [];
				this.selectionRect = null;
			}
		}
	}

	_deleteSelected() {
		if (this.selectedPointsTotal.size === 0) return;

		const lenBefore = this.polygons.length;
		this.polygons = this.polygons.filter(poly => !poly.points.every(v => this.selectedPointsTotal.has(v)));
		if (this.polygons.length !== lenBefore) this.currentPolygon = 0;

		for (const point of this.selectedPointsTotal) {
			for (const poly of this.polygons) {
				if (poly.points.length <= 3) continue;
				const idx = poly.points.indexOf(point);
				if (idx !== -1) { poly.points.splice(idx, 1); break; }
			}
		}

		this.selectedPointsTotal = new Set();
		this.selectedPoints = [];
	}

	// --- Save and load ---

	clear() {
		this.loadFromData(null);
	}

	saveData(drawingName) {

		const dataURL = scene.canvas.canvas.toDataURL("image/png");

		const data = {

			startPoint: [this.startPoint.x, this.startPoint.y],
			targetPoint: [this.targetPoint.x, this.targetPoint.y],
			polygons: this.polygons.map(poly => poly.points.map(p => [p.x, p.y])),
			polygonColors: this.polygons.map(poly => poly.color),

			currentPolygon: this.currentPolygon,
			currentPolygonVertex: this.currentPolygonVertex,
			scrollSensitivity: this.scrollSensitivity,
			snapping: this.snapping,
			showVertexLine: this.showVertexLine,

			camera: {
				position: [this.canvas.camera.x, this.canvas.camera.y],
				unitsToPixels: this.canvas.camera.unitsToPixels,
			},

			dataURL: dataURL,
			width: this.canvas.width,
			height: this.canvas.height,

			drawingName: drawingName,
		};

		return JSON.stringify(data);
	}

	async saveToDB(drawingName) {
		const data = this.saveData(drawingName);

		const response = await fetch(settings.SAVE_DRAWING_ENDPOINT_URL, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: data,
		});

		const json = await response.json();

		return json.id;
	}

	async updateDrawingData(drawingId, drawingName) {

		const data = this.saveData(drawingName);

		await fetch(`${settings.UPDATE_DRAWING_ENDPOINT_URL}/${drawingId}`, {
			method: "PUT",
			headers: { "Content-Type": "application/json" },
			body: data,
		});
	}

	loadFromData(data) {

		data = data || {
			startPoint: settings.INITIAL_START_POINT,
			targetPoint: settings.INITIAL_TARGET_POINT,
			polygons: settings.INITIAL_POLYGONS,
			polygonColors: settings.POLYGON_COLORS,

			currentPolygon: 0,
			currentPolygonVertex: 0,
			scrollSensitivity: settings.SCROLL_SENSITIVITY,

			snapping: null,
			showVertexLine: null,

			camera: {
				position: settings.INITIAL_CAMERA_POSITION,
				unitsToPixels: settings.INITIAL_UNITS_TO_PIXELS,
			},
		};

		this.startPoint = new Vector2(data.startPoint);
		this.targetPoint = new Vector2(data.targetPoint);

		const colors = data.polygonColors || settings.POLYGON_COLORS;
		this.polygons = data.polygons.map((poly, i) => new Polygon(poly, colors[i % colors.length]));

		this.currentPolygon = data.currentPolygon || 0;
		this.currentPolygonVertex = data.currentPolygonVertex || 0;
		this.scrollSensitivity = data.scrollSensitivity || settings.SCROLL_SENSITIVITY;

		this.snapButton = document.getElementById(settings.SNAP_BUTTON_ID);
		this.triangleButton = document.getElementById(settings.MAKE_TRIANGLE_BUTTON_ID);
		this.vertexLineButton = document.getElementById(settings.SHOW_VERTEX_LINE_BUTTON_ID);

		this.snapping = false;
		this.showVertexLine = false;

		this._updateSnapping(data.snapping === null ? this.snapButton.classList.contains("active") : data.snapping);
		this._updateVertexLine(data.showVertexLine === null ? this.vertexLineButton.classList.contains("active") : data.showVertexLine);

		const cameraData = data.camera || {};
		const cameraPos = cameraData.position || settings.INITIAL_CAMERA_POSITION;
		const unitsToPixels = cameraData.unitsToPixels || settings.INITIAL_UNITS_TO_PIXELS;

		this.canvas = new Canvas(cameraPos, unitsToPixels);

		this.canvas.camera.position = new Vector2(cameraPos);
		this.canvas.camera.unitsToPixels = unitsToPixels;

		this.mouseHeld = false;
		this.mouseLocation = new Vector2(0, 0);
		this.dragging = null;
		this.isDraggingCanvas = false;

		this.selectionRect = null;
		this.selectedPoints = [];
		this.selectedPointsTotal = new Set();

		this.lastClickTime = 0;
		this.lastClickPosition = new Vector2(0, 0);
		this.lastShiftPressTime = 0;
		this.lastShiftPosition = new Vector2(0, 0);

		this.changedPolygon = false;

		this._initInput();
	}
}

const scene = new Scene();
window.scene = scene; // Expose for exporting

function animate() {
	scene.draw();
	requestAnimationFrame(animate);
}

animate();
