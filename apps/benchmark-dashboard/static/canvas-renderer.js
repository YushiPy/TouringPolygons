export function drawCanvasScene(renderer) {
	if (!renderer.canvas || !renderer.ctx) {
		return;
	}
	renderer.drawGrid();
	const current = renderer.currentCase();
	if (!current) {
		renderer.setStatus("Create or select an editable campaign.");
		return;
	}
	const ctx = renderer.ctx;
	if (renderer.layers.solution && renderer.solutionPath && renderer.solutionPath.length >= 2) {
		ctx.save();
		ctx.strokeStyle = "#facc15";
		ctx.lineWidth = 4;
		ctx.globalAlpha = renderer.solutionStale ? 0.45 : 1;
		if (renderer.solutionStale) {
			ctx.setLineDash([9, 6]);
		}
		ctx.beginPath();
		renderer.solutionPath.forEach((point, index) => {
			const canvasPoint = renderer.worldToCanvas(point);
			if (index === 0) {
				ctx.moveTo(canvasPoint.x, canvasPoint.y);
			} else {
				ctx.lineTo(canvasPoint.x, canvasPoint.y);
			}
		});
		ctx.stroke();
		ctx.restore();
	}
	current.polygons.forEach((polygon, index) => {
		if (polygon.length === 0) {
			return;
		}
		const color = ["#38bdf8", "#a3e635", "#f97316", "#f472b6", "#c084fc"][index % 5];
		ctx.beginPath();
		polygon.forEach((point, pointIndex) => {
			const canvasPoint = renderer.worldToCanvas(point);
			if (pointIndex === 0) {
				ctx.moveTo(canvasPoint.x, canvasPoint.y);
			} else {
				ctx.lineTo(canvasPoint.x, canvasPoint.y);
			}
		});
		if (polygon.length >= 3 && index !== renderer.activePolygon) {
			ctx.closePath();
			ctx.fillStyle = `${color}33`;
			ctx.fill();
		}
		ctx.strokeStyle = color;
		ctx.lineWidth = 2;
		ctx.stroke();
		if (renderer.layers.decomposition && polygon.length >= 3) {
			renderer.drawDecomposition(polygon, color);
		}
		polygon.forEach((point) => renderer.drawPoint(point, color));
		if (index === renderer.activePolygon && polygon.length > 0) {
			const last = renderer.worldToCanvas(polygon.at(-1));
			const preview = renderer.worldToCanvas(renderer.snap(renderer.canvasToWorld(renderer.mouseCanvas.x, renderer.mouseCanvas.y)));
			ctx.strokeStyle = color;
			ctx.setLineDash([7, 5]);
			ctx.beginPath();
			ctx.moveTo(last.x, last.y);
			ctx.lineTo(preview.x, preview.y);
			ctx.stroke();
			ctx.setLineDash([]);
			ctx.fillStyle = renderer.isClosingPolygon(renderer.mouseCanvas.x, renderer.mouseCanvas.y) ? "#facc15" : "#f8fafc";
			ctx.beginPath();
			ctx.arc(preview.x, preview.y, 4, 0, Math.PI * 2);
			ctx.fill();
		}
	});
	for (const selected of renderer.selectedPoints) {
		renderer.drawSelectedRing(selected.point);
		if (selected.kind === "vertex") {
			renderer.drawPoint(selected.point, "#f8fafc");
		}
	}
	if (renderer.layers.solution && renderer.solutionPath && renderer.solutionPath.length > 2) {
		renderer.solutionPath.slice(1, -1).forEach((point) => renderer.drawPoint(point, "#f97316"));
	}
	if (renderer.selectionRect) {
		const left = Math.min(renderer.selectionRect.start.x, renderer.selectionRect.end.x);
		const top = Math.min(renderer.selectionRect.start.y, renderer.selectionRect.end.y);
		const width = Math.abs(renderer.selectionRect.start.x - renderer.selectionRect.end.x);
		const height = Math.abs(renderer.selectionRect.start.y - renderer.selectionRect.end.y);
		ctx.fillStyle = "rgba(37, 99, 235, 0.18)";
		ctx.strokeStyle = "#93c5fd";
		ctx.setLineDash([6, 4]);
		ctx.fillRect(left, top, width, height);
		ctx.strokeRect(left, top, width, height);
		ctx.setLineDash([]);
	}
	renderer.drawPoint(current.start, "#22c55e", renderer.layers.labels ? "s" : "", renderer.labelDirections.start);
	renderer.drawPoint(current.target, "#ef4444", renderer.layers.labels ? "t" : "", renderer.labelDirections.target);
}
