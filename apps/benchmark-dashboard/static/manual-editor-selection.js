export function samePointSelection(left, right) {
	return left?.kind === right?.kind
		&& left?.polygonIndex === right?.polygonIndex
		&& left?.vertexIndex === right?.vertexIndex;
}

export function mergeSelections(base, selected) {
	const merged = [...base];
	for (const point of selected) {
		if (!merged.some((existing) => samePointSelection(existing, point))) {
			merged.push(point);
		}
	}
	return merged;
}

export function pointsInRect(caseData, rect, project) {
	if (!caseData) {
		return [];
	}
	const left = Math.min(rect.start.x, rect.end.x);
	const right = Math.max(rect.start.x, rect.end.x);
	const top = Math.min(rect.start.y, rect.end.y);
	const bottom = Math.max(rect.start.y, rect.end.y);
	const selected = [];
	const candidates = [
		{ kind: "start", point: caseData.start },
		{ kind: "target", point: caseData.target },
	];
	caseData.polygons.forEach((polygon, polygonIndex) => {
		polygon.forEach((point, vertexIndex) => {
			candidates.push({ kind: "vertex", point, polygonIndex, vertexIndex });
		});
	});
	for (const candidate of candidates) {
		const canvasPoint = project(candidate.point);
		if (canvasPoint.x >= left && canvasPoint.x <= right
			&& canvasPoint.y >= top && canvasPoint.y <= bottom) {
			selected.push(candidate);
		}
	}
	return selected;
}
