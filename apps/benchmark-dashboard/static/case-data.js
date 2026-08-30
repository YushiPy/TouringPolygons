export function instanceLabel(index) {
	return Number(index) + 1;
}

export function cloneCaseData(data) {
	return {
		name: data?.name || "",
		generated: Boolean(data?.generated),
		start: [...(data?.start || [0, 0])],
		target: [...(data?.target || [1, 0])],
		polygons: (data?.polygons || []).map((polygon) => polygon.map((point) => [...point])),
	};
}

export function emptyCaseData() {
	return {
		name: "",
		generated: false,
		start: [0, 0],
		target: [1, 0],
		polygons: [],
	};
}

export function casePayload(data) {
	const clone = cloneCaseData(data);
	clone.polygons = clone.polygons.filter((polygon) => polygon.length >= 3);
	return clone;
}
