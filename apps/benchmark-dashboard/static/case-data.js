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
		background: data?.background ? JSON.parse(JSON.stringify(data.background)) : null,
		map_view: data?.map_view ? { ...data.map_view } : null,
	};
}

export function emptyCaseData() {
	return {
		name: "",
		generated: false,
		start: [0, 0],
		target: [1, 0],
		polygons: [],
		background: null,
		map_view: null,
	};
}

export function casePayload(data) {
	const clone = cloneCaseData(data);
	clone.polygons = clone.polygons.filter((polygon) => polygon.length >= 3);
	return clone;
}
