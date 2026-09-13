function pathLength(path) {
	let length = 0;
	for (let index = 1; index < path.length; index += 1) {
		length += Math.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]);
	}
	return length;
}

export async function solveFixedOrder(caseData, signal) {
	const response = await fetch("/api/editor/solve", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			start: caseData.start,
			target: caseData.target,
			polygons: caseData.polygons.filter(polygon => polygon.length >= 3),
			visit_order: "fixed",
		}),
		signal,
	});
	const result = await response.json();
	if (!response.ok) {
		const detail = Array.isArray(result.detail) ? result.detail.map(item => item.msg).join("; ") : result.detail;
		throw new Error(detail || "Fixed-order solver failed.");
	}
	return { ...result, length: pathLength(result.path), source: "server" };
}
