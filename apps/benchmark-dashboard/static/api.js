export async function requestJSON(url, options = {}) {
	const response = await fetch(url, {
		headers: { "Content-Type": "application/json" },
		...options,
	});
	const data = await response.json();
	if (!response.ok) {
		throw new Error(data.detail || data.output || `Request failed: ${response.status}`);
	}
	return data;
}
