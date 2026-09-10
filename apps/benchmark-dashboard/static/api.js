export async function requestJSON(url, options = {}) {
	const { timeoutMs = 30_000, ...fetchOptions } = options;
	const controller = fetchOptions.signal ? null : new AbortController();
	const timeout = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
	try {
		const response = await fetch(url, {
			headers: { "Content-Type": "application/json" },
			...fetchOptions,
			signal: fetchOptions.signal || controller?.signal,
		});
		const data = await response.json();
		if (!response.ok) {
			throw new Error(data.detail || data.output || `Request failed: ${response.status}`);
		}
		return data;
	} catch (error) {
		if (error.name === "AbortError") {
			throw new Error(`Request timed out: ${url}`);
		}
		throw error;
	} finally {
		if (timeout) clearTimeout(timeout);
	}
}
