export const $ = (selector) => document.querySelector(selector);

export function setOutput(target, text) {
	if (!target) {
		return;
	}
	target.textContent = text || "";
}

export function escapeHTML(value) {
	return String(value ?? "").replace(/[&<>"']/g, (character) => ({
		"&": "&amp;",
		"<": "&lt;",
		">": "&gt;",
		'"': "&quot;",
		"'": "&#39;",
	})[character]);
}
