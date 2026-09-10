export const $ = (selector) => document.querySelector(selector);

export function setOutput(target, text) {
	if (!target) {
		return;
	}
	const next = text || "";
	if (target.textContent === next) return;
	const selection = window.getSelection?.();
	if (selection && !selection.isCollapsed && selection.rangeCount > 0 && target.contains(selection.anchorNode)) {
		target.dataset.pendingOutput = next;
		return;
	}
	target.textContent = target.dataset.pendingOutput || next;
	delete target.dataset.pendingOutput;
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
