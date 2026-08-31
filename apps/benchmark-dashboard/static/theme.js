import { $ } from "./dom.js";
import { THEME_STORAGE_KEY } from "./storage.js";

export function applyTheme(theme) {
	const dark = theme === "dark";
	document.documentElement.dataset.theme = dark ? "dark" : "light";
	const button = $("#theme-toggle");
	if (button) {
		button.innerHTML = `${themeIcon(dark)}<span>${dark ? "Light" : "Dark"}</span>`;
		button.setAttribute("aria-pressed", dark ? "true" : "false");
		button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
	}
	localStorage.setItem(THEME_STORAGE_KEY, dark ? "dark" : "light");
}

export function toggleTheme() {
	applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
}

function themeIcon(dark) {
	return dark
		? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5a.6.6 0 01.6.6v1.2a.6.6 0 11-1.2 0V2.1a.6.6 0 01.6-.6zm4.6 2.5a.6.6 0 010 .8l-.8.8a.6.6 0 11-.8-.8l.8-.8a.6.6 0 01.8 0zM8 5.2A2.8 2.8 0 108 10.8 2.8 2.8 0 008 5.2zm6.5 2.8a.6.6 0 01-.6.6h-1.2a.6.6 0 010-1.2h1.2a.6.6 0 01.6.6zM12.6 12a.6.6 0 01-.8 0l-.8-.8a.6.6 0 11.8-.8l.8.8a.6.6 0 010 .8zM8 12.1a.6.6 0 01.6.6v1.2a.6.6 0 11-1.2 0v-1.2a.6.6 0 01.6-.6zM5 10.4a.6.6 0 010 .8l-.8.8a.6.6 0 11-.8-.8l.8-.8a.6.6 0 01.8 0zM3.9 8a.6.6 0 01-.6.6H2.1a.6.6 0 110-1.2h1.2a.6.6 0 01.6.6zM5 4.8a.6.6 0 01-.8.8l-.8-.8a.6.6 0 11.8-.8l.8.8z"></path></svg>'
		: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M9.7 1.5a6.1 6.1 0 104.8 8.8.6.6 0 00-.8-.8A4.4 4.4 0 016.5 4.4c0-.7.2-1.5.5-2.1a.6.6 0 00-.7-.8z"></path></svg>';
}
