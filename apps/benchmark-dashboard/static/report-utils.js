import { escapeHTML } from "./dom.js";

export function metricCard(label, value) {
	return `<div class="metric-card"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`;
}

export function shortNumber(value) {
	const number = Number(value);
	if (!Number.isFinite(number)) {
		return value ?? "-";
	}
	if (Math.abs(number) >= 1000) {
		return number.toFixed(1);
	}
	return number.toFixed(3);
}

export function parseNumber(value) {
	const cleaned = String(value).replaceAll("_", "").replace("%", "").match(/-?\d+(?:\.\d+)?/);
	return cleaned ? Number(cleaned[0]) : null;
}

export function parseDurationSeconds(value) {
	const number = parseNumber(value);
	if (!Number.isFinite(number)) {
		return null;
	}
	const text = String(value).toLowerCase();
	if (text.includes("us") || text.includes("µs")) {
		return number / 1000000;
	}
	if (text.includes("ms")) {
		return number / 1000;
	}
	return number;
}

export function parseTimingDetail(value) {
	const text = String(value || "").replace(" of measured work", "");
	const seconds = text.match(/[-+]?(?:\d+(?:\.\d*)?|\.\d+)s/);
	const percent = text.match(/\(([^)]+%)\)/);
	return {
		value: parseNumber(seconds?.[0] || text),
		time: seconds?.[0] || text,
		percent: percent?.[1] || "",
	};
}

export function meanSecondsPerCall(row) {
	const explicit = parseNumber(row.mean_seconds_per_call);
	if (Number.isFinite(explicit)) {
		return explicit;
	}
	const solverSeconds = parseNumber(row.convex_solver_seconds);
	const calls = parseNumber(row.total_convex_calls);
	if (!Number.isFinite(solverSeconds) || !Number.isFinite(calls) || calls <= 0) {
		return null;
	}
	return solverSeconds / calls;
}

export function findTable(report, title) {
	return report.tables.find((table) => table.title === title);
}

export function instanceTotalSeconds(row) {
	const parts = [
		parseNumber(row.decomposition_seconds),
		parseNumber(row.approximation_seconds),
		parseNumber(row.bnb_seconds),
	].filter(Number.isFinite);
	if (parts.length > 0) {
		return parts.reduce((sum, value) => sum + value, 0);
	}
	return parseNumber(row.total_seconds);
}

export function solverLabel(name) {
	const labels = {
		linear_search_lazy: "Linear Intersections",
		linear_search_disjoint: "Linear Disjoint",
		binary_search_lazy: "Binary Intersections",
		binary_search_disjoint: "Binary Disjoint",
		binary_search_eager: "Binary Eager",
		tan_jiang: "Tan Jiang",
		gurobi: "Gurobi",
		linear: "Linear Intersections",
		linear_disjoint: "Linear Disjoint",
		binary: "Binary Intersections",
		binary_disjoint: "Binary Disjoint",
		tan: "Tan Jiang",
	};
	return labels[name] || name;
}
