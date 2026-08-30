export function shellQuote(value) {
	const text = String(value ?? "");
	if (/^[A-Za-z0-9_./:=@+-]+$/.test(text)) {
		return text;
	}
	return `'${text.replaceAll("'", "'\\''")}'`;
}

function csvCell(value) {
	const text = String(value ?? "");
	return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function downloadCSV(filename, rows) {
	if (!rows || rows.length === 0) {
		return;
	}
	const headers = Object.keys(rows[0]);
	const csv = [
		headers.map(csvCell).join(","),
		...rows.map((row) => headers.map((header) => csvCell(row[header])).join(",")),
	].join("\n");
	const link = document.createElement("a");
	link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
	link.download = filename;
	link.click();
	URL.revokeObjectURL(link.href);
}

export function formatElapsed(seconds) {
	const value = Math.max(0, Number(seconds) || 0);
	const minutes = Math.floor(value / 60);
	const remaining = value - minutes * 60;
	if (minutes > 0) {
		return `${minutes}:${remaining.toFixed(1).padStart(4, "0")}`;
	}
	return `${remaining.toFixed(1)}s`;
}

export function formatSeconds(value) {
	if (value === null || value === undefined || value === "") {
		return "-";
	}
	const number = Number(value);
	if (!Number.isFinite(number)) {
		return "-";
	}
	if (Math.abs(number) >= 1) {
		return `${number.toFixed(3)} s`;
	}
	if (Math.abs(number) >= 0.001) {
		return `${(number * 1000).toFixed(2)} ms`;
	}
	if (Math.abs(number) >= 0.000001) {
		return `${(number * 1000000).toFixed(2)} us`;
	}
	return `${(number * 1000000000).toFixed(2)} ns`;
}

export function formatMicroseconds(seconds) {
	if (seconds === null || seconds === undefined || seconds === "") {
		return "-";
	}
	const number = Number(seconds);
	if (!Number.isFinite(number)) {
		return "-";
	}
	const microseconds = number * 1000000;
	if (Math.abs(microseconds) >= 100) {
		return `${microseconds.toFixed(1)} us`;
	}
	if (Math.abs(microseconds) >= 1) {
		return `${microseconds.toFixed(2)} us`;
	}
	return `${microseconds.toFixed(3)} us`;
}
