import { $, escapeHTML } from "./dom.js";
import { downloadCSV, formatMicroseconds, formatSeconds } from "./format.js";
import { findTable, instanceTotalSeconds, meanSecondsPerCall, metricCard, parseDurationSeconds, parseNumber, parseTimingDetail, shortNumber, solverLabel } from "./report-utils.js";

function renderBarChart(rows, options = {}) {
	const values = rows
		.map((row) => ({
			label: row.label,
			value: row.numeric ?? parseNumber(row.value),
			time: row.time || row.detail || row.value,
			percent: row.percent || "",
		}))
		.filter((row) => Number.isFinite(row.value) && row.value > 0)
		.slice(0, options.limit || 8);
	if (values.length === 0) {
		return "";
	}
	const max = Math.max(...values.map((row) => row.value));
	return `
		<div class="chart-bars">
			${values.map((row) => `
				<div class="chart-row">
					<span>${escapeHTML(row.label)}</span>
					<div><i style="width: ${Math.max(2, Math.round((row.value / max) * 100))}%"></i></div>
					<strong>${escapeHTML(row.time)}</strong>
					<em>${escapeHTML(row.percent)}</em>
				</div>
			`).join("")}
		</div>
	`;
}

function tableValue(table, label, labelKey) {
	const row = table?.rows.find((item) => item[labelKey] === label);
	return row?.Value ?? null;
}

function formatSummaryValue(label, value) {
	if (value === null || value === undefined) {
		return "-";
	}
	if (label === "Mean seconds per call") {
		return formatMicroseconds(parseDurationSeconds(value));
	}
	if (label === "Total work in seconds") {
		return formatSeconds(parseDurationSeconds(value));
	}
	return value;
}

function renderHistogram(values, formatter) {
	const finite = values.filter(Number.isFinite);
	if (finite.length === 0) {
		return '<p class="inline-note">No values available.</p>';
	}
	const min = Math.min(...finite);
	const max = Math.max(...finite);
	const binCount = Math.min(12, Math.max(4, Math.ceil(Math.sqrt(finite.length))));
	const width = max === min ? 1 : (max - min) / binCount;
	const bins = Array.from({ length: binCount }, (_, index) => ({
		start: min + index * width,
		end: index === binCount - 1 ? max : min + (index + 1) * width,
		count: 0,
	}));
	finite.forEach((value) => {
		const index = max === min ? 0 : Math.min(binCount - 1, Math.floor((value - min) / width));
		bins[index].count += 1;
	});
	const peak = Math.max(...bins.map((bin) => bin.count), 1);
	return `
		<div class="histogram">
			${bins.map((bin) => `
				<div class="histogram-row">
					<span>${escapeHTML(formatter(bin.start))}-${escapeHTML(formatter(bin.end))}</span>
					<div><i style="width: ${Math.max(3, Math.round((bin.count / peak) * 100))}%"></i></div>
					<strong>${bin.count}</strong>
				</div>
			`).join("")}
		</div>
	`;
}

function renderBenchmarkSummaryRows(timing, metrics, counters) {
	const rows = [
		[
			["Benchmarked instances", tableValue(metrics, "Benchmarked instances", "Metric")],
			["Fully solved runs", tableValue(metrics, "Fully solved runs", "Metric")],
			["Capped by calls runs", tableValue(metrics, "Capped by calls runs", "Metric")],
			["Capped by time runs", tableValue(metrics, "Capped by time runs", "Metric")],
		],
		[
			["Worker threads", tableValue(metrics, "Worker threads", "Metric")],
			["Convex solver name", tableValue(metrics, "Convex solver name", "Metric")],
		],
		[
			["Wall-clock total", tableValue(timing, "Wall-clock total", "Timing")],
			["Total work in seconds", tableValue(timing, "Measured work", "Timing")],
			["Mean seconds per call", tableValue(timing, "Mean seconds per call", "Timing")],
		],
		[
			["Total convex calls", tableValue(counters, "Total convex calls", "B&B Counter")],
			["Bound solves", tableValue(counters, "Bound solves", "B&B Counter")],
			["Leaf solves", tableValue(counters, "Leaf solves", "B&B Counter")],
		],
	];
	return `
		<div class="benchmark-summary-rows">
			${rows.map((row) => `
				<div class="benchmark-summary-row" style="--summary-columns: ${row.length}">
					${row.map(([label, value]) => metricCard(label, formatSummaryValue(label, value))).join("")}
				</div>
			`).join("")}
		</div>
	`;
}

export function renderBenchmarkReport(report) {
	const root = $("#benchmark-report");
	if (!report || report.files.length === 0) {
		root.classList.add("is-hidden");
		root.innerHTML = "";
		return;
	}
	const timing = findTable(report, "Timing");
	const metrics = findTable(report, "Metric");
	const counters = findTable(report, "B&B Counter");
	const resultRows = report.files[0].rows || [];
	const timingRows = timing?.rows
		.filter((row) => ["Decomposition", "Approximation", "B&B", "Convex solver"].includes(row.Timing))
		.map((row) => {
			const detail = parseTimingDetail(row.Value);
			return {
				label: row.Timing,
				value: row.Value,
				numeric: detail.value,
				time: detail.time,
				percent: detail.percent,
			};
		}) || [];
	const timeHistogram = renderHistogram(
		resultRows.map(instanceTotalSeconds),
		(value) => formatSeconds(value),
	);
	const callHistogram = renderHistogram(
		resultRows.map((row) => parseNumber(row.calls)),
		(value) => shortNumber(value),
	);
	root.innerHTML = `
		<header class="report-header">
			<div>
				<h3>Latest Markdown Summary</h3>
				<p>${escapeHTML(report.files[0].path)}${report.input_file ? ` | Test case: ${escapeHTML(report.input_file)}` : ""}</p>
			</div>
			<button class="secondary" type="button" data-export-benchmark>Export CSV</button>
		</header>
		${renderBenchmarkSummaryRows(timing, metrics, counters)}
		<div class="report-grid report-grid-single">
			<section class="report-panel">
				<h4>Timing Share</h4>
				${renderBarChart(timingRows)}
			</section>
			<section class="report-panel">
				<div class="histogram-head">
					<h4>Instance Histograms</h4>
					<div>
						<button class="secondary" type="button" data-toggle-histogram="time">Time</button>
						<button class="secondary" type="button" data-toggle-histogram="calls">Calls</button>
					</div>
				</div>
				<div class="histogram-panel is-hidden" data-histogram-panel="time">
					<h5>Instance Time</h5>
					${timeHistogram}
				</div>
				<div class="histogram-panel is-hidden" data-histogram-panel="calls">
					<h5>Convex Calls</h5>
					${callHistogram}
				</div>
			</section>
		</div>
	`;
	root.querySelectorAll("[data-toggle-histogram]").forEach((button) => {
		button.addEventListener("click", () => {
			const target = button.dataset.toggleHistogram;
			const panel = root.querySelector(`[data-histogram-panel="${target}"]`);
			const hidden = panel.classList.toggle("is-hidden");
			button.classList.toggle("is-active", !hidden);
			button.setAttribute("aria-pressed", !hidden ? "true" : "false");
		});
	});
	root.querySelector("[data-export-benchmark]")?.addEventListener("click", () => {
		downloadCSV(`${report.files[0].path.split("/").pop() || "benchmark"}.csv`, resultRows);
	});
	root.classList.remove("is-hidden");
}

export function renderComparisonReport(data) {
	const root = $("#comparison-report");
	const rows = data?.rows || [];
	if (rows.length === 0) {
		root.classList.add("is-hidden");
		root.innerHTML = "";
		return;
	}

	const completed = rows.filter((row) => row.status === "completed").length;
	const fastest = rows
		.map((row) => ({ solver: row.solver, seconds: parseNumber(row.wall_clock_seconds) }))
		.filter((row) => Number.isFinite(row.seconds))
		.sort((left, right) => left.seconds - right.seconds)[0];
	const calls = rows
		.map((row) => ({ solver: row.solver, calls: parseNumber(row.total_convex_calls) }))
		.filter((row) => Number.isFinite(row.calls));
	const maxCalls = calls.length ? Math.max(...calls.map((row) => row.calls)) : 0;
	const timeRows = rows.map((row) => ({
		label: solverLabel(row.solver),
		value: row.wall_clock_seconds,
	}));
	const meanRows = rows.map((row) => {
		const mean = meanSecondsPerCall(row);
		return {
			label: solverLabel(row.solver),
			value: row.mean_seconds_per_call,
			numeric: mean,
			time: mean === null ? "-" : formatMicroseconds(mean),
		};
	});
	const bestMean = rows
		.map((row) => ({ solver: row.solver, seconds: meanSecondsPerCall(row) }))
		.filter((row) => Number.isFinite(row.seconds))
		.sort((left, right) => left.seconds - right.seconds)[0];

	root.innerHTML = `
		<header class="report-header">
			<div>
				<h3>Latest Solver Comparison</h3>
				<p>${rows.length} solver${rows.length === 1 ? "" : "s"} in the latest comparison run${data?.input_file ? ` | Test case: ${escapeHTML(data.input_file)}` : ""}</p>
			</div>
			<button class="secondary" type="button" data-export-comparison>Export CSV</button>
		</header>
		<div class="summary-grid">
			${metricCard("Completed", `${completed}/${rows.length}`)}
			${metricCard("Fastest", fastest ? solverLabel(fastest.solver) : "-")}
			${metricCard("Best wall clock", fastest ? `${shortNumber(fastest.seconds)} s` : "-")}
			${metricCard("Best avg solve", bestMean ? formatMicroseconds(bestMean.seconds) : "-")}
			${metricCard("Total calls max", maxCalls || "-")}
		</div>
		<div class="report-grid">
			<section class="report-panel">
				<h4>Wall Clock</h4>
				${renderBarChart(timeRows)}
			</section>
			<section class="report-panel">
				<h4>Average Convex Solve</h4>
				${renderBarChart(meanRows)}
			</section>
			<section class="report-panel comparison-table-panel">
				<h4>Solver Details</h4>
				<div class="comparison-table-wrap">
					<table class="comparison-table">
						<thead>
							<tr>
								<th>Solver</th>
								<th>Status</th>
								<th>Wall</th>
								<th>Work</th>
								<th>Avg solve</th>
								<th>Calls</th>
								<th>Solved</th>
							</tr>
						</thead>
						<tbody>
							${rows.map((row) => `
								<tr>
									<td>${escapeHTML(solverLabel(row.solver))}</td>
									<td>${escapeHTML(row.status)}</td>
									<td>${shortNumber(row.wall_clock_seconds)} s</td>
									<td>${shortNumber(row.convex_solver_seconds)} s</td>
									<td>${meanSecondsPerCall(row) === null ? "-" : formatMicroseconds(meanSecondsPerCall(row))}</td>
									<td>${escapeHTML(row.total_convex_calls || "-")}</td>
									<td>${escapeHTML(row.fully_solved_runs || "-")}</td>
								</tr>
							`).join("")}
						</tbody>
					</table>
				</div>
			</section>
		</div>
	`;
	root.querySelector("[data-export-comparison]")?.addEventListener("click", () => {
		downloadCSV(`${data?.path?.split("/").at(-2) || "comparison"}.csv`, rows);
	});
	root.classList.remove("is-hidden");
}
