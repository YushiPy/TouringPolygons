export function jobPanel(job) {
	return job.kind === "comparison" ? "comparison-panel" : "benchmark-panel";
}

export function jobKindLabel(job) {
	return job.kind === "comparison" ? "Comparison" : "Benchmark";
}

export function jobProgressLabel(job) {
	if (job.kind === "comparison") {
		const solverTotal = job.solver_progress_total || 0;
		const solverCompleted = job.solver_progress_completed || 0;
		const instanceTotal = job.progress_total || 0;
		const instanceCompleted = job.progress_completed || 0;
		const solverText = solverTotal ? `${solverCompleted}/${solverTotal} solvers` : "Comparing solvers";
		const instanceText = instanceTotal ? `, ${instanceCompleted}/${instanceTotal} instances` : "";
		return `${solverText}${instanceText}`;
	}
	if (job.progress_total) {
		return `${job.progress_completed || 0}/${job.progress_total} instances`;
	}
	return "Compiling";
}

export function jobTerminalState(job) {
	if (job.status === "failed") {
		return "failed";
	}
	if (job.status === "canceled") {
		return "canceled";
	}
	if (job.status === "completed") {
		return "completed";
	}
	return "running";
}

export function jobDockStatusClass(job) {
	return `is-${jobTerminalState(job)}`;
}
