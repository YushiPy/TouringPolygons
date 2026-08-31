export function createFormSubmissionController({
	$, state, formData, boolField, renderBenchmarkReport, renderRunProgressCard,
	renderRunSummary, setOutput, setStopButton, requestJSON, pollJob, cancelJobId,
	switchPanel, renderComparisonReport, renderComparisonProgress, solverLabel,
	pollComparisonJob,
}) {
	async function runCampaign(event) {
		event.preventDefault();
		if (state.currentRunJob) {
			const jobId = $("#run-submit-button")?.dataset.job || $("#stop-run-button")?.dataset.job;
			if (jobId && jobId !== "pending") {
				$("#run-submit-button").disabled = true;
				$("#run-submit-button").textContent = "Stopping...";
				await cancelJobId(jobId);
			}
			switchPanel("benchmark-panel");
			return;
		}
		const form = event.currentTarget;
		const values = formData(form);
		const payload = {
			name: values.name,
			threads: values.threads ? Number(values.threads) : null,
			solver: values.solver || null,
			max_calls: values.max_calls,
			max_instances: values.max_instances ? Number(values.max_instances) : null,
			max_seconds: values.max_seconds || null,
			timeout: values.timeout ? Number(values.timeout) : null,
			dry_run: boolField(form, "dry_run"),
			force: boolField(form, "force"),
			no_build: boolField(form, "no_build"),
		};
		const output = $("#run-output");
		renderBenchmarkReport(null);
		const activeCampaign = state.campaigns.find((item) => item.name === state.selectedCampaign);
		renderRunProgressCard(activeCampaign, true, activeCampaign ? {
			completed: 0, total: 0, ratio: 0, label: "Compiling", elapsed_seconds: 0,
			counts: activeCampaign.run_index?.counts || {}, phase: "compile",
		} : null);
		setOutput(output, "Starting run...");
		renderRunSummary();
		try {
			state.currentRunJob = "pending";
			const runButton = $("#run-submit-button");
			if (runButton) { runButton.disabled = true; runButton.textContent = "Starting..."; }
			const data = await requestJSON("/api/runs", { method: "POST", body: JSON.stringify(payload) });
			state.currentJob = data.job;
			state.currentRunJob = data.job;
			setStopButton("#stop-run-button", data.job);
			await pollJob(data.job);
		} catch (error) {
			state.currentRunJob = null;
			setStopButton("#stop-run-button", null);
			setOutput(output, error.message);
		}
	}

	async function runComparison(event) {
		event.preventDefault();
		if (state.currentComparisonJob) {
			switchPanel("comparison-panel");
			return;
		}
		const form = event.currentTarget;
		const values = formData(form);
		const solvers = [...form.querySelectorAll('input[name="solvers"]:checked')].map((input) => input.value);
		const payload = {
			name: values.name, threads: values.threads ? Number(values.threads) : null,
			solvers, max_calls: values.max_calls,
			max_instances: values.max_instances ? Number(values.max_instances) : null,
			max_seconds: values.max_seconds || null, no_build: boolField(form, "no_build"),
		};
		const output = $("#compare-output");
		const progress = $("#compare-progress");
		renderComparisonReport(null);
		renderComparisonProgress(progress, {
			active: true, status: "Starting comparison",
			testCase: state.selectedComparisonCampaign || values.name || "",
			elapsedSeconds: 0, currentSolver: solvers.map(solverLabel).join(", "),
			instanceCompleted: 0, instanceTotal: 0, solverCompleted: 0, solverTotal: solvers.length,
		});
		setOutput(output, "Starting comparison...");
		try {
			state.currentComparisonJob = "pending";
			const data = await requestJSON("/api/comparisons", { method: "POST", body: JSON.stringify(payload) });
			state.currentJob = data.job;
			state.currentComparisonJob = data.job;
			await pollComparisonJob(data.job);
		} catch (error) {
			state.currentComparisonJob = null;
			setOutput(output, error.message);
			setStopButton("#stop-compare-button", null);
		}
	}

	return { runCampaign, runComparison };
}
