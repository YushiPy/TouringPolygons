import { shellQuote } from "./format.js";

const CLI_SOLVERS = {
	linear: "linear_search_lazy",
	linear_disjoint: "linear_search_disjoint",
	binary: "binary_search_lazy",
	binary_disjoint: "binary_search_disjoint",
	tan: "tan_jiang",
	gurobi: "gurobi",
};

export function formData(form) {
	return Object.fromEntries(new FormData(form).entries());
}

export function boolField(form, name) {
	const input = form.querySelector(`[name="${name}"]`);
	return input.type === "checkbox" ? input.checked : input.value === "1";
}

export function runCommandFromForm(form = document.querySelector("#run-form")) {
	const values = formData(form);
	const command = ["python3", "benchmarks/tpp.py", "run", values.name];
	if (values.threads) {
		command.push("--threads", values.threads);
	}
	if (values.solver) {
		command.push("--solver", CLI_SOLVERS[values.solver] || values.solver);
	}
	if (values.max_instances) {
		command.push("--max-instances", values.max_instances);
	}
	command.push("--max-calls", values.max_calls || "1000000");
	if (values.max_seconds) {
		command.push("--max-seconds", values.max_seconds);
	}
	if (values.timeout) {
		command.push("--timeout", values.timeout);
	}
	if (boolField(form, "force")) {
		command.push("--force");
	}
	if (boolField(form, "no_build")) {
		command.push("--no-build");
	}
	if (boolField(form, "dry_run")) {
		command.push("--dry-run");
	}
	return command.map(shellQuote).join(" ");
}

export function compareCommandFromForm(form = document.querySelector("#compare-form")) {
	const values = formData(form);
	const solvers = [...form.querySelectorAll('input[name="solvers"]:checked')].map((input) => input.value);
	const inputDir = shellQuote(`benchmarks/campaigns/${values.name}/inputs`);
	const command = [
		"python3",
		"benchmarks/tpp.py",
		"compare-solvers",
		"--suite",
		`$(find ${inputDir} -name '*.bin' | sort | head -n 1)`,
		"--output",
		`benchmarks/campaigns/${values.name}/results/comparisons`,
		"--max-calls",
		values.max_calls || "1000000",
	];
	if (values.max_instances) {
		command.push("--max-instances", values.max_instances);
	}
	command.push("--max-polygons", "-1", "--max-branching", "-1", "--keep-going");
	if (values.threads) {
		command.push("--threads", values.threads);
	}
	if (values.max_seconds) {
		command.push("--max-seconds", values.max_seconds);
	}
	if (boolField(form, "no_build")) {
		command.push("--no-build");
	}
	for (const solver of solvers) {
		command.push("--solver", CLI_SOLVERS[solver] || solver);
	}
	return command.map((part) => String(part).startsWith("$(") ? part : shellQuote(part)).join(" ");
}
