export function visitOrder() {
	return document.querySelector("#run-visit-order")?.value || "fixed";
}

export async function solveFreeOrder(caseData, signal) {
	const response = await fetch("/api/editor/solve", {
		method: "POST", headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ start: caseData.start, target: caseData.target, polygons: caseData.polygons.filter(polygon => polygon.length >= 3), visit_order: "free" }), signal,
	});
	const result = await response.json();
	if (!response.ok) {
		const detail = Array.isArray(result.detail) ? result.detail.map(item => item.msg).join("; ") : result.detail;
		throw new Error(detail || "Free-order solver failed.");
	}
	return result;
}

export function setupVisitOrder(onChange) {
	const selectors = [...document.querySelectorAll("[data-visit-order]")];
	const apply = (value) => {
		selectors.forEach((input) => { input.value = value; });
		document.querySelectorAll("[data-visit-order-picker]").forEach((picker) => {
			picker.querySelectorAll("[data-value]").forEach((button) => {
				const selected = button.dataset.value === value;
				button.classList.toggle("is-active", selected);
				button.setAttribute("aria-pressed", selected ? "true" : "false");
			});
		});
		for (const id of ["run-form", "compare-form"]) {
			const form = document.getElementById(id);
			form.querySelectorAll("[data-order-only]").forEach((group) => {
				const enabled = group.dataset.orderOnly === value;
				group.classList.toggle("is-hidden", !enabled);
				group.querySelectorAll("input,button").forEach((input) => { input.disabled = !enabled; });
			});
			form.querySelector('[name="max_seconds"]').placeholder = value === "free" ? "30 seconds per instance" : "unlimited";
			const timeout = form.querySelector('[name="timeout"]');
			if (timeout) timeout.disabled = value === "free";
		}
	};
	selectors.forEach((input) => input.addEventListener("change", () => {
		apply(input.value);
		onChange();
	}));
	document.querySelectorAll("[data-visit-order-picker] [data-value]").forEach((button) => button.addEventListener("click", () => {
		apply(button.dataset.value);
		onChange();
	}));
	apply("fixed");
}
