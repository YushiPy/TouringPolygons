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
		selectors.forEach((select) => { select.value = value; });
		for (const id of ["run-form", "compare-form"]) {
			const form = document.getElementById(id);
			form.querySelectorAll("[data-order-only]").forEach((group) => {
				const enabled = group.dataset.orderOnly === value;
				group.classList.toggle("is-hidden", !enabled);
				group.querySelectorAll("input,button").forEach((input) => { input.disabled = !enabled; });
			});
			form.querySelectorAll('input[name="threads"], input[id$="threads-slider"]').forEach((input) => {
				input.disabled = value === "free";
				if (value === "free") input.value = "1";
			});
			form.querySelector('[name="max_seconds"]').placeholder = value === "free" ? "30 seconds per instance" : "unlimited";
			const timeout = form.querySelector('[name="timeout"]');
			if (timeout) timeout.disabled = value === "free";
		}
	};
	selectors.forEach((select) => select.addEventListener("change", () => {
		apply(select.value);
		onChange();
	}));
	apply("fixed");
}
