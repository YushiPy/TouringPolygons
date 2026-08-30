export function createDashboardControls({ $, campaignInstanceTotal, state, updateCreateMode }) {
	function setupThreadsControl(sliderSelector = "#threads-slider", inputSelector = "#threads-input", maxLabelSelector = "#threads-max-label") {
		const slider = $(sliderSelector);
		const input = $(inputSelector);
		const maxLabel = $(maxLabelSelector);

		function clamp(value) {
			const parsed = Number(value);
			if (!Number.isFinite(parsed)) {
				return 1;
			}
			return Math.max(1, Math.min(state.cpuCount, Math.round(parsed)));
		}

		function setThreads(value) {
			const clamped = clamp(value);
			slider.max = String(state.cpuCount);
			input.max = String(state.cpuCount);
			slider.value = String(clamped);
			input.value = String(clamped);
			maxLabel.textContent = String(state.cpuCount);
			slider.style.setProperty("--value", String(clamped));
			slider.style.setProperty("--max", String(state.cpuCount));
			slider.style.setProperty("--progress", `${((clamped - 1) / Math.max(1, state.cpuCount - 1)) * 100}%`);
			slider.style.setProperty("--tick-step", `${100 / Math.max(1, state.cpuCount - 1)}%`);
		}

		slider.addEventListener("input", () => setThreads(slider.value));
		input.addEventListener("input", () => setThreads(input.value));
		input.addEventListener("blur", () => setThreads(input.value));
		setThreads(state.cpuCount);
	}

	function setupBoundedSliders() {
		document.querySelectorAll("[data-range-for]").forEach((slider) => {
			const input = document.querySelector(`[name="${slider.dataset.rangeFor}"]`);
			if (!input) {
				return;
			}
			const min = Number(slider.min);
			const max = Number(slider.max);
			const step = Number(slider.step) || 1;
			const integer = Number.isInteger(step) && step >= 1;

			function clamp(value) {
				const parsed = Number(value);
				if (!Number.isFinite(parsed)) {
					return Number(input.value || slider.value || min);
				}
				const bounded = Math.max(min, Math.min(max, parsed));
				const snapped = Math.round((bounded - min) / step) * step + min;
				return integer ? Math.round(snapped) : Number(snapped.toFixed(6));
			}

			function setValue(value) {
				const clamped = clamp(value);
				slider.value = String(clamped);
				input.value = String(clamped);
				slider.style.setProperty("--progress", `${((clamped - min) / Math.max(step, max - min)) * 100}%`);
			}

			slider.addEventListener("input", () => setValue(slider.value));
			input.addEventListener("input", () => setValue(input.value));
			input.addEventListener("blur", () => setValue(input.value));
			setValue(input.value || slider.value);
		});

		const polygonSize = document.querySelector('[name="grid_polygon_size"]');
		const cellSize = document.querySelector('[name="grid_cell_size"]');
		if (!polygonSize || !cellSize) {
			return;
		}
		function clampGridCell() {
			const polygon = Number(polygonSize.value);
			const cell = Number(cellSize.value);
			if (!Number.isFinite(polygon) || !Number.isFinite(cell)) {
				return;
			}
			if (cell <= polygon) {
				cellSize.value = String(Number((polygon + 0.1).toFixed(6)));
				const slider = document.querySelector('[data-range-for="grid_cell_size"]');
				if (slider) {
					slider.value = cellSize.value;
				}
			}
		}
		polygonSize.addEventListener("input", clampGridCell);
		cellSize.addEventListener("blur", clampGridCell);
		clampGridCell();
	}

	function setupMaxInstancesControl() {
		const slider = $("#max-instances-slider");
		const input = $("#max-instances-input");
		const maxLabel = $("#max-instances-label");

		function clamp(value) {
			const max = campaignInstanceTotal(state.selectedCampaign);
			const parsed = Number(value);
			if (!Number.isFinite(parsed)) {
				return max;
			}
			return Math.max(1, Math.min(max, Math.round(parsed)));
		}

		function setValue(value = null) {
			const max = campaignInstanceTotal(state.selectedCampaign);
			const clamped = clamp(value ?? max);
			slider.max = String(max);
			input.max = String(max);
			slider.value = String(clamped);
			input.value = String(clamped);
			maxLabel.textContent = String(max);
			slider.style.setProperty("--progress", `${((clamped - 1) / Math.max(1, max - 1)) * 100}%`);
			slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
		}

		slider.addEventListener("input", () => setValue(slider.value));
		input.addEventListener("input", () => setValue(input.value));
		input.addEventListener("blur", () => setValue(input.value));
		setValue();
	}

	function resetMaxInstancesControl() {
		const slider = $("#max-instances-slider");
		const input = $("#max-instances-input");
		const maxLabel = $("#max-instances-label");
		const max = campaignInstanceTotal(state.selectedCampaign);
		slider.max = String(max);
		input.max = String(max);
		slider.value = String(max);
		input.value = String(max);
		maxLabel.textContent = String(max);
		slider.style.setProperty("--progress", "100%");
		slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
	}

	function setupCompareMaxInstancesControl() {
		const slider = $("#compare-max-instances-slider");
		const input = $("#compare-max-instances-input");
		const maxLabel = $("#compare-max-instances-label");

		function clamp(value) {
			const max = campaignInstanceTotal(state.selectedComparisonCampaign);
			const parsed = Number(value);
			if (!Number.isFinite(parsed)) {
				return max;
			}
			return Math.max(1, Math.min(max, Math.round(parsed)));
		}

		function setValue(value = null) {
			const max = campaignInstanceTotal(state.selectedComparisonCampaign);
			const clamped = clamp(value ?? max);
			slider.max = String(max);
			input.max = String(max);
			slider.value = String(clamped);
			input.value = String(clamped);
			maxLabel.textContent = String(max);
			slider.style.setProperty("--progress", `${((clamped - 1) / Math.max(1, max - 1)) * 100}%`);
			slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
		}

		slider.addEventListener("input", () => setValue(slider.value));
		input.addEventListener("input", () => setValue(input.value));
		input.addEventListener("blur", () => setValue(input.value));
		setValue();
	}

	function resetCompareMaxInstancesControl() {
		const slider = $("#compare-max-instances-slider");
		const input = $("#compare-max-instances-input");
		const maxLabel = $("#compare-max-instances-label");
		const max = campaignInstanceTotal(state.selectedComparisonCampaign);
		slider.max = String(max);
		input.max = String(max);
		slider.value = String(max);
		input.value = String(max);
		maxLabel.textContent = String(max);
		slider.style.setProperty("--progress", "100%");
		slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
	}

	function setupFilterInput(selector, stateKey, render) {
		const input = $(selector);
		if (!input) {
			return;
		}
		input.addEventListener("input", () => {
			state[stateKey] = input.value.trim().toLowerCase();
			render();
		});
	}

	function setupSegmentedControls() {
		document.querySelectorAll(".segmented").forEach((group) => {
			const input = document.querySelector(`[name="${group.dataset.input}"]`);
			group.querySelectorAll(".segment").forEach((button) => {
				button.addEventListener("click", () => {
					input.value = button.dataset.value;
					group.querySelectorAll(".segment").forEach((item) => {
						item.classList.toggle("is-active", item === button);
					});
					if (group.dataset.input === "campaign_type") {
						updateCreateMode();
					}
				});
			});
		});
	}

	function setupToggleButtons() {
		document.querySelectorAll(".toggle-button").forEach((button) => {
			const input = document.querySelector(`[name="${button.dataset.input}"]`);
			button.addEventListener("click", () => {
				const active = input.value !== "1";
				input.value = active ? "1" : "";
				button.classList.toggle("is-active", active);
				button.setAttribute("aria-pressed", active ? "true" : "false");
			});
		});
	}

	return {
		resetCompareMaxInstancesControl,
		resetMaxInstancesControl,
		setupBoundedSliders,
		setupCompareMaxInstancesControl,
		setupFilterInput,
		setupMaxInstancesControl,
		setupSegmentedControls,
		setupThreadsControl,
		setupToggleButtons,
	};
}
