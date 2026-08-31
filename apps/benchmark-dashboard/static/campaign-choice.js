import { sortCampaigns } from "./sorting.js";

export function renderCampaignChoiceGrid(grid, selectedName, onSelect, { state, escapeHTML, sortMode = "default", reverse = false }) {
	grid.innerHTML = "";
	for (const campaign of sortCampaigns(state.campaigns, sortMode, reverse)) {
		const button = document.createElement("button");
		const generation = campaign.generation || {};
		button.type = "button";
		button.className = "choice-card";
		button.dataset.value = campaign.name;
		button.setAttribute("role", "option");
		button.innerHTML = `
      <strong>${escapeHTML(campaign.name)}</strong>
			<small>${campaign.instance_progress.total || generation.instances || "-"} instances</small>
    `;
		button.addEventListener("click", () => onSelect(campaign.name));
		grid.appendChild(button);
	}
	document.querySelectorAll(`#${grid.id} .choice-card`).forEach((button) => {
		const active = button.dataset.value === selectedName;
		button.classList.toggle("is-active", active);
		button.setAttribute("aria-selected", active ? "true" : "false");
	});
}
