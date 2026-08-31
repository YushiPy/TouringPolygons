import { escapeHTML } from "./dom.js";
import { state } from "./state.js";
import { sortCampaigns } from "./sorting.js";

export function renderCampaignList(root, { closeIconSVG, trashIconSVG, deleteCampaign, openCampaignModal, runProgress }) {
	if (!root) {
		return;
	}
	root.innerHTML = "";
	const campaigns = sortCampaigns(state.campaigns, state.campaignSort, state.campaignSortReverse).filter((campaign) => {
		const query = state.campaignFilter;
		if (!query) {
			return true;
		}
		return [campaign.name, JSON.stringify(campaign.generation || {})]
			.some((value) => String(value || "").toLowerCase().includes(query));
	});
	if (campaigns.length === 0) {
		root.innerHTML = '<div class="empty-choice">No campaigns match the current filter.</div>';
		return;
	}
	const deleteIconSVG = trashIconSVG || closeIconSVG;
	for (const campaign of campaigns) {
		const generation = campaign.generation || {};
		const progress = runProgress(campaign);
		const instanceCount = campaign.instance_progress?.total ?? generation.instances ?? generation.instances_per_file ?? 0;
		const previewMarkup = campaign.has_preview
			? `<img class="preview" src="/api/campaigns/${encodeURIComponent(campaign.name)}/preview?v=${campaign.version || ""}" alt="Preview for ${escapeHTML(campaign.name)}" loading="lazy">`
			: instanceCount === 0
				? '<div class="preview empty-preview">This campaign has no instances.</div>'
				: "";
		const card = document.createElement("article");
		card.className = "campaign-card";
		card.tabIndex = 0;
		card.innerHTML = `
			<button class="campaign-delete" type="button" data-delete-campaign="${escapeHTML(campaign.name)}" aria-label="Delete ${escapeHTML(campaign.name)}">${deleteIconSVG()}</button>
			<h3>${escapeHTML(campaign.name)}</h3>
			<div class="meta">
				<div><span>Instances</span><br>${instanceCount || "-"}</div>
				<div><span>Progress</span><br>${progress.label}</div>
			</div>
			<div class="bar" aria-label="Benchmark progress">
				<div class="bar-fill" style="width: ${Math.round(progress.ratio * 100)}%"></div>
			</div>
			${previewMarkup}
		`;
		card.querySelector(".campaign-delete").addEventListener("click", (event) => {
			event.stopPropagation();
			deleteCampaign(event);
		});
		card.addEventListener("click", () => openCampaignModal(campaign));
		card.addEventListener("keydown", (event) => {
			if (event.key === "Enter" || event.key === " ") {
				event.preventDefault();
				openCampaignModal(campaign);
			}
		});
		root.appendChild(card);
	}
}
