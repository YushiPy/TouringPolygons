import { escapeHTML } from "./dom.js";
import { state } from "./state.js";

export function renderCampaignList(root, { closeIconSVG, deleteCampaign, describeVertices, openCampaignModal, runProgress }) {
	if (!root) {
		return;
	}
	root.innerHTML = "";
	const campaigns = state.campaigns.filter((campaign) => {
		const query = state.campaignFilter;
		if (!query) {
			return true;
		}
		return [campaign.name, campaign.type, JSON.stringify(campaign.generation || {})]
			.some((value) => String(value || "").toLowerCase().includes(query));
	});
	if (campaigns.length === 0) {
		root.innerHTML = '<div class="empty-choice">No campaigns match the current filter.</div>';
		return;
	}
	for (const campaign of campaigns) {
		const generation = campaign.generation || {};
		const progress = runProgress(campaign);
		const card = document.createElement("article");
		card.className = "campaign-card";
		card.tabIndex = 0;
		card.innerHTML = `
			<button class="campaign-delete" type="button" data-delete-campaign="${escapeHTML(campaign.name)}" aria-label="Delete ${escapeHTML(campaign.name)}">${closeIconSVG()}</button>
			<h3>${escapeHTML(campaign.name)}</h3>
			<div class="meta">
				<div><span>Type</span><br>${escapeHTML(campaign.type)}</div>
				<div><span>Instances</span><br>${generation.instances ?? generation.instances_per_file ?? "-"}</div>
				<div><span>Polygon Count</span><br>${generation.polygons ?? generation.polygon_counts ?? "-"}</div>
				<div><span>Vertices</span><br>${escapeHTML(describeVertices(generation))}</div>
				<div><span>Progress</span><br>${progress.label}</div>
			</div>
			<div class="bar" aria-label="Benchmark progress">
				<div class="bar-fill" style="width: ${Math.round(progress.ratio * 100)}%"></div>
			</div>
			${campaign.has_preview ? `<img class="preview" src="/api/campaigns/${encodeURIComponent(campaign.name)}/preview?v=${campaign.version || ""}" alt="Preview for ${escapeHTML(campaign.name)}">` : ""}
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
