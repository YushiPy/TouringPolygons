import { instanceLabel } from "./case-data.js";

export function instancePreviewUrl(campaign, index) {
	return `/api/campaigns/${encodeURIComponent(campaign.name)}/preview/instance-${index}?v=${campaign.version || ""}`;
}

export function solutionPreviewUrl(campaign, item) {
	return `/api/campaigns/${encodeURIComponent(campaign.name)}/solution-preview/${item.case_index}?repeat_index=${item.repeat_index}&v=${campaign.version || ""}`;
}

export function benchmarkedPreviewHTML(campaign, item) {
	if (item.solution_available) {
		return `<img src="${solutionPreviewUrl(campaign, item)}" alt="Solved instance ${instanceLabel(item.case_index)} with path and decomposition" loading="lazy">`;
	}
	if (item.preview) {
		return `<img src="${instancePreviewUrl(campaign, item.case_index)}" alt="Benchmarked instance ${instanceLabel(item.case_index)}" loading="lazy">`;
	}
	return '<div class="missing-preview">No preview</div>';
}
