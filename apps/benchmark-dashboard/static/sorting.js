function instanceName(item, index) {
	return String(item.name || `Instance ${Number(index) + 1}`).toLowerCase();
}

export function sortCampaigns(campaigns, mode = "default") {
	return [...campaigns].sort((left, right) => {
		if (mode === "name") {
			return left.name.localeCompare(right.name, undefined, { sensitivity: "base", numeric: true });
		}
		if (mode === "count") {
			const count = (campaign) => Number(campaign.instance_progress?.total ?? campaign.generation?.instances ?? 0);
			return count(right) - count(left) || left.name.localeCompare(right.name, undefined, { sensitivity: "base" });
		}
		const leftOrder = Number.isFinite(left.order) ? left.order : Number.MAX_SAFE_INTEGER;
		const rightOrder = Number.isFinite(right.order) ? right.order : Number.MAX_SAFE_INTEGER;
		return leftOrder - rightOrder || left.name.localeCompare(right.name, undefined, { sensitivity: "base" });
	});
}

export function sortInstances(cases, mode = "default") {
	return cases
		.map((item, index) => ({ item, index }))
		.sort((left, right) => {
			if (mode === "name") {
				return instanceName(left.item, left.index).localeCompare(instanceName(right.item, right.index), undefined, { numeric: true }) || left.index - right.index;
			}
			if (mode === "count") {
				return right.item.polygons.length - left.item.polygons.length || left.index - right.index;
			}
			return left.index - right.index;
		});
}
