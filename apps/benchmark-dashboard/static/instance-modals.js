export function createInstanceModalController({
	$, state, escapeHTML, instanceLabel, instanceDisplayName, instanceTitle,
	loadCampaignCaseMetadata, requestJSON, casePayload, cloneCaseData,
	setOutput, renderManualCases, renderCampaigns, renderSolvedPreview,
	readonlyInstanceDetail, setupReadonlyInstanceDetail, manualEditor, metricCard,
	shortNumber, formatSeconds, parseNumber, formatMicroseconds, editInstance,
}) {
	let activeViewer = null;

	function cancelActiveViewer() {
		activeViewer?.destroy?.();
		activeViewer = null;
	}

	function instanceModalTitle(campaign, index) {
		const total = campaign.instance_progress?.total || campaign.generation?.instances || campaign.generation?.instances_per_file || "?";
		return `<span class="modal-title-main">${escapeHTML(campaign.name)}</span><span class="modal-title-sub">${instanceLabel(index)}/${escapeHTML(total)}: <button class="instance-name-button modal-title-rename" type="button" data-modal-rename-trigger>${escapeHTML(instanceDisplayName(campaign, index))}</button></span>`;
	}

	function setupModalTitleRename(campaign, index, afterRename) {
		const title = $("#modal-title");
		const trigger = title.querySelector("[data-modal-rename-trigger]");
		if (!trigger) return;
		trigger.addEventListener("click", () => {
			const input = document.createElement("input");
			input.className = "instance-name-input modal-title-input";
			input.value = instanceDisplayName(campaign, index);
			trigger.replaceWith(input);
			input.focus();
			input.select();
			let committed = false;
			const commit = async () => {
				if (committed) return;
				committed = true;
				try {
					await renameCampaignInstance(campaign, index, input.value);
					afterRename?.();
				} catch (error) {
					setOutput($("#inspect-output"), error.message);
					title.innerHTML = instanceModalTitle(campaign, index);
					setupModalTitleRename(campaign, index, afterRename);
				}
			};
			input.addEventListener("blur", commit);
			input.addEventListener("keydown", (event) => {
				event.stopPropagation();
				if (event.key === "Enter") { event.preventDefault(); commit(); }
				if (event.key === "Escape") {
					event.preventDefault(); committed = true;
					title.innerHTML = instanceModalTitle(campaign, index);
					setupModalTitleRename(campaign, index, afterRename);
				}
			});
		});
	}

	async function renameCampaignInstance(campaign, index, value) {
		const cases = await loadCampaignCaseMetadata(campaign.name);
		if (!cases[index]) throw new Error("Case does not exist.");
		cases[index].name = value.trim();
		const data = await requestJSON(`/api/campaigns/${encodeURIComponent(campaign.name)}/cases`, {
			method: "PUT", body: JSON.stringify({ cases: cases.map(casePayload) }),
		});
		const campaignIndex = state.campaigns.findIndex((item) => item.name === data.campaign.name);
		if (campaignIndex !== -1) {
			state.campaigns[campaignIndex] = data.campaign;
			campaign.version = data.campaign.version;
		}
		state.campaignCaseMetadata.set(campaign.name, cases.map(cloneCaseData));
		if (state.manualCampaign === campaign.name) {
			state.manualCases = cases.map(cloneCaseData);
			renderManualCases();
		}
		renderCampaigns();
		renderSolvedPreview(state.campaigns.find((item) => item.name === campaign.name) || campaign);
	}

	function setInstanceModalBackButton(modal) {
		const closeButton = modal.querySelector(".modal-x-button");
		if (!closeButton) return;
		closeButton.setAttribute("data-modal-back-instance", "true");
		closeButton.classList.add("modal-back-button");
		closeButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 12H5M11 6l-6 6 6 6" /></svg>';
		closeButton.setAttribute("aria-label", "Back to campaign details");
	}

	async function openInstanceModal(campaign, index) {
		cancelActiveViewer();
		const modal = $("#campaign-modal");
		state.instanceModalReturn = { campaign };
		setInstanceModalBackButton(modal);
		const cases = await loadCampaignCaseMetadata(campaign.name);
		const caseData = cases[index];
		const body = $("#modal-body");
		const title = instanceTitle(campaign, index);
		$("#modal-title").innerHTML = instanceModalTitle(campaign, index);
		body.innerHTML = `${caseData ? readonlyInstanceDetail(`${title} detail`) : '<div class="missing-preview detail-missing">No case data available.</div>'}`;
		body.querySelector("[data-edit-instance]")?.addEventListener("click", () => editInstance(campaign, index));
		if (caseData) activeViewer = setupReadonlyInstanceDetail(body, caseData, manualEditor);
		setupModalTitleRename(campaign, index, () => openInstanceModal(campaign, index));
		modal.classList.remove("is-hidden");
	}

	async function openBenchmarkedInstanceModal(campaign, item) {
		cancelActiveViewer();
		const modal = $("#campaign-modal");
		state.instanceModalReturn = { campaign, panel: "benchmark-panel" };
		setInstanceModalBackButton(modal);
		const cases = await loadCampaignCaseMetadata(campaign.name);
		const caseData = cases[item.case_index];
		const body = $("#modal-body");
		const title = instanceTitle(campaign, item.case_index);
		$("#modal-title").innerHTML = instanceModalTitle(campaign, item.case_index);
		body.innerHTML = `<div class="modal-summary">${metricCard("Status", item.status)}${metricCard("Final length", shortNumber(item.final_length))}${metricCard("Solve time", formatSeconds(parseNumber(item.total_seconds)))}${metricCard("Calls", item.calls ?? "-")}${metricCard("Avg convex solve", formatMicroseconds(parseNumber(item.seconds_per_call)))}${metricCard("Decomposed pieces", item.decomposed_pieces ?? "-")}${metricCard("Visited nodes", item.visited_nodes ?? "-")}${metricCard("Pruned nodes", item.pruned_nodes ?? "-")}</div>${caseData ? readonlyInstanceDetail(`${title} detail`) : '<div class="missing-preview detail-missing">No case data available.</div>'}`;
		body.querySelector("[data-edit-instance]")?.addEventListener("click", () => editInstance(campaign, item.case_index));
		if (caseData) activeViewer = setupReadonlyInstanceDetail(body, caseData, manualEditor);
		setupModalTitleRename(campaign, item.case_index, () => openBenchmarkedInstanceModal(campaign, item));
		modal.classList.remove("is-hidden");
	}

	return { instanceModalTitle, setupModalTitleRename, renameCampaignInstance, setInstanceModalBackButton, openInstanceModal, openBenchmarkedInstanceModal, cancelActiveViewer };
}
