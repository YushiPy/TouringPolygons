export function createConfirmationController({ $ }) {
	let pendingConfirmation = null;

	function askConfirmation(message, action = "Delete") {
		const modal = $("#confirm-modal");
		$("#confirm-message").textContent = message;
		modal.querySelector("[data-confirm-ok]").textContent = action;
		modal.classList.remove("is-hidden");
		return new Promise((resolve) => {
			pendingConfirmation = resolve;
		});
	}

	function closeConfirmation(value) {
		$("#confirm-modal").classList.add("is-hidden");
		if (pendingConfirmation) {
			pendingConfirmation(value);
			pendingConfirmation = null;
		}
	}

	return { askConfirmation, closeConfirmation };
}

export function createModalController({
	$, state, setCloseIcon, metricCard, describeVertices, runProgress,
	renderPreviewPanels, renderBenchmarkedInstanceSection, refreshBenchmarkedInstances,
	selectManualCampaign, switchPanel, deleteCampaign,
}) {
	function openCampaignModal(campaign) {
		const modal = $("#campaign-modal");
		const body = $("#modal-body");
		const closeButton = modal.querySelector(".modal-x-button");
		const generation = campaign.generation || {};
		const progress = runProgress(campaign);
		state.instanceModalReturn = null;
		closeButton?.removeAttribute("data-modal-back-instance");
		if (closeButton) {
			closeButton.classList.remove("modal-back-button");
			setCloseIcon(closeButton);
			closeButton.setAttribute("aria-label", "Close campaign details");
		}
		$("#modal-title").textContent = campaign.name;
		body.innerHTML = `
    <div class="modal-summary">
      ${metricCard("Type", campaign.type)}
      ${metricCard("Instances", generation.instances ?? generation.instances_per_file ?? "-")}
      ${metricCard("Polygon Count", generation.polygons ?? generation.polygon_counts ?? "-")}
      ${metricCard("Vertices", describeVertices(generation))}
      ${metricCard("Progress", progress.label)}
    </div>
    <div class="preview-layout modal-previews"></div>
    <section class="result-preview-section modal-results is-hidden" data-benchmarked-section="${campaign.name}"></section>
		<h3 class="generation-metadata-title">Generation Metadata</h3>
    <pre class="output modal-json">${JSON.stringify(generation, null, 2)}</pre>
    <div class="modal-actions">
      ${campaign.type === "manual" ? `<button class="secondary" type="button" data-edit-campaign="${campaign.name}">Edit Cases</button>` : ""}
      <button class="danger" type="button" data-delete-campaign="${campaign.name}">Delete Campaign</button>
    </div>
  `;
		body.querySelector("[data-edit-campaign]")?.addEventListener("click", async (event) => {
			closeCampaignModal();
			await selectManualCampaign(event.currentTarget.dataset.editCampaign);
			switchPanel("cases-panel");
		});
		body.querySelector("[data-delete-campaign]").addEventListener("click", deleteCampaign);
		renderPreviewPanels(body.querySelector(".modal-previews"), campaign);
		const instances = state.benchmarkedInstances.get(campaign.name) || [];
		renderBenchmarkedInstanceSection(body.querySelector(".modal-results"), campaign, instances);
		if (progress.completed > 0 && instances.length === 0) {
			refreshBenchmarkedInstances(campaign.name);
		}
		modal.classList.remove("is-hidden");
	}

	function closeCampaignModal() {
		$("#campaign-modal").classList.add("is-hidden");
		state.instanceModalReturn = null;
	}

	function returnToCampaignModal() {
		const modalReturn = state.instanceModalReturn;
		if (!modalReturn) {
			closeCampaignModal();
			return;
		}
		if (modalReturn.panel) {
			closeCampaignModal();
			switchPanel(modalReturn.panel);
			return;
		}
		openCampaignModal(modalReturn.campaign || modalReturn);
	}

	return { openCampaignModal, closeCampaignModal, returnToCampaignModal };
}
