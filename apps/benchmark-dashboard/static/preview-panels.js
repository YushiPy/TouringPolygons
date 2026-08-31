const PREVIEW_SAMPLE_THRESHOLD = 100;

function sampleInstanceIndices(count, limit = 20) {
	if (count <= limit) {
		return Array.from({ length: count }, (_, index) => index);
	}
	return Array.from({ length: limit }, (_, index) => Math.round(index * (count - 1) / (limit - 1)));
}

function makeFoldablePanel(tagName, className, title, folded = false) {
	const panel = document.createElement(tagName);
	panel.className = className;
	panel.classList.toggle("is-folded", folded);
	panel.innerHTML = `
    <figcaption class="fold-header">
      <button class="fold-toggle" type="button" aria-expanded="${folded ? "false" : "true"}">
        <span class="fold-caret" aria-hidden="true"></span>
        <span>${title}</span>
      </button>
    </figcaption>
    <div class="fold-content"></div>
  `;
	const button = panel.querySelector(".fold-toggle");
	button.addEventListener("click", () => {
		if (panel.classList.contains("mobile-only-fold") && !window.matchMedia("(max-width: 760px)").matches) {
			return;
		}
		const isFolded = panel.classList.toggle("is-folded");
		button.setAttribute("aria-expanded", isFolded ? "false" : "true");
	});
	return panel;
}

export function createPreviewPanelController({
	instancePreviewButton,
	instancePreviewUrl,
	previewUrl,
	renderCanvasPlaceholder,
}) {
	async function populateInstancePreviewPanels(root, campaign, options = {}) {
		const previewCount = campaign.instance_previews?.length || 0;
		if (previewCount === 0) {
			root.classList.toggle("is-hidden", root.children.length === 0);
			return;
		}
		root.innerHTML = "";

		const selected = makeFoldablePanel("figure", "preview-panel preview-selected mobile-only-fold", "Selected Instance");
		selected.querySelector(".fold-content").appendChild(instancePreviewButton(
			campaign,
			0,
			"selected-instance-button",
			{ src: previewUrl(campaign, "selected") || instancePreviewUrl(campaign, 0) },
		));
		root.appendChild(selected);

		const four = makeFoldablePanel("figure", "preview-panel preview-four mobile-only-fold", "Four Instances");
		const grid = document.createElement("div");
		grid.className = "four-instance-grid";
		Array.from({ length: Math.min(4, previewCount) }, (_, index) => index).forEach((index) => {
			grid.appendChild(instancePreviewButton(campaign, index));
		});
		four.querySelector(".fold-content").appendChild(grid);
		root.appendChild(four);

		const panel = makeFoldablePanel("figure", "preview-panel preview-instances", "All Instances");
		const instanceGrid = document.createElement("div");
		instanceGrid.className = "instance-grid";
		const showAll = options.sampleAll || previewCount <= PREVIEW_SAMPLE_THRESHOLD;
		const indices = showAll
			? Array.from({ length: previewCount }, (_, index) => index)
			: sampleInstanceIndices(previewCount, 20);
		indices.forEach((index) => instanceGrid.appendChild(instancePreviewButton(campaign, index)));
		const content = panel.querySelector(".fold-content");
		content.appendChild(instanceGrid);
		if (!showAll) {
			const loadAll = document.createElement("button");
			loadAll.type = "button";
			loadAll.className = "secondary preview-load-all";
			loadAll.textContent = `Load all ${previewCount} instances`;
			loadAll.addEventListener("click", () => {
				populateInstancePreviewPanels(root, campaign, { ...options, sampleAll: true });
			});
			content.appendChild(loadAll);
		}
		root.appendChild(panel);
		root.classList.remove("is-hidden");
	}

	function renderPreviewPanels(root, campaign, options = {}) {
		root.innerHTML = "";
		renderCanvasPlaceholder(root);
		populateInstancePreviewPanels(root, campaign, options).catch(() => {
			root.innerHTML = "";
			root.classList.add("is-hidden");
		});
	}

	return { populateInstancePreviewPanels, renderPreviewPanels };
}
