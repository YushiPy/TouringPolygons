export function createOsmPicker({ $, state, requestJSON, escapeHTML }) {
	function formatBytes(value) {
		if (!Number.isFinite(value)) {
			return "-";
		}
		const units = ["B", "KB", "MB", "GB"];
		let current = value;
		let unit = 0;
		while (current >= 1024 && unit < units.length - 1) {
			current /= 1024;
			unit += 1;
		}
		return `${current.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
	}

	function selectOsmFile(path) {
		document.querySelector('[name="pbf_path"]').value = path;
		$("#osm-file-status").textContent = path || "No file selected.";
		document.querySelectorAll("#osm-file-grid .choice-card").forEach((button) => {
			const active = button.dataset.value === path;
			button.classList.toggle("is-active", active);
			button.setAttribute("aria-selected", active ? "true" : "false");
		});
	}

	function renderOsmFiles() {
		const grid = $("#osm-file-grid");
		grid.innerHTML = "";
		const files = state.osmFiles.slice().sort((left, right) => right.size - left.size);
		if (files.length === 0) {
			grid.innerHTML = '<div class="empty-choice">No .osm.pbf files found.</div>';
			selectOsmFile("");
			return;
		}
		for (const file of files) {
			const button = document.createElement("button");
			button.type = "button";
			button.className = "choice-card osm-file-card";
			button.dataset.value = file.path;
			button.setAttribute("role", "option");
			button.innerHTML = `
      <strong>${escapeHTML(file.name)}</strong>
      <span>${formatBytes(file.size)}</span>
      <small>${escapeHTML(file.path)}</small>
    `;
			button.addEventListener("click", () => selectOsmFile(file.path));
			grid.appendChild(button);
		}
		selectOsmFile(files[0].path);
	}

	async function scanOsmFiles() {
		const status = $("#osm-file-status");
		state.osmScanStarted = true;
		status.textContent = "Scanning...";
		try {
			const data = await requestJSON("/api/osm-files");
			state.osmFiles = data.files;
			renderOsmFiles();
		} catch (error) {
			status.textContent = error.message;
		}
	}

	return { formatBytes, selectOsmFile, renderOsmFiles, scanOsmFiles };
}
