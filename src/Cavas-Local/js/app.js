import * as drawingStore from "./drawing-store.js";

let drawingName = null;
let drawingId = null;

export function setSavedDrawing(id, name) {
	drawingName = name;
	drawingId = id;

	const span = document.getElementById("drawing-name-text");
	if (span) {
		span.textContent = drawingName || "Untitled";
	}

	const saveOptionsButton = document.getElementById("save-options-button");
	if (saveOptionsButton) {
		if (id === null || id === undefined) {
			saveOptionsButton.setAttribute("disabled", "disabled");
		} else {
			saveOptionsButton.removeAttribute("disabled");
		}
	}
}

window.setSavedDrawing = setSavedDrawing;

function applyOverlay(id, show) {
	const overlay = document.getElementById(id);
	if (!overlay) return;
	overlay.style.display = show ? "flex" : "none";
	overlay.style.pointerEvents = show ? "auto" : "none";
	overlay.style.opacity = show ? "1" : "0";
}

export function toggleOverlay(id, show) {
	if (id === "rename-prompt" && show) {
		const input = document.getElementById("rename-prompt-input");
		if (input) input.value = drawingName || "";
	}
	if (id === "duplicate-prompt" && show) {
		const input = document.getElementById("duplicate-prompt-input");
		if (input) input.value = "";
	}
	applyOverlay(id, show);
}

window.toggleOverlay = toggleOverlay;

function toggleDropdown(el, show) {
	if (!el) return;
	if (show) {
		el.classList.remove("opacity-0", "pointer-events-none");
		el.classList.add("opacity-100");
	} else {
		el.classList.add("opacity-0", "pointer-events-none");
		el.classList.remove("opacity-100");
	}
}

function sanitizeFilename(name) {
	return String(name || "drawing")
		.replace(/[/\\?%*:|"<>]/g, "-")
		.trim() || "drawing";
}

function timeAgo(dateStr) {
	const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
	const intervals = [
		[31536000, "year"],
		[2592000, "month"],
		[604800, "week"],
		[86400, "day"],
		[3600, "hour"],
		[60, "minute"],
		[1, "second"],
	];

	for (const [secs, label] of intervals) {
		const count = Math.floor(seconds / secs);
		if (count >= 1) {
			const plural = count !== 1 ? "s" : "";
			return `${count} ${label}${plural} ago`;
		}
	}
	return "just now";
}

async function saveDrawing() {
	const input = document.getElementById("save-prompt-input");
	const name = input.value.trim();
	const scene = window.scene;
	const id = await scene.saveToDB(name);
	input.value = "";
	setSavedDrawing(id, name);
}

async function renameDrawing() {
	const input = document.getElementById("rename-prompt-input");
	const name = input.value.trim();
	const scene = window.scene;
	await scene.updateDrawingData(drawingId, name);
	input.value = "";
	setSavedDrawing(drawingId, name);
}

async function duplicateDrawing() {
	const input = document.getElementById("duplicate-prompt-input");
	const name = input.value.trim();
	const scene = window.scene;
	const id = await scene.saveToDB(name);
	input.value = "";
	setSavedDrawing(id, name);
}

window.saveDrawing = saveDrawing;
window.renameDrawing = renameDrawing;
window.duplicateDrawing = duplicateDrawing;

async function onSaveClick() {
	const scene = window.scene;
	if (!scene) return;

	if (drawingId != null) {
		await scene.updateDrawingData(drawingId, drawingName || "Untitled");
	} else {
		toggleOverlay("save-prompt", true);
		const input = document.getElementById("save-prompt-input");
		if (input) {
			input.value = drawingName || "";
			input.focus();
		}
	}
}

function buildLibraryHtml(items) {
	if (!items.length) {
		return `
<div id="library-backdrop" class="fixed inset-0 z-[1000] flex justify-center items-start bg-black/50 pt-[15vh] px-4" role="presentation">
  <div class="drawings-panel w-[min(900px,92vw)] max-h-[80vh] overflow-y-auto rounded-lg border-2 border-blue-600 bg-neutral-800 p-5 shadow-xl" onclick="event.stopPropagation()">
    <div class="mb-4 flex flex-row items-center justify-between text-neutral-100">
      <span class="text-2xl font-medium">Saved drawings</span>
      <button type="button" id="new-drawing-from-library" class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 active:bg-blue-700">New drawing</button>
    </div>
    <p class="text-neutral-300">No drawings in your local library yet. Use &quot;Save&quot; to store in this browser or import a JSON file.</p>
  </div>
</div>`;
	}

	const cards = items
		.map(
			(d) => `
<div class="drawing-card group relative cursor-pointer overflow-hidden rounded-lg transition hover:scale-[1.02]" data-id="${d.id}">
  <div class="drawing-thumbnail">
    <img src="${d.dataURL}" alt="" class="aspect-video w-full object-cover" loading="lazy">
  </div>
  <div class="flex flex-col gap-1 p-2">
    <span class="drawing-title truncate text-sm font-medium text-neutral-100">${escapeHtml(d.drawing_name)}</span>
    <span class="drawing-date text-xs text-neutral-400" data-date="${d.modified_at}">${timeAgo(d.modified_at)}</span>
  </div>
  <button type="button" class="drawing-delete absolute right-2 top-2 flex h-[26px] w-[26px] items-center justify-center rounded-md bg-black/45 p-0 text-white opacity-0 transition hover:bg-red-600 group-hover:opacity-100" data-delete-id="${d.id}" aria-label="Delete">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>
  </button>
</div>`
		)
		.join("");

	return `
<div id="library-backdrop" class="fixed inset-0 z-[1000] flex justify-center items-start bg-black/50 pt-[15vh] px-4" role="presentation">
  <div class="drawings-panel w-[min(900px,92vw)] max-h-[80vh] overflow-y-auto rounded-lg border-2 border-blue-600 bg-neutral-800 p-5 shadow-xl" onclick="event.stopPropagation()">
    <div class="mb-4 flex flex-row items-center justify-between gap-3 text-neutral-100">
      <span class="text-2xl font-medium">Saved drawings</span>
      <button type="button" id="new-drawing-from-library" class="shrink-0 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500">New drawing</button>
    </div>
    <input id="drawings-search" type="search" placeholder="Search…" class="mb-4 w-full rounded-md border border-blue-600 bg-neutral-900 px-3 py-2 text-neutral-100 placeholder:text-neutral-500">
    <div class="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4">${cards}</div>
  </div>
</div>`;
}

function escapeHtml(s) {
	const div = document.createElement("div");
	div.textContent = s;
	return div.innerHTML;
}

async function openLibrary() {
	const overlay = document.getElementById("saved-drawings-overlay");
	const items = await drawingStore.listDrawingsMeta();
	overlay.innerHTML = buildLibraryHtml(items);
	overlay.style.display = "flex";
	overlay.style.opacity = "1";
	overlay.style.pointerEvents = "auto";

	const backdrop = document.getElementById("library-backdrop");
	backdrop?.addEventListener("click", (e) => {
		if (e.target === backdrop) closeLibrary();
	});

	const search = document.getElementById("drawings-search");
	if (search) {
		search.addEventListener("input", () => {
			const q = search.value.toLowerCase();
			let visible = 0;
			overlay.querySelectorAll(".drawing-card").forEach((card) => {
				const title = card.querySelector(".drawing-title")?.textContent?.toLowerCase() ?? "";
				const show = title.includes(q);
				card.style.display = show ? "" : "none";
				if (show) visible++;
			});
		});
	}

	overlay.querySelector("#new-drawing-from-library")?.addEventListener("click", () => {
		window.scene.clear();
		closeLibrary();
		setSavedDrawing(null, null);
	});

	overlay.querySelectorAll(".drawing-card").forEach((card) => {
		card.addEventListener("click", (e) => {
			if (e.target.closest(".drawing-delete")) return;
			const id = Number(card.dataset.id);
			loadDrawingFromStore(id);
		});
	});

	overlay.querySelectorAll(".drawing-delete").forEach((btn) => {
		btn.addEventListener("click", (e) => {
			e.stopPropagation();
			const id = Number(btn.dataset.deleteId);
			showDeleteModal(id);
		});
	});
}

function closeLibrary() {
	const overlay = document.getElementById("saved-drawings-overlay");
	overlay.innerHTML = "";
	overlay.style.opacity = "0";
	overlay.style.pointerEvents = "none";
	overlay.style.display = "none";
}

async function loadDrawingFromStore(id) {
	try {
		const data = await drawingStore.getDrawing(id);
		window.scene.loadFromData(data);
		closeLibrary();
		setSavedDrawing(id, data.drawingName ?? null);
	} catch (err) {
		console.error(err);
		alert("Could not load the drawing.");
	}
}

function showDeleteModal(idToDelete) {
	const modal = document.getElementById("delete-modal");
	const cancel = document.getElementById("delete-cancel");
	const confirm = document.getElementById("delete-confirm");

	modal.classList.remove("hidden");
	modal.classList.add("flex");

	function close() {
		modal.classList.add("hidden");
		modal.classList.remove("flex");
	}

	function onBackdrop(e) {
		if (e.target === modal) close();
	}

	async function onConfirm() {
		try {
			await drawingStore.deleteDrawing(idToDelete);
			document.querySelector(`.drawing-card[data-id="${idToDelete}"]`)?.remove();
			if (idToDelete === drawingId) {
				setSavedDrawing(null, null);
			}
		} catch {
			alert("Could not delete the drawing.");
		}
		close();
	}

	modal.addEventListener("click", onBackdrop, { once: true });
	cancel.addEventListener("click", close, { once: true });
	confirm.addEventListener("click", onConfirm, { once: true });
}

function exportDrawingFile() {
	const scene = window.scene;
	if (!scene) return;
	const name = drawingName || "untitled";
	const json = scene.saveData(name, { includePreview: false });
	const blob = new Blob([json], { type: "application/json" });
	const a = document.createElement("a");
	const url = URL.createObjectURL(blob);
	a.href = url;
	a.download = `${sanitizeFilename(name)}.tpp.json`;
	a.click();
	URL.revokeObjectURL(url);
}

function triggerImport(inputId) {
	document.getElementById(inputId)?.click();
}

async function onImportFile(event) {
	const file = event.target.files?.[0];
	event.target.value = "";
	if (!file) return;

	try {
		const text = await file.text();
		const data = JSON.parse(text);
		window.scene.loadFromData(data);
		setSavedDrawing(null, data.drawingName ?? null);
	} catch {
		alert("Invalid JSON file.");
	}
}

function openTPPinfo() {
	window.open("./tpp-info.html", "_blank", "noopener,noreferrer");
}

function initSaveDropdown() {
	const button = document.querySelector(".save-options-button");
	const dropdown = document.querySelector("#save-dropdown-menu");
	if (!button || !dropdown) return;

	toggleDropdown(dropdown, false);

	button.addEventListener("click", (event) => {
		event.stopPropagation();
		const hidden = dropdown.classList.contains("opacity-0");
		toggleDropdown(dropdown, hidden);
	});

	document.body.addEventListener("click", () => toggleDropdown(dropdown, false));
}

function initVertexLineToggle() {
	const vertexLineToggle = document.getElementById("vertex-line-toggle");
	const vertexLineToggleOnIcon = document.getElementById("vertex-line-toggle-on");
	const vertexLineToggleOffIcon = document.getElementById("vertex-line-toggle-off");
	if (!vertexLineToggle || !vertexLineToggleOnIcon || !vertexLineToggleOffIcon) return;

	vertexLineToggle.toggleIcons = () => {
		const isActive = vertexLineToggle.classList.contains("active");
		vertexLineToggleOnIcon.classList.toggle("hidden", !isActive);
		vertexLineToggleOffIcon.classList.toggle("hidden", isActive);
	};
	vertexLineToggle.toggleIcons();
}

document.addEventListener("DOMContentLoaded", () => {
	initVertexLineToggle();

	document.getElementById("folder-button")?.addEventListener("click", (e) => {
		e.stopPropagation();
		openLibrary();
	});

	document.getElementById("save-drawing-button")?.addEventListener("click", onSaveClick);

	document.getElementById("export-button")?.addEventListener("click", exportDrawingFile);

	document.getElementById("import-input")?.addEventListener("change", onImportFile);
	document.getElementById("import-button")?.addEventListener("click", () => triggerImport("import-input"));

	document.getElementById("logo-button")?.addEventListener("click", openTPPinfo);

	initSaveDropdown();

	const overlays = ["save-prompt", "rename-prompt", "duplicate-prompt"];
	for (const id of overlays) {
		toggleOverlay(id, false);
		const root = document.getElementById(id);
		const promptBox = root?.querySelector(".prompt-box");
		root?.addEventListener("click", (event) => {
			if (promptBox && !promptBox.contains(event.target)) {
				toggleOverlay(id, false);
			}
		});
		const submitButton = root?.querySelector(".submit-button");
		submitButton?.addEventListener("click", () => toggleOverlay(id, false));
	}

	setSavedDrawing(null, null);
});
