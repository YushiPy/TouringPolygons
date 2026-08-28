const state = {
  campaigns: [],
  currentJob: null,
  currentRunJob: null,
  currentComparisonJob: null,
  cpuCount: 1,
  selectedCampaign: "",
  selectedComparisonCampaign: "",
  osmFiles: [],
  resultFiles: [],
  recentJobs: [],
  osmScanStarted: false,
  benchmarkedInstances: new Map(),
  benchmarkedSort: "case",
  campaignFilter: "",
  resultFilter: "",
};

const $ = (selector) => document.querySelector(selector);

let pendingConfirmation = null;

function formData(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function boolField(form, name) {
  const input = form.querySelector(`[name="${name}"]`);
  return input.type === "checkbox" ? input.checked : input.value === "1";
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.output || `Request failed: ${response.status}`);
  }
  return data;
}

function setOutput(target, text) {
  target.textContent = text || "";
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadCSV(filename, rows) {
  if (!rows || rows.length === 0) {
    return;
  }
  const headers = Object.keys(rows[0]);
  const csv = [
    headers.map(csvCell).join(","),
    ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(",")),
  ].join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function instanceLabel(index) {
  return Number(index) + 1;
}

function formatElapsed(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remaining = value - minutes * 60;
  if (minutes > 0) {
    return `${minutes}:${remaining.toFixed(1).padStart(4, "0")}`;
  }
  return `${remaining.toFixed(1)}s`;
}

function formatSeconds(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (Math.abs(number) >= 1) {
    return `${number.toFixed(3)} s`;
  }
  if (Math.abs(number) >= 0.001) {
    return `${(number * 1000).toFixed(2)} ms`;
  }
  return `${(number * 1000000).toFixed(2)} us`;
}

function formatMicroseconds(seconds) {
  if (seconds === null || seconds === undefined || seconds === "") {
    return "-";
  }
  const number = Number(seconds);
  if (!Number.isFinite(number)) {
    return "-";
  }
  const microseconds = number * 1000000;
  if (Math.abs(microseconds) >= 100) {
    return `${microseconds.toFixed(1)} us`;
  }
  if (Math.abs(microseconds) >= 1) {
    return `${microseconds.toFixed(2)} us`;
  }
  return `${microseconds.toFixed(3)} us`;
}

function switchPanel(panelId) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.panel === panelId);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === panelId);
  });
}

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

function setupSegmentedControls() {
  document.querySelectorAll(".segmented").forEach((group) => {
    const input = document.querySelector(`[name="${group.dataset.input}"]`);
    group.querySelectorAll(".segment").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.value;
        group.querySelectorAll(".segment").forEach((item) => {
          item.classList.toggle("is-active", item === button);
        });
        if (group.dataset.input === "campaign_type") {
          updateCreateMode();
        }
      });
    });
  });
}

function setupToggleButtons() {
  document.querySelectorAll(".toggle-button").forEach((button) => {
    const input = document.querySelector(`[name="${button.dataset.input}"]`);
    button.addEventListener("click", () => {
      const active = input.value !== "1";
      input.value = active ? "1" : "";
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
  });
}

function campaignExists(name) {
  return state.campaigns.some((campaign) => campaign.name === name);
}

function campaignInstanceTotal(name) {
  const campaign = state.campaigns.find((item) => item.name === name);
  if (!campaign) {
    return 1;
  }
  const generation = campaign.generation || {};
  return Math.max(
    1,
    Number(campaign.instance_progress?.total)
      || Number(generation.instances)
      || Number(generation.instances_per_file)
      || 1,
  );
}

function updateCampaignNameIndicator() {
  const name = $("#create-name").value.trim();
  const exists = campaignExists(name);
  const indicator = $("#campaign-name-indicator");
  const overwriteInput = document.querySelector('[name="overwrite"]');
  const submit = $("#create-submit");
  overwriteInput.value = exists ? "1" : "";
  submit.textContent = exists ? "Overwrite" : "Create";
  indicator.textContent = exists
    ? "A campaign with this name already exists. Creating will overwrite it."
    : "";
  indicator.classList.toggle("is-warning", exists);
}

function setupThreadsControl(sliderSelector = "#threads-slider", inputSelector = "#threads-input", maxLabelSelector = "#threads-max-label") {
  const slider = $(sliderSelector);
  const input = $(inputSelector);
  const maxLabel = $(maxLabelSelector);

  function clamp(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return 1;
    }
    return Math.max(1, Math.min(state.cpuCount, Math.round(parsed)));
  }

  function setThreads(value) {
    const clamped = clamp(value);
    slider.max = String(state.cpuCount);
    input.max = String(state.cpuCount);
    slider.value = String(clamped);
    input.value = String(clamped);
    maxLabel.textContent = String(state.cpuCount);
    slider.style.setProperty("--value", String(clamped));
    slider.style.setProperty("--max", String(state.cpuCount));
    slider.style.setProperty("--progress", `${((clamped - 1) / Math.max(1, state.cpuCount - 1)) * 100}%`);
    slider.style.setProperty("--tick-step", `${100 / Math.max(1, state.cpuCount - 1)}%`);
  }

  slider.addEventListener("input", () => setThreads(slider.value));
  input.addEventListener("input", () => setThreads(input.value));
  input.addEventListener("blur", () => setThreads(input.value));
  setThreads(state.cpuCount);
}

function setupBoundedSliders() {
  document.querySelectorAll("[data-range-for]").forEach((slider) => {
    const input = document.querySelector(`[name="${slider.dataset.rangeFor}"]`);
    if (!input) {
      return;
    }
    const min = Number(slider.min);
    const max = Number(slider.max);
    const step = Number(slider.step) || 1;
    const integer = Number.isInteger(step) && step >= 1;

    function clamp(value) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) {
        return Number(input.value || slider.value || min);
      }
      const bounded = Math.max(min, Math.min(max, parsed));
      const snapped = Math.round((bounded - min) / step) * step + min;
      return integer ? Math.round(snapped) : Number(snapped.toFixed(6));
    }

    function setValue(value) {
      const clamped = clamp(value);
      slider.value = String(clamped);
      input.value = String(clamped);
      slider.style.setProperty("--progress", `${((clamped - min) / Math.max(step, max - min)) * 100}%`);
    }

    slider.addEventListener("input", () => setValue(slider.value));
    input.addEventListener("input", () => setValue(input.value));
    input.addEventListener("blur", () => setValue(input.value));
    setValue(input.value || slider.value);
  });

  const polygonSize = document.querySelector('[name="grid_polygon_size"]');
  const cellSize = document.querySelector('[name="grid_cell_size"]');
  if (!polygonSize || !cellSize) {
    return;
  }
  function clampGridCell() {
    const polygon = Number(polygonSize.value);
    const cell = Number(cellSize.value);
    if (!Number.isFinite(polygon) || !Number.isFinite(cell)) {
      return;
    }
    if (cell <= polygon) {
      cellSize.value = String(Number((polygon + 0.1).toFixed(6)));
      const slider = document.querySelector('[data-range-for="grid_cell_size"]');
      if (slider) {
        slider.value = cellSize.value;
      }
    }
  }
  polygonSize.addEventListener("input", clampGridCell);
  cellSize.addEventListener("blur", clampGridCell);
  clampGridCell();
}

function setupMaxInstancesControl() {
  const slider = $("#max-instances-slider");
  const input = $("#max-instances-input");
  const maxLabel = $("#max-instances-label");

  function clamp(value) {
    const max = campaignInstanceTotal(state.selectedCampaign);
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return max;
    }
    return Math.max(1, Math.min(max, Math.round(parsed)));
  }

  function setValue(value = null) {
    const max = campaignInstanceTotal(state.selectedCampaign);
    const clamped = clamp(value ?? max);
    slider.max = String(max);
    input.max = String(max);
    slider.value = String(clamped);
    input.value = String(clamped);
    maxLabel.textContent = String(max);
    slider.style.setProperty("--progress", `${((clamped - 1) / Math.max(1, max - 1)) * 100}%`);
    slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
  }

  slider.addEventListener("input", () => setValue(slider.value));
  input.addEventListener("input", () => setValue(input.value));
  input.addEventListener("blur", () => setValue(input.value));
  setValue();
}

function resetMaxInstancesControl() {
  const slider = $("#max-instances-slider");
  const input = $("#max-instances-input");
  const maxLabel = $("#max-instances-label");
  const max = campaignInstanceTotal(state.selectedCampaign);
  slider.max = String(max);
  input.max = String(max);
  slider.value = String(max);
  input.value = String(max);
  maxLabel.textContent = String(max);
  slider.style.setProperty("--progress", "100%");
  slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
}

function setupCompareMaxInstancesControl() {
  const slider = $("#compare-max-instances-slider");
  const input = $("#compare-max-instances-input");
  const maxLabel = $("#compare-max-instances-label");

  function clamp(value) {
    const max = campaignInstanceTotal(state.selectedComparisonCampaign);
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return max;
    }
    return Math.max(1, Math.min(max, Math.round(parsed)));
  }

  function setValue(value = null) {
    const max = campaignInstanceTotal(state.selectedComparisonCampaign);
    const clamped = clamp(value ?? max);
    slider.max = String(max);
    input.max = String(max);
    slider.value = String(clamped);
    input.value = String(clamped);
    maxLabel.textContent = String(max);
    slider.style.setProperty("--progress", `${((clamped - 1) / Math.max(1, max - 1)) * 100}%`);
    slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
  }

  slider.addEventListener("input", () => setValue(slider.value));
  input.addEventListener("input", () => setValue(input.value));
  input.addEventListener("blur", () => setValue(input.value));
  setValue();
}

function setupFilterInput(selector, stateKey, render) {
  const input = $(selector);
  if (!input) {
    return;
  }
  input.addEventListener("input", () => {
    state[stateKey] = input.value.trim().toLowerCase();
    render();
  });
}

function resetCompareMaxInstancesControl() {
  const slider = $("#compare-max-instances-slider");
  const input = $("#compare-max-instances-input");
  const maxLabel = $("#compare-max-instances-label");
  const max = campaignInstanceTotal(state.selectedComparisonCampaign);
  slider.max = String(max);
  input.max = String(max);
  slider.value = String(max);
  input.value = String(max);
  maxLabel.textContent = String(max);
  slider.style.setProperty("--progress", "100%");
  slider.style.setProperty("--tick-step", `${100 / Math.max(1, max - 1)}%`);
}

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
  selectOsmFile(state.osmFiles[0].path);
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

function renderCampaignChoiceGrid(grid, selectedName, onSelect) {
  grid.innerHTML = "";
  for (const campaign of state.campaigns) {
    const button = document.createElement("button");
    const generation = campaign.generation || {};
    button.type = "button";
    button.className = "choice-card";
    button.dataset.value = campaign.name;
    button.setAttribute("role", "option");
    button.innerHTML = `
      <strong>${escapeHTML(campaign.name)}</strong>
      <span>${escapeHTML(campaign.type)}</span>
      <small>${campaign.inputs.existing}/${campaign.inputs.total} input files</small>
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

function renderCampaignOptions() {
  const input = $("#run-campaign");
  const selected = state.selectedCampaign || input.value;
  const fallback = state.campaigns[0]?.name || "";
  const nextSelected = state.campaigns.some((campaign) => campaign.name === selected) ? selected : fallback;
  selectRunCampaign(nextSelected, { resetCap: nextSelected !== state.selectedCampaign });
  selectComparisonCampaign(state.campaigns.some((campaign) => campaign.name === state.selectedComparisonCampaign) ? state.selectedComparisonCampaign : fallback);
}

function selectRunCampaign(name, options = {}) {
  const resetCap = options.resetCap ?? true;
  state.selectedCampaign = name;
  $("#run-campaign").value = name;
  renderCampaignChoiceGrid($("#run-campaign-grid"), name, selectRunCampaign);
  if (resetCap) {
    resetMaxInstancesControl();
  }
  renderRunSummary();
  if (name) {
    refreshBenchmarkReport(name);
  }
}

function selectComparisonCampaign(name) {
  state.selectedComparisonCampaign = name;
  $("#compare-campaign").value = name;
  renderCampaignChoiceGrid($("#compare-campaign-grid"), name, selectComparisonCampaign);
  resetCompareMaxInstancesControl();
  if (name) {
    refreshComparisonReport(name);
  }
}

function runProgress(campaign) {
  const progress = campaign.instance_progress || {};
  const total = progress.total || 0;
  const completed = progress.completed || 0;
  if (total === 0) {
    return { label: "not started", ratio: 0, total: 0, completed: 0, counts: campaign.run_index.counts || {} };
  }
  return {
    label: `${completed}/${total} instances`,
    ratio: progress.ratio || completed / total,
    total,
    completed,
    counts: campaign.run_index.counts || {},
  };
}

function describeVertices(generation) {
  const vertices = generation.vertices;
  if (!Array.isArray(vertices) || vertices.length === 0) {
    return "-";
  }
  if (new Set(vertices).size === 1) {
    return `${vertices[0]} each`;
  }
  return vertices.join(",");
}

function previewUrl(campaign, kind) {
  if (!campaign.previews || !campaign.previews[kind]) {
    return null;
  }
  return `/api/campaigns/${encodeURIComponent(campaign.name)}/preview/${kind}?v=${campaign.version || ""}`;
}

function instancePreviewUrl(campaign, index) {
  return `/api/campaigns/${encodeURIComponent(campaign.name)}/preview/instance-${index}?v=${campaign.version || ""}`;
}

function detailPreviewUrl(campaign, kind) {
  return `/api/campaigns/${encodeURIComponent(campaign.name)}/preview/${kind}?v=${campaign.version || ""}`;
}

function solutionPreviewUrl(campaign, item) {
  return `/api/campaigns/${encodeURIComponent(campaign.name)}/solution-preview/${item.case_index}?repeat_index=${item.repeat_index}&v=${campaign.version || ""}`;
}

function zoomableImage(src, alt, extraClass = "", options = {}) {
  return `
    <figure class="instance-detail zoomable-detail ${extraClass}">
      <div class="zoom-toolbar" aria-label="Zoom controls">
        <div class="zoom-toolbar-toggles">
          ${options.inspectable ? '<button class="secondary zoom-toggle has-dropdown" type="button" data-style-panel aria-pressed="false" aria-expanded="false">Style</button>' : ""}
          ${options.inspectable ? '<button class="secondary zoom-toggle" type="button" data-toggle-grid aria-pressed="false">Grid</button>' : ""}
          ${options.inspectable ? '<button class="secondary zoom-toggle is-active" type="button" data-toggle-endpoints aria-pressed="true">S/T</button>' : ""}
          ${options.inspectable ? '<button class="secondary zoom-toggle" type="button" data-toggle-bounces aria-pressed="false">Bounces</button>' : ""}
          ${options.inspectable ? '<button class="secondary zoom-toggle is-active" type="button" data-toggle-pieces aria-pressed="true">Pieces</button>' : ""}
        </div>
        <div class="zoom-toolbar-main">
          <button class="secondary" type="button" data-zoom-out>-</button>
          <button class="secondary" type="button" data-zoom-reset>Fit</button>
          <button class="secondary" type="button" data-zoom-in>+</button>
        </div>
      </div>
      ${options.inspectable ? `
        <div class="zoom-style-panel is-hidden">
          <label>
            Grid step
            <input data-inspect-grid-step type="number" min="4" max="200" step="1" value="40">
          </label>
          <label>
            Point radius
            <input data-inspect-point-radius type="number" min="1" max="24" step="1" value="6">
          </label>
          <label>
            Path width
            <input data-inspect-path-width type="number" min="0.5" max="16" step="0.5" value="3">
          </label>
          <label>
            Polygon opacity
            <input data-inspect-polygon-opacity type="range" min="0" max="0.8" step="0.02" value="0.24">
          </label>
          <label class="switch-row">
            <input data-inspect-path-dashed type="checkbox">
            <span class="switch-control"></span>
            <span>Dashed path</span>
          </label>
        </div>
      ` : ""}
      <div class="zoom-viewport">
        <img src="${src}" alt="${alt}" draggable="false" data-zoom-src="${src}">
      </div>
    </figure>
  `;
}

function parsePointList(value) {
  return String(value || "").trim().split(/\s+/)
    .map((pair) => pair.split(",").map(Number))
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    .map(([x, y]) => ({ x, y }));
}

function svgViewBox(svg) {
  const raw = svg.getAttribute("viewBox");
  if (raw) {
    const values = raw.trim().split(/[\s,]+/).map(Number);
    if (values.length === 4 && values.every(Number.isFinite)) {
      return { x: values[0], y: values[1], width: values[2], height: values[3] };
    }
  }
  return {
    x: 0,
    y: 0,
    width: Number(svg.getAttribute("width")) || 1,
    height: Number(svg.getAttribute("height")) || 1,
  };
}

function extendBounds(bounds, point) {
  bounds.minX = Math.min(bounds.minX, point.x);
  bounds.maxX = Math.max(bounds.maxX, point.x);
  bounds.minY = Math.min(bounds.minY, point.y);
  bounds.maxY = Math.max(bounds.maxY, point.y);
}

function svgGeometryBounds(svg) {
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  svg.querySelectorAll("polygon, polyline").forEach((element) => {
    parsePointList(element.getAttribute("points")).forEach((point) => extendBounds(bounds, point));
  });
  svg.querySelectorAll("circle").forEach((circle) => {
    const x = Number(circle.getAttribute("cx"));
    const y = Number(circle.getAttribute("cy"));
    if (Number.isFinite(x) && Number.isFinite(y)) {
      extendBounds(bounds, { x, y });
    }
  });
  if (!Number.isFinite(bounds.minX)) {
    const viewBox = svgViewBox(svg);
    return { minX: viewBox.x, minY: viewBox.y, maxX: viewBox.x + viewBox.width, maxY: viewBox.y + viewBox.height };
  }
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
  const slack = span * 0.08;
  return {
    minX: bounds.minX - slack,
    minY: bounds.minY - slack,
    maxX: bounds.maxX + slack,
    maxY: bounds.maxY + slack,
  };
}

function setupInspectableSvg(root, svg, controls) {
  const ns = "http://www.w3.org/2000/svg";
  const gridGroup = document.createElementNS(ns, "g");
  gridGroup.classList.add("inspection-grid");
  gridGroup.dataset.inspectionOverlay = "grid";
  const background = svg.querySelector("rect");
  if (background) {
    background.after(gridGroup);
  } else {
    svg.prepend(gridGroup);
  }

  const bounceGroup = document.createElementNS(ns, "g");
  bounceGroup.dataset.inspectionOverlay = "bounces";
  svg.appendChild(bounceGroup);

  const polygons = [...svg.querySelectorAll("polygon[fill]:not([fill='none'])")];
  const piecePolygons = [...svg.querySelectorAll("polygon[fill='none']")];
  const path = svg.querySelector("polyline");
  const endpoints = [...svg.querySelectorAll("circle")].slice(-2);
  const endpointLabels = [...svg.querySelectorAll("text")].slice(-2);
  const pathPoints = path ? parsePointList(path.getAttribute("points")) : [];

  function renderGrid() {
    gridGroup.replaceChildren();
    if (!controls.showGrid) {
      return;
    }
    const viewBox = svgViewBox(svg);
    const step = Math.max(4, Number(controls.gridStep.value) || 40);
    const minX = Math.floor(viewBox.x / step) * step;
    const maxX = viewBox.x + viewBox.width;
    const minY = Math.floor(viewBox.y / step) * step;
    const maxY = viewBox.y + viewBox.height;
    for (let x = minX; x <= maxX; x += step) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", x);
      line.setAttribute("x2", x);
      line.setAttribute("y1", viewBox.y);
      line.setAttribute("y2", maxY);
      gridGroup.appendChild(line);
    }
    for (let y = minY; y <= maxY; y += step) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", viewBox.x);
      line.setAttribute("x2", maxX);
      line.setAttribute("y1", y);
      line.setAttribute("y2", y);
      gridGroup.appendChild(line);
    }
  }

  function renderBounces() {
    bounceGroup.replaceChildren();
    if (!controls.showBounces || pathPoints.length <= 2) {
      return;
    }
    const radius = Math.max(1, Number(controls.pointRadius.value) || 6) * 0.72;
    pathPoints.slice(1, -1).forEach((point) => {
      const circle = document.createElementNS(ns, "circle");
      circle.classList.add("inspection-bounce-point");
      circle.setAttribute("cx", point.x);
      circle.setAttribute("cy", point.y);
      circle.setAttribute("r", radius);
      bounceGroup.appendChild(circle);
    });
  }

  function applyStyle() {
    const radius = Math.max(1, Number(controls.pointRadius.value) || 6);
    const strokeWidth = Math.max(0.5, Number(controls.pathWidth.value) || 3);
    const opacity = Math.max(0, Math.min(0.8, Number(controls.polygonOpacity.value) || 0));
    controls.polygonOpacity?.style.setProperty("--progress", `${(opacity / 0.8) * 100}%`);
    polygons.forEach((polygon) => polygon.setAttribute("fill-opacity", opacity));
    piecePolygons.forEach((polygon) => {
      polygon.style.display = controls.showPieces ? "" : "none";
    });
    endpoints.forEach((circle) => {
      circle.style.display = controls.showEndpoints ? "" : "none";
      circle.setAttribute("r", radius);
    });
    endpointLabels.forEach((label) => {
      label.style.display = controls.showEndpoints ? "" : "none";
    });
    if (path) {
      path.setAttribute("stroke-width", strokeWidth);
      path.setAttribute("stroke-dasharray", controls.pathDashed.checked ? `${strokeWidth * 3} ${strokeWidth * 2}` : "");
    }
    renderGrid();
    renderBounces();
  }

  [controls.gridStep, controls.pointRadius, controls.pathWidth, controls.polygonOpacity, controls.pathDashed].forEach((input) => {
    input?.addEventListener("input", applyStyle);
  });
  applyStyle();
  return applyStyle;
}

async function setupZoomableDetail(root) {
  const viewport = root.querySelector(".zoom-viewport");
  let image = root.querySelector(".zoom-viewport img");
  if (!viewport || !image) {
    return;
  }
  let allowedBounds = null;
  let viewBox = null;
  const inspectable = Boolean(root.querySelector("[data-style-panel]"));
  const controls = {
    showGrid: false,
    showEndpoints: true,
    showBounces: false,
    showPieces: true,
    gridStep: root.querySelector("[data-inspect-grid-step]"),
    pointRadius: root.querySelector("[data-inspect-point-radius]"),
    pathWidth: root.querySelector("[data-inspect-path-width]"),
    polygonOpacity: root.querySelector("[data-inspect-polygon-opacity]"),
    pathDashed: root.querySelector("[data-inspect-path-dashed]"),
  };
  let applySvgStyle = null;

  const originalAlt = image.alt;
  if (inspectable) {
    try {
      const response = await fetch(image.dataset.zoomSrc);
      if (!response.ok) {
        throw new Error("Failed to load SVG preview.");
      }
      const text = await response.text();
      const documentSvg = new DOMParser().parseFromString(text, "image/svg+xml").querySelector("svg");
      if (documentSvg) {
        image.replaceWith(document.importNode(documentSvg, true));
        image = viewport.querySelector("svg");
        image.removeAttribute("width");
        image.removeAttribute("height");
        image.setAttribute("role", "img");
        image.setAttribute("aria-label", originalAlt || "Solution detail");
        viewBox = svgViewBox(image);
        allowedBounds = svgGeometryBounds(image);
        applySvgStyle = setupInspectableSvg(root, image, controls);
      }
    } catch {
      allowedBounds = null;
    }
  }

  const state = { scale: 1, minScale: 0.5, x: 0, y: 0, dragging: false, lastX: 0, lastY: 0 };

  function baseRectForBounds(bounds) {
    if (!bounds || !viewBox) {
      return { left: 0, top: 0, right: viewport.clientWidth, bottom: viewport.clientHeight };
    }
    const scale = Math.min(viewport.clientWidth / viewBox.width, viewport.clientHeight / viewBox.height);
    const offsetX = (viewport.clientWidth - viewBox.width * scale) / 2;
    const offsetY = (viewport.clientHeight - viewBox.height * scale) / 2;
    return {
      left: offsetX + (bounds.minX - viewBox.x) * scale,
      right: offsetX + (bounds.maxX - viewBox.x) * scale,
      top: offsetY + (bounds.minY - viewBox.y) * scale,
      bottom: offsetY + (bounds.maxY - viewBox.y) * scale,
    };
  }

  function updateMinScale() {
    if (!allowedBounds || !viewBox) {
      state.minScale = 0.5;
      return;
    }
    const rect = baseRectForBounds(allowedBounds);
    state.minScale = Math.max(
      viewport.clientWidth / Math.max(1, rect.right - rect.left),
      viewport.clientHeight / Math.max(1, rect.bottom - rect.top),
      0.5,
    );
  }

  function clampPan() {
    if (!allowedBounds || !viewBox) {
      return;
    }
    const rect = baseRectForBounds(allowedBounds);
    const left = rect.left * state.scale + state.x;
    const right = rect.right * state.scale + state.x;
    const top = rect.top * state.scale + state.y;
    const bottom = rect.bottom * state.scale + state.y;
    if (right - left <= viewport.clientWidth) {
      state.x += (viewport.clientWidth - (left + right)) / 2;
    } else {
      state.x += Math.min(0, viewport.clientWidth - right);
      state.x += Math.max(0, -left);
    }
    if (bottom - top <= viewport.clientHeight) {
      state.y += (viewport.clientHeight - (top + bottom)) / 2;
    } else {
      state.y += Math.min(0, viewport.clientHeight - bottom);
      state.y += Math.max(0, -top);
    }
  }

  function render() {
    clampPan();
    image.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
  }

  function clampScale(value) {
    return Math.max(state.minScale, Math.min(12, value));
  }

  function zoomAt(nextScale, originX = viewport.clientWidth / 2, originY = viewport.clientHeight / 2) {
    const previous = state.scale;
    state.scale = clampScale(nextScale);
    const ratio = state.scale / previous;
    state.x = originX - (originX - state.x) * ratio;
    state.y = originY - (originY - state.y) * ratio;
    render();
  }

  function fitToBounds() {
    updateMinScale();
    state.scale = state.minScale;
    state.x = 0;
    state.y = 0;
    clampPan();
    render();
  }

  root.querySelector("[data-zoom-in]")?.addEventListener("click", () => zoomAt(state.scale * 1.25));
  root.querySelector("[data-zoom-out]")?.addEventListener("click", () => zoomAt(state.scale / 1.25));
  root.querySelector("[data-zoom-reset]")?.addEventListener("click", fitToBounds);
  root.querySelector("[data-style-panel]")?.addEventListener("click", (event) => {
    const panel = root.querySelector(".zoom-style-panel");
    if (!panel) {
      return;
    }
    const hidden = panel.classList.toggle("is-hidden");
    event.currentTarget.classList.toggle("is-active", !hidden);
    event.currentTarget.setAttribute("aria-pressed", !hidden ? "true" : "false");
    event.currentTarget.setAttribute("aria-expanded", !hidden ? "true" : "false");
  });
  [
    ["[data-toggle-grid]", "showGrid"],
    ["[data-toggle-endpoints]", "showEndpoints"],
    ["[data-toggle-bounces]", "showBounces"],
    ["[data-toggle-pieces]", "showPieces"],
  ].forEach(([selector, key]) => {
    root.querySelector(selector)?.addEventListener("click", (event) => {
      controls[key] = !controls[key];
      event.currentTarget.classList.toggle("is-active", controls[key]);
      event.currentTarget.setAttribute("aria-pressed", controls[key] ? "true" : "false");
      applySvgStyle?.();
    });
  });
  viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(state.scale * factor, event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });
  viewport.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!state.dragging) {
      return;
    }
    state.x += event.clientX - state.lastX;
    state.y += event.clientY - state.lastY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    render();
  });
  viewport.addEventListener("pointerup", () => {
    state.dragging = false;
  });
  viewport.addEventListener("pointercancel", () => {
    state.dragging = false;
  });
  window.addEventListener("resize", fitToBounds);
  fitToBounds();
}

function renderPreviewPanels(root, campaign) {
  root.innerHTML = "";
  const panels = [
    ["selected", "Selected Instance"],
    ["four", "Four Instances"],
  ];
  for (const [kind, title] of panels) {
    const url = previewUrl(campaign, kind);
    if (!url) {
      continue;
    }
    const panel = document.createElement("figure");
    panel.className = `preview-panel preview-${kind}`;
    panel.innerHTML = `
      <figcaption>${title}</figcaption>
      <img src="${url}" alt="${title} preview for ${campaign.name}">
    `;
    panel.classList.add("is-clickable");
    panel.tabIndex = 0;
    panel.addEventListener("click", () => openPreviewModal(campaign, kind, title));
    panel.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPreviewModal(campaign, kind, title);
      }
    });
    root.appendChild(panel);
  }
  if (root.children.length === 0 && campaign.instance_previews && campaign.instance_previews.length > 0) {
    const selected = document.createElement("figure");
    selected.className = "preview-panel preview-selected is-clickable";
    selected.tabIndex = 0;
    selected.innerHTML = `
      <figcaption>Selected Instance</figcaption>
      <img src="${instancePreviewUrl(campaign, 0)}" alt="Selected instance preview for ${campaign.name}">
    `;
    selected.addEventListener("click", () => openInstanceModal(campaign, 0));
    selected.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openInstanceModal(campaign, 0);
      }
    });
    const four = document.createElement("figure");
    four.className = "preview-panel preview-four";
    four.innerHTML = "<figcaption>Four Instances</figcaption>";
    const grid = document.createElement("div");
    grid.className = "four-instance-grid";
    campaign.instance_previews.slice(0, 4).forEach((_, index) => {
      const button = document.createElement("button");
      button.className = "instance-thumb";
      button.type = "button";
      button.innerHTML = `
        <img src="${instancePreviewUrl(campaign, index)}" alt="Instance ${instanceLabel(index)} preview">
        <span>${instanceLabel(index)}</span>
      `;
      button.addEventListener("click", () => openInstanceModal(campaign, index));
      grid.appendChild(button);
    });
    four.appendChild(grid);
    root.appendChild(selected);
    root.appendChild(four);
  }
  if (campaign.instance_previews && campaign.instance_previews.length > 0) {
    const panel = document.createElement("figure");
    panel.className = "preview-panel preview-instances";
    panel.innerHTML = "<figcaption>All Instances</figcaption>";
    const grid = document.createElement("div");
    grid.className = "instance-grid";
    campaign.instance_previews.forEach((_, index) => {
      const button = document.createElement("button");
      button.className = "instance-thumb";
      button.type = "button";
      button.innerHTML = `
        <img src="${instancePreviewUrl(campaign, index)}" alt="Instance ${instanceLabel(index)} preview">
        <span>${instanceLabel(index)}</span>
      `;
      button.addEventListener("click", () => openInstanceModal(campaign, index));
      grid.appendChild(button);
    });
    panel.appendChild(grid);
    root.appendChild(panel);
  }
  root.classList.toggle("is-hidden", root.children.length === 0);
}

function renderBenchmarkedInstanceSection(root, campaign, instances) {
  if (!campaign || instances.length === 0) {
    root.innerHTML = "";
    root.classList.add("is-hidden");
    return;
  }
  root.innerHTML = `
    <header class="section-subheader">
      <div>
        <h3>Benchmarked Instances</h3>
        <p>Completed rows with available previews and benchmark metrics.</p>
      </div>
      <label class="compact-select">
        Sort
        <select data-benchmarked-sort>
          <option value="case" ${state.benchmarkedSort === "case" ? "selected" : ""}>Case</option>
          <option value="time" ${state.benchmarkedSort === "time" ? "selected" : ""}>Solve time</option>
          <option value="calls" ${state.benchmarkedSort === "calls" ? "selected" : ""}>Convex calls</option>
        </select>
      </label>
    </header>
    <div class="benchmarked-grid"></div>
  `;
  const grid = root.querySelector(".benchmarked-grid");
  root.querySelector("[data-benchmarked-sort]")?.addEventListener("change", (event) => {
    state.benchmarkedSort = event.currentTarget.value;
    renderBenchmarkedInstanceSection(root, campaign, instances);
  });
  const sortedInstances = instances.slice().sort((left, right) => {
    if (state.benchmarkedSort === "time") {
      return (parseNumber(right.total_seconds) || 0) - (parseNumber(left.total_seconds) || 0);
    }
    if (state.benchmarkedSort === "calls") {
      return (parseNumber(right.calls) || 0) - (parseNumber(left.calls) || 0);
    }
    return Number(left.case_index) - Number(right.case_index);
  });
  for (const item of sortedInstances) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "benchmarked-card";
    const preview = item.solution_available
      ? `<img src="${solutionPreviewUrl(campaign, item)}" alt="Solved instance ${instanceLabel(item.case_index)} with path and decomposition">`
      : item.preview
        ? `<img src="${instancePreviewUrl(campaign, item.case_index)}" alt="Benchmarked instance ${instanceLabel(item.case_index)}">`
      : '<div class="missing-preview">No preview</div>';
    button.innerHTML = `
      ${preview}
      <div class="benchmarked-meta">
        <strong>Case ${instanceLabel(item.case_index)}</strong>
        <span class="status-pill ${item.status === "solved" ? "is-solved" : "is-capped"}">${item.status}</span>
        <small>final ${shortNumber(item.final_length)}</small>
        <small>${formatSeconds(parseNumber(item.total_seconds))}</small>
        <small>${escapeHTML(item.calls ?? "-")} calls</small>
        <small>${item.solution_available ? "path + pieces" : `${escapeHTML(item.decomposed_pieces ?? "-")} pieces`}</small>
      </div>
    `;
    button.addEventListener("click", () => openBenchmarkedInstanceModal(campaign, item));
    grid.appendChild(button);
  }
  root.classList.remove("is-hidden");
}

function renderSolvedPreview(campaign) {
  const root = $("#solved-preview");
  const instances = campaign ? state.benchmarkedInstances.get(campaign.name) || [] : [];
  renderBenchmarkedInstanceSection(root, campaign, instances);
}

function renderCampaigns() {
  const root = $("#campaign-list");
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
      <button class="campaign-delete" type="button" data-delete-campaign="${escapeHTML(campaign.name)}" aria-label="Delete ${escapeHTML(campaign.name)}">x</button>
      <h3>${escapeHTML(campaign.name)}</h3>
      <div class="meta">
        <div><span>Type</span><br>${escapeHTML(campaign.type)}</div>
        <div><span>Input files</span><br>${campaign.inputs.existing}/${campaign.inputs.total}</div>
        <div><span>Instances</span><br>${generation.instances ?? generation.instances_per_file ?? "-"}</div>
        <div><span>Polygon Count</span><br>${generation.polygons ?? generation.polygon_counts ?? "-"}</div>
        <div><span>Vertices</span><br>${escapeHTML(describeVertices(generation))}</div>
        <div><span>Progress</span><br>${progress.label}</div>
      </div>
      <div class="bar" aria-label="Benchmark progress">
        <div class="bar-fill" style="width: ${Math.round(progress.ratio * 100)}%"></div>
      </div>
      ${campaign.has_preview ? `<img class="preview" src="/api/campaigns/${encodeURIComponent(campaign.name)}/preview?v=${campaign.version || ""}" alt="Preview for ${campaign.name}">` : ""}
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

function renderResults(files) {
  const root = $("#result-list");
  root.innerHTML = "";
  const query = state.resultFilter;
  const filtered = files
    .filter((file) => !query || file.path.toLowerCase().includes(query))
    .slice()
    .sort((left, right) => right.mtime - left.mtime);
  if (filtered.length === 0) {
    root.textContent = "No result files found.";
    return;
  }
  for (const file of filtered.slice(0, 40)) {
    const row = document.createElement("div");
    row.className = "result-row";
    row.textContent = `${file.path} (${file.size} bytes)`;
    root.appendChild(row);
  }
}

function renderJobs(jobs) {
  const root = $("#job-list");
  if (!root) {
    return;
  }
  root.innerHTML = "";
  if (!jobs || jobs.length === 0) {
    root.textContent = "No jobs recorded.";
    return;
  }
  for (const job of jobs.slice(0, 20)) {
    const row = document.createElement("div");
    row.className = "result-row job-row";
    const started = new Date((job.started_at || 0) * 1000);
    const elapsed = job.finished_at
      ? formatElapsed(job.finished_at - job.started_at)
      : formatElapsed(Date.now() / 1000 - job.started_at);
    row.innerHTML = `
      <strong>${escapeHTML(job.kind || "job")} / ${escapeHTML(job.status || "unknown")}</strong>
      <span>${escapeHTML(job.campaign || "-")} | ${Number.isNaN(started.getTime()) ? "-" : started.toLocaleString()} | ${elapsed}</span>
    `;
    root.appendChild(row);
  }
}

function metricCard(label, value) {
  return `<div class="metric-card"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`;
}

function shortNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return value ?? "-";
  }
  if (Math.abs(number) >= 1000) {
    return number.toFixed(1);
  }
  return number.toFixed(3);
}

function parseNumber(value) {
  const cleaned = String(value).replaceAll("_", "").replace("%", "").match(/-?\d+(?:\.\d+)?/);
  return cleaned ? Number(cleaned[0]) : null;
}

function parseDurationSeconds(value) {
  const number = parseNumber(value);
  if (!Number.isFinite(number)) {
    return null;
  }
  const text = String(value).toLowerCase();
  if (text.includes("us") || text.includes("µs")) {
    return number / 1000000;
  }
  if (text.includes("ms")) {
    return number / 1000;
  }
  return number;
}

function parseTimingDetail(value) {
  const text = String(value || "").replace(" of measured work", "");
  const seconds = text.match(/[-+]?(?:\d+(?:\.\d*)?|\.\d+)s/);
  const percent = text.match(/\(([^)]+%)\)/);
  return {
    value: parseNumber(seconds?.[0] || text),
    time: seconds?.[0] || text,
    percent: percent?.[1] || "",
  };
}

function meanSecondsPerCall(row) {
  const explicit = parseNumber(row.mean_seconds_per_call);
  if (Number.isFinite(explicit)) {
    return explicit;
  }
  const solverSeconds = parseNumber(row.convex_solver_seconds);
  const calls = parseNumber(row.total_convex_calls);
  if (!Number.isFinite(solverSeconds) || !Number.isFinite(calls) || calls <= 0) {
    return null;
  }
  return solverSeconds / calls;
}

function findTable(report, title) {
  return report.tables.find((table) => table.title === title);
}

function renderBarChart(rows, options = {}) {
  const values = rows
    .map((row) => ({
      label: row.label,
      value: row.numeric ?? parseNumber(row.value),
      time: row.time || row.detail || row.value,
      percent: row.percent || "",
    }))
    .filter((row) => Number.isFinite(row.value) && row.value > 0)
    .slice(0, options.limit || 8);
  if (values.length === 0) {
    return "";
  }
  const max = Math.max(...values.map((row) => row.value));
  return `
    <div class="chart-bars">
      ${values.map((row) => `
        <div class="chart-row">
          <span>${escapeHTML(row.label)}</span>
          <div><i style="width: ${Math.max(2, Math.round((row.value / max) * 100))}%"></i></div>
          <strong>${escapeHTML(row.time)}</strong>
          <em>${escapeHTML(row.percent)}</em>
        </div>
      `).join("")}
    </div>
  `;
}

function renderCounterMetrics(table) {
  if (!table) {
    return "";
  }
  const rows = ["Total convex calls", "Bound solves", "Leaf solves"]
    .map((label) => {
      const row = table.rows.find((item) => item["B&B Counter"] === label);
      if (!row) {
        return "";
      }
      const value = label === "Total convex calls" && !String(row.Value).includes("%")
        ? `${row.Value} (100%)`
        : row.Value;
      return metricCard(label, value);
    })
    .join("");
  return rows ? `<div class="counter-summary">${rows}</div>` : "";
}

function tableValue(table, label, labelKey) {
  const row = table?.rows.find((item) => item[labelKey] === label);
  return row?.Value ?? null;
}

function formatSummaryValue(label, value) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (label === "Mean seconds per call") {
    return formatMicroseconds(parseDurationSeconds(value));
  }
  if (label === "Total work in seconds") {
    return formatSeconds(parseDurationSeconds(value));
  }
  return value;
}

function instanceTotalSeconds(row) {
  const parts = [
    parseNumber(row.decomposition_seconds),
    parseNumber(row.approximation_seconds),
    parseNumber(row.bnb_seconds),
  ].filter(Number.isFinite);
  if (parts.length > 0) {
    return parts.reduce((sum, value) => sum + value, 0);
  }
  return parseNumber(row.total_seconds);
}

function renderHistogram(values, formatter) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return '<p class="inline-note">No values available.</p>';
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const binCount = Math.min(12, Math.max(4, Math.ceil(Math.sqrt(finite.length))));
  const width = max === min ? 1 : (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: min + index * width,
    end: index === binCount - 1 ? max : min + (index + 1) * width,
    count: 0,
  }));
  finite.forEach((value) => {
    const index = max === min ? 0 : Math.min(binCount - 1, Math.floor((value - min) / width));
    bins[index].count += 1;
  });
  const peak = Math.max(...bins.map((bin) => bin.count), 1);
  return `
    <div class="histogram">
      ${bins.map((bin) => `
        <div class="histogram-row">
          <span>${escapeHTML(formatter(bin.start))}-${escapeHTML(formatter(bin.end))}</span>
          <div><i style="width: ${Math.max(3, Math.round((bin.count / peak) * 100))}%"></i></div>
          <strong>${bin.count}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function renderBenchmarkSummaryRows(timing, metrics, counters) {
  const rows = [
    [
      ["Benchmarked instances", tableValue(metrics, "Benchmarked instances", "Metric")],
      ["Fully solved runs", tableValue(metrics, "Fully solved runs", "Metric")],
      ["Capped by calls runs", tableValue(metrics, "Capped by calls runs", "Metric")],
      ["Capped by time runs", tableValue(metrics, "Capped by time runs", "Metric")],
    ],
    [
      ["Worker threads", tableValue(metrics, "Worker threads", "Metric")],
      ["Convex solver name", tableValue(metrics, "Convex solver name", "Metric")],
    ],
    [
      ["Wall-clock total", tableValue(timing, "Wall-clock total", "Timing")],
      ["Total work in seconds", tableValue(timing, "Measured work", "Timing")],
      ["Mean seconds per call", tableValue(timing, "Mean seconds per call", "Timing")],
    ],
    [
      ["Total convex calls", tableValue(counters, "Total convex calls", "B&B Counter")],
      ["Bound solves", tableValue(counters, "Bound solves", "B&B Counter")],
      ["Leaf solves", tableValue(counters, "Leaf solves", "B&B Counter")],
    ],
  ];
  return `
    <div class="benchmark-summary-rows">
      ${rows.map((row) => `
        <div class="benchmark-summary-row" style="--summary-columns: ${row.length}">
          ${row.map(([label, value]) => metricCard(label, formatSummaryValue(label, value))).join("")}
        </div>
      `).join("")}
    </div>
  `;
}

function renderBenchmarkReport(report) {
  const root = $("#benchmark-report");
  if (!report || report.files.length === 0) {
    root.classList.add("is-hidden");
    root.innerHTML = "";
    return;
  }
  const timing = findTable(report, "Timing");
  const metrics = findTable(report, "Metric");
  const counters = findTable(report, "B&B Counter");
  const resultRows = report.files[0].rows || [];
  const timingRows = timing?.rows
    .filter((row) => ["Decomposition", "Approximation", "B&B", "Convex solver"].includes(row.Timing))
    .map((row) => {
      const detail = parseTimingDetail(row.Value);
      return {
        label: row.Timing,
        value: row.Value,
        numeric: detail.value,
        time: detail.time,
        percent: detail.percent,
      };
    }) || [];
  const timeHistogram = renderHistogram(
    resultRows.map(instanceTotalSeconds),
    (value) => formatSeconds(value),
  );
  const callHistogram = renderHistogram(
    resultRows.map((row) => parseNumber(row.calls)),
    (value) => shortNumber(value),
  );
  root.innerHTML = `
    <header class="report-header">
      <div>
        <h3>Latest Markdown Summary</h3>
        <p>${escapeHTML(report.files[0].path)}${report.input_file ? ` | Test case: ${escapeHTML(report.input_file)}` : ""}</p>
      </div>
      <button class="secondary" type="button" data-export-benchmark>Export CSV</button>
    </header>
    ${renderBenchmarkSummaryRows(timing, metrics, counters)}
    <div class="report-grid report-grid-single">
      <section class="report-panel">
        <h4>Timing Share</h4>
        ${renderBarChart(timingRows)}
      </section>
      <section class="report-panel">
        <div class="histogram-head">
          <h4>Instance Histograms</h4>
          <div>
            <button class="secondary" type="button" data-toggle-histogram="time">Time</button>
            <button class="secondary" type="button" data-toggle-histogram="calls">Calls</button>
          </div>
        </div>
        <div class="histogram-panel is-hidden" data-histogram-panel="time">
          <h5>Instance Time</h5>
          ${timeHistogram}
        </div>
        <div class="histogram-panel is-hidden" data-histogram-panel="calls">
          <h5>Convex Calls</h5>
          ${callHistogram}
        </div>
      </section>
    </div>
  `;
  root.querySelectorAll("[data-toggle-histogram]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.toggleHistogram;
      const panel = root.querySelector(`[data-histogram-panel="${target}"]`);
      const hidden = panel.classList.toggle("is-hidden");
      button.classList.toggle("is-active", !hidden);
      button.setAttribute("aria-pressed", !hidden ? "true" : "false");
    });
  });
  root.querySelector("[data-export-benchmark]")?.addEventListener("click", () => {
    downloadCSV(`${report.files[0].path.split("/").pop() || "benchmark"}.csv`, resultRows);
  });
  root.classList.remove("is-hidden");
}

function solverLabel(name) {
  const labels = {
    linear_search_lazy: "Linear Intersections",
    linear_search_disjoint: "Linear Disjoint",
    binary_search_lazy: "Binary Intersections",
    binary_search_disjoint: "Binary Disjoint",
    binary_search_eager: "Binary Eager",
    tan_jiang: "Tan Jiang",
    gurobi: "Gurobi",
    linear: "Linear Intersections",
    linear_disjoint: "Linear Disjoint",
    binary: "Binary Intersections",
    binary_disjoint: "Binary Disjoint",
    tan: "Tan Jiang",
  };
  return labels[name] || name;
}

function renderComparisonReport(data) {
  const root = $("#comparison-report");
  const rows = data?.rows || [];
  if (rows.length === 0) {
    root.classList.add("is-hidden");
    root.innerHTML = "";
    return;
  }

  const completed = rows.filter((row) => row.status === "completed").length;
  const fastest = rows
    .map((row) => ({ solver: row.solver, seconds: parseNumber(row.wall_clock_seconds) }))
    .filter((row) => Number.isFinite(row.seconds))
    .sort((left, right) => left.seconds - right.seconds)[0];
  const calls = rows
    .map((row) => ({ solver: row.solver, calls: parseNumber(row.total_convex_calls) }))
    .filter((row) => Number.isFinite(row.calls));
  const maxCalls = calls.length ? Math.max(...calls.map((row) => row.calls)) : 0;
  const timeRows = rows.map((row) => ({
    label: solverLabel(row.solver),
    value: row.wall_clock_seconds,
  }));
  const meanRows = rows.map((row) => {
    const mean = meanSecondsPerCall(row);
    return {
      label: solverLabel(row.solver),
      value: row.mean_seconds_per_call,
      numeric: mean,
      time: mean === null ? "-" : formatMicroseconds(mean),
    };
  });
  const bestMean = rows
    .map((row) => ({ solver: row.solver, seconds: meanSecondsPerCall(row) }))
    .filter((row) => Number.isFinite(row.seconds))
    .sort((left, right) => left.seconds - right.seconds)[0];

  root.innerHTML = `
    <header class="report-header">
      <div>
        <h3>Latest Solver Comparison</h3>
        <p>${rows.length} solver${rows.length === 1 ? "" : "s"} in the latest comparison run${data?.input_file ? ` | Test case: ${escapeHTML(data.input_file)}` : ""}</p>
      </div>
      <button class="secondary" type="button" data-export-comparison>Export CSV</button>
    </header>
    <div class="summary-grid">
      ${metricCard("Completed", `${completed}/${rows.length}`)}
      ${metricCard("Fastest", fastest ? solverLabel(fastest.solver) : "-")}
      ${metricCard("Best wall clock", fastest ? `${shortNumber(fastest.seconds)} s` : "-")}
      ${metricCard("Best avg solve", bestMean ? formatMicroseconds(bestMean.seconds) : "-")}
      ${metricCard("Total calls max", maxCalls || "-")}
    </div>
    <div class="report-grid">
      <section class="report-panel">
        <h4>Wall Clock</h4>
        ${renderBarChart(timeRows)}
      </section>
      <section class="report-panel">
        <h4>Average Convex Solve</h4>
        ${renderBarChart(meanRows)}
      </section>
      <section class="report-panel comparison-table-panel">
        <h4>Solver Details</h4>
        <div class="comparison-table-wrap">
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Solver</th>
                <th>Status</th>
                <th>Wall</th>
                <th>Work</th>
                <th>Avg solve</th>
                <th>Calls</th>
                <th>Solved</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map((row) => `
                <tr>
                  <td>${escapeHTML(solverLabel(row.solver))}</td>
                  <td>${escapeHTML(row.status)}</td>
                  <td>${shortNumber(row.wall_clock_seconds)} s</td>
                  <td>${shortNumber(row.convex_solver_seconds)} s</td>
                  <td>${meanSecondsPerCall(row) === null ? "-" : formatMicroseconds(meanSecondsPerCall(row))}</td>
                  <td>${escapeHTML(row.total_convex_calls || "-")}</td>
                  <td>${escapeHTML(row.fully_solved_runs || "-")}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
  root.querySelector("[data-export-comparison]")?.addEventListener("click", () => {
    downloadCSV(`${data?.path?.split("/").at(-2) || "comparison"}.csv`, rows);
  });
  root.classList.remove("is-hidden");
}

async function refreshComparisonReport(campaignName) {
  if (!campaignName) {
    renderComparisonReport(null);
    return;
  }
  const data = await requestJSON(`/api/campaigns/${encodeURIComponent(campaignName)}/comparisons`);
  renderComparisonReport(data);
}

function renderRunProgressCard(campaign, running = false, liveProgress = null) {
  const root = $("#run-progress");
  if (!campaign) {
    root.innerHTML = "";
    root.classList.add("is-hidden");
    return;
  }
  const progress = liveProgress || runProgress(campaign);
  const percent = Math.round(progress.ratio * 100);
  const visiblePercent = running && percent === 0 ? 8 : percent;
  root.innerHTML = `
    <div class="run-progress-header">
      <strong>${running ? "Running benchmark" : "Benchmark progress"}</strong>
      <span>${progress.label}</span>
    </div>
    <div class="run-progress-meta">
      <span>Wall clock</span>
      <strong>${formatElapsed(progress.elapsed_seconds || 0)}</strong>
    </div>
    <div class="run-progress-line">
      <div class="run-progress-track ${running ? "is-running" : ""}">
        <div class="run-progress-fill" style="width: ${visiblePercent}%"></div>
      </div>
      <strong class="run-progress-percent">${percent}%</strong>
    </div>
  `;
  root.classList.remove("is-hidden");
}

function renderRunSummary() {
  const root = $("#run-summary");
  const selected = $("#run-campaign").value;
  const campaign = state.campaigns.find((item) => item.name === selected);
  if (!campaign) {
    root.innerHTML = "";
    return;
  }
  const progress = runProgress(campaign);
  const failedFiles = Object.entries(progress.counts)
    .filter(([status]) => status !== "completed" && status !== "pending")
    .reduce((total, [, count]) => total + count, 0);
  const pendingInstances = Math.max(0, progress.total - progress.completed);
  root.innerHTML = `
    ${metricCard("Instances", `${progress.completed}/${progress.total || "-"}`)}
    ${metricCard("Completed", progress.completed)}
    ${metricCard("Pending", pendingInstances || 0)}
    ${metricCard("Failed input files", failedFiles)}
    <div class="summary-progress">
      <span>${progress.label}</span>
      <div class="bar"><div class="bar-fill" style="width: ${Math.round(progress.ratio * 100)}%"></div></div>
    </div>
  `;
  renderRunProgressCard(campaign);
  renderSolvedPreview(campaign);
}

function openCampaignModal(campaign) {
  const modal = $("#campaign-modal");
  const body = $("#modal-body");
  const generation = campaign.generation || {};
  const progress = runProgress(campaign);
  $("#modal-title").textContent = campaign.name;
  body.innerHTML = `
    <div class="modal-summary">
      ${metricCard("Type", campaign.type)}
      ${metricCard("Input files", `${campaign.inputs.existing}/${campaign.inputs.total}`)}
      ${metricCard("Instances", generation.instances ?? generation.instances_per_file ?? "-")}
      ${metricCard("Polygon Count", generation.polygons ?? generation.polygon_counts ?? "-")}
      ${metricCard("Vertices", describeVertices(generation))}
      ${metricCard("Progress", progress.label)}
    </div>
    <div class="preview-layout modal-previews"></div>
    <section class="result-preview-section modal-results is-hidden" data-benchmarked-section="${campaign.name}"></section>
    <h3>Generation Metadata</h3>
    <pre class="output modal-json">${JSON.stringify(generation, null, 2)}</pre>
    <div class="modal-actions">
      <button class="danger" type="button" data-delete-campaign="${campaign.name}">Delete Campaign</button>
    </div>
  `;
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
}

function openInstanceModal(campaign, index) {
  const modal = $("#campaign-modal");
  const body = $("#modal-body");
  $("#modal-title").textContent = `${campaign.name} / instance ${instanceLabel(index)}`;
  body.innerHTML = zoomableImage(
    instancePreviewUrl(campaign, index),
    `Instance ${instanceLabel(index)} detail`,
  );
  setupZoomableDetail(body);
  modal.classList.remove("is-hidden");
}

function openBenchmarkedInstanceModal(campaign, item) {
  const modal = $("#campaign-modal");
  const body = $("#modal-body");
  $("#modal-title").textContent = `${campaign.name} / case ${instanceLabel(item.case_index)}`;
  const previewUrl = item.solution_available
    ? solutionPreviewUrl(campaign, item)
    : item.preview
      ? instancePreviewUrl(campaign, item.case_index)
      : null;
  body.innerHTML = `
    <div class="modal-summary">
      ${metricCard("Status", item.status)}
      ${metricCard("Final length", shortNumber(item.final_length))}
      ${metricCard("Solve time", formatSeconds(parseNumber(item.total_seconds)))}
      ${metricCard("Calls", item.calls ?? "-")}
      ${metricCard("Avg convex solve", formatMicroseconds(parseNumber(item.seconds_per_call)))}
      ${metricCard("Decomposed pieces", item.decomposed_pieces ?? "-")}
      ${metricCard("Visited nodes", item.visited_nodes ?? "-")}
      ${metricCard("Pruned nodes", item.pruned_nodes ?? "-")}
    </div>
    ${previewUrl
      ? zoomableImage(previewUrl, `Solved instance ${instanceLabel(item.case_index)} detail`, "benchmarked-detail", { inspectable: item.solution_available })
      : '<div class="missing-preview detail-missing">No preview available.</div>'}
    ${item.solution_available ? "" : '<p class="inline-note">This is an older run. Rerun the benchmark to generate the path/decomposition overlay SVG for this case.</p>'}
  `;
  setupZoomableDetail(body);
  modal.classList.remove("is-hidden");
}

function openPreviewModal(campaign, kind, title) {
  const modal = $("#campaign-modal");
  const body = $("#modal-body");
  $("#modal-title").textContent = `${campaign.name} / ${title}`;
  body.innerHTML = zoomableImage(detailPreviewUrl(campaign, kind), `${title} detail`);
  setupZoomableDetail(body);
  modal.classList.remove("is-hidden");
}

async function refresh() {
  const [campaignData, resultData, jobData] = await Promise.all([
    requestJSON("/api/campaigns"),
    requestJSON("/api/results"),
    requestJSON("/api/jobs"),
  ]);
  state.campaigns = campaignData.campaigns;
  state.resultFiles = resultData.files;
  state.recentJobs = jobData.jobs;
  renderCampaignOptions();
  renderCampaigns();
  renderResults(state.resultFiles);
  renderJobs(state.recentJobs);
  updateCampaignNameIndicator();
}

function updateCreateMode() {
  const mode = document.querySelector('[name="campaign_type"]').value;
  const synthetic = mode === "synthetic";
  document.querySelectorAll(".synthetic-field").forEach((field) => {
    field.classList.toggle("is-hidden", !synthetic);
    field.querySelectorAll("input, select").forEach((input) => {
      input.disabled = !synthetic;
    });
  });
  document.querySelectorAll(".osm-field").forEach((field) => {
    field.classList.toggle("is-hidden", synthetic);
    field.querySelectorAll("input, select").forEach((input) => {
      input.disabled = synthetic;
    });
  });
  if (!synthetic && !state.osmScanStarted) {
    scanOsmFiles();
  }
}

async function refreshBenchmarkReport(campaignName) {
  if (!campaignName) {
    renderBenchmarkReport(null);
    renderSolvedPreview(null);
    return;
  }
  const report = await requestJSON(`/api/campaigns/${encodeURIComponent(campaignName)}/summaries`);
  renderBenchmarkReport(report);
  await refreshBenchmarkedInstances(campaignName);
}

async function refreshBenchmarkedInstances(campaignName) {
  if (!campaignName) {
    return;
  }
  const data = await requestJSON(`/api/campaigns/${encodeURIComponent(campaignName)}/benchmarked-instances`);
  state.benchmarkedInstances.set(campaignName, data.instances);
  const campaign = state.campaigns.find((item) => item.name === campaignName);
  renderSolvedPreview(campaign);
  document.querySelectorAll("[data-benchmarked-section]").forEach((root) => {
    if (root.dataset.benchmarkedSection === campaignName) {
      renderBenchmarkedInstanceSection(root, campaign, data.instances);
    }
  });
}

async function createCampaign(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const values = formData(form);
  const synthetic = values.campaign_type === "synthetic";
  const payload = synthetic ? {
    name: values.name,
    vertices: values.vertices,
    polygons: Number(values.polygons),
    instances: Number(values.instances),
    shape: values.shape,
    seed: Number(values.seed),
    no_preview: boolField(form, "no_preview"),
    overwrite: boolField(form, "overwrite"),
  } : {
    name: values.name,
    pbf_path: values.pbf_path,
    polygon_counts: Number(values.polygon_counts),
    sample_size: values.sample_size ? Number(values.sample_size) : null,
    instances: Number(values.instances),
    seed: Number(values.seed),
    simplify_tolerance: Number(values.simplify_tolerance),
    normalization: values.normalization,
    scale: Number(values.scale),
    sampling: values.sampling,
    local_pool_size: Number(values.local_pool_size),
    layout: values.layout,
    grid_polygon_size: Number(values.grid_polygon_size),
    grid_cell_size: Number(values.grid_cell_size),
    grid_columns: values.grid_columns ? Number(values.grid_columns) : null,
    grid_placement: values.grid_placement,
    convex_replacement_fraction: Number(values.convex_replacement_fraction),
    convex_replacement_vertices: Number(values.convex_replacement_vertices),
    convex_replacement_position: values.convex_replacement_position,
    order: values.order,
    endpoint_mode: values.endpoint_mode,
    no_preview: boolField(form, "no_preview"),
    overwrite: boolField(form, "overwrite"),
  };
  const output = $("#create-output");
  const preview = $("#create-preview");
  setOutput(output, "Creating campaign...");
  preview.innerHTML = "";
  preview.classList.add("is-hidden");
  try {
    const data = await requestJSON(synthetic ? "/api/campaigns/synthetic" : "/api/campaigns/osm", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setOutput(output, data.output);
    await refresh();
    const campaign = state.campaigns.find((item) => item.name === data.campaign.name) || data.campaign;
    renderPreviewPanels(preview, campaign);
  } catch (error) {
    setOutput(output, error.message);
    setStopButton("#stop-run-button", null);
  }
}

async function importCanonicalSuite() {
  const output = $("#inspect-output");
  try {
    const data = await requestJSON("/api/campaigns/canonical", {
      method: "POST",
      body: JSON.stringify({ name: "canonical-v1", overwrite: campaignExists("canonical-v1") }),
    });
    await refresh();
    switchPanel("inspect-panel");
    setOutput(output, `Imported canonical suite as ${data.campaign.name}.`);
  } catch (error) {
    switchPanel("inspect-panel");
    setOutput(output, error.message);
  }
}

async function importGermanInstances() {
  const output = $("#inspect-output");
  try {
    const data = await requestJSON("/api/campaigns/german", {
      method: "POST",
      body: JSON.stringify({ name: "german-instances", overwrite: campaignExists("german-instances") }),
    });
    await refresh();
    switchPanel("inspect-panel");
    setOutput(output, `Imported German instances as ${data.campaign.name}.`);
  } catch (error) {
    switchPanel("inspect-panel");
    setOutput(output, error.message);
  }
}

async function deleteCampaign(event) {
  const name = event.currentTarget.dataset.deleteCampaign;
  if (!name || !(await askConfirmation(`Delete campaign "${name}"? This removes its inputs, previews, results, and metadata.`, "Delete"))) {
    return;
  }
  try {
    await requestJSON(`/api/campaigns/${encodeURIComponent(name)}`, { method: "DELETE" });
    closeCampaignModal();
    state.benchmarkedInstances.delete(name);
    if (state.selectedCampaign === name) {
      state.selectedCampaign = "";
    }
    if (state.selectedComparisonCampaign === name) {
      state.selectedComparisonCampaign = "";
    }
    await refresh();
  } catch (error) {
    setOutput($("#inspect-output"), error.message);
  }
}

function setStopButton(selector, jobId) {
  const button = $(selector);
  if (!button) {
    return;
  }
  button.dataset.job = jobId || "";
  button.disabled = !jobId;
  button.classList.toggle("is-hidden", !jobId);
  button.textContent = "Stop";
}

async function cancelJob(selector, outputSelector) {
  const button = $(selector);
  const jobId = button?.dataset.job;
  if (!jobId || button.disabled) {
    return;
  }
  button.disabled = true;
  button.textContent = "Stopping...";
  try {
    await requestJSON(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  } catch (error) {
    setOutput($(outputSelector), error.message);
    button.disabled = false;
    button.textContent = "Stop";
  }
}

function renderComparisonProgress(progress, options) {
  const active = options.active;
  const instancePercent = options.instanceTotal
    ? Math.round((options.instanceCompleted / options.instanceTotal) * 100)
    : active ? 0 : 100;
  const visibleInstancePercent = active && instancePercent === 0 ? 8 : instancePercent;
  const solverPercent = options.solverTotal
    ? Math.round((options.solverCompleted / options.solverTotal) * 100)
    : active ? 0 : 100;
  const visibleSolverPercent = active && solverPercent === 0 ? 8 : solverPercent;
  progress.innerHTML = `
    <div class="comparison-progress-head">
      <strong>Status: ${options.status}</strong>
      <span>Test case: ${options.testCase || "-"}</span>
      <span>Wall clock: <strong>${formatElapsed(options.elapsedSeconds || 0)}</strong></span>
      <span>Current solver: ${options.currentSolver || "-"}</span>
    </div>
    <div class="comparison-progress-lines">
      <div class="comparison-progress-line">
        <span>Instances Solved</span>
        <div class="run-progress-track ${active ? "is-running" : ""}">
          <div class="run-progress-fill" style="width: ${visibleInstancePercent}%"></div>
        </div>
        <strong>${instancePercent}%</strong>
      </div>
      <div class="comparison-progress-line">
        <span>Solvers Tested</span>
        <div class="run-progress-track ${active ? "is-running" : ""}">
          <div class="run-progress-fill" style="width: ${visibleSolverPercent}%"></div>
        </div>
        <strong>${options.solverTotal ? `${options.solverCompleted}/${options.solverTotal}` : "-"}</strong>
      </div>
    </div>
  `;
  progress.classList.remove("is-hidden");
}

async function pollJob(jobId) {
  const output = $("#run-output");
  setStopButton("#stop-run-button", jobId);
  let lastRefresh = 0;
  while (true) {
    const job = await requestJSON(`/api/jobs/${jobId}`);
    const command = `+ ${job.command.join(" ")}\n\n`;
    setOutput(output, command + (job.output || ""));
    if (Date.now() - lastRefresh > 1000) {
      await refresh();
      lastRefresh = Date.now();
    }
    const campaign = state.campaigns.find((item) => item.name === state.selectedCampaign);
    const liveProgress = job.progress_total
      ? {
        completed: job.progress_completed || 0,
        total: job.progress_total,
        ratio: (job.progress_completed || 0) / job.progress_total,
        label: `${job.progress_completed || 0}/${job.progress_total} instances`,
        elapsed_seconds: job.elapsed_seconds,
        counts: campaign?.run_index?.counts || {},
      }
      : campaign ? {
        ...runProgress(campaign),
        elapsed_seconds: job.elapsed_seconds,
      } : null;
    const jobActive = job.status === "running" || job.status === "stopping";
    renderRunProgressCard(campaign, jobActive, liveProgress);
    if (!jobActive) {
      let logText = "";
      if (job.status === "failed") {
        const selected = state.selectedCampaign;
        const logs = await requestJSON(`/api/campaigns/${encodeURIComponent(selected)}/logs`);
        logText = logs.logs.map((log) => `\n\n--- ${log.path} ---\n${log.tail}`).join("");
      }
      await refresh();
      await refreshBenchmarkReport(state.selectedCampaign);
      const finalCampaign = state.campaigns.find((item) => item.name === state.selectedCampaign);
      renderRunProgressCard(finalCampaign, false);
      renderSolvedPreview(finalCampaign);
      setOutput(output, command + (job.output || "") + logText + `\nstatus: ${job.status}`);
      setStopButton("#stop-run-button", null);
      state.currentRunJob = null;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function pollComparisonJob(jobId) {
  const output = $("#compare-output");
  const progress = $("#compare-progress");
  setStopButton("#stop-compare-button", jobId);
  let lastReportRefresh = 0;
  while (true) {
    const job = await requestJSON(`/api/jobs/${jobId}`);
    const command = `+ ${job.command.join(" ")}\n\n`;
    setOutput(output, command + (job.output || ""));
    const jobActive = job.status === "running" || job.status === "stopping";
    const solverTotal = job.solver_progress_total || 0;
    const solverCompleted = job.solver_progress_completed || 0;
    renderComparisonProgress(progress, {
      active: jobActive,
      status: jobActive ? (job.status === "stopping" ? "Stopping comparison" : "Running comparison") : "Comparison finished",
      testCase: state.selectedComparisonCampaign || job.campaign || "",
      elapsedSeconds: job.elapsed_seconds || 0,
      currentSolver: job.current_solver ? solverLabel(job.current_solver) : "-",
      instanceCompleted: job.progress_completed || 0,
      instanceTotal: job.progress_total || 0,
      solverCompleted,
      solverTotal,
    });
    if (Date.now() - lastReportRefresh > 1000 && jobActive) {
      refreshComparisonReport(state.selectedComparisonCampaign || job.campaign);
      lastReportRefresh = Date.now();
    }
    if (!jobActive) {
      await refreshComparisonReport(state.selectedComparisonCampaign || job.campaign);
      setOutput(output, command + (job.output || "") + `\nstatus: ${job.status}`);
      setStopButton("#stop-compare-button", null);
      state.currentComparisonJob = null;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function runCampaign(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const values = formData(form);
  const payload = {
    name: values.name,
    threads: values.threads ? Number(values.threads) : null,
    solver: values.solver || null,
    max_calls: values.max_calls,
    max_instances: values.max_instances ? Number(values.max_instances) : null,
    max_seconds: values.max_seconds || null,
    timeout: values.timeout ? Number(values.timeout) : null,
    dry_run: boolField(form, "dry_run"),
    force: boolField(form, "force"),
    no_build: boolField(form, "no_build"),
  };
  const output = $("#run-output");
  renderBenchmarkReport(null);
  const activeCampaign = state.campaigns.find((item) => item.name === state.selectedCampaign);
  renderRunProgressCard(activeCampaign, true, activeCampaign ? {
    ...runProgress(activeCampaign),
    elapsed_seconds: 0,
  } : null);
  setOutput(output, "Starting run...");
  renderRunSummary();
  try {
    const data = await requestJSON("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJob = data.job;
    state.currentRunJob = data.job;
    await pollJob(data.job);
  } catch (error) {
    setOutput(output, error.message);
  }
}

async function runComparison(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const values = formData(form);
  const solvers = [...form.querySelectorAll('input[name="solvers"]:checked')]
    .map((input) => input.value);
  const payload = {
    name: values.name,
    threads: values.threads ? Number(values.threads) : null,
    solvers,
    max_calls: values.max_calls,
    max_instances: values.max_instances ? Number(values.max_instances) : null,
    max_seconds: values.max_seconds || null,
    no_build: boolField(form, "no_build"),
  };
  const output = $("#compare-output");
  const progress = $("#compare-progress");
  renderComparisonReport(null);
  renderComparisonProgress(progress, {
    active: true,
    status: "Starting comparison",
    testCase: state.selectedComparisonCampaign || values.name || "",
    elapsedSeconds: 0,
    currentSolver: solvers.map(solverLabel).join(", "),
    instanceCompleted: 0,
    instanceTotal: 0,
    solverCompleted: 0,
    solverTotal: solvers.length,
  });
  setOutput(output, "Starting comparison...");
  try {
    const data = await requestJSON("/api/comparisons", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJob = data.job;
    state.currentComparisonJob = data.job;
    await pollComparisonJob(data.job);
  } catch (error) {
    setOutput(output, error.message);
    setStopButton("#stop-compare-button", null);
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchPanel(tab.dataset.panel));
});

document.querySelectorAll("[data-close-modal]").forEach((element) => {
  element.addEventListener("click", closeCampaignModal);
});

document.querySelectorAll("[data-confirm-cancel]").forEach((element) => {
  element.addEventListener("click", () => closeConfirmation(false));
});

document.querySelector("[data-confirm-ok]").addEventListener("click", () => closeConfirmation(true));

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeCampaignModal();
    closeConfirmation(false);
  }
});

setupSegmentedControls();
setupToggleButtons();
setupBoundedSliders();
setupMaxInstancesControl();
setupCompareMaxInstancesControl();
$("#refresh-button").addEventListener("click", refresh);
$("#import-canonical-button").addEventListener("click", importCanonicalSuite);
$("#import-german-button").addEventListener("click", importGermanInstances);
$("#scan-osm-files").addEventListener("click", scanOsmFiles);
$("#create-form").addEventListener("submit", createCampaign);
$("#run-form").addEventListener("submit", runCampaign);
$("#compare-form").addEventListener("submit", runComparison);
$("#stop-run-button").addEventListener("click", () => cancelJob("#stop-run-button", "#run-output"));
$("#stop-compare-button").addEventListener("click", () => cancelJob("#stop-compare-button", "#compare-output"));
$("#create-name").addEventListener("input", updateCampaignNameIndicator);
setupFilterInput("#campaign-filter", "campaignFilter", renderCampaigns);
setupFilterInput("#result-filter", "resultFilter", () => renderResults(state.resultFiles));
updateCreateMode();

requestJSON("/api/system")
  .then((system) => {
    state.cpuCount = system.cpu_count || 1;
    setupThreadsControl();
    setupThreadsControl("#compare-threads-slider", "#compare-threads-input", "#compare-threads-max-label");
    return refresh();
  })
  .catch((error) => {
    setupThreadsControl();
    setupThreadsControl("#compare-threads-slider", "#compare-threads-input", "#compare-threads-max-label");
    setOutput($("#create-output"), error.message);
  });
