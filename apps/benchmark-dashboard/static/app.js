const state = {
  campaigns: [],
  currentJob: null,
  cpuCount: 1,
  selectedCampaign: "",
  selectedComparisonCampaign: "",
  osmFiles: [],
  osmScanStarted: false,
  benchmarkedInstances: new Map(),
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
  if (state.osmFiles.length === 0) {
    grid.innerHTML = '<div class="empty-choice">No .osm.pbf files found.</div>';
    selectOsmFile("");
    return;
  }
  for (const file of state.osmFiles) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "choice-card osm-file-card";
    button.dataset.value = file.path;
    button.setAttribute("role", "option");
    button.innerHTML = `
      <strong>${file.name}</strong>
      <span>${formatBytes(file.size)}</span>
      <small>${file.path}</small>
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
      <strong>${campaign.name}</strong>
      <span>${campaign.type}</span>
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
  selectRunCampaign(state.campaigns.some((campaign) => campaign.name === selected) ? selected : fallback);
  selectComparisonCampaign(state.campaigns.some((campaign) => campaign.name === state.selectedComparisonCampaign) ? state.selectedComparisonCampaign : fallback);
}

function selectRunCampaign(name) {
  state.selectedCampaign = name;
  $("#run-campaign").value = name;
  renderCampaignChoiceGrid($("#run-campaign-grid"), name, selectRunCampaign);
  resetMaxInstancesControl();
  renderRunSummary();
  if (name) {
    refreshBenchmarkReport(name);
  }
}

function selectComparisonCampaign(name) {
  state.selectedComparisonCampaign = name;
  $("#compare-campaign").value = name;
  renderCampaignChoiceGrid($("#compare-campaign-grid"), name, selectComparisonCampaign);
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
    </header>
    <div class="benchmarked-grid"></div>
  `;
  const grid = root.querySelector(".benchmarked-grid");
  for (const item of instances) {
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
        <small>${item.calls ?? "-"} calls</small>
        <small>${item.solution_available ? "path + pieces" : `${item.decomposed_pieces ?? "-"} pieces`}</small>
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
  for (const campaign of state.campaigns) {
    const generation = campaign.generation || {};
    const progress = runProgress(campaign);
    const card = document.createElement("article");
    card.className = "campaign-card";
    card.tabIndex = 0;
    card.innerHTML = `
      <button class="campaign-delete" type="button" data-delete-campaign="${campaign.name}" aria-label="Delete ${campaign.name}">x</button>
      <h3>${campaign.name}</h3>
      <div class="meta">
        <div><span>Type</span><br>${campaign.type}</div>
        <div><span>Input files</span><br>${campaign.inputs.existing}/${campaign.inputs.total}</div>
        <div><span>Instances</span><br>${generation.instances ?? generation.instances_per_file ?? "-"}</div>
        <div><span>Polygon Count</span><br>${generation.polygons ?? generation.polygon_counts ?? "-"}</div>
        <div><span>Vertices</span><br>${describeVertices(generation)}</div>
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
  if (files.length === 0) {
    root.textContent = "No result files found.";
    return;
  }
  for (const file of files.slice().reverse().slice(0, 40)) {
    const row = document.createElement("div");
    row.className = "result-row";
    row.textContent = `${file.path} (${file.size} bytes)`;
    root.appendChild(row);
  }
}

function metricCard(label, value) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`;
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
          <span>${row.label}</span>
          <div><i style="width: ${Math.max(2, Math.round((row.value / max) * 100))}%"></i></div>
          <strong>${row.time}</strong>
          <em>${row.percent}</em>
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
  const cards = [];
  for (const label of [
    "Benchmarked instances",
    "Worker threads",
    "Convex solver name",
    "Wall-clock total",
    "Fully solved runs",
    "Capped by calls runs",
    "Capped by time runs",
  ]) {
    const table = label === "Wall-clock total" ? timing : metrics;
    const key = label === "Wall-clock total" ? "Timing" : "Metric";
    const row = table?.rows.find((item) => item[key] === label);
    if (row) {
      cards.push(metricCard(label, row.Value));
    }
  }
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
  root.innerHTML = `
    <header class="report-header">
      <div>
        <h3>Latest Markdown Summary</h3>
        <p>${report.files[0].path}</p>
      </div>
    </header>
    <div class="summary-grid">${cards.join("")}</div>
    <div class="report-grid report-grid-single">
      <section class="report-panel">
        <h4>Timing Share</h4>
        ${renderBarChart(timingRows)}
        ${renderCounterMetrics(counters)}
      </section>
    </div>
  `;
  root.classList.remove("is-hidden");
}

function solverLabel(name) {
  const labels = {
    linear_search_lazy: "Linear",
    binary_search_lazy: "Binary",
    tan_jiang: "Tan Jiang",
    gurobi: "Gurobi",
    linear: "Linear",
    binary: "Binary",
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

  root.innerHTML = `
    <header class="report-header">
      <div>
        <h3>Latest Solver Comparison</h3>
        <p>${rows.length} solver${rows.length === 1 ? "" : "s"} in the latest comparison run</p>
      </div>
    </header>
    <div class="summary-grid">
      ${metricCard("Completed", `${completed}/${rows.length}`)}
      ${metricCard("Fastest", fastest ? solverLabel(fastest.solver) : "-")}
      ${metricCard("Best wall clock", fastest ? `${shortNumber(fastest.seconds)} s` : "-")}
      ${metricCard("Total calls max", maxCalls || "-")}
    </div>
    <div class="report-grid">
      <section class="report-panel">
        <h4>Wall Clock</h4>
        ${renderBarChart(timeRows)}
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
                <th>Calls</th>
                <th>Solved</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map((row) => `
                <tr>
                  <td>${solverLabel(row.solver)}</td>
                  <td>${row.status}</td>
                  <td>${shortNumber(row.wall_clock_seconds)} s</td>
                  <td>${shortNumber(row.convex_solver_seconds)} s</td>
                  <td>${row.total_convex_calls || "-"}</td>
                  <td>${row.fully_solved_runs || "-"}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
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
  body.innerHTML = `
    <figure class="instance-detail">
      <img src="${instancePreviewUrl(campaign, index)}" alt="Instance ${instanceLabel(index)} detail">
    </figure>
  `;
  modal.classList.remove("is-hidden");
}

function openBenchmarkedInstanceModal(campaign, item) {
  const modal = $("#campaign-modal");
  const body = $("#modal-body");
  $("#modal-title").textContent = `${campaign.name} / case ${instanceLabel(item.case_index)}`;
  const preview = item.solution_available
    ? `<img src="${solutionPreviewUrl(campaign, item)}" alt="Solved instance ${instanceLabel(item.case_index)} with path and decomposition">`
    : item.preview
      ? `<img src="${instancePreviewUrl(campaign, item.case_index)}" alt="Benchmarked instance ${instanceLabel(item.case_index)}">`
    : '<div class="missing-preview">No preview available.</div>';
  body.innerHTML = `
    <div class="modal-summary">
      ${metricCard("Status", item.status)}
      ${metricCard("Final length", shortNumber(item.final_length))}
      ${metricCard("Calls", item.calls ?? "-")}
      ${metricCard("Decomposed pieces", item.decomposed_pieces ?? "-")}
      ${metricCard("Visited nodes", item.visited_nodes ?? "-")}
      ${metricCard("Pruned nodes", item.pruned_nodes ?? "-")}
    </div>
    <figure class="instance-detail benchmarked-detail">
      ${preview}
    </figure>
    ${item.solution_available ? "" : '<p class="inline-note">This is an older run. Rerun the benchmark to generate the path/decomposition overlay SVG for this case.</p>'}
  `;
  modal.classList.remove("is-hidden");
}

function openPreviewModal(campaign, kind, title) {
  const modal = $("#campaign-modal");
  const body = $("#modal-body");
  $("#modal-title").textContent = `${campaign.name} / ${title}`;
  body.innerHTML = `
    <figure class="instance-detail">
      <img src="${detailPreviewUrl(campaign, kind)}" alt="${title} detail">
    </figure>
  `;
  modal.classList.remove("is-hidden");
}

async function refresh() {
  const [campaignData, resultData] = await Promise.all([
    requestJSON("/api/campaigns"),
    requestJSON("/api/results"),
  ]);
  state.campaigns = campaignData.campaigns;
  renderCampaignOptions();
  renderCampaigns();
  renderResults(resultData.files);
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

async function pollJob(jobId) {
  const output = $("#run-output");
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
    renderRunProgressCard(campaign, job.status === "running", liveProgress);
    if (job.status !== "running") {
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
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function pollComparisonJob(jobId) {
  const output = $("#compare-output");
  const progress = $("#compare-progress");
  let lastReportRefresh = 0;
  while (true) {
    const job = await requestJSON(`/api/jobs/${jobId}`);
    const command = `+ ${job.command.join(" ")}\n\n`;
    setOutput(output, command + (job.output || ""));
    const liveProgress = job.progress_total
      ? `${job.progress_completed || 0}/${job.progress_total} instances`
      : state.selectedComparisonCampaign || job.campaign || "";
    const progressPercent = job.progress_total
      ? Math.round(((job.progress_completed || 0) / job.progress_total) * 100)
      : job.status === "running" ? 0 : 100;
    const visiblePercent = job.status === "running" && progressPercent === 0 ? 8 : progressPercent;
    progress.innerHTML = `
      <div class="run-progress-header">
        <strong>${job.status === "running" ? "Running comparison" : "Comparison finished"}</strong>
        <span>${liveProgress}</span>
      </div>
      <div class="run-progress-meta">
        <span>Wall clock</span>
        <strong>${formatElapsed(job.elapsed_seconds || 0)}</strong>
      </div>
      <div class="run-progress-line">
        <div class="run-progress-track ${job.status === "running" ? "is-running" : ""}">
          <div class="run-progress-fill" style="width: ${visiblePercent}%"></div>
        </div>
        <strong class="run-progress-percent">${progressPercent}%</strong>
      </div>
    `;
    progress.classList.remove("is-hidden");
    if (Date.now() - lastReportRefresh > 1000 && job.status === "running") {
      refreshComparisonReport(state.selectedComparisonCampaign || job.campaign);
      lastReportRefresh = Date.now();
    }
    if (job.status !== "running") {
      await refreshComparisonReport(state.selectedComparisonCampaign || job.campaign);
      setOutput(output, command + (job.output || "") + `\nstatus: ${job.status}`);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
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
    max_seconds: values.max_seconds || null,
    no_build: boolField(form, "no_build"),
  };
  const output = $("#compare-output");
  const progress = $("#compare-progress");
  renderComparisonReport(null);
  progress.innerHTML = `
    <div class="run-progress-header">
      <strong>Starting comparison</strong>
      <span>${solvers.map(solverLabel).join(", ")}</span>
    </div>
    <div class="run-progress-meta">
      <span>Wall clock</span>
      <strong>${formatElapsed(0)}</strong>
    </div>
    <div class="run-progress-line">
      <div class="run-progress-track is-running">
        <div class="run-progress-fill" style="width: 8%"></div>
      </div>
      <strong class="run-progress-percent">0%</strong>
    </div>
  `;
  progress.classList.remove("is-hidden");
  setOutput(output, "Starting comparison...");
  try {
    const data = await requestJSON("/api/comparisons", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJob = data.job;
    await pollComparisonJob(data.job);
  } catch (error) {
    setOutput(output, error.message);
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
$("#refresh-button").addEventListener("click", refresh);
$("#import-canonical-button").addEventListener("click", importCanonicalSuite);
$("#import-german-button").addEventListener("click", importGermanInstances);
$("#scan-osm-files").addEventListener("click", scanOsmFiles);
$("#create-form").addEventListener("submit", createCampaign);
$("#run-form").addEventListener("submit", runCampaign);
$("#compare-form").addEventListener("submit", runComparison);
$("#create-name").addEventListener("input", updateCampaignNameIndicator);
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
