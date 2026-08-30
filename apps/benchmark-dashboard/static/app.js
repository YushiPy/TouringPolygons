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
  manualCampaign: "",
  loadedManualCampaign: "",
  manualCases: [],
  manualCaseIndex: 0,
  manualAutosaveTimer: null,
  manualAutosaving: false,
  manualAutosaveQueued: false,
  manualRenamingIndex: null,
  campaignCaseMetadata: new Map(),
  instanceModalReturn: null,
  finishedDockJob: null,
};

const $ = (selector) => document.querySelector(selector);
const KEYBIND_STORAGE_KEY = "benchmarkDashboardManualEditorKeybinds";
const THEME_STORAGE_KEY = "benchmarkDashboardTheme";
const CLI_SOLVERS = {
  linear: "linear_search_lazy",
  linear_disjoint: "linear_search_disjoint",
  binary: "binary_search_lazy",
  binary_disjoint: "binary_search_disjoint",
  tan: "tan_jiang",
  gurobi: "gurobi",
};
const defaultKeybinds = {
  closePolygon: ["Enter", "C"],
  deleteSelection: ["X"],
  clearSelection: ["Z"],
  toggleSnap: ["S"],
  fitInstance: ["F"],
  toggleGrid: ["G"],
  togglePath: ["P"],
  toggleDecomposition: ["D"],
  toggleLabels: ["L"],
};
let editorKeybinds = loadEditorKeybinds();
let pendingKeybindAction = null;
let floatingTooltip = null;

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
  if (!target) {
    return;
  }
  target.textContent = text || "";
}

function shellQuote(value) {
  const text = String(value ?? "");
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(text)) {
    return text;
  }
  return `'${text.replaceAll("'", "'\\''")}'`;
}

function runCommandFromForm(form = $("#run-form")) {
  const values = formData(form);
  const command = ["python3", "benchmarks/tpp.py", "run", values.name];
  if (values.threads) {
    command.push("--threads", values.threads);
  }
  if (values.solver) {
    command.push("--solver", CLI_SOLVERS[values.solver] || values.solver);
  }
  if (values.max_instances) {
    command.push("--max-instances", values.max_instances);
  }
  command.push("--max-calls", values.max_calls || "1000000");
  if (values.max_seconds) {
    command.push("--max-seconds", values.max_seconds);
  }
  if (values.timeout) {
    command.push("--timeout", values.timeout);
  }
  if (boolField(form, "force")) {
    command.push("--force");
  }
  if (boolField(form, "no_build")) {
    command.push("--no-build");
  }
  if (boolField(form, "dry_run")) {
    command.push("--dry-run");
  }
  return command.map(shellQuote).join(" ");
}

function compareCommandFromForm(form = $("#compare-form")) {
  const values = formData(form);
  const solvers = [...form.querySelectorAll('input[name="solvers"]:checked')].map((input) => input.value);
  const inputDir = shellQuote(`benchmarks/campaigns/${values.name}/inputs`);
  const command = [
    "python3",
    "benchmarks/tpp.py",
    "compare-solvers",
    "--suite",
    `$(find ${inputDir} -name '*.bin' | sort | head -n 1)`,
    "--output",
    `benchmarks/campaigns/${values.name}/results/comparisons`,
    "--max-calls",
    values.max_calls || "1000000",
  ];
  if (values.max_instances) {
    command.push("--max-instances", values.max_instances);
  }
  command.push("--max-polygons", "-1", "--max-branching", "-1", "--keep-going");
  if (values.threads) {
    command.push("--threads", values.threads);
  }
  if (values.max_seconds) {
    command.push("--max-seconds", values.max_seconds);
  }
  if (boolField(form, "no_build")) {
    command.push("--no-build");
  }
  for (const solver of solvers) {
    command.push("--solver", CLI_SOLVERS[solver] || solver);
  }
  return command.map((part) => String(part).startsWith("$(") ? part : shellQuote(part)).join(" ");
}

async function copyText(text, button) {
  if (!text) {
    return;
  }
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
  } else {
    const selection = window.getSelection();
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    document.execCommand("copy");
    document.body.removeChild(scratch);
    selection?.removeAllRanges();
  }
  if (button) {
    const original = button.textContent;
    button.textContent = "Copied";
    clearTimeout(button._copyTimer);
    button._copyTimer = setTimeout(() => {
      button.textContent = original;
    }, 1400);
  }
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

function bindTapZoom(button, action) {
  if (!button) {
    return;
  }
  button.addEventListener("click", () => {
    if (Date.now() - (button._lastTouchZoomAt || 0) < 450) {
      return;
    }
    action();
  });
  button.addEventListener("touchend", (event) => {
    event.preventDefault();
    button._lastTouchZoomAt = Date.now();
    action();
  }, { passive: false });
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
  if (Math.abs(number) >= 0.000001) {
    return `${(number * 1000000).toFixed(2)} us`;
  }
  return `${(number * 1000000000).toFixed(2)} ns`;
}

function instanceDisplayName(campaign, index) {
  const cases = state.campaignCaseMetadata.get(campaign.name) || [];
  return cases[index]?.name || `Instance ${instanceLabel(index)}`;
}

function instanceTitle(campaign, index) {
  const total = campaign.instance_progress?.total || campaign.generation?.instances || campaign.generation?.instances_per_file || "?";
  return `${instanceLabel(index)}/${total}: ${instanceDisplayName(campaign, index)}`;
}

function autoGeneratedSticker(label = "This test case was generated automatically.") {
  return `
    <span class="auto-sticker" tabindex="0" data-tooltip="${escapeHTML(label)}" aria-label="${escapeHTML(label)}">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M6.1 1.4h1.8l.3 1.2c.3.1.6.2.9.4l1.1-.6 1.3 1.3-.6 1.1c.2.3.3.6.4.9l1.2.3v1.8l-1.2.3c-.1.3-.2.6-.4.9l.6 1.1-1.3 1.3-1.1-.6c-.3.2-.6.3-.9.4l-.3 1.2H6.1l-.3-1.2c-.3-.1-.6-.2-.9-.4l-1.1.6-1.3-1.3.6-1.1c-.2-.3-.3-.6-.4-.9l-1.2-.3V6l1.2-.3c.1-.3.2-.6.4-.9l-.6-1.1 1.3-1.3 1.1.6c.3-.2.6-.3.9-.4l.3-1.2zm.9 3.5a2.2 2.2 0 100 4.4 2.2 2.2 0 000-4.4zm4.9 5.9h1.2l.2.8.6.3.7-.4.9.9-.4.7.3.6.8.2v1.2l-.8.2-.3.6.4.7-.9.9-.7-.4-.6.3-.2.8h-1.2l-.2-.8-.6-.3-.7.4-.9-.9.4-.7-.3-.6-.8-.2v-1.2l.8-.2.3-.6-.4-.7.9-.9.7.4.6-.3.2-.8zm.6 2.3a1.4 1.4 0 100 2.8 1.4 1.4 0 000-2.8z"></path>
      </svg>
    </span>
  `;
}

function showFloatingTooltip(anchor) {
  const label = anchor?.dataset?.tooltip;
  if (!anchor || !label) {
    return;
  }
  if (!floatingTooltip) {
    floatingTooltip = document.createElement("div");
    floatingTooltip.className = "floating-tooltip";
    document.body.appendChild(floatingTooltip);
  }
  floatingTooltip.textContent = label;
  floatingTooltip.classList.add("is-visible");
  const rect = anchor.getBoundingClientRect();
  const tooltipRect = floatingTooltip.getBoundingClientRect();
  const gap = 10;
  let left = rect.right + gap;
  let top = rect.top + rect.height / 2 - tooltipRect.height / 2;
  if (left + tooltipRect.width > window.innerWidth - 8) {
    left = rect.left - tooltipRect.width - gap;
    floatingTooltip.classList.add("is-left");
  } else {
    floatingTooltip.classList.remove("is-left");
  }
  top = Math.max(8, Math.min(window.innerHeight - tooltipRect.height - 8, top));
  floatingTooltip.style.left = `${left}px`;
  floatingTooltip.style.top = `${top}px`;
}

function hideFloatingTooltip() {
  floatingTooltip?.classList.remove("is-visible");
}

function applyTheme(theme) {
  const dark = theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const button = $("#theme-toggle");
  if (button) {
    button.innerHTML = `${themeIcon(dark)}<span>${dark ? "Light" : "Dark"}</span>`;
    button.setAttribute("aria-pressed", dark ? "true" : "false");
    button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  }
  localStorage.setItem(THEME_STORAGE_KEY, dark ? "dark" : "light");
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
}

function themeIcon(dark) {
  return dark
    ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5a.6.6 0 01.6.6v1.2a.6.6 0 11-1.2 0V2.1a.6.6 0 01.6-.6zm4.6 2.5a.6.6 0 010 .8l-.8.8a.6.6 0 11-.8-.8l.8-.8a.6.6 0 01.8 0zM8 5.2A2.8 2.8 0 108 10.8 2.8 2.8 0 008 5.2zm6.5 2.8a.6.6 0 01-.6.6h-1.2a.6.6 0 010-1.2h1.2a.6.6 0 01.6.6zM12.6 12a.6.6 0 01-.8 0l-.8-.8a.6.6 0 11.8-.8l.8.8a.6.6 0 010 .8zM8 12.1a.6.6 0 01.6.6v1.2a.6.6 0 11-1.2 0v-1.2a.6.6 0 01.6-.6zM5 10.4a.6.6 0 010 .8l-.8.8a.6.6 0 11-.8-.8l.8-.8a.6.6 0 01.8 0zM3.9 8a.6.6 0 01-.6.6H2.1a.6.6 0 110-1.2h1.2a.6.6 0 01.6.6zM5 4.8a.6.6 0 01-.8.8l-.8-.8a.6.6 0 11.8-.8l.8.8z"></path></svg>'
    : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M9.7 1.5a6.1 6.1 0 104.8 8.8.6.6 0 00-.8-.8A4.4 4.4 0 016.5 4.4c0-.7.2-1.5.5-2.1a.6.6 0 00-.7-.8z"></path></svg>';
}

function cloneCaseData(data) {
  return {
    name: data?.name || "",
    generated: Boolean(data?.generated),
    start: [...(data?.start || [0, 0])],
    target: [...(data?.target || [1, 0])],
    polygons: (data?.polygons || []).map((polygon) => polygon.map((point) => [...point])),
  };
}

function emptyCaseData() {
  return {
    name: "",
    generated: false,
    start: [0, 0],
    target: [1, 0],
    polygons: [],
  };
}

function casePayload(data) {
  const clone = cloneCaseData(data);
  clone.polygons = clone.polygons.filter((polygon) => polygon.length >= 3);
  return clone;
}

function loadEditorKeybinds() {
  try {
    const loaded = JSON.parse(localStorage.getItem(KEYBIND_STORAGE_KEY) || "{}");
    return {
      closePolygon: normalizeBindings(loaded.closePolygon, defaultKeybinds.closePolygon),
      deleteSelection: normalizeBindings(loaded.deleteSelection, defaultKeybinds.deleteSelection),
      clearSelection: normalizeBindings(loaded.clearSelection, defaultKeybinds.clearSelection),
      toggleSnap: normalizeBindings(loaded.toggleSnap, defaultKeybinds.toggleSnap),
      fitInstance: normalizeBindings(loaded.fitInstance, defaultKeybinds.fitInstance),
      toggleGrid: normalizeBindings(loaded.toggleGrid, defaultKeybinds.toggleGrid),
      togglePath: normalizeBindings(loaded.togglePath, defaultKeybinds.togglePath),
      toggleDecomposition: normalizeBindings(loaded.toggleDecomposition, defaultKeybinds.toggleDecomposition),
      toggleLabels: normalizeBindings(loaded.toggleLabels, defaultKeybinds.toggleLabels),
    };
  } catch {
    return {
      closePolygon: [...defaultKeybinds.closePolygon],
      deleteSelection: [...defaultKeybinds.deleteSelection],
      clearSelection: [...defaultKeybinds.clearSelection],
      toggleSnap: [...defaultKeybinds.toggleSnap],
      fitInstance: [...defaultKeybinds.fitInstance],
      toggleGrid: [...defaultKeybinds.toggleGrid],
      togglePath: [...defaultKeybinds.togglePath],
      toggleDecomposition: [...defaultKeybinds.toggleDecomposition],
      toggleLabels: [...defaultKeybinds.toggleLabels],
    };
  }
}

function normalizeBindings(value, fallback) {
  if (Array.isArray(value)) {
    return value.filter(Boolean);
  }
  return value ? [value] : [...fallback];
}

function saveEditorKeybinds() {
  localStorage.setItem(KEYBIND_STORAGE_KEY, JSON.stringify(editorKeybinds));
}

function keyEventToBinding(event) {
  const parts = [];
  if (event.ctrlKey) {
    parts.push("Ctrl");
  }
  if (event.altKey) {
    parts.push("Alt");
  }
  if (event.shiftKey) {
    parts.push("Shift");
  }
  if (event.metaKey) {
    parts.push("Meta");
  }
  if (["Control", "Alt", "Shift", "Meta"].includes(event.key)) {
    return "";
  }
  const key = event.key === " " ? "Space" : event.key.length === 1 ? event.key.toUpperCase() : event.key;
  parts.push(key);
  return parts.join("+");
}

function keyMatchesBinding(event, bindings) {
  return bindings.includes(keyEventToBinding(event));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function updateKeybindUI() {
  renderKeybindControl("closePolygon", "#close-polygon-keybinds");
  renderKeybindControl("deleteSelection", "#delete-selection-keybinds");
  renderKeybindControl("clearSelection", "#clear-selection-keybinds");
  renderKeybindControl("toggleSnap", "#toggle-snap-keybinds");
  renderKeybindControl("fitInstance", "#fit-instance-keybinds");
  renderKeybindControl("toggleGrid", "#toggle-grid-keybinds");
  renderKeybindControl("togglePath", "#toggle-path-keybinds");
  renderKeybindControl("toggleDecomposition", "#toggle-decomposition-keybinds");
  renderKeybindControl("toggleLabels", "#toggle-labels-keybinds");
}

function renderKeybindControl(action, selector) {
  const root = $(selector);
  if (!root) return;
  root.innerHTML = "";
  editorKeybinds[action].forEach((binding, index) => {
    const wrapper = document.createElement("span");
    wrapper.className = "keybind-chip";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary keybind-input";
    button.textContent = pendingKeybindAction?.action === action && pendingKeybindAction.index === index ? "Press keys..." : binding;
    button.addEventListener("click", () => {
      pendingKeybindAction = { action, index };
      updateKeybindUI();
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "keybind-remove";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${binding}`);
    remove.addEventListener("click", () => {
      editorKeybinds[action].splice(index, 1);
      saveEditorKeybinds();
      updateKeybindUI();
    });
    wrapper.append(button, remove);
    root.appendChild(wrapper);
  });
  const add = document.createElement("button");
  add.type = "button";
  add.className = "secondary keybind-input";
  add.textContent = pendingKeybindAction?.action === action && pendingKeybindAction.index === editorKeybinds[action].length ? "Press keys..." : "...";
  add.addEventListener("click", () => {
    pendingKeybindAction = { action, index: editorKeybinds[action].length };
    updateKeybindUI();
  });
  root.appendChild(add);
}

function openKeybinds() {
  pendingKeybindAction = null;
  updateKeybindUI();
  $("#keybind-modal")?.classList.add("is-top-modal");
  $("#keybind-modal")?.classList.remove("is-hidden");
}

function closeKeybinds() {
  if (pendingKeybindAction) {
    editorKeybinds[pendingKeybindAction.action].splice(pendingKeybindAction.index, 1);
    editorKeybinds[pendingKeybindAction.action] = editorKeybinds[pendingKeybindAction.action].filter(Boolean);
    saveEditorKeybinds();
  }
  pendingKeybindAction = null;
  updateKeybindUI();
  $("#keybind-modal")?.classList.remove("is-top-modal");
  $("#keybind-modal")?.classList.add("is-hidden");
}

function signedArea(points) {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const next = points[(index + 1) % points.length];
    area += point[0] * next[1] - next[0] * point[1];
  }
  return area / 2;
}

function pointInTriangle(point, a, b, c) {
  const area = (u, v, w) => (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
  const ab = area(a, b, point);
  const bc = area(b, c, point);
  const ca = area(c, a, point);
  return (ab >= -1e-9 && bc >= -1e-9 && ca >= -1e-9) || (ab <= 1e-9 && bc <= 1e-9 && ca <= 1e-9);
}

function earClipDecomposition(polygon) {
  if (polygon.length < 3) {
    return [];
  }
  if (polygonIsConvex(polygon)) {
    return [polygon];
  }
  const oriented = signedArea(polygon) >= 0 ? polygon.map((point) => [...point]) : [...polygon].reverse().map((point) => [...point]);
  const remaining = oriented.map((_, index) => index);
  const pieces = [];
  let guard = 0;
  while (remaining.length > 3 && guard < oriented.length * oriented.length) {
    let clipped = false;
    for (let index = 0; index < remaining.length; index += 1) {
      const previousIndex = remaining[(index + remaining.length - 1) % remaining.length];
      const currentIndex = remaining[index];
      const nextIndex = remaining[(index + 1) % remaining.length];
      const previous = oriented[previousIndex];
      const current = oriented[currentIndex];
      const next = oriented[nextIndex];
      const cross = (current[0] - previous[0]) * (next[1] - current[1]) - (current[1] - previous[1]) * (next[0] - current[0]);
      if (cross <= 1e-9) {
        continue;
      }
      const containsOther = remaining.some((candidateIndex) => (
        candidateIndex !== previousIndex
        && candidateIndex !== currentIndex
        && candidateIndex !== nextIndex
        && pointInTriangle(oriented[candidateIndex], previous, current, next)
      ));
      if (containsOther) {
        continue;
      }
      pieces.push([previous, current, next].map((point) => [...point]));
      remaining.splice(index, 1);
      clipped = true;
      break;
    }
    if (!clipped) {
      return [polygon];
    }
    guard += 1;
  }
  if (remaining.length === 3) {
    pieces.push(remaining.map((index) => [...oriented[index]]));
  }
  return pieces;
}

function convexDecomposition(polygon) {
  return earClipDecomposition(polygon).filter((piece) => piece.length >= 3);
}

function solutionDirectionAt(path, index) {
  if (!path || path.length < 2) {
    return index === 0 ? [1, 0] : [-1, 0];
  }
  if (index === 0) {
    return [path[0][0] - path[1][0], path[0][1] - path[1][1]];
  }
  const last = path.length - 1;
  return [path[last][0] - path[last - 1][0], path[last][1] - path[last - 1][1]];
}

let editorWasmModule = null;
let editorWasmLoad = null;
let editorWasmFailed = false;
let editorGeometryModule = null;

function loadEditorWasm() {
  if (editorWasmLoad) {
    return editorWasmLoad;
  }
  editorWasmLoad = import("/visualizer-static/wasm/tpp_convex_wasm.js")
    .then((module) => module.default({
      locateFile: (path) => path.endsWith(".wasm") ? `/visualizer-static/wasm/${path}` : path,
    }))
    .then((module) => {
      editorWasmModule = module;
      return module;
    })
    .catch(() => {
      editorWasmFailed = true;
      editorWasmModule = null;
      return null;
    });
  return editorWasmLoad;
}

async function loadEditorGeometry() {
  if (!globalThis.polygonClipping) {
    return null;
  }
  if (editorGeometryModule) {
    return editorGeometryModule;
  }
  try {
    editorGeometryModule = {
      vector: await import("/visualizer-static/js/vector2.js"),
      partition: await import("/visualizer-static/js/convex-partition.js"),
    };
  } catch {
    editorGeometryModule = null;
  }
  return editorGeometryModule;
}

function polygonIsConvex(polygon) {
  let gotNegative = false;
  let gotPositive = false;
  for (let index = 0; index < polygon.length; index += 1) {
    const a = polygon[index];
    const b = polygon[(index + 1) % polygon.length];
    const c = polygon[(index + 2) % polygon.length];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (cross < 0) {
      gotNegative = true;
    } else if (cross > 0) {
      gotPositive = true;
    }
    if (gotNegative && gotPositive) {
      return false;
    }
  }
  return true;
}

function solveEditorWasm(caseData, maxCalls = 200000, maxSeconds = 3) {
  if (!editorWasmModule) {
    return null;
  }
  const polygons = caseData.polygons;
  const totalVertices = polygons.reduce((sum, polygon) => sum + polygon.length, 0);
  const pointsPtr = editorWasmModule._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
  const sizesPtr = editorWasmModule._malloc(polygons.length * Int32Array.BYTES_PER_ELEMENT);
  try {
    const points = new Float64Array(editorWasmModule.HEAPF64.buffer, pointsPtr, totalVertices * 2);
    const sizes = new Int32Array(editorWasmModule.HEAP32.buffer, sizesPtr, polygons.length);
    let pointIndex = 0;
    polygons.forEach((polygon, polygonIndex) => {
      sizes[polygonIndex] = polygon.length;
      polygon.forEach((point) => {
        points[2 * pointIndex] = point[0];
        points[2 * pointIndex + 1] = point[1];
        pointIndex += 1;
      });
    });
    const pathSize = editorWasmModule._tpp_solve(
      caseData.start[0],
      caseData.start[1],
      caseData.target[0],
      caseData.target[1],
      pointsPtr,
      sizesPtr,
      polygons.length,
      maxCalls,
      maxSeconds,
    );
    if (pathSize < 0) {
      return null;
    }
    const outputPtr = editorWasmModule._tpp_get_path_points();
    const output = new Float64Array(editorWasmModule.HEAPF64.buffer, outputPtr, pathSize * 2);
    const path = [];
    for (let index = 0; index < pathSize; index += 1) {
      path.push([output[2 * index], output[2 * index + 1]]);
    }
    return {
      path,
      exact: editorWasmModule._tpp_solution_exact() === 1,
      calls: editorWasmModule._tpp_solution_calls(),
      seconds: editorWasmModule._tpp_solution_seconds(),
      source: "wasm",
    };
  } finally {
    editorWasmModule._free(pointsPtr);
    editorWasmModule._free(sizesPtr);
  }
}

function solveEditorWasmGroups(caseData, pieceGroups, maxCalls = 200000, maxSeconds = 3) {
  if (!editorWasmModule) {
    return null;
  }
  const pieces = pieceGroups.flat();
  const totalVertices = pieces.reduce((sum, piece) => sum + piece.length, 0);
  const pointsPtr = editorWasmModule._malloc(totalVertices * 2 * Float64Array.BYTES_PER_ELEMENT);
  const pieceSizesPtr = editorWasmModule._malloc(pieces.length * Int32Array.BYTES_PER_ELEMENT);
  const groupSizesPtr = editorWasmModule._malloc(pieceGroups.length * Int32Array.BYTES_PER_ELEMENT);
  try {
    const points = new Float64Array(editorWasmModule.HEAPF64.buffer, pointsPtr, totalVertices * 2);
    const pieceSizes = new Int32Array(editorWasmModule.HEAP32.buffer, pieceSizesPtr, pieces.length);
    const groupSizes = new Int32Array(editorWasmModule.HEAP32.buffer, groupSizesPtr, pieceGroups.length);
    let pointIndex = 0;
    let pieceIndex = 0;
    pieceGroups.forEach((group, groupIndex) => {
      groupSizes[groupIndex] = group.length;
      group.forEach((piece) => {
        pieceSizes[pieceIndex] = piece.length;
        pieceIndex += 1;
        piece.forEach((point) => {
          points[2 * pointIndex] = point[0];
          points[2 * pointIndex + 1] = point[1];
          pointIndex += 1;
        });
      });
    });
    const pathSize = editorWasmModule._tpp_solve_piece_groups(
      caseData.start[0],
      caseData.start[1],
      caseData.target[0],
      caseData.target[1],
      pointsPtr,
      pieceSizesPtr,
      groupSizesPtr,
      pieceGroups.length,
      maxCalls,
      maxSeconds,
    );
    if (pathSize < 0) {
      return null;
    }
    const outputPtr = editorWasmModule._tpp_get_path_points();
    const output = new Float64Array(editorWasmModule.HEAPF64.buffer, outputPtr, pathSize * 2);
    const path = [];
    for (let index = 0; index < pathSize; index += 1) {
      path.push([output[2 * index], output[2 * index + 1]]);
    }
    return {
      path,
      exact: editorWasmModule._tpp_solution_exact() === 1,
      calls: editorWasmModule._tpp_solution_calls(),
      seconds: editorWasmModule._tpp_solution_seconds(),
      source: "wasm",
    };
  } finally {
    editorWasmModule._free(pointsPtr);
    editorWasmModule._free(pieceSizesPtr);
    editorWasmModule._free(groupSizesPtr);
  }
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
  if (panelId === "cases-panel") {
    requestAnimationFrame(() => manualEditor.frameCurrentCase());
  }
  dismissFinishedJobForPanel(panelId);
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

const manualEditor = {
  canvas: null,
  ctx: null,
  mode: "move",
  scale: 70,
  minScale: 0.1,
  maxScale: 50000,
  offsetX: 0,
  offsetY: 0,
  activePoint: null,
  selectedPoint: null,
  selectedPoints: [],
  selectionBase: [],
  dragPolygon: null,
  dragSelection: null,
  panDrag: null,
  pointerStart: null,
  activePointers: new Map(),
  pinchGesture: null,
  selectionRect: null,
  mouseCanvas: { x: 0, y: 0 },
  selectionSpinStarted: performance.now(),
  selectionAnimationRunning: false,
  snapping: false,
  activePolygon: null,
  solutionPath: null,
  solutionStale: false,
  solutionTimer: null,
  solutionFrame: null,
  solutionAbort: null,
  solutionRevision: 0,
  layers: {
    grid: true,
    solution: true,
    decomposition: true,
    labels: true,
  },
  labelDirections: {
    start: [1, 0],
    target: [-1, 0],
  },
  labelAnimation: null,
  expanded: false,

  init() {
    this.canvas = $("#manual-case-canvas");
    if (!this.canvas) {
      return;
    }
    this.ctx = this.canvas.getContext("2d");
    this.canvas.tabIndex = 0;
    this.loadCamera();
    this.resize();
    new ResizeObserver(() => {
      this.resize();
      this.draw();
    }).observe(this.canvas);
    this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
    $("#cases-panel")?.addEventListener("click", (event) => {
      if (this.expanded && event.target === event.currentTarget) {
        this.toggleExpanded(false);
      }
    });
    const finishPointer = (event) => {
      this.activePointers.delete(event.pointerId);
      if (this.pinchGesture && this.activePointers.size < 2) {
        this.pinchGesture = null;
      }
      this.activePoint = null;
      this.dragSelection = null;
      this.dragPolygon = null;
      const pointerStart = this.pointerStart;
      this.panDrag = null;
      this.pointerStart = null;
      if (this.selectionRect) {
        this.finishSelection();
      } else if (pointerStart?.kind === "pan") {
        const moved = Math.hypot(event.clientX - pointerStart.clientX, event.clientY - pointerStart.clientY) > 4;
        if (!moved) {
          this.clearSelection();
        }
      }
      if (event.pointerId !== undefined && this.canvas.hasPointerCapture?.(event.pointerId)) {
        this.canvas.releasePointerCapture(event.pointerId);
      }
      this.updateCursor();
    };
    window.addEventListener("pointerup", finishPointer);
    window.addEventListener("pointercancel", finishPointer);
    this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    document.addEventListener("keydown", (event) => this.onKeyDown(event));
    loadEditorWasm();
    loadEditorGeometry();
    this.syncCloseButton();
    this.toggleSnapping(false);
    this.draw();
  },

  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.canvas.offsetWidth * dpr));
    this.canvas.height = Math.max(1, Math.round(this.canvas.offsetHeight * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.offsetX === 0 && this.offsetY === 0) {
      this.offsetX = this.canvas.offsetWidth / 2;
      this.offsetY = this.canvas.offsetHeight / 2;
    }
  },

  loadCamera() {
    try {
      const camera = JSON.parse(localStorage.getItem("benchmarkDashboardManualEditorCamera") || "null");
      if (!camera) {
        return;
      }
      this.scale = Number(camera.scale) || this.scale;
      this.offsetX = Number(camera.offsetX) || this.offsetX;
      this.offsetY = Number(camera.offsetY) || this.offsetY;
    } catch {
      // Ignore stale local camera data.
    }
  },

  saveCamera() {
    localStorage.setItem("benchmarkDashboardManualEditorCamera", JSON.stringify({
      scale: this.scale,
      offsetX: this.offsetX,
      offsetY: this.offsetY,
    }));
  },

  currentCase() {
    return state.manualCases[state.manualCaseIndex] || null;
  },

  caseBounds(caseData = this.currentCase()) {
    if (!caseData) {
      return null;
    }
    const points = [caseData.start, caseData.target, ...caseData.polygons.flat()];
    if (points.length === 0) {
      return null;
    }
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    return {
      minX: Math.min(...xs),
      minY: Math.min(...ys),
      maxX: Math.max(...xs),
      maxY: Math.max(...ys),
    };
  },

  updateZoomLimits(bounds = this.caseBounds()) {
    if (!bounds || !this.canvas) {
      this.minScale = 0.1;
      this.maxScale = 50000;
      return;
    }
    const width = Math.max(this.canvas.offsetWidth, 1);
    const height = Math.max(this.canvas.offsetHeight, 1);
    const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
    const fitScale = Math.min(width / spanX, height / spanY);
    this.minScale = Math.max(0.00001, fitScale * 0.025);
    this.maxScale = Math.min(2000000, Math.max(fitScale * 180, this.minScale * 10));
    this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale));
  },

  frameCurrentCase() {
    const bounds = this.caseBounds();
    if (!bounds || !this.canvas) {
      return;
    }
    this.updateZoomLimits(bounds);
    const width = this.canvas.offsetWidth;
    const height = this.canvas.offsetHeight;
    const pad = 42;
    const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
    this.scale = Math.max(this.minScale, Math.min(this.maxScale, Math.min(
      (width - 2 * pad) / spanX,
      (height - 2 * pad) / spanY,
    )));
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    this.offsetX = width / 2 - centerX * this.scale;
    this.offsetY = height / 2 + centerY * this.scale;
    this.saveCamera();
    this.draw();
  },

  toggleLayer(layer, force = null) {
    if (!(layer in this.layers)) {
      return;
    }
    this.layers[layer] = force === null ? !this.layers[layer] : force;
    const button = document.querySelector(`[data-editor-layer="${layer}"]`);
    button?.classList.toggle("is-active", this.layers[layer]);
    button?.setAttribute("aria-pressed", this.layers[layer] ? "true" : "false");
    this.draw();
  },

  pointRadius(kind = "vertex") {
    const base = Math.sqrt(Math.max(0.1, this.scale)) * 0.62;
    const radius = Math.max(1.4, Math.min(kind === "endpoint" ? 6.5 : 5.2, base));
    return radius;
  },

  worldToCanvas(point) {
    return {
      x: this.offsetX + point[0] * this.scale,
      y: this.offsetY - point[1] * this.scale,
    };
  },

  canvasToWorld(x, y) {
    return [
      (x - this.offsetX) / this.scale,
      -(y - this.offsetY) / this.scale,
    ];
  },

  snapStep() {
    return this.gridMetrics().subGridSpacing;
  },

  gridMetrics() {
    const decisionValue = 83 / this.scale;
    let exponent = Math.ceil(Math.log10(decisionValue)) || 0;
    let multiplier = 1;
    let subGridCount = 4;
    const gridScale = 10 ** exponent;
    if (gridScale / 5 > decisionValue) {
      subGridCount = 3;
      exponent -= 1;
      multiplier = 2;
    } else if (gridScale / 2 > decisionValue) {
      exponent -= 1;
      multiplier = 5;
    }
    const gridSpacing = (10 ** exponent) * multiplier;
    return {
      gridSpacing,
      subGridCount,
      subGridSpacing: gridSpacing / (subGridCount + 1),
    };
  },

  snap(point) {
    if (!this.snapping) {
      return point;
    }
    const step = this.snapStep();
    return point.map((value) => Math.round(value / step) * step);
  },

  setMode(mode) {
    this.mode = mode;
    document.querySelectorAll("[data-manual-mode] .segment").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.mode === mode);
    });
    this.updateCursor();
    this.syncCloseButton();
    this.draw();
  },

  toggleSnapping(force = null) {
    this.snapping = force === null ? !this.snapping : force;
    const button = $("#toggle-manual-snapping");
    button?.classList.toggle("is-active", this.snapping);
    button?.setAttribute("aria-pressed", this.snapping ? "true" : "false");
    this.draw();
  },

  zoomBy(factor) {
    const centerX = this.canvas.offsetWidth / 2;
    const centerY = this.canvas.offsetHeight / 2;
    const before = this.canvasToWorld(centerX, centerY);
    this.updateZoomLimits();
    this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
    const after = this.canvasToWorld(centerX, centerY);
    this.offsetX += (after[0] - before[0]) * this.scale;
    this.offsetY -= (after[1] - before[1]) * this.scale;
    this.saveCamera();
    this.draw();
  },

  toggleExpanded(force = null) {
    const centerWorld = this.canvasToWorld(this.canvas.offsetWidth / 2, this.canvas.offsetHeight / 2);
    this.expanded = force === null ? !this.expanded : force;
    const panel = $("#cases-panel");
    const button = $("#manual-editor-expand");
    panel?.classList.toggle("editor-expanded", this.expanded);
    document.body.classList.toggle("manual-editor-is-expanded", this.expanded);
    button?.setAttribute("aria-pressed", this.expanded ? "true" : "false");
    if (button) {
      button.textContent = this.expanded ? "Collapse" : "Expand";
    }
    requestAnimationFrame(() => {
      this.resize();
      this.offsetX = this.canvas.offsetWidth / 2 - centerWorld[0] * this.scale;
      this.offsetY = this.canvas.offsetHeight / 2 + centerWorld[1] * this.scale;
      this.saveCamera();
      this.draw();
    });
  },

  setSolveStatus(message) {
    const status = $("#manual-solve-status");
    if (status) {
      status.textContent = message || "";
    }
  },

  setSaveStatus(message) {
    const status = $("#manual-save-status");
    if (status) {
      status.textContent = message || "";
    }
  },

  setStatus(message) {
    this.setSolveStatus(message);
  },

  syncCloseButton() {
    const button = $("#close-manual-polygon");
    if (!button) {
      return;
    }
    const current = this.currentCase();
    const polygon = current && this.activePolygon !== null ? current.polygons[this.activePolygon] : null;
    const canClose = Boolean(polygon && polygon.length >= 3);
    button.classList.toggle("is-disabled", !canClose);
    button.disabled = !canClose;
    button.setAttribute("aria-disabled", canClose ? "false" : "true");
  },

  showCloseHint() {
    const button = $("#close-manual-polygon");
    if (!button) {
      return;
    }
    button.classList.add("show-tooltip");
    clearTimeout(button._tooltipTimer);
    button._tooltipTimer = setTimeout(() => button.classList.remove("show-tooltip"), 1800);
  },

  updateLabelDirections(animate = false) {
    const targets = {
      start: solutionDirectionAt(this.solutionPath, 0),
      target: solutionDirectionAt(this.solutionPath, 1),
    };
    if (!animate) {
      this.labelDirections = targets;
      return;
    }
    if (this.labelAnimation) {
      cancelAnimationFrame(this.labelAnimation);
    }
    let frame = 0;
    const step = () => {
      frame += 1;
      for (const key of ["start", "target"]) {
        this.labelDirections[key] = [
          this.labelDirections[key][0] + (targets[key][0] - this.labelDirections[key][0]) * 0.24,
          this.labelDirections[key][1] + (targets[key][1] - this.labelDirections[key][1]) * 0.24,
        ];
      }
      this.draw();
      if (frame < 14) {
        this.labelAnimation = requestAnimationFrame(step);
      } else {
        this.labelDirections = targets;
        this.labelAnimation = null;
        this.draw();
      }
    };
    step();
  },

  selectNearestPoint(x, y) {
    const current = this.currentCase();
    if (!current) {
      return null;
    }
    const candidates = [
      { kind: "start", point: current.start },
      { kind: "target", point: current.target },
    ];
    current.polygons.forEach((polygon, polygonIndex) => {
      polygon.forEach((point, vertexIndex) => {
        candidates.push({ kind: "vertex", point, polygonIndex, vertexIndex });
      });
    });
    let best = null;
    let bestDistance = 20;
    for (const candidate of candidates) {
      const canvasPoint = this.worldToCanvas(candidate.point);
      const distance = Math.hypot(canvasPoint.x - x, canvasPoint.y - y);
      if (distance <= bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    return best;
  },

  isClosingPolygon(x, y) {
    const current = this.currentCase();
    if (!current || this.mode !== "polygon" || this.activePolygon === null) {
      return false;
    }
    const polygon = current.polygons[this.activePolygon];
    if (!polygon || polygon.length < 3) {
      return false;
    }
    const first = this.worldToCanvas(polygon[0]);
    return Math.hypot(first.x - x, first.y - y) <= 16;
  },

  pointInPolygon(world, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0];
      const yi = polygon[i][1];
      const xj = polygon[j][0];
      const yj = polygon[j][1];
      if (((yi > world[1]) !== (yj > world[1]))
        && world[0] < ((xj - xi) * (world[1] - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  },

  selectPolygon(x, y) {
    const current = this.currentCase();
    if (!current) {
      return -1;
    }
    const world = this.canvasToWorld(x, y);
    for (let index = current.polygons.length - 1; index >= 0; index -= 1) {
      const polygon = current.polygons[index];
      if (polygon.length >= 3 && this.pointInPolygon(world, polygon)) {
        return index;
      }
    }
    return -1;
  },

  pointsInRect(rect) {
    const current = this.currentCase();
    if (!current) {
      return [];
    }
    const left = Math.min(rect.start.x, rect.end.x);
    const right = Math.max(rect.start.x, rect.end.x);
    const top = Math.min(rect.start.y, rect.end.y);
    const bottom = Math.max(rect.start.y, rect.end.y);
    const selected = [];
    [
      { kind: "start", point: current.start },
      { kind: "target", point: current.target },
    ].forEach((point) => {
      const canvasPoint = this.worldToCanvas(point.point);
      if (canvasPoint.x >= left && canvasPoint.x <= right && canvasPoint.y >= top && canvasPoint.y <= bottom) {
        selected.push(point);
      }
    });
    current.polygons.forEach((polygon, polygonIndex) => {
      polygon.forEach((point, vertexIndex) => {
        const canvasPoint = this.worldToCanvas(point);
        if (canvasPoint.x >= left && canvasPoint.x <= right && canvasPoint.y >= top && canvasPoint.y <= bottom) {
          selected.push({ kind: "vertex", point, polygonIndex, vertexIndex });
        }
      });
    });
    return selected;
  },

  samePointSelection(left, right) {
    return left?.kind === right?.kind
      && left?.polygonIndex === right?.polygonIndex
      && left?.vertexIndex === right?.vertexIndex;
  },

  mergeSelections(base, selected) {
    const merged = [...base];
    for (const point of selected) {
      if (!merged.some((existing) => this.samePointSelection(existing, point))) {
        merged.push(point);
      }
    }
    return merged;
  },

  activeTouchPoints() {
    return [...this.activePointers.values()].filter((pointer) => pointer.pointerType === "touch");
  },

  startPinchGesture() {
    const touches = this.activeTouchPoints().slice(0, 2);
    if (touches.length < 2) {
      return false;
    }
    const rect = this.canvas.getBoundingClientRect();
    const midpoint = {
      x: (touches[0].clientX + touches[1].clientX) / 2 - rect.left,
      y: (touches[0].clientY + touches[1].clientY) / 2 - rect.top,
    };
    this.pinchGesture = {
      distance: Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY),
      midpoint,
    };
    this.activePoint = null;
    this.dragSelection = null;
    this.dragPolygon = null;
    this.panDrag = null;
    this.selectionRect = null;
    this.pointerStart = { kind: "pinch", clientX: midpoint.x + rect.left, clientY: midpoint.y + rect.top };
    return true;
  },

  updatePinchGesture() {
    if (!this.pinchGesture && !this.startPinchGesture()) {
      return false;
    }
    const gesture = this.pinchGesture;
    if (!gesture || gesture.distance <= 0) {
      return false;
    }
    const touches = this.activeTouchPoints().slice(0, 2);
    const distance = Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
    if (distance <= 0) {
      return false;
    }
    const rect = this.canvas.getBoundingClientRect();
    const midpoint = {
      x: (touches[0].clientX + touches[1].clientX) / 2 - rect.left,
      y: (touches[0].clientY + touches[1].clientY) / 2 - rect.top,
    };
    const before = this.canvasToWorld(midpoint.x, midpoint.y);
    this.updateZoomLimits();
    this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * (distance / gesture.distance)));
    const after = this.canvasToWorld(midpoint.x, midpoint.y);
    this.offsetX += (after[0] - before[0]) * this.scale;
    this.offsetY -= (after[1] - before[1]) * this.scale;
    this.pinchGesture = { distance, midpoint };
    this.saveCamera();
    this.draw();
    return true;
  },

  onPointerDown(event) {
    event.preventDefault();
    const current = this.currentCase();
    if (!current) {
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    this.mouseCanvas = { x, y };
    this.activePointers.set(event.pointerId, {
      pointerType: event.pointerType,
      clientX: event.clientX,
      clientY: event.clientY,
    });
    if (event.pointerType === "touch" && this.activeTouchPoints().length >= 2) {
      this.canvas.setPointerCapture(event.pointerId);
      this.startPinchGesture();
      return;
    }
    const world = this.snap(this.canvasToWorld(x, y));
    if (this.mode === "select" || (event.shiftKey && this.mode === "move")) {
      this.selectionRect = { start: { x, y }, end: { x, y } };
      this.selectionBase = [...this.selectedPoints];
      this.selectedPoint = null;
      this.draw();
      return;
    }
    if (this.mode === "polygon") {
      if (this.isClosingPolygon(x, y)) {
        this.closePolygon();
        return;
      }
      if (this.activePolygon === null) {
        current.polygons.push([]);
        this.activePolygon = current.polygons.length - 1;
      }
      current.polygons[this.activePolygon].push(world);
      this.changed();
      return;
    }
    this.activePoint = this.selectNearestPoint(x, y);
    this.selectedPoint = this.activePoint;
    if (this.activePoint) {
      if (this.selectedPoints.some((point) => this.samePointSelection(point, this.activePoint))) {
        this.dragSelection = {
          lastWorld: this.canvasToWorld(x, y),
        };
      }
      this.pointerStart = { kind: "point", clientX: event.clientX, clientY: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.style.cursor = "grabbing";
      this.animateSelection();
      this.draw();
      return;
    }
    const polygonIndex = this.selectPolygon(x, y);
    if (polygonIndex !== -1) {
      this.dragPolygon = {
        index: polygonIndex,
        lastWorld: this.canvasToWorld(x, y),
      };
      this.pointerStart = { kind: "polygon", clientX: event.clientX, clientY: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.style.cursor = "grabbing";
      this.draw();
      return;
    }
    this.selectedPoint = null;
    this.panDrag = { x, y };
    this.pointerStart = { kind: "pan", clientX: event.clientX, clientY: event.clientY };
    this.canvas.setPointerCapture(event.pointerId);
  },

  onPointerMove(event) {
    event.preventDefault();
    if (this.activePointers.has(event.pointerId)) {
      this.activePointers.set(event.pointerId, {
        pointerType: event.pointerType,
        clientX: event.clientX,
        clientY: event.clientY,
      });
    }
    const current = this.currentCase();
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    this.mouseCanvas = { x, y };
    if (!current) {
      return;
    }
    if (this.pinchGesture || (event.pointerType === "touch" && this.activeTouchPoints().length >= 2)) {
      if (this.updatePinchGesture()) {
        return;
      }
    }
    this.updateCursor();
    if (this.selectionRect) {
      this.selectionRect.end = { x, y };
      this.selectedPoints = this.mergeSelections(this.selectionBase, this.pointsInRect(this.selectionRect));
      this.draw();
      return;
    }
    if (this.panDrag) {
      this.offsetX += x - this.panDrag.x;
      this.offsetY += y - this.panDrag.y;
      this.panDrag = { x, y };
      this.saveCamera();
      this.draw();
      return;
    }
    if (this.dragSelection) {
      const world = this.snap(this.canvasToWorld(x, y));
      const dx = world[0] - this.dragSelection.lastWorld[0];
      const dy = world[1] - this.dragSelection.lastWorld[1];
      for (const selected of this.selectedPoints) {
        const next = this.snap([selected.point[0] + dx, selected.point[1] + dy]);
        if (selected.kind === "start") {
          current.start = next;
        } else if (selected.kind === "target") {
          current.target = next;
        } else if (selected.kind === "vertex") {
          current.polygons[selected.polygonIndex][selected.vertexIndex] = next;
        }
        selected.point = next;
      }
      this.dragSelection.lastWorld = world;
      this.changed();
      return;
    }
    if (this.dragPolygon) {
      const world = this.snap(this.canvasToWorld(x, y));
      const dx = world[0] - this.dragPolygon.lastWorld[0];
      const dy = world[1] - this.dragPolygon.lastWorld[1];
      current.polygons[this.dragPolygon.index] = current.polygons[this.dragPolygon.index]
        .map(([px, py]) => this.snap([px + dx, py + dy]));
      for (const selected of this.selectedPoints) {
        if (selected.kind === "vertex" && selected.polygonIndex === this.dragPolygon.index) {
          selected.point = current.polygons[selected.polygonIndex][selected.vertexIndex];
        }
      }
      this.dragPolygon.lastWorld = world;
      this.changed();
      return;
    }
    if (!this.activePoint) {
      if (this.mode === "polygon") {
        this.draw();
      }
      return;
    }
    const world = this.snap(this.canvasToWorld(x, y));
    if (this.activePoint.kind === "start") {
      current.start = world;
    } else if (this.activePoint.kind === "target") {
      current.target = world;
    } else if (this.activePoint.kind === "vertex") {
      current.polygons[this.activePoint.polygonIndex][this.activePoint.vertexIndex] = world;
      this.activePoint.point = world;
    }
    this.changed();
  },

  onWheel(event) {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const before = this.canvasToWorld(event.clientX - rect.left, event.clientY - rect.top);
    const delta = Math.max(-80, Math.min(80, event.deltaY));
    const factor = Math.exp(-delta * (event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? 0.0028 : 0.08));
    this.updateZoomLimits();
    this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
    const after = this.canvasToWorld(event.clientX - rect.left, event.clientY - rect.top);
    this.offsetX += (after[0] - before[0]) * this.scale;
    this.offsetY -= (after[1] - before[1]) * this.scale;
    this.saveCamera();
    this.draw();
  },

  updateCursor() {
    if (!this.canvas) {
      return;
    }
    if (this.mode === "polygon") {
      this.canvas.style.cursor = this.isClosingPolygon(this.mouseCanvas.x, this.mouseCanvas.y) ? "copy" : "crosshair";
      return;
    }
    if (this.mode === "select") {
      this.canvas.style.cursor = "crosshair";
      return;
    }
    if (this.selectNearestPoint(this.mouseCanvas.x, this.mouseCanvas.y)) {
      this.canvas.style.cursor = "grab";
      return;
    }
    if (this.selectPolygon(this.mouseCanvas.x, this.mouseCanvas.y) !== -1) {
      this.canvas.style.cursor = "grab";
      return;
    }
    this.canvas.style.cursor = "all-scroll";
  },

  onKeyDown(event) {
    if (pendingKeybindAction) {
      event.preventDefault();
      const binding = keyEventToBinding(event);
      if (event.key === "Escape") {
        pendingKeybindAction = null;
        updateKeybindUI();
      } else if (binding) {
        const bindings = editorKeybinds[pendingKeybindAction.action];
        if (!bindings.includes(binding)) {
          bindings[pendingKeybindAction.index] = binding;
        }
        editorKeybinds[pendingKeybindAction.action] = bindings.filter(Boolean);
        saveEditorKeybinds();
        pendingKeybindAction = null;
        updateKeybindUI();
      }
      return;
    }
    if (!$("#keybind-modal")?.classList.contains("is-hidden")) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeKeybinds();
      }
      return;
    }
    if (event.target?.closest?.("input, textarea, select, [contenteditable='true']")) {
      return;
    }
    if (!$("#cases-panel")?.classList.contains("is-active")) {
      return;
    }
    if (keyMatchesBinding(event, editorKeybinds.toggleSnap)) {
      event.preventDefault();
      this.toggleSnapping();
    } else if (keyMatchesBinding(event, editorKeybinds.deleteSelection) || event.key === "Backspace" || event.key === "Delete") {
      event.preventDefault();
      this.deleteSelection();
    } else if (keyMatchesBinding(event, editorKeybinds.clearSelection)) {
      event.preventDefault();
      this.clearSelection();
    } else if (keyMatchesBinding(event, editorKeybinds.fitInstance)) {
      event.preventDefault();
      this.frameCurrentCase();
    } else if (keyMatchesBinding(event, editorKeybinds.toggleGrid)) {
      event.preventDefault();
      this.toggleLayer("grid");
    } else if (keyMatchesBinding(event, editorKeybinds.togglePath)) {
      event.preventDefault();
      this.toggleLayer("solution");
    } else if (keyMatchesBinding(event, editorKeybinds.toggleDecomposition)) {
      event.preventDefault();
      this.toggleLayer("decomposition");
    } else if (keyMatchesBinding(event, editorKeybinds.toggleLabels)) {
      event.preventDefault();
      this.toggleLayer("labels");
    } else if (keyMatchesBinding(event, editorKeybinds.closePolygon)) {
      event.preventDefault();
      this.closePolygon();
    } else if (event.key === "Escape") {
      if (this.expanded) {
        event.preventDefault();
        this.toggleExpanded(false);
        return;
      }
      this.activePolygon = null;
      this.selectedPoint = null;
      this.selectedPoints = [];
      this.selectionRect = null;
      this.setMode("move");
    } else if (event.key === "1") {
      this.setMode("move");
    } else if (event.key === "2") {
      this.setMode("polygon");
    } else if (event.key === "3") {
      this.setMode("select");
    }
  },

  finishSelection() {
    const selected = this.pointsInRect(this.selectionRect);
    const moved = Math.hypot(
      this.selectionRect.end.x - this.selectionRect.start.x,
      this.selectionRect.end.y - this.selectionRect.start.y,
    ) > 4;
    if (!moved && selected.length === 0) {
      this.selectedPoints = [];
      this.selectedPoint = null;
      this.selectionRect = null;
      this.selectionBase = [];
      this.draw();
      return;
    }
    this.selectedPoints = this.mergeSelections(this.selectionBase, selected);
    this.selectedPoint = this.selectedPoints[0] || null;
    this.selectionRect = null;
    this.selectionBase = [];
    this.animateSelection();
    this.draw();
  },

  closePolygon() {
    const current = this.currentCase();
    if (!current || this.activePolygon === null) {
      this.showCloseHint();
      return;
    }
    if ((current.polygons[this.activePolygon] || []).length < 3) {
      this.showCloseHint();
      return;
    }
    this.activePolygon = null;
    this.syncCloseButton();
    this.changed();
    this.canvas?.focus({ preventScroll: true });
  },

  deleteSelection() {
    const current = this.currentCase();
    const selected = this.selectedPoints;
    if (!current || selected.length === 0) {
      return;
    }
    const byPolygon = new Map();
    for (const point of selected) {
      if (point.kind !== "vertex") {
        continue;
      }
      const vertices = byPolygon.get(point.polygonIndex) || [];
      vertices.push(point.vertexIndex);
      byPolygon.set(point.polygonIndex, vertices);
    }
    [...byPolygon.entries()]
      .sort(([left], [right]) => right - left)
      .forEach(([polygonIndex, vertexIndices]) => {
        const polygon = current.polygons[polygonIndex];
        [...new Set(vertexIndices)].sort((left, right) => right - left).forEach((vertexIndex) => {
          polygon.splice(vertexIndex, 1);
        });
        if (polygon.length < 3) {
          current.polygons.splice(polygonIndex, 1);
        }
      });
    if (this.activePolygon !== null && this.activePolygon >= current.polygons.length) {
      this.activePolygon = null;
    }
    this.activePoint = null;
    this.selectedPoint = null;
    this.selectedPoints = [];
    this.changed();
  },

  clearSelection() {
    this.selectedPoint = null;
    this.selectedPoints = [];
    this.selectionBase = [];
    this.selectionRect = null;
    this.draw();
  },

  changed() {
    this.solutionRevision += 1;
    this.solutionStale = Boolean(this.solutionPath);
    this.updateLabelDirections(false);
    this.syncCloseButton();
    updateManualCaseListMetadata();
    this.draw();
    this.scheduleSolve();
    scheduleManualAutosave();
  },

  scheduleSolve() {
    const current = this.currentCase();
    if (!current) {
      return;
    }
    if (this.solutionStale) {
      this.setStatus("Updating solution...");
    }
    if (this.solutionTimer) {
      clearTimeout(this.solutionTimer);
      this.solutionTimer = null;
    }
    if (this.solutionFrame) {
      cancelAnimationFrame(this.solutionFrame);
    }
    this.solutionFrame = requestAnimationFrame(() => {
      this.solutionFrame = null;
      this.fetchSolution(cloneCaseData(this.currentCase()), this.solutionRevision);
    });
  },

  async fetchSolution(caseData, revision = this.solutionRevision) {
    if (!caseData) {
      return;
    }
    if (this.solutionAbort) {
      this.solutionAbort.abort();
    }
    this.solutionAbort = new AbortController();
    if (caseData.polygons.length === 0) {
      if (revision !== this.solutionRevision) {
        return;
      }
      this.solutionPath = [caseData.start, caseData.target];
      this.solutionStale = false;
      this.updateLabelDirections(true);
      this.setStatus("Solution: exact, 0 calls");
      this.draw();
      return;
    }
    this.setStatus(editorWasmModule ? "Solving..." : editorWasmFailed ? "WASM solver unavailable." : "Loading solver...");
    try {
      await loadEditorWasm();
      if (!editorWasmModule) {
        if (revision !== this.solutionRevision) {
          return;
        }
        this.solutionPath = null;
        this.solutionStale = false;
        this.updateLabelDirections(true);
        this.setStatus("WASM solver unavailable.");
        this.draw();
        return;
      }
      const geometry = await loadEditorGeometry();
      let wasmResult = null;
      if (caseData.polygons.every(polygonIsConvex)) {
        wasmResult = solveEditorWasm(caseData);
      } else if (geometry) {
        const pieceGroups = caseData.polygons.map((polygon) => (
          geometry.partition.convexPartition(polygon.map(([x, y]) => new geometry.vector.Vector2(x, y)))
            .filter((piece) => piece.length >= 3)
            .map((piece) => piece.map((point) => [point.x, point.y]))
        ));
        wasmResult = solveEditorWasmGroups(caseData, pieceGroups);
      } else {
        wasmResult = solveEditorWasmGroups(caseData, caseData.polygons.map(convexDecomposition));
      }
      if (wasmResult) {
        if (revision !== this.solutionRevision) {
          return;
        }
        this.solutionPath = wasmResult.path;
        this.solutionStale = false;
        this.updateLabelDirections(true);
        this.setStatus(`Solution: ${wasmResult.exact ? "exact" : "approximate"}, ${wasmResult.calls} calls, ${formatSeconds(wasmResult.seconds)} via WASM`);
        this.draw();
        return;
      }
      if (revision !== this.solutionRevision) {
        return;
      }
      this.solutionPath = null;
      this.solutionStale = false;
      this.updateLabelDirections(true);
      this.setStatus("WASM solver could not solve this case.");
      this.draw();
    } catch (error) {
      if (error.name !== "AbortError") {
        this.setStatus(error.message);
      }
    }
  },

  drawGrid() {
    const width = this.canvas.offsetWidth;
    const height = this.canvas.offsetHeight;
    const ctx = this.ctx;
    ctx.fillStyle = cssVar("--editor-bg") || "#121417";
    ctx.fillRect(0, 0, width, height);
    if (!this.layers.grid) {
      return;
    }
    const { gridSpacing, subGridCount } = this.gridMetrics();
    const left = (0 - this.offsetX) / this.scale;
    const right = (width - this.offsetX) / this.scale;
    const top = (this.offsetY - 0) / this.scale;
    const bottom = (this.offsetY - height) / this.scale;
    const drawWorldLine = (a, b, color) => {
      const start = this.worldToCanvas(a);
      const end = this.worldToCanvas(b);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
    };
    const firstX = Math.floor(left / gridSpacing) * gridSpacing;
    const firstY = Math.floor(bottom / gridSpacing) * gridSpacing;
    for (let x = firstX; x <= right + gridSpacing; x += gridSpacing) {
      drawWorldLine([x, bottom], [x, top], cssVar("--editor-grid-major") || "#515a67");
      for (let index = 0; index < subGridCount; index += 1) {
        const subX = x + (gridSpacing * (index + 1)) / (subGridCount + 1);
        drawWorldLine([subX, bottom], [subX, top], cssVar("--editor-grid-minor") || "#2a2f38");
      }
    }
    for (let y = firstY; y <= top + gridSpacing; y += gridSpacing) {
      drawWorldLine([left, y], [right, y], cssVar("--editor-grid-major") || "#515a67");
      for (let index = 0; index < subGridCount; index += 1) {
        const subY = y + (gridSpacing * (index + 1)) / (subGridCount + 1);
        drawWorldLine([left, subY], [right, subY], cssVar("--editor-grid-minor") || "#2a2f38");
      }
    }
    const origin = this.worldToCanvas([0, 0]);
    ctx.strokeStyle = cssVar("--editor-axis") || "#9aa3ad";
    ctx.beginPath();
    ctx.moveTo(0, origin.y);
    ctx.lineTo(width, origin.y);
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, height);
    ctx.stroke();
  },

  drawPoint(point, color, label = "", labelDirection = [1, -1], options = {}) {
    const ctx = this.ctx;
    const canvasPoint = this.worldToCanvas(point);
    const radius = options.radius ?? this.pointRadius(label ? "endpoint" : "vertex");
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(canvasPoint.x, canvasPoint.y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (label) {
      const length = Math.hypot(labelDirection[0], labelDirection[1]) || 1;
      const offset = Math.max(13, radius + 9);
      const offsetX = (labelDirection[0] / length) * offset;
      const offsetY = -(labelDirection[1] / length) * offset;
      ctx.fillStyle = "#f8fafc";
      ctx.font = "12px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, canvasPoint.x + offsetX, canvasPoint.y + offsetY);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }
  },

  drawSelectedRing(point) {
    const ctx = this.ctx;
    const canvasPoint = this.worldToCanvas(point);
    const radius = this.pointRadius("vertex") + 5;
    const rotation = ((performance.now() - this.selectionSpinStarted) / 700) * Math.PI * 2;
    ctx.save();
    ctx.translate(canvasPoint.x, canvasPoint.y);
    ctx.rotate(rotation);
    ctx.strokeStyle = "#f8fafc";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  },

  animateSelection() {
    if (this.selectionAnimationRunning) {
      return;
    }
    this.selectionAnimationRunning = true;
    const tick = () => {
      if (this.selectedPoints.length === 0) {
        this.selectionAnimationRunning = false;
        return;
      }
      this.draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  },

  drawDecomposition(polygon, color) {
    const pieces = convexDecomposition(polygon);
    if (pieces.length <= 1) {
      return;
    }
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.82;
    ctx.setLineDash([5, 5]);
    for (const piece of pieces) {
      ctx.beginPath();
      piece.forEach((point, pointIndex) => {
        const canvasPoint = this.worldToCanvas(point);
        if (pointIndex === 0) {
          ctx.moveTo(canvasPoint.x, canvasPoint.y);
        } else {
          ctx.lineTo(canvasPoint.x, canvasPoint.y);
        }
      });
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  },

  draw() {
    if (!this.canvas || !this.ctx) {
      return;
    }
    this.drawGrid();
    const current = this.currentCase();
    if (!current) {
      this.setStatus("Create or select an editable campaign.");
      return;
    }
    const ctx = this.ctx;
    if (this.layers.solution && this.solutionPath && this.solutionPath.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 4;
      ctx.globalAlpha = this.solutionStale ? 0.45 : 1;
      if (this.solutionStale) {
        ctx.setLineDash([9, 6]);
      }
      ctx.beginPath();
      this.solutionPath.forEach((point, index) => {
        const canvasPoint = this.worldToCanvas(point);
        if (index === 0) {
          ctx.moveTo(canvasPoint.x, canvasPoint.y);
        } else {
          ctx.lineTo(canvasPoint.x, canvasPoint.y);
        }
      });
      ctx.stroke();
      ctx.restore();
    }
    current.polygons.forEach((polygon, index) => {
      if (polygon.length === 0) {
        return;
      }
      const color = ["#38bdf8", "#a3e635", "#f97316", "#f472b6", "#c084fc"][index % 5];
      ctx.beginPath();
      polygon.forEach((point, pointIndex) => {
        const canvasPoint = this.worldToCanvas(point);
        if (pointIndex === 0) {
          ctx.moveTo(canvasPoint.x, canvasPoint.y);
        } else {
          ctx.lineTo(canvasPoint.x, canvasPoint.y);
        }
      });
      if (polygon.length >= 3 && index !== this.activePolygon) {
        ctx.closePath();
        ctx.fillStyle = `${color}33`;
        ctx.fill();
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      if (this.layers.decomposition && polygon.length >= 3) {
        this.drawDecomposition(polygon, color);
      }
      polygon.forEach((point) => this.drawPoint(point, color));
      if (index === this.activePolygon && polygon.length > 0) {
        const last = this.worldToCanvas(polygon.at(-1));
        const preview = this.worldToCanvas(this.snap(this.canvasToWorld(this.mouseCanvas.x, this.mouseCanvas.y)));
        ctx.strokeStyle = color;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(preview.x, preview.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = this.isClosingPolygon(this.mouseCanvas.x, this.mouseCanvas.y) ? "#facc15" : "#f8fafc";
        ctx.beginPath();
        ctx.arc(preview.x, preview.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    for (const selected of this.selectedPoints) {
      this.drawSelectedRing(selected.point);
      if (selected.kind === "vertex") {
        this.drawPoint(selected.point, "#f8fafc");
      }
    }
    if (this.layers.solution && this.solutionPath && this.solutionPath.length > 2) {
      this.solutionPath.slice(1, -1).forEach((point) => this.drawPoint(point, "#f97316"));
    }
    if (this.selectionRect) {
      const left = Math.min(this.selectionRect.start.x, this.selectionRect.end.x);
      const top = Math.min(this.selectionRect.start.y, this.selectionRect.end.y);
      const width = Math.abs(this.selectionRect.start.x - this.selectionRect.end.x);
      const height = Math.abs(this.selectionRect.start.y - this.selectionRect.end.y);
      ctx.fillStyle = "rgba(37, 99, 235, 0.18)";
      ctx.strokeStyle = "#93c5fd";
      ctx.setLineDash([6, 4]);
      ctx.fillRect(left, top, width, height);
      ctx.strokeRect(left, top, width, height);
      ctx.setLineDash([]);
    }
    this.drawPoint(current.start, "#22c55e", this.layers.labels ? "s" : "", this.labelDirections.start);
    this.drawPoint(current.target, "#ef4444", this.layers.labels ? "t" : "", this.labelDirections.target);
  },
};

function renderCanvasPlaceholder(root, text = "Loading instances...") {
  root.innerHTML = `<div class="missing-preview">${escapeHTML(text)}</div>`;
  root.classList.remove("is-hidden");
}

function createReadonlyInstanceViewer(canvas, caseData, options = {}) {
  const viewer = {
    canvas,
    ctx: canvas.getContext("2d"),
    caseData: cloneCaseData(caseData),
    scale: 70,
    minScale: 0.1,
    maxScale: 50000,
    offsetX: 0,
    offsetY: 0,
    activePolygon: null,
    selectedPoints: [],
    selectionRect: null,
    mouseCanvas: { x: 0, y: 0 },
    solutionPath: null,
    solutionStale: false,
    solutionRevision: 0,
    labelDirections: {
      start: [1, 0],
      target: [-1, 0],
    },
    layers: {
      grid: options.grid ?? true,
      solution: options.solution ?? true,
      decomposition: options.decomposition ?? true,
      labels: options.labels ?? true,
    },
    currentCase() {
      return this.caseData;
    },
    saveCamera() {},
    setStatus(message) {
      if (options.status) {
        options.status.textContent = message || "";
      }
    },
    resize: manualEditor.resize,
    caseBounds: manualEditor.caseBounds,
    updateZoomLimits: manualEditor.updateZoomLimits,
    frameCurrentCase: manualEditor.frameCurrentCase,
    pointRadius: manualEditor.pointRadius,
    worldToCanvas: manualEditor.worldToCanvas,
    canvasToWorld: manualEditor.canvasToWorld,
    gridMetrics: manualEditor.gridMetrics,
    drawGrid: manualEditor.drawGrid,
    drawPoint: manualEditor.drawPoint,
    drawDecomposition: manualEditor.drawDecomposition,
    updateLabelDirections: manualEditor.updateLabelDirections,
    draw: manualEditor.draw,
    zoomBy(factor) {
      const centerX = this.canvas.offsetWidth / 2;
      const centerY = this.canvas.offsetHeight / 2;
      const before = this.canvasToWorld(centerX, centerY);
      this.updateZoomLimits();
      this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
      const after = this.canvasToWorld(centerX, centerY);
      this.offsetX += (after[0] - before[0]) * this.scale;
      this.offsetY -= (after[1] - before[1]) * this.scale;
      this.draw();
    },
    async fetchSolution(caseData, revision = this.solutionRevision) {
      if (!caseData) {
        return;
      }
      if (caseData.polygons.length === 0) {
        if (revision !== this.solutionRevision) {
          return;
        }
        this.solutionPath = [caseData.start, caseData.target];
        this.solutionStale = false;
        this.updateLabelDirections(true);
        this.setStatus("Solution: exact, 0 calls");
        this.draw();
        return;
      }
      this.setStatus(editorWasmModule ? "Solving..." : editorWasmFailed ? "WASM solver unavailable." : "Loading solver...");
      try {
        await loadEditorWasm();
        if (!editorWasmModule) {
          if (revision !== this.solutionRevision) {
            return;
          }
          this.solutionPath = null;
          this.solutionStale = false;
          this.setStatus("WASM solver unavailable.");
          this.draw();
          return;
        }
        const geometry = await loadEditorGeometry();
        let wasmResult = null;
        if (caseData.polygons.every(polygonIsConvex)) {
          wasmResult = solveEditorWasm(caseData);
        } else if (geometry) {
          const pieceGroups = caseData.polygons.map((polygon) => (
            geometry.partition.convexPartition(polygon.map(([x, y]) => new geometry.vector.Vector2(x, y)))
              .filter((piece) => piece.length >= 3)
              .map((piece) => piece.map((point) => [point.x, point.y]))
          ));
          wasmResult = solveEditorWasmGroups(caseData, pieceGroups);
        } else {
          wasmResult = solveEditorWasmGroups(caseData, caseData.polygons.map(convexDecomposition));
        }
        if (revision !== this.solutionRevision) {
          return;
        }
        this.solutionPath = wasmResult?.path || null;
        this.solutionStale = false;
        this.updateLabelDirections(true);
        this.setStatus(wasmResult
          ? `Solution: ${wasmResult.exact ? "exact" : "approximate"}, ${wasmResult.calls} calls, ${formatSeconds(wasmResult.seconds)} via WASM`
          : "WASM solver could not solve this case.");
        this.draw();
      } catch (error) {
        this.setStatus(error.message);
      }
    },
  };

  viewer.resize();
  viewer.frameCurrentCase();
  new ResizeObserver(() => {
    viewer.resize();
    viewer.frameCurrentCase();
  }).observe(canvas);
  canvas.addEventListener("wheel", (event) => {
    if (!options.interactive) {
      return;
    }
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const before = viewer.canvasToWorld(event.clientX - rect.left, event.clientY - rect.top);
    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
    viewer.updateZoomLimits();
    viewer.scale = Math.max(viewer.minScale, Math.min(viewer.maxScale, viewer.scale * factor));
    const after = viewer.canvasToWorld(event.clientX - rect.left, event.clientY - rect.top);
    viewer.offsetX += (after[0] - before[0]) * viewer.scale;
    viewer.offsetY -= (after[1] - before[1]) * viewer.scale;
    viewer.draw();
  }, { passive: false });
  if (options.interactive) {
    let pan = null;
    canvas.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      pan = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!pan) {
        return;
      }
      viewer.offsetX += event.clientX - pan.x;
      viewer.offsetY += event.clientY - pan.y;
      pan = { x: event.clientX, y: event.clientY };
      viewer.draw();
    });
    const finishPan = () => {
      pan = null;
      canvas.style.cursor = "grab";
    };
    canvas.addEventListener("pointerup", finishPan);
    canvas.addEventListener("pointercancel", finishPan);
  }
  if (options.solve) {
    viewer.fetchSolution(viewer.caseData);
  }
  return viewer;
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
  renderManualCampaigns();
  if (state.manualCampaign && state.loadedManualCampaign !== state.manualCampaign) {
    loadManualCases(state.manualCampaign);
  }
}

function editableCampaigns() {
  return state.campaigns;
}

function renderManualCampaigns() {
  const grid = $("#manual-campaign-grid");
  if (!grid) {
    return;
  }
  grid.innerHTML = "";
  const campaigns = editableCampaigns();
  if (!campaigns.some((campaign) => campaign.name === state.manualCampaign)) {
    state.manualCampaign = campaigns[0]?.name || "";
  }
  if (campaigns.length === 0) {
    grid.innerHTML = '<div class="empty-choice">No editable campaigns yet.</div>';
    renderManualCases();
    return;
  }
  for (const campaign of campaigns) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "choice-card";
    button.dataset.value = campaign.name;
    button.setAttribute("role", "option");
    button.innerHTML = `
      <strong>${escapeHTML(campaign.name)}${campaign.type === "manual" ? "" : autoGeneratedSticker()}</strong>
      <span>${campaign.instance_progress.total || 0} instances</span>
      <small>${campaign.instance_progress.completed || 0} benchmarked</small>
    `;
    button.addEventListener("click", () => selectManualCampaign(campaign.name));
    grid.appendChild(button);
  }
  grid.querySelectorAll(".choice-card").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.value === state.manualCampaign);
  });
}

function renderManualCases() {
  const root = $("#manual-case-list");
  if (!root) {
    return;
  }
  root.innerHTML = "";
  if (!state.manualCampaign) {
    root.textContent = "Create or select a manual campaign.";
    state.manualCases = [];
    manualEditor.draw();
    return;
  }
  if (state.manualCases.length === 0) {
    root.textContent = "No instances yet.";
    manualEditor.draw();
    return;
  }
  state.manualCases.forEach((item, index) => {
    const active = index === state.manualCaseIndex;
    const row = document.createElement("div");
    row.className = "manual-case-row";
    row.classList.toggle("is-active", active);
    row.dataset.caseIndex = String(index);
    const selectArea = document.createElement("div");
    selectArea.className = "manual-case-select";
    selectArea.setAttribute("role", active ? "group" : "button");
    if (!active) {
      selectArea.tabIndex = 0;
    }
    let nameControl;
    if (state.manualRenamingIndex === index) {
      nameControl = document.createElement("input");
      nameControl.className = "instance-name-input";
      nameControl.value = item.name || `Instance ${instanceLabel(index)}`;
      const commit = () => finishRenameManualCase(index, nameControl.value);
      nameControl.addEventListener("click", (event) => event.stopPropagation());
      nameControl.addEventListener("blur", commit);
      nameControl.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (event.key === "Enter") {
          event.preventDefault();
          commit();
        } else if (event.key === "Escape") {
          event.preventDefault();
          state.manualRenamingIndex = null;
          renderManualCases();
        }
      });
      requestAnimationFrame(() => {
        nameControl.focus();
        nameControl.select();
      });
    } else {
      nameControl = document.createElement("button");
      nameControl.type = "button";
      nameControl.className = "instance-name-button";
      nameControl.textContent = item.name || `Instance ${instanceLabel(index)}`;
      nameControl.addEventListener("click", (event) => {
        event.stopPropagation();
        renameManualCase(index);
      });
    }
    const count = document.createElement("span");
    count.dataset.casePolygonCount = "";
    count.innerHTML = `${item.polygons.length} polygons${item.generated ? autoGeneratedSticker("This instance was generated automatically.") : ""}`;
    selectArea.append(nameControl, count);
    const select = () => {
      if (active) {
        return;
      }
      state.manualCaseIndex = index;
      manualEditor.activePolygon = null;
      manualEditor.solutionPath = null;
      renderManualCases();
      manualEditor.frameCurrentCase();
      manualEditor.scheduleSolve();
    };
    selectArea.addEventListener("click", select);
    selectArea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    const actions = document.createElement("div");
    actions.className = "manual-case-actions";
    const duplicateButton = document.createElement("button");
    duplicateButton.type = "button";
    duplicateButton.className = "secondary manual-case-action";
    duplicateButton.textContent = "⧉";
    duplicateButton.title = "Duplicate";
    duplicateButton.setAttribute("aria-label", `Duplicate instance ${instanceLabel(index)}`);
    duplicateButton.addEventListener("click", () => duplicateManualCase(index));
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "danger manual-case-action";
    deleteButton.textContent = "×";
    deleteButton.title = "Delete";
    deleteButton.setAttribute("aria-label", `Delete instance ${instanceLabel(index)}`);
    deleteButton.addEventListener("click", () => deleteManualCase(index));
    actions.append(duplicateButton, deleteButton);
    row.append(selectArea, actions);
    root.appendChild(row);
  });
  manualEditor.draw();
}

function updateManualCaseListMetadata() {
  const current = manualEditor.currentCase();
  const row = document.querySelector(`.manual-case-row[data-case-index="${state.manualCaseIndex}"]`);
  const label = row?.querySelector("[data-case-polygon-count]");
  if (label && current) {
    label.innerHTML = `${current.polygons.length} polygons${current.generated ? autoGeneratedSticker("This instance was generated automatically.") : ""}`;
  }
}

function renameManualCase(index) {
  const current = state.manualCases[index];
  if (!current) {
    return;
  }
  state.manualRenamingIndex = index;
  renderManualCases();
}

function finishRenameManualCase(index, value) {
  const current = state.manualCases[index];
  if (!current) {
    return;
  }
  state.manualRenamingIndex = null;
  current.name = value.trim();
  renderManualCases();
  scheduleManualAutosave({ immediate: true });
}

async function selectManualCampaign(name) {
  state.manualCampaign = name;
  state.loadedManualCampaign = "";
  state.manualCases = [];
  state.manualCaseIndex = 0;
  renderManualCampaigns();
  await loadManualCases(name);
}

async function loadManualCases(name) {
  if (!name) {
    renderManualCases();
    return;
  }
  try {
    const data = await requestJSON(`/api/campaigns/${encodeURIComponent(name)}/cases`);
    state.manualCases = data.cases.map(cloneCaseData);
    state.campaignCaseMetadata.set(name, state.manualCases.map(cloneCaseData));
    state.loadedManualCampaign = name;
    state.manualCaseIndex = Math.min(state.manualCaseIndex, Math.max(0, state.manualCases.length - 1));
    clearTimeout(state.manualAutosaveTimer);
    renderManualCases();
    manualEditor.frameCurrentCase();
    manualEditor.scheduleSolve();
  } catch (error) {
    manualEditor.setStatus(error.message);
  }
}

async function loadCampaignCaseMetadata(campaignName) {
  if (state.campaignCaseMetadata.has(campaignName)) {
    return state.campaignCaseMetadata.get(campaignName);
  }
  try {
    const data = await requestJSON(`/api/campaigns/${encodeURIComponent(campaignName)}/cases`);
    const cases = data.cases.map(cloneCaseData);
    state.campaignCaseMetadata.set(campaignName, cases);
    return cases;
  } catch {
    state.campaignCaseMetadata.set(campaignName, []);
    return [];
  }
}

async function editInstance(campaign, index) {
  closeCampaignModal();
  await selectManualCampaign(campaign.name);
  state.manualCaseIndex = Math.min(index, Math.max(0, state.manualCases.length - 1));
  renderManualCases();
  switchPanel("cases-panel");
  manualEditor.frameCurrentCase();
  manualEditor.scheduleSolve();
}

async function createManualCampaign(event) {
  event.preventDefault();
  const values = formData(event.currentTarget);
  if (campaignExists(values.name) && !(await askConfirmation(`Overwrite campaign "${values.name}"? This removes its inputs, previews, results, and metadata.`, "Overwrite"))) {
    return;
  }
  try {
    const data = await requestJSON("/api/campaigns/manual", {
      method: "POST",
      body: JSON.stringify({ name: values.name, overwrite: campaignExists(values.name) }),
    });
    await refresh();
    await selectManualCampaign(data.campaign.name);
    switchPanel("cases-panel");
  } catch (error) {
    manualEditor.setStatus(error.message);
  }
}

async function saveManualCasesNow() {
  if (!state.manualCampaign) {
    manualEditor.setSaveStatus("Select a campaign first.");
    return;
  }
  if (state.manualAutosaving) {
    state.manualAutosaveQueued = true;
    return;
  }
  const campaignName = state.manualCampaign;
  state.manualAutosaving = true;
  manualEditor.setSaveStatus("Saving...");
  try {
    const data = await requestJSON(`/api/campaigns/${encodeURIComponent(campaignName)}/cases`, {
      method: "PUT",
      body: JSON.stringify({ cases: state.manualCases.map(casePayload) }),
    });
    if (state.manualCampaign === campaignName) {
      const campaignIndex = state.campaigns.findIndex((campaign) => campaign.name === data.campaign.name);
      if (campaignIndex === -1) {
        state.campaigns.push(data.campaign);
      } else {
        state.campaigns[campaignIndex] = data.campaign;
      }
      state.loadedManualCampaign = campaignName;
      state.campaignCaseMetadata.set(campaignName, state.manualCases.map(cloneCaseData));
      renderManualCampaigns();
      manualEditor.setSaveStatus("Autosaved.");
    }
  } catch (error) {
    manualEditor.setSaveStatus(error.message);
  } finally {
    state.manualAutosaving = false;
    if (state.manualAutosaveQueued) {
      state.manualAutosaveQueued = false;
      scheduleManualAutosave({ immediate: true });
    }
  }
}

function scheduleManualAutosave(options = {}) {
  if (!state.manualCampaign) {
    return;
  }
  clearTimeout(state.manualAutosaveTimer);
  if (options.immediate) {
    saveManualCasesNow();
    return;
  }
  state.manualAutosaveTimer = setTimeout(saveManualCasesNow, 1200);
}

function newManualCase() {
  if (!state.manualCampaign) {
    manualEditor.setSaveStatus("Create or select a campaign first.");
    return;
  }
  state.manualCases.push(emptyCaseData());
  state.manualCaseIndex = state.manualCases.length - 1;
  manualEditor.activePolygon = null;
  manualEditor.solutionPath = null;
  renderManualCases();
  manualEditor.frameCurrentCase();
  manualEditor.changed();
  scheduleManualAutosave({ immediate: true });
}

function duplicateManualCase(index = state.manualCaseIndex) {
  const current = state.manualCases[index] || manualEditor.currentCase();
  if (!current) {
    newManualCase();
    return;
  }
  state.manualCases.splice(index + 1, 0, cloneCaseData(current));
  state.manualCaseIndex = index + 1;
  renderManualCases();
  manualEditor.frameCurrentCase();
  manualEditor.changed();
  scheduleManualAutosave({ immediate: true });
}

async function deleteManualCase(index = state.manualCaseIndex) {
  if (!state.manualCampaign || state.manualCases.length === 0) {
    return;
  }
  if (!(await askConfirmation(`Delete instance ${instanceLabel(index)} from "${state.manualCampaign}"?`, "Delete"))) {
    return;
  }
  state.manualCases.splice(index, 1);
  state.manualCaseIndex = Math.min(state.manualCaseIndex, Math.max(0, state.manualCases.length - 1));
  manualEditor.activePolygon = null;
  manualEditor.solutionPath = null;
  renderManualCases();
  manualEditor.changed();
  scheduleManualAutosave({ immediate: true });
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

function instanceModalTitle(campaign, index) {
  const total = campaign.instance_progress?.total || campaign.generation?.instances || campaign.generation?.instances_per_file || "?";
  const name = instanceDisplayName(campaign, index);
  return `
    <span class="modal-title-main">${escapeHTML(campaign.name)}</span>
    <span class="modal-title-sub">${instanceLabel(index)}/${escapeHTML(total)}: <button class="instance-name-button modal-title-rename" type="button" data-modal-rename-trigger>${escapeHTML(name)}</button></span>
  `;
}

function setupModalTitleRename(campaign, index, afterRename) {
  const title = $("#modal-title");
  const trigger = title.querySelector("[data-modal-rename-trigger]");
  if (!trigger) {
    return;
  }
  trigger.addEventListener("click", () => {
    const input = document.createElement("input");
    input.className = "instance-name-input modal-title-input";
    input.value = instanceDisplayName(campaign, index);
    trigger.replaceWith(input);
    input.focus();
    input.select();
    let committed = false;
    const commit = async () => {
      if (committed) {
        return;
      }
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
      if (event.key === "Enter") {
        event.preventDefault();
        commit();
      } else if (event.key === "Escape") {
        event.preventDefault();
        committed = true;
        title.innerHTML = instanceModalTitle(campaign, index);
        setupModalTitleRename(campaign, index, afterRename);
      }
    });
  });
}

function readonlyInstanceDetail(title) {
  return `
    <figure class="instance-detail readonly-instance-detail">
      <div class="case-toolbar readonly-toolbar">
        <div class="readonly-toolbar-main">
          <button class="secondary tap-zoom-button" type="button" data-readonly-zoom-out>-</button>
          <button class="secondary tap-zoom-button" type="button" data-readonly-zoom-in>+</button>
          <button class="secondary" type="button" data-readonly-fit>Fit</button>
          <div class="editor-layer-toggles" aria-label="Viewer layers">
            <button class="secondary is-active" data-readonly-layer="grid" type="button" aria-pressed="true">Grid</button>
            <button class="secondary is-active" data-readonly-layer="solution" type="button" aria-pressed="true">Path</button>
            <button class="secondary is-active" data-readonly-layer="decomposition" type="button" aria-pressed="true">Decomp</button>
            <button class="secondary is-active" data-readonly-layer="labels" type="button" aria-pressed="true">Labels</button>
          </div>
        </div>
        <button class="secondary readonly-edit-button" type="button" data-edit-instance>Edit Instance</button>
      </div>
      <canvas class="readonly-instance-canvas" width="960" height="620" aria-label="${escapeHTML(title)}"></canvas>
      <div class="editor-status-row"><span data-readonly-status></span></div>
    </figure>
  `;
}

function setupReadonlyInstanceDetail(root, caseData) {
  const canvas = root.querySelector(".readonly-instance-canvas");
  if (!canvas) {
    return null;
  }
  const viewer = createReadonlyInstanceViewer(canvas, caseData, {
    interactive: true,
    solve: true,
    status: root.querySelector("[data-readonly-status]"),
  });
  const zoomOut = root.querySelector("[data-readonly-zoom-out]");
  const zoomIn = root.querySelector("[data-readonly-zoom-in]");
  bindTapZoom(zoomOut, () => viewer.zoomBy(1 / 1.2));
  bindTapZoom(zoomIn, () => viewer.zoomBy(1.2));
  root.querySelector("[data-readonly-fit]")?.addEventListener("click", () => viewer.frameCurrentCase());
  root.querySelectorAll("[data-readonly-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      const layer = button.dataset.readonlyLayer;
      viewer.layers[layer] = !viewer.layers[layer];
      button.classList.toggle("is-active", viewer.layers[layer]);
      button.setAttribute("aria-pressed", viewer.layers[layer] ? "true" : "false");
      viewer.draw();
    });
  });
  return viewer;
}

async function renameCampaignInstance(campaign, index, value) {
  const cases = await loadCampaignCaseMetadata(campaign.name);
  if (!cases[index]) {
    throw new Error("Case does not exist.");
  }
  cases[index].name = value.trim();
  const data = await requestJSON(`/api/campaigns/${encodeURIComponent(campaign.name)}/cases`, {
    method: "PUT",
    body: JSON.stringify({ cases: cases.map(casePayload) }),
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
          <button class="secondary tap-zoom-button" type="button" data-zoom-out>-</button>
          <button class="secondary" type="button" data-zoom-reset>Fit</button>
          <button class="secondary tap-zoom-button" type="button" data-zoom-in>+</button>
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
  const restyleSvg = () => {
    const background = svg.querySelector("rect");
    if (background) {
      background.setAttribute("fill", cssVar("--editor-bg") || "#121417");
    }
    svg.querySelectorAll("text").forEach((text) => {
      text.setAttribute("fill", cssVar("--text") || "#f8fafc");
    });
    const path = svg.querySelector("polyline");
    if (path) {
      path.setAttribute("stroke", cssVar("--accent-strong") || "#facc15");
    }
  };
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
    restyleSvg();
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

  const zoomIn = root.querySelector("[data-zoom-in]");
  const zoomOut = root.querySelector("[data-zoom-out]");
  bindTapZoom(zoomIn, () => zoomAt(state.scale * 1.25));
  bindTapZoom(zoomOut, () => zoomAt(state.scale / 1.25));
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

function instancePreviewButton(campaign, index, className = "instance-thumb") {
  const button = document.createElement("button");
  button.className = className;
  button.type = "button";
  button.innerHTML = `
    <img src="${instancePreviewUrl(campaign, index)}" alt="${escapeHTML(instanceDisplayName(campaign, index))} preview">
    <span>${instanceLabel(index)}</span>
  `;
  button.addEventListener("click", () => openInstanceModal(campaign, index));
  return button;
}

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
        <span>${escapeHTML(title)}</span>
      </button>
    </figcaption>
    <div class="fold-content"></div>
  `;
  const button = panel.querySelector(".fold-toggle");
  button.addEventListener("click", () => {
    const isFolded = panel.classList.toggle("is-folded");
    button.setAttribute("aria-expanded", isFolded ? "false" : "true");
  });
  return panel;
}

async function populateInstancePreviewPanels(root, campaign) {
  const previewCount = campaign.instance_previews?.length || 0;
  if (previewCount === 0) {
    root.classList.toggle("is-hidden", root.children.length === 0);
    return;
  }
  root.innerHTML = "";

  const selected = makeFoldablePanel("figure", "preview-panel preview-selected", "Selected Instance");
  selected.querySelector(".fold-content").appendChild(instancePreviewButton(campaign, 0, "selected-instance-button"));
  root.appendChild(selected);

  const four = makeFoldablePanel("figure", "preview-panel preview-four", "Four Instances");
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
  sampleInstanceIndices(previewCount, 20).forEach((index) => {
    instanceGrid.appendChild(instancePreviewButton(campaign, index));
  });
  panel.querySelector(".fold-content").appendChild(instanceGrid);
  root.appendChild(panel);
  root.classList.remove("is-hidden");
}

function renderPreviewPanels(root, campaign) {
  root.innerHTML = "";
  renderCanvasPlaceholder(root);
  populateInstancePreviewPanels(root, campaign).catch(() => {
    root.innerHTML = "";
    root.classList.add("is-hidden");
  });
}

function renderBenchmarkedInstanceSection(root, campaign, instances) {
  if (!campaign || instances.length === 0) {
    root.innerHTML = "";
    root.classList.add("is-hidden");
    return;
  }
  if (!state.campaignCaseMetadata.has(campaign.name)) {
    loadCampaignCaseMetadata(campaign.name).then(() => renderBenchmarkedInstanceSection(root, campaign, instances));
  }
  root.innerHTML = `
    <header class="section-subheader foldable-section-header">
      <div>
        <h3>
          <button class="fold-toggle" type="button" aria-expanded="true">
            <span class="fold-caret" aria-hidden="true"></span>
            <span>Benchmarked Instances</span>
          </button>
        </h3>
        <p>Completed rows with available previews and benchmark metrics.</p>
      </div>
      <div class="benchmarked-sort" aria-label="Sort benchmarked instances">
        <button class="segment ${state.benchmarkedSort === "case" ? "is-active" : ""}" data-benchmarked-sort="case" type="button" aria-pressed="${state.benchmarkedSort === "case" ? "true" : "false"}">Case</button>
        <button class="segment ${state.benchmarkedSort === "time" ? "is-active" : ""}" data-benchmarked-sort="time" type="button" aria-pressed="${state.benchmarkedSort === "time" ? "true" : "false"}">Solve time</button>
        <button class="segment ${state.benchmarkedSort === "calls" ? "is-active" : ""}" data-benchmarked-sort="calls" type="button" aria-pressed="${state.benchmarkedSort === "calls" ? "true" : "false"}">Convex calls</button>
      </div>
    </header>
    <div class="fold-content">
      <div class="benchmarked-grid"></div>
    </div>
  `;
  const grid = root.querySelector(".benchmarked-grid");
  const foldButton = root.querySelector(".fold-toggle");
  foldButton.addEventListener("click", () => {
    const isFolded = root.classList.toggle("is-folded");
    foldButton.setAttribute("aria-expanded", isFolded ? "false" : "true");
  });
  root.querySelectorAll("[data-benchmarked-sort]").forEach((button) => button.addEventListener("click", (event) => {
    state.benchmarkedSort = event.currentTarget.dataset.benchmarkedSort;
    renderBenchmarkedInstanceSection(root, campaign, instances);
  }));
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
    const preview = item.preview
      ? `<img src="${instancePreviewUrl(campaign, item.case_index)}" alt="Benchmarked instance ${instanceLabel(item.case_index)}">`
      : item.solution_available
        ? `<img src="${solutionPreviewUrl(campaign, item)}" alt="Solved instance ${instanceLabel(item.case_index)} with path and decomposition">`
        : '<div class="missing-preview">No preview</div>';
    button.innerHTML = `
      ${preview}
      <div class="benchmarked-meta">
        <strong>${escapeHTML(instanceTitle(campaign, item.case_index))}</strong>
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

function runningJobs() {
  return state.recentJobs.filter((job) => job.status === "running" || job.status === "stopping");
}

function jobPanel(job) {
  return job.kind === "comparison" ? "comparison-panel" : "benchmark-panel";
}

function jobKindLabel(job) {
  return job.kind === "comparison" ? "Comparison" : "Benchmark";
}

function dismissFinishedJobForPanel(panelId) {
  if (state.finishedDockJob && jobPanel(state.finishedDockJob) === panelId) {
    const dock = $("#job-dock");
    dock?.classList.add("is-dismissing");
    setTimeout(() => {
      state.finishedDockJob = null;
      renderJobDock();
    }, 180);
  }
}

function jobProgressLabel(job) {
  if (job.kind === "comparison") {
    const solverTotal = job.solver_progress_total || 0;
    const solverCompleted = job.solver_progress_completed || 0;
    const instanceTotal = job.progress_total || 0;
    const instanceCompleted = job.progress_completed || 0;
    const solverText = solverTotal ? `${solverCompleted}/${solverTotal} solvers` : "Comparing solvers";
    const instanceText = instanceTotal ? `, ${instanceCompleted}/${instanceTotal} instances` : "";
    return `${solverText}${instanceText}`;
  }
  if (job.progress_total) {
    return `${job.progress_completed || 0}/${job.progress_total} instances`;
  }
  return "Compiling";
}

function renderJobDock(previousJobs = []) {
  const dock = $("#job-dock");
  if (!dock) {
    return;
  }
  const previousActive = new Map(previousJobs
    .filter((job) => job.status === "running" || job.status === "stopping")
    .map((job) => [job.id, job]));
  for (const job of state.recentJobs) {
    if (previousActive.has(job.id) && job.status !== "running" && job.status !== "stopping") {
      state.finishedDockJob = job;
    }
  }
  const activeJobs = runningJobs();
  const visibleJobs = activeJobs.length > 0 ? activeJobs : state.finishedDockJob ? [state.finishedDockJob] : [];
  dock.classList.toggle("is-hidden", visibleJobs.length === 0);
  dock.classList.remove("is-dismissing");
  const signature = visibleJobs.map((job) => `${job.id}:${job.status}`).join(",");
  if (dock.dataset.signature === signature) {
    visibleJobs.forEach((job) => updateJobDockItem(dock.querySelector(`[data-job-id="${CSS.escape(job.id)}"]`), job));
    return;
  }
  dock.dataset.signature = signature;
  dock.innerHTML = visibleJobs.map((job) => {
    const active = job.status === "running" || job.status === "stopping";
    return `
    <div class="job-dock-item ${active ? "is-active" : "is-complete"}" data-job-panel="${jobPanel(job)}" data-job-id="${escapeHTML(job.id)}">
      <button class="job-dock-main" type="button" data-job-open>
        <span data-job-dock-status>${jobKindLabel(job)} ${active ? "running..." : "done"}</span>
        <strong data-job-dock-campaign>${escapeHTML(job.campaign || "-")}</strong>
        <small data-job-dock-progress>${escapeHTML(jobProgressLabel(job))} | ${formatElapsed((job.finished_at || Date.now() / 1000) - (job.started_at || Date.now() / 1000))}</small>
      </button>
      ${active ? '<button class="job-dock-stop" type="button" data-job-stop aria-label="Stop job">×</button>' : '<span class="job-dock-check" aria-hidden="true">✓</span>'}
    </div>
  `;
  }).join("");
  dock.querySelectorAll("[data-job-open]").forEach((button) => {
    button.addEventListener("click", () => {
      if (dock.dataset.suppressClick === "true") {
        return;
      }
      const item = button.closest(".job-dock-item");
      switchPanel(item.dataset.jobPanel);
      if (item.classList.contains("is-complete")) {
        item.classList.add("is-dismissing");
        setTimeout(() => {
          state.finishedDockJob = null;
          renderJobDock();
        }, 180);
      }
    });
  });
  dock.querySelectorAll("[data-job-stop]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const item = button.closest(".job-dock-item");
      button.disabled = true;
      button.textContent = "×";
      await cancelJobId(item.dataset.jobId);
    });
  });
}

function updateJobDockItem(item, job) {
  if (!item) {
    return;
  }
  const active = job.status === "running" || job.status === "stopping";
  item.dataset.jobPanel = jobPanel(job);
  item.classList.toggle("is-active", active);
  item.classList.toggle("is-complete", !active);
  item.querySelector("[data-job-dock-status]").textContent = `${jobKindLabel(job)} ${active ? "running..." : "done"}`;
  item.querySelector("[data-job-dock-campaign]").textContent = job.campaign || "-";
  item.querySelector("[data-job-dock-progress]").textContent = `${jobProgressLabel(job)} | ${formatElapsed((job.finished_at || Date.now() / 1000) - (job.started_at || Date.now() / 1000))}`;
}

function setupJobDockDrag() {
  const dock = $("#job-dock");
  if (!dock) {
    return;
  }
  let drag = null;
  dock.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button:not(.job-dock-main)")) {
      return;
    }
    const rect = dock.getBoundingClientRect();
    drag = {
      pointerId: event.pointerId,
      dx: event.clientX - rect.left,
      dy: event.clientY - rect.top,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
    dock.setPointerCapture(event.pointerId);
  });
  dock.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4) {
      drag.moved = true;
      dock.dataset.dragging = "true";
    }
    const left = Math.max(8, Math.min(window.innerWidth - dock.offsetWidth - 8, event.clientX - drag.dx));
    const top = Math.max(8, Math.min(window.innerHeight - dock.offsetHeight - 8, event.clientY - drag.dy));
    dock.style.left = `${left}px`;
    dock.style.top = `${top}px`;
    dock.style.right = "auto";
    dock.style.bottom = "auto";
  });
  const finish = (event) => {
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    if (drag.moved) {
      dock.dataset.suppressClick = "true";
      setTimeout(() => {
        delete dock.dataset.suppressClick;
      }, 120);
    }
    delete dock.dataset.dragging;
    drag = null;
    dock.releasePointerCapture?.(event.pointerId);
  };
  dock.addEventListener("pointerup", finish);
  dock.addEventListener("pointercancel", finish);
}

async function cancelJobId(jobId) {
  if (!jobId) {
    return;
  }
  try {
    await requestJSON(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  } catch (error) {
    setOutput($("#run-output"), error.message);
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
  const phase = progress.phase || (running ? "benchmark" : "completed");
  const compiling = running && phase === "compile";
  const percent = Math.round(progress.ratio * 100);
  const visiblePercent = compiling ? 100 : running && percent === 0 ? 8 : percent;
  const title = compiling ? "Compiling benchmark" : running ? "Running tests" : "Benchmark progress";
  const label = compiling ? "Build in progress" : progress.label;
  root.innerHTML = `
    <div class="run-progress-header">
      <strong>${title}</strong>
      <span>${label}</span>
    </div>
    <div class="run-progress-meta">
      <span>${compiling ? "Compile wall clock" : running ? "Test wall clock" : "Wall clock"}</span>
      <strong>${formatElapsed(progress.elapsed_seconds || 0)}</strong>
    </div>
    <div class="run-progress-line">
      <div class="run-progress-track ${running ? "is-running" : ""} ${compiling ? "is-compiling" : ""}">
        <div class="run-progress-fill" style="width: ${visiblePercent}%"></div>
      </div>
      <strong class="run-progress-percent">${compiling ? "build" : `${percent}%`}</strong>
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
  const closeButton = modal.querySelector(".modal-x-button");
  const generation = campaign.generation || {};
  const progress = runProgress(campaign);
  state.instanceModalReturn = null;
  closeButton?.removeAttribute("data-modal-back-instance");
  if (closeButton) {
    closeButton.classList.remove("modal-back-button");
    closeButton.innerHTML = "×";
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

function setInstanceModalBackButton(modal) {
  const closeButton = modal.querySelector(".modal-x-button");
  if (!closeButton) {
    return;
  }
  closeButton.setAttribute("data-modal-back-instance", "true");
  closeButton.classList.add("modal-back-button");
  closeButton.innerHTML = `
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M19 12H5M11 6l-6 6 6 6" />
    </svg>
  `;
  closeButton.setAttribute("aria-label", "Back to campaign details");
}

async function openInstanceModal(campaign, index) {
  const modal = $("#campaign-modal");
  state.instanceModalReturn = { campaign };
  setInstanceModalBackButton(modal);
  const cases = await loadCampaignCaseMetadata(campaign.name);
  const caseData = cases[index];
  const body = $("#modal-body");
  const title = instanceTitle(campaign, index);
  $("#modal-title").innerHTML = instanceModalTitle(campaign, index);
  body.innerHTML = `
    ${caseData ? readonlyInstanceDetail(`${title} detail`) : '<div class="missing-preview detail-missing">No case data available.</div>'}
  `;
  body.querySelector("[data-edit-instance]")?.addEventListener("click", () => editInstance(campaign, index));
  if (caseData) {
    setupReadonlyInstanceDetail(body, caseData);
  }
  setupModalTitleRename(campaign, index, () => openInstanceModal(campaign, index));
  modal.classList.remove("is-hidden");
}

async function openBenchmarkedInstanceModal(campaign, item) {
  const modal = $("#campaign-modal");
  state.instanceModalReturn = { campaign, panel: "benchmark-panel" };
  setInstanceModalBackButton(modal);
  const cases = await loadCampaignCaseMetadata(campaign.name);
  const caseData = cases[item.case_index];
  const body = $("#modal-body");
  const title = instanceTitle(campaign, item.case_index);
  $("#modal-title").innerHTML = instanceModalTitle(campaign, item.case_index);
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
    ${caseData ? readonlyInstanceDetail(`${title} detail`) : '<div class="missing-preview detail-missing">No case data available.</div>'}
  `;
  body.querySelector("[data-edit-instance]")?.addEventListener("click", () => editInstance(campaign, item.case_index));
  if (caseData) {
    setupReadonlyInstanceDetail(body, caseData);
  }
  setupModalTitleRename(campaign, item.case_index, () => openBenchmarkedInstanceModal(campaign, item));
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
  const previousJobs = state.recentJobs;
  const [campaignData, resultData, jobData] = await Promise.all([
    requestJSON("/api/campaigns"),
    requestJSON("/api/results"),
    requestJSON("/api/jobs"),
  ]);
  state.campaigns = campaignData.campaigns;
  state.resultFiles = resultData.files;
  state.recentJobs = jobData.jobs;
  renderJobDock(previousJobs);
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
  button.classList.toggle("is-hidden", selector === "#stop-run-button" || !jobId);
  button.textContent = "Stop";
  if (selector === "#stop-run-button") {
    const runButton = $("#run-submit-button");
    if (runButton) {
      runButton.dataset.job = jobId || "";
      runButton.disabled = false;
      runButton.textContent = jobId ? "Stop" : "Run";
      runButton.classList.toggle("stop-button", Boolean(jobId));
    }
  }
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
    await cancelJobId(jobId);
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
    const previousJobs = state.recentJobs;
    state.recentJobs = [job, ...state.recentJobs.filter((item) => item.id !== job.id)];
    renderJobDock(previousJobs);
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
        phase: "benchmark",
      }
      : {
        completed: 0,
        total: 0,
        ratio: 0,
        label: "Compiling",
        elapsed_seconds: job.elapsed_seconds,
        counts: campaign?.run_index?.counts || {},
        phase: "compile",
      };
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
      await refresh();
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
    const previousJobs = state.recentJobs;
    state.recentJobs = [job, ...state.recentJobs.filter((item) => item.id !== job.id)];
    renderJobDock(previousJobs);
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
      await refresh();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function runCampaign(event) {
  event.preventDefault();
  if (state.currentRunJob) {
    const jobId = $("#run-submit-button")?.dataset.job || $("#stop-run-button")?.dataset.job;
    if (jobId && jobId !== "pending") {
      $("#run-submit-button").disabled = true;
      $("#run-submit-button").textContent = "Stopping...";
      await cancelJobId(jobId);
    }
    switchPanel("benchmark-panel");
    return;
  }
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
    completed: 0,
    total: 0,
    ratio: 0,
    label: "Compiling",
    elapsed_seconds: 0,
    counts: activeCampaign.run_index?.counts || {},
    phase: "compile",
  } : null);
  setOutput(output, "Starting run...");
  renderRunSummary();
  try {
    state.currentRunJob = "pending";
    const runButton = $("#run-submit-button");
    if (runButton) {
      runButton.disabled = true;
      runButton.textContent = "Starting...";
    }
    const data = await requestJSON("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJob = data.job;
    state.currentRunJob = data.job;
    setStopButton("#stop-run-button", data.job);
    await pollJob(data.job);
  } catch (error) {
    state.currentRunJob = null;
    setStopButton("#stop-run-button", null);
    setOutput(output, error.message);
  }
}

async function runComparison(event) {
  event.preventDefault();
  if (state.currentComparisonJob) {
    switchPanel("comparison-panel");
    return;
  }
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
    state.currentComparisonJob = "pending";
    const data = await requestJSON("/api/comparisons", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJob = data.job;
    state.currentComparisonJob = data.job;
    await pollComparisonJob(data.job);
  } catch (error) {
    state.currentComparisonJob = null;
    setOutput(output, error.message);
    setStopButton("#stop-compare-button", null);
  }
}

document.addEventListener("pointerover", (event) => {
  const sticker = event.target?.closest?.(".auto-sticker[data-tooltip]");
  if (sticker) {
    showFloatingTooltip(sticker);
  }
});

document.addEventListener("pointerout", (event) => {
  if (event.target?.closest?.(".auto-sticker[data-tooltip]")) {
    hideFloatingTooltip();
  }
});

document.addEventListener("focusin", (event) => {
  const sticker = event.target?.closest?.(".auto-sticker[data-tooltip]");
  if (sticker) {
    showFloatingTooltip(sticker);
  }
});

document.addEventListener("focusout", (event) => {
  if (event.target?.closest?.(".auto-sticker[data-tooltip]")) {
    hideFloatingTooltip();
  }
});

document.addEventListener("click", (event) => {
  const sticker = event.target?.closest?.(".auto-sticker[data-tooltip]");
  if (sticker) {
    event.stopPropagation();
    showFloatingTooltip(sticker);
    clearTimeout(sticker._tooltipTimer);
    sticker._tooltipTimer = setTimeout(hideFloatingTooltip, 2200);
  }
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchPanel(tab.dataset.panel));
});

document.querySelectorAll("[data-close-modal]").forEach((element) => {
  element.addEventListener("click", (event) => {
    if (event.currentTarget.dataset.modalBackInstance === "true") {
      returnToCampaignModal();
      return;
    }
    closeCampaignModal();
  });
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
$("#theme-toggle").addEventListener("click", toggleTheme);
$("#import-canonical-button").addEventListener("click", importCanonicalSuite);
$("#import-german-button").addEventListener("click", importGermanInstances);
$("#scan-osm-files").addEventListener("click", scanOsmFiles);
$("#create-form").addEventListener("submit", createCampaign);
$("#manual-campaign-form").addEventListener("submit", createManualCampaign);
$("#run-form").addEventListener("submit", runCampaign);
$("#compare-form").addEventListener("submit", runComparison);
$("#copy-run-command").addEventListener("click", (event) => copyText(runCommandFromForm(), event.currentTarget));
$("#copy-compare-command").addEventListener("click", (event) => copyText(compareCommandFromForm(), event.currentTarget));
$("#stop-run-button").addEventListener("click", () => cancelJob("#stop-run-button", "#run-output"));
$("#stop-compare-button").addEventListener("click", () => cancelJob("#stop-compare-button", "#compare-output"));
$("#create-name").addEventListener("input", updateCampaignNameIndicator);
document.querySelectorAll("[data-manual-mode] .segment").forEach((button) => {
  button.addEventListener("click", () => manualEditor.setMode(button.dataset.mode));
});
$("#new-manual-case").addEventListener("click", newManualCase);
$("#close-manual-polygon").addEventListener("click", () => manualEditor.closePolygon());
$("#delete-manual-selection").addEventListener("click", () => manualEditor.deleteSelection());
$("#clear-manual-selection").addEventListener("click", () => manualEditor.clearSelection());
$("#toggle-manual-snapping").addEventListener("click", () => manualEditor.toggleSnapping());
bindTapZoom($("#manual-zoom-out"), () => manualEditor.zoomBy(1 / 1.2));
bindTapZoom($("#manual-zoom-in"), () => manualEditor.zoomBy(1.2));
$("#manual-fit-instance").addEventListener("click", () => manualEditor.frameCurrentCase());
document.querySelectorAll("[data-editor-layer]").forEach((button) => {
  button.addEventListener("click", () => manualEditor.toggleLayer(button.dataset.editorLayer));
});
$("#manual-editor-expand").addEventListener("click", () => manualEditor.toggleExpanded());
$("#manual-editor-close").addEventListener("click", () => manualEditor.toggleExpanded(false));
$("#manual-keybinds-button").addEventListener("click", openKeybinds);
document.querySelectorAll("[data-close-keybinds]").forEach((button) => {
  button.addEventListener("click", closeKeybinds);
});
setupJobDockDrag();
setupFilterInput("#campaign-filter", "campaignFilter", renderCampaigns);
setupFilterInput("#result-filter", "resultFilter", () => renderResults(state.resultFiles));
updateKeybindUI();
applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || "light");
manualEditor.init();
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
