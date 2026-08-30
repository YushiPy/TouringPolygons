# Benchmark Dashboard Refactor TODO

This file tracks cleanup and performance work for `apps/benchmark-dashboard`.

## Luna Execution Brief

Use this when continuing the refactor with a smaller model:

```text
You are working in /Users/gabrielushijima/Documents/Scripts/TouringPolygons, specifically apps/benchmark-dashboard.

The benchmark dashboard has already had several backend, frontend, template, and stylesheet refactors. Continue improving performance, storage, and maintainability without changing behavior.

Rules:
- Read TODO.md before starting.
- Make small, mechanical, verifiable changes.
- Do not revert existing changes.
- Preserve all IDs, classes, names, aria attributes, and data-* hooks used by JavaScript.
- Before editing, inspect the relevant files.
- Prefer extraction by explicit dependency injection over rewrites.
- Do not redesign the UI.
- Do not remove compatibility with existing campaign/result files.
- If a change looks risky, stop and explain the risk before continuing.
- After each meaningful step, update TODO.md and run the relevant checks.

Validation commands:
- cd apps/benchmark-dashboard
- node --check static/*.js
- npm run lint:js
- uv run python -m unittest tests/test_main.py
- uv run ruff check .
- git diff --check
- If HTML, CSS, frontend boot, or routes changed: run uvicorn locally and then npm run test:browser.

Recommended first action:
Read TODO.md, main.py, dashboard_routes.py, static/app.js, static/manual-cases.js, templates/index.html, and the list of static/*.css files. Then execute only the next unchecked task from "Luna Task Queue".
```

### Luna Operating Rules

- Do one small task group at a time. Avoid broad "clean everything" edits.
- Keep route moves mechanical: copy route body, inject dependencies, register router, delete old route, run tests.
- Keep JS moves mechanical: create module/factory, inject dependencies, wire in `app.js`, remove old block, run JS checks.
- Keep CSS moves mechanical: preserve selector behavior and loading order.
- Do not move the manual canvas editor object all at once unless all intermediate checks are green.
- Do not remove `manual.bin` until tests prove it can be regenerated from `manual-cases.json`.
- If a command fails due macOS sandbox/cache permissions, rerun the same command with the appropriate approval instead of changing code around it.
- Always stop local dev servers before finishing.
- Final response after each batch must include files changed, size changes when relevant, and checks run.

### Luna Task Queue

1. [ ] Inspect current size and structure:
	- run `wc -l main.py static/app.js static/*.css templates/index.html templates/partials/*.html`;
	- run `rg -n "^@app\\.|register_.*routes|create.*Controller|const manualEditor" main.py static/app.js`;
	- summarize only the next safe extraction boundary.
2. [ ] Extract campaign list/detail/delete routes from `main.py` into `dashboard_campaign_routes.py`:
	- `GET /api/campaigns`;
	- `GET /api/campaigns/{name}`;
	- `DELETE /api/campaigns/{name}`.
3. [ ] Register campaign routes through `register_campaign_routes(app, ...)` with explicit dependencies.
4. [ ] Run `uv run ruff check .` and `uv run python -m unittest tests/test_main.py` after the first route extraction.
5. [ ] Extract campaign preview routes into `dashboard_campaign_routes.py`:
	- `GET /api/campaigns/{name}/preview`;
	- `GET /api/campaigns/{name}/preview/{kind}`;
	- `GET /api/campaigns/{name}/solution-preview/{case_index}`.
6. [ ] Run Python tests, Ruff, and browser smoke after preview route extraction.
7. [ ] Extract campaign creation/import routes:
	- `POST /api/campaigns/synthetic`;
	- `POST /api/campaigns/osm`;
	- `POST /api/campaigns/canonical`;
	- `POST /api/campaigns/german`;
	- `POST /api/campaigns/manual`.
8. [ ] Extract manual case mutation routes:
	- `GET /api/campaigns/{name}/cases`;
	- `PUT /api/campaigns/{name}/cases`;
	- `POST /api/campaigns/{name}/cases`;
	- `PUT /api/campaigns/{name}/cases/{case_index}`;
	- `DELETE /api/campaigns/{name}/cases/{case_index}`.
9. [ ] Extract editor solve route:
	- `POST /api/editor/solve`;
	- keep WASM/live solver behavior unchanged.
10. [ ] Remove imports from `main.py` that become unused after route extraction.
11. [ ] If `dashboard_campaign_routes.py` becomes too large, split it into:
	- `dashboard_campaign_routes.py`;
	- `dashboard_manual_routes.py`;
	- `dashboard_import_routes.py`.
12. [ ] Extract job orchestration helpers from `main.py` only after route extraction is stable:
	- `active_job`;
	- `persist_jobs`;
	- `load_jobs`;
	- `refresh_job_progress_from_logs`;
	- `update_job_progress`;
	- `run_job`.
13. [ ] Keep the `jobs` registry ownership obvious. Do not introduce background workers or queues.
14. [ ] Add or update tests for any route extraction that changes dependency wiring.
15. [ ] Add a Jinja render regression test:
	- render `templates/index.html`;
	- verify required IDs are present;
	- verify partial includes render correctly.
16. [ ] Required template IDs for the render test:
	- `create-form`;
	- `manual-case-canvas`;
	- `run-form`;
	- `compare-form`;
	- `campaign-modal`;
	- `confirm-modal`;
	- `keybind-modal`;
	- `job-dock`.
17. [ ] Add a CSS manifest test:
	- parse `static/style.css`;
	- verify every imported CSS file exists.
18. [ ] Add a CSS template loading test:
	- parse `templates/index.html`;
	- verify all linked `static/*.css` files exist.
19. [ ] Extract modal helpers from `static/app.js` into `static/modals.js`:
	- confirmation modal;
	- campaign modal open/close;
	- shared close handling.
20. [ ] Keep modal extraction dependency-injected. Do not import `state` into too many modules unless already established.
21. [ ] Extract instance modal orchestration into `static/instance-modals.js`:
	- `openInstanceModal`;
	- `openBenchmarkedInstanceModal`;
	- `instanceModalTitle`;
	- `setupModalTitleRename`;
	- `renameCampaignInstance`.
22. [ ] Preserve `manualCases.editInstance(...)` behavior when extracting instance modals.
23. [ ] Extract OSM picker helpers from `app.js` into `static/osm-picker.js`:
	- `formatBytes`;
	- `selectOsmFile`;
	- `renderOsmFiles`;
	- `scanOsmFiles`.
24. [ ] Extract campaign choice grid rendering into `static/campaign-choice.js`.
25. [ ] Extract run/comparison form submission orchestration if it can be done without changing polling behavior.
26. [ ] Do not extract the manual canvas editor object yet if `app.js` is still above 2000 lines but checks are unstable.
27. [ ] Start manual canvas editor extraction with pure camera helpers:
	- bounds calculation;
	- zoom limits;
	- world/canvas coordinate conversion;
	- camera persistence keys.
28. [ ] Create `static/manual-editor-camera.js` only for pure camera functions.
29. [ ] Add JS tests for camera helpers if extracted.
30. [ ] Extract manual selection helpers:
	- selected point lookup;
	- selection rectangle logic;
	- active polygon reset helpers.
31. [ ] Create `static/manual-editor-selection.js` only if the helper functions are mostly pure.
32. [ ] Convert `manualEditor` to a factory only after camera and selection helpers are extracted:
	- `createManualEditor({ $, state, requestJSON, drawCanvasScene, ... })`.
33. [ ] Move the full `manualEditor` object to `static/manual-editor.js` only after the factory passes browser smoke.
34. [ ] Keep `app.js` as the bootstrap/wiring file:
	- imports;
	- dependency construction;
	- top-level event listeners;
	- refresh startup.
35. [ ] Target `app.js` size below 1500 lines before doing deeper behavior changes.
36. [ ] Add JS unit test scaffolding if simple:
	- Node `assert`;
	- no browser dependency for pure modules.
37. [ ] Add JS tests for `case-data.js`.
38. [ ] Add JS tests for `command-builders.js`.
39. [ ] Add JS tests for `report-utils.js`.
40. [ ] Add JS tests for `job-utils.js`.
41. [ ] Add JS tests for `manual-cases.js` only if DOM stubbing stays small.
42. [ ] Centralize localStorage keys into `static/storage.js`.
43. [ ] Replace repeated localStorage strings with exported constants.
44. [ ] Confirm key names do not change unless migration fallback is included.
45. [ ] Improve thumbnail network pressure:
	- inspect where preview images are created;
	- add `loading="lazy"` to generated `<img>` tags where safe;
	- avoid changing visible behavior.
46. [ ] Consider limiting initial thumbnail rendering for very large grids.
47. [ ] Do not implement complex virtualization until simple lazy loading is measured/validated.
48. [ ] Add tests around `dashboard_files.py` JSON cache invalidation when files change.
49. [ ] Add tests around CSV cache invalidation when files change.
50. [ ] Add tests around cache trimming beyond `FILE_CACHE_LIMIT`.
51. [ ] Review `dashboard_previews.py` for any non-atomic preview writes.
52. [ ] Add regression test for lazy single-instance preview generation.
53. [ ] If feasible, assert single-instance preview generation does not read all cases.
54. [ ] Review `dashboard_reports.py` with malformed or empty Markdown summaries.
55. [ ] Add tests for:
	- empty summary file;
	- missing CSV;
	- comparison file missing;
	- partially written result CSV.
56. [ ] Document current manual-case storage flow:
	- `manual-cases.json` as canonical editable data;
	- `inputs/manual.bin` as solver compatibility artifact.
57. [ ] Before removing duplicate storage, add test:
	- save manual cases;
	- delete `inputs/manual.bin`;
	- call endpoint that needs solver input;
	- verify binary is regenerated.
58. [ ] Add helper `ensure_manual_binary_cache(path)` if missing.
59. [ ] Treat `manual.bin` as generated cache only after all manual binary tests pass.
60. [ ] Do not change the solver CLI contract unless explicitly requested.
61. [ ] Inspect generated preview/storage directories for redundant files before deleting anything.
62. [ ] Never delete user data or campaign artifacts without an explicit backup/migration path.
63. [ ] Add README notes for the dashboard architecture after major extraction stabilizes.
64. [ ] Update README with:
	- how to run dashboard;
	- validation commands;
	- file/module layout;
	- manual storage model.
65. [ ] Keep `TODO.md` current after every completed task group.
66. [ ] After each route extraction batch, run:
	- `uv run python -m unittest tests/test_main.py`;
	- `uv run ruff check .`;
	- `git diff --check`.
67. [ ] After each frontend extraction batch, run:
	- `node --check static/*.js`;
	- `npm run lint:js`;
	- `git diff --check`.
68. [ ] After HTML/CSS/frontend boot changes, run browser smoke:
	- start `uv run uvicorn main:app --host 127.0.0.1 --port 8017`;
	- run `npm run test:browser`;
	- stop the server.
69. [ ] Before finalizing a long Luna session, run all checks:
	- `node --check static/*.js`;
	- `npm run lint:js`;
	- `uv run python -m unittest tests/test_main.py`;
	- `uv run ruff check .`;
	- `git diff --check`;
	- `npm run test:browser`.
70. [ ] End each batch with a compact summary:
	- files changed;
	- line count changes;
	- checks passed;
	- next recommended task number.

## Planned

- [ ] Split `main.py` into focused backend modules:
	- [x] request/data models and shared types;
	- [x] path and campaign metadata helpers;
	- [x] binary case parsing and writing;
	- [x] preview generation;
	- job orchestration and persistence;
	- [x] report parsing;
	- [x] support/read-only FastAPI routes;
	- campaign mutation FastAPI routes.
- [ ] Split `static/app.js` into ES modules:
	- [x] API client;
	- [x] shared state;
	- [x] campaign rendering;
	- [x] job dock and polling;
	- [x] benchmark/comparison reports;
	- [x] editor geometry;
	- [x] editor renderer;
	- [x] manual campaign/case list and autosave controller;
	- manual canvas editor interactions;
	- [x] read-only instance viewer;
	- [x] command builders and form controls;
	- [x] theme, keybinding, and UI utility helpers.
- [x] Split `templates/index.html` into focused Jinja partials.
- [x] Split `static/style.css` into focused domain stylesheets.
- [x] Consolidate duplicated `static/style.css` rules for common controls and containers.
- [x] Replace eager per-instance preview generation with lazy generation on first request.
- [x] Separate manual autosave from benchmark artifact rebuilding.
- [x] Stream binary case reads instead of loading whole `.bin` files into memory.
- [x] Add optional binary case offset indexes for fast single-case reads.
- [x] Cache completed-instance counts for unchanged result files.
- [x] Move stale preview migration out of `GET /api/campaigns`.
- [x] Replace unbounded JSON/CSV globals with a bounded file-signature cache.
- [x] Cache frontend convex decomposition per polygon version.
- [x] Extract a shared canvas renderer used by both the manual editor and read-only viewer.
- [x] Reduce job polling/report refresh pressure, possibly with server-sent events.
- [x] Add regression tests around campaign summaries, manual save/rebuild behavior, preview paths, and binary parsing.
- [x] Add browser-level smoke-test scaffolding and static quality configuration.
- [x] Add resource limits to dashboard API requests.
- [ ] Remove duplicate manual-case storage by replacing the solver binary with a generated cache.

## Done

- Replaced eager per-instance preview materialization with lazy generation.
	- `write_imported_previews()` now records virtual `previews/instances/case-XXXX.svg` paths without writing every instance SVG immediately.
	- `/api/campaigns/{name}/preview/instance-{index}` now creates the missing SVG on first request and reuses it afterward.
	- Existing preview files remain compatible.
	- Added a regression test for lazy instance preview generation.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
- Reworked binary case parsing to stream from disk instead of reading entire `.bin` files into memory.
	- `binary_case_count()` now scans with file reads/seeks.
	- `read_binary_cases()` now reads only up to the requested limit.
	- Shared binary format helpers now use cached `struct.Struct` instances.
	- Added a regression test for counting and limited reads.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
- Replaced unbounded JSON/CSV file caches with bounded least-recently-used caches.
	- Cache hits are refreshed with `move_to_end()`.
	- New inserts trim both caches to `FILE_CACHE_LIMIT`.
	- Explicit invalidation through `.pop()` still works for mutated campaign files.
	- Added a regression test for cache trimming.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
- Moved stale overview preview regeneration out of campaign summaries.
	- `campaign_summary()` is now read-only with respect to preview migration.
	- Non-instance preview requests still refresh stale overview SVGs before serving them.
	- Added a regression test to keep `campaign_summary()` from calling preview regeneration.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
- Cached frontend convex decomposition results per polygon object and coordinate signature.
	- Editor redraws now reuse decompositions while polygon coordinates are unchanged.
	- The cache invalidates naturally when point coordinates change because the signature changes.
	- Verified with `node --check apps/benchmark-dashboard/static/app.js`.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
- Started splitting `static/app.js` into ES modules.
	- Extracted API requests into `static/api.js`.
	- Extracted DOM helpers into `static/dom.js`.
	- Extracted shell quoting, CSV download, and elapsed/seconds formatting into `static/format.js`.
	- Switched the dashboard script tag to `type="module"`.
	- Verified module syntax with `node --input-type=module --check`.
	- Verified the app and module files are served by Uvicorn on a temporary local port.
- Extracted pure editor geometry helpers into `static/editor-geometry.js`.
	- Moved convexity checks, ear-clipping decomposition, decomposition caching, and solution direction helpers out of `app.js`.
	- Kept the module dependency-free so it can be tested and reused independently.
	- Verified all dashboard JavaScript modules pass Node syntax checks.
- Extracted the mutable dashboard state object into `static/state.js`.
	- Kept the same object shape and Map instances so existing event handlers retain their behavior.
	- Verified all dashboard JavaScript modules pass Node syntax checks.
- Extracted pure job status helpers into `static/job-utils.js`.
	- Moved job panel selection, labels, progress text, terminal states, and status classes out of `app.js`.
	- Kept DOM event handling and polling orchestration in `app.js`.
	- Verified all dashboard JavaScript modules pass syntax checks.
- Extracted campaign list rendering into `static/campaign-rendering.js`.
	- Kept filtering and card interaction behavior unchanged while leaving modal and deletion workflows in `app.js`.
	- Passed existing campaign helpers into the renderer to avoid circular module dependencies.
- Extracted report parsing and formatting helpers into `static/report-utils.js`.
	- Moved numeric parsing, timing parsing, solver labels, metric cards, and report lookup helpers out of `app.js`.
	- Kept report DOM rendering in `app.js` until its event and formatting dependencies can be separated cleanly.
- Extracted backend request models and shared case types into `dashboard_models.py`.
	- Kept `main.py` compatibility imports so the existing API and test loader continue to work.
- Extracted binary case IO and offset indexing into `dashboard_binary.py`.
	- Kept the offset cache shared with the existing dashboard cache controls.
	- Verified the full 12-test dashboard suite after the extraction.
- Extracted shared canvas scene composition into `static/canvas-renderer.js`.
	- Manual editing and read-only viewing now call the same polygon, solution, selection, and label renderer.
	- Preserved mode-specific interaction and camera behavior around the shared renderer.
- Added browser and static-quality tooling.
	- Added `tests/browser_smoke.mjs` and `npm run test:browser` using Playwright.
	- Added ESLint and Ruff configuration for the dashboard sources.
	- Browser execution remains environment-dependent because the local browser runner may block localhost URLs.
- Added explicit upper bounds for generated instances, polygon counts, worker threads, benchmark sizes, and manual editor payloads.
- Kept `manual-cases.json` as the canonical editable source and `inputs/manual.bin` as a compatibility artifact for the existing solver CLI.
	- Full removal of the binary artifact requires changing the solver input contract and is intentionally still pending.
- Added atomic, locked preview writes using temporary files and `os.replace()`.
- Added compact `/api/jobs/{job_id}/progress` responses for active polling.
- Added regression tests for resource limits and compact job snapshots.
- Separated manual autosave persistence from preview generation.
	- Autosave still updates the editable JSON, binary input, campaign metadata, and invalidates benchmark results.
	- Autosave now skips expensive preview rendering and invalidates old preview metadata.
	- Preview requests regenerate the previews lazily when they are needed.
- Reduced duplicate polling work in the frontend.
	- Centralized job and report polling intervals.
	- Prevented overlapping comparison report requests when a prior refresh is still running.
	- Removed a duplicate final dashboard refresh after benchmark jobs complete.
- Added an in-memory binary case offset cache for fast single-case reads.
	- `binary_case_offsets()` scans each `.bin` file once per file signature and caches valid case offsets.
	- `binary_case_count()` now uses the offset cache.
	- `read_binary_case()` can jump directly to a specific case.
	- `read_campaign_case()` resolves global campaign case indexes across multiple input files.
	- Lazy instance preview generation now reads only the requested case instead of all campaign cases.
	- Added regression tests for offset invalidation and multi-input campaign lookup.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
- Extracted shared file cache and report helpers from `main.py`.
	- Moved bounded JSON/CSV file cache logic into `dashboard_files.py`.
	- Moved markdown summary parsing and comparison report discovery into `dashboard_reports.py`.
	- Kept compatibility imports in `main.py` for existing tests and callers.
- Extracted campaign metadata and preview generation helpers from `main.py`.
	- Moved input counting, preview metadata lookup, campaign case lookup, and input-file labels into `dashboard_campaigns.py`.
	- Moved SVG preview rendering, stale-preview detection, lazy instance preview generation, and preview metadata rewriting into `dashboard_previews.py`.
	- Kept route-level behavior and compatibility imports unchanged.
- Cached completed-instance counts for unchanged benchmark outputs.
	- `completed_instance_count()` now keys cached counts by `run-index.csv` and referenced result CSV file signatures.
	- Campaign list refreshes avoid rescanning unchanged result CSV rows.
	- Added a regression test for count cache invalidation when a result CSV changes.
- Continued splitting `static/app.js` into focused modules.
	- Moved case cloning/payload helpers into `static/case-data.js`.
	- Moved CLI command preview builders into `static/command-builders.js`.
	- Moved theme handling into `static/theme.js`.
	- Moved shared UI helpers and tooltip handling into `static/ui-utils.js`.
	- Moved WASM solver loading and calls into `static/editor-solver.js`.
	- Moved read-only instance viewer rendering and interactions into `static/readonly-viewer.js`.
	- Moved benchmark/comparison report rendering into `static/report-rendering.js`.
	- Moved job dock rendering/interactions into `static/job-dock.js`.
	- Moved keybinding persistence and editing UI into `static/keybinds.js`.
	- Moved segmented controls, sliders, and filter wiring into `static/controls.js`.
- Split the dashboard HTML into focused Jinja partials.
	- `index.html` now contains only the page shell, tabs, includes, and module script tag.
	- Moved each primary panel into `templates/partials/*_panel.html`.
	- Moved dock and modal markup into `templates/partials/modals.html`.
	- Verified the rendered template preserves the same JS-facing ids/classes/data attributes as the original file.
- Consolidated repeated stylesheet rules.
	- Shared range input styling now covers thread sliders, numeric range controls, and zoom style sliders through custom properties.
	- Shared custom scrollbar styling now covers instance and benchmarked preview grids.
	- Shared report/progress container styling now covers benchmark, comparison, result-preview, and progress cards.
	- Deduplicated common icon and modal card primitives.
- Split the stylesheet by UI domain.
	- Added `base.css`, `forms.css`, `campaigns.css`, `editor.css`, `previews.css`, `reports.css`, `overlays.css`, and `responsive.css`.
	- `index.html` loads the focused CSS files directly for better cache granularity.
	- Kept `style.css` as an import manifest for compatibility with older direct references.
- Extracted manual campaign and case-list orchestration from `app.js`.
	- Added `static/manual-cases.js` for editable campaign selection, case list rendering, rename/duplicate/delete, and autosave.
	- Kept the canvas editor object in `app.js` while injecting it into the controller to avoid a circular module dependency.
- Extracted support/read-only route registration from `main.py`.
	- Added `dashboard_routes.py` for job status/cancel, system, OSM file scan, logs, summaries, benchmarked instances, comparisons, and result listing routes.
	- Kept route behavior registered on the same FastAPI app via explicit dependency injection.
