# Benchmark Dashboard

Local FastAPI dashboard for creating, appending, editing, running, comparing, and inspecting TPP benchmark campaigns.

The dashboard works with local instances, campaigns, and solver evidence. The
public research presentation and its frozen data live in
[`../siicusp34/`](../siicusp34/README.md). The former `/evento` pages and their
duplicated snapshots have been retired from this application.

## Run

From the repository root:

```bash
cd apps/benchmark-dashboard
uv sync
npm start
```

Then open `http://127.0.0.1:8017`.

For automatic reload during development, or to choose another port:

```bash
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Offline editor

The lightweight editor at `/editor/offline` is the migration target for the old
former standalone editor workflow. It keeps a browser-local library of instances,
imports legacy `startPoint`/`targetPoint` JSON as well as dashboard JSON, and
imports/exports the library as a single file. It does not call the dashboard API;
convex decomposition runs in JavaScript, solving uses optional WebAssembly,
and the `Last-step map` layer uses the WASM solver's extracted cone buffers for
fixed-order convex pairwise-disjoint instances. It deliberately does not carry
over the old JavaScript solver.

With the dashboard server running, open
`http://127.0.0.1:8017/editor/offline`. To serve the static page without
FastAPI, run `python3 -m http.server 8020` from this directory and open
`http://127.0.0.1:8020/offline-editor/`.

The page remains useful without the generated solver files. To enable local
fixed-order solving, build the browser assets with `bash wasm/build.sh`; the
generated files under `static/wasm/` are ignored by Git.

## Main Views

- `Create`: builds synthetic or OSM-derived campaigns, with optional preview generation. Generated batches can also be appended to an existing campaign.
- `Cases`: opens a campaign in the manual instance editor. The editor supports moving endpoints and vertices, drawing polygons, selecting/deleting vertices, deleting instances, zooming, fitting, grid snapping, convex decomposition display, labels, and live path solving.
- `Inspect`: lists campaigns, imports bundled suites, shows SVG preview grids, opens individual instances, and shows recent result files and jobs.
- `Benchmark`: runs a selected campaign through a solver and displays progress, summary metrics, histograms, and solved instance cards.
- `Comparison`: runs the same campaign through multiple solvers and compares wall time, convex solve time, calls, and solved counts.

## Instance Inspection

Campaign preview grids intentionally use generated SVG previews, so large suites with hundreds of instances stay cheap to browse.

Clicking an instance opens a read-only version of the editor renderer. This expanded inspection view supports:

- pan, zoom, and fit;
- grid, path, decomposition, and label toggles;
- live computed path rendering;
- an `Edit Instance` shortcut in the viewer toolbar;
- inline instance renaming from the modal title.

The modal title shows the campaign/test-case name on the first line and the selected instance position/name on the second line.

## Notes

Campaign data lives under `benchmarks/workspace/campaigns` and benchmark outputs under `benchmarks/workspace/runs` (see `benchmarks/README.md`). Campaigns created here record `"origin": "dashboard"`.

Generated and imported campaigns can be inspected and edited through the same case API. Saving edits updates the canonical manual case JSON and campaign metadata; preview images and the solver compatibility binary are regenerated lazily when requested.

Appending generated cases converts the destination campaign to the editable manual-json format, marks appended instances as generated, stores a compact edit-history entry, and clears stale benchmark outputs.

Shift-clicking a delete button bypasses the confirmation for faster editing sessions. Normal delete clicks and overwrite prompts still ask for confirmation.

## Module Layout

- `main.py`: FastAPI application setup, compatibility imports, and route wiring.
- `dashboard/`: backend models, file and binary IO, campaign and preview helpers, reports, jobs, and route groups.
- `static/app.js`: frontend bootstrap and cross-feature wiring.
- `static/*.js`: focused API, rendering, preview, editor, report, job, and form modules.
- `templates/partials`: page panels, dialogs, and shared modal markup.
- `static/*css`: stylesheet modules loaded by the page shell.
- `scripts/run-tests.sh`: dashboard test runner used by `npm run test:all`.

## Validation

From `apps/benchmark-dashboard`:

```bash
npm run test:all
```

The full runner prints each command's detailed output and finishes with a compact summary of passed suites and test counts.

Browser smoke is opt-in because it starts a local dashboard server and launches Playwright:

```bash
RUN_BROWSER=1 npm run test:all
```

Use another port if needed:

```bash
RUN_BROWSER=1 BROWSER_PORT=8020 npm run test:all
```

The Python suite includes route-level integration coverage for manual campaign mutation, append behavior, stale benchmark invalidation, and lazy regeneration of a missing or stale `inputs/manual.bin` compatibility artifact.

Manual campaigns use `manual-cases.json` as the canonical editable representation. The generated `inputs/manual.bin` file remains a solver compatibility artifact and is rebuilt on demand for previews, benchmark runs, and comparisons.

## Visit order

Choose `Fixed order` or `Free order` in Benchmark, Comparison, or Cases. The selection is synchronized across these views. Fixed order keeps the existing solver pipeline. Free order uses the native nonconvex TPP branch-and-bound, with fixed start and target points. The live editor uses a three-second budget and displays the incumbent path and gap when the search has not finished.

Free-order campaigns currently run with one worker. An empty time limit means 30 seconds per instance. Comparison supports `Our TPP B&B` and `External TSPN`; the external checkout and its Python environment must be installed, and its time limit must be an integer number of seconds. The fixed-order editor uses the optional WASM solver when available and falls back to the local fixed-order API otherwise. Rebuild the browser solver with `bash wasm/build.sh`; generated files under `static/wasm/` remain ignored.

Reports include per-instance bounds, gaps, timing, termination, and our saved paths and first-visit orders. Results are saved separately under `benchmarks/workspace/campaigns/<campaign>/results/<run>/report.json`. Matching completed configurations are reused unless forced. They never populate fixed-order summary files.

In Comparison, click `Show recorded free-order comparison (60 instances)` to inspect the measured development suite, including numerical tolerances and endpoint validation differences. This requires the local artifacts under `benchmarks/workspace/runs/unordered/final-dev.jsonl` and `benchmarks/workspace/runs/tspn-socg`; those benchmark artifacts are not tracked in Git.

The same campaign runner is available from the repository root:

```bash
apps/benchmark-dashboard/.venv/bin/python benchmarks/tpp.py free-order CAMPAIGN --solver tpp-ours --solver tpp-fekete --max-seconds 2
```

The additional browser test exercises mode switching, the saved comparison, a live free-order solve, a campaign run, and an actual external comparison. Start the dashboard, then run:

```bash
DASHBOARD_URL=http://127.0.0.1:8137 npm run test:browser:free
```
