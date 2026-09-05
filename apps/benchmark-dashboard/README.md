# Benchmark Dashboard

Local FastAPI dashboard for creating, appending, editing, running, comparing, and inspecting TPP benchmark campaigns.

## Run

From the repository root:

```bash
cd apps/benchmark-dashboard
uv sync
uv run uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000`.

To use a fixed port:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8017
```

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

Campaign data lives under `benchmarks/campaigns`. Benchmark outputs live under `benchmarks/results`.

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

Free-order campaigns currently run with one worker. An empty time limit means 30 seconds per instance. Comparison supports `Our TPP B&B` and `External TSPN`; the external checkout and its Python environment must be installed, and its time limit must be an integer number of seconds.

Reports include per-instance bounds, gaps, timing, termination, and our saved paths and first-visit orders. Results are saved separately under `benchmarks/campaigns/<campaign>/results/free-order/<run>/report.json`. Matching completed configurations are reused unless forced. They never populate fixed-order summary files.

In Comparison, click `Show recorded free-order comparison (60 instances)` to inspect the measured development suite, including numerical tolerances and endpoint validation differences. This requires the local artifacts under `benchmarks/results/unordered/final-dev.jsonl` and `tspn-comparison/results/unordered-final`; those benchmark artifacts are not tracked in Git.

The same campaign runner is available from the repository root:

```bash
apps/benchmark-dashboard/.venv/bin/python benchmarks/tpp.py free-order CAMPAIGN --solver unordered --solver tspn --max-seconds 2
```

The additional browser test exercises mode switching, the saved comparison, a live free-order solve, a campaign run, and an actual external comparison. Start the dashboard, then run:

```bash
DASHBOARD_URL=http://127.0.0.1:8137 npm run test:browser:free
```
