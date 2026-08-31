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
