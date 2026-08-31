# Benchmark Dashboard

Local FastAPI dashboard for creating, editing, running, comparing, and inspecting TPP benchmark campaigns.

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

- `Create`: builds synthetic or OSM-derived campaigns, with optional preview generation.
- `Cases`: opens a campaign in the manual instance editor. The editor supports moving endpoints and vertices, drawing polygons, selecting/deleting vertices, zooming, fitting, grid snapping, convex decomposition display, labels, and live path solving.
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

Generated and imported campaigns can be inspected and edited through the same case API. Saving edits rebuilds the campaign input binary and its preview metadata.

## Module Layout

- `main.py`: FastAPI application setup, compatibility imports, and route wiring.
- `dashboard_*`: backend models, file and binary IO, campaign and preview helpers, reports, jobs, and route groups.
- `static/app.js`: frontend bootstrap and cross-feature wiring.
- `static/*.js`: focused API, rendering, preview, editor, report, job, and form modules.
- `templates/partials`: page panels, dialogs, and shared modal markup.
- `static/*css`: stylesheet modules loaded by the page shell.

## Validation

From `apps/benchmark-dashboard`:

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
npm run lint:js
npm run check:js
for file in static/*.js; do node --check "$file"; done
node --test tests/*.test.mjs
DASHBOARD_URL=http://127.0.0.1:8017 npm run test:browser
```

The Python suite also includes route-level integration coverage for manual campaign mutation and regeneration of a missing `inputs/manual.bin` compatibility artifact.

Manual campaigns use `manual-cases.json` as the canonical editable representation. The generated `inputs/manual.bin` file remains a solver compatibility artifact and is rebuilt when manual cases are saved.
