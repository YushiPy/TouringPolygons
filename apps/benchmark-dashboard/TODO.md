# Benchmark Dashboard Refactor TODO

This file tracks cleanup and performance work for `apps/benchmark-dashboard`.

## Planned

- [ ] Split `main.py` into focused backend modules:
	- request/data models and shared types;
	- path and campaign metadata helpers;
	- binary case parsing and writing;
	- preview generation;
	- job orchestration and persistence;
	- report parsing;
	- FastAPI routers.
- [ ] Split `static/app.js` into ES modules:
	- API client;
	- shared state;
	- campaign rendering;
	- job dock and polling;
	- benchmark/comparison reports;
	- editor geometry;
	- editor renderer;
	- manual editor interactions;
	- read-only instance viewer.
- [x] Replace eager per-instance preview generation with lazy generation on first request.
- [ ] Separate manual autosave from benchmark artifact rebuilding.
- [ ] Stream binary case reads instead of loading whole `.bin` files into memory.
- [ ] Add optional binary case offset indexes for fast single-case reads.
- [ ] Move stale preview migration out of `GET /api/campaigns`.
- [ ] Replace unbounded JSON/CSV globals with a bounded file-signature cache.
- [ ] Cache frontend convex decomposition per polygon version.
- [ ] Extract a shared canvas renderer used by both the manual editor and read-only viewer.
- [ ] Reduce job polling/report refresh pressure, possibly with server-sent events.
- [ ] Add regression tests around campaign summaries, manual save/rebuild behavior, preview paths, and binary parsing.

## Done

- Replaced eager per-instance preview materialization with lazy generation.
	- `write_imported_previews()` now records virtual `previews/instances/case-XXXX.svg` paths without writing every instance SVG immediately.
	- `/api/campaigns/{name}/preview/instance-{index}` now creates the missing SVG on first request and reuses it afterward.
	- Existing preview files remain compatible.
	- Added a regression test for lazy instance preview generation.
	- Verified with `uv run python -m unittest apps/benchmark-dashboard/tests/test_main.py`.
