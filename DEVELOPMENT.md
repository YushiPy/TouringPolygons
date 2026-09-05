# Development Guide

This repository is a research monorepo. Keep maintained solver code, generated artifacts, and historical experiments separated so benchmark results remain reproducible.

## Maintained entry points

- `README.md`: project overview, algorithm summary, and fresh-clone setup.
- `scripts/sanity_check.sh`: broad local verification for a fresh clone.
- `benchmarks/tpp.py`: public benchmark and instance-generation command entry point.
- `CMakePresets.json`: IDE-friendly CMake presets for maintained C++ solver targets.

## Directory boundaries

- `packages/`: maintained package code, package-local tests, and intentional regression fixtures.
- `benchmarks/scripts/`: command internals used by `benchmarks/tpp.py`; prefer shared helpers here over copy-pasted parsing or benchmark logic.
- `benchmarks/suites/`: tracked canonical input corpora only. Derived suites and run outputs are ignored.
- `apps/`: runnable visualizers and their UI/server dependencies.
- `docs/`: reports, bibliography notes, and other project documentation.

## Generated and local files

Generated benchmark campaigns, benchmark results, derived suites, CMake build trees, Python caches, local virtual environments, and local source archives are ignored by `.gitignore`. If a generated file is needed for reproducibility, document the command that recreates it rather than committing the output, unless it is a small canonical fixture.

## Public repository policy

This repository must not contain private supervision material or third-party
content that is not ours to redistribute. Keep meeting recordings, transcripts,
summaries, and other correspondence under `docs/meetings/`, which is ignored.
Keep previously submitted material written by others under
`docs/reports/SIICUSP/resultados-anteriores/`, external paper PDFs under
`docs/bibliography/`, and external repository checkouts such as
`tspn-comparison/` ignored as well.

`benchmarks/campaigns/` is a local experiment workspace and remains ignored.
Historical implementation experiments under `experiments/` are local-only and
remain ignored as well.
When a case from a campaign becomes a durable regression test, copy or export
only that case into `benchmarks/suites/`, preferably as a small documented
fixture. Commit the stable input required by the test harness and document the
command that produces any derived binary or preview. Do not commit a campaign's
run history, previews, local paths, or unrelated cases merely to preserve one
test.

## Python benchmark scripts

Use `benchmarks/tpp.py` for user-facing commands. The common workflow should
stay visible as `create`, `run`, and `status`; more specialized scripts can
remain as lower-level commands. Lower-level scripts under `benchmarks/scripts/`
should remain importable and testable as modules, with command-line parsing
isolated in `make_parser()` and `main()`.

Shared binary case parsing and geometry filtering belongs in `benchmarks/scripts/benchmark_cases.py`.

## C++ packages

Maintained C++ package dependencies should flow in this direction:

```text
tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex
```

Avoid reintroducing local copies of shared geometry or convex solver code in downstream packages. If a package needs shared behavior, add it to the upstream package and link it explicitly through CMake.

## Verification

Run the broad check before larger changes:

```bash
scripts/sanity_check.sh --no-install
```

For faster benchmark-script checks during refactors:

```bash
python3 -m compileall benchmarks/scripts benchmarks/tpp.py
python3 benchmarks/tpp.py generate-suites
```
