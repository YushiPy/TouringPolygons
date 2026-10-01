# Cooperative TSPN portfolio holdout screening — 2026-09-29

**Status: six-case holdout completed, followed by a three-repetition run.** The initial screening directories preserve one repetition per mode; the less noisy three-repetition comparison is in [`repeat3/analysis.md`](repeat3/analysis.md). These six preselected cases are a diagnostic sample, not a broad corpus estimate.

## Measurement

Same simplified closed polygon inputs and formulation as the parent campaign: free cyclic order, arbitrary visit order, closed polygon regions, no fixed point. The six cases were selected before timing by a seeded filename hash, excluding Bangalore, and are preserved in `selection.json` and `holdout-inputs.json`. Each backend received a 1-second solver limit and an 8-second process cap. The gap target is `UB <= (1 + 1e-6) LB`; native relative tolerance is `1e-6/(1+1e-6)`, absolute gap zero. Feasibility tolerance is `1e-8`, independent tour validation tolerance `1e-7`. Fekete used one Gurobi thread. Native portfolio runs used two search workers and include the cooperative join tail in whole-solver time. Full per-run config, raw JSONL, bounds, validation and telemetry are stored in each mode directory.

| Initial one-repetition screen | Ours valid / closed | Fekete closed | Mutually closed | Median matched speedup | Portfolio median proof / join (ms) |
|---|---:|---:|---:|---:|---:|
| Default | 6/6 / 6/6 | 5/6 | 5/6 | 2.26× | — / — |
| Isolated dfs-bfs | 6/6 / 6/6 | 5/6 | 5/6 | 3.05× | — / — |
| Independent race | 6/6 / 6/6 | 5/6 | 5/6 | 2.23× | 3.841 / 0.201 |
| Cooperative shared-incumbent | 6/6 / 6/6 | 5/6 | 5/6 | 1.98× | 4.212 / 0.305 |

The single-repetition screen is retained as the initial pass. The three-repetition follow-up on the same inputs reports per-run gap closure, strict `exactly_covers` counts, portfolio publication/import totals and proof/join timings. `validation.md` records native suites for the preceding oracle and portfolio changes.

## Reproduction and provenance

The native executable was built from the workspace state recorded by `base-commit.txt`, `source.patch`, `source-files.sha256`, and the exact untracked portfolio header under `source-overlay/`. The overlay hash matches the hash captured by each native build configuration. The same frozen native binary was used for the initial screen and all three-repetition runs. Each variant ran through the public `benchmarks/tpp.py tspn-benchmark` command with `--skip-build`; invocations and outputs are in `measurement.txt` and the mode logs. Binary hashes, vendor commit, input hash, limits, tolerances, and thread counts are recorded in `provenance.json` and each mode’s `config.json`. The Fekete source checkout is read-only.
