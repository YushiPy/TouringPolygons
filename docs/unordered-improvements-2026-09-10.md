# Free-order nonconvex TPP improvements — 2026-09-10

The retained implementation improves both development and held-out results without
changing the final requested optimality gap or adding a production Gurobi dependency.
The improved executable is installed locally at `.build/unordered/tpp`.

## Results

Single sequential sweeps, Apple arm64, Apple Clang 17 release build (`-O3`).
The original baseline is commit `d29b4df3943f8489c9e51037ba794c903457c332`.
Both versions use fixed endpoints, a 10,000,000-call limit, and the same final gap
`1e-7 + 1e-9 * upper_bound`. “Closed” means that numerical gap was certified,
not exact arithmetic. Time budgets are soft: an in-progress oracle can overrun.

| Workload | Budget/case | Baseline closed | Improved closed | Baseline total time | Improved total time |
|---|---:|---:|---:|---:|---:|
| Development, 60 cases | 3 s | 43 | 55 | 62.78 s | 21.36 s |
| Held-out canonical, 60 cases | 1 s | 32 | 53 | 33.76 s | 13.41 s |
| Held-out reference corpus, 49 cases | 1 s | 29 | 43 | 24.11 s | 8.83 s |
| All held-out, 109 cases | 1 s | 61 | 96 | 57.87 s | 22.24 s |

Every returned path in these comparisons passed independent Shapely validation
at `1e-7`, including endpoints and recomputed length. Held-out bound intervals
were mutually consistent; every case closed by the baseline was also closed by
the improved solver. There are no duplicate case hashes between development and
held-out inputs, or between the two held-out sets. They come from related corpora,
so this is instance holdout, not evidence of generalization to every distribution.
Totals include time-limited runs; their ratios are not uncensored speedup estimates.
These are single sweeps, not repeated statistical measurements.

## Retained changes and experiments

Development ablations below are cumulative unless marked rejected.

| Variant | Closed / 60 | Total time | Decision |
|---|---:|---:|---|
| Original | 43 | 62.78 s | Baseline |
| Repair contacts on variable-length geometric paths | 46 | 54.92 s | Keep |
| Stop oracle refinement once a bound permits pruning | 50 | 43.06 s | Keep |
| Adaptive internal oracle precision; strict feasible-leaf refinement | 53 | 28.46 s | Keep |
| Analytic contact optimization and repeated 2-opt | 53 | 28.41 s | Keep; also supports insertion bounds |
| Full per-child support-dual screening | 53 | 26.54 s | Replace with incremental version |
| Incremental insertion screening | 55 | 25.19 s | Keep |
| Bounding-box rejection in polygon visit checks | 55 | 22.26 s | Keep |
| Alternative root choice, relative to preceding row | 55 | 22.18 s | Reject: negligible benefit |
| Node coverage cache, relative to bounding-box version | 55 | 22.12 s | Reject: negligible benefit |
| Squared-distance visit comparisons, bounding-box version | 55 | 21.36 s | Keep; final candidate |

The original profile spent 56.69 of 62.78 seconds in certified-oracle fallback.
Contact repair now recovers ordered contacts even when the geometric solver omits
pass-through vertices. Slightly displaced candidates are projected back onto
their actual polygons; the independent dual certificate still determines whether
fallback can be avoided. This is important: the geometric solver alone can return
invalid paths on overlapping convex regions, so bypassing certification is unsafe.

The new insertion bound follows from `||d|| >= u·d` for `||u|| <= 1`.
For regions `R_0,...,R_(n-1)`, endpoints `s,t`, and segment vectors `u_0,...,u_n`,
the dual value is

```text
D = (t-s)·u_n + sum_i min_{v in R_i} (v-s)·(u_i-u_(i+1)).
```

Reference contacts need not be feasible. Inserting a region changes only its
support term, adjacent terms, and possibly the endpoint term. Reusing the other
terms makes screening cheap. Zero-length segments receive alternative valid dual
directions; accumulation uses long double and a scale-dependent safety subtraction.
Pruning still uses lower bounds, and coarse feasible leaves are refined to the
original final tolerance. Existing floating-point safety conventions remain;
this is not an interval-arithmetic proof for arbitrary coordinate magnitudes.

The reference paper's cycle-specific convex-hull order arguments were not applied
to fixed-endpoint paths without a corresponding proof. Farthest-missed insertion,
lazy convex decomposition, and the original root selection remain intact.

## Correctness checks

- 86 exhaustive-order cases and 344 interrupted-search checks passed.
- Added analytical overlapping-region oracle/cutoff and contact regressions.
- Added 200 arbitrary reference paths, including coincident and infeasible
  contacts, testing all three insertion bounds against fixed-order solves.
- All 24 independent exhaustive Gurobi comparisons passed; eight exercised
  explicit nonconvex decomposition branches.
- Changed Python scripts compile; scoped whitespace checks pass. The full
  repository sanity campaign was not completed during this work.

## Comparison with the reference solver

The installed native `tspn_bnb2` 0.2.1 implementation was run on the same 60
development cases in path mode, one thread, 3 seconds/case, `eps=1e-9`, feasibility
tolerance `1e-8`. Its total native solver time was 70.84 seconds. Only 5 cases
reported optimality at that requested tolerance; 41 reported relative gaps at
most `1e-6`. Independent validation at `1e-7` accepted only 13 raw paths, or 28
after snapping endpoints. Maximum deviations were about `8.15e-5` at endpoints
and `3.42e-5` from polygons. Some reported gaps were negative.

Consequently, these results do **not** establish universal search superiority:
the reference solver's numerical behavior prevents a clean tight-tolerance
comparison. A common-looser-tolerance rerun and repeated timings remain future
work. Our improvement over our own baseline is the stronger comparison here.
The private comparison adapter loads the same native extension directly to avoid
stalled cloud-offloaded optional Python dependencies; its C++ solver is unchanged.

## Reproduction and artifacts

The new runner rotates solver execution order and writes paths, counters, independent
validation, and executable/input SHA-256 metadata. Results and binaries are local
ignored artifacts, not committed benchmark data:

- `benchmarks/results/free-order-improvements/*-dev.jsonl` and `.meta.json`:
  ablations named as in the experiment sequence (`squared` is final).
- `benchmarks/results/free-order-improvements/heldout-final.jsonl` and metadata:
  final paired held-out comparison.
- `.build/unordered-improvement-baseline/tpp`, `.build/unordered-improved/tpp`:
  frozen comparison binaries; intermediate binaries in `.build/unordered-ablation/`.
- `tspn-comparison/results/free-order-improvements-native-dev/20260910-162456/`:
  successful reference run. Earlier incomplete import attempts are excluded.

Held-out selection: remove development case hashes from `canonical-v1.bin`, then
take every fourth remaining case (60). From the generated reference input
`benchmarks/campaigns/german-instances/inputs/german-instances.bin`, remove both
previous sets' hashes, then select every ninth remaining case, capped at 60 (49).
Case hashes and suite hashes are recorded in the results and metadata.

```bash
.venv/bin/python benchmarks/tpp.py free-order-ablation \
  --suite benchmarks/results/free-order-improvements/heldout.bin \
  --suite benchmarks/results/free-order-improvements/paper-heldout.bin \
  --solver baseline=.build/unordered-improvement-baseline/tpp \
  --solver improved=.build/unordered-improved/tpp \
  --seconds 1 --quiet \
  --output benchmarks/results/free-order-improvements/heldout-rerun.jsonl

tspn-comparison/solver/.venv/bin/python \
  packages/nonconvex-tpp/cpp/tests/validate_unordered_gurobi.py \
  --solver .build/unordered-improved/tpp --cases 24
```

Runner options include `--repeats`, `--absolute-gap`, `--relative-gap`, and
`--oracle-relative-gap`. The old baseline executable does not support the new
gap CLI flags; use unchanged defaults for that paired comparison. Setting the
new internal `--oracle-relative-gap 0` disables coarse oracle precision without
disabling other improvements. No remaining experiments are running.
