# Free-order TPP oracle comparison

Date: 2026-09-10

## What was implemented

The original German `tspn-bnb2` checkout remains unchanged at
`tspn-comparison/solver`. The working integration is at
`tspn-comparison/solver-oracle` and accepts an `oracle_backend` toggle:

- `socp`: the original Gurobi reduced-SOCP oracle.
- `tpp`: the specialized certified convex TPP oracle from this repository.

The adapter normalizes active regions to CCW convex hulls. If the certified
refinement encounters its specific invalid-domain failure, that logical call
falls back to SOCP and increments `tpp_socp_fallback_calls`. Other errors are
not swallowed.

Because `tspn-comparison` is intentionally ignored, the portable integration
is backed up in
`benchmarks/patches/tspn-bnb2-tpp-oracle.patch`. It applies cleanly to the
untouched checkout with:

```sh
patch -p1 -d tspn-comparison/solver < benchmarks/patches/tspn-bnb2-tpp-oracle.patch
```

The tracked benchmark drivers are:

- `benchmarks/scripts/tspn_run_comparison.py`: free-order whole-solver run.
- `benchmarks/scripts/tspn_oracle_backends.py`: identical fixed-sequence oracle comparison.
- `benchmarks/scripts/free_order_metamorphic.py`: translation, rotation,
  reflection, scale, input-order, and vertex-order invariance checks.
- `benchmarks/scripts/analyze_oracle_swap.py`: matched summary by instance hash.

## Whole-solver experiment

All 60 rows use the same encoded instances, one thread, a 3-second limit,
`eps=1e-9`, independent feasibility tolerance `1e-7`, and are matched by
SHA-256.

| Solver | Proven optimal | Independently valid | Total time | Median time | Total calls | Median calls |
|---|---:|---:|---:|---:|---:|---:|
| Standalone TPP | 55/60 | 60/60 | 21.357 s | 0.003983 s | 648,898 | 110.5 |
| German search + SOCP | 5/60 | 13/60 | 70.839 s | 0.249273 s | 128,669 | 577.0 |
| German search + TPP | 43/60 | 51/60 | 74.458 s | 0.250638 s | 619,251 | 207.5 |

Paired outcomes:

- Standalone TPP was faster than both German variants on 60/60 instances.
- Within the German search, TPP was faster on 38/60 instances.
- Within the German search, TPP used fewer calls on 38/60 instances.
- The hybrid required 2 SOCP fallback calls out of 619,251 oracle calls, both
  on instance 57.

The equal aggregate runtime of the two German variants does not mean equal
progress. Both saturate the 3-second budget on hard cases. With the TPP oracle,
the German search proves 43 cases instead of 5 and performs much more search
work. The time limit is soft because an in-progress native oracle call cannot
be interrupted, so a few rows exceed three seconds.

Oracle-call totals are comparable only between the two German-search rows.
Replacing the oracle changes bounds, incumbents, and therefore the search tree,
so it can change both the number of calls and the cost of each call. Standalone
TPP calls are from a different algorithm and are not identical work units.

## Identical-sequence oracle experiment

The isolated experiment used the same original-order convex-hull relaxation
for both backends on 30 instances, with three alternating-order repetitions,
for 90 paired runs.

| Metric | SOCP | TPP |
|---|---:|---:|
| Valid paths at `1e-7` | 21/90 | 90/90 |
| Median call time | 1.178 ms | 0.127 ms |
| Total time | 0.120 s | 15.734 s |

TPP was faster on 63/90 paired runs and 21/30 instance medians. The median of
the paired SOCP/TPP speed ratios was 11.57x. However, nine instances triggered
expensive certified refinement, including one with a 3.47-second median. Those
outliers dominate total time, so the current oracle is faster typically but
not in total on this sample. This is the clearest remaining performance target.

The largest observed objective difference was 0.00908, or about
`7.15e-7` relative. Most SOCP paths failed the strict independent geometry
check, so objective disagreements involving those paths are not evidence that
TPP is less accurate.

## German regression fixtures

The German repository ships 54 small regression instances in
`tests/instances/optimal_solutions.json.xz`. They are closed-tour TSPN
instances, while the specialized TPP oracle solves fixed-endpoint paths. Calling
the path oracle directly for those cycles would be mathematically incorrect.

For a compatible experiment, each published reference cycle was cut at its
first trajectory point, and that point was used as both the start and end depot.
All original polygons were preserved. If the published cycle is optimal, this
anchored path has exactly the same optimum: it is a restriction of the cycle
problem, and the published cycle itself is feasible for the restriction.

Results on all 54 adapted native geometries, with one thread, `eps=1e-9`, and a
3-second limit:

| Solver | Proven at `1e-9` | Valid at `1e-7` | Total time | Median time | Total calls | Median calls |
|---|---:|---:|---:|---:|---:|---:|
| Standalone TPP | 54/54 | 54/54 | 0.005 s | 0.041 ms | 382 | 4.5 |
| German search + SOCP | 4/54 | 18/54 | 0.432 s | 5.440 ms | 1,628 | 22.0 |
| German search + TPP | 54/54 | 54/54 | 0.012 s | 0.163 ms | 875 | 9.0 |

The TPP-backed German search was faster on 54/54 cases and used fewer calls on
38/54. Standalone TPP was faster than German + SOCP on 54/54 and faster than
German + TPP on 51/54; the other three comparisons are tens of microseconds and
dominated by harness overhead.

The German SOCP run completed quickly, but only four results met the deliberately
strict `1e-9` proof criterion. Every SOCP relative gap was below `4.0e-7`, so all
54 meet the repository's native `1e-3` criterion. Its path violations were also
numerical: 18/54 raw paths pass at `1e-7`, 37/54 after endpoint snapping, and
54/54 pass at `1e-4`. TPP passed 54/54 without snapping at `1e-7`; its maximum
polygon distance was `1.55e-10`.

On an additional 162-pair identical-order oracle microbenchmark using these
geometries, TPP was faster in 158/162 runs. Median oracle time was 0.0255 ms for
TPP versus 0.2369 ms for SOCP, a median paired ratio of 9.04x. Total time was
0.0070 s versus 0.0417 s. TPP produced 162/162 valid paths at `1e-7`; SOCP
produced 24/162. All objective differences were below `4.0e-7` relative.

These fixtures are not the full experimental corpus. The full corrected
comparison follows.

## German experimental corpus, 558 instances

The repository's evaluation scripts use
`instances/instances_socg_simplified.zip`, containing 558 instances with 4 to
60 polygons. This is the simplified corpus described in their evaluation
README: holes are removed, convex-hull fill is applied, and supersites are
removed. The non-simplified sibling archive cannot be represented exactly by
the current TPP input format when polygons contain holes.

These instances are also cycles. To compare endpoint TPP implementations, the
converter retains every simplified polygon and adds deterministic exterior
depots at opposite corners of the geometry bounding box, with a 10% margin.
These are therefore the Germans' actual benchmark geometries, but the objective
is an adapted free-order endpoint path, not their original closed-tour
objective.

All variants used four concurrent instance processes, one solver thread per
instance, a three-second soft limit, `eps=1e-9`, solver feasibility tolerance
`1e-8`, and independent validation tolerance `1e-7`.

| Solver | Proven at `1e-9` | Valid at `1e-7` | Sum of solver times | Median time | Total calls | Median calls |
|---|---:|---:|---:|---:|---:|---:|
| Standalone TPP | 525/558 | 558/558 | 187.125 s | 0.00516 s | 5,281,888 | 155.5 |
| German search + SOCP | 49/558 | 107/558 | 640.335 s | 0.26132 s | 1,354,407 on 557 cases | 725.0 |
| German search + TPP | 393/558 | 504/558 | 666.541 s | 0.09293 s | 5,221,455 | 277.0 |

The 504 valid TPP rows are all rows with an incumbent; the remaining 54 timed
out without one. There were no invalid TPP incumbents. SOCP returned 557
incumbents, of which 107 passed the strict `1e-7` endpoint-and-polygon check;
all 557 pass at the repository's native `1e-3` tolerance. One SOCP run failed
when Gurobi reported model status 3 (infeasible) for a reduced relaxation. TPP
solved that same case without an error.

Paired results:

- Standalone TPP was faster than German + SOCP on 556/558 cases.
- Standalone TPP was faster than German + TPP on 556/558 cases.
- Within the German search, TPP was faster on 361/558 cases.
- Within the German search, TPP used fewer calls on 341/557 recorded pairs.
- The integrated TPP backend used SOCP fallback on only 2 of 5,221,455 calls.

| Polygon count | Cases | Standalone proofs | German + SOCP proofs | German + TPP proofs |
|---:|---:|---:|---:|---:|
| 4–10 | 128 | 128 | 5 | 128 |
| 11–20 | 136 | 136 | 13 | 131 |
| 21–30 | 75 | 74 | 16 | 63 |
| 31–40 | 74 | 73 | 13 | 51 |
| 41–50 | 75 | 71 | 2 | 16 |
| 51–60 | 70 | 43 | 0 | 4 |

The TPP oracle does more total work in the German search because it enables
that search to explore far more nodes before the same limit. Therefore total
calls and summed time are not per-call speed measurements. Median instance time
improves, proofs increase eightfold, and median calls fall, but certified
fallback outliers can make a soft-limited call run past three seconds. The
worst observed row took 25.97 seconds.

### Deadline-aware certified refinement follow-up

On 2026-09-11, the certified oracle was made aware of the caller's remaining
time budget. Long-double and extended-precision refinement now stop safely at
the deadline while retaining the best feasible upper bound and certified lower
bound already obtained. The standalone solver passes its global remaining time
directly. The German integration aligns the oracle deadline with its
branch-and-bound timer.

| German search + TPP | Baseline | Deadline-aware |
|---|---:|---:|
| Proven at `1e-9` | 393/558 | **394/558** |
| Valid at `1e-7` | 504/558 | **504/558** |
| Sum of solver times | 666.541 s | **566.862 s** |
| Median time | 0.09293 s | **0.08904 s** |
| Maximum time | 25.969 s | **3.068 s** |
| Runs over 4 seconds | 20 | **0** |

No proof, valid incumbent, or meaningfully better incumbent from the baseline
was lost. Three timeout incumbents improved and one additional proof completed;
these small search-quality changes may be affected by parallel scheduling. The
tail reduction is direct and repeatable: rerunning the 12 worst baseline cases
reduced their total from 100.65 seconds to 36.03 seconds, with every case at
approximately the requested three-second limit.

The standalone rerun proved 526/558 instead of 525/558, validated all 558
paths, and reduced summed time from 187.125 to 184.582 seconds. Its baseline
already respected the limit closely, so these small differences should not be
treated as a robust speed claim.

### Bidirectional incumbent experiment

The standalone nearest-neighbor/contact/2-opt initializer was also run from
both endpoints, retaining the better initial path. Proof count remained
526/558, all paths remained valid, and summed time was unchanged within noise
(184.58 s versus 184.60 s). Total calls fell by 1.62%, from 5,374,437 to
5,287,246. Four timeout incumbents improved by 1.13% to 1.45%, while four
worsened by 0.10% to 0.89% because the new upper bound changed the subsequent
time-limited search trajectory.

The reverse initializer reproduced the German search's better incumbent on
cases 65 and 556 to numerical precision. This supports endpoint direction as
one component of the German incumbent advantage, but the heuristic is disabled
by default until both search trajectories can be combined without sacrificing
their respective incumbents. It can be enabled with
`tpp-unordered --bidirectional-initial`.

## Correctness checks

- C++ exhaustive suite: 86 exhaustive-order cases and 344 interrupted-search
  checks passed, including oracle certificate/contact regressions.
- Metamorphic suite: 60/60 transformed runs were exact, independently valid,
  and objective-consistent.
- Final 60-case hybrid run: no integration errors and 51 independently valid
  incumbents. The remaining nine are time-limit rows without a validated
  incumbent, not invalid reported solutions.

## Reproduction commands

Build and install the integrated checkout into the comparison environment:

```sh
cd tspn-comparison/solver-oracle
UV_CACHE_DIR=/tmp/tpp-oracle-uv-cache uv pip install --python ../solver/.venv/bin/python . --no-deps
cd ../..
```

Generate the anchored suite from the German repository's corpus:

```sh
python3 benchmarks/scripts/convert_tspn_native_instances.py \
	--input tspn-comparison/solver/tests/instances/optimal_solutions.json.xz \
	--output tspn-comparison/instances/german-native-anchored-v1.bin
```

Generate the full 558-instance adapted path suite used in the corrected run:

```sh
python3 benchmarks/scripts/convert_tspn_native_instances.py \
	--input tspn-comparison/solver/instances/instances_socg_simplified.zip \
	--output tspn-comparison/instances/socg-simplified-bbox-path-v1.bin \
	--depot-strategy bbox-corners
```

Use `--workers 4` with `tspn_run_comparison.py` and
`unordered_benchmark.py` to reproduce the parallel campaign. The external
runner writes every completed case immediately and can continue a partial CSV
with `--resume PATH`; the standalone runner uses `--resume` with the same JSONL
output path.

Run either whole-solver backend by changing `--oracle-backend`:

```sh
tspn-comparison/solver/.venv/bin/python benchmarks/scripts/tspn_run_comparison.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin \
	--tspn-repo tspn-comparison/solver-oracle \
	--output tspn-comparison/results/oracle-swap \
	--mode path --time-limit 3 --threads 1 \
	--eps 1e-9 --feasibility-tolerance 1e-8 \
	--validation-tolerance 1e-7 \
	--oracle-backend tpp --oracle-tolerance 1e-7
```

Run the isolated paired oracle benchmark:

```sh
tspn-comparison/solver/.venv/bin/python benchmarks/scripts/tspn_oracle_backends.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin \
	--tspn-repo tspn-comparison/solver-oracle \
	--output tspn-comparison/results/oracle-backends.jsonl \
	--repeat 3 --limit 30 \
	--validation-tolerance 1e-7 --oracle-tolerance 1e-7
```

Add `--strict` when using the paired command as a regression gate. Without it,
invalid paths and objective disagreements are reported as experimental outcomes;
native exceptions still make the command fail.

Run the metamorphic checks:

```sh
tspn-comparison/solver/.venv/bin/python benchmarks/scripts/free_order_metamorphic.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin \
	--solver .build/unordered/tpp \
	--output benchmarks/results/free-order-improvements/metamorphic.jsonl \
	--limit 20 --seconds 3 --tolerance 1e-7
```

## Conclusion

The evidence supports that the standalone free-order algorithm is substantially
better, not merely that its oracle is faster. On the corrected German corpus it
proves 525/558 cases and validates all 558 returned paths. The oracle swap alone
improves the German search from 49 to 393 proofs, confirming that the specialized
oracle is a major contributor. Still, “strictly faster oracle on every input”
would be too strong because certified-refinement outliers remain.

Before making a publication-level claim, repeat the full comparison on a frozen
held-out suite, use several wall-time seeds/orderings, and report censored
time-limit statistics. Deadline-aware refinement removed the previously
observed multi-second certification overruns.
