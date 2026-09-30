# Completed optimization checkpoint — 2026-09-30

Work remains in `/private/tmp/tpp-convex-cycle`, branch
`codex/convex-cycle-disjoint`. Existing local changes are preserved. The tested
production version is candidate E, native binary SHA-256
`277452400f5c472919e3acbace5ac03a07abb6d5ec252ee5fd04f8c7d70fb3f2`.
The preceding checkpoint is preserved in `CHECKPOINT-2026-09-29.md`.

## Implemented and tested

The general cycle search now checks one boundary vertex of each region before
reconstructing a stationary point on the first region's edge. It caches those
contacts for the full search. Both scalar backends share the implementation;
all candidates retain independent exact certification. The original complete
boundary search remains the fallback. Complexity is unchanged asymptotically.

The general rational API also accepts a default-infinity pruning cutoff. The
cutoff and existing feasible contacts survive rational recovery inside the
double oracle and the B&B adapter. A verified bound reaching the cutoff returns
`CertifiedBound`, never an unsupported claim of optimality. Default rational
solves still seek the exact optimum. No acceptance epsilon or discretization
was added.

Direct reconstruction of inherited features (A) and an extra rotated proposal
pass (B) failed the two-second isolated screen and were removed. Combining B
with the vertex pass (C) worked, but the vertex pass alone (D) was faster. E
adds cutoff propagation. All concrete candidates in this follow-up were tested;
this is not a claim to exhaust all possible research improvements.

## Results

- Captured 19-region relaxation: an uninstrumented v3 control exceeded eight
  seconds; E returned in about 110 ms (median of three), a greater-than-70x
  lower bound on the isolated improvement. Independent rational reconstruction
  also certified the optimum. Rounded contacts can have a weaker dual, so the
  independently proved rational bound is retained.
- Lazy OSM39, 2 s nominal / 8 s process cap: D and E both returned 3/3 validated
  tours, with no process timeout; their gaps remained about 8.51% and 8.50%.
  These are budget-limited feasible solutions, not completed proofs. The cutoff
  change has no substantial isolated speed benefit demonstrated here.
- Selected configuration remains **cache + features + root (CFR)**. On OSM39,
  candidate E closed the requested gap in 3/3 repetitions, median **0.736 s**.
  The earlier v3 median was 0.788 s; do not attribute this small timing change
  to the new algorithm without stronger repeated controls.
- The fresh CFR small6 + large6 screen returned 12/12 valid tours and closed
  9/12 gaps. On the seven cases where both solvers were valid and closed the
  matched gap, ours won 7/7, median speedup **3.686x**. This is a small sample
  stratified by OSM/random/tessellation and size, not the full collection.
- Large OSM50, random59 and tessellation60 remained open. Fekete's OSM39 run
  also stayed open, so it has no matched completed-solve speedup.

The accidental no-switch controls were retained and explicitly distinguished
from CFR; they must not be presented as CFR measurements. Fekete rows are reused
only with matching inputs, settings, repetition keys and binary hashes.

## Verification and handoff

Cycle tests with GMP ON and OFF passed, including the new captured-case
regression, rational cutoff/NaN checks and independent exact certificates.
TSPN tests passed (399 optimization/call-cap checks, 38 concurrency checks and
inherited-dual checks); endpoint tests passed (86 exhaustive-order cases and
344 interrupted-search checks). `git diff --check` passed. Full repository,
dashboard/browser and WASM suites were not rerun.

The TSPN comparison retains the existing numerical contract: target relative
gap 1e-6, native feasibility 1e-8, independent validation 1e-7 and the strict
Gurobi settings in each config. This does not make the entire TSPN B&B rationally
exact. The convex oracle's accepted exact optimum is independently certified.
A cycle call remains cooperative with the outer time budget; arbitrary future
instances can still exceed that budget inside one oracle.

See `oracle-maps-followup/` for captures, ablations, test evidence, source
provenance and analysis, and `runs/*-cfr-followup` plus
`runs/osm39-repeat3-e-cfr-10s` for the maintained-CLI benchmark records.
Do not rerun the completed matrix merely to resume. All long commands and
measurements must continue to run through GPT-6-Luna as the user requested.
