# Completed optimization checkpoint — paused at user request

All six options (five idea families), their complete combination, selected
subsets, and repeated controls have been measured. No next strategy is started.
Changes remain local in `/private/tmp/tpp-convex-cycle` on
`codex/convex-cycle-disjoint`; pre-existing changes are preserved. Options remain
opt-in. Production source is the tested v3 snapshot (`79bcf69b…5a8d2`).

## Current result

`cache + features + root` (CFR) is the selected one-thread configuration for
this sample. The implementation also eliminates repeated rational conversion
and translation within sibling bounds and unused feature-metadata copies.

| OSM39, three repetitions | Median | Range | Valid / requested gap closed |
|---|---:|---:|---:|
| Frozen baseline, one thread | 4.696 s | 4.676–4.725 s | 3/3 |
| CFR, one thread | 0.788 s | 0.781–0.788 s | 3/3 |
| Cache + features portfolio, two threads | 0.957 s | 0.929–0.967 s | 3/3 |

CFR is approximately 5.96 times faster than our frozen baseline on OSM39.
Fekete returned in 2.540–2.602 s but did not close the requested gap, so this
case has no matched solved-instance speedup against Fekete.

For the six small cases, three repetitions each: CFR returned 18/18 valid tours
and closed all requested gaps. Fekete also returned valid tours but closed the
gap in only five cases. On those five matched cases CFR wins 5/5, with median
speedup **5.793746x** against Fekete. Per-class speedup medians are OSM 7.67x
(two matched cases), random 4.68x (two), tessellation 1.60x (one). This small,
partly millisecond-scale sample is not evidence for the full collection.

The root heuristic is not universally beneficial: small tessellation10 took
4.453 ms with CFR versus 2.759 ms in the frozen baseline and 1.616 ms with
cache + features alone. On large tessellation60 the two-second screen remained
open: baseline gap 15.6661%, CFR 12.75%, Fekete 7.88%. The two-thread portfolio
was slower than standalone CFR on OSM39 and had weaker open-case bounds in its
large screen. These limitations remain visible in the raw results.

## Correctness, scope, and unsuccessful variants

The convex oracle still checks candidates with its exact certificate. No new
numerical acceptance epsilon or discretization was added. TSPN comparison uses
the existing numerical B&B contract: target relative gap 1e-6, native feasibility
1e-8, independent validation 1e-7, and the strict Gurobi settings recorded in each
config. A gap-closed TSPN run is not a claim of rational zero-error optimality.

TSPN tests passed, including 399 optimization/call-cap comparisons and 38
combined concurrency checks. Endpoint tests (86 exhaustive cases), convex cycle
suites with GMP on/off, and added warm/cold comparisons passed. Certificate
suites passed before the unchanged certificate implementation was reused.
Full dashboard/WASM and repository-wide sanity suites were not run.

Dual reuse and stronger branching did not repay their extra work in this
sample. Enabling everything regressed. Lazy evaluation hit the firm eight-second
process cap on OSM39 without returning a tour. A bounded native stack sample
localized that time to exact intersection-map recovery (`DirectionalMaps`
expansion). Trying the retained contacts before the cold rational constructor
preserves certification but did not eliminate the timeout or demonstrate a
separate measurable improvement; it must not be advertised as fixing it.

## Reproduction and next checkpoint

Use the maintained CLI with the tested configuration:

```sh
tpp-unordered --cycle --cycle-optimization cache \
  --cycle-optimization features --cycle-optimization root
```

Benchmark through `python3 benchmarks/tpp.py tspn-benchmark`; exact inputs,
commands/settings, raw results, objective intervals and hashes are in `runs/`.
OSM39 repetitions use 10-second nominal / 15-second process limits; the broad
screen uses 2/8 seconds. Reused Fekete rows are explicitly identified and checked
against input/settings/binary hashes. `provenance/v2` and `provenance/v3` preserve
source snapshots; executable files remain in ignored `.build/` storage.

Upon an explicit resume, first read this checkpoint and STATUS.md. Do not rerun
completed experiments. Any new strategy or deeper lazy/oracle investigation is
future work. Keep timed commands and builds delegated to GPT-6-Luna.
