# TSPN SOCG stratified screening — 2026-09-28

**Follow-up on 2026-09-29:** the pending oracle patch subsequently passed the
GMP-ON/OFF cycle and certificate suites. Native B&B and the new cooperative
portfolio were also tested; the previously reserved holdout was measured in
[the portfolio campaign](../tspn-portfolio-holdout-2026-09-29/README.md).
The historical measurements below remain unchanged.

**Status at the end of 2026-09-28: investigation incomplete.** The measured candidate is GMP + certified
cutoff + inherited contacts. A subsequent segment fix passed native tests but
was measured on only one case. The final early rational recovery / retained
lower-bound patch has **not been compiled or tested**: the delegated GPT-Luna
agent exhausted its account usage allowance. No performance claim below
includes that final patch. The updated benchmark runner also awaits execution.

## Formulation and selection

TSPN on closed polygon regions, arbitrary cyclic visiting order, free endpoints,
possible overlap and nonconvexity. Compare the maintained C++ B&B with the
pinned Fekete B&B using its original SOCP backend. This is not a comparison of
fixed-order convex subroutines alone.

The supplied simplified ZIP contains **558 entries**: 320 OSM, 160 random and
78 tessellation. Comparing original and simplified WKT sets identifies 360
changed instances (201 OSM + 159 random), not an archive containing only 360.
69 cases change polygon count. The simplified inputs have no holes. Class
comes from supplier metadata; size comes from the actual simplified geometry.
For example, the selected random n050 file has 48 polygons.

`selection.json` fixes 12 cases, one per class and band 5–10, 11–20, 21–40,
41–60, before measurement. Lexicographic selection puts all four OSM cases in
Bangalore: this is a diagnostic sample, not an unbiased estimate of the whole
corpus. `holdout/` fixes six previously unmeasured cases in the first two
bands using a seeded filename hash and excluding Bangalore. **No holdout
measurement is available yet.** It must not be selected again after timing.

## Measurement contract

- One repetition per screening case, sequential solvers, one thread each.
- Two-second native B&B budget; 12-second process cap. A single expensive
  relaxation can overrun the native budget. Process timeouts are censored.
- Matched stopping criterion `UB <= (1 + 1e-6) LB`; our relative parameter is
  `1e-6 / (1 + 1e-6)` with absolute gap zero.
- B&B feasibility tolerance 1e-8; independent tour validation 1e-7. Strict
  Gurobi settings and spanning tolerance are recorded in each `config.json`.
- These are numerical B&B stopping / validation settings. The convex rational
  solver and independent certificates use exact predicates, no optimization
  epsilon or spatial discretization. Returned binary64 bounds round outward.
- Fekete bounds are numerical, not exact certificates. A valid tour alone is
  not an optimum. Time comparisons between two completed runs count as solved
  speedups only when both close the matched gap and pass tour validation.

## Measured results

| Class | Actual k | Baseline ours ms | GMP + cutoff + seed ours ms | Fekete ms in that stage | Gap closed ours/Fekete |
|---|---:|---:|---:|---:|---:|
| OSM | 5 | 5.259 | 1.689 | 2.841 | yes / yes |
| OSM | 15 | 28.852 | 12.791 | 28.089 | yes / yes |
| OSM | 30 | 238.116 | 102.139 | 298.517 | yes / no |
| OSM | 50 | 2002.085 | 2001.281 | 2026.291 | no / no |
| random | 5 | 2.539 | 1.322 | 3.530 | yes / yes |
| random | 15 | 20.835 | 11.503 | 26.666 | yes / no |
| random | 30 | 3565.695 | 2282.042 | 550.968 | no / no |
| random | 48 | censored at 12 s | censored at 12 s | 2010.546 | no / no |
| tessellation | 10 | 1877.206 | 1156.674 | 9.076 | yes / yes |
| tessellation | 14 | 19.574 | 10.711 | 4.865 | yes / yes |
| tessellation | 25 | 2057.748 | 1404.775 | 209.732 | yes / no |
| tessellation | 48 | 4357.920 | 2887.479 | 959.555 | no / no |

Ours closes 7/12 gaps in baseline and 8/12 after GMP + cutoff + seed; Fekete
closes 5/12. All 11 returned native tours and all 12 Fekete tours pass validation
at each full screening stage. Reported objective intervals do not separate on
any paired completed case; this is consistency evidence, not an independent
exactness proof of Fekete's bounds.

Among the **five mutually gap-closed cases**, ours wins 3/5 after the measured
changes, with median speedup about **1.68×**. The earlier 5.11× / 492-of-550
endpoint-TPP target is **not achieved** by this TSPN sample. OSM-30 improves
2.33× relative to our baseline; it is not a matched solved speedup against
Fekete because that Fekete run does not close the requested gap. Large open
cases must not be included in solved speedup averages.

The segment fix alone changes tessellation-10 from 1.157 s to 1.097 s in one
measurement, while Fekete changes from 9 to 18 ms. That comparison is too noisy
to claim a small speedup; the major tessellation bottleneck remains.

## Diagnosis and pending patch

`gmp/diagnostics/tessellation10-cycle-profile.json` profiles extracted,
normalized convex hull relaxations. Despite its directory, this diagnostic
used the **older cpp_rational binary**, SHA256
`23622ef83b616b490d1f8a7e7b41cfb6f8f8097f14d8254e07fdabd6dec93324`,
not GMP. It supports a structural diagnosis, not a latest-build speedup.

For sequence `[0,4,2,3,9,8]`, double took 0.750 s, evaluated 2,354 candidates,
and used 776 rational recoveries (73 anchor, 702 feature, one complete cycle).
The rational solve took 0.0153 s with ten candidates. The independently proven
rational lower bound was also lost when contacts were rounded, yielding a much
weaker double lower bound and causing another full rational solve in B&B.

The pending patch invokes complete rational recovery after the bounded initial
feature proposal fails, and keeps the already certified lower bound across
contact rounding. Its upper bound still comes from independently verified
feasible exported contacts. A new regression fixture checks recovery counts
and retained bounds. No new acceptance tolerance is introduced. These changes
are currently **unvalidated**, and no speedup is asserted for them.

## Validation and continuation

Before that final patch, GPT-Luna ran the native cycle, certificate, TSPN and
endpoint B&B suites successfully. Coverage includes 19 exhaustive TSPN cases,
152 interrupted runs, 240 arbitrary dual checks, parallel batches, 86 exhaustive
endpoint cases, 344 interrupted endpoint runs, and 1,040 certificate ray pairs.
The segment revision passed cycle/TSPN/endpoint suites. The GMP-OFF cycle and
certificate suites passed at the earlier GMP-only revision; they need rerunning
against the final changes. Global sanity, dashboard and WASM tests were not run.

On this machine, run from a terminal:

```bash
bash /private/tmp/tpp-convex-cycle/benchmarks/results-saved/tspn-socg-stratified-2026-09-28/reproduce.sh
```

The script is campaign reproduction, not a new benchmark CLI. It rebuilds and
runs GMP-ON/OFF cycle and certificate tests, TSPN and endpoint tests, then calls
`benchmarks/tpp.py tspn-benchmark` for the 12-case screening and six-case holdout
(three repetitions). Tests fail fast. Benchmarks retain censored runs and
report them. It reuses this machine's configured builds and frozen Fekete
binary, requires existing dependencies, and saves logs/results/source diff in
a fresh ignored `.build/tspn-socg-2026-09-28.*` directory. It may take several
minutes; it has **not yet been executed**. Share the printed output path when
resuming. The new runner applies the same 12-second cap to each repetition of
either backend and reports aggregates in `strata.json`.

## Artifact provenance

Each measured stage contains inputs, raw runs, configuration, per-case summary
and analysis. The baseline binary was frozen at `b70cda3`; vendor commit and
ZIP/binary hashes are in `provenance.json`. Historical `config.json` fields named
`source_sha256` are workspace snapshots at measurement, **not a guarantee that
those files built a supplied frozen binary**. Binary hashes identify what was
executed; the baseline's source identity is its recorded commit. Intermediate
candidate hashes describe uncommitted revisions and do not by themselves
reconstruct their full source trees. New runner configs distinguish workspace
hashes explicitly and record only build commands actually executed. The final
source patch and current run hashes will be captured by `reproduce.sh`.
