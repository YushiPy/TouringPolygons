> **2026-09-30 update:** Candidate E completed the oracle follow-up. Read [the current checkpoint](CHECKPOINT.md) and [the follow-up campaign](oracle-maps-followup/README.md) first. Earlier v3/paused statements below describe the previous checkpoint; the captured lazy timeout has now been resolved on the measured case.

# TSPN optimization checkpoint — 2026-09-29

Completed and paused at the user's request. See [CHECKPOINT.md](CHECKPOINT.md)
for the results, scope of exactness, limitations, test status and reproduction.

The selected `cache + features + root` configuration achieved about 5.96x over
our frozen baseline on OSM39 (three repetitions). On five small cases where
both solvers closed the requested gap, its median speedup over Fekete was
5.793746x, with 5/5 wins. This small sample does not establish collection-wide
performance; root and lazy regressions remain documented.

Inputs and selection metadata are in `inputs/`; raw data, configs, intervals,
per-class summaries and repetition timings are in `runs/`. Source/build
snapshots are in `provenance/`; executables remain in ignored `.build/`.
[STATUS.md](STATUS.md) and [RESUME.md](RESUME.md) identify the stopping point.
The initial plan and full variant matrix are retained alongside this report.
