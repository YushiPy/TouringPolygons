# Status — bounded campaign complete

All requested builds, targeted tests, screens, and the selected interval repeats are complete. No build or benchmark process is running. The frozen binaries live only under ignored `.build/`; no binary or generated build data is in this saved campaign.

- Candidate V1 hash: `6e9724b988fd314c16e17a53a90f87b239e924a1d8419c6300311ec940b1ffcb`.
- Candidate V2 (interned shared-bound index) hash: `be321d7befc4839bd54f76f18acafacc50e57e2b0bd7483ae7e36c608ad28f2d`.
- V1 and V2 passed the relevant GMP-on tests; final V2 source also passed all four affected GMP-off suites. See `provenance/*/build-tests.txt` and `provenance/test-outcomes-gmp-off.md`.
- 17 raw run directories contain 96 valid native rows and 96 verified Fekete-reference rows reused through the CLI. There were 60 closed proofs, 36 valid tours stopped at the time limit, and no native process timeouts.
- Repeated CFR+interval improved 9/9 matched closed large6 runs (median paired speedup 1.263x); repeated OSM39 portfolio CFR+memo+interval improved 3/3 matched closed runs (median paired speedup 1.481x). Dual-screen and share-bounds showed no supported aggregate wall-time benefit and remain opt-in.

To reproduce calculations, run `python3 provenance/summarize-results.py` and `python3 provenance/audit-objectives.py` from this campaign directory or the repository root using their repository-relative paths; `provenance/campaign-commands.json` records the CLI argv for all 17 runs, with verified reference paths and frozen-binary hashes. The first generates `provenance/campaign-summary.json`; the second generates `provenance/objective-interval-audit.json`. Candidate binaries can be rebuilt from `campaign-plan.json`'s source commit, each `tracked-source.patch`, and its `source-untracked/cycle_interval.h` overlay. Exact per-run settings are retained in `config.json`; `provenance/campaign-commands.json` reconstructs the public CLI invocations. Dashboard browser tests, WASM checks, and the full repository sanity script were not run.
