> **2026-09-30 update:** Candidate E completed the oracle follow-up. Read [the current checkpoint](CHECKPOINT.md) and [the follow-up campaign](oracle-maps-followup/README.md) first. Earlier v3/paused statements below describe the previous checkpoint; the captured lazy timeout has now been resolved on the measured case.

# Paused after completing the current optimization checkpoint

All planned isolated variants, the complete combination, selected subsets,
three-repetition controls and the portfolio comparison have completed.
No new strategy is scheduled. Read [CHECKPOINT.md](CHECKPOINT.md) for final
results, limitations, reproducible settings and validation status.

Current production source/binary snapshot: v3 (`79bcf69b…5a8d2`), preserved
under `provenance/v3/`; executable remains in ignored `.build/`.
All modifications are local on `codex/convex-cycle-disjoint` in
`/private/tmp/tpp-convex-cycle`. Existing changes are preserved.

Future work, only after explicit resume: broader confirmation on new inputs
or a separate investigation of the lazy exact-map timeout. Completed runs must
not be repeated or overwritten merely because this task resumes.
