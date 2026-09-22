# External repositories and comparison code

`tspn-comparison/` and `paula-tspn/` are local checkouts of other people's
repositories. They are useful inputs to experiments, but they are deliberately
not part of this repository's tracked source. The root `.gitignore` protects
against accidentally committing their files.

The current arrangement is intentional:

- keep the checkouts at their existing paths because benchmark scripts and local
  workflows refer to them;
- preserve the original remote, commit, and local modifications in each
  checkout's own Git metadata;
- put compatibility changes that belong to this project in
  `benchmarks/patches/`, with a clear target revision and application order;
- record the external revision and patch set in benchmark metadata whenever a
  result is meant to be reproducible.

Do not replace these directories with a fresh download during a refactor: local
compatibility edits and cached instances are part of the working experiment.
Git submodules are also not a good default here because the comparison code is
actively adapted locally. A submodule or vendored snapshot should be considered
only if a future release needs a frozen, redistributable dependency and the
license/provenance have been reviewed.

## Current checkout audit

The working tree currently contains the two ignored directories, but neither
contains nested Git metadata. Their exact upstream revisions therefore cannot be
proved from this repository. Treat benchmark results that depend on them as
machine-local until a revision is recorded in the checkout itself or in the
benchmark metadata.

Before recording a new comparison result, capture:

1. the upstream URL and commit for the external solver or instance collection;
2. the local compatibility diff, saved as a patch under `benchmarks/patches/`;
3. the patch application order and any required build command;
4. the resulting checkout path and revision in the result's provenance file.

The existing patches target the adapted solver under `tspn-comparison/solver/`.
They are intentionally not applied automatically because the local checkout has
no pinned revision to validate against.
