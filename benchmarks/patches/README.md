# Third-party compatibility patches

These patches adapt the ignored checkout under `tspn-comparison/solver/` to
the local benchmark protocol. They are source artifacts, not a vendored copy of
the external project.

Before applying one, record the external checkout's upstream URL and commit in
the benchmark provenance. The current local checkout does not contain nested
Git metadata, so its revision must not be guessed. Apply patches in the order
needed by the target revision and record that order with the benchmark result.

- `tspn-bnb2-tpp-oracle.patch` adds the repository's convex oracle adapter.
- `tspn-bnb2-tpp-deadline.patch` adds the remaining-time deadline and counters.
