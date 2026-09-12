# Benchmark Suites

This directory contains binary benchmark fixtures for the non-convex TPP code.

## Promoting a campaign case

Campaigns are local and ignored. If one case becomes a regression fixture,
export only the selected instance into the appropriate suite here, give it a
stable descriptive name, and record its origin and conversion command in the
commit or this file. Do not add campaign metadata, preview images, run history,
or private experiment notes.

- `nonconvex/test_cases.bin`: tracked source corpus converted from a TSPN JSON
  archive. The source instances are cycles, so the converter uses each instance
  bounding box lower-left corner as `start` and upper-right corner as `target`.
- `nonconvex/custom_tests.bin`: small tracked hand-made/debug fixture.
- `algorithm-dev-v1.bin`: ignored generated development suite, regenerated from
  `nonconvex/test_cases.bin`.
- `canonical-v1.bin`: ignored generated benchmark suite, regenerated from
  `nonconvex/test_cases.bin`.

Regenerate the source corpus:

```bash
python3 benchmarks/scripts/convert_instances.py
```

Regenerate the derived development and canonical suites:

```bash
python3 benchmarks/tpp.py generate-suites
```
