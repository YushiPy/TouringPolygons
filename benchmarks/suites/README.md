# Benchmark Suites

This directory contains binary benchmark fixtures for the non-convex TPP code.

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
