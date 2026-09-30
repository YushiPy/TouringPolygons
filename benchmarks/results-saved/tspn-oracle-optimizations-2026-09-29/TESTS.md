# Build and test record

The v1 build used `/private/tmp/tpp-tspn-comparison-build`, Apple clang 21, and the CMake target’s `-O3` flags. The same native target build was repeated for v2 and v3.

Passed on v3:

- `cmake --build /private/tmp/tpp-tspn-comparison-build --target tpp-unordered tpp-tspn-tests tpp-unordered-tests -j 4`
- `tpp-tspn-tests`: 19 exhaustive cases, interruption/portfolio/concurrency checks, 399 optimization/call-cap comparisons, and 38 combined concurrency checks.
- `tpp-unordered-tests`: 86 exhaustive-order cases, 344 interrupted-search checks, oracle certificate/contact regressions and multi-threaded child evaluation.
- GMP-on and GMP-off `main-cycle_tests`: 96 seeded cases, active-contact regressions, 67 double comparisons and v3 warm-versus-cold overlaps passed.

The cycle certificate suite passed with GMP both enabled and disabled on v2. The v3 edit was confined to cycle recovery; certificate code did not change afterward. CLI loading was checked with `python3 benchmarks/tpp.py tspn-benchmark --help`. No build products, environments, license stderr or binaries are in this results directory.
