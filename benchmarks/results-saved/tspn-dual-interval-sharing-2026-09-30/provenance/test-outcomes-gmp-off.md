# GMP-off test outcomes

Configuration: Release, `TPP_ENABLE_GMP_RATIONAL=OFF` (Boost cpp_rational fallback), AppleClang 21.0.0, macOS arm64. The nonconvex target was incrementally rebuilt after V2; cycle and certificate executables were rebuilt with GMP off from the same source (V2 does not change convex sources).

- `tpp-tspn-tests`: passed; 19 exhaustive cases, 798 option/call-cap comparisons, 38 combined concurrency checks.
- `tpp-unordered-tests`: passed; 86 exhaustive-order cases, 344 interrupted-search checks, concurrent child evaluation.
- `main-cycle_tests`: passed; prepared geometry/dual feasibility, 96 seeded cases, active-contact and boundary recovery tests, 67 double comparisons.
- `main-cycle_certificate_tests`: passed; 1,040 zero-link ray pairs, 269 optimal.

The initial nonconvex CMake build emitted existing designated-initializer order warnings in unordered.cpp. All affected targets and suites passed. Full repository sanity, dashboard browser tests, and WASM intersection tests were not run for this targeted campaign.
