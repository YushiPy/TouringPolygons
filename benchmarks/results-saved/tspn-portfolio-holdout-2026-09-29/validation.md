# Native validation record

The focused native checks completed successfully for the measured C++ state. No dashboard or WASM tests were run; the change is confined to native solvers, tests and benchmark reporting.

- GMP ON cycle suite passed after updating the double-bound regression to permit the retained independently solved rational lower bound. The intersecting-cycle regressions included 96 seeded cases; all 5 active-contact rational/double relaxations passed.
- GMP ON cycle-certificate suite passed: 1,040 zero-link ray pairs, 269 optimal.
- GMP OFF fallback cycle and certificate suites both passed.
- TSPN suite passed: 19 exhaustive cases, 152 interrupted searches, 38 full portfolio solves, 152 shared-call-cap portfolio runs, 1 decomposition case, 2 concurrent oracle batches and 240 arbitrary-hint dual checks.
- Endpoint unordered suite passed: 86 exhaustive-order cases, 344 interrupted-search checks, multi-threaded child evaluation, and the added endpoint portfolio regression for shared and unshared modes with call caps 0, 1 and 3.

The portfolio implementation uses two native search workers while Fekete uses one Gurobi thread. The portfolio cap tests check aggregate calls against the single global cap, preserve certified bounds and validate endpoint paths.
