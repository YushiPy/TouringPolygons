
## Tests directory

This directory contains the handwritten test cases for the TPP for convex
polygons C++ implementation. The generated stress suites are intentionally not
tracked; generate and run them with:

```bash
packages/convex-tpp/cpp/run_generated_tests.sh
```

The script writes generated `.bin` files under `.build/convex-generated-tests/`
and runs the verifier against both the tracked handwritten fixtures and the
generated fixtures.

The ordered-cycle optimality certificate has focused tests in
`src/main-cycle_certificate_tests.cpp`. Configure that executable with
`-DTARGET=main-cycle_certificate_tests`, build the `tpp-convex` target, then run
the resulting executable. The tests cover exact small cycles, invalid inputs
and candidates, primal-dual bounds, and the saved Gurobi references described
in `benchmarks/results-saved/convex-cycle-gurobi-reference-2026-09-25/`.

## Test case format

Every file in the `tests` directory with a `.bin` extension contains one or more test cases. Each test case is structured as follows:


- `16` bytes: Start point (`2 doubles`)
- `16` bytes: Target point (`2 doubles`)
- `8` bytes: Number of polygons (`size_t`)
- For each polygon:
	- `8` bytes: Number of vertices (`size_t`)
	- For each vertex:
		- `16` bytes: Vertex position (`2 doubles`)
- `8 `bytes: Number of points in the solution (size_t)
- For each point in the solution:
	- `16` bytes: Point position (`2 doubles`)

The file may contains multiple test cases, one after the other, following the same format. The number of test cases in the file can be determined by reading until the end of the file.

## Intersecting-polygon audit

The ordinary intersection target now checks ordered visitation along complete
segments. `validate_ordered_path` is a separate feasibility validator; unlike
`is_valid_solution`, it does not impose the legacy disjoint local-bend rules.
The legacy validator also runs this ordered feasibility check before its local
checks, preventing premature success when later polygons have not been visited.

A separate `main-intersection_audit` target contains exact paper counterexamples,
Wrong1 in both input orientations, metamorphic checks, a contact-continuity
family, and optional random/oracle comparisons. The corrected directional maps
now pass these production checks. Additional closed-degeneracy, continuity,
workspace, and randomized tests are in `main-directional_tests`. See
[fixture instructions](../../../../benchmarks/suites/intersection-audit/README.md)
and [the correction report](../../../../docs/algorithms/intersecting-tpp-correction.md).

## Floating ordered-cycle solvers

`src/main-cycle_tests.cpp` exercises both scalar instantiations of the cycle
solver and requires an exact support certificate for every rational result.
It also audits the strict double constructor (recovery disabled) against all
four saved Gurobi instances: an arithmetic failure must remain explicit and
must not be labeled optimal. Objective agreement is required for the default
double backend with counted rational recovery. Tests compare lengths and certified intervals,
not Gurobi contact coordinates. Configure with `-DTARGET=main-cycle_tests`;
the campaign path is supplied by CMake, so the executable works from any cwd.

Intersection coverage includes common points, containment, zero-length links,
an orthic cycle whose anchor endpoints are nonsmooth, cyclic shifts, and 96
seeded cases compared with the unaccelerated exact boundary search. Selected
fixtures also force pure-double boundary construction. The certificate suite
checks 1,040 independent analytical ray-pair cases and variants with longer
zero blocks. Rational scales beyond the binary64 exponent range are exercised.

See [`docs/algorithms/convex-cycle.md`](../../../../docs/algorithms/convex-cycle.md)
for the API, exactness proof, complexity and floating-point limitations.
