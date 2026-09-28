# Ordered convex-cycle SOCP references

This is a small reference campaign for the disjoint ordered-cycle problem.
Each record contains one contact in each polygon, the Gurobi objective and
best bound, solve status, and the objective recomputed from the returned
contacts. The input sequence is cyclic: its final link closes from the last
contact to the first.

The instances are full-dimensional, counterclockwise convex polygons with
pairwise positive separation. They are intentionally small and fixed so the
future C++ solver can compare objective values against an independent solver.
The returned contact coordinates are numerical references, not exact
certificates. Compare objectives with a documented tolerance; a different
optimal contact tuple can be equally valid.

## Model and provenance

`generate_references.cpp` adapts the second-order-cone model used by the
`third_party/tspn-socg` SOCP oracle. For each polygon it creates a point
constrained by the polygon's edge halfplanes. For each cyclic link it adds a
Gurobi Euclidean norm constraint and minimizes the sum of link lengths. There
are no binary variables because the visit order is fixed.

The model was run with Gurobi Optimizer 13.0.3 through the installed C++ API.
Its inputs, settings, source, raw output, and short analysis are kept together
in this directory. No license file or license key is included.

## Reproduction

From the repository root on macOS, with Gurobi 13 installed and a valid local
license:

```sh
clang++ -std=c++20 \
  -I/Library/gurobi1303/macos_universal2/include \
  benchmarks/results-saved/convex-cycle-gurobi-reference-2026-09-25/generate_references.cpp \
  -L/Library/gurobi1303/macos_universal2/lib \
  -Wl,-rpath,/Library/gurobi1303/macos_universal2/lib \
  -lgurobi_c++ -lgurobi130 \
  -o /private/tmp/convex-cycle-gurobi-reference
/private/tmp/convex-cycle-gurobi-reference \
  > benchmarks/results-saved/convex-cycle-gurobi-reference-2026-09-25/raw.json
```

The run configuration is recorded in `config.json`, inputs in `instances.json`,
the captured Gurobi output in `raw.json`, and the results summary in
`analysis.md`. `feasible_contacts` in the raw output are the Gurobi contacts
contracted toward each polygon's centroid by the factor in `config.json`. This
small adjustment makes the floating-point contacts exactly feasible under the
independent binary-rational checker; their lengths are still close to the
Gurobi objective.
