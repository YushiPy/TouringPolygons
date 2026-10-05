# Packages

The maintained path is the C++ package graph:

```text
common-geometry -> convex-tpp -> optimal-convex-partition -> nonconvex-tpp
```

- `common-geometry/`: shared vectors and low-level geometry primitives.
- `convex-tpp/`: maintained convex TPP APIs, including fixed-order primitives
  and certified interfaces.
- `optimal-convex-partition/`: shared CGAL-backed decomposition library.
- `nonconvex-tpp/`: maintained fixed-order and free-order C++ solvers.
- `fenced-tpp/`: legacy fenced-TPP research code (Python); it is not part of
  the main build graph.

Benchmark input generation lives in `benchmarks/_internal/` (`gen_instances.py`,
`generate_benchmark_matrix.py`); see `docs/benchmarks/instance-generation.md`.

Every `cpp/src/main-*.cpp` is also a named CMake target (`tpp-…` in
`nonconvex-tpp`, `tpp-convex-…` in `convex-tpp`). Build them with
`python3 benchmarks/tpp.py build TOOL`, which uses one shared `.build/tools`
directory. The legacy `-DTARGET=main-foo` mode still works.

Each package may keep package-local dependencies and tests. New maintained code
must use upstream geometry/solver targets rather than copying implementations.
Scratch experiments and generated matrices belong in ignored local directories.

Keep package directories limited to maintained source, package-local tests, and
intentional regression fixtures. Scratch files, alternate historical versions,
and generated benchmark matrices belong in ignored local directories.
