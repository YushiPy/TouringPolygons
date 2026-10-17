# Packages

The maintained path is the C++ package graph:

```text
common-geometry -> convex-tpp -> optimal-convex-partition -> nonconvex-tpp
```

- `common-geometry/`: shared vectors and low-level geometry primitives.
- `convex-tpp/`: maintained convex TPP APIs, including fixed-order primitives
  and certified interfaces.
- `optimal-convex-partition/`: shared optimal convex decomposition, a CGAL-free
  port of CGAL's `optimal_convex_partition_2` (CGAL only for its parity test).
- `nonconvex-tpp/`: maintained fixed-order and free-order C++ solvers.

Benchmark input generation lives in `benchmarks/_internal/` (`gen_instances.py`,
`generate_benchmark_matrix.py`); see `docs/benchmarks/instance-generation.md`.

Every `cpp/src/main-*.cpp` is also a named CMake target (`tpp-…` in
`nonconvex-tpp`, `tpp-convex-…` in `convex-tpp`). Build them with
`python3 benchmarks/tpp.py build TOOL`, which uses one shared `.build/tools`
directory. The legacy `-DTARGET=main-foo` mode still works.
The former package-local `run.sh` wrappers were removed; use the shared build
command from the repository root.

Keep package directories limited to maintained source, package-local dependencies
and tests, and intentional regression fixtures. New maintained code must use
upstream geometry/solver targets rather than copying implementations. Scratch
experiments and generated matrices belong in ignored local directories;
superseded implementations remain in Git history.
