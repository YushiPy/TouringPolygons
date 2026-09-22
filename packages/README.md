# Packages

The maintained path is the C++ package graph:

```text
common-geometry -> convex-tpp -> optimal-convex-partition -> nonconvex-tpp
```

- `common-geometry/`: shared vectors and low-level geometry primitives.
- `convex-tpp/`: maintained convex TPP APIs, including fixed-order primitives
  and certified interfaces; its Python directory is legacy/prototyping code.
- `optimal-convex-partition/`: shared CGAL-backed decomposition library.
- `nonconvex-tpp/`: maintained fixed-order and free-order C++ solvers, plus
  legacy Python implementations retained for comparison and historical use.
- `fenced-tpp/`: legacy fenced-TPP research code; it is not part of the main
  build graph.
- `instance-generation/`: reproducible input generation helpers.

Each package may keep package-local dependencies and tests. New maintained code
must use upstream geometry/solver targets rather than copying implementations.
Scratch experiments and generated matrices belong in ignored local directories.

Keep package directories limited to maintained source, package-local tests, and
intentional regression fixtures. Scratch files, alternate historical versions,
and generated benchmark matrices belong in ignored local directories.
