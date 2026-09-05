# Packages

Maintained solver and helper code lives here.

- `convex-tpp/`: convex Touring Polygons implementations.
- `common-geometry/`: shared C++ vector and low-level geometry primitives.
- `nonconvex-tpp/`: non-convex Touring Polygons implementations.
- `fenced-tpp/`: fenced variant, currently legacy/deprioritized.
- `instance-generation/`: instance generation code pending integration.

Each package may keep its own dependency files and build system. Maintained C++
packages should depend on `common-geometry` for shared geometry and avoid local
copies of solver implementations.

Keep package directories limited to maintained source, package-local tests, and
intentional regression fixtures. Scratch files, alternate historical versions,
and generated benchmark matrices belong in ignored local directories.
