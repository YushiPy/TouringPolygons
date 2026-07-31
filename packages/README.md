# Packages

Maintained solver and helper code lives here.

- `convex-tpp/`: convex Touring Polygons implementations.
- `nonconvex-tpp/`: non-convex Touring Polygons implementations.
- `fenced-tpp/`: fenced variant, currently legacy/deprioritized.
- `instance-generation/`: instance generation code pending integration.

Each package may keep its own dependency files and build system. Shared geometry code should be extracted only when two maintained packages use the same API and behavior.
