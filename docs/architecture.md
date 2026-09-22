# Repository architecture

This is a research monorepo with one maintained solver path and a small set of
explicitly retained legacy implementations.

## Maintained solver hierarchy

```text
packages/common-geometry
        |
packages/convex-tpp  --->  packages/optimal-convex-partition
        |                                  |
        +------------------------------> packages/nonconvex-tpp
```

The convex package provides the fixed-order convex TPP primitives and the
certified interfaces used by higher-level search. The non-convex package owns
both variants that matter to the project:

- fixed visit order, using convex decomposition and branch-and-bound;
- free visit order, using insertion/decomposition branching and the certified
  convex oracle.

The C++ targets are the source of truth. Python code under the solver packages
is retained for experiments, visualization, and historical comparison; it is
not a second public implementation to extend by default.

## Application lifecycle

`apps/benchmark-dashboard` is the maintained local application. It owns the
campaign workbench, event pages, editor, native solver integration, optional
browser WASM build, and browser regression tests. Its `/editor/offline` route
is the migration target for the old standalone editor: it now provides a
browser-local library, legacy/dashboard JSON compatibility, library export,
local convex decomposition, optional WASM solving, and the WASM-backed
last-step map for fixed-order convex pairwise-disjoint instances. The old
editor remains only while its vertex-line interaction is reviewed for a future
replacement.

The former standalone visualizers were retired after their maintained
capabilities moved to the dashboard. The `/editor/offline` route owns the
current local editing workflow, while
`apps/siicusp34` contains its own frozen fixed-order JavaScript solver copy so
the public event can be published independently. The former
`apps/visualizer-server` was retired because its
solver/WASM path was subsumed by the dashboard. Its account and saved-drawing
features were product experiments with no current caller; their code remains
available in Git history, but is not part of the active architecture.

`apps/siicusp34` is frozen event material. It should change only when a new
reviewed event export is intentionally produced.

## Generated and local boundaries

- `.build*/`, virtual environments, caches, node modules, WASM binaries, and
  benchmark campaign outputs are local/generated.
- `benchmarks/suites/` contains only small or canonical tracked inputs.
- `apps/benchmark-dashboard/static/event/` contains reviewed event snapshots;
  regeneration commands and provenance belong beside the exporter, not in an
  ad hoc result directory.
- Research notes under `docs/research/` preserve historical evidence. They do
  not override the current API or algorithm documentation in `docs/algorithms/`.

## Refactoring rules

1. Add shared geometry or solver behavior to its upstream C++ package.
2. Keep application adapters thin and do not copy solver implementations into
   JavaScript or Python merely for convenience. The SIICUSP copy is an explicit
   exception: it is a frozen publication artifact, tested against its embedded
   teaching instances, and intentionally isolated from production code.
3. Promote a campaign case to a tracked suite only with a stable name,
   provenance, and regeneration/validation instructions.
4. Keep external repositories local and patch them through explicit patch files;
   never commit a vendor snapshot by accident.
5. When a maintained path replaces an app or experiment, remove the active
   source after migrating its unique capabilities and record the decision here.
