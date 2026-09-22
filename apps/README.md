# Apps

The maintained user-facing applications are:

- `benchmark-dashboard/`: local FastAPI workbench for campaigns, editing,
  benchmarking, solver comparison, free-order reports, event pages, the
  optional browser WASM solver, and the new `/editor/offline` editor.
- `siicusp34/`: frozen, self-contained static event page and data package. Its
  small fixed-order convex solver copy lives beside the page so the event does
  not depend on another application.

The former `visualizer-server/` was removed after its solver and WASM build
path were consolidated into `benchmark-dashboard`. Its login and saved-drawing
features were not used by the research workflow; the Git history retains the
implementation if that product direction is revived.

Application code is kept separate from solver packages because its browser,
server, and presentation dependencies are different from the C++ libraries.
