# TPP Visualizer Server

This is the server-backed TPP visualizer. The browser handles drawing and sends
solve requests to the FastAPI backend, which calls the C++ TPP implementation.
Convex polygons with intersections are handled by the intersection-aware convex
solver; non-convex polygons are decomposed and solved through Branch and Bound.

## Requirements

- Python 3.12+
- `uv`
- CMake
- CGAL and the C++ dependencies required by the non-convex solver

## Setup

From this directory:

```bash
uv sync
```

## Run

From this directory:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8017
```

Then open:

```text
http://127.0.0.1:8017
```

The first solve request builds the visualizer solver binary if needed. To build
it manually from the repository root:

```bash
cmake --preset nonconvex-release -DTARGET=main-visualizer_solve -DTPP_ENABLE_GUROBI=OFF
cmake --build --preset nonconvex-release
```

## Smoke Test

With the server running:

```bash
curl -fsS http://127.0.0.1:8017/api/tpp/solve \
	-H 'Content-Type: application/json' \
	-d '{"start":[-2,0],"target":[3,0],"polygons":[[[-1,-1],[1,-1],[1,1],[-1,1]],[[0,-1],[2,-1],[2,1],[0,1]]],"maxCalls":200000,"maxSeconds":3}'
```

Expected response shape:

```json
{"path":[[-2.0,0.0],[3.0,0.0]],"exact":true,"calls":1,"seconds":0.0}
```
