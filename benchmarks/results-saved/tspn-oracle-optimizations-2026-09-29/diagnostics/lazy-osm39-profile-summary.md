# Lazy OSM k=39 sample

The frozen v1 native run used the 2-second solver limit and 8-second firm process cap. It timed out without a solver result. A one-second sample taken near 3 seconds showed all 849 sampled frames inside native cycle relaxation and exact intersecting-boundary recovery: 727 frames were in `search_intersecting_boundaries`; 241 reached exact contact solving; 186 reached `DirectionalMaps::query_path` / `locate` / `build_vertex`, with repeated recursion through `virtual_source` and `locate`. This localizes the delay to exact cycle contact/map expansion rather than the outer TSPN branch-and-bound or Fekete.

The v2 and v3 bounded runs also timed out at 8 seconds. The v3 warm-recovery path did not resolve it. Raw diagnostic sampling output is retained under ignored `.build/tspn-oracle-optimization-2026-09-29/diagnostics/`.
