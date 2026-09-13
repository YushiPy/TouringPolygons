# Intersecting convex TPP audit fixtures

`Wrong1.bin` contains only the `Wrong1` geometry exported from
`benchmarks/campaigns/Cool Instances/manual-cases.json`, in the existing convex
TPP test format (little-endian doubles and 64-bit counts). The original mixed
vertex orientations are deliberately preserved. No background image, map state,
other campaign case, or run history is included.

The stored reference route has five points. Let `s` be the start, `t` the target,
`h = polygons[0][0].y`, `b = polygons[1][3]`, and `c = polygons[2][1]`.
Reflect `b` across `y=h` to get `r`. Intersect `s→r` with `y=h` to get `a`.
The reference is `[s,a,b,c,t]`, with length `1.6729790817113108`.
The audit independently verifies a support-function dual certificate for this
route; it does not simply trust a solver's stored answer.

From the repository root:

```bash
cmake -S packages/convex-tpp/cpp -B .build/intersection-audit \
  -DTARGET=main-intersection_audit -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-audit -j 4
.build/intersection-audit/tpp-convex --proof-only
.build/intersection-audit/tpp-convex --random 40 --bench 5000
```

The default audit is a **failing production regression suite**, not an expected
failure test: it exits 1 while the reported solver bugs remain. `--proof-only`
is a separate passing check of the independent validator, the exact mathematical
counterexamples, the Wrong1 reference certificate, and the edge-contact
continuity oracles. It does not certify the production solver.

`--random` uses the existing certified convex API only as a test oracle. This
API independently checks support-function bounds and can use an interior-point
method; unresolved gaps are counted separately. It is never installed as a
fallback by this audit. The deterministic counterexamples have explicit
certificates and do not depend on that oracle. Random inputs use seed 20260913;
bit-for-bit reproduction is tied to the recorded AppleClang/libc++ toolchain.

`--bench` measures the unchanged public binary API on a seeded disjoint instance
(20 polygons, 8 vertices each) and on the known-incorrect intersecting example.
The latter timing does not measure a correct intersection algorithm.

See [the detailed mathematical audit](../../../docs/algorithms/intersecting-tpp-audit.md)
for exact inputs, proofs, limitations, source references, results, and remaining
implementation work. The two-polygon and ordered-visitation counterexamples are
also constructed directly by `main-intersection_audit.cpp`.
