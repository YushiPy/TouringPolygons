# Exact ordered cycles through closed convex polygons

The maintained C++ APIs in `tpp/convex/cycle.h`, exported by `tpp_convex.h`, are:

- `tpp_convex_solve_cycle`: rational construction and exact rational contacts;
- `tpp_convex_solve_cycle_double`: the same geometric proposal and boundary
  reduction with binary64 constructions;
- the existing `_disjoint` variants: preserve their stricter input contract.

They minimize `sum_i |q_(i+1)-q_i|` with one contact in each polygon in the
supplied cyclic order. Endpoints are free; the closing link is included and
the cycle may cross itself. The general APIs accept touching, overlap and
containment. Inputs need at least two positive-area convex polygons, in either
winding, optionally with redundant collinear/closing vertices. There is no
optimization epsilon or spatial discretization.

## Existing core and shared arithmetic

Dror et al., *Touring a Sequence of Polygons*, STOC 2003, gives fixed-source
last-step maps; its `s=t` case still fixes that source. The local source is
[`TPP-Dror/main.tex`](../bibliography/TPP-Dror/main.tex). The floating-cycle
reduction below is additional to that fixed-source result.

The solver uses the existing disjoint reflection recurrence, factored from
`rational_disjoint.cpp` into the scalar template `disjoint_arithmetic.h`.
Intersecting anchored paths use the existing corrected directional maps in
`intersecting_maps.cpp`, with exact rational input/output added to the same
implementation. Their correction and contracts are described in
[the fixed-source report](intersecting-tpp-correction.md). The binary64 map is
compiled from that same source. No second optimizer or SOCP dependency was
introduced into production.

`CycleRefinement<Scalar>` and `search_intersecting_boundaries<Scalar>` are
shared by rational and double. Input validation and the independent certificate
remain exact in both modes. Rational input never needs a floating construction.

Native CMake builds use GMP's unbounded integers and rationals through the
same Boost.Multiprecision interface when GMP is available. The aliases
`ConvexInteger` and `ConvexRational` select the representation; the geometric
algorithm, support predicates and outward-rounded bounds are unchanged.
`-DTPP_ENABLE_GMP_RATIONAL=OFF`, missing GMP, and WebAssembly retain the
header-only `cpp_int`/`cpp_rational` backend. This changes arithmetic bit costs,
not exactness or the number of geometric operations. The CMake target exports
the backend definition and dependency to consumers: rebuild the library and
its consumers together because the exact-coordinate types are part of the ABI.

## Fast construction by active contacts

For two disjoint regions the optimum is twice their distance. A closest pair
can be chosen with at least one polygon vertex, including when the supporting
edges are parallel. The shared arithmetic first enumerates vertex/segment
projections in both directions and submits the closest pair to the same
certificate. This takes O(mn) arithmetic operations for polygon sizes m and n,
with constant auxiliary storage. It avoids boundary search on nearly parallel
edges. Intersecting regions still use their common point for a zero cycle.

Start with polygon centroids. A coordinate update minimizes the two incident
links over its polygon: use a crossing segment when possible, otherwise test
vertex support and reflected edge contacts. Adjacent coincident contacts can
move together over the intersection of their regions.
That intersection may be a point, a segment, or a positive-area polygon.
For a segment the shared coordinate update also clips by its two endpoints;
the two edge halfplanes alone describe an unbounded line. This allows a block
on a common tessellation edge to slide jointly instead of being trapped by
separate contact updates.

The resulting vertex/edge/straight-through contacts suggest a combinatorial
path type. If it has fixed vertex contacts, unfold edge reflections between
successive fixed contacts and intersect the resulting straight segment with
each contact edge. With no fixed vertex, choose an anchor edge `x=a+t*e`;
affine unfolding gives a length `|(A-I)x+c|`. Its squared length is quadratic
in `t`, so its stationary parameter is obtained rationally. Restoring skipped
straight-through contacts preserves the visit order.

A contact strictly between its neighbours is straight-through even if it lies
on the polygon boundary: the two unit directions cancel, so that boundary
must not introduce a reflection. When unfolding places a contact beyond its
edge, the proposal pins it to the blocking endpoint and reconstructs the
remaining chain. Each such pivot adds a vertex pin, hence at most `k` pivots
occur in one sweep. A constructed chain feeds the next coordinate sweep so
that restored contacts which now bend can update their active features.
These are finite feature proposals, shared by both arithmetic types; only the
independent certificate accepts an optimum. Failure still uses the complete
boundary reduction. No displacement, convergence epsilon or discretization
is introduced.

Coincident vertex contacts can trap separate coordinate updates. An additional
proposal releases them onto their incident edges facing the neighboring
contacts, then solves those reflection equations together. This changes only
the proposed feature tuple, with no positional perturbation or epsilon.

Every complete proposal is checked with
`tpp_convex_verify_cycle_certificate`; a wrong active type cannot authorize
`Optimal`. The proposal phase tries at most `k+1` feature sweeps, stopping
sooner on a repeated feature tuple. This structural work limit only hands
control to the anchored search; it never accepts an objective or claims
convergence. Set `refine_contacts=false` to exercise the unaccelerated general
boundary reduction. Tests compare both paths.

In the filtered double mode, an exhausted initial feature proposal invokes the
complete rational reduction before starting double boundary probes. Recovery
starts from the best already-verified double contacts (or the supplied initial
hint if no candidate was retained), imported as exact binary rationals. If
that warm proposal fails, the usual centroid proposal is still attempted before
entering the complete boundary maps. Both are finite construction passes whose
candidates use the same exact certificate. This avoids discarding useful
contacts and avoids repeating rational repairs at hundreds of anchors when the same exact
reduction can solve the entire cycle. Disabling arithmetic recovery preserves
the double boundary path. If the rational recovery fails, boundary search
still runs; the proposal limit never authorizes acceptance.

When an independently verified rational solution or dual bound is exported to binary64, its
outward-rounded lower bound remains valid for the original input. The double
result keeps the larger of that bound and the bound obtained by independently
verifying the exported contacts. Its upper bound still comes from those
feasible exported contacts. This matters when rounding splits coincident
contacts and weakens the contact-derived dual. It does not promote the rounded
contacts to `Optimal`: they may still return `FloatingPointLimit`, or
`CertifiedBound` when the proven lower bound reaches the caller's cutoff.

Both arithmetic APIs accept optional `initial_contacts` as a proposal for this
same constructor. Their coordinates supply neither a bound nor a feasibility
assumption. A wrong proposal still falls back to the complete reduction. The
general rational and double APIs also accept `lower_bound_cutoff` for B&B callers. Only a
feasible candidate with an independently certified lower bound at least that
cutoff can return `CertifiedBound`. This status proves the requested bound,
not optimality or an arithmetic limit. The default cutoff is infinity, so the
standalone solver's optimality contract is unchanged. No tolerance is added.

The cutoff is passed through rational recovery as well as the initial double
construction. A rational recovery may therefore return a certified pruning
bound before finding its optimum. The double result retains this global lower
bound when exporting independently feasible contacts, just as it retains an
exact optimum's lower bound. A bound is never promoted to an optimality claim.

The opt-in `bound_first` experiment also checks supplied initial contacts before
constructing another candidate when a cheap floating support estimate suggests
that their exact certificate might reach the cutoff. The estimate only schedules
an additional check; it is never used to accept, prune, or report a bound.
It requests the verifier's filtered early dual-bound
test before KKT. Sufficient bounds return `CertifiedBound`; insufficient bounds
continue through the full certificate and the ordinary construction. General
rational recovery receives the same option. The legacy rational disjoint
fallback still solves its full optimum. `certificate_cutoff_skips` counts
certificates that avoided KKT, while the double API additionally reports
`initial_contact_checks` and `initial_contact_accepts`. These counters measure
actual certificate work; a skipped KKT test alone does not imply a speedup,
because the early dual and extra inherited-contact check also have a cost.

## Repeated relaxations and prepared certificates

`ConvexCycleCertificateGeometry` owns an immutable, independently validated
copy of its regions. The verifier overload accepting this object still checks
contact membership and the complete support certificate on every invocation.
Only input conversion and boundary validation are reused. The ordinary APIs
remain available for independent verification against the original input.

The general double solver accepts a per-worker `ConvexCycleWorkspace` through
its options. It caches canonical rational polygons by their complete binary64
coordinates, never by pointer identity. Mutation, changed winding, and polygon
reordering therefore cannot reuse the wrong geometry. Lookup costs at most
O(m log U) scalar comparisons for an m-vertex polygon among U cached polygons;
assembling an ordered problem still copies O(V) rational coordinates. Repeated
candidate checks avoid those conversions and validation passes, while retaining
the certificate's predicate cost. Storage grows with distinct polygon contents;
`clear()` releases it. B&B workspaces live for one instance and one worker.

`initial_features` is a checked proposal for the same shared refinement kernel.
Feature ids refer to the canonical boundary: vertex `2*j`, edge `2*j+1`,
straight-through `-1`, changed contact `-2`. Before a full sweep, it updates the
changed contacts and their neighbors and attempts to reconstruct the inherited
feature tuple. Out-of-range hints are ignored; incorrect tuples never bypass
the verifier. The ordinary constructor and complete fallback remain available.
This adds one bounded proposal, not a numerical acceptance threshold.
The result exports feature metadata only with `retain_active_features=true`;
ordinary solves avoid allocating/copying metadata that their caller will discard.

`tpp_convex_cycle_dual_bound` independently checks rational vector norms and
computes a downward-rounded support bound. No vector longer than one is
accepted. `tpp_convex_cycle_dual_directions` normalizes nonzero links with rational
upper enclosures of their norms and can retain inherited unit-disk vectors at
zero links. These witnesses prove lower bounds; they are not a replacement for
the complete zero-link KKT certificate and need not be optimal dual witnesses.

## General boundary reduction and correctness

First intersect all regions. A common point gives an optimal zero cycle.
Otherwise a positive optimal cycle has a contact on some polygon boundary.
Indeed, if every contact were interior, group consecutive coincident contacts.
Stationarity under translating each such block forces its incoming and outgoing
unit directions to agree. This would make all nonzero links point in the same
direction, which cannot close a positive cycle.

It is therefore sufficient to scan all polygon boundaries. A containing
polygon need not have an optimal boundary contact; choosing only that polygon
would be incorrect. Split each edge at intersections with all other polygon
boundaries, including collinear overlap endpoints.

For an anchor `x`, solve the remaining visits with fixed `s=t=x`. A prefix or
suffix of visits whose polygons contain `x` may be removed and restored with
contacts at `x`: removing constraints lower-bounds the anchored problem, and
these zero-cost restored visits attain that bound. The returned fixed-source
path must pass a certificate with the anchor region replaced by the singleton
`{x}` before its search direction is used by the rational solver.

Before edge bracketing, the shared search evaluates the first vertex of every
polygon and attempts the same finite contact reconstruction. It stores these
anchored contacts for reuse during the full scan. This changes only the order
of already required boundary queries: a difficult rational stationary parameter
on the first polygon no longer prevents an easy certificate at another polygon
from being tried first. If the preliminary queries fail, every split edge and
its original reconstruction search remain available. Every returned optimum
still passes the global certificate; a pruning bound needs only the independent
global dual certificate, not a derivative or anchored optimality claim.

Let `F(x)` be the anchored optimum and `f(t)=F(a+t*e)` on one split segment.
Partial minimization of a jointly convex sum of norms makes `F` convex and
2-Lipschitz. Membership of every anchor point in other regions is constant on
the open segment; hence its removed prefix and suffix are fixed there. The
first and last retained regions do not contain `x`, so the incident retained
links are nonzero. Their directions are unique among anchored optima: equality
in convexity of a Euclidean norm forces equal directions for any nonzero links
of two minimizing paths. Consequently `f` is differentiable in the open
segment, with

```
f'(t) = (u_in - u_out) dot e.
```

Here `u_in,u_out` skip the zero-cost suffix/prefix. At segment endpoints the
same expression is a valid subgradient: assign their repeated zero links those
same unit directions, making every removed visit's support coefficient zero.
Internal zero blocks use the complete disk/cone certificate.

Each endpoint is first tested globally, then against the restricted problem
whose anchor region is just this closed segment. The complete certificate
handles its nonsmooth endpoints as well. If either endpoint minimizes the
restriction, no interior search is needed. Otherwise convexity forces every
left endpoint subgradient to be negative and every right endpoint subgradient
to be positive. Exact bisection and simplest-rational reconstruction find an
interior stationary point. Only the full cyclic certificate can accept it as
a global polygon-constrained optimum; another boundary is examined if it
fails the inward support condition.

For disjoint polygons the chosen smallest polygon alone suffices. An interior
optimal contact would lie straight between its two exterior neighbors and
could be slid to its boundary without changing length. The original disjoint
reduction and its more economical fixed-source kernel remain available.

## Exact termination

There are finitely many combinatorial types of anchored paths. Intersection
vertices are rational, as are original vertices and edge reflections. On an
interval of one type, unfolding has one of two forms:

1. If a path has a fixed bend vertex, its variable contribution is the sum of
   distances from the anchor to two fixed rational images, plus a constant.
   The stationary anchor on its line is a rational reflection/intersection or
   perpendicular projection.
2. With only edge reflections, its variable length is `|(A-I)x+c|` for rational
   `A,c`. Its squared length is quadratic in the edge parameter; an isolated
   stationary parameter is rational, or the expression is constant.

The derivative in an open arrangement segment is continuous by the common
incident-direction argument. An isolated minimum at a type transition is
therefore stationary for an adjacent expression and rational as well. A flat
minimum contains a rational minimizer. Endpoint minima were already handled
by the restricted certificates.

For a minimizing rational parameter with denominator `Q`, a bracket narrower
than `1/Q^2` contains no other rational with denominator at most `Q`.
Continued fractions return the simplest rational in the bracket, so bisection
followed by that reconstruction reaches a minimizer in `O(1+log Q)` probes.
Bracket width is never an acceptance test. With rational arithmetic the
boundary reduction thus terminates, under the existing exact fixed-source
map contract. `OracleFailure` reports a violated construction/certification
contract; intersections are no longer skipped because of an incomplete zero
witness search or a nonsmooth anchor endpoint.

Feasibility plus the complete cyclic support certificate proves global
optimality, as detailed in [the certificate proof](convex-cycle-certificate.md).
The result stores rational coordinates and squared link lengths. Length itself
may be irrational, and is exactly represented as their sum of square roots.

## Complexity

Let `N` be the total input vertex count, `k` the polygon count, `A` the number
of split anchor segments, and `H=1+log Q` bound the rational reconstruction
height of a minimizer for a searched segment. Convexity/winding validation is
`O(N)`. Direct boundary splitting/separation checks cost `O(N^2)` operations.
Convexity gives `A=O(kN)` even though the implementation checks pairs of edges.

For the maintained intersecting fixed-source map, use the conservative bound
`C=O(N^2+k^2*N*log(kN))` and `O(kN)` stored values. The complete cycle
certificate costs `V=O(N+k^3)` operations, or `O(N)` without zero links.
With refinement disabled, the general reduction therefore costs

```
O(N^2 + A*H*(C+V+H)) arithmetic operations,
O(kN+H) stored arithmetic values.
```

The optional proposal uses `O(N)` geometric operations per sweep without zero
blocks, at most `O(k^2)` for endpoint pivots, and at most `O(N^2)` for block
intersections. Including its certificates,
its `k+1` sweeps add at most `R=O(k*(N^2+V))` per invocation. It is called before
the search and at searched contacts, so replacing `C+V+H` by `C+V+H+R` gives a
conservative bound for the accelerated implementation. The intended practical
benefit is obtaining a certificate before most anchored solves are needed.

The preliminary vertex pass adds at most `k*(C+V+R)` operations and `O(k^2)`
stored contact values, already covered by the bounds above because `A>=k`,
`H>=1`, and `N>=k`. It changes practical ordering, not worst-case complexity.

For the disjoint fallback, with `m` vertices on the smallest polygon and
`C_d=O(k*N*log(N/k))`, the boundary part remains
`O(N^2+m*H*(C_d+N+H))`, plus the bounded initial proposal. These count exact
arithmetic operations, not constant-time bit costs: reflections grow rational
numerators/denominators, and reporting bounds use integer square roots.

## Double behavior and limitations

The double mode uses the same construction and general search, with no epsilon
for feasibility, derivatives, or stopping. It can stop at a floating stationary
parameter or when no representable midpoint remains. Those are arithmetic
limits, not exact optimality claims. Only the exact independent certificate
authorizes `Optimal`; `FloatingPointLimit` carries a feasible incumbent and its
measured global bounds. A failed constructor can report `OracleFailure`.

`recover_arithmetic_failures` defaults to true. A proposed feature tuple can
be reconstructed with rational reflections; a failed fixed-source anchor can
be recomputed rationally; finally the general cycle can be recovered using the
same rational search. Separate counters expose all three recoveries. Set the
option false for double candidate construction without those recoveries.
Validation, certification and outward-rounding repair still use exact predicates.
When recovery is enabled, an exact common-region check handles a zero cycle
whose common point cannot be represented in binary64 before the anchor search;
this is counted as a cycle recovery.

Double cannot promise the same robustness as rationals for ill-conditioned
or unrepresentable contacts. An outward-rounded contact is moved inward by an
exactly computed displacement derived from `nextafter`, then rechecked. A thin
region may force a much larger representable displacement, which the reported
certificate exposes. Overflow/underflow can also defeat a double construction.
`FloatingPointLimit` does not promise a chosen error tolerance. The rational
API retains its contacts and objective exactly even beyond binary64's exponent
and coordinate-resolution ranges.

## Tests and measurement

Build `main-cycle_tests` and `main-cycle_certificate_tests` through the existing
CMake `TARGET` option. The cycle tests cover 2–5 visits, crossing cycles,
non-dyadic contacts, all-edge optima, closed intersections and containment,
cyclic/reversal variants, 96 seeded intersecting cases, and rational scales
`2^-1100` and `2^1100`. Accelerated and unaccelerated rational searches are
compared; selected intersecting cases also force the unaccelerated pure-double
search. The zero-link certificate has independent analytical ray-pair tests.

`python3 benchmarks/tpp.py cycle-benchmark --output NEW_DIRECTORY` compares
whole API calls with warmed, single-thread Gurobi SOCP calls, including model
construction for Gurobi and validation/certification for C++. Solver order
rotates, and every campaign saves inputs, individual repetitions, configuration,
source hashes and analysis. Compare objective values and independently certified
intervals, not contact coordinates or Gurobi's numerical bound as an exact bound.
Finite synthetic benchmarks do not establish universal speed dominance.
