# Exact ordered cycles through closed convex polygons

The maintained C++ APIs in `tpp/convex/cycle.h`, exported by `tpp_convex.h`, are:

- `tpp_convex_solve_cycle`: rational construction and exact rational contacts;
- `tpp_convex_solve_cycle_double`: the same geometric proposal and boundary
  reduction with binary64 constructions;
- the existing `_disjoint` variants: preserve their stricter input contract.

They minimize `sum_i |q_(i+1)-q_i|` with one contact in each polygon in the
supplied cyclic order. Endpoints are free; the closing link is included and
the cycle may cross itself. The general APIs accept touching, overlap and
containment. Inputs need at least two closed convex regions: points (one
vertex), segments (two vertices) or positive-area convex polygons in either
winding, optionally with redundant collinear/closing vertices. Points and
segments never take the disjoint fast path. The common-region clip bounds a
segment by its supporting line **and** its endpoints and tests a point by
membership (their edge halfplanes alone would describe a line or the plane);
the anchored search uses the same membership, and a point anchor is its own
only anchor. The `_disjoint` variants keep their positive-area contract. There is no
optimization epsilon or spatial discretization. A caller that only needs bounds
within a gap can request it with `ConvexCycleDoubleOptions::max_gap`; the
[binary64 stage](#binary64-stage-with-a-requested-gap-2026-10-08) then proves
them without rational arithmetic and returns `GapClosed`, never `Optimal`.

[`convex-cycle-roteiro.md`](convex-cycle-roteiro.md) (in Portuguese) orders the
results below as a study map, from the certificate to exact termination, with
the code that implements each one.

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

In binary64 the clipped intersection can lose part of a shared wall. Two
houses whose wall has the same binary64 endpoints in both can clip to the
wall's far corner alone (seen in `kuala_lumpur_n060_seed3530`). Moving the
block there lengthens the cycle and pins it, and separate coordinate updates
cannot leave that corner afterwards. The double construction therefore keeps
a block in place when the move would lengthen its two incident links. This
binary64 comparison only selects the proposal. In exact arithmetic the
block's point lies in the common region, so the move never lengthens the
cycle and the rational construction is unchanged.

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
independent certificate accepts an optimum. The rational pass tries the closed
reflection candidate before certifying its intermediate coordinate-sweep
contacts: the latter can have much larger denominators. Both proposals remain
available if the first fails; this ordering is not a convergence criterion.
One immutable prepared geometry is reused across rational candidate checks,
while each candidate still receives independent membership and support checks.
Failure still uses the complete boundary reduction. No displacement,
convergence epsilon or discretization is introduced.

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

The binary64 options also expose `proposal_only`, a separate finite-work
request used by the B&B adapter. It stops after the floating feature proposals,
without rational reconstruction or boundary search. `ProposalLimit` reports
that this requested phase ended; it does not assert optimality, convergence,
or that a feasible candidate was found. Every retained candidate and bound
still has a completed independent certificate. `Optimal` and `CertifiedBound`
keep their exact tests, and default standalone solves keep the complete path.
No tolerance is supplied to the constructor. The caller can use the certified
interval or request the complete solve when it needs a stronger bound.

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

`ConvexCycleDoubleOptions::interval_certificate` enables rigorous binary64
interval filters inside `tpp_convex_verify_cycle_certificate`. Input membership
and every inconclusive pruning cutoff retain rational fallback; `Optimal`
still requires the exact KKT predicate. The rational solver is unchanged.
Prepared geometry can cache an immutable binary64 view of canonical vertices.
`certificate_interval_uses` counts candidates whose reporting bounds used the
filter. See [the interval proof and environment guards](convex-cycle-certificate.md#optional-rigorous-binary64-interval-filter).

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

## Binary64 stage with a requested gap (2026-10-08)

`tpp_convex_solve_cycle_float_certified` (`tpp/convex/float_oracle.h`,
`solvers/cycle_float_oracle.cpp`) is the cycle version of the fixed-endpoint
floating-point oracle ([certified-convex-oracle.md](certified-convex-oracle.md#oráculo-em-ponto-flutuante-experimental-2026-10-08)).
It proves `L <= OPT <= U` on the input regions with directed rounding in
binary64 and never uses rational arithmetic. It needs `max_gap > 0` or a
finite cutoff, and it closes only when a cycle of proved contacts exists and
`U-L <= max_gap` (directed subtraction) or `L >= cutoff`. Algebraic optimality
and zero gaps stay with the exact path.

`tpp_convex_solve_cycle_double` runs it first when `max_gap > 0`. A closed
stage returns `GapClosed`, or `CertifiedBound` when `L` reaches the cutoff,
with `certificate.status=Feasible`, `optimality_check_skipped=true`, no exact
KKT test, no rational preparation and no recovery. Otherwise the existing
exact path runs unchanged, and the result keeps the larger of its lower bound
and the stage's (both bound the same optimum; `Optimal` is never promoted or
demoted). An interruption during the stage returns `Interrupted` with the
stage's proved bounds. `float_interval_closed`, `float_polish_closed`,
`float_polish_attempted`, `float_candidates`, `float_newton_iterations` and
the exclusive `timings.interval_proof_seconds` and `timings.polish_seconds`
describe the stage. The default `max_gap=0` keeps every previous contract.

### Steps

1. **Normalization.** Each region is normalized in binary64 to the set that
   `prepare_cycle_polygons` describes: consecutive and closing duplicates
   removed; one vertex is a point, two a segment; otherwise every turn
   `orient(p_(i-1),p_i,p_(i+1))` is decided exactly (an interval that excludes
   zero, else the 128-bit integer determinant `dyadic_orientation`), all
   turns must have one sign (reversed if negative), the boundary must wind
   once, collinear vertices must not backtrack (interval dot product) and are
   removed. An undecided sign or an invalid region returns `Unsupported`,
   and the exact path then reports the input as before.
2. **Proposals.** A common point of all regions (binary64 clipping, the zero
   cycle); every candidate of the shared construction kernel
   `CycleRefinement<double>` (the same feature sweeps and closed reflections
   as the exact path, seeded with the inherited contacts and features); and,
   if no candidate closes, the polish below. Proposals are never trusted.
3. **Proof** (the cycle analogue of `ProvaLimites`). Membership: a point is its
   own contact; on a segment `[a,b]`, a binary64 contact is accepted if it is
   exactly on the segment, otherwise `t=clamp((q-a).(b-a)/|b-a|^2,0,1)` and
   the exact point `a+t(b-a)`, which lies on the segment for every binary64
   `t` in `[0,1]`, is carried as an interval box; in a polygon every edge
   sign is decided exactly, and an unproved contact moves toward the vertex
   mean by `2^-45, 2^-40, 2^-30, 2^-20, 2^-10, 2^-6, 2^-3, 1/2, 1`. `U` is the
   upper end of the interval sum of the link lengths between boxes. `L` is a
   lower bound of `D(u)=sum_i min_(v in P_i) (v-r).(u_(i-1)-u_i)` for each
   proposed `u`, rounded to binary vectors proved to lie in the unit disk
   (`binary_dual_vector`), with `r` the mean of the regions' vertex means; and
   `L >= 0`. Since 2026-10-09 each support is the binary64 minimum widened by
   an a priori bound of its rounding error (`support_bounds`, `binary_dual.h`;
   proof in [unordered-tpp.md](unordered-tpp.md#certificado-convexo-e-interseções)),
   which costs about a third of the interval enclosure, and edge signs in
   polygons try a static orientation filter before the interval and the
   integer determinant. For construction candidates, links no longer than
   `max(32*eps*scale, max_gap/(16k))` get four proposals: their own
   directions, the nearest long link's on either side (cyclically), or zero.
   These four apply one rule to every short link. A fifth proposal
   (`cycle_dual_by_runs`, `float_proof.h`) chooses per run of short links,
   because a cycle through several blocks of coincident contacts usually
   needs a different vector in each block. `D(u)` separates over the runs.
   A run `j..j+m-1` enters only the supports of regions `j..j+m`, whose
   other vectors are the long links `j-1` and `j+m`. Dynamic programming
   over each run picks the vectors that maximize that part of the bound,
   among both long directions, their midpoint, the link's own direction and
   zero. It uses the same rounded supports, and every choice is a valid dual.
   The polish proposes its smoothed directions. Lengths only select
   proposals; no length is used as a tolerance or as a lower bound.
4. **Polish.** A log-barrier Newton method on the smoothed cycle length (see
   the proposition below), in coordinates `x=(p-r)/sigma` with `sigma` the
   largest coordinate difference from `r`. A polygon contact is a free 2D
   variable with a barrier per edge; a segment contact is `x=a+t*e` with the
   barrier `-mu*(log t+log(1-t))`; a point is fixed. It starts from the
   candidate with the smallest proved `U` (else the last candidate, else the
   inherited contacts), pulled `2^-7` toward the vertex mean or the segment
   midpoint, at `mu_0=max(mu_end,min(1e-3,1e4*mu_end))` (`0.1` without a
   seed), divides `mu` by 10 per level down to
   `mu_end=max(1e-15, max_gap/(2*sigma*(k+F)))`, and stops after 600 Newton
   iterations. Each level's contacts and directions go through step 3.
   Line search, stopping test and the step taken below the objective's
   resolution are those of the fixed-endpoint polish; both use the same
   chain kernel (`float_chain.h`).

### Correctness of the bounds

**Theorem.** Let `P_1,...,P_k` (`k >= 2`) be closed convex regions with
binary64 coordinates, each a point, a segment or a polygon of positive area.
Suppose that whenever `cycle_interval_environment()` accepts, binary64
operations follow IEEE 754 with round-to-nearest. If the stage returns
`(L,U)`, in any status, then `L <= OPT <= U`. When `U` is finite, it bounds the
length of a cycle of exact points `z_i in P_i`; the returned contacts are those
points, except on a segment, where they are binary64 approximations of them
(`a+t*(b-a)` evaluated in binary64, within a few units in the last place).
This holds whatever the construction kernel and the polish propose.

*Proof.* (a) Normalization decides each turn sign from an interval that
contains its exact value or from an exact integer determinant, and accepts
exactly the boundaries that `prepare_cycle_polygons` accepts (one turn sign,
one winding, no backtracking). Removing collinear vertices does not change
the set. So each normalized region is the input region.

(b) Each contact box contains an exact point of its region: a point region's
own point; a binary64 point proved on a segment by an exact zero orientation
and two nonnegative dot-product intervals, or `a+t(b-a)` with binary64
`t in [0,1]`, enclosed by interval operations; a binary64 point all of whose
edge signs are proved nonnegative in a counter-clockwise convex polygon. If
some contact has no proof, `U` is not updated. Otherwise the exact points
form a feasible cycle, and each link length lies in the interval
`sqrt(dx^2+dy^2)` of the difference of its boxes (a link between two equal
binary points is exactly zero). The upper end of the interval sum bounds the
feasible cycle's length, hence `OPT`.

(c) Each `u_i` is a binary vector whose squared norm has an interval upper end
`<= 1`, or zero. For any feasible cycle `q`, `|q_(i+1)-q_i| >= u_i.(q_(i+1)-q_i)`;
summing and regrouping by contact gives `sum_i q_i.(u_(i-1)-u_i)`. The
coefficients `u_(i-1)-u_i` sum to zero exactly, so this equals
`sum_i (q_i-r).(u_(i-1)-u_i) >= sum_i min_(v in P_i) (v-r).(u_(i-1)-u_i) = D(u)`,
and the minimum of a linear function over a convex polygon, a segment or a
point is attained at a vertex. Hence `D(u) <= OPT`. Every computed vertex term
is within `E` of its exact value (standard model with gradual underflow; `E`
as in `support_bounds`), so the computed minimum minus `E`, rounded down, is
`<=` the exact minimum, and the sum of these terms, rounded down one value per
addition, is `<= D(u)`. `OPT >= 0` trivially. Hence `L <= OPT`. (A filter
sign in (b) is decided only when the computed determinant exceeds its own a
priori error bound, so it is the exact sign.)

(d) The stage reports closure only with a finite `U` from (b), so a closed
result always has a feasible cycle, as `CertifiedBound` requires. In
`tpp_convex_solve_cycle_double`, an open stage contributes only its `L`, and
the exact path's bounds keep their own certificate contract. In all cases
`L <= OPT <= U`. No step used the optimality of a proposal. ∎

The theorem says the returned bounds are true, not that the stage closes.
It does not depend on the correctness of the construction kernel, of the
directional maps or of the polish. The correspondence between this proof and
the code was checked by reading and by tests, not by formal verification.

### Why the polish produces good lower bounds

**Proposition.** In scaled coordinates, let polygon `i` have faces
`n_f.x >= o_f` (unit inward normals), let a segment have `x=a+t*e` with faces
`t >= 0` and `-t >= -1`, and let a point be fixed. For `mu > 0` let
`f_mu(y) = sum_(i=1..k) sqrt(|x_(i+1)-x_i|^2+mu^2) - mu*sum_i sum_(f in F_i) log(n_f.y_i-o_f)`,
with `x_(k+1)=x_1`. If `y` is a stationary point and
`u_i=d_i/sqrt(|d_i|^2+mu^2)` with `d_i=x_(i+1)-x_i`, then
`D(u) >= sum_i |d_i| - mu*(k+F)`, where `F=sum_i |F_i|` (two per segment, none
per point). In original coordinates the loss is `sigma*mu*(k+F)`.

*Proof.* Let `n_i=u_(i-1)-u_i` and write `x_i=c_i+B_i y_i` (`B_i` the identity
for a polygon, the column `e` for a segment). The derivative of `f_mu` in
`y_i` is `B_i^T n_i - sum_f lambda_f n_f` with `lambda_f=mu/(n_f.y_i-o_f) > 0`,
so it vanishes at the stationary point. Any `v in P_i` is `v=c_i+B_i w` with
`n_f.w >= o_f`, and then
`v.n_i - x_i.n_i = (w-y_i).B_i^T n_i = sum_f lambda_f n_f.(w-y_i) >= -sum_f lambda_f (n_f.y_i-o_f) = -mu*|F_i|`.
A fixed point has `v=x_i`, a zero difference. Summing over `i`,
`D(u) >= sum_i x_i.n_i - mu*F`, and the cyclic regrouping of (c) gives
`sum_i x_i.(u_(i-1)-u_i) = sum_i u_i.d_i`. Finally
`u_i.d_i = |d_i|^2/sqrt(|d_i|^2+mu^2) >= |d_i|-mu`. The reference `r` drops out
because the `n_i` sum to zero. ∎

This only justifies `mu_end` (half the requested gap at exact stationarity);
the returned bounds are always those of step 3. The same argument with fixed
start and target nodes and an open chain gives the fixed-endpoint bound
`D(u) >= length - mu*(m+1+F)` with the same face count for points and
segments; the fixed-endpoint oracle uses it for regions with points or
segments.

### Newton system

The gradient in `y_i` involves only its two neighbours, so the Hessian is
block tridiagonal with 2×2 blocks in the order of the variable nodes. The link
from `x_i` to `x_(i+1)` adds `(I-u_i u_i^T)/sqrt(|d_i|^2+mu^2)`, projected by
`B_i` and `B_(i+1)`, to the two diagonal blocks and minus that to their
coupling; each face adds `(mu/s_f^2) n_f n_f^T`; every block gets `1e-14 I`,
and a segment's unused second component a unit diagonal (its step is zero).
A fixed node is not a variable: it
decouples its neighbours, and a cycle with a fixed node is ordered from the
node after it, so its Hessian is block tridiagonal. Without fixed nodes the
closing link couples the first and last variables. The solver then
eliminates the first `k-1` blocks for the right-hand side and the two columns
of the last block column (`T [z W] = [r b]`), solves the 2×2 Schur complement
`A_k - b^T W` for the last variable and back-substitutes. Each iteration costs
`O(k+F)` operations, like the open chain. The barrier keeps the Hessian
positive definite, so the elimination needs no pivoting.

### Complexity and limits

Normalization costs `O(N)` exact-sign predicates for `N` input vertices, each
proof `O(N+k)` interval operations, each Newton iteration `O(k+F)`, and at most
600 iterations run; the construction kernel keeps its own bound. Points and
segments need no special exact handling. With gaps far tighter than the
B&B's, binary64 can stop the polish before it closes (a subnormal `max_gap`
in the tests); its bounds stay valid and the exact path runs. On a segment,
contacts approximate exact segment points to a few units in the last place;
the exact path's segment contacts are also rounded when exported. The
environment check is the one of the interval certificate.

A construction detail found while testing: `CycleRefinement::coordinate`
treated a point region as the whole plane (its zero-length edges give no
halfplanes) and proposed points beside it. The exact certificate always
rejected those proposals; the kernel now returns the point itself (vertex
feature 0).

Tests: `binary64_stage` in `main-cycle_tests.cpp` checks, against the
rational solver, every interval of the stage (with and without the
construction, at relative gaps `1e-3`, `1e-6` and `1e-9`, and with a cutoff)
on segments, points, touching tessellations, shared-edge blocks, nesting,
crossing and 60 random mixed cases; the named cases close by the interval
proof (collinear boxes), by the polish (the report's coincident-contact
instance `t_A` as a cycle with points), stay open and fall back to the exact
path (subnormal gap), and reach a cutoff. Measurements:
[`tspn-cycle-float-2026-10-08`](../../benchmarks/results-saved/README.md#tspn-cycle-float-2026-10-08).

## Regions with many vertices (2026-10-21)

With `k` regions of `m` vertices each, the binary64 stage above cost
`Theta(km)` per call (normalization, supports, membership), and its
common-region clip cost `Theta(m^2)`. This section brings a warm call to
`O(k log m)` queries plus the polish. The queries that feed `L` and `U` keep
the loops' results; elsewhere only proposals change, and step 3 still
proves every bound on the input regions.

### Prepared regions

`ConvexCycleDoubleOptions::region_ids` (one stable identifier per region)
together with a `ConvexCycleWorkspace` makes the stage prepare each region
once: the normalized region (step 1), its kernel copy, its vertex mean and
the construction's centroid (`ConvexCycleWorkspace::prepared_region`).

- An entry is reused while the input's size and its first and last vertices
  are bitwise equal. That is a guard against a misused identifier, not a
  proof; the caller promises that an identifier always names the same region.
- Without identifiers, every call normalizes again (`O(m)` per region,
  through the content-keyed cache).
- The TSPN B&B labels each hull or convex piece `(polygon << 32) | piece`
  ([tspn.md](tspn.md#large-convex-regions-2026-10-21)).

### Logarithmic queries

Each query below applies to a normalized region (strictly convex,
counterclockwise) above a size threshold. Anything undecided runs the
previous loop.

1. **Membership proof** (`proved_inside`, from 16 vertices). A binary search
   on floating orientation signs picks the wedge `j` of the fan from `p_0`.
   Three exact signs (static filter, interval, then 128-bit determinant)
   decide the rest:
   - `q` in the closed triangle `(p_0, p_j, p_(j+1))` proves `q` in `P`, since
     the triangle lies in `P`;
   - `q` in the closed wedge but strictly beyond edge `j` is outside `P`,
     since `P` meets the wedge only in that triangle.

   The wedges `lo, lo-1, lo+1` cover a search misled by rounding.
2. **Supports** (`support_bounds_logarithmic`, from 64 vertices, without the
   width). Along the boundary of a strictly convex polygon, the exact values
   `t(v) = (v - o).n` have a single local minimum: a vertex, or an edge
   orthogonal to `n`. A linear function whose two edge neighbours at a
   vertex are not lower cannot decrease into the tangent cone, which those
   two edges span.
   - A binary search for the extreme vertex (KACTL's `extrVertex`) and a walk
     on computed values give a guess `w`.
   - The window is `w`, or `w` and one neighbour when the interval of their
     difference contains zero (an optimal edge contact makes `n` normal to
     the edge, so such ties are frequent).
   - Each outer neighbour must be certified not lower by an interval. Then
     the window holds an exact minimizer `t*`.
   - The reported `low` is the least computed `t` over the window and its
     neighbours. Every computed `t` is within `E` of its exact value, so
     `low - E <= t* <= low + E`. Here `E` uses the largest coordinate
     differences, read from the four coordinate extremes: comparisons of
     coordinates are exact.
   - The enclosure is therefore valid. It equals the loop's except when a
     distant vertex ties with the minimum within rounding.
3. **Coordinate update** (`coordinate_logarithmic`, construction only, from
   32 vertices). It finds the chord of the line `ab` from the two edges where
   the line crosses the boundary (binary search between the extremes on
   either side). Otherwise it locates the feature of the last-step map of
   Dror et al. from `a` at `b`, by binary search over the vertex cones (each
   cone built on demand). The located feature and its neighbours at offsets
   `±1, ±2` must pass the loop's own tests (`checked_feature`); otherwise
   the loop runs.
4. **Common region.**
   - `separated_toward` tests, in `O(log m)` with a margin over its rounding,
     whether an edge of one region, at its extreme toward the other's mean,
     leaves the other strictly outside. If so, the regions are disjoint, and
     the clip's result (empty) is known without clipping.
   - `intersect_convex` intersects two strictly convex counterclockwise
     regions with at least 32 vertices in binary64, in `O((m+n) log n)`. It
     uses the vertices of the first inside the second (fan tests), the
     crossings of the first's edges with the second's boundary (chords) and
     the second's boundary arcs from each exit to the next entry.
   - When the crossings do not alternate, the clip runs. The output matches
     the clip's region up to rounding, not bit for bit, so constructions may
     differ in rounding. It feeds `move_zero_blocks` (blocks of coincident
     contacts) and the zero-cycle test, both proposals; the rational path
     keeps the clip.
   - With identifiers, the common region of two regions with at least 32
     vertices is kept in the first one's prepared entry. The entry holds a
     weak reference to the other, so a replaced region is never matched.
     Each overlapping pair is therefore intersected once per workspace, both
     for two-region blocks and for regions 0 and 1 of the zero-cycle test.

### Polish on inner approximations

From 64 vertices, the polish works on the hull `Q_i` of a subset `S_i` of
the vertices of `P_i`, in index order (`region_node`, `float_proof.h`).

- **Bands.** A band around vertex `w` holds the vertices `w-4..w+4` and, at
  each coarser spacing `2^j` from 8 up to half the skeleton's (`m/32`), the
  four multiples of `2^j` at offsets `-1, 0, 1, 2` from `w` rounded down to
  that spacing. It is graded: fine at `w`, coarse away from it, with
  `O(log m)` vertices. The coarse vertices lie on a fixed grid, so a small
  move of `w` adds only fine ones.
- **Initial subset.** `S_i` holds 16 evenly spaced vertices, plus bands
  around two vertices:
  - the fan wedge holding the warm start (with `p_0`, so `Q_i` contains the
    warm start);
  - the vertex extreme toward the seed's descent direction (the sum of the
    unit vectors toward its neighbours, an outward normal at an optimal
    contact).
- **Feasibility.** Since `Q_i` lies inside `P_i`, every iterate is feasible
  and the line search needs no extra test.
- **Refinement.** Within a barrier level the faces stay fixed. When Newton
  stops, `refine` looks up the vertex `w` of `P_i` extreme toward the
  converged descent direction `u_i - u_(i-1) = -n_i`. That vertex minimizes
  `n_i.v`, the `i`-th term of `D(u)`. If `w` is already in `S_i`, that term
  is the same on `Q_i` and on `P_i`, and nothing changes. Otherwise the band
  around `w` is inserted and the faces are rebuilt. Inserting vertices only
  enlarges `Q_i`, so the current point stays inside.
- **Re-convergence.** If any approximation changed, the level runs again
  from the current point, up to eight times.
- **Why only on convergence.** Refining before every Newton step followed
  the extreme along the path of the iterates. With 16k vertices per region
  that path crosses dozens of vertices per step, and every step added a
  band. On smoothed `vienna` m=16384, the B&B took 10.6 s against 4.4 s with
  refinement on convergence only.
- **Gap and stopping level.** The proposition above holds on the `Q_i`: at a
  stationary point, `D_Q(u) >= length - mu*(k+F_Q)`, with `F_Q` the current
  faces. Term by term, `D_P(u) = D_Q(u)` whenever the minimizing vertex of
  `P_i` is in `S_i`, which is what the refinement targets. `mu_end` follows
  the current face count (`gap_target/(links+F)` after every level).
- **Bounds.** As always, they come from step 3 on the `P_i`, never on the
  `Q_i`.

A first version used an outer window instead: the faces around the
predicted contact, plus cutting planes for the faces that trial points
crossed. On the smoothed corpus regions (2048 vertices, mostly on arcs), one
Newton step crossed hundreds of short faces, and each cut cost an iteration.

Calls captured from `vienna` m=2048 (`k = 22, 23`):

| Polish | Newton iterations (2 calls) | Closed? |
|---|---|---|
| Outer window | 268, 395 | no: the exact path took 5–10 s |
| Outer window, stopping level from the current faces | 529, 567 | yes |
| Inner approximation | 63, 56 | yes |
| Every face | 67, 67 | yes |

Across all 199 captured calls, the polish took 3 ms in total with the inner
approximation against 91 ms with every face, and both closed the same
calls.

The subset only grows. Dropping bands far from the current extreme (keeping
the skeleton, and only while the current point stayed inside the smaller
hull) left some calls of `random_mixed` m=16384 open. Those calls then ran
the exact path for minutes, so the subset is never pruned.

**65,536 vertices (2026-10-22).** On the smoothed instances with m=65536
(2.4 to 3.9 million vertices), some calls still reached the exact path,
whose `O(m)` rational normalization then took the whole budget. Two calls
of `vienna` (`k = 30`, 1.97 million vertices) were captured and replayed.
Their polish stopped at 600 Newton iterations with the stage open. Two
things had gone wrong:
- **Refinement without progress.** The band was then only `w-4..w+4`. With
  arcs of thousands of short edges, each refinement moved `w` just past the
  last band. All eight rounds of every level ran and added faces, and `Q_i`
  never reached the extreme. The graded band covers a wide stretch around
  `w` from the first round. A region whose `w` is already in `S_i` is left
  as is, so the level ends once every extreme is in its subset.
- **Newton stalled at the rounding limit.** At small `mu` the Armijo test
  still accepted tiny steps (`alpha` near `4e-9`, slope `-9e-14`) that left
  the objective unchanged, and the level repeated them until the iteration
  cap. A step that leaves the objective unchanged now ends the level
  (`float_chain.h`), so it applies to every polish, the fixed-endpoint one
  included.

With these changes the two calls close in binary64 (0.08 and 0.07 s), and the
whole of `vienna` m=65536 closes the 0.1% gap in 12.9 s, with no exact call.
All six smoothed instances with m=65536 then close, in 2 to 13 s on the Mac.
With the changes switched off in combinations, `vienna` closes with the
Newton stop alone, but `warsaw` needs all three (the graded band, refinement
only on a new extreme, and the Newton stop). On the 558 corpus cases, calls,
nodes and bounds are unchanged bit for bit. On the smoothed instances up to
m=16384, calls change by at most 0.4% (construction only), and time grows by
up to 13% (1.055× in geometric mean); which change costs it was not isolated.
Measurements:
[`tspn-smoothed-m65536-mac-2026-10-22`](../../benchmarks/results-saved/README.md#tspn-smoothed-m65536-mac-2026-10-22).

### Complexity and what remains linear

A warm call with identifiers costs:
- `O(log m)` per membership, support and coordinate query;
- `O(k log m + F_Q)` per Newton iteration;
- `O((m+n) log n)` per common-region intersection of overlapping regions,
  once per pair and workspace when the regions have identifiers;
- copying a cached common region (its own vertices) into a block or the
  zero-cycle test.

Still linear or worse:
- a cold call (normalization, `O(m)` per region);
- `CycleRefinement::run` with `k = 2`, whose closest pair is `O(m^2)`;
- the exact rational path.

Tests (`main-cycle_tests.cpp`):
- `logarithmic_queries` compares memberships, supports (with ties),
  coordinate updates and separations with their loops, on lattice hulls of
  64 to 9000 vertices. It checks that calls with and without identifiers
  agree, and that the bounds enclose the exact optimum.
- `convex_intersections` compares `intersect_convex` with the clip on
  overlapping, nested and disjoint hulls.

Measurement, `k = 3` against Gurobi: [`oracle-vertex-scaling-2026-10-21`](../../benchmarks/results-saved/README.md#oracle-vertex-scaling-2026-10-21).

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

**Closed-form step (2026-10-23).** Each anchored solve returns a cycle, and
with it a combinatorial type: per region a vertex, an edge, or a straight
passage (`contact_features`; a contact strictly between its neighbours on one
line counts as straight even on the boundary). With the anchor reflecting on
the current edge, `CycleRefinement::close` solves that type in closed form by
unfolding. Its anchor parameter is the next probe on the edge, and its cycle
goes to the global certificate. Within one type `f` is a distance to fixed
images, so when a probe has the type of the edge's minimizer the next probe
lands on it exactly: this is the role of a Newton step, exact here. The step
runs at both ends of a bracketing piece and after every probe, in the
disjoint search (`search_cycle`) and in the general one, rational and binary64.
Midpoints and simplest rationals still follow when the step's parameter falls
outside the bracket, so termination is unchanged.

On 320 captured B&B calls solved to gap zero with the construction disabled,
four needed bisection, with about 450 probes each (16-30 s, one interrupted at
30 s); with the step no call needed more than one probe, and the most anchored
solves in a call fell from 482 to 22. With the construction first, as the exact
path runs, no call reached a probe either way. `ConvexCycleResult` counts
`anchored_solves`, `bisection_probes` and `closed_form_steps`; the tool
`tpp-convex-anchor-probes` replays captures. Measurements:
[`anchor-step-mac-2026-10-23`](../../benchmarks/results-saved/README.md#anchor-step-mac-2026-10-23).

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
minimum contains a rational minimizer. Flat minima are real here: unlike the
fixed-endpoint TPP, whose shortest path is unique (Dror et al.), a cycle has no
fixed point, and optimal cycles can form a continuum with the same link
directions. Four equal squares turned 45 degrees at the corners of a square
admit a segment of optimal rectangles, and `F` is constant along the inner edge
([`funcao-ancoragem`](../reports/funcao-ancoragem/README.md), in Portuguese). Endpoint minima were already handled
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

## Disks (2026-10-25)

`tpp_convex_solve_cycle_disks` (`disk_cycle.h`, `cycle_disk_oracle.cpp`)
solves the fixed-order cycle through closed disks (radius 0 is a point), the
subproblem of the close-enough TSP. It keeps the propose-and-certify contract;
only the regions change.

- **Lower bound.** For any `u_i` proved in the unit disk, weak duality gives
  `D(u) = sum_i min_{v in D_i} (v-r).(u_(i-1)-u_i)`. Over a disk the minimum is
  `(c_i-r).w - radius_i |w|`, in closed form: no vertex or extreme-point search.
  `L` is the lower end of its interval evaluation, with the proposal's smoothed
  link directions as `u`.
- **Upper bound.** Each proposed contact is proved inside its disk
  (`radius^2 - |x-c|^2 >= 0` with directed rounding), or moved toward the
  center until it is. `U` is the upper end of the interval sum of the links.
- **Proposal: geometry first.** The construction plays the role the
  last-step maps play for polygons, with the disk's reflection law in place of
  edge unfolding:
  - *Alhazen update.* Between fixed neighbours `a` and `b`, the point of a disk
    minimizing `|x-a|+|x-b|` is the foot of its center on `ab` if the segment
    meets the disk (a straight passage), else the reflection point on the arc
    between the radial directions of `a` and `b` (Newton on `g'(theta)`, kept
    in that bracket). Gauss–Seidel sweeps of these exact coordinate updates
    never lengthen the cycle and fix a combinatorial type: straight,
    reflecting, or a corner.
  - *Closing the type.* With the type fixed, straight contacts drop out and
    Newton on the reflection angles minimizes the reduced cycle. Its Hessian
    is cyclic tridiagonal (Thomas plus Sherman–Morrison), and each
    reflection's curvature adds `radius * lambda` to the diagonal, so it
    converges quadratically near the optimum. Straight contacts then return to
    their reduced segments in order along each (out of order they zigzag), and
    a sweep checks the type.
  - *Blocks.* With overlapping disks, a zero-length link holds two coincident
    contacts together: moving either alone never shortens the cycle, so the
    sweeps stall in a lens. A run of coincident contacts moves as a block, the
    shortest of: one point of the intersection of its disks (on the segment,
    at an Alhazen point inside the others, or at a corner where two circles
    cross); one member reflecting with the others straight; or one Alhazen
    pass each way. A corner is kept only if its dual splits (below);
    otherwise its members start a little apart and Newton separates them.
    Mennell's bubbles need this: circles of radius 10 at spacing 10 are
    tangent at a third disk's center, a single point no tour needs to share.
    Inside Newton, two reflections that meet at the crossing of their circles
    become a corner, and a reflection that meets a fixed bend stays there.
    Newton approaches a corner slowly (the reduced length is not smooth
    there), so two reflections are also merged at their crossing whenever
    that shortens the reduced cycle.
  - *Duals.* Every link takes the direction of the reduced segment around it,
    so straight passages bend by exactly zero. At a corner the zero-length
    link takes `u_mid = u_in + l1 n1`, where `l1 n1 + l2 n2 = u_out - u_in`
    for the outward normals there. The multipliers are nonnegative and
    `|u_mid| <= 1` exactly when the corner is optimal. At a closed optimal
    type, `D(u)` equals the length up to rounding.

  Every round is certified. The construction stops when the gap closes, or
  when it stops halving for three rounds. In that case the barrier runs from
  its contacts. The barrier is the chain Newton kernel with an optional disk
  per node and the term `-mu log(radius^2 - |y-c|^2)`. Its gradient is
  `2mu r/s` and its Hessian `(2mu/s) I + (4mu/s^2) r r^T`. A warm start begins
  at `mu = 1e-4` (scaled coordinates). Started far below the central path,
  Newton crawls.
- **No gap-zero path.** With polygons, a fixed combinatorial type is solved in
  rationals by unfolding (Theorem B). Reflection in a circle is not an
  isometry: a single disk between two points is Alhazen's problem (a quartic).
  So the disk oracle certifies `L <= OPT <= U` to any requested gap and has no
  exact path. The construction comes close: it certifies gaps of about
  `1e-12` relative, below what the barrier reaches within its iteration
  budget.

Tests (`main-disk_cycle_tests.cpp`):
- known optima;
- 60 random instances whose bounds must lie between the polygon oracle's
  bounds on inscribed and circumscribed 64-gons;
- 60 random instances, a third with heavy overlaps, where the construction and
  the barrier alone must certify the same value (the construction closes all
  of them);
- warm starts and cutoffs.

The kernel change leaves polygon nodes untouched: the 558 corpus cases are
identical bit for bit.

`tpp-convex-cetsp` is an exploration branch-and-bound on top of it. Its nodes
are cyclic sequences from the depot. A node's bound is its oracle `L`:
visiting a subset can only shorten the tour.
- **Branching.** A cycle that meets every disk is a feasible tour. Otherwise
  the farthest uncovered disk is inserted at every position, with the mirror
  symmetry removed at the second disk. Coverage uses a margin of `1e-9`
  times the radius, so a borderline disk is branched on rather than assumed
  covered. Strong branching (several candidates, keep the one whose weakest
  child is strongest) is optional. On rotatingDiamonds3 and
  concentricCircles2 it gave weaker bounds in the same time, even with the
  candidates compared by insertion bounds alone. So did lifting a node's
  bound by the best of several uncovered disks' smallest insertion bounds:
  it is valid, but rarely prunes and costs more than it saves.
- **Insertion bounds and lazy children** (as in the TSPN's `lazy`).
  `tpp_convex_disk_insertion_bounds` gives every position's child a weak
  dual bound from the parent's link directions. The inserted disk's
  proposed contact is its point nearest the link: no Newton, and its
  detour point whenever the link meets it. Only three support terms change
  per position, and a disk's support is closed form, so all k bounds cost
  O(k). Each child enters the queue with its bound and is solved only
  when expanded; a bound above the cutoff prunes it unsolved. On 588 random
  insertions the bounds never exceeded the child's optimum, and 89% were
  within 1% of its `L`.
  - Against solving every child when created, in the same 300 s of B&B:
    rotatingDiamonds3 reaches `L` = 371.2 instead of 360.8.
  - chaoSingleDep closes in 299 s instead of about 1,770 s.
  - `--mode screen` prunes with the bounds but solves the survivors at
    once; `--mode eager` solves every child.
- **Memory.** A node takes 16 bytes: 8 in the tree (parent and insertion)
  and 8 in the queue. The queue keeps the bound in single precision rounded
  down, still a lower bound. On expansion the sequence is rebuilt from the
  root, each disk entering at its point nearest the link between its
  neighbours there, and its cycle is solved warm from those points.
- **Discarded feasible nodes.** A feasible child gets no node. Its `L` still
  enters the final bound, so a gap the oracle left there is not lost.
- **Threads.** `--threads T` expands batches of 4T best nodes in parallel
  (the oracle has no shared state) and merges them in order. Cutoffs use
  the batch's incumbent, which only weakens pruning a little. With 8
  threads on the Mac the B&B is about 5x faster, with the same node counts
  on the instances measured.
- **Initial tour.** A CETSP tour is also a tour of every disk in some order,
  one contact each: a disk the cycle crosses anyway is a straight passage at
  no cost. Steps:
  1. A tour of the centers (nearest neighbour, 2-opt and Or-opt with the
     centers fixed), its contacts solved by the oracle.
  2. Within 30% of the budget, cheapest insertion and local search on
     sequences, expanded to a full order with the same length, if shorter.
  3. Iterated local search on full orders, one per thread with its own seed.
     - Moves: 2-opt and Or-opt (blocks of 1–3, either way round) toward
       each disk's 10 nearest. A moved block's contacts are placed again by
       Alhazen between its new neighbours, so moving a disk to where the
       cycle already crosses it costs nothing (with its old contact, that
       move looked expensive and was never made). A move applies when the
       cycle through the contacts, always feasible, gets shorter.
     - A work queue re-examines only disks near a change.
     - At a local optimum the contacts are swept, then re-optimized and
       certified by the oracle.
     - Kicks swap two adjacent segments of up to 50 disks; acceptance uses
       a threshold.

  The budget is 5% of the time. A search stops after max(4n, 200)
  iterations without a new best.
- **Independent check.** The tour can be written out and checked in exact
  rationals on the instance's decimal text: coverage of every disk, and a
  length bound with rounded-up square roots.
- **Integration with the TSPN B&B.** Not done: it would pay only if faster.
  The TSPN's two pieces that help disks, insertion screening and lazy
  children, are here, in O(1) closed form.
  - A node here costs about one oracle call (20–40 µs). A TSPN node costs
    at least that, plus polygon machinery: decomposition, polygon visit
    checks, rational fallbacks.
  - On the polygonal sandwich (128-gons), the TSPN B&B also builds no
    smaller trees: about 52 thousand nodes on rotatingDiamonds2, against
    33 thousand here.
  - So it could not be faster, only slower.

Measurements:
[`cetsp-campaign-mac-2026-10-25`](../../benchmarks/results-saved/README.md#cetsp-campaign-mac-2026-10-25)
(earlier versions:
[`cetsp-geometric`](../../benchmarks/results-saved/README.md#cetsp-geometric-mac-2026-10-25),
[`cetsp-native`](../../benchmarks/results-saved/README.md#cetsp-native-mac-2026-10-25)).

## Tests and measurement

Build `main-cycle_tests` and `main-cycle_certificate_tests` through the existing
CMake `TARGET` option. The cycle tests cover 2–5 visits, crossing cycles,
non-dyadic contacts, all-edge optima, closed intersections and containment,
cyclic/reversal variants, 96 seeded intersecting cases, and rational scales
`2^-1100` and `2^1100`. Accelerated and unaccelerated rational searches are
compared; selected intersecting cases also force the unaccelerated pure-double
search. The zero-link certificate has independent analytical ray-pair tests.
Four squares turned 45 degrees check optima interior to edges that no vertex
anchor reaches, a flat optimum that a vertex-only search would hit by
coincidence, and a strict non-global minimum of `F` along a closed boundary;
[`funcao-ancoragem`](../reports/funcao-ancoragem/README.md) (in Portuguese)
draws `F` for them against an independent SOCP.

`python3 benchmarks/tpp.py cycle-benchmark --output NEW_DIRECTORY` compares
whole API calls with warmed, single-thread Gurobi SOCP calls, including model
construction for Gurobi and validation/certification for C++. Solver order
rotates, and every campaign saves inputs, individual repetitions, configuration,
source hashes and analysis. Compare objective values and independently certified
intervals, not contact coordinates or Gurobi's numerical bound as an exact bound.
Finite synthetic benchmarks do not establish universal speed dominance.

## Cooperative interruption and exclusive timing

The general rational and binary64 options accept `max_seconds` and an optional
`stop_requested` callback. The default remains unlimited. A solve checks the
request between refinement contacts, reflection steps, boundary probes, map
queries and certificate invocations. Nested rational recovery shares its
caller's deadline. Interruption bypasses arithmetic-recovery exception handlers
so it cannot accidentally launch another fallback.

`Interrupted` is distinct from `Optimal`, `CertifiedBound` and arithmetic
failure. Empty contacts carry an infinite upper bound; otherwise they and the
attached interval come only from a completed independent certificate. It never certifies an
unfinished construction. A caller with an inherited lower bound can keep the
maximum of that bound and the returned certificate. No tolerance, rational
truncation or acceptance epsilon is introduced. Checkpoints cannot preempt one
GMP operation or an external decomposition, so this remains a cooperative
budget, not a guarantee of a hard wall-clock ceiling.

The result exposes exclusive construction, certification and rational-recovery
seconds. All certificate work is attributed to certification even while called
from rational recovery. These are diagnostic measurements, never inputs to
geometric acceptance or a proof of optimality.
