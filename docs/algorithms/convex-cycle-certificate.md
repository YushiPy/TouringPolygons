# Exact certificate for an ordered convex cycle

For contacts `q_i in P_i` in fixed cyclic order, the objective is
`L(q) = sum_i |q_(i+1)-q_i|`. Self-intersections of the cycle are allowed.
`tpp_convex_verify_cycle_certificate` checks feasibility, optimality, and
outward-rounded global objective bounds independently of cycle construction.
Binary64 inputs are interpreted as exact binary rationals; the rational
overload preserves rational inputs and contacts without exporting them to double.

Polygons must be convex closed regions. Either winding and redundant collinear
or closing vertices are accepted. The checker also accepts a point or a segment
for restricted anchor problems. Public cycle solvers require positive-area
input polygons and at least two visits.

## Necessary and sufficient support conditions

Let `d_i = q_(i+1)-q_i`. For nonzero links the norm subgradient is fixed:
`u_i = d_i/|d_i|`. At a zero link any vector `|u_i| <= 1` is admissible.
The certificate asks whether these choices can satisfy

```
(u_(i-1) - u_i) dot (v-q_i) >= 0     for every vertex v of P_i.
```

Linearity extends each inequality to its full polygon. The supporting-plane
inequality for every link norm, summed with these inequalities, gives
`L(z) >= L(q)` for every feasible `z`. Conversely, the objective is a finite
continuous convex function and the feasible set is a product of convex sets;
the convex subgradient/normal-cone optimality condition gives exactly these
inequalities. Thus the complete test is necessary and sufficient, including
coincident contacts. A feasible all-coincident cycle attains the universal
lower bound zero immediately.

For nonzero links, each sign is `sign(a/sqrt(A)-b/sqrt(B))` with rational
`a,b,A,B` and positive `A,B`. Signs of `a,b`, followed when needed by comparison
of `a*a*B` and `b*b*A`, decide it exactly. No numerical square root or epsilon
enters the optimality test.

For a nonzero-link contact, membership is checked before the support test.
The implementation then uses the generators of the polygon's tangent cone:
the two incident rays at a strict vertex; both directions of the edge and an
inward normal at an edge-interior or redundant collinear vertex; the whole
plane in the strict interior. In the edge case, the two opposite inequalities
are one exact tangency equality. In the interior, equal normalized incoming
and outgoing directions are necessary and sufficient. A point imposes no
condition; a segment uses its two endpoint directions. These cones generate
every feasible displacement, so this is equivalent to the all-vertex
inequalities above. Exact side predicates still locate the contact; no feature
hint, interval width or epsilon can accept the certificate. The zero-block
reachability algorithm below is unchanged. Tests compare this reduction with
all-vertex radical signs across vertices, edge/interior contacts, collinear
vertices, point/segment regions and widely scaled rational inputs.

## Complete zero-block test by disk/cone reachability

Consider a maximal block of `b` coincident contacts between two nonzero links.
Its incoming and outgoing unit directions are fixed. Write `D` for the closed
unit disk and `N_i` for the outward normal cone of `P_i` at the shared contact.
The support condition is equivalent to `u_i-u_(i-1) in N_i`. Starting from the
singleton incoming direction, propagate

```
R_0 = {u_in},
R_j = D intersect (R_(j-1) + N_j).
```

The outgoing direction belongs to `R_b` exactly when the block admits all its
dual variables. These are sets of vectors inside the disk, not just unit
vectors: unit-only witness enumeration is insufficient. Keeping a vector
unchanged is always allowed, so the sets grow monotonically; reaching the
outgoing direction early permits an immediate success.

The implementation in `cycle_zero_certificate.h` represents each reachable
set as the disk intersected with finitely many halfplanes

```
n dot u <= n dot d / sqrt(d dot d),
```

where both `n` and the direction `d` have rational coordinates. The following
invariants make this representation exact and computable rationally.

1. **All extreme points lie on the unit circle.** This holds for the initial
   singleton. An extreme point of a Minkowski sum with a cone must already be
   an extreme point of the original set: a nonzero cone displacement can be
   varied along its ray. Intersecting with the disk can create new extreme
   points only on its boundary. This proves the invariant inductively.
2. **Propagation requires only inherited halfplanes and at most two new ones.**
   The polar of `N_j` is the feasible tangent cone `T_j`. Retain old halfplanes
   whose normals lie in `T_j`; at each extreme ray `t` of `T_j`, add
   `t dot u <= h_R(t)`, using the old set's support value. Circular support
   constraints are already imposed by `D`. Other supports in `T_j` are covered
   by the retained straight faces and the supports at its extreme rays:
   within a vertex normal cone they are positive combinations of its boundary
   supports. This is the support description of `R + N_j`, intersected with D.
3. **Support directions remain rational.** A linear functional on the reachable
   set reaches its maximum at a circular point. Candidates are its unconstrained
   disk maximum, and the two circle endpoints of every stored chord. One chord
   endpoint is `d/|d|`; the other has rational direction
   `2*n*(n dot d)/(n dot n)-d`, its reflection in the normal axis. Feasibility
   and support comparisons use the exact normalized-dot sign above. The maximum
   therefore supplies another rational direction for the new halfplane.

At an interior polygon contact the normal cone is zero and the set stays
unchanged. At a fixed point it becomes the whole disk. Segment endpoints and
interiors give a tangent ray or line, respectively, and follow the same update.
These cases are required for restricted anchor certificates.

There is no direction sampling, recursion budget, numerical disk tolerance, or
incomplete witness enumeration. With `h <= 1+2b` halfplanes, one support query
costs `O(h^2)` exact arithmetic operations. A block costs `O(N_block+b^3)`
operations and `O(b)` auxiliary values. The complete cycle certificate costs
`O(N + sum_blocks b^3)`, bounded by `O(N+k^3)`; without zero links it is `O(N)`.
Bit costs depend on the rational operands, and are not treated as constant.

## Bounds and status

`Optimal` is assigned only after feasibility and the support conditions pass.
`Feasible` means the candidate passes membership but not the optimality test;
`InvalidInput` and `InvalidCandidate` distinguish malformed regions and contacts.
No interval gap, iteration count, or displayed length controls acceptance.

The verifier overloads optionally accept `lower_bound_cutoff` (default infinity).
For a finite request they check membership, then use a floating estimate of the
same support expression to decide whether an early rational test is worth
trying. That estimate is never returned as a bound and cannot authorize pruning.
If promising, the same rational unit-disk dual bound is computed before KKT.
If its downward-rounded lower bound
reaches the requested cutoff, the verifier returns `Feasible`, the primal upper
bound, and `optimality_check_skipped=true`. In this case `Feasible` says that
optimality was not tested, not that it failed. A candidate that does not reach
the cutoff follows the complete existing test, reusing its computed dual bound
if KKT fails. A floating false positive costs an unsuccessful rational check;
a false negative uses the ordinary certificate and construction. Neither can
change the validity of an accepted bound. Nonfinite estimates defer to the
exact test, and no epsilon is used in the proposal filter.
Invalid contacts are rejected even for a negative cutoff, and
NaN cutoffs are invalid. The default infinity retains the complete test.
This early return certifies a pruning inequality, never exact optimality; no
epsilon is added to the comparison. The rational and binary64 overloads share
this implementation.

For an optimal candidate, lower and upper bounds enclose its actual length.
For any other feasible candidate, a dual lower bound uses rational vectors of
norm at most one and polygon support minima; the primal upper bound encloses
the candidate length. The lower bound is at least zero, using the all-zero
dual when it improves the contact-derived bound. Every integer square-root enclosure is scaled relative
to its norm with 96 reporting bits, then rounded outward to binary64. This
reporting precision is not an optimization tolerance. It handles underflow
and overflow explicitly; the exact solver also returns rational squared link
lengths, representing the objective as a sum of radicals.

Tests include 1,040 independent two-ray cases: solving the intersection of two
affine lines gives the unique intermediate dual, whose rational squared norm
is compared with one. Repeated-region, interior-contact and fixed-contact
variants check longer zero blocks. General cycle tests also disable the fast
proposal so the complete anchored search and its restriction certificates run.
See [the cycle solver](convex-cycle.md) for construction and termination.

## Optional rigorous binary64 interval filter

The binary64 overloads accept `interval_filter=true`; prepared geometry must
also request its immutable binary64 view. The rational overload stays entirely
rational. This filter is a part of the existing verifier, not another solver.

Each arithmetic primitive stores a binary64 result and expands it toward both
infinities with `nextafter`. Volatile stores separate operations and prevent
FMA contraction or excess-precision intermediates. There is no error epsilon.
The filter requires IEEE binary64, round-to-nearest and working subnormals;
fast-math, other rounding modes, unrepresentable norm enclosures and overflow
fall back to the rational path. No process rounding mode is changed.

Contact halfplane signs are decided by intervals when possible and by the
original rational determinant otherwise. Boundary ambiguity never accepts an
outside contact. Point/segment regions use the exact membership predicate.
For each nonzero link, let `h_i` be an interval upper bound for its norm.
The **exact** vector `(q_(i+1)-q_i)/h_i` belongs to the unit disk. Interval
division encloses this vector; it does not treat rounded direction coordinates
as exact unit vectors. Zero links use zero. Interval polygon support minima,
summed after a common translation, therefore enclose a valid global dual
bound. Independently summed norm intervals enclose the candidate length.

If the lower endpoint reaches a finite cutoff, the result is `Feasible`, with
`optimality_check_skipped=true`. If the cutoff lies between the interval dual
lower bound and primal upper bound, the rational certificate resolves it. If
the primal upper bound is already below the cutoff, or no cutoff was requested,
the full exact KKT predicate still runs and the intervals provide reporting
bounds. `Optimal` always requires that predicate. `interval_bounds_used` marks
successful interval reporting; rational recovery remains available.

For N vertices and k contacts, interval membership and bounds cost O(N+k)
fixed-precision operations and O(k) temporary storage, in addition to exact
predicates for ambiguous signs and any full KKT/fallback work. The prepared
binary64 view costs O(N) memory. This changes practical arithmetic costs, not
the exact solver's worst-case bit complexity or B&B's numerical gap contract.
