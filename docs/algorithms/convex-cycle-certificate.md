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
