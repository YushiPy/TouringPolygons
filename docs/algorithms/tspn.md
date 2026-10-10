# TSPN with the maintained insertion/decomposition branch and bound

`tpp_nonconvex_tspn_solve(polygons, options)` in `tpp/nonconvex/unordered.h`
minimizes a closed Euclidean tour visiting all supplied regions, with free
cyclic order and no prescribed geometric point. A region is a point (one
vertex), a closed segment (two vertices) or a simple polygon of positive area;
polygons may be nonconvex, and regions may overlap, touch or contain each other.
With a point region the tour is solved as the closed endpoint path through
that point (see [the endpoint algorithm](unordered-tpp.md#escopo));
`result.cycle_point_anchor` names that region. Otherwise points and segments
are convex pieces of the cycle oracle. Polygon holes are not represented by this
API. The returned `path` repeats its first point at the end. With zero or one
region the optimum is zero.

This is a topology option inside the existing `unordered.cpp` search, not a
second B&B. The endpoint API retains its behavior. The queue, diving, lazy
convex decomposition, missing-region selection, incumbent checks, interrupted
frontier accounting, traces and diagnostics are shared. See
[the endpoint algorithm](unordered-tpp.md) for their contracts.
The same `options.threads` setting evaluates sibling cycle relaxations in
parallel; their bounds and incumbents are recorded after each batch. The
default remains one thread.

## Cyclic relaxation and branching

A node stores a partial cyclic sequence. Each entry represents either an
original region's convex hull or one convex decomposition piece. Its relaxation
uses `tpp_convex_solve_cycle_double` with the call's tolerance as `max_gap`
(`options.cycle_float_oracle`, on by default; `--no-cycle-float` disables it).
Its [binary64 stage](convex-cycle.md#binary64-stage-with-a-requested-gap-2026-10-08)
proves `L <= OPT <= U` with directed rounding, from the shared construction and,
if needed, an interior-point polish, and returns `GapClosed` or `CertifiedBound`
without rational arithmetic. A call it leaves open, and every zero-gap call,
uses the exact path: the exact independent certificate and counted rational
recoveries; if the contact-derived lower bound remains too weak, the rational
cycle solve strengthens it. The retained path is independently feasible for
the convex regions, except that a segment contact is a binary64 approximation
of an exact segment point (the B&B uses paths only through its own visit checks
and lengths). No optimization epsilon is added to the convex-cycle
constructor or its certificate; the gap is the B&B's own oracle request.

Each child seeds the shared convex constructor with its parent's contacts,
replacing or inserting only the contact of the new region. This is a proposal,
not a bound: certificates still validate every complete candidate. The oracle
also receives the incumbent cutoff already used by B&B. A feasible candidate
whose independently certified lower bound reaches that cutoff returns
`CertifiedBound`; the node can then be pruned without constructing its exact
optimum. This is the same bound comparison as pruning after a complete solve,
with no extra error allowance. Standalone cycle solves default to an infinite
cutoff and retain their original exact-termination contract.

By default, the root represents region 0's hull and has bound zero. A single-region relaxed
cycle may be any point in that hull. A vertex seeds the initial feasible
heuristic; it is never fixed in subsequent relaxations. Rooting at a *region*
removes cyclic rotation without imposing an artificial depot.

If a missing original region is absent from a sequence of size m, insert its
hull into every cyclic gap, including the closing edge: m positions rather than
the m+1 positions of an endpoint path. The root's first region remains first (region 0 with the default strategy). At m=2 the two
children are reversals of the same triangle; only one is needed. Subsequent
insertions retain every cyclic gap. If the missing region is already represented
by its hull, branch over its existing convex decomposition pieces.

Every feasible closed tour admits a chosen visit to region 0 as cyclic origin.
The visits to represented regions occur in a cyclic order; any further chosen
visit belongs to one of its gaps. Reversing the full tour preserves its length
and visits, so the size-two symmetry reduction loses no objective value. A
visit to a nonconvex region belongs to some decomposition piece. These facts
prove coverage of the branching alternatives. Omitting regions and replacing
them with convex hulls relaxes the constraints, so the cycle optimum is a valid
node lower bound. A relaxed tour that covers all original regions is a feasible
global incumbent.

As in the endpoint search, crossings of a partial tour and repeated visits to
one region do not authorize pruning. The existing counterexamples and their
reasoning still apply.

## Cooperative search portfolio

Set `options.portfolio=true` or use `tpp-unordered --cycle --portfolio` for two
independent instances of this same B&B. Each has one oracle thread, its own
frontier, decomposition cache, normalized geometry and oracle workspace. The
original input is immutable and shared. Leave `options.threads=1`; combining
sibling parallelism with the portfolio is rejected to avoid silently using
more than two workers. Native OS threads are used, without CPU affinity.
WebAssembly rejects this optional mode.

Worker 0 uses the maintained best-bound queue with the configured diving
interval. Worker 1 uses a DFS/BFS policy inspired by the pinned supplier's
`strategies/search_strategy.h`: visit the cheapest child depth first, then
return to the globally cheapest frontier after a pruned or feasible node.
Children are ordered by bound, relaxed path length, then serial number.
Comparison is strict; the supplier's approximate-tie threshold is not copied.
An independent bound index always tracks the minimum over the entire DFS
frontier, including children whose oracles were deferred by a budget limit.

For cycles, worker 1 starts with a farthest pair of original regions plus the
region maximizing the sum of its distances to that pair, following the root
selection described in the supplier's `root_node_strategy.cpp`. There is one
unoriented cyclic order on three distinct labels. Therefore choosing that
triple removes neither a geometric constraint nor a class of feasible tours:
all later cyclic insertion gaps remain represented, and each hull can still
be refined into its decomposition pieces. Root distance computations only rank
regions; they are not used as certified lower bounds. No geometric contact is
fixed. For endpoint paths the second worker uses the endpoint-distance root
choice in the same existing branching code.

Incumbent publication occurs only after the existing feasibility check. Workers
share a synchronized path/value snapshot in their identical normalized
coordinate system, with a fast atomic upper-bound check avoiding a lock when
there is no improvement. The receiving worker validates the path again before
adopting it. Updating a feasible upper bound changes pruning strength without
removing any solution that could improve that bound. There is no new gap,
feasibility tolerance or discretization.

`max_calls` is a **global** atomic reservation budget, including both workers'
initial convex polishing calls. `max_seconds` is one common wall deadline.
A worker requests peer cancellation only after its full solve, including
original-coordinate feasibility restoration, closes the requested gap. A
numerical limit, exhausted budget or worker exception is not an optimality
proof. An exception in one worker is reported in its diagnostics while the
other can still complete; if both fail, the call throws. Shutdown is
cooperative. Cycle construction, rational recovery and certification check the
common deadline/cancellation between geometric operations; an individual exact
arithmetic operation or decomposition still finishes before returning. No
detached thread outlives its inputs.

Each search covers the complete feasible set. Consequently the maximum of
its final valid lower bounds is a global lower bound, and the smallest upper
bound is returned with the corresponding validated path. An interrupted
portfolio retains this interval and its incomplete termination status.
`exact` still means the requested numerical B&B gap was closed, not a rational
zero-error solution of the complete TSPN.

For controlled comparisons, `--search-strategy best-bound` and
`--search-strategy dfs-bfs` run the policies independently.
`--portfolio-no-sharing` runs the same two-worker race with incumbent exchange
disabled; proof cancellation remains active. These flags are mutually
exclusive with an explicit isolated strategy in the CLI. The Python public
`tspn-benchmark` command exposes the same switches and records two native
workers versus one Fekete worker; that is a resource advantage, not a matched
single-core comparison. The portfolio is opt-in and does not promise to be
faster than either standalone strategy on every input.

Result telemetry includes `portfolio_runs`, `portfolio_winner`, publication
and import counts, `portfolio_proof_seconds`, and `portfolio_join_seconds`.
The main `seconds` includes both startup and joining workers. Work counters
and phase/oracle times sum worker work and can exceed wall time; `peak_queue`
is the sum of individual peaks, an upper estimate of simultaneous occupancy.
Initial-solution metadata and the retained trace belong to the worker supplying
the returned path, followed by a `portfolio_complete` trace event. On a worker
exception its partial statistics are unavailable, but reserved calls remain
included in the global `calls` counter. The winner is absent when neither
individual worker proves the requested gap.

## Cyclic insertion bounds

For arbitrary vectors `|u_i| <= 1`, the cycle dual bound is

```
D(u) = sum_i min_(v in C_i) (u_(i-1)-u_i) dot v.
```

The inequality `u_i dot (q_(i+1)-q_i) <= |q_(i+1)-q_i|` telescopes around the
cycle. Support minima are taken over polygon vertices. This formula has no
fixed-endpoint term. Translating every vertex by the same origin leaves it
unchanged.

Splitting edge i to insert a region changes only the new region's support and
the two neighboring supports. These terms are updated from the parent dual.
The one-region case combines both neighbor changes into its single support
term. Unit-ball directions and all support arithmetic are rational; a floating
norm is rounded upward and checked by rational squaring before division. The
reported result is rounded downward and bounded below by zero. This remains
valid even when reference contacts are infeasible or coincident. The endpoint
insertion bound is unchanged.

**Binary64 screening (default since 2026-10-09).** Without an inherited dual
(`dual` off, the protocol configuration), `insertion_lower_bounds` evaluates
the same formula with `tpp_convex_binary_cycle_insertion_bounds`
(`tpp/convex/dual.h`). Each link direction, and each direction to and from the
proposed contact, is a binary64 vector shortened by `2^-49` whose norm is
proved at most one. A computed squared norm at most `1-2^-50` proves it in the
standard model; otherwise the interval squares of `binary_dual_feasible` do,
and an unproved vector becomes zero. Every support is the plain binary64
minimum widened by an a priori bound of its rounding error, and sums are
rounded outward. The proof is in
[`unordered-tpp.md`](unordered-tpp.md#certificado-convexo-e-interseções).
Each bound is therefore a weak-duality value on the child cycle, exactly like
the rational one, with slightly different vectors; it is not a tolerance.
The rational evaluation below is kept: it is the test reference, the
evaluation with an inherited rational dual, and the fallback outside the
interval environment. `--no-cycle-binary-insertion` /
`cycle_binary_insertion = false` restores it for ablation.

On the SoCG regression this removes the remaining rational hot spot: the
search is unchanged (the same calls in 48/48 cases) and the screening time
drops from 22.4 to 0.6 s (results in
[`tspn-cycle-insertion-binary64-2026-10-09`](../../benchmarks/results-saved/README.md#tspn-cycle-insertion-binary64-2026-10-09)).
`insertion_bound_calls` and `insertion_bound_seconds` measure the screening of
the branched region (one call per insertion expansion).

Within one cyclic sibling set, each polygon vertex is imported and translated
to the common rational origin once. Proposed insertion contacts and link
directions are also shared between the ordinary and inherited dual evaluations.
Only support coefficients differ between these evaluations. This preserves
the exact support formulas and downward-rounded output while removing repeated
conversion and contact construction. It does not change branch selection or
pruning bounds for a fixed set of directions.

The support scan represents the exact translated binary64 vertices with a
common power-of-two denominator `D`: each vertex is `(X/D,Y/D)` with integer
`X,Y`. Original vertices and the origin are imported exactly, scaled to `D`,
and subtracted as integers, avoiding per-coordinate rational normalization.
Thus neither overflow nor rounding of a floating subtraction changes these
coordinates. For a rational
normal `(a/b,c/d)`, every support candidate has numerator `X*a*d + Y*c*b`
over the same positive denominator `D*b*d`. Comparing these integers therefore
selects exactly the same minimum as the rational dot-product scan. Only the
minimum is normalized back to a rational. Equal normal denominators are shared
directly. This is an exact representation change, not coordinate snapping or
an acceptance tolerance; dual directions, rational sums and downward rounding
remain unchanged. Tests compare the support value by exact rational equality,
including subnormals, opposite finite extremes and large rational normals.

## Independently selectable acceleration experiments

The native CLI and public benchmark command accept repeated
`--cycle-optimization cache|dual|features|lazy|root|branch|one-tree|learn|memo|bound-first|dual-screen|interval|share-bounds|proposal-bound|primal-starts` options. Each changes
the existing solver; none introduces a parallel implementation. They initially
remain opt-in so that regressions are measurable against the same executable.

- `cache` reuses canonical rational polygons across calls and immutable prepared
  geometry across candidate checks, with a separate workspace per worker.
- `dual` transports the parent's rational unit-disk directions. Inserting a
  region duplicates the split link's direction; replacing a convex piece keeps
  the same link indices. New nonzero contacts replace their directions, while
  zero links retain inherited witnesses. Insertion screening takes the maximum
  of the original and inherited-witness bounds, so screening cannot weaken.
  The extra rational work and per-frontier-node witness storage can cost more
  than the saved oracle calls.
- `features` transports the active vertex/edge tuple with the contacts and marks
  only the inserted/replaced feature unknown. A local repair and reconstruction
  precedes the ordinary constructor; every resulting candidate is certified.
- `lazy` queues children with their already-certified insertion/parent bounds
  and solves the convex relaxation only when a child is selected for expansion.
  A newer incumbent may prune it before that call. The global frontier includes
  these unsolved children, including at interruptions; an unrefined bound is
  never promoted to an exact relaxation. This is a two-stage bound policy and
  can delay incumbent discovery compared with evaluating all siblings eagerly.
- `root` also enables the separated triple root for best-bound search. As with
  DFS/BFS, all tours contain one unoriented cyclic order on three fixed labels;
  the reduction loses no tour. Its pairwise region-distance heuristic costs
  O(n²) geometric distance queries before the search.
- `branch` compares insertion lower bounds for up to three farthest unvisited
  regions, choosing the largest minimum child bound. Decomposition branches
  keep their existing priority. This adds at most three insertion-screening
  passes per branch; it orders a complete search and is not an extra pruning rule.
- `one-tree` adds a certified all-region Held–Karp bound, described below.
- `learn` ranks missing regions using bound gains observed in previous children,
  without extra oracle calls. It takes precedence over `branch` when both are
  selected; both remain opt-in.
- `memo` caches previously certified relaxation results for identical cyclic
  constraints, with rotation/reversal normalization as described below.
- `bound-first` checks inherited contacts and a rational dual bound before
  seeking another construction or a complete KKT certificate. The existing
  cutoff propagation remains in effect with either setting.
- `dual-screen` extends dual reuse to decomposition children, using one new
  support term per replacement piece before invoking the oracle.
- `interval` selects the existing certificate's rigorous interval filter.
- `proposal-bound` first requests the finite floating feature proposal and its
  independent certificate, without rational reconstruction or boundary search.
  The B&B adapter accepts its interval only when the lower bound reaches the
  incumbent cutoff or its width meets the existing `oracle_relative_gap`/node
  gap request. Otherwise it performs the full cycle solve with the retained
  contacts and remaining time. Precise leaf requests use the full path directly.
  This schedules construction effort; it adds no oracle acceptance tolerance.
  The cycle constructor receives no gap parameter and returns `ProposalLimit`
  for an exhausted proposal unless an exact optimum or cutoff was certified.
  A partial result never asserts exact cycle optimality. Any returned lower
  bound remains valid for all extensions of the node, and the feasible path is
  checked against all original regions before it can update the incumbent.
  On interruption between stages, the best completed feasible path and maximum
  certified lower bound survive; unfinished work remains in the frontier.
- `share-bounds` imports compatible certified subcycle bounds in a sharing
  portfolio before reserving an oracle call.
- `primal-starts` diversifies the existing greedy/2-opt/contact initial
  heuristic from up to seven additional original polygon vertices. All starts
  share an extra budget capped at 5% of the solve deadline, after retaining the
  ordinary initial candidate. They reuse the original regions and the existing
  feasibility checks, and can only supply incumbent upper bounds. They neither
  restrict the free cyclic order nor alter any convex relaxation or lower bound.

All retain the same numerical B&B contract below. The experiment matrix,
including unsuccessful variants and interrupted runs, belongs to the saved
campaign rather than to this algorithm contract.

## All-region one-tree bound

`options.cycle_one_tree` includes every original label, including labels absent
from the partial cycle. A label denotes its convex hull until the node assigns
a decomposition piece. For each pair of these convex sets, existing floating
point projections propose a separating direction. Its norm is bounded above
and checked with rational arithmetic, giving an exact vector `|u| <= 1`. Then

```
c_ij = max(0, min_(y in C_j) u dot y - max_(x in C_i) u dot x)
```

is no greater than any distance between the two regions. Support arithmetic
is rational; the stored binary64 cost is rounded downward and reimported as
an exact dyadic rational. Incorrect or degenerate floating proposals can only
weaken this bound. Intersections, containment and touching require no epsilon.

For arbitrary vertex prices pi, construct a minimum spanning tree on all labels
except label 0 and add the two cheapest distinct edges incident to label 0,
using exact weights `c_ij + pi_i + pi_j`. Its weight minus `2 sum_i pi_i` is a
valid lower bound: every Hamiltonian cycle is such a one-tree, and each cycle
has degree two at every label. Every feasible geometric tour supplies a visit
per label; shortcutting these visits and replacing distances by `c_ij` cannot
increase its length. The graph need not satisfy the triangle inequality.
For two labels the bound is `2 c_01`; fewer labels give zero.

The bound is maximized with the existing node bound, never added to it.
Partial-cycle adjacent labels are not forced to be graph edges: future
insertions may separate them. Thus this bound deliberately relaxes visit order.
Each evaluated one-tree is safe independently of the price update heuristic.
At most 32 subgradient iterations propose prices in doubles, interpreted exactly
for the next tree computation. The iteration count is a work budget, not an
optimality tolerance. A nearest-neighbor graph tour and at most n improving
2-opt sweeps supply an ascent target. This graph tour is **not** a feasible
geometric incumbent: consecutive graph edges may require incompatible contacts.
It is used only to choose price steps and stop ascent.

Distances and geometry are cached per worker. A bound cache, cleared at 4096
assignments, is keyed by the complete hull/piece assignment; visit order does
not affect it. Root computation supplies all insertion descendants with the
same bound. Only decomposition children need another assignment lookup or
computation. An interrupted computation returns an already valid bound or
zero. Caches never contain pointers to temporary polygons.

For polygon sizes m_i and m_j, a new pair uses O(m_i m_j) floating proposal work
and O(m_i + m_j) rational support work. After pair caching, I tree passes take
O(I n²) rational arithmetic operations; the graph-tour proposal takes O(n²)
plus at most O(n³) work for the capped 2-opt sweeps. The working graph occupies
O(n²) space. These are arithmetic-operation counts, not constant-cost bit
complexity claims. Touching tessellations can have zero pair distances along
an entire graph tour, leaving this relaxation weak even when the geometric
optimum is positive.

`one_tree_calls`, `one_tree_cache_hits`, `one_tree_iterations`,
`one_tree_distance_queries`, `one_tree_improvements` and `one_tree_seconds`
separate its cost from convex-oracle work. Portfolio counters sum both workers.

## Branch ranking from observed gains

`options.cycle_learned_branching` maintains an online mean of
`max(0, child_bound - parent_bound) / missing_region_distance`, separately per
original region and per branch type (insertion or decomposition). Only ordinary
child oracle results contribute; a later precision refinement does not train
the same child twice. Nonfinite observations are ignored. Workers keep separate
means, so collecting a parallel sibling batch cannot race with another worker.

For an uncovered region with k observations, mix its mean with the global mean
for the same branch type using weight `k / (k + 4)`. Rank by its current
missing-region distance times one plus that mixed mean. Four prior observations
regularize sparse data; this is not an acceptance tolerance. Before any data,
ranking is the existing farthest-region choice. Ties follow original label order.
Ranking takes O(n m) work for a partial sequence of length m and O(n) stored
statistics. It calls no additional convex oracle or insertion-bound evaluator.

All insertion positions and decomposition pieces remain represented, and the
learned score is never used as a pruning bound. Search coverage and the B&B
gap contract therefore remain unchanged. `learned_branch_observations`,
`learned_branch_decisions` and `learned_branch_changes` expose whether it was
actually exercised. Gains observed on one instance need not predict useful
branching on another; statistics are reset for every solve.

## Reusing certified relaxations

`options.cycle_memo` caches the complete ordered `(original region, hull/piece)`
tuple of a convex relaxation. This includes partial B&B sequences: their
relaxations are themselves closed cycles. Each original label occurs once, so
starting at the smallest tuple and comparing the forward and reverse sequences
gives an O(k) canonical key. Distinct piece assignments and non-equivalent visit
orders have different keys. The search's normalized hulls and lazily prepared
pieces are immutable for the entire cache lifetime.

A stored path is reused only if its certified bound reaches the current cutoff
or its certified interval already meets the current oracle accuracy request.
Contacts and feature hints are permuted back to the requested cyclic order.
The existing exact certificate independently rechecks the contacts before every
cache return. A stronger stored lower bound, including one from rational
recovery, remains valid because the geometric constraints are identical.
Reversal and rotation preserve the sum of link lengths and feasibility.
The cache does not transfer a bound to merely similar or differently ordered
polygons. A tighter request may force another solve.

Outside the cooperative portfolio, each oracle worker owns a separate cache,
cleared when it reaches 4096 entries and destroyed at the end of the solve.
Lookup takes O(k log M) tuple comparisons
for M entries, followed on a hit by the existing O(N+k³) certificate bound;
storing a result takes O(k) contacts/features plus its key. It saves construction
work, not independent verification. `calls` still counts relaxation requests,
including cache hits, so the global call-budget contract is unchanged.
`cycle_memo_queries`, `cycle_memo_repeated` and `cycle_memo_hits` distinguish
lookups, retained matching keys, and successfully reused results. Eviction can
hide repetitions; zero hits in a sample is not proof that none occur elsewhere.

With `portfolio` and incumbent sharing enabled, `memo` also shares these results
between the two searches. A shared key contains original labels and **complete
polygon coordinates**, rather than assuming that both workers give a piece the
same local index. Canonical rotation/reversal is unchanged. A short separate
mutex protects lookup/publication; a retrieved snapshot is copied under the
lock and independently certified after releasing it. Concurrent publications
retain the stronger lower bound and the feasible path with the smaller upper
bound, always for identical constraints. The shared cache has the same 4096
entry cap; its keys take O(N) coordinates and lookup O(N log M) comparisons.
`--portfolio-no-sharing` disables this exchange as well as incumbent exchange.
All workers still cover their original search space, and reservation/proof
cancellation rules are unchanged. Performance comparisons must use the same
two-worker portfolio with and without memoization.

`options.cycle_bound_first` selects the convex solver's early inherited-contact
and dual-bound checks. A floating support estimate filters unpromising extra
checks; inaccurate estimates can only change work, never authorize acceptance.
Any early pruning still requires exact membership and a
valid rational support bound; no full optimality claim follows from reaching
the cutoff. `cycle_certificate_cutoff_skips`, `cycle_initial_contact_checks` and
`cycle_initial_contact_accepts` expose the avoided work. Both new options are
independent, disabled by default, and preserve the existing numerical B&B
contract below. Portfolio counters sum work across the two private searches.

## Scope of exactness and limits

### Decomposition screening and compatible shared bounds

`cycle_dual_screen` constructs the parent's unit-disk dual directions using
the existing exact routine. All unchanged polygon support minima are summed
once; for a replacement piece only its own support term is substituted. This
does not require the parent's contacts to belong to the new piece. Weak duality
proves the resulting bound for every candidate in that child. It is maximized
with the inherited node bound and cannot weaken pruning. Insertion already
has its own incremental dual screening; this option does not duplicate it.
For N parent vertices and M vertices across all replacement pieces, the batch
costs O(N+M+k) rational operations and O(k+b) storage for k visits and b pieces.
Rational bit sizes remain part of the arithmetic cost. Counters record children,
prunes and total screening time.

`cycle_share_bounds` uses the existing shared, full-geometry cycle cache. A
query checks the target key and its k one-region-deleted cyclic subsequences,
canonicalized under rotation and reversal. Every remaining polygon must match
by all coordinates. A feasible tour for the target can be shortcut to any such
subcycle without increasing length (triangle inequality), so a certified
lower bound for that subcycle also bounds the target. No path is imported by
this operation and no new candidate requires certification. Published bounds
come only from certified convex relaxations, never from a node bound that might
include additional unrepresented constraints. Ordinary memoized candidate
returns continue to be independently verified.

There is no transfer from a larger constraint set, a different cyclic order,
or a merely similar polygon. One-deletion lookup deliberately does not search
all 2^k subsets or assume containment between differently rounded pieces.
`--portfolio-no-sharing` disables these queries and publications; outside a
sharing portfolio the option does no work. `memo` is independently selectable.
Queries happen before oracle reservation and may avoid a call entirely, just
like insertion screening. Frontier bounds and proof cancellation are unchanged.

The optional bound index interns complete polygon coordinate vectors into
IDs using exact container equality; it does not use unchecked fingerprints or
workers' private decomposition IDs. For N target vertices, P distinct polygon
geometries and M cache entries, one query costs O(N log P + k² log M) in the
worst case and O(N+k) temporary key storage. Stored compact bounds cost O(Mk)
plus interned geometry. The existing cap is 4096 cache entries, and eviction
clears the corresponding bound index. This index is built only when this option
is queried, leaving ordinary memoization unchanged. A separate mutex protects
snapshots and bound lookup; search policies
remain independent. `cycle_shared_bound_queries`, `hits`, `improvements`,
`prunes` and `seconds` distinguish reuse from its overhead. All three new
options remain disabled by default pending stratified measurements.

### Existing numerical B&B contract

The convex rational oracle is exact on its input regions, with objective stored
as a sum of radicals. The complete TSPN B&B retains the existing numerical
geometry/normalization, visit tolerance and user-selected global gap contract.
Its `exact` field means that this requested gap was closed; it does **not** mean
a zero-error rational solution of the complete nonconvex TSPN. Bounds and
`termination` are returned for time, call and numerical limits. An oracle failure
does not authorize pruning or a claim of optimality. The time limit is
cooperative. Cycle calls check it inside refinement, boundary search, maps and
certificates. A single arithmetic primitive or a decomposition
can still exceed it; this is not a hard process deadline. An interrupted
relaxation retains its completed certified bound and any verified contacts.
The B&B retains the unfinished node and every unevaluated sibling in its
frontier, combining inherited bounds with completed certificates. It returns
the independently validated global incumbent with an incomplete status unless
those bounds already close the requested numerical gap.

The incumbent pruning cutoff is preserved when a cycle relaxation enters
rational recovery, including the adapter's secondary recovery for a wide
rounded-contact gap. That recovery also receives the already verified contacts
as a proposal. It may return `CertifiedBound` once the independent global lower
bound reaches the cutoff; this is sufficient to prune and does not assert an
exact optimal tour. Default standalone rational cycle solves still require
the exact optimality certificate. When that secondary recovery is optimal, its
contacts rounded to binary64 replace a longer double path. Rounded double
contacts can stay well above the relaxation optimum on points and segments,
which have no interior to round into; the search then kept a node whose path
covered every region but whose gap never closed (`numerical_limit`). The
replacement path is used like any relaxation path, only through the B&B's own
visit checks and length. Its upper bound is the exact optimum's outward-rounded
length (since 2026-10-08; before, the plain binary64 length of the rounded
contacts, which was not a proved bound).

With n regions, orders contribute at most `(n-1)!/2` unoriented cyclic orders
for n>=3, and decomposition choices multiply the worst-case search. Each branch
contains at most one insertion and one refinement per region. The search remains
exponential; the convex oracle's operation/bit complexity is documented in
[convex-cycle.md](convex-cycle.md). Cyclic dual screening reuses all unchanged
supports and avoids solving children already excluded by their lower bound.

## Evaluation on GTSP-derived instances with points and segments

Paula's corpus (GTSP-Lib/MOM-Lib clusters turned into nonconvex polygons,
points and segments; local, not redistributable) exposed four defects that the
OSM/random/tessellation suite never reached: cycles rejected points and
segments; the common-region clip and anchored membership treated a point or a
segment as the whole plane or line; rounded double contacts on degenerate
regions left a relaxation gap the search could not close; and a block of
coincident contacts whose optimum is a crossing of two edges was unrepresentable
by vertex/edge features, so the rational boundary bisection ran until its
deadline. All are fixed, with regression tests that need no third-party data.
Above about 40 regions these TSPLIB-like instances remain open for every
method tried (ours, Fekete, the paper's CPLEX model). Stronger branching,
lookahead, DFS/BFS, primal starts and exact LNS did not close more; graph
relaxations with region-to-region distances (Held–Karp) or per-region triple
costs are weaker than the existing insertion relaxations even at the optimal
order, so neither was implemented.
With a point region the tour is an endpoint path, and its oracle calls contain
points and segments; since 2026-10-08 they also try the fixed-endpoint binary64
oracle first (`options.float_degenerate`, `--no-float-degenerate` disables it;
[contract](certified-convex-oracle.md#oráculo-em-ponto-flutuante-experimental-2026-10-08)),
and cycle calls with segments use the binary64 stage above.
On those open instances the deliverable is a good tour, not a certificate. The
draft's appendix reports, per instance, the best of ten runs of Paula's
ILS-BCD (each up to 1,200 s of CPU). The optional initial ILS
(`--primal-ils F`, [`unordered-tpp.md`](unordered-tpp.md#busca-local-iterada-inicial-2026-10-07))
reproduces her neighbourhoods and acceptance and adds what a coordinate descent
cannot do: windows of eight consecutive regions are re-solved exactly, order
and contacts, by this branch and bound between fixed contacts. That step is
what closes the gap on the instances with heavily overlapping polygons, where
consecutive contacts coincide.
Results and protocol: [`tspn-paula-cycle-2026-10-06`](../../benchmarks/results-saved/README.md#tspn-paula-cycle-2026-10-06).

With that incumbent the certificate is limited by the lower bound alone.
`--multi-insertion-bound` raises every expanded node's bound with the
insertion gains of all absent regions under the node's contact-direction
dual. Gains of non-adjacent gaps add, adjacent gaps lose at most the width of
their shared region, and a covering dual prices the worst assignment of
regions to gaps. It works for cycles and anchored paths, and its argument is
in [`unordered-tpp.md`](unordered-tpp.md#limite-de-inserções-múltiplas-2026-10-07).
Once the incumbent is near-optimal, diving and eager child evaluation stop
paying for themselves, so the lower bound also rises with
`--cycle-optimization lazy --lazy-oracles --dive-interval 0`. All three stay
opt-in. Measurements:
[`tspn-paula-lower-bound-2026-10-07`](../../benchmarks/results-saved/README.md#tspn-paula-lower-bound-2026-10-07).

**Revised 2026-10-09.** With the binary64 oracle and screening, the node
count limits the lower bound, and the multi-insertion bound (`O(m·k)` gains
per node) no longer pays: on the 25 validation instances `lazy` without dives
closes 39.5% of the base gap and adding `--multi-insertion-bound` only 11.5%.
The recommended pipeline for a certificate on Paula's open instances is
`--primal-ils 0.5 --cycle-optimization lazy --lazy-oracles --dive-interval 0`.
At 600 s on the 38 instances open at 60 s, it reaches a median certified gap
of 2.7% (5.7% on 2026-10-07) and closes 17 (4). Without a supplied or ILS
incumbent the defaults (eager evaluation with dives) stay better. See
[`tspn-search-retune-2026-10-09`](../../benchmarks/results-saved/README.md#tspn-search-retune-2026-10-09).

## Next directions (2026-10-09)

Done in this round, from the list after the binary64 cycle oracle:

- **Cyclic insertion screening in binary64** (item 1):
  `tpp_convex_binary_cycle_insertion_bounds`, the same search in 48/48 SoCG
  cases, screening 22.4 → 0.15 s. The inherited rational dual (`dual`) and
  `dual-screen` stay rational; both are off and showed no gain.
- **Proofs instead of margins** (item 2): path insertion, multi-insertion
  and the endpoint floor are proved (contract in
  [`unordered-tpp.md`](unordered-tpp.md#certificado-convexo-e-interseções)).
  Every pruning bound of the free-order B&B is now rational, interval or
  backed by an a priori rounding-error bound.
- **Re-tune** (item 3): with an ILS incumbent, `lazy` without dives raises
  the lower bound most, and the multi-insertion bound no longer pays (see
  [`tspn-search-retune-2026-10-09`](../../benchmarks/results-saved/README.md#tspn-search-retune-2026-10-09)).
- **Records** (item 7): the report's "Alcance" is updated, and
  `tpp-convex-path-oracle-replay` accepts calls with points and segments
  (50lin105: 5,051 sampled calls, all closed in binary64, no incompatible
  bounds).
- **New:** the binary64 proofs in the cycle oracle use the a priori error bound
  (1.29× on SoCG), and the best-bound frontier has a memory budget. The
  frontier now grows ~50× faster than before, and 60 s of 100pr1002 reached
  3–3.8 GB.

Open, in order of expected value:

1. **Frontier memory per node.** Long runs are now bounded by memory, not by
   the oracle. A queued node costs ~216 B of header (64-bit indices, the
   learning fields, an empty rational dual), ~16 B per contact of its path and
   the features. Above the budget, nodes are stripped and solved again, or
   discarded. Shrinking the header (32-bit indices, rarely used fields moved
   out) and keeping paths in an arena would roughly double the nodes per
   budget. Chunked node storage would also remove the transient 3× of a
   vector reallocation, a 1.3 → 1.8 GB RSS peak at a 1 GiB budget.
2. **Cost of the multi-insertion bound.** Its `O(m·k)` gains per node
   (`best_contact` plus three supports each) now cost more than the node's
   oracle calls; on the closed instances it is 3–5× slower. Incremental gains
   (cache per region and gap of the parent's contacts) or evaluating it only
   at nodes near the frontier minimum could make it pay again.
3. **The oracle is again most of the time.** SoCG: 1.8 of 2.7 s
   (construction 0.8, polish 0.6, proofs 0.4). Visit checks take ~20% on the
   largest cases. Polish iterations and the construction kernel are the
   remaining constant factors.
4. **`memo` and `share-bounds`.** A hit rechecks the stored contacts with the
   exact certificate (rational), which now costs more than solving the call
   again (~13 µs), and binary64 segment contacts are never reused. Measure the
   portfolio without `memo`, then drop it from the portfolio defaults or
   re-prove hits with the binary64 cycle proof.
5. **Feature hints after a polish closure** (item 4). Low value now: the
   construction is 0.8 of 2.7 s on SoCG, and ~10% of calls close by the polish.
6. **Calls the polish leaves open:** 3 in 99 M on Paula's collection (80rd400).
7. **Measurements.** Repeat on an idle machine (dantzig, one process,
   repetitions) and run the full 558-case path corpus once with this binary
   (the identity checks covered 40 cases).

## Use and validation

```cpp
tpp::UnorderedTppSolveOptions options;
options.max_seconds = 3;
auto result = tpp::tpp_nonconvex_tspn_solve(polygons, options);
```

The existing CLI accepts `tpp-unordered --cycle`; its text input format stays
unchanged and the two input endpoints are ignored. A supplied `--initial-path`
must be closed and feasible, and can start anywhere. JSON includes `mode` to
prevent comparing different formulations accidentally.

`tpp-tspn-tests` compares the B&B with exhaustive cyclic order/piece enumeration,
checks missing-depot behavior, zero tours, overlaps and nonconvex decomposition,
and checks lower/upper bounds under call caps. Its enumeration intentionally
shares the convex oracle; the external comparison is a separate correctness
check. `tpp-unordered-tests` protects the old endpoint contract.

`python3 benchmarks/tpp.py tspn-benchmark --output NEW_DIRECTORY` compares the
whole native solve with the pinned Fekete B&B's original SOCP backend. It builds
the external source list unchanged outside the submodule. Source revisions,
hashes, settings, inputs, raw tours, intervals, timings and analysis are saved
together. Independent tour validation uses binary-rational intersection and
containment tests plus numerical distance/length diagnostics. Raw feasibility
and reported gap closure are kept separate; coordinate equality is not required.

## Oracle timing diagnostics

The shared B&B exposes `oracle_profiled_calls`, `oracle_max_call_seconds`,
`oracle_fallback_call_seconds`, and two fixed-size arrays:
`oracle_call_histogram` and `oracle_seconds_histogram`. Their exclusive buckets
are ≤10 µs, (10,100] µs, (0.1,1] ms, (1,10] ms, (10,100] ms, (0.1,1] s, >1 s.
These are reporting bins, not geometric tolerances or stopping criteria.
They reuse the existing per-call clock and count completed search/refinement
requests, including memo hits. Initial optional polishing and failed requests
are excluded. Cooperatively interrupted calls are included; a hard-killed
process still has no final counters. Counts sum to `oracle_profiled_calls`; bucket times sum
to `profile.convex_oracle_seconds`. Portfolio aggregation sums counts and work
and takes the maximum of the workers' longest calls.

Fallback-attributed time is the **whole call** that used recovery, not exclusive
rational recovery time. The cycle adapter additionally reports exclusive
`cycle_construction_seconds`,
`cycle_certification_seconds`, `cycle_rational_recovery_seconds`, and for the
binary64 stage `cycle_interval_proof_seconds` and `cycle_polish_seconds`
(construction inside the stage counts as construction). As in the path mode,
each cycle call closes in exactly one way:
`calls = oracle_interval_bound_calls + oracle_float_calls + oracle_rational_calls`,
the binary64 stage by an interval proof of a construction candidate, by the
polish, or neither (the exact path, an interruption, or a memo hit rechecked by
the exact certificate; a relaxation with fewer than two regions counts as an
interval closure). `fallback_calls` still counts calls with a rational
recovery; `cycle_polish_calls` and `cycle_polish_newton_iterations` count polish
attempts and their iterations.
Certification includes all checks, including checks inside rational recovery;
rational recovery excludes that certification time. Construction includes
preparation and binary64 proposal work. Their sum is at most the full oracle
time (adapter overhead remains outside them). Older records lacking these
fields cannot be retrospectively split into phases. These constant-memory
diagnostics do not change geometry, pruning, tolerances, or the solver's
numerical gap contract.

`benchmarks/tpp.py tspn-benchmark --capture-oracles` writes local diagnostic
JSONL under the campaign's `oracle-captures/`. Each input is flushed before the
oracle starts, followed by its completed phase timings and bounds. A missing
end record identifies an in-flight request after a process failure. Capture is
disabled by default and its I/O overhead makes such runs unsuitable for a
speedup claim. Captures include the normalized regions, inherited contacts and
features, incumbent cutoff and remaining budget; they are local campaign data,
not regression fixtures to archive wholesale.
