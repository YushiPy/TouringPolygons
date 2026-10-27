#pragma once
#include "tpp/geometry/vec2.h"
#include <cstddef>
#include <limits>
#include <vector>

namespace tpp {
// A closed disk; radius 0 is a point.
struct ConvexDisk { Vector2 center; double radius = 0; };

// GapClosed: U-L <= max_gap; CertifiedBound: L >= lower_bound_cutoff;
// Limit: neither within the iteration budget (the bounds are still valid).
enum class DiskCycleStatus { InvalidInput, GapClosed, CertifiedBound, Limit };

struct DiskCycleOptions {
    double max_gap = 0; // absolute
    double lower_bound_cutoff = std::numeric_limits<double>::infinity();
    // Optional warm start, one contact per disk; pulled toward each center.
    std::vector<Vector2> initial_contacts;
    std::size_t max_iterations = 600;
    // Barrier parameter of the first level in scaled coordinates (radius of the
    // instance about 1); 0 picks it from the gap and the warm start.
    double mu_start = 0;
    // The geometric construction first (Alhazen updates, the type closed by
    // Newton on the reflection law); false runs only the barrier.
    bool construction = true;
};

struct DiskCycleResult {
    DiskCycleStatus status = DiskCycleStatus::InvalidInput;
    // Contacts proved inside their disks; upper_bound bounds their cycle.
    std::vector<Vector2> contacts;
    double lower_bound = 0, upper_bound = std::numeric_limits<double>::infinity();
    // Barrier Newton iterations and levels (zero if the construction closed).
    std::size_t newton_iterations = 0, levels = 0;
    // The construction: whether its bounds closed the call, its rounds
    // (close the type, then one sweep), and its Newton steps on the angles.
    bool construction_closed = false;
    std::size_t construction_rounds = 0, closing_iterations = 0;
};

// Shortest closed tour visiting the disks in the given cyclic order. Proposals
// come from geometry first: Alhazen updates (the point of a disk with the
// shortest detour between its neighbours: a straight passage, or the
// reflection point on its circle) give a combinatorial type, which Newton on
// the reflection law closes; a log-barrier Newton method on the smoothed
// length is the fallback. Both bounds are proved with directed rounding on the
// input, independently of the proposal:
// - U: the length of a cycle through contacts proved inside their disks;
// - L: the weak dual D(u) = sum_i min_{v in D_i} (v-r).(u_{i-1}-u_i), with each
//   u_i proved in the unit disk; for a disk the minimum is
//   (c_i-r).w - radius_i |w|, so no vertex search is needed.
// L <= OPT <= U always holds; contacts are never exact optima.
DiskCycleResult tpp_convex_solve_cycle_disks(const std::vector<ConvexDisk> &disks,
    const DiskCycleOptions &options = {});

// The point of the disk minimizing |x-a|+|x-b| (Alhazen's problem): the foot
// of the center on ab if the segment meets the disk, else the reflection
// point on its circle. A proposal in binary64, not a proved point.
Vector2 tpp_convex_disk_detour_point(Vector2 a, Vector2 b, const ConvexDisk &disk);

// Lower bounds on the shortest cycle through `disks` (k >= 1) with `inserted`
// added at position p = 1..k, between disks[p-1] and disks[p mod k]; entry
// p-1 is position p. Weak duality as in tpp_convex_solve_cycle_disks, with
// directed rounding: the link directions of the cycle through `contacts`
// (one per disk), and for the inserted disk the directions to and from its
// point nearest the segment between the two neighbouring contacts (its
// detour point when the segment meets it; no Newton). Only three support
// terms change per position, so all k bounds cost O(k). Valid for any
// contacts; close to the child's optimum when the parent's contacts are
// optimal and the insertion leaves them in place. Empty on invalid input.
std::vector<double> tpp_convex_disk_insertion_bounds(const std::vector<ConvexDisk> &disks,
    const std::vector<Vector2> &contacts, const ConvexDisk &inserted);
} // namespace tpp
