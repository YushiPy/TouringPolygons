#pragma once

#include "unordered_geometry.h"
#include "tpp/convex/rational.h"
#include <functional>
#include <map>

namespace tpp::unordered_detail {
    // Canonical rotation/reversal for a cycle of distinct immutable region labels.
    // Returns canonical-index -> original-index, also used to transport contacts.
    std::vector<size_t> canonical_cycle_indices(const std::vector<std::pair<size_t,size_t>> &);
	// Lower bounds for all insertion positions. References may be infeasible;
	// only the fixed endpoints and the ordered region definitions matter.
	// Optional insertion_contacts receives the existing open-path proposals.
	// Open-path insertion bounds split in two: the node's dual directions and
	// supports (independent of the inserted region) and the per-region bounds.
	// insertion_lower_bounds without an inherited dual is their composition.
	struct PathInsertionDual {
		std::vector<Vector2> directions;
		std::vector<double> supports;
		long double value = 0;
		double scale = 1;
	};
	PathInsertionDual path_insertion_dual(const Polygon &contacts, const std::vector<const Polygon *> &regions);
	// One position of path_insertion_bounds, with identical arithmetic.
	double path_insertion_bound_at(const PathInsertionDual &dual, const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, size_t position,
		Vector2 *insertion_contact = nullptr);
	std::vector<double> path_insertion_bounds(const PathInsertionDual &dual, const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, Polygon *insertion_contacts = nullptr);
	std::vector<double> insertion_lower_bounds(const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted,
		bool cycle = false, const ConvexRationalPolygon &inherited_dual = {},
        Polygon *insertion_contacts = nullptr);
    // Keep the parent's dual directions, replacing only one support term.
    // Each returned bound is valid even when the old contacts leave the piece.
    std::vector<double> cycle_replacement_lower_bounds(const Polygon &contacts,
        const std::vector<const Polygon *> &regions,const std::vector<Polygon> &pieces,
        size_t position,const ConvexRationalPolygon &inherited_dual = {});

    struct OneTreeResult {
        double lower_bound = 0;
        size_t iterations = 0, distance_queries = 0;
        bool cached = false;
    };
    // Exact graph arithmetic on downward-rounded, independently certified
    // pair-distance bounds. The iteration budget affects strength, not validity.
    OneTreeResult held_karp_bound(const std::vector<std::vector<ConvexRational>> &costs,
        double upper_bound, size_t iterations = 32, const std::function<bool()> &stop = {});
    class CycleOneTreeWorkspace {
        struct Region { const Polygon *input; ConvexRationalPolygon exact; };
        std::vector<Region> regions_;
        std::map<const Polygon *,size_t> ids_;
        std::map<std::pair<size_t,size_t>,ConvexRational> distances_;
        std::map<std::vector<size_t>,double> bounds_;
    public:
        // Region pointers must remain valid and immutable for this workspace.
        OneTreeResult bound(const std::vector<const Polygon *> &regions,
            double upper_bound, const std::function<bool()> &stop = {});
    };
}
