#include "tpp/convex/certified.h"
#include "certified_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace tpp {
	CertifiedConvexTppResult tpp_convex_solve_certified(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		DynamicConvexTppWorkspace &workspace, double tolerance, double cutoff
	) {
		using Clock = std::chrono::steady_clock;
		using namespace certified_detail;
		auto duration = [](auto since) { return std::chrono::duration<double>(Clock::now() - since).count(); };
		const auto began = Clock::now();
		CertifiedConvexTppResult result;
		if (polygons.empty()) {
			result.path = {start, target};
			result.lower_bound = result.upper_bound = start.distance_to(target);
			result.seconds = duration(began);
			return result;
		}
		double scale = std::max(1.0, start.distance_to(target));
		for (const auto &polygon : polygons) for (auto vertex : polygon)
			scale = std::max(scale, start.distance_to(vertex));
		const double safety = 1e-12 * scale * (polygons.size() + 1);
		Polygon contacts;
		const auto geometric_began = Clock::now();
		tpp_convex_solve_binary_search_lazy(start, target, polygons, workspace, result.path);
		result.geometric_solver_seconds = duration(geometric_began);
		const auto certificate_began = Clock::now();
		bool geometric_path_valid = recover_contacts(result.path, polygons, contacts);
		if (!geometric_path_valid && repair_contacts(result.path, polygons, contacts)) {
			geometric_path_valid = true;
			result.repaired_geometric_path = true;
		}
		if (geometric_path_valid) {
			result.upper_bound = path_length(contacts);
			result.lower_bound = std::max(start.distance_to(target), dual_bound(contacts, polygons) - safety);
			if (result.upper_bound - result.lower_bound <= tolerance || result.lower_bound >= cutoff) {
				result.path = std::move(contacts);
				result.certificate_verification_seconds = duration(certificate_began);
				result.seconds = duration(began);
				return result;
			}
		}
		result.certificate_verification_seconds = duration(certificate_began);
		result.used_fallback = true;
		result.fallback_geometric_path_invalid = !geometric_path_valid;
		result.fallback_certificate_gap = geometric_path_valid;
		const auto fallback_began = Clock::now();
		if (!geometric_path_valid) {
			result.upper_bound = std::numeric_limits<double>::infinity();
			result.lower_bound = start.distance_to(target);
		} else result.path = std::move(contacts);
		const auto long_double_began = Clock::now();
		result = refine_long_double(start, target, polygons, tolerance, scale, safety, cutoff, std::move(result));
		result.fallback_long_double_seconds = duration(long_double_began);
		if (result.upper_bound - result.lower_bound > tolerance && result.lower_bound < cutoff) {
			result.used_extended_precision = true;
			const auto extended_began = Clock::now();
			result = refine_extended_precision(start, target, polygons, tolerance, scale, safety, cutoff, std::move(result));
			result.fallback_extended_precision_seconds = duration(extended_began);
		}
		result.fallback_seconds = duration(fallback_began);
		result.seconds = duration(began);
		return result;
	}
}
