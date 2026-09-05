#pragma once
#include "tpp/convex/solver.h"

namespace tpp {
	struct CertifiedConvexTppResult {
		std::vector<Vector2> path;
		double lower_bound = 0;
		double upper_bound = 0;
		bool used_fallback = false;
	};

	// Verifies the geometric oracle against a support-function dual bound.
	// Uses a smooth interior-point fallback if the certificate does not close.
	CertifiedConvexTppResult tpp_convex_solve_certified(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		DynamicConvexTppWorkspace &workspace, double tolerance
	);
}
