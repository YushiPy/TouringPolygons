#include "certified_internal.h"

#include <Eigen/Dense>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
	using Polygon = tpp::certified_detail::Polygon;

	template<class Real>
	tpp::CertifiedConvexTppResult refine(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &polygons,
		double tolerance, double scale, double safety, double cutoff, tpp::CertifiedConvexTppResult result
	) {
		using V = Eigen::Matrix<Real, 2, 1>;
		using M = Eigen::Matrix<Real, 2, 2>;
		struct Face { V normal; Real offset; };
		using std::sqrt;
		using std::log;
		using std::isfinite;
		Polygon q;
		const size_t n = polygons.size();
		std::vector<std::vector<Face>> faces(n);
		std::vector<V> x(n + 2, V::Zero());
		x.back() = V{(Real(target.x) - Real(start.x)) / Real(scale), (Real(target.y) - Real(start.y)) / Real(scale)};
		for (size_t i = 0; i < n; ++i) {
			for (auto v : polygons[i]) x[i + 1] += V{(Real(v.x) - Real(start.x)) / Real(scale), (Real(v.y) - Real(start.y)) / Real(scale)} / Real(polygons[i].size());
			for (size_t j = 0; j < polygons[i].size(); ++j) {
				const auto a = polygons[i][j], b = polygons[i][(j + 1) % polygons[i].size()];
				V normal{Real(a.y) - Real(b.y), Real(b.x) - Real(a.x)};
				if (normal.norm() == 0) continue;
				normal.normalize();
				faces[i].push_back({normal, normal.dot(V{(Real(a.x) - Real(start.x)) / Real(scale), (Real(a.y) - Real(start.y)) / Real(scale)})});
			}
		}
		if constexpr (std::numeric_limits<Real>::digits > 64) {
			if (result.path.size() == n + 2) for (size_t i = 0; i < n; ++i) {
				const auto p = result.path[i + 1];
				x[i + 1] = Real(.00001L) * x[i + 1] + Real(.99999L) * V{(Real(p.x) - Real(start.x)) / Real(scale), (Real(p.y) - Real(start.y)) / Real(scale)};
			}
		}
		auto objective = [&](const std::vector<V> &z, Real mu) -> Real {
			Real value = 0;
			for (size_t i = 1; i < z.size(); ++i) value += sqrt((z[i] - z[i - 1]).squaredNorm() + mu * mu);
			for (size_t i = 0; i < n; ++i) for (auto face : faces[i]) {
				const Real slack = face.normal.dot(z[i + 1]) - face.offset;
				if (slack <= 0) return std::numeric_limits<Real>::infinity();
				value -= mu * log(slack);
			}
			return value;
		};
		if (!isfinite(objective(x, .1L))) {
			if constexpr (std::numeric_limits<Real>::digits <= 64) return result;
			throw std::runtime_error("Convex fallback requires CCW convex polygons with positive area.");
		}
		for (Real mu = std::numeric_limits<Real>::digits > 64 ? 1e-5L : .1L; mu >= 1e-17L; mu *= .15L) {
			for (size_t iteration = 0; iteration < 80; ++iteration) {
				std::vector<M> diagonal(n, M::Zero()), off(n, M::Zero());
				std::vector<V> gradient(n, V::Zero()), step(n, V::Zero());
				for (size_t i = 0; i <= n; ++i) {
					const V d = x[i + 1] - x[i];
					const Real norm = sqrt(d.squaredNorm() + mu * mu);
					const V g = d / norm;
					const M h = (M::Identity() - g * g.transpose()) / norm;
					if (i > 0) { gradient[i - 1] -= g; diagonal[i - 1] += h; }
					if (i < n) { gradient[i] += g; diagonal[i] += h; }
					if (i > 0 && i < n) off[i] = -h;
				}
				for (size_t i = 0; i < n; ++i) {
					for (auto face : faces[i]) {
						const Real slack = face.normal.dot(x[i + 1]) - face.offset;
						gradient[i] -= (mu / slack) * face.normal;
						diagonal[i] += (mu / (slack * slack)) * face.normal * face.normal.transpose();
					}
					diagonal[i] += 1e-14L * M::Identity();
				}
				std::vector<M> inverse(n), factor(n, M::Zero());
				std::vector<V> rhs(n);
				for (size_t i = 0; i < n; ++i) {
					rhs[i] = -gradient[i];
					if (i) {
						factor[i] = off[i] * inverse[i - 1];
						diagonal[i] -= factor[i] * off[i].transpose();
						rhs[i] -= factor[i] * rhs[i - 1];
					}
					inverse[i] = diagonal[i].ldlt().solve(M::Identity());
				}
				for (size_t i = n; i-- > 0;) step[i] = inverse[i] * (rhs[i] - (i + 1 < n ? V(off[i + 1].transpose() * step[i + 1]) : V::Zero()));
				Real slope = 0;
				for (size_t i = 0; i < n; ++i) slope += gradient[i].dot(step[i]);
				if (!isfinite(slope) || slope >= 0 || -slope < mu * (std::numeric_limits<Real>::digits > 64 ? 1e-22L : 1e-7L)) break;
				const Real before = objective(x, mu);
				Real alpha = 1;
				std::vector<V> trial = x;
				for (; alpha > 1e-18L; alpha *= .5L) {
					for (size_t i = 0; i < n; ++i) trial[i + 1] = x[i + 1] + alpha * step[i];
					if (objective(trial, mu) <= before + .01L * alpha * slope) break;
				}
				if (alpha <= 1e-18L) break;
				x = std::move(trial);
			}
			q.clear();
			for (auto v : x) q.emplace_back(start.x + scale * double(v.x()), start.y + scale * double(v.y()));
			q.front() = start;
			q.back() = target;
			const double value = tpp::certified_detail::path_length(q);
			if (value < result.upper_bound) { result.path = q; result.upper_bound = value; }
			std::vector<V> directions;
			for (size_t i = 1; i < x.size(); ++i) directions.push_back((x[i] - x[i - 1]) / sqrt((x[i] - x[i - 1]).squaredNorm() + mu * mu));
			Real bound = x.back().dot(directions.back());
			for (size_t i = 0; i < n; ++i) {
				Real support = std::numeric_limits<Real>::infinity();
				for (auto v : polygons[i]) {
					const V p{(Real(v.x) - Real(start.x)) / Real(scale), (Real(v.y) - Real(start.y)) / Real(scale)};
					support = std::min(support, p.dot(directions[i] - directions[i + 1]));
				}
				bound += support;
			}
			result.lower_bound = std::max(result.lower_bound, double(bound) * scale - safety);
			if (result.upper_bound - result.lower_bound <= tolerance || result.lower_bound >= cutoff) return result;
		}
		return result;
	}
}

namespace tpp::certified_detail {
	CertifiedConvexTppResult refine_long_double(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &polygons,
		double tolerance, double scale, double safety, double cutoff, CertifiedConvexTppResult result
	) {
		return refine<long double>(start, target, polygons, tolerance, scale, safety, cutoff, std::move(result));
	}

	CertifiedConvexTppResult refine_extended_precision(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &polygons,
		double tolerance, double scale, double safety, double cutoff, CertifiedConvexTppResult result
	) {
		return refine<boost::multiprecision::cpp_bin_float_quad>(start, target, polygons, tolerance, scale, safety, cutoff, std::move(result));
	}
}
