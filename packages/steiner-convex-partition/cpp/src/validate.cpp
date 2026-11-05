#include "exact_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace steiner_convex_partition::exact {

	namespace {

		struct Box {
			double min_x = std::numeric_limits<double>::infinity();
			double min_y = std::numeric_limits<double>::infinity();
			double max_x = -std::numeric_limits<double>::infinity();
			double max_y = -std::numeric_limits<double>::infinity();

			void add(const Vec2 &p) {
				min_x = std::min(min_x, p.x);
				min_y = std::min(min_y, p.y);
				max_x = std::max(max_x, p.x);
				max_y = std::max(max_y, p.y);
			}

			void inflate() {
				const double scale = std::max({std::fabs(min_x), std::fabs(min_y), std::fabs(max_x), std::fabs(max_y), 1.0});
				const double margin = scale * 1.0e-9;
				min_x -= margin;
				min_y -= margin;
				max_x += margin;
				max_y += margin;
			}

			bool overlaps(const Box &other) const {
				return min_x <= other.max_x && other.min_x <= max_x && min_y <= other.max_y && other.min_y <= max_y;
			}
		};

		BigInt determinant(const HPoint &a, const HPoint &b, const HPoint &c) {
			return a.x * (b.y * c.w - c.y * b.w)
				- a.y * (b.x * c.w - c.x * b.w)
				+ a.w * (b.x * c.y - c.x * b.y);
		}

		// Does the closed segment [u, w] meet the open interior of the convex
		// counterclockwise polygon q?
		bool segment_meets_open_convex(const HPoint &u, const HPoint &w, const std::vector<HPoint> &q) {
			BigInt low_num(0);
			BigInt low_den(1);
			BigInt high_num(1);
			BigInt high_den(1);
			const std::size_t m = q.size();
			for (std::size_t k = 0; k < m; ++k) {
				const HPoint &a = q[k];
				const HPoint &b = q[(k + 1) % m];
				const BigInt alpha = determinant(a, b, u) * w.w;
				const BigInt beta = determinant(a, b, w) * u.w;
				const int sa = alpha.sign();
				const int sb = beta.sign();
				if (sa > 0 && sb > 0) continue;
				if (sa <= 0 && sb <= 0) return false;
				if (sa > 0) {
					// t < alpha / (alpha - beta)
					const BigInt num = alpha;
					const BigInt den = alpha - beta;
					if (compare(num * high_den, high_num * den) < 0) {
						high_num = num;
						high_den = den;
					}
				} else {
					// t > -alpha / (beta - alpha)
					const BigInt num = -alpha;
					const BigInt den = beta - alpha;
					if (compare(num * low_den, low_num * den) > 0) {
						low_num = num;
						low_den = den;
					}
				}
			}
			return compare(low_num * high_den, high_num * low_den) < 0;
		}

		bool strictly_inside_polygon(const HPoint &p, const std::vector<HPoint> &boundary) {
			bool inside = false;
			const std::size_t n = boundary.size();
			for (std::size_t e = 0; e < n; ++e) {
				const HPoint &u = boundary[e];
				const HPoint &w = boundary[(e + 1) % n];
				const bool u_above = compare(u.y * p.w, p.y * u.w) > 0;
				const bool w_above = compare(w.y * p.w, p.y * w.w) > 0;
				if (u_above == w_above) continue;
				const int o = orient(u, w, p);
				// Upward edge: the crossing is to the right of p when p is left of it.
				if (w_above ? o > 0 : o < 0) inside = !inside;
			}
			return inside;
		}

		bool separated(const std::vector<HPoint> &a, const std::vector<HPoint> &b) {
			for (int pass = 0; pass < 2; ++pass) {
				const std::vector<HPoint> &p = pass == 0 ? a : b;
				const std::vector<HPoint> &q = pass == 0 ? b : a;
				for (std::size_t k = 0; k < p.size(); ++k) {
					const HPoint &u = p[k];
					const HPoint &w = p[(k + 1) % p.size()];
					bool all_right = true;
					for (const HPoint &v : q) {
						if (orient(u, w, v) > 0) {
							all_right = false;
							break;
						}
					}
					if (all_right) return true;
				}
			}
			return false;
		}

		// Sum of fractions grouped by equal denominators.
		class FractionSum {
		public:
			bool add(const BigInt &num, const BigInt &den) {
				for (auto &[d, n] : groups_) {
					if (compare(d, den) == 0) {
						n = n + num;
						return true;
					}
				}
				if (groups_.size() >= 48) return false;
				groups_.emplace_back(den, num);
				return true;
			}

			int sign() const {
				BigInt total_num(0);
				BigInt total_den(1);
				for (const auto &[d, n] : groups_) {
					total_num = total_num * d + n * total_den;
					total_den = total_den * d;
				}
				return total_num.sign() * total_den.sign();
			}

		private:
			std::vector<std::pair<BigInt, BigInt>> groups_;
		};

		void add_cross_terms(FractionSum &sum, bool &exact_ok, long double &approx, const std::vector<HPoint> &ring, int sign, const Frame &frame) {
			const std::size_t m = ring.size();
			for (std::size_t k = 0; k < m; ++k) {
				const HPoint &a = ring[k];
				const HPoint &b = ring[(k + 1) % m];
				BigInt num = a.x * b.y - b.x * a.y;
				if (sign < 0) num = -num;
				if (exact_ok && !sum.add(num, a.w * b.w)) exact_ok = false;
				const Vec2 pa = frame.approximate(a);
				const Vec2 pb = frame.approximate(b);
				approx += sign * (static_cast<long double>(pa.x) * pb.y - static_cast<long double>(pb.x) * pa.y);
			}
		}
	}

	ValidationReport validate_partition(const std::vector<HPoint> &boundary, const std::vector<std::vector<HPoint>> &pieces, const Frame &frame) {
		ValidationReport report;
		if (pieces.empty()) {
			report.reason = "no pieces";
			return report;
		}
		std::vector<Box> boxes(pieces.size());
		for (std::size_t p = 0; p < pieces.size(); ++p) {
			std::string why;
			if (!is_weakly_convex(pieces[p], &why)) {
				report.reason = "piece " + std::to_string(p) + ": " + why;
				return report;
			}
			for (const HPoint &v : pieces[p]) boxes[p].add(frame.approximate(v));
			boxes[p].inflate();
		}

		// Each piece's interior misses the boundary and one interior point is
		// inside the polygon, so the whole piece is.
		const std::size_t n = boundary.size();
		for (std::size_t p = 0; p < pieces.size(); ++p) {
			const std::vector<HPoint> &q = pieces[p];
			for (std::size_t e = 0; e < n; ++e) {
				const HPoint &u = boundary[e];
				const HPoint &w = boundary[(e + 1) % n];
				Box edge_box;
				edge_box.add(frame.approximate(u));
				edge_box.add(frame.approximate(w));
				edge_box.inflate();
				if (!edge_box.overlaps(boxes[p])) continue;
				if (segment_meets_open_convex(u, w, q)) {
					report.reason = "polygon edge " + std::to_string(e) + " crosses piece " + std::to_string(p);
					return report;
				}
			}
			std::size_t k = 1;
			while (k + 1 < q.size() && orient(q[0], q[k], q[k + 1]) <= 0) ++k;
			const HPoint &a = q[0];
			const HPoint &b = q[k];
			const HPoint &c = q[k + 1];
			HPoint centroid;
			centroid.x = a.x * b.w * c.w + b.x * a.w * c.w + c.x * a.w * b.w;
			centroid.y = a.y * b.w * c.w + b.y * a.w * c.w + c.y * a.w * b.w;
			centroid.w = BigInt(3) * a.w * b.w * c.w;
			if (!strictly_inside_polygon(centroid, boundary)) {
				report.reason = "piece " + std::to_string(p) + " lies outside the polygon";
				return report;
			}
		}

		for (std::size_t p = 0; p < pieces.size(); ++p) {
			for (std::size_t q = p + 1; q < pieces.size(); ++q) {
				if (!boxes[p].overlaps(boxes[q])) continue;
				if (!separated(pieces[p], pieces[q])) {
					report.reason = "pieces " + std::to_string(p) + " and " + std::to_string(q) + " overlap";
					return report;
				}
			}
		}

		FractionSum sum;
		bool exact_ok = true;
		long double approx = 0.0L;
		long double scale = 0.0L;
		add_cross_terms(sum, exact_ok, approx, boundary, -1, frame);
		scale = std::fabs(approx);
		for (const std::vector<HPoint> &q : pieces) add_cross_terms(sum, exact_ok, approx, q, 1, frame);
		if (exact_ok) {
			report.exact_area = true;
			if (sum.sign() != 0) {
				report.reason = "piece areas do not add up to the polygon's area";
				return report;
			}
		} else if (std::fabs(approx) > 1.0e-9L * scale) {
			report.reason = "piece areas do not add up to the polygon's area (approximate check)";
			return report;
		}
		report.ok = true;
		return report;
	}
}
