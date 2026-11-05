#pragma once

// Exact arithmetic without external dependencies: a small signed big integer,
// filtered predicates on doubles, and homogeneous points with big integer
// coordinates for the few points that are not doubles.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace steiner_convex_partition::exact {

	class BigInt {
	public:
		BigInt() = default;

		BigInt(std::int64_t value) {
			if (value < 0) {
				negative_ = true;
				std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1;
				set_magnitude(magnitude);
			} else {
				set_magnitude(static_cast<std::uint64_t>(value));
			}
		}

		int sign() const {
			if (limbs_.empty()) return 0;
			return negative_ ? -1 : 1;
		}

		bool is_zero() const {
			return limbs_.empty();
		}

		BigInt operator-() const {
			BigInt result = *this;
			if (!result.limbs_.empty()) result.negative_ = !result.negative_;
			return result;
		}

		friend BigInt operator+(const BigInt &a, const BigInt &b) {
			if (a.negative_ == b.negative_) {
				BigInt result;
				result.limbs_ = add_magnitudes(a.limbs_, b.limbs_);
				result.negative_ = a.negative_;
				result.trim();
				return result;
			}
			const int order = compare_magnitudes(a.limbs_, b.limbs_);
			if (order == 0) return BigInt();
			BigInt result;
			if (order > 0) {
				result.limbs_ = subtract_magnitudes(a.limbs_, b.limbs_);
				result.negative_ = a.negative_;
			} else {
				result.limbs_ = subtract_magnitudes(b.limbs_, a.limbs_);
				result.negative_ = b.negative_;
			}
			result.trim();
			return result;
		}

		friend BigInt operator-(const BigInt &a, const BigInt &b) {
			return a + (-b);
		}

		friend BigInt operator*(const BigInt &a, const BigInt &b) {
			BigInt result;
			if (a.limbs_.empty() || b.limbs_.empty()) return result;
			result.limbs_.assign(a.limbs_.size() + b.limbs_.size(), 0);
			for (std::size_t i = 0; i < a.limbs_.size(); ++i) {
				std::uint64_t carry = 0;
				const std::uint64_t ai = a.limbs_[i];
				for (std::size_t j = 0; j < b.limbs_.size(); ++j) {
					const std::uint64_t current = result.limbs_[i + j] + ai * b.limbs_[j] + carry;
					result.limbs_[i + j] = static_cast<std::uint32_t>(current);
					carry = current >> 32;
				}
				std::size_t k = i + b.limbs_.size();
				while (carry != 0) {
					const std::uint64_t current = result.limbs_[k] + carry;
					result.limbs_[k] = static_cast<std::uint32_t>(current);
					carry = current >> 32;
					++k;
				}
			}
			result.negative_ = a.negative_ != b.negative_;
			result.trim();
			return result;
		}

		BigInt shifted_left(unsigned bits) const {
			if (limbs_.empty()) return *this;
			BigInt result;
			const unsigned limb_shift = bits / 32;
			const unsigned bit_shift = bits % 32;
			result.limbs_.assign(limbs_.size() + limb_shift + 1, 0);
			for (std::size_t i = 0; i < limbs_.size(); ++i) {
				const std::uint64_t value = static_cast<std::uint64_t>(limbs_[i]) << bit_shift;
				result.limbs_[i + limb_shift] |= static_cast<std::uint32_t>(value);
				result.limbs_[i + limb_shift + 1] |= static_cast<std::uint32_t>(value >> 32);
			}
			result.negative_ = negative_;
			result.trim();
			return result;
		}

		friend int compare(const BigInt &a, const BigInt &b) {
			if (a.sign() != b.sign()) return a.sign() < b.sign() ? -1 : 1;
			const int order = compare_magnitudes(a.limbs_, b.limbs_);
			return a.negative_ ? -order : order;
		}

		// Approximation as mantissa * 2^exponent, with the mantissa in [0.5, 1).
		std::pair<long double, long> approximate() const {
			if (limbs_.empty()) return {0.0L, 0};
			long double mantissa = 0.0L;
			const std::size_t top = limbs_.size();
			const std::size_t used = std::min<std::size_t>(top, 3);
			for (std::size_t k = 0; k < used; ++k) {
				mantissa = mantissa * 4294967296.0L + static_cast<long double>(limbs_[top - 1 - k]);
			}
			int exponent = 0;
			mantissa = std::frexp(mantissa, &exponent);
			const long total = static_cast<long>(exponent) + 32L * static_cast<long>(top - used);
			return {negative_ ? -mantissa : mantissa, total};
		}

		std::size_t bit_length() const {
			if (limbs_.empty()) return 0;
			std::size_t bits = 32 * (limbs_.size() - 1);
			std::uint32_t top = limbs_.back();
			while (top != 0) {
				++bits;
				top >>= 1;
			}
			return bits;
		}

		std::string to_string() const {
			if (limbs_.empty()) return "0";
			std::vector<std::uint32_t> digits = limbs_;
			std::string out;
			while (!digits.empty()) {
				std::uint64_t remainder = 0;
				for (std::size_t i = digits.size(); i-- > 0;) {
					const std::uint64_t current = (remainder << 32) | digits[i];
					digits[i] = static_cast<std::uint32_t>(current / 1000000000ULL);
					remainder = current % 1000000000ULL;
				}
				while (!digits.empty() && digits.back() == 0) digits.pop_back();
				std::string chunk = std::to_string(remainder);
				if (!digits.empty()) chunk.insert(0, 9 - chunk.size(), '0');
				out.insert(0, chunk);
			}
			if (negative_) out.insert(0, "-");
			return out;
		}

	private:
		bool negative_ = false;
		std::vector<std::uint32_t> limbs_;

		void set_magnitude(std::uint64_t magnitude) {
			limbs_.clear();
			while (magnitude != 0) {
				limbs_.push_back(static_cast<std::uint32_t>(magnitude));
				magnitude >>= 32;
			}
			if (limbs_.empty()) negative_ = false;
		}

		void trim() {
			while (!limbs_.empty() && limbs_.back() == 0) limbs_.pop_back();
			if (limbs_.empty()) negative_ = false;
		}

		static int compare_magnitudes(const std::vector<std::uint32_t> &a, const std::vector<std::uint32_t> &b) {
			if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
			for (std::size_t i = a.size(); i-- > 0;) {
				if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
			}
			return 0;
		}

		static std::vector<std::uint32_t> add_magnitudes(const std::vector<std::uint32_t> &a, const std::vector<std::uint32_t> &b) {
			const std::vector<std::uint32_t> &longer = a.size() >= b.size() ? a : b;
			const std::vector<std::uint32_t> &shorter = a.size() >= b.size() ? b : a;
			std::vector<std::uint32_t> result(longer.size() + 1, 0);
			std::uint64_t carry = 0;
			for (std::size_t i = 0; i < longer.size(); ++i) {
				const std::uint64_t current = static_cast<std::uint64_t>(longer[i])
					+ (i < shorter.size() ? shorter[i] : 0) + carry;
				result[i] = static_cast<std::uint32_t>(current);
				carry = current >> 32;
			}
			result[longer.size()] = static_cast<std::uint32_t>(carry);
			return result;
		}

		// Requires |a| >= |b|.
		static std::vector<std::uint32_t> subtract_magnitudes(const std::vector<std::uint32_t> &a, const std::vector<std::uint32_t> &b) {
			std::vector<std::uint32_t> result(a.size(), 0);
			std::int64_t borrow = 0;
			for (std::size_t i = 0; i < a.size(); ++i) {
				std::int64_t current = static_cast<std::int64_t>(a[i]) - borrow - (i < b.size() ? static_cast<std::int64_t>(b[i]) : 0);
				borrow = 0;
				if (current < 0) {
					current += 4294967296LL;
					borrow = 1;
				}
				result[i] = static_cast<std::uint32_t>(current);
			}
			return result;
		}
	};

	// A finite double as an integer mantissa times a power of two.
	struct Dyadic {
		std::int64_t mantissa = 0;
		int exponent = 0;
	};

	inline Dyadic decompose(double value) {
		if (value == 0.0) return {};
		int exponent = 0;
		const double fraction = std::frexp(value, &exponent);
		Dyadic result;
		result.mantissa = static_cast<std::int64_t>(std::ldexp(fraction, 53));
		result.exponent = exponent - 53;
		while (result.mantissa != 0 && (result.mantissa & 1) == 0) {
			result.mantissa /= 2;
			++result.exponent;
		}
		return result;
	}

	// Exact integers for doubles scaled by a common power of two.
	template <std::size_t N>
	std::vector<BigInt> scaled_integers(const double (&values)[N]) {
		Dyadic parts[N];
		int lowest = 0;
		bool any = false;
		for (std::size_t i = 0; i < N; ++i) {
			parts[i] = decompose(values[i]);
			if (parts[i].mantissa == 0) continue;
			if (!any || parts[i].exponent < lowest) lowest = parts[i].exponent;
			any = true;
		}
		std::vector<BigInt> result;
		result.reserve(N);
		for (std::size_t i = 0; i < N; ++i) {
			if (parts[i].mantissa == 0) {
				result.emplace_back();
			} else {
				result.push_back(BigInt(parts[i].mantissa).shifted_left(static_cast<unsigned>(parts[i].exponent - lowest)));
			}
		}
		return result;
	}

	struct Vec2 {
		double x = 0.0;
		double y = 0.0;

		friend bool operator==(const Vec2 &a, const Vec2 &b) {
			return a.x == b.x && a.y == b.y;
		}
	};

	inline int sign_of(double value) {
		return value > 0 ? 1 : (value < 0 ? -1 : 0);
	}

	namespace expansion {

		inline void two_sum(double a, double b, double &sum, double &error) {
			sum = a + b;
			const double bv = sum - a;
			const double av = sum - bv;
			error = (a - av) + (b - bv);
		}

		inline void two_product(double a, double b, double &product, double &error) {
			product = a * b;
			error = std::fma(a, b, -product);
		}

		// Sign of (a1 - a0)(b1 - b0) + s (c1 - c0)(d1 - d0) with s = +-1, exactly,
		// through floating-point expansions (Shewchuk). Returns 2 when an
		// intermediate could underflow and the result might not be exact.
		inline int sign_of_two_products(double a0, double a1, double b0, double b1, double c0, double c1, double d0, double d1, int s) {
			double hi[4];
			double lo[4];
			two_sum(a1, -a0, hi[0], lo[0]);
			two_sum(b1, -b0, hi[1], lo[1]);
			two_sum(c1, -c0, hi[2], lo[2]);
			two_sum(d1, -d0, hi[3], lo[3]);
			for (int k = 0; k < 4; ++k) {
				const double l = std::fabs(lo[k]);
				const double h = std::fabs(hi[k]);
				if ((l != 0.0 && l < 1.0e-140) || (h != 0.0 && h < 1.0e-140) || h > 1.0e140) return 2;
			}
			double terms[16];
			int count = 0;
			auto product = [&](double x, double y, double sign) {
				double p;
				double e;
				two_product(x, y, p, e);
				terms[count++] = sign * p;
				terms[count++] = sign * e;
			};
			const double second = s > 0 ? 1.0 : -1.0;
			product(hi[0], hi[1], 1.0);
			product(hi[0], lo[1], 1.0);
			product(lo[0], hi[1], 1.0);
			product(lo[0], lo[1], 1.0);
			product(hi[2], hi[3], second);
			product(hi[2], lo[3], second);
			product(lo[2], hi[3], second);
			product(lo[2], lo[3], second);
			// Grow a nonoverlapping expansion term by term; its largest nonzero
			// component carries the sign of the sum.
			double expansion[17];
			int size = 0;
			for (int t = 0; t < count; ++t) {
				double q = terms[t];
				int out = 0;
				for (int k = 0; k < size; ++k) {
					double sum;
					double error;
					two_sum(q, expansion[k], sum, error);
					if (error != 0.0) expansion[out++] = error;
					q = sum;
				}
				if (q != 0.0) expansion[out++] = q;
				size = out;
			}
			for (int k = size; k-- > 0;) {
				if (expansion[k] > 0.0) return 1;
				if (expansion[k] < 0.0) return -1;
			}
			return 0;
		}
	}

	// sign((b - a) x (d - c)), exactly.
	inline int cross_sign(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d) {
		const double t1 = (b.x - a.x) * (d.y - c.y);
		const double t2 = (b.y - a.y) * (d.x - c.x);
		const double det = t1 - t2;
		const double bound = (std::fabs(t1) + std::fabs(t2)) * 1.0e-15;
		if (det > bound) return 1;
		if (-det > bound) return -1;
		const int fast = expansion::sign_of_two_products(a.x, b.x, c.y, d.y, a.y, b.y, c.x, d.x, -1);
		if (fast != 2) return fast;
		const double values[8] = {a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y};
		const std::vector<BigInt> v = scaled_integers(values);
		const BigInt exact = (v[2] - v[0]) * (v[7] - v[5]) - (v[3] - v[1]) * (v[6] - v[4]);
		return exact.sign();
	}

	// sign((b - a) . (d - c)), exactly.
	inline int dot_sign(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d) {
		const double t1 = (b.x - a.x) * (d.x - c.x);
		const double t2 = (b.y - a.y) * (d.y - c.y);
		const double sum = t1 + t2;
		const double bound = (std::fabs(t1) + std::fabs(t2)) * 1.0e-15;
		if (sum > bound) return 1;
		if (-sum > bound) return -1;
		const int fast = expansion::sign_of_two_products(a.x, b.x, c.x, d.x, a.y, b.y, c.y, d.y, 1);
		if (fast != 2) return fast;
		const double values[8] = {a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y};
		const std::vector<BigInt> v = scaled_integers(values);
		const BigInt exact = (v[2] - v[0]) * (v[6] - v[4]) + (v[3] - v[1]) * (v[7] - v[5]);
		return exact.sign();
	}

	// Positive when a, b, c turn counterclockwise.
	inline int orient(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
		return cross_sign(a, b, a, c);
	}

	// For collinear a, o, b: is o strictly between a and b?
	inline bool strictly_between_collinear(const Vec2 &a, const Vec2 &o, const Vec2 &b) {
		if (a.x != b.x) {
			return (a.x < o.x && o.x < b.x) || (b.x < o.x && o.x < a.x);
		}
		return (a.y < o.y && o.y < b.y) || (b.y < o.y && o.y < a.y);
	}

	// For collinear a, o, b: is o on the closed segment [a, b]?
	inline bool on_closed_collinear(const Vec2 &a, const Vec2 &o, const Vec2 &b) {
		return std::min(a.x, b.x) <= o.x && o.x <= std::max(a.x, b.x)
			&& std::min(a.y, b.y) <= o.y && o.y <= std::max(a.y, b.y);
	}

	// Does the open segment (p, q) meet the closed segment [u, w]?
	inline bool open_segment_meets_closed(const Vec2 &p, const Vec2 &q, const Vec2 &u, const Vec2 &w) {
		const int o1 = orient(p, q, u);
		const int o2 = orient(p, q, w);
		if (o1 * o2 > 0) return false;
		const int o3 = orient(u, w, p);
		const int o4 = orient(u, w, q);
		if (o1 == 0 && o2 == 0) {
			// Collinear: compare the projections on one axis, which keep the order.
			const bool use_x = p.x != q.x;
			const double sp = use_x ? p.x : p.y;
			const double sq = use_x ? q.x : q.y;
			const double su = use_x ? u.x : u.y;
			const double sw = use_x ? w.x : w.y;
			const double a = std::min(sp, sq);
			const double b = std::max(sp, sq);
			const double c = std::min(su, sw);
			const double d = std::max(su, sw);
			return c < b && d > a;
		}
		if (o3 * o4 > 0) return false;
		if (o1 * o2 < 0 && o3 * o4 < 0) return true;
		if (o1 == 0 && strictly_between_collinear(p, u, q)) return true;
		if (o2 == 0 && strictly_between_collinear(p, w, q)) return true;
		// An end of (p, q) inside the edge: the segment touches the edge there.
		if (o3 == 0 && strictly_between_collinear(u, p, w)) return true;
		if (o4 == 0 && strictly_between_collinear(u, q, w)) return true;
		return false;
	}

	// Do the closed segments [p, q] and [u, w] meet?
	inline bool closed_segments_meet(const Vec2 &p, const Vec2 &q, const Vec2 &u, const Vec2 &w) {
		const int o1 = orient(p, q, u);
		const int o2 = orient(p, q, w);
		const int o3 = orient(u, w, p);
		const int o4 = orient(u, w, q);
		if (o1 * o2 < 0 && o3 * o4 < 0) return true;
		if (o1 == 0 && on_closed_collinear(p, u, q)) return true;
		if (o2 == 0 && on_closed_collinear(p, w, q)) return true;
		if (o3 == 0 && on_closed_collinear(u, p, w)) return true;
		if (o4 == 0 && on_closed_collinear(u, q, w)) return true;
		return false;
	}

	// Counterclockwise angle from u to v is in (0, 180] for directions u = a - o, v = b - o.
	inline bool gap_at_most_half_turn(const Vec2 &o, const Vec2 &a, const Vec2 &b) {
		const int s = orient(o, a, b);
		if (s > 0) return true;
		if (s < 0) return false;
		return dot_sign(o, a, o, b) < 0;
	}

	// Homogeneous point with big integer coordinates and w > 0, in a common
	// integer frame (see Frame).
	struct HPoint {
		BigInt x;
		BigInt y;
		BigInt w = BigInt(1);
	};

	inline int orient(const HPoint &a, const HPoint &b, const HPoint &c) {
		const BigInt det =
			a.x * (b.y * c.w - c.y * b.w)
			- a.y * (b.x * c.w - c.x * b.w)
			+ a.w * (b.x * c.y - c.x * b.y);
		return det.sign();
	}

	// sign((b - a) . (d - c)) for homogeneous points.
	inline int dot_sign(const HPoint &a, const HPoint &b, const HPoint &c, const HPoint &d) {
		const BigInt bax = b.x * a.w - a.x * b.w;
		const BigInt bay = b.y * a.w - a.y * b.w;
		const BigInt dcx = d.x * c.w - c.x * d.w;
		const BigInt dcy = d.y * c.w - c.y * d.w;
		return (bax * dcx + bay * dcy).sign();
	}

	inline bool same_point(const HPoint &a, const HPoint &b) {
		return compare(a.x * b.w, b.x * a.w) == 0 && compare(a.y * b.w, b.y * a.w) == 0;
	}

	// Intersection of the lines (p1, p2) and (q1, q2); w == 0 when parallel.
	inline HPoint line_intersection(const HPoint &p1, const HPoint &p2, const HPoint &q1, const HPoint &q2) {
		const BigInt la = p1.y * p2.w - p2.y * p1.w;
		const BigInt lb = p2.x * p1.w - p1.x * p2.w;
		const BigInt lc = p1.x * p2.y - p2.x * p1.y;
		const BigInt ma = q1.y * q2.w - q2.y * q1.w;
		const BigInt mb = q2.x * q1.w - q1.x * q2.w;
		const BigInt mc = q1.x * q2.y - q2.x * q1.y;
		HPoint result;
		result.x = lb * mc - lc * mb;
		result.y = lc * ma - la * mc;
		result.w = la * mb - lb * ma;
		if (result.w.sign() < 0) {
			result.x = -result.x;
			result.y = -result.y;
			result.w = -result.w;
		}
		return result;
	}

	// Common integer frame: every double coordinate of interest becomes an
	// integer after multiplying by 2^-lowest_exponent.
	class Frame {
	public:
		void include(double value) {
			const Dyadic part = decompose(value);
			if (part.mantissa == 0) return;
			if (!any_ || part.exponent < lowest_) lowest_ = part.exponent;
			any_ = true;
		}

		void include(const Vec2 &p) {
			include(p.x);
			include(p.y);
		}

		// Exact frame coordinates of a double point; a point finer than the frame
		// gets a power-of-two denominator.
		HPoint point(const Vec2 &p) const {
			const Dyadic dx = decompose(p.x);
			const Dyadic dy = decompose(p.y);
			const int sx = dx.exponent - lowest_;
			const int sy = dy.exponent - lowest_;
			int s = 0;
			if (dx.mantissa != 0) s = std::min(s, sx);
			if (dy.mantissa != 0) s = std::min(s, sy);
			HPoint result;
			if (dx.mantissa != 0) result.x = BigInt(dx.mantissa).shifted_left(static_cast<unsigned>(sx - s));
			if (dy.mantissa != 0) result.y = BigInt(dy.mantissa).shifted_left(static_cast<unsigned>(sy - s));
			result.w = BigInt(1).shifted_left(static_cast<unsigned>(-s));
			return result;
		}

		long double to_long_double(const BigInt &numerator, const BigInt &denominator) const {
			const auto [nm, ne] = numerator.approximate();
			const auto [dm, de] = denominator.approximate();
			if (nm == 0.0L) return 0.0L;
			return std::ldexp(nm / dm, static_cast<int>(ne - de + lowest_));
		}

		Vec2 approximate(const HPoint &p) const {
			return {
				static_cast<double>(to_long_double(p.x, p.w)),
				static_cast<double>(to_long_double(p.y, p.w)),
			};
		}

		// Is p exactly the double point q?
		bool equals(const HPoint &p, const Vec2 &q) const {
			const HPoint h = point(q);
			return same_point(p, h);
		}

		int lowest_exponent() const {
			return lowest_;
		}

	private:
		bool any_ = false;
		int lowest_ = 0;
	};
}
