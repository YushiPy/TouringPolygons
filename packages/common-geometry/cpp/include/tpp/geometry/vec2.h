#pragma once

#include <cmath>
#include <format>
#include <limits>
#include <ostream>
#include <stdexcept>

template <class T>
class Vec2 {
	public:
	T x;
	T y;

	static const Vec2 ZERO;
	static const Vec2 ONE;
	static const Vec2 INF;
	static const Vec2 NaN;
	static const Vec2 LEFT;
	static const Vec2 RIGHT;
	static const Vec2 UP;
	static const Vec2 DOWN;

	constexpr Vec2() noexcept : x(0), y(0) {}
	constexpr Vec2(T x, T y) noexcept : x(x), y(y) {}

	template <class U>
	constexpr explicit Vec2(const Vec2<U> &from) noexcept : x(static_cast<T>(from.x)), y(static_cast<T>(from.y)) {}

	static Vec2 from_angle(T angle) noexcept {
		return {std::cos(angle), std::sin(angle)};
	}

	T angle() const noexcept {
		return std::atan2(y, x);
	}

	T angle_to_point(const Vec2 &point) const noexcept {
		return (point - *this).angle();
	}

	Vec2 rotated(T angle) const noexcept {
		const T sine = std::sin(angle);
		const T cosine = std::cos(angle);
		return {x * cosine - y * sine, x * sine + y * cosine};
	}

	constexpr T cross(const Vec2 &other) const noexcept {
		return x * other.y - y * other.x;
	}

	constexpr T dot(const Vec2 &other) const noexcept {
		return x * other.x + y * other.y;
	}

	constexpr T length_squared() const noexcept {
		return x * x + y * y;
	}

	T length() const noexcept {
		return std::sqrt(length_squared());
	}

	constexpr T distance_squared_to(const Vec2 &other) const noexcept {
		return (*this - other).length_squared();
	}

	T distance_to(const Vec2 &other) const noexcept {
		return (*this - other).length();
	}

	constexpr Vec2 lerp(const Vec2 &to, T weight) const noexcept {
		return *this + (to - *this) * weight;
	}

	void normalize() noexcept {
		const T len_sq = length_squared();

		if (len_sq == T(0)) {
			x = 0;
			y = 0;
			return;
		}

		const T inv_len = T(1) / std::sqrt(len_sq);
		x *= inv_len;
		y *= inv_len;
	}

	Vec2 normalized() const noexcept {
		const T len_sq = length_squared();

		if (len_sq == T(0)) {
			return ZERO;
		}

		return *this * (T(1) / std::sqrt(len_sq));
	}

	constexpr Vec2 reflect(const Vec2 &line) const noexcept {
		return T(2) * line * dot(line) / line.length_squared() - *this;
	}

	constexpr Vec2 reflect_line(const Vec2 &point1, const Vec2 &point2) const noexcept {
		return point1 + (*this - point1).reflect(point2 - point1);
	}

	bool is_equal_approx(const Vec2 &other) const noexcept {
		return is_equal_approx(other, T(1e-8));
	}

	bool is_equal_approx(const Vec2 &other, T epsilon) const noexcept {
		return std::fabs(x - other.x) < epsilon && std::fabs(y - other.y) < epsilon;
	}

	bool is_same_direction(const Vec2 &other) const noexcept {
		return std::fabs(cross(other)) < T(1e-8) && dot(other) > T(0);
	}

	bool is_finite() const noexcept {
		return std::isfinite(x) && std::isfinite(y);
	}

	bool is_nan() const noexcept {
		return std::isnan(x) || std::isnan(y);
	}

	constexpr Vec2 operator+(const Vec2 &right) const noexcept {
		return {x + right.x, y + right.y};
	}

	constexpr Vec2 operator-(const Vec2 &right) const noexcept {
		return {x - right.x, y - right.y};
	}

	constexpr Vec2 operator*(const Vec2 &right) const noexcept {
		return {x * right.x, y * right.y};
	}

	constexpr Vec2 operator*(T scalar) const noexcept {
		return {x * scalar, y * scalar};
	}

	constexpr Vec2 operator/(const Vec2 &right) const noexcept {
		return {x / right.x, y / right.y};
	}

	constexpr Vec2 operator/(T scalar) const noexcept {
		return {x / scalar, y / scalar};
	}

	constexpr Vec2 operator+() const noexcept {
		return *this;
	}

	constexpr Vec2 operator-() const noexcept {
		return {-x, -y};
	}

	constexpr void operator+=(const Vec2 &right) noexcept {
		x += right.x;
		y += right.y;
	}

	constexpr void operator-=(const Vec2 &right) noexcept {
		x -= right.x;
		y -= right.y;
	}

	constexpr void operator*=(const Vec2 &right) noexcept {
		x *= right.x;
		y *= right.y;
	}

	constexpr void operator*=(T scalar) noexcept {
		x *= scalar;
		y *= scalar;
	}

	constexpr void operator/=(const Vec2 &right) noexcept {
		x /= right.x;
		y /= right.y;
	}

	constexpr void operator/=(T scalar) noexcept {
		x /= scalar;
		y /= scalar;
	}

	constexpr bool operator==(const Vec2 &right) const noexcept {
		return x == right.x && y == right.y;
	}

	constexpr bool operator!=(const Vec2 &right) const noexcept {
		return !(*this == right);
	}

	constexpr bool operator<(const Vec2 &right) const noexcept {
		return x == right.x ? y < right.y : x < right.x;
	}

	constexpr bool operator<=(const Vec2 &right) const noexcept {
		return x == right.x ? y <= right.y : x < right.x;
	}

	constexpr bool operator>(const Vec2 &right) const noexcept {
		return x == right.x ? y > right.y : x > right.x;
	}

	constexpr bool operator>=(const Vec2 &right) const noexcept {
		return x == right.x ? y >= right.y : x > right.x;
	}

	T operator[](int index) const {
		if (index == 0) {
			return x;
		}

		if (index == 1 || index == -1) {
			return y;
		}

		throw std::out_of_range(std::format("Index {} out of range for Vec2", index));
	}
};

template <class T>
inline const Vec2<T> Vec2<T>::ZERO = {T(0), T(0)};

template <class T>
inline const Vec2<T> Vec2<T>::ONE = {T(1), T(1)};

template <class T>
inline const Vec2<T> Vec2<T>::INF = {
	std::numeric_limits<T>::infinity(),
	std::numeric_limits<T>::infinity(),
};

template <class T>
inline const Vec2<T> Vec2<T>::NaN = {
	std::numeric_limits<T>::quiet_NaN(),
	std::numeric_limits<T>::quiet_NaN(),
};

template <class T>
inline const Vec2<T> Vec2<T>::LEFT = {T(-1), T(0)};

template <class T>
inline const Vec2<T> Vec2<T>::RIGHT = {T(1), T(0)};

template <class T>
inline const Vec2<T> Vec2<T>::UP = {T(0), T(-1)};

template <class T>
inline const Vec2<T> Vec2<T>::DOWN = {T(0), T(1)};

template <class T>
constexpr Vec2<T> operator*(T scalar, const Vec2<T> &vector) noexcept {
	return vector * scalar;
}

template <class T>
constexpr Vec2<T> operator/(T scalar, const Vec2<T> &vector) noexcept {
	return {scalar / vector.x, scalar / vector.y};
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Vec2<T>& v) {
	os << "Vec2(" << v.x << ", " << v.y << ")";
	return os;
}

using Vector2 = Vec2<double>;
using Vector2f = Vec2<float>;
using Vector2d = Vec2<double>;

template<>
struct std::formatter<Vector2> : std::formatter<double> {
	auto format(const Vector2& v, std::format_context& ctx) const {
		auto out = ctx.out();
		out = std::format_to(out, "{{");
		out = std::formatter<double>::format(v.x, ctx);
		out = std::format_to(out, ", ");
		out = std::formatter<double>::format(v.y, ctx);
		return std::format_to(out, "}}");
	}
};

