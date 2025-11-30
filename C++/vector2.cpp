/**************************************************************************/
/*  vector2.cpp                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "vector2.h"

#include <algorithm>
#include <cmath>
#include <format>


/*Auxiliary functions and macros*/
#define CMP_EPSILON 1e-8

double sign(double x) {
	return (x < 0) ? -1.0 : ((x > 0) ? 1.0 : 0.0);
}

double snapped(double p_value, double p_step) {
	return p_step ? floor(p_value / p_step + 0.5) * p_step : p_value;
}


double bezier_interpolate(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
	/* Formula from Wikipedia article on Bezier curves. */
	double omt = (1.0 - p_t);
	double omt2 = omt * omt;
	double omt3 = omt2 * omt;
	double t2 = p_t * p_t;
	double t3 = t2 * p_t;

	return p_start * omt3 + p_control_1 * omt2 * p_t * 3.0 + p_control_2 * omt * t2 * 3.0 + p_end * t3;
}

double bezier_derivative(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
	/* Formula from Wikipedia article on Bezier curves. */
	double omt = (1.0 - p_t);
	double omt2 = omt * omt;
	double t2 = p_t * p_t;
	
	return (p_control_1 - p_start) * 3.0 * omt2 + (p_control_2 - p_control_1) * 6.0 * omt * p_t + (p_end - p_control_2) * 3.0 * t2;
}

double cubic_interpolate(double p_from, double p_to, double p_pre, double p_post, double p_weight) {
	return 0.5 * (
		(p_from * 2.0) +
		(-p_pre + p_to) * p_weight +
		(2.0 * p_pre - 5.0 * p_from + 4.0 * p_to - p_post) * (p_weight * p_weight) +
		(-p_pre + 3.0 * p_from - 3.0 * p_to + p_post) * (p_weight * p_weight * p_weight)
	);
}

double cubic_interpolate_in_time(double p_from, double p_to, double p_pre, double p_post, double p_weight, double p_to_t, double p_pre_t, double p_post_t) {
	
	/* Barry-Goldman method */
	double t = std::lerp(0.0, p_to_t, p_weight);
	double a1 = std::lerp(p_pre, p_from, p_pre_t == 0 ? 0.0 : (t - p_pre_t) / -p_pre_t);
	double a2 = std::lerp(p_from, p_to, p_to_t == 0 ? 0.5 : t / p_to_t);
	double a3 = std::lerp(p_to, p_post, p_post_t - p_to_t == 0 ? 1.0 : (t - p_to_t) / (p_post_t - p_to_t));
	double b1 = std::lerp(a1, a2, p_to_t - p_pre_t == 0 ? 0.0 : (t - p_pre_t) / (p_to_t - p_pre_t));
	double b2 = std::lerp(a2, a3, p_post_t == 0 ? 1.0 : t / p_post_t);
	
	return std::lerp(b1, b2, p_to_t == 0 ? 0.5 : t / p_to_t);
}


bool is_equal_approx(double p_left, double p_right, double p_tolerance) {
	// Check for exact equality first, required to handle "infinity" values.
	if (p_left == p_right) {
		return true;
	}
	// Then check for approximate equality.
	return abs(p_left - p_right) < p_tolerance;
}

bool is_equal_approx(double p_left, double p_right) {

	// Check for exact equality first, required to handle "infinity" values.
	if (p_left == p_right) {
		return true;
	}
	// Then check for approximate equality.
	double tolerance = CMP_EPSILON * abs(p_left);

	if (tolerance < CMP_EPSILON) {
		tolerance = CMP_EPSILON;
	}
	return abs(p_left - p_right) < tolerance;
}

bool is_zero_approx(double p_value) {
	return abs(p_value) < CMP_EPSILON;
}

bool is_same(double p_left, double p_right) {
	return (p_left == p_right) || (std::isnan(p_left) && std::isnan(p_right));
}


double fposmod(double p_x, double p_y) {
	double value = fmod(p_x, p_y);
	if (((value < 0) && (p_y > 0)) || ((value > 0) && (p_y < 0))) {
		value += p_y;
	}
	value += 0.0;
	return value;
}


/*Definition of static constant vectors*/
const Vector2 Vector2::ZERO = Vector2(0.0, 0.0);
const Vector2 Vector2::ONE = Vector2(1.0, 1.0);
const Vector2 Vector2::INF = Vector2(INFINITY, INFINITY);
const Vector2 Vector2::LEFT = Vector2(-1.0, 0.0);
const Vector2 Vector2::RIGHT = Vector2(1.0, 0.0);
const Vector2 Vector2::UP = Vector2(0.0, -1.0);
const Vector2 Vector2::DOWN = Vector2(0.0, 1.0);


/*Constructors*/
Vector2::Vector2() : x(0), y(0) {}
Vector2::Vector2(const Vector2& from) : x(from.x), y(from.y) {}
Vector2::Vector2(double p_x, double p_y) : x(p_x), y(p_y) {}


/*Method definitions*/
Vector2 Vector2::abs() const {
	return Vector2(std::fabs(x), std::fabs(y));
}

double Vector2::angle() const {
	return std::atan2(y, x);
}

double Vector2::angle_to(const Vector2 &to) const {
	return std::atan2(cross(to), dot(to));
}

double Vector2::angle_to_point(const Vector2 &point) const {
	return (point - *this).angle();
}

double Vector2::aspect() const {
	return x / y;
}


Vector2 Vector2::bezier_derivative(const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end, double t) const {
	return Vector2(
		::bezier_derivative(x, control_1.x, control_2.x, end.x, t),
		::bezier_derivative(y, control_1.y, control_2.y, end.y, t)
	);
}

Vector2 Vector2::bezier_interpolate(const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end, double t) const {
	return Vector2(
		::bezier_interpolate(x, control_1.x, control_2.x, end.x, t),
		::bezier_interpolate(y, control_1.y, control_2.y, end.y, t)
	);
}

Vector2 Vector2::bounce(const Vector2 &normal) const {
	return - reflect(normal);
}

Vector2 Vector2::ceil() const {
	return Vector2(std::ceil(x), std::ceil(y));
}

Vector2 Vector2::clamp(const Vector2 &p_min, const Vector2 &p_max) const {
	return Vector2(
		std::clamp(x, p_min.x, p_max.x),
		std::clamp(y, p_min.y, p_max.y)
	);
}

Vector2 Vector2::clamp(double p_min, double p_max) const {
	return Vector2(
		std::clamp(x, p_min, p_max),
		std::clamp(y, p_min, p_max)
	);
}

Vector2 Vector2::clampf(double p_min, double p_max) const {
	return clamp(p_min, p_max);
}

double Vector2::cross(const Vector2 &other) const {
	return x * other.y - y * other.x;
}

Vector2 Vector2::cubic_interpolate(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight) const {
	return Vector2(
		::cubic_interpolate(x, b.x, pre_a.x, post_b.x, weight),
		::cubic_interpolate(y, b.y, pre_a.y, post_b.y, weight)
	);
}

Vector2 Vector2::cubic_interpolate_in_time(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight, double b_t, double pre_a_t, double post_b_t) const {
	return Vector2(
		::cubic_interpolate_in_time(x, b.x, pre_a.x, post_b.x, weight, b_t, pre_a_t, post_b_t),
		::cubic_interpolate_in_time(y, b.y, pre_a.y, post_b.y, weight, b_t, pre_a_t, post_b_t)
	);
}

Vector2 Vector2::direction_to(const Vector2 &to) const {
	return (to - *this).normalized();
}

double Vector2::distance_squared_to(const Vector2 &to) const {
	return (*this - to).length_squared();
}

double Vector2::distance_to(const Vector2 &to) const {
	return (*this - to).length();
}

double Vector2::dot(const Vector2 &with) const {
	return x * with.x + y * with.y;
}

Vector2 Vector2::floor() const {
	return Vector2(std::floor(x), std::floor(y));
}

Vector2 Vector2::from_angle(double angle) {
	return Vector2(std::cos(angle), std::sin(angle));
}

bool Vector2::is_equal_approx(const Vector2 &v) const {
	return ::is_equal_approx(x, v.x) && ::is_equal_approx(y, v.y);
}

bool Vector2::is_finite() const {
	return std::isfinite(x) && std::isfinite(y);
}

bool Vector2::is_normalized() const {
	// use length_squared() instead of length() to avoid sqrt(), makes it more stringent.
	return ::is_equal_approx(length_squared(), 1.0);
}

bool Vector2::is_zero_approx() const {
	return ::is_zero_approx(x) && ::is_zero_approx(y);
}

double Vector2::length() const {
	return std::sqrt(x * x + y * y);
}

double Vector2::length_squared() const {
	return x * x + y * y;
}

Vector2 Vector2::lerp(const Vector2 &to, double weight) const {
	return Vector2(
		std::lerp(x, to.x, weight),
		std::lerp(y, to.y, weight)
	);
}

Vector2 Vector2::limit_length(double length) const {
	
	const double l = this->length();
	Vector2 v = *this;
	
	if (l > 0 && length < l) {
		v /= l;
		v *= length;
	}

	return v;
}

Vector2 Vector2::max(const Vector2 &with) const {
	return Vector2(
		std::max(x, with.x),
		std::max(y, with.y)
	);
}

Vector2 Vector2::max(double with) const {
	return Vector2(
		std::max(x, with),
		std::max(y, with)
	);
}

Vector2::Axis Vector2::max_axis_index() const {
	return x >= y ? AXIS_X : AXIS_Y;
}

Vector2 Vector2::maxf(double with) const {
	return max(with);
}

Vector2 Vector2::min(const Vector2 &with) const {
	return Vector2(
		std::min(x, with.x),
		std::min(y, with.y)
	);
}

Vector2 Vector2::min(double with) const {
	return Vector2(
		std::min(x, with),
		std::min(y, with)
	);
}

Vector2::Axis Vector2::min_axis_index() const {
	return x <= y ? AXIS_X : AXIS_Y;
}

Vector2 Vector2::minf(double with) const {
	return min(with);
}

Vector2 Vector2::move_toward(const Vector2 &to, double delta) const {
	Vector2 v = *this;
	Vector2 diff = to - v;
	const double len = diff.length();
	return len <= delta || len < CMP_EPSILON ? to : v + diff / len * delta;
}

void Vector2::normalize() {
	double l = length_squared();
	if (l != 0) {
		l = std::sqrt(l);
		x /= l;
		y /= l;
	}
}

Vector2 Vector2::normalized() const {
	Vector2 v = *this;
	v.normalize();
	return v;
}

Vector2 Vector2::orthogonal() const {
	return Vector2(-y, x);
}

Vector2 Vector2::posmod(double mod) const {
	return Vector2(fposmod(x, mod), fposmod(y, mod));
}

Vector2 Vector2::posmod(const Vector2 &modv) const {
	return Vector2(fposmod(x, modv.x), fposmod(y, modv.y));
}

Vector2 Vector2::posmodv(const Vector2 &modv) const {
	return posmod(modv);
}

Vector2 Vector2::project(const Vector2 &b) const {
	return b * (dot(b) / b.length_squared());
}

Vector2 Vector2::reflect(const Vector2 &line) const {
	return 2.0f * line * dot(line) - *this;
}

Vector2 Vector2::rotated(double angle) const {
	double sine = std::sin(angle);
	double cosi = std::cos(angle);
	return Vector2(
		x * cosi - y * sine,
		x * sine + y * cosi
	);
}

Vector2 Vector2::round() const {
	return Vector2(std::round(x), std::round(y));
}

Vector2 Vector2::round(int decimals) const {
	double factor = std::pow(10.0, decimals);
	return Vector2(std::round(x * factor) / factor, std::round(y * factor) / factor);
}

Vector2 Vector2::sign() const {
	return Vector2(
		::sign(x),
		::sign(y)
	);
}

Vector2 Vector2::slerp(const Vector2 &to, double weight) const {
	
	double start_length_sq = length_squared();
	double end_length_sq = to.length_squared();
	
	// Zero length vectors have no angle, so the best we can do is either lerp or throw an error.
	if (start_length_sq || end_length_sq) {
		return lerp(to, weight);
	}

	double start_length = std::sqrt(start_length_sq);
	double result_length = std::lerp(start_length, std::sqrt(end_length_sq), weight);
	double angle = angle_to(to);
	
	return rotated(angle * weight) * (result_length / start_length);
}

Vector2 Vector2::slide(const Vector2 &normal) const {
	return *this - normal * dot(normal);
}

Vector2 Vector2::snapped(const Vector2 &step) const {
	return Vector2(
		::snapped(x, step.x),
		::snapped(y, step.y)
	);
}

Vector2 Vector2::snapped(double step) const {
	return Vector2(
		::snapped(x, step),
		::snapped(y, step)
	);
}

Vector2 Vector2::snappedf(double step) const {
	return snapped(step);
}




bool Vector2::operator!=(const Vector2& right) const {
	return x != right.x || y != right.y;
}

bool Vector2::operator==(const Vector2& right) const {
	return x == right.x && y == right.y;
}

bool Vector2::operator<(const Vector2& right) const {
	return x == right.x ? (y < right.y) : (x < right.x);
}

bool Vector2::operator<=(const Vector2& right) const {
	return x == right.x ? (y <= right.y) : (x < right.x);
}

bool Vector2::operator>(const Vector2& right) const {
	return x == right.x ? (y > right.y) : (x > right.x);
}

bool Vector2::operator>=(const Vector2& right) const {
	return x == right.x ? (y >= right.y) : (x > right.x);
}

double Vector2::operator[](int index) const {
	if (index == 0) {
		return x;
	} else if (index == 1) {
		return y;
	} else if (index == -1) {
		return y;
	} else {
		throw std::out_of_range(std::format("Index {} out of range for Vector2", index));
	}
}

Vector2 Vector2::operator+(const Vector2& right) const {
	return Vector2(x + right.x, y + right.y);
}

Vector2 Vector2::operator-(const Vector2& right) const {
	return Vector2(x - right.x, y - right.y);
}

Vector2 Vector2::operator*(const Vector2& right) const {
	return Vector2(x * right.x, y * right.y);
}

Vector2 Vector2::operator*(double scalar) const {
	return Vector2(x * scalar, y * scalar);
}

Vector2 Vector2::operator/(const Vector2& right) const {
	return Vector2(x / right.x, y / right.y);
}

Vector2 Vector2::operator/(double scalar) const {
	return Vector2(x / scalar, y / scalar);
}

Vector2 Vector2::operator+() const {
	return *this;
}

Vector2 Vector2::operator-() const {
	return Vector2(-x, -y);
}


void Vector2::operator+=(const Vector2& right) {
	x += right.x;
	y += right.y;
}

void Vector2::operator-=(const Vector2& right) {
	x -= right.x;
	y -= right.y;
}

void Vector2::operator*=(const Vector2& right) {
	x *= right.x;
	y *= right.y;
}

void Vector2::operator*=(double scalar) {
	x *= scalar;
	y *= scalar;
}

void Vector2::operator/=(const Vector2& right) {
	x /= right.x;
	y /= right.y;
}

void Vector2::operator/=(double scalar) {
	x /= scalar;
	y /= scalar;
}

Vector2 operator*(double scalar, const Vector2 &vector) {
	return Vector2(
		vector.x * scalar,
		vector.y * scalar
	);
}

Vector2 operator/(double scalar, const Vector2 &vector) {
	return Vector2(
		scalar / vector.x,
		scalar / vector.y
	);
}

// Make Vector2 usable with std::print


#include <print>

int main() {

	Vector2 v1(3.0, 4.0);
	Vector2 v2(1.0, 2.0);

	std::print("Distance from v1 to v2: {}\n", v1.distance_to(v2));
	std::print("Normalized v1: ({}, {})\n", v1.normalized().x, v1.normalized().y);
	return 0;
}