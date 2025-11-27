
#pragma once

class Vector2 {

	public:
	double x, y;

	enum Axis {
		AXIS_X,
		AXIS_Y,
	};

	static const Vector2 ZERO;
	static const Vector2 ONE;
	static const Vector2 INF;
	static const Vector2 LEFT;
	static const Vector2 RIGHT;
	static const Vector2 UP;
	static const Vector2 DOWN;

	Vector2();
	Vector2(const Vector2& from);
	Vector2(double x, double y);

	Vector2 abs() const;
	double  angle() const;
	double  angle_to(const Vector2 &to) const;
	double  angle_to_point(const Vector2 &point) const;
	double  aspect() const;
	Vector2 bezier_derivative(const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end, double t) const;
	Vector2 bezier_interpolate(const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end, double t) const;
	Vector2 bounce(const Vector2 &normal) const;
	Vector2 ceil() const;
	Vector2 clamp(const Vector2 &min, const Vector2 &max) const;
	Vector2 clamp(double min, double max) const;
	Vector2 clampf(double min, double max) const;
	double  cross(const Vector2 &other) const;
	Vector2 cubic_interpolate(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight) const;
	Vector2 cubic_interpolate_in_time(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight, double b_t, double pre_a_t, double post_b_t) const;
	Vector2 direction_to(const Vector2 &to) const;
	double  distance_squared_to(const Vector2 &to) const;
	double  distance_to(const Vector2 &to) const;
	double  dot(const Vector2 &other) const;
	Vector2 floor() const;
	static  Vector2 from_angle(double angle);
	bool    is_equal_approx(const Vector2 &v) const;
	bool    is_finite() const;
	bool    is_normalized() const;
	bool    is_zero_approx() const;
	double  length() const;
	double  length_squared() const;
	Vector2 lerp(const Vector2 &to, double weight) const;
	Vector2 limit_length(double length = 1.0) const;
	Vector2 max(const Vector2 &with) const;
	Vector2 max(double with) const;
	Axis    max_axis_index() const;
	Vector2 maxf(double with) const;
	Vector2 min(const Vector2 &with) const;
	Vector2 min(double with) const;
	Axis    min_axis_index() const;
	Vector2 minf(double with) const;
	Vector2 move_toward(const Vector2 &to, double delta) const;
	Vector2 normalized() const;
	Vector2 orthogonal() const;
	Vector2 posmod(double mod) const;
	Vector2 posmod(const Vector2 &modv) const;
	Vector2 posmodv(const Vector2 &modv) const;
	Vector2 project(const Vector2 &b) const;
	Vector2 reflect(const Vector2 &line) const;
	Vector2 rotated(double angle) const;
	Vector2 round() const;
	Vector2 sign() const;
	Vector2 slerp(const Vector2 &to, double weight) const;
	Vector2 slide(const Vector2 &normal) const;
	Vector2 snapped(const Vector2 &step) const;
	Vector2 snapped(double step) const;
	Vector2 snappedf(double step) const;

	bool    operator!=(const Vector2& other) const;
	bool    operator==(const Vector2& other) const;
	bool    operator<(const Vector2& other) const;
	bool    operator<=(const Vector2& other) const;
	bool    operator>(const Vector2& other) const;
	bool    operator>=(const Vector2& other) const;
	double  operator[](int index) const;
	Vector2 operator+(const Vector2& other) const;
	Vector2 operator-(const Vector2& other) const;
	Vector2 operator*(double scalar) const;
	Vector2 operator/(double scalar) const;
	Vector2 operator+() const;
	Vector2 operator-() const;
};
