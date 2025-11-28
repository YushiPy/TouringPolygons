
#pragma once

class Vector2 {

	public:
	/*The vector's X component. Also accessible by using the index position `[0]`.*/
	double x = 0.0;
	/*The vector's Y component. Also accessible by using the index position `[1]`.*/
	double y = 0.0;

	enum Axis {
		/*Enumerated value for the X axis. Returned by `max_axis_index()` and `min_axis_index()`.*/
		AXIS_X,
		/*Enumerated value for the Y axis. Returned by `max_axis_index()` and `min_axis_index()`.*/
		AXIS_Y,
	};

	/*Zero vector, a vector with all components set to `0`.*/
	static const Vector2 ZERO;
	/*One vector, a vector with all components set to `1`.*/
	static const Vector2 ONE;
	/*Infinity vector, a vector with all components set to `INFINITY`.*/
	static const Vector2 INF;
	/*Left unit vector. Represents the direction of left.*/
	static const Vector2 LEFT;
	/*Right unit vector. Represents the direction of right.*/
	static const Vector2 RIGHT;
	/*Up unit vector. Y is down in 2D, so this vector points -Y.*/
	static const Vector2 UP;
	/*Down unit vector. Y is down in 2D, so this vector points +Y.*/
	static const Vector2 DOWN;

	/*Constructs a default-initialized Vector2 with all components set to 0.*/
	Vector2();
	/*Constructs a Vector2 as a copy of the given Vector2.*/
	Vector2(const Vector2& from);
	/*Constructs a new Vector2 from the given `x` and `y`.*/
	Vector2(double x, double y);

	/*Returns a new vector with all components in absolute values (i.e. positive).*/
	Vector2 abs() const;
	/*Returns this vector's angle with respect to the positive X axis, or `(1, 0)` vector, in radians. 
	Equivalent to `atan2(y, x)`. 
	[Illustration of returned angle](https://raw.githubusercontent.com/godotengine/godot-docs/master/img/vector2_angle.png).*/
	double  angle() const;
	/*Returns the signed angle to the given vector, in radians.
	[Illustration of returned angle](https://raw.githubusercontent.com/godotengine/godot-docs/master/img/vector2_angle_to.png).*/
	double  angle_to(const Vector2 &to) const;
	/*Returns the angle between the line connecting the two points and the X axis, in radians. 
	`a.angle_to_point(b)` is equivalent of doing `(b - a).angle()`.
	[Illustration of returned angle](https://raw.githubusercontent.com/godotengine/godot-docs/master/img/vector2_angle_to_point.png).*/
	double  angle_to_point(const Vector2 &point) const;
	/*Returns the aspect ratio of this vector, the ratio of `x` to `y`.*/
	double  aspect() const;
	/*Returns the derivative at the given `t` on the [Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve) defined by this vector and the given `control_1`, `control_2`, and `end` points.*/
	Vector2 bezier_derivative(const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end, double t) const;
	/*Returns the point at the given t on the [Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve) defined by this vector and the given control_1, control_2, and end points.*/
	Vector2 bezier_interpolate(const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end, double t) const;
	/*Returns the vector "bounced off" from a line defined by the given `normal` perpendicular to the line.
	Note: `bounce()` performs the operation that most engines and frameworks call `reflect()`.*/
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
