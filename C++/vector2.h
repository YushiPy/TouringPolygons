
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
	/*Returns a new vector with all components rounded up (towards positive infinity).*/
	Vector2 ceil() const;
	/*Returns a new vector with all components clamped between the components of `min` and `max`, by running `clamp()` on each component.*/
	Vector2 clamp(const Vector2 &min, const Vector2 &max) const;
	/*Returns a new vector with all components clamped between `min` and `max`, by running `clamp()` on each component.*/
	Vector2 clamp(double min, double max) const;
	/*Alias of `clamp(min, max)`.*/
	Vector2 clampf(double min, double max) const;
	/*Returns the 2D analog of the cross product for this vector and `with`.

	This is the signed area of the parallelogram formed by the two vectors. If the second vector is clockwise from the first vector, then the cross product is the positive area. If counter-clockwise, the cross product is the negative area. If the two vectors are parallel this returns zero, making it useful for testing if two vectors are parallel.

	Note: Cross product is not defined in 2D mathematically. This method embeds the 2D vectors in the XY plane of 3D space and uses their cross product's Z component as the analog.*/
	double  cross(const Vector2 &with) const;
	/*Performs a cubic interpolation between this vector and `b` using `pre_a` and `post_b` as handles, and returns the result at position `weight`. `weight` is on the range of `0.0` to `1.0`, representing the amount of interpolation.*/
	Vector2 cubic_interpolate(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight) const;
	/*Performs a cubic interpolation between this vector and `b` using `pre_a` and `post_b` as handles, and returns the result at position `weight`. `weight` is on the range of `0.0` to `1.0`, representing the amount of interpolation.

	It can perform smoother interpolation than `cubic_interpolate()` by the time values.*/
	Vector2 cubic_interpolate_in_time(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight, double b_t, double pre_a_t, double post_b_t) const;
	/*Returns the normalized vector pointing from this vector to `to`. This is equivalent to using `(b - a).normalized()`.*/
	Vector2 direction_to(const Vector2 &to) const;
	/*Returns the squared distance between this vector and `to`.

	This method runs faster than `distance_to()`, so prefer it if you need to compare vectors or need the squared distance for some formula.*/
	double  distance_squared_to(const Vector2 &to) const;
	/*Returns the distance between this vector and `to`.*/
	double  distance_to(const Vector2 &to) const;
	/*Returns the dot product of this vector and `with`. This can be used to compare the angle between two vectors. For example, this can be used to determine whether an enemy is facing the player.

	The dot product will be `0` for a right angle (90 degrees), greater than 0 for angles narrower than 90 degrees and lower than 0 for angles wider than 90 degrees.

	When using unit (normalized) vectors, the result will always be between `-1.0` (180 degree angle) when the vectors are facing opposite directions, and `1.0` (0 degree angle) when the vectors are aligned.

	Note: `a.dot(b)` is equivalent to `b.dot(a)`.*/
	double  dot(const Vector2 &with) const;
	/*Returns a new vector with all components rounded down (towards negative infinity).*/
	Vector2 floor() const;
	/*Creates a Vector2 rotated to the given `angle` in radians. This is equivalent to doing `Vector2(cos(angle), sin(angle))` or `Vector2.RIGHT.rotated(angle)`.

	```
	Vector2.from_angle(0) -> (1.0, 0.0)
	Vector2(1, 0).angle() -> 0.0, which is the angle used above.
	Vector2.from_angle(PI / 2) -> (0.0, 1.0)
	```
	
	Note: The length of the returned Vector2 is approximately `1.0`, but is is not guaranteed to be exactly `1.0` due to floating-point precision issues. Call `normalized()` on the returned Vector2 if you require a unit vector.*/
	static  Vector2 from_angle(double angle);
	/*Returns `true` if this vector and to are approximately equal, by checking if each component is approximately equal.*/
	bool    is_equal_approx(const Vector2 &v) const;
	/*Returns `true` if this vector is finite, by checking if each component is finite.*/
	bool    is_finite() const;
	/*Returns `true` if the vector is normalized, i.e. its length is approximately equal to 1.*/
	bool    is_normalized() const;
	/*
	Returns `true` if this vector's values are approximately zero, by running checking if each component is approximately zero.

	This method is faster than using `is_equal_approx()` with one value as a zero vector.*/
	bool    is_zero_approx() const;
	/*Returns the length (magnitude) of this vector.*/
	double  length() const;
	/*Returns the squared length (squared magnitude) of this vector.

	This method runs faster than `length()`, so prefer it if you need to compare vectors or need the squared distance for some formula.*/
	double  length_squared() const;
	/*Returns the result of the linear interpolation between this vector and `to` by amount `weight`. `weight` is on the range of `0.0` to `1.0`, representing the amount of interpolation.*/
	Vector2 lerp(const Vector2 &to, double weight) const;
	/*Returns the vector with a maximum length by limiting its length to `length`. If the vector is non-finite, the result is undefined.*/
	Vector2 limit_length(double length = 1.0) const;
	/*Returns the component-wise maximum of `this` and `with`, equivalent to `Vector2(max(x, with.x), max(y, with.y))`.*/
	Vector2 max(const Vector2 &with) const;
	/*Returns the component-wise maximum of `this` and `with`, equivalent to `Vector2(maxf(x, with), maxf(y, with))`.*/
	Vector2 max(double with) const;
	/*Returns the axis of the vector's highest value. See `AXIS_*` constants. If all components are equal, this method returns `AXIS_X`.*/
	Axis    max_axis_index() const;
	/*Alias of `max(with)`*/
	Vector2 maxf(double with) const;
	/*Returns the component-wise minimum of `this` and `with`, equivalent to `Vector2(min(x, with.x), min(y, with.y))`.*/
	Vector2 min(const Vector2 &with) const;
	/*Returns the component-wise minimum of `this` and `with`, equivalent to `Vector2(min(x, with), min(y, with))`.*/
	Vector2 min(double with) const;
	/*Returns the axis of the vector's lowest value. See `AXIS_*` constants. If all components are equal, this method returns `AXIS_Y`.*/
	Axis    min_axis_index() const;
	/*Alias of `min(with)`*/
	Vector2 minf(double with) const;
	/*Returns a new vector moved toward `to` by the fixed `delta` amount. Will not go past the final value.*/
	Vector2 move_toward(const Vector2 &to, double delta) const;
	/*Returns the result of scaling the vector to unit length. Equivalent to `v / v.length()`. Returns `(0, 0)` if `v.length() == 0`. See also `is_normalized()`.

	Note: This function may return incorrect values if the input vector length is near zero.*/
	Vector2 normalized() const;
	/*Returns a perpendicular vector rotated 90 degrees counter-clockwise compared to the original, with the same length.*/
	Vector2 orthogonal() const;
	/*Returns a vector composed of the `posmod()` of this vector's components and `mod`.*/
	Vector2 posmod(double mod) const;
	/*Returns a vector composed of the `posmod()` of this vector's components and `modv`'s components.*/
	Vector2 posmod(const Vector2 &modv) const;
	/*Alias of `posmod(modv)`*/
	Vector2 posmodv(const Vector2 &modv) const;
	/*Returns a new vector resulting from projecting this vector onto the given vector `b`. The resulting new vector is parallel to `b`. See also `slide()`.

	Note: If the vector `b` is a zero vector, the components of the resulting new vector will be `NAN`.*/
	Vector2 project(const Vector2 &b) const;
	/*
	Returns the result of reflecting the vector from a line defined by the given direction vector `line`.

	Note: `reflect()` differs from what other engines and frameworks call `reflect()`. In other engines, `reflect()` takes a normal direction which is a direction perpendicular to the line. Here, you specify the direction of the line directly. See also `bounce()` which does what most engines call `reflect()`.
	*/
	Vector2 reflect(const Vector2 &line) const;
	/*Returns the result of rotating this vector by `angle` (in radians)*/
	Vector2 rotated(double angle) const;
	/*Returns a new vector with all components rounded to the nearest integer, with halfway cases rounded away from zero.*/
	Vector2 round() const;
	/*Returns a new vector with each component set to `1.0` if it's positive, `-1.0` if it's negative, and `0.0` if it's zero. The result is identical to calling `sign()` on each component.*/
	Vector2 sign() const;
	/*
	Returns the result of spherical linear interpolation between this vector and `to`, by amount `weight`. `weight` is on the range of `0.0` to `1.0`, representing the amount of interpolation.

	This method also handles interpolating the lengths if the input vectors have different lengths. For the special case of one or both input vectors having zero length, this method behaves like `lerp()`.
	*/
	Vector2 slerp(const Vector2 &to, double weight) const;
	/*
	Returns a new vector resulting from sliding this vector along a line with normal `normal`. The resulting new vector is perpendicular to `normal`, and is equivalent to this vector minus its projection on `normal`. See also `project()`.

	Note: The vector `normal` must be normalized. See also normalized().
	*/
	Vector2 slide(const Vector2 &normal) const;
	/*Returns a new vector with each component snapped to the nearest multiple of the corresponding component in `step`. This can also be used to round the components to an arbitrary number of decimals.*/
	Vector2 snapped(const Vector2 &step) const;
	/*Returns a new vector with each component snapped to the nearest multiple of `step`. This can also be used to round the components to an arbitrary number of decimals.*/
	Vector2 snapped(double step) const;
	/*Alias of `snapped(step)`*/
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
