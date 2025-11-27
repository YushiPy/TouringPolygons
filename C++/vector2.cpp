
#include "vector2.h"

Vector2::Vector2() : x(0), y(0) {}
Vector2::Vector2(const Vector2& from) : x(from.x), y(from.y) {}
Vector2::Vector2(double p_x, double p_y) : x(p_x), y(p_y) {}

Vector2 Vector2::operator+(const Vector2& other) const {
	return Vector2(x + other.x, y + other.y);
}

Vector2 Vector2::operator-(const Vector2& other) const {
	return Vector2(x - other.x, y - other.y);
}

Vector2 Vector2::operator*(double scalar) const {
	return Vector2(x * scalar, y * scalar);
}

Vector2 Vector2::operator/(double scalar) const {
	return Vector2(x / scalar, y / scalar);
}
