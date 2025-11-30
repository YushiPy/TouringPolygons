
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

// std math
#include <cmath>

/*Definition of static constant vectors*/
const Vector2 Vector2::ZERO = Vector2(0.0, 0.0);
const Vector2 Vector2::ONE = Vector2(1.0, 1.0);
const Vector2 Vector2::INF = Vector2(INFINITY, INFINITY);
const Vector2 Vector2::LEFT = Vector2(-1.0, 0.0);
const Vector2 Vector2::RIGHT = Vector2(1.0, 0.0);
const Vector2 Vector2::UP = Vector2(0.0, -1.0);
const Vector2 Vector2::DOWN = Vector2(0.0, 1.0);

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

#include <print>

int main() {
	std::print("Hello, World!\n");
	Vector2 v(3.0, 4.0);
}