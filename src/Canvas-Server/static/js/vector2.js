
export class Vector2 {

	constructor(x, y) {

		// Check if x is an object with x and y properties
		if (typeof x === "object" && x !== null && "x" in x && "y" in x) {
			this.x = x.x;
			this.y = x.y;
		} else if (typeof x === "object" && x !== null) {
			this.x = x[0];
			this.y = x[1];
		} else {
			this.x = x;
			this.y = y;
		}

		// Check if x and y are valid numbers
		if (typeof this.x !== "number" || typeof this.y !== "number") {
			throw new Error("Invalid arguments for Vector2 constructor");
		}
	}

	add(other) {
		return new Vector2(this.x + other.x, this.y + other.y);
	}

	sub(other) {
		return new Vector2(this.x - other.x, this.y - other.y);
	}

	mul(scalar) {
		return new Vector2(this.x * scalar, this.y * scalar);
	}

	cross(other) {
		return this.x * other.y - this.y * other.x;
	}

	dot(other) {
		return this.x * other.x + this.y * other.y;
	}

	isSameDirection(other) {
		return this.cross(other) === 0 && this.dot(other) >= 0;
	}

	length() {
		return Math.sqrt(this.x ** 2 + this.y ** 2);
	}

	normalize() {
		const length = this.length();

		if (length === 0) {
			return new Vector2(0, 0);
		}

		return new Vector2(this.x / length, this.y / length);
	}

	reflect(normal) {
		const normalizedNormal = normal.normalize();
		const dot = this.dot(normalizedNormal);
		return this.sub(normalizedNormal.mul(2 * dot));
	}

	perpendicular() {
		return new Vector2(-this.y, this.x);
	}

	reflectSegment(start, end) {
		return start.add(this.sub(start).reflect(end.sub(start).perpendicular()));
	}

	distanceTo(other) {
		return this.sub(other).length();
	}

	/**
	 * Clamps the vector's components between `min` and `max` or their corresponding components.
	 *
	 * @param {(number | Vector2)} min - The first number or the vector containing the minimum values for x and y.
	 * @param {(number | Vector2)} max - The second number or the vector containing the maximum values for x and y.
	 * @returns {Vector2} A new vector with the clamped components.
	 */
	clamp(min, max) {

		let minX, minY, maxX, maxY;

		if (typeof min === "number") {
			minX = minY = min;
		} else {
			minX = min.x;
			minY = min.y;
		}

		if (typeof max === "number") {
			maxX = maxY = max;
		} else {
			maxX = max.x;
			maxY = max.y;
		}

		return new Vector2(
			Math.max(minX, Math.min(this.x, maxX)),
			Math.max(minY, Math.min(this.y, maxY))
		);
	}

	clone() {
		return new Vector2(this.x, this.y);
	}
}