
import math
from operator import neg

from itertools import islice

from collections.abc import Callable, Iterable, Iterator
from typing import Literal, Self, overload


class Vector2:

	x: float
	y: float

	INF: Self

	@overload
	def __init__(self) -> None: ...
	@overload
	def __init__(self, x: float, y: float) -> None: ...
	@overload
	def __init__(self, x: Iterable[float]) -> None: ...

	def __init__(self, x: float | Iterable[float] = 0, y: float = 0) -> None:

		if isinstance(x, Iterable):
			x, y = x

		self.x = x
		self.y = y

	@classmethod
	def inf(cls) -> Self:
		"""Returns a vector with both components set to infinity."""
		return cls(math.inf, math.inf)

	@overload
	@staticmethod
	def from_polar(r: float, theta: float) -> "Vector2": ...
	@overload
	@staticmethod
	def from_polar(r: Iterable[float]) -> "Vector2": ...

	@overload
	@staticmethod
	def from_polar(r: float = 1.0, *, theta: float) -> "Vector2": ...

	@staticmethod
	def from_polar(r: float | Iterable[float] = 1.0, theta: float | None = None) -> "Vector2":

		if theta is None:
			if isinstance(r, Iterable):
				r, theta = r
			else:
				r, theta = 1.0, r

		elif isinstance(r, Iterable):
			raise ValueError("If theta is provided, r must be a float.")

		x = r * math.cos(theta)
		y = r * math.sin(theta)

		return Vector2(x, y)
	
	@staticmethod
	def from_bytes(data: bytes) -> "Vector2":
		"""Creates a Vector2 from its IEE-754 representation as 16 bytes."""

		import struct

		if len(data) != 16:
			raise ValueError("Data must be exactly 16 bytes long to convert to Vector2.")

		x = struct.unpack('<d', data[:8])[0]
		y = struct.unpack('<d', data[8:])[0]

		return Vector2(x, y)

	def dot(self, other: Self) -> float:
		"""Returns the dot product with the other vector."""
		return self.x * other.x + self.y * other.y
	
	def cross(self, other: Self) -> float:
		"""Returns the cross product with the other vector."""
		return self.x * other.y - self.y * other.x
	
	def magnitude(self) -> float:
		"""Returns the magnitude of the vector."""
		return math.sqrt(self.magnitude_squared())

	def magnitude_squared(self) -> float:
		"""Returns the squared magnitude of the vector."""
		return self.x * self.x + self.y * self.y
	
	def length(self) -> float:
		"""Returns the length of the vector."""
		return self.magnitude()

	def length_squared(self) -> float:
		"""Returns the squared length of the vector."""
		return self.magnitude_squared()

	def normalize(self) -> Self:
		"""Returns a normalized version of the vector (magnitude == 1)."""
		return self.scale_to_length(1)

	def normalize_ip(self) -> None:
		"""Normalizes the vector in place (magnitude == 1)."""
		self.scale_to_length_ip(1)

	def scale_to_length(self, value: float) -> Self:
		"""Returns a scaled version of the vector with the desired length, maintaining the direction."""

		result = self.copy()
		result.scale_to_length_ip(value)

		return result
	
	def scale_to_length_ip(self, value: float) -> None:
		"""Scales the vector to the desired length, maintaining the direction."""
		mult = value / self.magnitude()

		if mult == 0:
			raise ValueError("Can't scale a vector of length Zero")

		self.x *= mult
		self.y *= mult
	
	def reflect(self, normal: Self) -> Self:
		"""Returns a reflected version of the vector by the given normal."""

		result = self.copy()
		result.reflect_ip(normal)

		return result

	def reflect_ip(self, normal: Self) -> None:
		"""Reflects the vector in place using the given normal vector."""
		if normal.magnitude_squared() == 0:
			raise ValueError("Can't reflect a vector with a zero-length normal")

		self -= 2 * self.dot(normal) / normal.magnitude_squared() * normal

	def reflect_segment(self, start: Self, end: Self) -> Self:
		"""Reflects the vector across the line segment defined by start and end points."""
		return (self - start).reflect((end - start).perpendicular()) + start

	def distance_to(self, other: Self) -> float:
		"""Returns the euclidean distance between the tips of the vectors."""
		return (self - other).magnitude()

	def distance_squared_to(self, other: Self) -> float:
		"""Returns the squared euclidean distance between the tips of the vectors."""
		return (self - other).magnitude_squared()

	def move_towards(self, target: Self, step: float) -> Self:
		"""Returns a version of the vector that is moved towards the target by the given step size."""

		result = self.copy()
		result.move_towards_ip(target, step)

		return result

	def move_towards_ip(self, target: Self, step: float) -> None:

		diff = target - self
		distance = diff.magnitude()

		if step < 0:
			diff *= -1
			step = -step
			distance = step

		if step >= distance:
			self.update(target)
			return

		self += diff.scale_to_length(step)

	@overload
	def lerp(self, other: Self, rate: float) -> Self: ...
	@overload
	def lerp(self, other: Self, rate: Iterable[float]) -> Self: ...

	def lerp(self, other: Self, rate: float | Iterable[float]) -> Self:
		"""
		Returns the linear interpolation of `this` vector with `other`.
		
		If `rate` is a float, uses it for both components.
		If `rate` is an iterable, uses the first value for x and the second for y.
		"""
		result = self.copy()
		result.lerp_ip(other, rate)

		return result

	@overload
	def lerp_ip(self, other: Self, rate: float) -> Self: ...
	@overload
	def lerp_ip(self, other: Self, rate: Iterable[float]) -> Self: ...

	def lerp_ip(self, other: Self, rate: float | Iterable[float]) -> Self:
		"""
		Linearly interpolates `this` vector with `other`.
		
		If `rate` is a float, uses it for both components.
		If `rate` is an iterable, uses the first value for x and the second for y.
		"""

		if isinstance(rate, Iterable):

			values = list(islice(rate, 2))

			if len(values) != 2:
				raise ValueError("Rate must be an iterable with two values (x, y).")
			
			rate_x, rate_y = values
		else:
			rate_x = rate_y = rate

		self.x += (other.x - self.x) * rate_x
		self.y += (other.y - self.y) * rate_y

		return self

	def slerp(self, other: Self, rate: float) -> Self: 
		"""Returns the spherical interpolation of `this` vector with `other`."""
		if rate <= 0:
			return self.copy()
		if rate >= 1:
			return other.copy()

		a = self.normalize()
		b = other.normalize()

		dot_product = a.dot(b)

		if dot_product == 1:
			if a == b:
				return self.copy()
			else:
				raise ValueError("Slerp with 180 degrees is undefined.")

		omega = math.acos(dot_product)
		sin_omega = math.sqrt(1 - dot_product * dot_product)

		a_rate = math.sin((1 - rate) * omega) / sin_omega
		b_rate = math.sin(rate * omega) / sin_omega

		result = a * a_rate + b * b_rate
		mag = (self.magnitude() * (1 - rate) + other.magnitude() * rate)

		return result * mag

	def slerp_ip(self, other: Self, rate: float) -> None:
		"""Sphericaly interpolates `this` with `other` in place."""
		self.update(self.slerp(other, rate))

	def rotate(self, angle: float) -> Self:
		"""Returns a vector rotated counterclockwise by the given angle in radians."""

		result = self.copy()
		result.rotate_ip(angle)

		return result

	def rotate_ip(self, angle: float) -> None:

		cos_angle = math.cos(angle)
		sin_angle = math.sin(angle)

		new_x = self.x * cos_angle - self.y * sin_angle
		new_y = self.y * cos_angle + self.x * sin_angle

		self.x = new_x
		self.y = new_y

	def angle(self) -> float:
		"""Returns the angle of the vector with the x-axis."""
		return math.atan2(self.y, self.x)

	def angle_to(self, other: Self) -> float:
		"""Returns the angle between this vector and the other vector in radians."""

		a = self.magnitude()
		b = other.magnitude()

		if a == 0 or b == 0:
			raise ValueError("Can't calculate angle with a zero-length vector")

		return math.acos(self.dot(other) / (a * b))

	def as_polar(self) -> Self:
		"""Returns the polar coordinates of the vector as a Vector2(r, theta)."""
		
		r = self.magnitude()
		theta = math.atan2(self.y, self.x)
		
		return type(self)(r, theta)
	
	def quadrant(self) -> Literal[0, 1, 2, 3]:
		"""
		Returns the quadrant of the vector in the Cartesian plane.
		0: First quadrant (x >= 0, y >= 0)
		1: Second quadrant (x < 0, y >= 0)
		2: Third quadrant (x < 0, y < 0)
		3: Fourth quadrant (x >= 0, y < 0)
		"""
		
		if self.x >= 0 and self.y >= 0:
			return 0
		elif self.x < 0 and self.y >= 0:
			return 1
		elif self.x < 0 and self.y < 0:
			return 2
		else:
			return 3
	
	def bbox(self, other: Self) -> tuple[Self, Self]:
		"""Returns the bounding box of this vector and the other vector as two corners."""
		min_x = min(self.x, other.x)
		max_x = max(self.x, other.x)
		min_y = min(self.y, other.y)
		max_y = max(self.y, other.y)

		return (type(self)(min_x, min_y), type(self)(max_x, max_y))

	def project(self, other: Self) -> Self:
		"""Returns the projection of `this` onto `other`."""

		mag_sq = other.magnitude_squared()

		if mag_sq == 0:
			raise ValueError("Can't project a vector onto a zero-length vector")

		return other * (self.dot(other) / mag_sq)

	def project_ip(self, other: Self) -> None:
		"""Projects `this` onto `other` in place."""
		self.update(self.project(other))

	def clamp_magnitude(self, min_magnitude: float, max_maginitude: float) -> Self:
		"""Returns a version of the vector clamped to the given minimum and maximum values."""

		vector = self.copy()
		vector.clamp_magnitude_ip(min_magnitude, max_maginitude)

		return vector

	def clamp_magnitude_ip(self, min_magnitude: float, max_magnitude: float) -> None:
		"""Clamps the vector in place to the given minimum and maximum values."""

		mag = self.magnitude()

		if mag < min_magnitude:
			self.scale_to_length_ip(min_magnitude)
		if mag > max_magnitude:
			self.scale_to_length_ip(max_magnitude)

	def to_bytes(self) -> bytes:
		"""
		Returns the IEE-754 representation of the vector as 16 bytes.
		Uses little-endian format, with the x coordinate as the first 8 bytes and the y coordinate as the second 8 bytes.
		"""
		import struct
		return struct.pack('<d', self.x) + struct.pack('<d', self.y)

	def to_tuple(self) -> tuple[float, float]:
		"""Returns the vector as a tuple of (x, y)."""
		return (self.x, self.y)

	@overload
	def update(self) -> None: 
		"""Sets coordinates of `this` to `0`."""
		...
	@overload
	def update(self, x: float) -> None: 
		"""Sets coordinates of `this` to `x`, respectively."""
		...
	@overload
	def update(self, x: Self) -> None: 
		"""Sets coordinates of `this` to the coordinates of `x` vector."""
		...
	@overload
	def update(self, x: float, y: float) -> None: 
		"""Sets coordinates of `this` to `x` and `y`, respectively."""
		...

	def update(self, x: float | Self = 0.0, y: float = 0.0) -> None:

		if isinstance(x, Vector2):
			x, y = x.x, x.y # type: ignore

		self.x = x
		self.y = y

	def map(self, func: Callable[[float], float]) -> Self:
		"""Returns a new vector with the components mapped by the given function."""
		return type(self)(func(self.x), func(self.y))

	def perpendicular(self) -> Self:
		"""Returns a vector that is perpendicular to this vector and has same length."""
		return type(self)(-self.y, self.x)

	def copy(self) -> Self:
		"""Returns a copy of the vector."""
		return type(self)(self.x, self.y)
	
	def xx(self) -> Self:
		"""Returns a vector with both components equal to x."""
		return type(self)(self.x, self.x)

	def xy(self) -> Self:
		"""Returns a vector with x and y components."""
		return type(self)(self.x, self.y)

	def yx(self) -> Self:
		"""Returns a vector with y and x components."""
		return type(self)(self.y, self.x)

	def yy(self) -> Self:
		"""Returns a vector with both components equal to y."""
		return type(self)(self.y, self.y)

	def is_close(self, other: Self, eps: float = 1e-8) -> bool:
		"""Checks if this vector is close to the other vector within a given epsilon."""
		return (self - other).magnitude_squared() < eps * eps

	def __add__(self, other: Self) -> Self:
		"""Returns a new vector that is the sum of this vector and the other vector."""
		return type(self)(self.x + other.x, self.y + other.y)

	def __iadd__(self, other: Self) -> Self:
		"""Adds the other vector to this vector in place."""

		self.x += other.x
		self.y += other.y

		return self

	def __sub__(self, other: Self) -> Self:
		"""Returns a new vector that is the difference of this vector and the other vector."""
		return type(self)(self.x - other.x, self.y - other.y)

	def __isub__(self, other: Self) -> Self:
		"""Subtracts the other vector from this vector in place."""

		self.x -= other.x
		self.y -= other.y

		return self

	def __neg__(self) -> Self:
		"""Returns a new vector that is the negation of this vector."""
		return self.map(neg)

	def __pos__(self) -> Self:
		"""Returns a new vector that is the same as this vector."""
		return self.copy()

	def __mul__(self, scalar: float) -> Self:
		"""Returns a new vector that is this vector scaled by the scalar."""
		return self.map(lambda x: x * scalar)

	def __rmul__(self, scalar: float) -> Self:
		"""Returns a new vector that is this vector scaled by the scalar."""
		return self * scalar

	def __imul__(self, scalar: float) -> Self:
		"""Scales this vector by the scalar in place."""

		self.x *= scalar
		self.y *= scalar

		return self

	def __truediv__(self, scalar: float) -> Self:
		"""Returns a new vector that is this vector divided by the scalar."""
		return self.map(lambda x: x / scalar)

	def __itruediv__(self, scalar: float) -> Self:
		"""Divides this vector by the scalar in place."""

		if scalar == 0:
			raise ValueError("Division by zero")

		self.x /= scalar
		self.y /= scalar

		return self

	def __eq__(self, other: object) -> bool:
		"""Checks if this vector is equal to the other vector."""

		if not isinstance(other, type(self)):
			return False

		return self.x == other.x and self.y == other.y

	def __ne__(self, other: object) -> bool:
		"""Checks if this vector is not equal to the other vector."""
		return not self.__eq__(other)

	def __lt__(self, other: Self) -> bool:
		"""Checks if this vector is less than the other vector based on the magnitude."""
		return self.magnitude_squared() < other.magnitude_squared()
	
	def __le__(self, other: Self) -> bool:
		"""Checks if this vector is less than or equal to the other vector based on the magnitude."""
		return self.magnitude_squared() <= other.magnitude_squared()

	def __gt__(self, other: Self) -> bool:
		"""Checks if this vector is greater than the other vector based on the magnitude."""
		return self.magnitude_squared() > other.magnitude_squared()

	def __ge__(self, other: Self) -> bool:
		"""Checks if this vector is greater than or equal to the other vector based on the magnitude."""
		return self.magnitude_squared() >= other.magnitude_squared()

	def __iter__(self) -> Iterator[float]:
		"""Returns an iterator over the components of the vector."""
		yield self.x
		yield self.y

	def __getitem__(self, index: int) -> float:
		"""Returns the component of the vector at the given index."""

		if index == 0:
			return self.x
		elif index == 1:
			return self.y
		else:
			raise IndexError("Index out of range for Vector2")

	def __setitem__(self, index: int, value: float) -> None:
		"""Sets the component of the vector at the given index to the value."""

		if index == 0:
			self.x = value
		elif index == 1:
			self.y = value
		else:
			raise IndexError("Index out of range for Vector2")
	
	def __len__(self) -> int:
		"""Returns the number of components in the vector."""
		return 2

	def __bool__(self) -> bool:
		"""Returns True if the vector is not zero, False otherwise."""
		return self.x != 0 or self.y != 0
	
	def __hash__(self) -> int:
		return hash((self.x, self.y))

	def __nonzero__(self) -> bool:
		"""Returns True if the vector is not zero, False otherwise."""
		return self.__bool__()

	def __abs__(self) -> Self:
		"""Returns the absolute value of the vector."""
		return self.map(abs)

	def __floor__(self) -> Self:
		"""Returns a new vector with the components floored to the nearest integer."""
		return self.map(math.floor)

	def __ceil__(self) -> Self:
		"""Returns a new vector with the components ceiled to the nearest integer."""
		return self.map(math.ceil)

	def __round__(self, ndigits: int = 0) -> Self:
		"""Returns a new vector with the components rounded to the given number of decimal places."""
		return self.map(lambda x: round(x, ndigits))

	def __str__(self) -> str:
		"""Returns a string representation of the vector."""
		return f"({round(self.x, 6)}, {round(self.y, 6)})"

	def __repr__(self) -> str:
		"""Returns a string representation of the vector."""
		return f"{type(self).__name__}({self.x}, {self.y})"

	def __copy__(self) -> Self:
		"""Returns a shallow copy of the vector."""
		return type(self)(self.x, self.y)

	def __sizeof__(self) -> int:
		return super().__sizeof__() + self.x.__sizeof__() + self.y.__sizeof__()
