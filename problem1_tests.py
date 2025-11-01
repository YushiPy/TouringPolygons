
from math import pi, tau

from problem1_draw import Drawing
from vector2 import Vector2
from polygon2 import Polygon2

def regular(n: int, r: float, start: Vector2 = Vector2(), angle: float = 0) -> Polygon2:
	"""
	Create a regular polygon with `n` vertices and radius `r`.

	:param int n: The number of vertices.
	:param float r: The radius of the polygon.

	:return: A Polygon2 object representing the regular polygon.
	"""
	return Polygon2(start + Vector2.from_polar(r, i * tau / n + angle) for i in range(n))

sol = Drawing(Vector2(-3, 0), Vector2(3, 0), [
	regular(5, 1, start=Vector2(-1, 3), angle=0.1),
	Polygon2([Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)]),
])

test1 = Drawing(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
)

test2 = Drawing(
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		Polygon2([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		Polygon2([Vector2.from_polar(2, i * tau / 3 + pi /4) + Vector2(-3, 4) for i in range(3)]),
		Polygon2([Vector2.from_polar(2, i * tau / 10) + Vector2(5, -4) for i in range(10)]),
		Polygon2([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
		Polygon2([Vector2.from_polar(2, i * tau / 30 + pi / 4) + Vector2(0, -8) for i in range(30)]),
	]
)

test3 = Drawing(
	Vector2(4, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(2.5, 5.), Vector2(4.7, 5), Vector2(4, 6), Vector2(3, 6)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)])
	]
)

test4 = Drawing(
	Vector2(4, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(-1, 1), Vector2(1, 4), Vector2(3, 0), Vector2(-1, 0)]),
	]
)

test5 = Drawing(
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		Polygon2([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		Polygon2([Vector2.from_polar(2, i * tau / 3) + Vector2(5, -4) for i in range(3)]),
		Polygon2([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
	]
)

test6 = Drawing(
	Vector2(-3, 0),
	Vector2(0, 2),
	[
		Polygon2(regular(3, 1, Vector2(0, 0), pi / 3))
	]
)

test7 = Drawing(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 2)]),
		Polygon2([Vector2(3, 3), Vector2(5, 3), Vector2(4.5, 4), Vector2(3.5, 4)]),
		regular(5, 1.3, Vector2(5, 6), 0.1),
	]
)
"""
test8 = Drawing(
	Vector2(0, -5), Vector2(0, 2),
	[
		Polygon2([
			(0, 0), (-0.5, 1), (-1, 1), (-1.5, 0), (-1.5, -1), 
			(-0.25, -3), (0.25, -3), (1.5, -1), (1.5, 0), (1, 1), (0.5, 1),
		]),
	]

)

test9 = Drawing(
	Vector2(-3, -0.5), Vector2(3, -0.5),
	[
		Polygon2([
			(0, 0), (-0.5, 1), (-1, 1), (-1.5, 0), (-1.5, -1), 
			(-0.25, -3), (0.25, -3), (1.5, -1), (1.5, 0), (1, 1), (0.5, 1),
		]),
	]

)
"""

#test1.draw()
#test2.draw()
#test3.draw([1])
#test4.draw()
#test5.draw()
#test6.draw()
test7.draw()
