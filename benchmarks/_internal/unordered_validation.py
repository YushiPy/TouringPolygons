"""Independent validation shared by unordered-TPP benchmark adapters."""
from __future__ import annotations

import math
from fractions import Fraction
from collections.abc import Sequence

Point = tuple[float, float]


def orient_path(start: Point, target: Point, path: Sequence[Sequence[float]]) -> list[list[float]]:
	points = [[float(point[0]), float(point[1])] for point in path]
	if len(points) < 2:
		return points
	forward = math.dist(points[0], start) + math.dist(points[-1], target)
	reverse = math.dist(points[0], target) + math.dist(points[-1], start)
	return list(reversed(points)) if reverse < forward else points


def validate_path(
	start: Point,
	target: Point,
	polygons: Sequence[Sequence[Sequence[float]]],
	path: Sequence[Sequence[float]],
	tolerance: float,
) -> dict[str, float | bool]:
	from shapely.geometry import LineString, Polygon

	points = [[float(point[0]), float(point[1])] for point in path]
	if len(points) < 2:
		return {
			'valid': False,
			'endpoint_valid': False,
			'polygon_valid': False,
			'start_distance': math.inf,
			'target_distance': math.inf,
			'max_polygon_distance': math.inf,
			'recomputed_length': math.inf,
		}
	line = LineString(points)
	start_distance = math.dist(points[0], start)
	target_distance = math.dist(points[-1], target)
	max_polygon_distance = max((line.distance(Polygon(polygon)) for polygon in polygons), default=0.0)
	endpoint_valid = start_distance <= tolerance and target_distance <= tolerance
	polygon_valid = max_polygon_distance <= tolerance
	return {
		'valid': endpoint_valid and polygon_valid,
		'endpoint_valid': endpoint_valid,
		'polygon_valid': polygon_valid,
		'start_distance': start_distance,
		'target_distance': target_distance,
		'max_polygon_distance': max_polygon_distance,
		'recomputed_length': line.length,
	}


def validate_cycle(polygons, path, tolerance: float) -> dict:
	"""Independent closed-tour check; no solver geometry or optional libraries.

	Intersections and containment use exact binary-rational predicates. Distances
	and the explicit reporting/feasibility tolerance are numerical diagnostics.
	"""
	points = [tuple(map(float, p)) for p in path]
	if len(points) < 2 or any(not math.isfinite(x) for p in points for x in p):
		return {'valid': False, 'recomputed_length': None, 'max_polygon_distance': None}
	def cross(a, b, c):
		return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
	def intersects(a, b, c, d):
		if any(max(min(a[j],b[j]),min(c[j],d[j])) > min(max(a[j],b[j]),max(c[j],d[j])) for j in (0,1)):
			return False
		return cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0
	def inside(q, ring):
		parity = False
		for a,b in zip(ring, ring[1:]+ring[:1]):
			if intersects(q,q,a,b):return True
			if (a[1] > q[1]) != (b[1] > q[1]):
				x = a[0]+(q[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
				if q[0] < x:parity = not parity
		return parity
	def point_segment(q,a,b):
		x,y=b[0]-a[0],b[1]-a[1];squared=x*x+y*y
		t=0 if squared == 0 else max(0,min(1,((q[0]-a[0])*x+(q[1]-a[1])*y)/squared))
		return math.hypot(q[0]-a[0]-t*x,q[1]-a[1]-t*y)
	exact = [tuple(map(Fraction,p)) for p in points]
	distances=[]
	for polygon in polygons:
		ring=[tuple(map(Fraction,p)) for p in polygon]
		edges=list(zip(ring,ring[1:]+ring[:1]))
		if inside(exact[0],ring) or any(intersects(a,b,c,d) for a,b in zip(exact,exact[1:]) for c,d in edges):
			distances.append(0.0);continue
		floating=[tuple(map(float,p)) for p in polygon]
		distances.append(min(min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))
			for a,b in zip(points,points[1:]) for c,d in zip(floating,floating[1:]+floating[:1])))
	closure=math.dist(points[0],points[-1]);missing=max(distances,default=0.0)
	return {'valid': closure <= tolerance and missing <= tolerance, 'closure_distance': closure,
		'max_polygon_distance': missing, 'exactly_covers': closure == 0 and missing == 0,
		'recomputed_length': sum(math.dist(a,b) for a,b in zip(points,points[1:])),
		'validator': 'binary-rational intersections/containment, numerical distances'}
