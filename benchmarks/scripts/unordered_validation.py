"""Independent validation shared by unordered-TPP benchmark adapters."""
from __future__ import annotations

import math
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
