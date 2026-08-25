"""Shared helpers for Touring Polygons benchmark case files."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class EncodedCase:
	data: bytes
	digest: str
	case_index: int
	polygons: tuple[Polygon, ...]

	@property
	def polygon_count(self) -> int:
		return len(self.polygons)

	@property
	def vertex_count(self) -> int:
		return sum(len(polygon) for polygon in self.polygons)


def read_u64(data: bytes, offset: int, path: Path) -> tuple[int, int]:
	if offset + 8 > len(data):
		raise ValueError(f"Truncated size value at byte {offset} in {path}")
	return struct.unpack_from("<Q", data, offset)[0], offset + 8


def skip_bytes(data: bytes, offset: int, count: int, path: Path) -> int:
	end = offset + count
	if end > len(data):
		raise ValueError(f"Truncated case payload at byte {offset} in {path}")
	return end


def read_encoded_cases(path: Path) -> list[EncodedCase]:
	data = path.read_bytes()
	offset = 0
	cases: list[EncodedCase] = []

	while offset < len(data):
		start = offset
		offset = skip_bytes(data, offset, 32, path)  # start and target vectors
		polygon_count, offset = read_u64(data, offset, path)
		polygons: list[Polygon] = []

		for _ in range(polygon_count):
			vertex_count, offset = read_u64(data, offset, path)
			vertices = tuple(
				struct.unpack_from("<dd", data, offset + 16 * vertex_index)
				for vertex_index in range(vertex_count)
			)
			polygons.append(vertices)
			offset = skip_bytes(data, offset, 16 * vertex_count, path)

		solution_count, offset = read_u64(data, offset, path)
		offset = skip_bytes(data, offset, 16 * solution_count, path)
		encoded = data[start:offset]
		cases.append(EncodedCase(
			data=encoded,
			digest=hashlib.sha256(encoded).hexdigest(),
			case_index=len(cases),
			polygons=tuple(polygons),
		))

	return cases


def write_encoded_cases(path: Path, cases: Sequence[EncodedCase]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as file:
		for case in cases:
			file.write(case.data)


def orientation(a: Point, b: Point, c: Point) -> float:
	return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(points: Sequence[Point]) -> Polygon:
	unique = sorted(set(points))
	if len(unique) <= 1:
		return tuple(unique)

	def half_hull(sorted_points: Sequence[Point]) -> list[Point]:
		hull: list[Point] = []
		for point in sorted_points:
			while len(hull) > 1 and orientation(hull[-2], hull[-1], point) <= 0.0:
				hull.pop()
			hull.append(point)
		return hull[:-1]

	return tuple(half_hull(unique) + half_hull(list(reversed(unique))))


def point_on_segment(point: Point, a: Point, b: Point) -> bool:
	epsilon = 1e-12
	return (
		abs(orientation(a, b, point)) <= epsilon
		and min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
		and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
	)


def segments_intersect_or_touch(a: Point, b: Point, c: Point, d: Point) -> bool:
	o1 = orientation(a, b, c)
	o2 = orientation(a, b, d)
	o3 = orientation(c, d, a)
	o4 = orientation(c, d, b)
	if o1 * o2 < 0.0 and o3 * o4 < 0.0:
		return True
	return (
		point_on_segment(c, a, b)
		or point_on_segment(d, a, b)
		or point_on_segment(a, c, d)
		or point_on_segment(b, c, d)
	)


def point_in_polygon_or_on_boundary(point: Point, polygon: Sequence[Point]) -> bool:
	inside = False
	px, py = point
	for index, a in enumerate(polygon):
		b = polygon[(index + 1) % len(polygon)]
		if point_on_segment(point, a, b):
			return True
		if (a[1] > py) != (b[1] > py):
			x_crossing = (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]
			if px <= x_crossing:
				inside = not inside
	return inside


def polygons_intersect_or_touch(first: Sequence[Point], second: Sequence[Point]) -> bool:
	first_bounds = (
		min(point[0] for point in first), min(point[1] for point in first),
		max(point[0] for point in first), max(point[1] for point in first),
	)
	second_bounds = (
		min(point[0] for point in second), min(point[1] for point in second),
		max(point[0] for point in second), max(point[1] for point in second),
	)
	if (
		first_bounds[2] < second_bounds[0]
		or second_bounds[2] < first_bounds[0]
		or first_bounds[3] < second_bounds[1]
		or second_bounds[3] < first_bounds[1]
	):
		return False

	for first_index, a in enumerate(first):
		b = first[(first_index + 1) % len(first)]
		for second_index, c in enumerate(second):
			d = second[(second_index + 1) % len(second)]
			if segments_intersect_or_touch(a, b, c, d):
				return True
	return point_in_polygon_or_on_boundary(first[0], second) or point_in_polygon_or_on_boundary(second[0], first)


def case_has_intersections(case: EncodedCase) -> bool:
	return any(
		polygons_intersect_or_touch(case.polygons[first], case.polygons[second])
		for first in range(len(case.polygons))
		for second in range(first + 1, len(case.polygons))
	)


def case_has_intersecting_hulls(case: EncodedCase) -> bool:
	hulls = [convex_hull(polygon) for polygon in case.polygons]
	return any(
		polygons_intersect_or_touch(hulls[first], hulls[second])
		for first in range(len(hulls))
		for second in range(first + 1, len(hulls))
	)
