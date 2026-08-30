from __future__ import annotations

import os
import struct
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

from dashboard_models import CaseData, Point


SIZE_STRUCT = struct.Struct("<Q")
POINT_STRUCT = struct.Struct("<dd")
FILE_CACHE_LIMIT = 128
_binary_offset_cache: OrderedDict[Path, tuple[int, int, list[int]]] = OrderedDict()


def file_signature(path: Path) -> tuple[int, int]:
	stat = path.stat()
	return stat.st_mtime_ns, stat.st_size


def trim_binary_cache() -> None:
	while len(_binary_offset_cache) > FILE_CACHE_LIMIT:
		_binary_offset_cache.popitem(last=False)


def read_exact(file: BinaryIO, size: int) -> bytes | None:
	data = file.read(size)
	return data if len(data) == size else None


def read_size(file: BinaryIO) -> int | None:
	data = read_exact(file, SIZE_STRUCT.size)
	return SIZE_STRUCT.unpack(data)[0] if data is not None else None


def read_point(file: BinaryIO) -> Point | None:
	data = read_exact(file, POINT_STRUCT.size)
	return POINT_STRUCT.unpack(data) if data is not None else None


def skip_bytes(file: BinaryIO, byte_count: int, file_size: int) -> bool:
	if file.tell() + byte_count > file_size:
		return False
	file.seek(byte_count, os.SEEK_CUR)
	return True


def skip_binary_case(file: BinaryIO, file_size: int) -> bool:
	if not skip_bytes(file, 2 * POINT_STRUCT.size, file_size):
		return False
	polygon_count = read_size(file)
	if polygon_count is None:
		return False
	for _ in range(polygon_count):
		vertex_count = read_size(file)
		if vertex_count is None or not skip_bytes(file, vertex_count * POINT_STRUCT.size, file_size):
			return False
	return read_size(file) is not None


def binary_case_offsets(path: Path) -> list[int]:
	signature = file_signature(path)
	cached = _binary_offset_cache.get(path)
	if cached and cached[:2] == signature:
		_binary_offset_cache.move_to_end(path)
		return list(cached[2])
	offsets: list[int] = []
	file_size = signature[1]
	with path.open("rb") as file:
		while file.tell() < file_size:
			offset = file.tell()
			if not skip_binary_case(file, file_size):
				break
			offsets.append(offset)
	_binary_offset_cache[path] = (*signature, offsets)
	trim_binary_cache()
	return list(offsets)


def read_binary_case_from_file(file: BinaryIO) -> CaseData | None:
	start = read_point(file)
	target = read_point(file)
	polygon_count = read_size(file)
	if start is None or target is None or polygon_count is None:
		return None
	polygons: list[list[Point]] = []
	for _ in range(polygon_count):
		vertex_count = read_size(file)
		if vertex_count is None:
			return None
		polygon: list[Point] = []
		for _ in range(vertex_count):
			point = read_point(file)
			if point is None:
				return None
			polygon.append(point)
		polygons.append(polygon)
	if read_size(file) is None:
		return None
	return start, target, polygons


def binary_case_count(path: Path) -> int:
	return len(binary_case_offsets(path))


def read_binary_case(path: Path, index: int) -> CaseData | None:
	if index < 0:
		return None
	offsets = binary_case_offsets(path)
	if index >= len(offsets):
		return None
	with path.open("rb") as file:
		file.seek(offsets[index])
		return read_binary_case_from_file(file)


def read_binary_cases(path: Path, limit: int) -> list[CaseData]:
	cases: list[CaseData] = []
	offsets = binary_case_offsets(path)
	with path.open("rb") as file:
		for offset in offsets[:max(0, limit)]:
			file.seek(offset)
			case = read_binary_case_from_file(file)
			if case is None:
				return cases
			cases.append(case)
	return cases


def write_vector(file: BinaryIO, point: Point) -> None:
	file.write(POINT_STRUCT.pack(point[0], point[1]))


def write_size(file: BinaryIO, value: int) -> None:
	file.write(SIZE_STRUCT.pack(value))


def write_binary_cases(path: Path, cases: list[CaseData]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as file:
		for start, target, polygons in cases:
			write_vector(file, start)
			write_vector(file, target)
			write_size(file, len(polygons))
			for polygon in polygons:
				write_size(file, len(polygon))
				for point in polygon:
					write_vector(file, point)
			write_size(file, 0)
