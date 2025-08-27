
from itertools import count
from math import ceil

import os
import random

from pygame import Vector2
import pygame as pg

from game_template import Game
from text import PlainText, Text

def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
	"""
	Convert HSV color to RGB color.

	:param h: Hue value (0.0 to 1.0)
	:param s: Saturation value (0.0 to 1.0)
	:param v: Value (brightness) value (0.0 to 1.0)

	:return: Tuple of (R, G, B) values (0 to 255)
	"""

	c = v * s
	x = c * (1 - abs((h * 6) % 2 - 1))
	m = v - c

	match int(h * 6):
		case 0: r, g, b = c, x, 0
		case 1: r, g, b = x, c, 0
		case 2: r, g, b = 0, c, x
		case 3: r, g, b = 0, x, c
		case 4: r, g, b = x, 0, c
		case 5: r, g, b = c, 0, x
		case _: r, g, b = 0, 0, 0
	
	r = round((r + m) * 255)
	g = round((g + m) * 255)
	b = round((b + m) * 255)

	return r, g, b

def random_color(seed: int | None = None, offset: int = random.randrange(0, 2 ** 63)) -> tuple[int, int, int]:

	if seed is not None:
		seed += offset

	rand = random.Random(seed).random

	h = rand()
	s = 0.8 + rand() / 5
	v = 0.8 + rand() / 5

	return hsv_to_rgb(h, s, v)

def dashed_line(surface: pg.Surface, color: pg.Color, start: Vector2, end: Vector2, dash_length: int = 5) -> None:

	dash_count = ceil(start.distance_to(end) / dash_length)
	dash_count += not dash_count % 2

	dash_vector = (end - start) / dash_count

	for i in range(0, dash_count, 2):
		pg.draw.line(surface, color, start + i * dash_vector, start + (i + 1) * dash_vector, 1)

class Gameplay(Game):

	def __init__(self, __fps: int | float = 125) -> None:
		super().__init__(__fps)

		self.polygons: list[list[Vector2]] = [[]]
		self.current_polygon: int = 0

		self.last_mouse_keys = pg.mouse.get_pressed()
		self.held_vertex: tuple[int, int] | None = None

		self.snap_to_grid = False
		self.grid_size = 50

		self.last_loaded: str | None = None

	def load(self, filename: str) -> None:

		with open(filename, "r") as file:
			data = file.read()
		
		center = Vector2(self.surface.get_size()) / 2

		result: list[list[Vector2]] = []

		for polygon in eval(data):
			result.append([])
			for p in polygon:
				v = Vector2(p) * self.grid_size
				v.y *= -1
				v += center
				result[-1].append(v)

		self.polygons = result

	def export(self) -> None:

		scale = 1 / self.grid_size

		string = "[\n"

		for polygon in self.polygons:

			string += "\t["

			for vertex in polygon:
				vertex = round((vertex - Vector2(self.surface.get_size()) / 2) * scale, 3)
				vertex.y *= -1
				string += f"({vertex.x}, {vertex.y}), "

			string = string[:-2] + "],\n"

		string += "]"

		if not os.path.exists("PolygonsOut"):
			os.makedirs("PolygonsOut")
		
		index = next(i for i in count(1) if not os.path.exists(f"PolygonsOut/polygons{i}.txt"))

		with open(f"PolygonsOut/polygons{index}.txt", "w") as file:
			file.write(string)

	@property
	def shifting(self) -> bool:
		return self.held_keys[pg.K_LSHIFT] or self.held_keys[pg.K_RSHIFT]

	def point_indices(self) -> list[tuple[int, int]]:
		"""Get a list of indices for all points in all polygons."""
		return [(i, j) for i, polygon in enumerate(self.polygons) for j in range(len(polygon))]

	def mouse_released(self) -> list[bool]:
		return [a and not b for a, b in zip(pg.mouse.get_pressed(), self.last_mouse_keys)]

	def remove_point(self, mouse_pos: tuple[int, int]) -> None:

		indeces = min(self.point_indices(), key=lambda v: self.polygons[v[0]][v[1]].distance_to(mouse_pos), default=None)

		if indeces is None:
			return
		
		vertex = self.polygons[indeces[0]][indeces[1]]

		if vertex.distance_to(mouse_pos) > 30:
			return
		
		self.polygons[indeces[0]].pop(indeces[1])

	def snap_point(self, point: Vector2 | tuple[int, int]) -> Vector2:

		point = Vector2(point)

		if not self.snap_to_grid:
			return point

		center = Vector2(self.surface.get_size()) / 2
		gs = round(self.grid_size)

		return round((point - center) / gs) * gs + center # type: ignore

	def move_point(self, mouse_pos: tuple[int, int]) -> None:

		if self.held_vertex is None:

			def key(x: tuple[int, int]) -> float:
				return self.polygons[x[0]][x[1]].distance_to(mouse_pos) / (1 + (x[0] == self.current_polygon))
			
			self.held_vertex = min(self.point_indices(), key=key, default=None)

			if self.held_vertex is not None and self.polygons[self.held_vertex[0]][self.held_vertex[1]].distance_to(mouse_pos) > 30:
				self.held_vertex = None
				return
			

		if self.held_vertex is None:
			return

		self.polygons[self.held_vertex[0]][self.held_vertex[1]] = self.snap_point(mouse_pos)

	def draw_grid(self, surface: pg.Surface, grid_size: int) -> None:

		width, height = surface.get_size()
		color = (50, 50, 50)

		center = Vector2(width, height) / 2

		xstart = round(center.x % grid_size)
		ystart = round(center.y % grid_size)

		for x in range(xstart, width, grid_size):
			pg.draw.line(surface, color, (x, 0), (x, height), 1)

		for y in range(ystart, height, grid_size):
			pg.draw.line(surface, color, (0, y), (width, y), 1)

	def draw(self, surface: pg.Surface) -> None:

		surface.fill("#001418")

		if self.snap_to_grid:
			self.draw_grid(surface, round(self.grid_size))

		for i, polygon in enumerate(self.polygons):

			color = pg.color.Color(random_color(i))
			light_color = color.lerp("white", 0.3)

			if len(polygon) >= 3:
				pg.draw.aalines(surface, color, False, polygon)
				dashed_line(surface, light_color.lerp("white", 0.5), polygon[0], polygon[-1], 10)

			for j, vertex in enumerate(polygon):
				v_color = light_color.lerp("white", 0.5 * j / len(polygon))
				size = 5 + 5 * (j / len(polygon))
				pg.draw.circle(surface, v_color, vertex, size)

			if len(polygon) < 3:
				continue

			center_of_mass = sum(polygon, Vector2()) / len(polygon)
			PlainText(f"{i + 1}", center_of_mass, light_color).draw(surface)

		color = pg.color.Color(random_color(self.current_polygon)).lerp("white", 0.3)

		string = f"""\\color({list(color)})Current Polygon: {self.current_polygon + 1}\\color(white)
Grid Snapping: {'On' if self.snap_to_grid else 'Off'}
Grid Size: {round(self.grid_size)}

Controls:
- Left Click: Add Point
- Right Click: Remove Point
- Hold Shift + Left Click: Move Point

- Enter: New Polygon
- Tab: Next Polygon
- Shift + Tab: Previous Polygon
- Left/Right Arrow: Change Polygon
- Backspace: Remove Last Point / Polygon

- S: Toggle Grid Snapping
- Mouse Wheel: Change Grid Size
- L: Load Polygons
- E: Export Polygons"""

		Text(string, (50, 50), 20, Text.Alignment.LEFT).draw(surface)

	def fixed_update(self, down_keys: set[int], up_keys: set[int], events: set[int]) -> None | bool:

		mouse_held = pg.mouse.get_pressed()
		mouse_released = self.mouse_released()
		mouse_pos = pg.mouse.get_pos()

		if mouse_released[0] and not self.shifting:
			self.polygons[self.current_polygon].append(self.snap_point(mouse_pos))

		if mouse_held[0] and self.shifting:
			self.move_point(mouse_pos)
		else:
			self.held_vertex = None

		if mouse_released[2]:
			self.remove_point(mouse_pos)

		if pg.K_l in up_keys:

			options = os.listdir("PolygonsOut")
			options = [f for f in options if f.startswith("polygons") and f.endswith(".txt")]

			if self.last_loaded in options:
				filename = options[(options.index(self.last_loaded) + 1) % len(options)]
			else:
				filename = options[0] if options else None

			if filename:
				self.load(os.path.join("PolygonsOut", filename))
				self.last_loaded = filename

		if pg.K_e in up_keys:
			self.export()

		if pg.K_s in up_keys:
			self.snap_to_grid = not self.snap_to_grid

		for event in self.all_events:
			if event.type == pg.MOUSEWHEEL:

				diff: float = (event.y * self.grid_size / 30)

				self.grid_size = min(300, max(10, self.grid_size + diff))

		if pg.K_RETURN in up_keys:
			self.polygons.append([])
			self.current_polygon = len(self.polygons) - 1

		if pg.K_TAB in up_keys:

			if self.shifting:
				self.current_polygon = (self.current_polygon - 1) % len(self.polygons)
			elif self.current_polygon == len(self.polygons) - 1:
				self.polygons.append([])
				self.current_polygon += 1
			else:
				self.current_polygon = (self.current_polygon + 1) % len(self.polygons)

		self.current_polygon = (self.current_polygon + (pg.K_RIGHT in down_keys) - (pg.K_LEFT in down_keys)) % len(self.polygons)

		if pg.K_BACKSPACE in down_keys:
			
			if self.polygons[self.current_polygon]:
				self.polygons[self.current_polygon].pop()
			else:
				self.polygons.pop(self.current_polygon)

		self.last_mouse_keys = mouse_held


print(Gameplay().run().get_info())
