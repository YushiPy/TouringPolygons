
from math import ceil
from pygame import Vector2
import pygame as pg

from game_template import Game
from text import Text

COLORS = [
	"green", "blue", "yellow", "cyan", "magenta", "orange", "purple",
	"pink", "brown", "gray", "black", "white", "lightblue", "lightgreen", "lightyellow"
]

def dashed_line(surface: pg.Surface, color: pg.Color, start: Vector2, end: Vector2, dash_length: int = 5) -> None:

	dash_count = ceil(start.distance_to(end) / dash_length)
	dash_count += dash_count % 2

	dash_vector = (end - start) / dash_count

	for i in range(0, dash_count, 2):
		pg.draw.line(surface, color, start + i * dash_vector, start + (i + 1) * dash_vector, 1)

class Gameplay(Game):

	def __init__(self, __fps: int | float = 125) -> None:
		super().__init__(__fps)

		self.polygons: list[list[Vector2]] = [[]]
		self.current_polygon: int = 0

		self.last_mouse_keys = pg.mouse.get_pressed()

	def mouse_released(self) -> list[bool]:
		return [a and not b for a, b in zip(pg.mouse.get_pressed(), self.last_mouse_keys)]

	def remove_point(self, mouse_pos: tuple[int, int]) -> None:

		polygon = self.polygons[self.current_polygon]

		if not polygon:
			return
		
		vertex = min(polygon, key=lambda v: v.distance_to(mouse_pos))

		if vertex.distance_to(mouse_pos) > 30:
			return
		
		polygon.remove(vertex)

	def draw(self, surface: pg.Surface) -> None:

		surface.fill("#001418")

		for i, polygon in enumerate(self.polygons):

			color = pg.color.Color(COLORS[i % len(COLORS)])
			light_color = color.lerp("white", 0.3)

			if len(polygon) >= 3:
				pg.draw.aalines(surface, color, False, polygon)
				dashed_line(surface, light_color.lerp("white", 0.5), polygon[0], polygon[-1], 10)

			for vertex in polygon:
				pg.draw.circle(surface, light_color, (int(vertex.x), int(vertex.y)), 5)

			if len(polygon) < 3:
				continue

			center_of_mass = sum(polygon, Vector2()) / len(polygon)
			Text(f"{i + 1}", center_of_mass, light_color).draw(surface)

		color = pg.color.Color(COLORS[self.current_polygon % len(COLORS)]).lerp("white", 0.3)
		Text(f"Selected Polygon: {self.current_polygon + 1}", (200, 100), color).draw(surface)

	def fixed_update(self, down_keys: set[int], up_keys: set[int], held_keys: set[int], events: set[int]) -> None | bool:

		mouse_released = self.mouse_released()
		mouse_pos = pg.mouse.get_pos()

		if mouse_released[0]:
			self.polygons[self.current_polygon].append(Vector2(mouse_pos))

		if mouse_released[2]:
			self.remove_point(mouse_pos)

		if pg.K_RETURN in up_keys:
			self.polygons.append([])
			self.current_polygon = len(self.polygons) - 1
		
		if pg.K_RIGHT in down_keys:
			self.current_polygon = (self.current_polygon + 1) % len(self.polygons)
		if pg.K_LEFT in down_keys:
			self.current_polygon = (self.current_polygon - 1) % len(self.polygons)

		if pg.K_BACKSPACE in down_keys:
			
			if self.polygons[self.current_polygon]:
				self.polygons[self.current_polygon].pop()

		self.last_mouse_keys = pg.mouse.get_pressed()


print(Gameplay().run().get_info())

