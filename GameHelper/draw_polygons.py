
from pygame import Vector2
import pygame as pg

from game_template import Game
from text import Text

COLORS = [
	"green", "blue", "yellow", "cyan", "magenta", "orange", "purple",
	"pink", "brown", "gray", "black", "white", "lightblue", "lightgreen", "lightyellow"
]

class Gameplay(Game):

	def __init__(self, __fps: int | float = 125) -> None:
		super().__init__(__fps)

		self.polygons: list[list[Vector2]] = [[]]
		self.current_polygon: int = 0

		self.last_mouse_keys = pg.mouse.get_pressed()

	def draw(self, surface: pg.Surface) -> None:

		surface.fill("#001418")

		for i, polygon in enumerate(self.polygons):

			color = pg.color.Color(COLORS[i % len(COLORS)])
			light_color = color.lerp("white", 0.3)

			if len(polygon) >= 3:
				pg.draw.aalines(surface, color, True, polygon)

			for vertex in polygon:
				pg.draw.circle(surface, light_color, (int(vertex.x), int(vertex.y)), 5)

			if len(polygon) < 3:
				continue

			center_of_mass = sum(polygon, Vector2()) / len(polygon)
			Text(f"{i + 1}", center_of_mass, light_color).draw(surface)

		color = pg.color.Color(COLORS[self.current_polygon % len(COLORS)]).lerp("white", 0.3)
		Text(f"Selected Polygon: {self.current_polygon + 1}", (200, 100), color).draw(surface)

	def fixed_update(self, down_keys: set[int], up_keys: set[int], held_keys: set[int], events: set[int]) -> None | bool:

		mouse_keys = pg.mouse.get_pressed()

		if mouse_keys[0] and not self.last_mouse_keys[0]:
			mouse_pos = pg.mouse.get_pos()
			self.polygons[self.current_polygon].append(Vector2(mouse_pos))

		if mouse_keys[2] and not self.last_mouse_keys[2]:
			exit()

		if pg.K_RETURN in up_keys:
			self.polygons.append([])
			self.current_polygon = len(self.polygons) - 1
		
		if pg.K_RIGHT in down_keys:
			self.current_polygon = (self.current_polygon + 1) % len(self.polygons)
		elif pg.K_LEFT in down_keys:
			self.current_polygon = (self.current_polygon - 1) % len(self.polygons)

		self.last_mouse_keys = mouse_keys

print(Gameplay().run().get_info())

