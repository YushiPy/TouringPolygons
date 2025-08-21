
from pygame import Vector2
import pygame as pg

from game_template import Game

class Gameplay(Game):

	def __init__(self, __fps: int | float = 125) -> None:
		super().__init__(__fps)

		self.polygons: list[list[Vector2]] = [[]]
		self.current_polygon: int = 0

	def draw(self, surface: pg.Surface) -> None:

		for polygon in self.polygons:

			for vertex in polygon:
				pg.draw.circle(surface, (0, 255, 0), (int(vertex.x), int(vertex.y)), 5)

			if len(polygon) >= 3:
				pg.draw.aalines(surface, pg.color.Color("red"), True, polygon)

	def fixed_update(self, down_keys: set[int], up_keys: set[int], held_keys: set[int], events: set[int]) -> None | bool:

		if pg.MOUSEBUTTONUP in events:
			mouse_pos = pg.mouse.get_pos()
			self.polygons[-1].append(Vector2(mouse_pos))

print(Gameplay().run().get_info())

