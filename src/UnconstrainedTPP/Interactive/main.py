
from collections.abc import Callable, Sequence
import enum
from math import ceil, cos, floor, log, pi, sin
from typing import Any, Self
import pygame as pg

import sys
import os

os.chdir(os.path.dirname(__file__))
sys.path.append("..")

from polygon2 import Polygon2
from vector2 import Vector2
from u_tpp import tpp_solve
from u_tpp_fast_locate import Solution

pg.init()

CAPTION = "Unconstrained TPP Interactive"

FRAME_RATE = 120

BACKGROUND_COLOR = 255, 255, 255
AXIS_COLOR = 25, 25, 25
GRID_COLOR = 153, 153, 153
SUBGRID_COLOR = 224, 224, 224

POINTS_COLORS = [
	(184, 77, 70),
	(63, 111, 174),
	(74, 130, 73),
	(234, 133, 57),
	(92, 67, 160),
	(0, 0, 0),
]

class ButtonColors(enum.Enum):
	BACKGROUND_IDLE = (220, 220, 220)
	BACKGROUND_HOVER = (200, 200, 200)
	BACKGROUND_ACTIVE = (180, 180, 180)
	BORDER = (120, 120, 120)
	TEXT = (0, 0, 0)

FRICTION = 0.9

FONT = pg.font.SysFont("arial", 16)
EXPORT_FILE = "export.txt"
LOAD_FILE = "load.txt"


def create_display() -> pg.Surface:

	display = pg.display.set_mode((0, 0), pg.NOFRAME | pg.RESIZABLE)
	pg.display.set_caption(CAPTION)

	return display

class FRect:

	left: float
	top: float
	right: float
	bottom: float

	def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom

	@property
	def width(self) -> float:
		return self.right - self.left

	@property
	def height(self) -> float:
		return self.bottom - self.top

class Point:

	position: pg.Vector2
	radius: int
	color: tuple[int, int, int]
	focus_time: float

	max_focus_time: float = 0.1 # seconds to reach full focus

	def __init__(self, position: pg.Vector2, radius: int, color: tuple[int, int, int] = (255, 0, 0)) -> None:
		self.position = position
		self.radius = radius
		self.color = color
		self.light_color = tuple(min(255, c * 2) for c in color) # type: ignore
		self.focus_time = 0.0
	
	def focus_radius(self) -> int:

		factor = min(self.focus_time / self.max_focus_time, 1.0)
		base = 0.4

		return int(self.radius * (factor * (1 - base) + base))

class Polygon:

	vertices: list[Point]
	color: tuple[int, int, int]

	def __init__(self, vertices: list[Point], color: tuple[int, int, int] = (0, 0, 0)) -> None:
		self.vertices = vertices
		self.color = color

class Button:

	_text: str
	_action: Callable[[Self], None]

	def __init__(self, text: str, action: Callable[..., None]) -> None:
		self._text = text
		self._action = action

	@property
	def text(self) -> str:
		return self._text

	def call(self, *args: Any, **kwargs: Any) -> None:
		self._action(self, *args, **kwargs)

class ToggleButton(Button):

	is_active: bool
	inactive_text: str
	active_text: str

	def __init__(self, inactive_text: str, active_text: str, action: Callable[..., None], is_active: bool = False) -> None:

		self.is_active = is_active
		self.inactive_text = inactive_text
		self.active_text = active_text

		self._text = ""
		self._action = action
	
	@property
	def text(self) -> str:
		return self.active_text if self.is_active else self.inactive_text

	def call(self, *args: Any, **kwargs: Any) -> None:
		self.is_active = not self.is_active
		super().call(args, kwargs)


def clamp_point(point: pg.Vector2, rect: FRect) -> pg.Vector2:

	x = max(rect.left, min(rect.right, point.x))
	y = max(rect.top, min(rect.bottom, point.y))

	return pg.Vector2(x, y)

class Main:

	screen: pg.Surface
	clock: pg.time.Clock
	delta_time: float

	events: list[pg.event.Event]

	position: pg.Vector2
	velocity: pg.Vector2
	units_to_pixels: float
	new_zoom: float

	render_area: pg.Rect

	points: list[Point]
	focused_point: Point | None

	start: Point
	target: Point

	polygons: list[Polygon]
	buttons: list[Button]

	def __init__(self) -> None:
		
		self.screen = create_display()
		self.update_screen_size()

		self.clock = pg.time.Clock()
		self.delta_time = 0.0

		self.events = []

		self.position = pg.Vector2(0, 0)
		self.velocity = pg.Vector2(0, 0)
		self.units_to_pixels = self.render_area.width / 20 # 20 units fit in screen width, by default

		self.new_zoom = self.units_to_pixels

		self.start = Point(pg.Vector2(), 0, (0, 0, 0))
		self.target = Point(pg.Vector2(), 0, (0, 0, 0))

		self.points = []
		self.focused_point = None

		self.polygons = []
		self.ghost_polygon = None
		self.snapping = False

		self.buttons = []

		def wrap_function(func: Callable[[], None] | None = None) -> Callable[..., None]:

			def wrapper(button: Button, *args: Any, **kwargs: Any) -> None:
				if callable(func):
					func()

			return wrapper

		self.snap_button = ToggleButton("Snap", "Snap", wrap_function(self.toggle_snapping))

		self.buttons.append(self.snap_button)
		self.buttons.append(Button("Triangle", wrap_function(self.make_ghost_polygon)))

		self.show_line_button = ToggleButton("Show Line", "Show Line", wrap_function())
		self.buttons.append(self.show_line_button)

		self.polygon_index = 0

		self.change_button = Button("Polygon 1", lambda x, y = 1, *_, **__: self.set_polygon_index(self.polygon_index + y))
		self.buttons.append(self.change_button)

		self.export_button = Button("Export", wrap_function(self.export))
		self.buttons.append(self.export_button)

		self.buttons.append(Button("Clear Polygons", wrap_function(self.clear_polygons)))

		self.map_button = ToggleButton("Draw Map", "Draw Map", wrap_function(), is_active=True)
		self.buttons.append(self.map_button)

		self.loaded_index = 0
		self.load_button = Button("Load Next", wrap_function(self.load_next))
		self.buttons.append(self.load_button)

	def load_next(self) -> None:

		if not os.path.exists(LOAD_FILE):
			return

		with open(LOAD_FILE, "r") as f:
			lines = f.readlines()
		
		if not lines:
			return

		self.loaded_index = self.loaded_index % len(lines)
		line = lines[self.loaded_index]

		try:
			data = eval(line)
			start, target, polygons = data
			self.load_polygons(start, target, polygons)
		except Exception as e:
			print(f"Failed to load data: {e}")

		self.loaded_index += 1

	def load_polygons(self, start: tuple[float, float], target: tuple[float, float], polygons: list[list[tuple[float, float]]]) -> None:

		self.clear_polygons()

		self.start.position = pg.Vector2(start)
		self.target.position = pg.Vector2(target)

		for i, polygon in enumerate(polygons):
			color = POINTS_COLORS[i % len(POINTS_COLORS)]
			self.place_polygon(polygon, color)

		self.set_polygon_index(0)

		self.points.append(self.start)
		self.points.append(self.target)

	def clear_polygons(self) -> None:

		self.polygons.clear()
		self.points.clear()
		self.set_polygon_index(0)

		self.points.append(self.start)
		self.points.append(self.target)

	def set_polygon_index(self, index: int) -> None:
		self.polygon_index = index % len(self.polygons) if self.polygons else 0
		self.change_button._text = f"Polygon {self.polygon_index + 1}"

	def export(self) -> None:

		with open(EXPORT_FILE, "a") as f:
			start = tuple(self.start.position)
			target = tuple(self.target.position)
			polygons = [[tuple(vertex.position) for vertex in polygon.vertices] for polygon in self.polygons]
			f.write(str((start, target, polygons)) + "\n")

	def get_button_rects(self) -> list[pg.Rect]:

		currentx = self.render_area.left + 20
		result = []

		for button in self.buttons:

			text_rect = pg.Rect(0, 0, *FONT.size(button.text))
			border_rect = text_rect.inflate(20, 10)

			border_rect.left = currentx
			border_rect.centery = self.render_area.top // 2

			result.append(border_rect)
			
			currentx += border_rect.width + 20

		return result

	def make_ghost_polygon(self, sides: int = 3) -> None:

		if self.ghost_polygon is not None:
			self.ghost_polygon = None
			return

		radius = min(self.render_area.width, self.render_area.height) / 6
		self.ghost_polygon = [(radius * cos(2 * pi * i / sides), radius * sin(2 * pi * i / sides)) for i in range(sides)]

	def place_ghost_polygon(self) -> None:

		if self.ghost_polygon is None:
			raise ValueError("No ghost polygon to place.")

		mouse_pos = pg.Vector2(pg.mouse.get_pos())
		color = POINTS_COLORS[len(self.polygons) % len(POINTS_COLORS)]
		coords = [self.from_screen_pos(mouse_pos + pg.Vector2(p)) for p in self.ghost_polygon]
		self.ghost_polygon = None

		self.place_polygon(coords, color) # type: ignore

	def place_polygon(self, points: Sequence[tuple[float, float]], color: tuple[int, int, int]) -> None:

		coords = [pg.Vector2(p) for p in points]
		polygon = Polygon([Point(p, 8, color) for p in coords], color)
		self.polygons.append(polygon)
		self.points.extend(polygon.vertices)
		self.set_polygon_index(len(self.polygons) - 1)

	def toggle_snapping(self) -> None:
		self.snapping = not self.snapping

	def update_screen_size(self) -> None:

		self.screen_rect = self.screen.get_rect()
		self.render_area = self.screen.get_rect()
		self.render_area.x = self.render_area.width // 4
		self.render_area.top = self.render_area.height // 10

	def visible_area(self) -> FRect:
		"""Returns the visible area in world coordinates as (left, top, right, bottom)."""

		width = self.render_area.width / self.units_to_pixels
		height = self.render_area.height / self.units_to_pixels

		left = self.position.x - width / 2
		right = self.position.x + width / 2
		top = self.position.y - height / 2
		bottom = self.position.y + height / 2

		return FRect(left, top, right, bottom)

	def change_zoom(self, new_zoom: float, fixed_point: pg.Vector2 | None = None) -> None:

		if fixed_point is None:
			fixed_point = pg.Vector2(pg.mouse.get_pos())

		factor = new_zoom / self.units_to_pixels
		self.units_to_pixels *= factor
		diff = (fixed_point - self.render_area.center) * (1 - factor) / self.units_to_pixels
		diff.y *= -1
		self.position -= diff

	def to_screen_pos(self, world_pos: pg.Vector2, relative: pg.Vector2 | None = None) -> pg.Vector2:

		if relative is None:
			relative = self.position

		x = (world_pos.x - relative.x) * self.units_to_pixels
		y = (world_pos.y - relative.y) * self.units_to_pixels * -1

		position = pg.Vector2(x, y) + self.render_area.center

		return position

	def from_screen_pos(self, screen_pos: pg.Vector2, relative: pg.Vector2 | None = None) -> pg.Vector2:

		if relative is None:
			relative = self.position

		x = (screen_pos.x - self.render_area.center[0]) / self.units_to_pixels + relative.x
		y = -1 * (screen_pos.y - self.render_area.center[1]) / self.units_to_pixels + relative.y

		return pg.Vector2(x, y)

	def try_focus_point(self) -> None:
		
		mouse_pos = self.from_screen_pos(pg.Vector2(pg.mouse.get_pos()))
		self.focused_point = min(self.points, key=lambda p: (p.position - mouse_pos).length_squared(), default=None)

		if self.focused_point is not None and (self.focused_point.position - mouse_pos).length() * self.units_to_pixels > self.focused_point.radius:
			self.focused_point = None

	def update(self) -> None:

		held_keys = pg.key.get_pressed()

		for event in self.events:
			if event.type == pg.KEYDOWN:
				if event.key == pg.K_PLUS or event.key == pg.K_EQUALS:
					#self.change_zoom(self.units_to_pixels * 1.1)
					self.new_zoom = self.units_to_pixels * 1.1
				elif event.key == pg.K_MINUS or event.key == pg.K_UNDERSCORE:
					#self.change_zoom(self.units_to_pixels / 1.1)
					self.new_zoom = self.units_to_pixels / 1.1
				elif event.key == pg.K_SPACE:

					polygon = self.polygons[self.polygon_index]
					hull = Polygon2(p.position for p in polygon.vertices).convex_hull()

					for p in polygon.vertices:
						self.points.remove(p)

					polygon.vertices = [Point(pg.Vector2(p[0], p[1]), 8, polygon.color) for p in hull]

					for point in polygon.vertices:
						self.points.append(point)
				
				elif event.key in [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9]:

					index = event.key - pg.K_1 if event.key != pg.K_0 else 9

					if index < len(self.buttons):
						button = self.buttons[index]
						button.call()
				elif event.key == pg.K_UP:
					self.change_button.call()
				elif event.key == pg.K_DOWN:
					self.change_button.call(-1)

			if event.type == pg.MOUSEWHEEL:
				self.new_zoom = self.units_to_pixels * 1.15 ** event.y

			if event.type == pg.MOUSEBUTTONUP:
				mouse_pos = pg.Vector2(pg.mouse.get_pos())
				if self.render_area.collidepoint(mouse_pos):
					if event.button == 1: # Left click
						self.focused_point = None
					elif event.button == 3: # Right click

						if self.ghost_polygon is None:
						
							if held_keys[pg.K_LSHIFT]:
								
								self.try_focus_point()
								
								if self.focused_point is None:
									continue
								
								polygon = next((poly for poly in self.polygons if self.focused_point in poly.vertices), None)

								if polygon is not None:
									polygon.vertices.remove(self.focused_point)
									self.points.remove(self.focused_point)
									if not polygon.vertices:
										self.polygons.remove(polygon)
									self.focused_point = None
							elif self.polygons:
								world_pos = self.from_screen_pos(mouse_pos)
								polygon = self.polygons[self.polygon_index]
								new_point = Point(world_pos, 8, polygon.color)
								self.points.append(new_point)
								polygon.vertices.append(new_point)
								self.focused_point = new_point
						
						else:
							self.place_ghost_polygon()
				else:
					
					for rect, button in zip(self.get_button_rects(), self.buttons):
						if rect.collidepoint(mouse_pos):
							button.call()
							break

		diff = self.new_zoom - self.units_to_pixels

		if abs(diff) > 1e-3:
			self.change_zoom(self.units_to_pixels + diff * min(self.delta_time * 10, 1))
		else:
			self.change_zoom(self.new_zoom)

		mouse_buttons = pg.mouse.get_pressed()
		mouse_pos = pg.Vector2(pg.mouse.get_pos())
		mouse_movement = pg.Vector2(pg.mouse.get_rel())

		movement = pg.Vector2(mouse_movement) / self.units_to_pixels
		movement.y *= -1

		if mouse_buttons[0] and self.focused_point is not None:
			bbox = self.visible_area()
			self.focused_point.position = self.from_screen_pos(mouse_pos)
			self.focused_point.position.x = max(bbox.left, min(bbox.right, self.focused_point.position.x))
			self.focused_point.position.y = max(bbox.top, min(bbox.bottom, self.focused_point.position.y))	

			if self.snapping:
				_, subgrid_spacing = self.get_spacing()
				self.focused_point.position.x = round(self.focused_point.position.x / subgrid_spacing) * subgrid_spacing
				self.focused_point.position.y = round(self.focused_point.position.y / subgrid_spacing) * subgrid_spacing

		elif mouse_buttons[0]:
			self.position -= movement
		
		for point in self.points:
			if point == self.focused_point:
				point.focus_time += self.delta_time
			else:
				point.focus_time = 0.0

		if not mouse_buttons[0]:
			self.try_focus_point()

	def draw_line(self, start: pg.Vector2, end: pg.Vector2, color: tuple[int, int, int], width: int = 1) -> None:

		start_screen = self.to_screen_pos(start)
		end_screen = self.to_screen_pos(end)

		if width == 1:
			pg.draw.aaline(self.screen, color, start_screen, end_screen)
		else:
			pg.draw.line(self.screen, color, start_screen, end_screen, width)

	def get_spacing(self) -> tuple[float, float]:

		visible_area = self.visible_area()

		target = visible_area.width / 10
		x = 10 ** floor(log(target, 10))

		if 2 * x >= target:
			grid_spacing = x
			subgrid_spacing = grid_spacing / 5
		elif 5 * x >= target:
			grid_spacing = 2 * x
			subgrid_spacing = grid_spacing / 4
		else:
			grid_spacing = 5 * x
			subgrid_spacing = grid_spacing / 5

		return grid_spacing, subgrid_spacing

	def draw_grid(self) -> None:

		visible_area = self.visible_area()
		grid_spacing, subgrid_spacing = self.get_spacing()

		# Draw subgrid lines
		current_x = ceil(visible_area.left / subgrid_spacing) * subgrid_spacing

		while current_x <= visible_area.right:
			self.draw_line(pg.Vector2(current_x, visible_area.top), pg.Vector2(current_x, visible_area.bottom), SUBGRID_COLOR)
			current_x += subgrid_spacing

		current_y = ceil(visible_area.top / subgrid_spacing) * subgrid_spacing

		while current_y <= visible_area.bottom:
			self.draw_line(pg.Vector2(visible_area.left, current_y), pg.Vector2(visible_area.right, current_y), SUBGRID_COLOR)
			current_y += subgrid_spacing

		# Draw main grid lines
		current_x = ceil(visible_area.left / grid_spacing) * grid_spacing
	
		while current_x <= visible_area.right:
			self.draw_line(pg.Vector2(current_x, visible_area.top), pg.Vector2(current_x, visible_area.bottom), GRID_COLOR, 2)
			current_x += grid_spacing

		current_y = ceil(visible_area.top / grid_spacing) * grid_spacing

		while current_y <= visible_area.bottom:
			self.draw_line(pg.Vector2(visible_area.left, current_y), pg.Vector2(visible_area.right, current_y), GRID_COLOR, 2)
			current_y += grid_spacing

		# Draw x and y axes
		self.draw_line(pg.Vector2(visible_area.left, 0), pg.Vector2(visible_area.right, 0), AXIS_COLOR, 2)
		self.draw_line(pg.Vector2(0, visible_area.top), pg.Vector2(0, visible_area.bottom), AXIS_COLOR, 2)


		# Draw numbers
		current_x = ceil(visible_area.left / grid_spacing) * grid_spacing

		while current_x <= visible_area.right:

			if abs(current_x) > 1e-5:
				
				screen_pos = self.to_screen_pos(pg.Vector2(current_x, 0))

				number = f"{abs(current_x):.6g}"
				width, height = FONT.size(number)

				text_surf = FONT.render(f"{current_x:.6g}", True, "black")
				text_rect = text_surf.get_rect()

				text_rect.center = (screen_pos.x - (text_rect.width - width) / 2, screen_pos.y + height) # type: ignore

				text_rect.y = max(text_rect.y, self.render_area.top + 5) # prevent cutting off, 5 pixels padding
				text_rect.y = min(text_rect.y, self.render_area.height - text_rect.height - 5) # prevent going off screen at bottom

				rect = text_rect.copy()
				center = rect.center

				rect.width *= 1.2 # type: ignore
				rect.height *= 1.2 # type: ignore

				rect.center = center

				self.screen.fill(BACKGROUND_COLOR, rect)
				self.screen.blit(text_surf, text_rect)

			current_x += grid_spacing

		current_y = ceil(visible_area.top / grid_spacing) * grid_spacing

		while current_y <= visible_area.bottom:

			if abs(current_y) > 1e-5:
				
				screen_pos = self.to_screen_pos(pg.Vector2(0, current_y))

				number = f"{abs(current_y):.6g}"
				width, height = FONT.size(number)

				text_surf = FONT.render(f"{current_y:.6g}", True, "black")
				text_rect = text_surf.get_rect()

				text_rect.center = (screen_pos.x - width, screen_pos.y - (text_rect.height - height) / 2) # type: ignore

				text_rect.x = max(text_rect.x, self.render_area.left + 5) # prevent cutting off, 5 pixels padding
				text_rect.x = min(text_rect.x, self.render_area.width - text_rect.width - 5) # prevent going off screen at right

				rect = text_rect.copy()
				center = rect.center

				rect.width *= 1.2 # type: ignore
				rect.height *= 1.2 # type: ignore

				rect.center = center

				self.screen.fill(BACKGROUND_COLOR, rect)
				self.screen.blit(text_surf, text_rect)

			current_y += grid_spacing
		
		# Render 0
		screen_pos = self.to_screen_pos(pg.Vector2(0, 0))
		number = "0"

		text_surf = FONT.render("0", True, "black")
		text_rect = text_surf.get_rect()
		text_rect.center = (screen_pos.x - text_rect.width, screen_pos.y + text_rect.height) # type: ignore
		self.screen.fill(BACKGROUND_COLOR, text_rect)
		self.screen.blit(text_surf, text_rect)

	def draw_point(self, point: Point) -> None:

		screen_pos = self.to_screen_pos(point.position)
		radius = point.radius

		pg.draw.circle(self.screen, point.light_color, screen_pos, radius + 1)
		pg.draw.circle(self.screen, point.color, screen_pos, point.focus_radius())

	def draw_polygon(self, polygon: Polygon) -> None:

		if len(polygon.vertices) < 3:
			return

		screen_points = [self.to_screen_pos(v.position) for v in polygon.vertices]
		surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
		
		pg.draw.polygon(surface, polygon.color + (100,), screen_points)
		pg.draw.polygon(surface, polygon.color + (255,), screen_points, 2)
		
		for point in polygon.vertices:
			self.draw_point(point)

		p2 = Polygon2((v.position for v in polygon.vertices))
		
		if not p2.is_convex():
			centroid_point = sum(screen_points, pg.Vector2(0, 0)) / len(screen_points)
			text = FONT.render("Not Convex", True, (255, 0, 0))
			text_rect = text.get_rect()
			text_rect.center = centroid_point # type: ignore
			surface.blit(text, text_rect)

			hull_screen_points = [self.to_screen_pos(pg.Vector2(*p)) for p in p2.convex_hull()]
			pg.draw.polygon(surface, polygon.color + (100,), hull_screen_points, 3)

		self.screen.blit(surface, (0, 0))

	def draw_table(self) -> None:

		pg.draw.rect(self.screen, BACKGROUND_COLOR, (0, 0, self.render_area.left, self.render_area.height))
		pg.draw.rect(self.screen, GRID_COLOR, (0, 0, self.render_area.left, self.render_area.height), 2)

		font = pg.font.SysFont("cambria", 20)
		rect = pg.Rect(0, 0, self.render_area.left, font.get_height() * 2)

		current_y = self.render_area.top

		for i, point in enumerate(self.points, 1):

			rect.y = current_y
			pg.draw.rect(self.screen, (200, 200, 200), rect, 1)

			x = rect.width // 8
			pg.draw.rect(self.screen, (200, 200, 200), (0, rect.y, x, rect.height))

			number_surf = font.render(f"{i}", True, "black")
			number_rect = number_surf.get_rect()
			number_rect.center = (x // 2, rect.centery)
			self.screen.blit(number_surf, number_rect)

			text_surf = font.render(f"P_{i} = ({point.position.x:.3g}, {point.position.y:.3g})", True, "black")
			text_rect = text_surf.get_rect()
			text_rect.centery = rect.centery
			text_rect.x = x + 10

			self.screen.fill(BACKGROUND_COLOR, text_rect)
			self.screen.blit(text_surf, text_rect)

			current_y += rect.height + 5

	def draw_buttons(self) -> None:

		def draw_button(button: Button, rect: pg.Rect) -> None:

			text_surf = FONT.render(button.text, True, "black")

			text_rect = text_surf.get_rect()
			border_rect = rect

			text_rect.center = border_rect.center

			mouse_pos = pg.Vector2(pg.mouse.get_pos())
			is_held = pg.mouse.get_pressed()[0]
			
			if border_rect.collidepoint(mouse_pos):
				if is_held:
					color = ButtonColors.BACKGROUND_ACTIVE
				else:
					color = ButtonColors.BACKGROUND_HOVER
			else:
				color = ButtonColors.BACKGROUND_IDLE

			pg.draw.rect(self.screen, color.value, border_rect)
			pg.draw.rect(self.screen, ButtonColors.BORDER.value, border_rect, 2)
			self.screen.blit(text_surf, text_rect) # type: ignore

			if isinstance(button, ToggleButton):
				color = (0, 200, 0) if button.is_active else (200, 0, 0)
				indicator_rect = pg.Rect(0, 0, 10, 10)
				indicator_rect.center = border_rect.topright
				pg.draw.circle(self.screen, (0, 0, 0), indicator_rect.center, 7)
				pg.draw.circle(self.screen, color, indicator_rect.center, 5)

		rect = pg.Rect(self.render_area.left, 0, self.render_area.width - self.render_area.left, self.render_area.top)

		pg.draw.rect(self.screen, BACKGROUND_COLOR, rect)
		pg.draw.rect(self.screen, GRID_COLOR, rect, 2)

		for rect, button in zip(self.get_button_rects(), self.buttons):
			draw_button(button, rect)

	def draw_ghost_polygon(self) -> None:

		if self.ghost_polygon is None:
			return

		surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA)

		center = pg.mouse.get_pos()
		center = clamp_point(pg.Vector2(center), FRect(*self.render_area))

		points = [pg.Vector2(p) + center for p in self.ghost_polygon]

		color = POINTS_COLORS[len(self.polygons) % len(POINTS_COLORS)]
		
		pg.draw.polygon(surface, color + (100,), points)
		pg.draw.polygon(surface, color + (255,), points, 2)

		self.screen.blit(surface, (0, 0))

	def draw_solution(self) -> None:

		if not self.polygons:
			return

		if any(len(poly.vertices) < 3 or not Polygon2((v.position for v in poly.vertices)).is_convex() for poly in self.polygons):
			return
		
		start = tuple(self.start.position)
		target = tuple(self.target.position)
		polygons = [Polygon2(p.position for p in poly.vertices) for poly in self.polygons] # type: ignore

		try:
			path = tpp_solve(start, target, polygons) # type: ignore
		except Exception as e:
			print(f"Error solving TPP: {e}")
			print(f"{start}\n{target}\n{polygons}")
			return

		pg.draw.lines(self.screen, (0, 0, 255), False, [self.to_screen_pos(pg.Vector2(p)) for p in path], 3) # type: ignore

		for point in path[1:-1]:
			screen_pos = self.to_screen_pos(pg.Vector2(point)) # type: ignore
			pg.draw.circle(self.screen, (0, 0, 255), screen_pos, 5)

	def draw_last_step_map(self) -> None:

		if not self.map_button.is_active:
			return

		if not self.polygons:
			return

		index = self.polygon_index

		start = tuple(self.start.position)
		target = tuple(self.target.position)
		polygons = [Polygon2(p.position for p in poly.vertices) for poly in self.polygons[:index + 1]]
	
		try:
			solution = Solution(start, target, polygons)
			solution.solve()
		except Exception as e:
			print(f"Error solving TPP: {e}")
			print(f"{start}\n{target}\n{polygons}")
			return


		from vector2 import Vector2
		from math import isclose, inf

		def intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:

			cross = direction1.cross(direction2)

			if abs(cross) < 1e-8:
				return None
			
			sdiff = start2 - start1

			rate1 = sdiff.cross(direction2) / cross
			rate2 = sdiff.cross(direction1) / cross

			return rate1, rate2

		def locate_ray(start: Vector2, direction: Vector2, bbox: tuple[float, float, float, float]) -> Vector2:
			"""
			Locate the ray starting from `start` in the direction of `direction` within the bounding box `bbox`.
			
			:param Vector2 start: The starting point of the ray.
			:param Vector2 direction: The direction vector of the ray.
			:param tuple bbox: The bounding box defined as (minx, miny, maxx, maxy).

			:return: The point where the ray intersects with the bounding box.
			"""

			minx, miny, maxx, maxy = bbox
			dx = maxx - minx
			dy = maxy - miny

			walls = [
				(Vector2(minx, miny), Vector2(dx, 0)),
				(Vector2(maxx, miny), Vector2(0, dy)),
				(Vector2(maxx, maxy), Vector2(-dx, 0)),
				(Vector2(minx, maxy), Vector2(0, -dy))
			]

			for wall_start, wall_dir in walls:

				rates = intersection_rates(start, direction, wall_start, wall_dir)

				if rates is not None and rates[0] >= 0 and 0 <= rates[1] <= 1:
					return start + direction * rates[0]

			# Should be unreachable if the ray is inside the bounding box.
			return Vector2(inf, inf)

		def locate_edge(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2, bbox: tuple[float, float, float, float]) -> list[Vector2]:
			"""
			Locate the edge defined by two directions starting from `start1` and `start2` within the bounding box `bbox`.

			:param Vector2 start1: The starting point of the first edge.
			:param Vector2 direction1: The direction vector of the first edge.
			:param Vector2 start2: The starting point of the second edge.
			:param Vector2 direction2: The direction vector of the second edge.
			:param tuple bbox: The bounding box defined as (minx, miny, maxx, maxy).

			:return: A list of Vector2 points representing the edge's vertices.
			"""

			def get_wall(point: Vector2) -> int:
				"""
				Determine which wall of the bounding box the point is closest to.
				Returns an index corresponding to the wall:
				0: bottom, 1: right, 2: top, 3: left.
				"""

				if isclose(point.y, miny):
					return 0
				elif isclose(point.x, maxx):
					return 1
				elif isclose(point.y, maxy):
					return 2
				elif isclose(point.x, minx):
					return 3

				# Should not happen if the point is within the bounding box
				return -1 

			minx, miny, maxx, maxy = bbox

			p1 = locate_ray(start1, direction1, bbox)
			p2 = locate_ray(start2, direction2, bbox)

			w1 = get_wall(p1)
			w2 = get_wall(p2)

			eps = 0

			corners = [
				Vector2(maxx + eps, miny -eps),
				Vector2(maxx + eps, maxy + eps),
				Vector2(minx - eps, maxy + eps),
				Vector2(minx - eps, miny - eps)
			]
			
			result = [start1, p1]

			while w1 != w2:
				result.append(corners[w1])
				w1 = (w1 + 1) % 4
			
			result.append(p2)
			result.append(start2)
			result.append(start1)

			return result

		def locate_cone(start: Vector2, direction1: Vector2, direction2: Vector2, bbox: tuple[float, float, float, float]) -> list[Vector2]:
			"""
			Locate the cone defined by two directions starting from `start` within the bounding box `bbox`.

			:param Vector2 start: The starting point of the cone.
			:param Vector2 direction1: The first direction vector of the cone.
			:param Vector2 direction2: The second direction vector of the cone.
			:param tuple bbox: The bounding box defined as (minx, miny, maxx, maxy).

			:return: A Polygon2 object representing the cone's vertices.
			"""

			return locate_edge(start, direction1, start, direction2, bbox)

		visible_area = self.visible_area()

		vertices = solution.polygons[index]
		cones = solution.cones[index]
		blocked = solution.blocked[index]

		points = list(vertices) + [pg.Vector2(visible_area.left, visible_area.bottom), pg.Vector2(visible_area.right, visible_area.top)]

		minx = min(point.x for point in points) - 1
		maxx = max(point.x for point in points) + 1
		miny = min(point.y for point in points) - 1
		maxy = max(point.y for point in points) + 1

		bbox = (minx, miny, maxx, maxy)

		surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
		lines_surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA)

		for i, vertex in enumerate(vertices):

			if blocked[i] and blocked[i - 1]:
				ray = cones[i][0]
				point = locate_ray(vertex, ray, bbox)
				screen_points = [self.to_screen_pos(p) for p in [vertex, point]] # type: ignore
				pg.draw.line(lines_surface, (0, 155, 155, 205), screen_points[0], screen_points[1], 3)
				continue

			ray1, ray2 = cones[i]
			points = locate_cone(vertex, ray1, ray2, bbox)

			screen_points = [self.to_screen_pos(p) for p in points] # type: ignore
			
			pg.draw.polygon(surface, (255, 0, 0, 80), screen_points)
			pg.draw.polygon(surface, (255, 0, 0, 255), screen_points, 5)
		
		for i in range(len(vertices)):

			v1 = vertices[i]
			v2 = vertices[(i + 1) % len(vertices)]

			ray1 = cones[i][1]
			ray2 = cones[(i + 1) % len(vertices)][0]

			points = locate_edge(v1, ray1, v2, ray2, bbox)
			screen_points = [self.to_screen_pos(p) for p in points] # type: ignore

			color = (0, 155, 0, 80) if not blocked[i] else (0, 155, 155, 80)
			# cyan: (0, 155, 155, 80)
			
			pg.draw.polygon(surface, color, screen_points)

		self.screen.blit(surface, (0, 0))
		self.screen.blit(lines_surface, (0, 0))

	def draw(self) -> None:

		self.update_screen_size()

		self.screen.fill(BACKGROUND_COLOR)
		
		self.draw_grid()

		self.draw_ghost_polygon()

		if self.show_line_button.is_active:
			if self.polygons and self.polygons[self.polygon_index].vertices:
				mouse_pos = self.from_screen_pos(pg.Vector2(pg.mouse.get_pos()))
				point = self.polygons[self.polygon_index].vertices[-1]
				position = point.position
				color = point.color

				self.draw_line(position, mouse_pos, color, 2)


		for point in self.points:
			self.draw_point(point)
		
		for polygon in self.polygons:
			self.draw_polygon(polygon)

		self.draw_last_step_map()
		self.draw_solution()

		self.draw_table()
		self.draw_buttons()


	def run_frame(self) -> int:

		self.events = pg.event.get()

		for event in self.events:

			if event.type == pg.QUIT:
				return 1

			if event.type == pg.KEYDOWN:
				if event.key == pg.K_ESCAPE:
					return 1
		
		self.update()
		self.draw()

		pg.display.flip()
		
		return 0

	def run(self) -> None:

		self.start = Point(pg.Vector2(-1, 1), 10, (105, 0, 0))
		self.target = Point(pg.Vector2(2, 1), 10, (0, 105, 0))

		self.points = [self.start, self.target]

		self.place_polygon([(2, 3), (2, 2), (-1, 2), (-1, 3)], (100, 100, 100))
		
		while not self.run_frame():
			self.delta_time = self.clock.tick(FRAME_RATE) / 1000.0

if __name__ == "__main__":
	Main().run()
