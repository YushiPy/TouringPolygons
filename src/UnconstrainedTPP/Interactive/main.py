
from math import ceil, floor, log
import pygame as pg

import sys
import os

os.chdir(os.path.dirname(__file__))
sys.path.append("..")

from u_tpp import tpp_solve

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

FRICTION = 0.9

FONT = pg.font.SysFont("arial", 16)

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

		self.start = Point(pg.Vector2(0, 0), 10, (105, 0, 0))
		self.target = Point(pg.Vector2(3, 3), 10, (0, 105, 0))

		self.points = [self.start, self.target]
		self.focused_point = None

		self.polygons = []

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

		for event in self.events:
			if event.type == pg.KEYDOWN:
				if event.key == pg.K_PLUS or event.key == pg.K_EQUALS:
					self.change_zoom(self.units_to_pixels * 1.1)
					self.new_zoom = self.units_to_pixels
				elif event.key == pg.K_MINUS or event.key == pg.K_UNDERSCORE:
					self.change_zoom(self.units_to_pixels / 1.1)
					self.new_zoom = self.units_to_pixels

			if event.type == pg.MOUSEWHEEL:
				self.new_zoom = self.units_to_pixels * 1.15 ** event.y

			if event.type == pg.MOUSEBUTTONUP:
				if event.button == 1: # Left click
					self.focused_point = None

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

		pg.draw.line(self.screen, color, start_screen, end_screen, width)

	def draw_grid(self) -> None:

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

	def draw(self) -> None:

		self.update_screen_size()

		self.screen.fill(BACKGROUND_COLOR)
		
		self.draw_grid()

		for point in self.points:
			self.draw_point(point)
		
		for polygon in self.polygons:
			self.draw_polygon(polygon)

		path = list(map(pg.Vector2, tpp_solve((self.start.position), (self.target.position), [[v.position for v in poly.vertices] for poly in self.polygons], simplify=True)))

		pg.draw.lines(self.screen, (255, 0, 255), False, [self.to_screen_pos(pg.Vector2(p.x, p.y)) for p in path], 3)

		self.draw_table()

		

		pass

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

		self.points.append(Point(pg.Vector2(2, 1), 13, POINTS_COLORS[0]))
		self.points.append(Point(pg.Vector2(1, 1), 13, POINTS_COLORS[1]))
		self.points.append(Point(pg.Vector2(2, 2), 13, POINTS_COLORS[2]))
		
		self.polygons.append(Polygon([self.points[-3], self.points[-2], self.points[-1]], (100, 150, 250)))

		self.points.append(Point(pg.Vector2(4, 1), 13, POINTS_COLORS[3]))
		self.points.append(Point(pg.Vector2(5, 1), 13, POINTS_COLORS[3]))
		self.points.append(Point(pg.Vector2(4, 2), 13, POINTS_COLORS[3]))
		self.points.append(Point(pg.Vector2(5, 2), 13, POINTS_COLORS[3]))

		self.polygons.append(Polygon([self.points[-4], self.points[-3], self.points[-1], self.points[-2]], POINTS_COLORS[3]))

		while not self.run_frame():
			self.delta_time = self.clock.tick(FRAME_RATE) / 1000.0

if __name__ == "__main__":
	Main().run()
