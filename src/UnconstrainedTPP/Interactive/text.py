
import enum
import re
from collections.abc import Callable, Iterable
from typing import Hashable, Any

from pygame.font import Font
import pygame as pg

pg.font.init()

type FontInfo = tuple[str, int, bool, bool]
type FontInput = tuple[str, int, bool, bool] | tuple[str, int, bool] | tuple[str, int] | tuple[str] | str

type ColorValue = tuple[int, int, int] | tuple[int, int, int, int] | str | pg.Color

default_font = 'ヒラキノ角コシックw0', 30, False, False

writable_keys: set[int | str] = {
	32, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
	57, 59, 61, 91, 92, 93, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 
	107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 
	" ", "'", ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
	';', '=', '[', '\\', ']', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
	'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
}

shift_map = {
	' ' : ' ', '`' : '~', '1' : '!', '2' : '@', '3' : '#', '4' : '$', '5' : '%', 
	'6' : '^', '7' : '&', '8' : '*', '9' : '(', '0' : ')', '-' : '_', '=' : '+', 
	'[' : '{', ']' : '}', '\\' : '|', ';' : ':', ',' : '<', '.' : '>', '/' : '?', 
	"'" : '"','a' : 'A', 'b' : 'B', 'c' : 'C', 'd' : 'D', 'e' : 'E', 'f' : 'F', 
	'g' : 'G', 'h' : 'H', 'i' : 'I', 'j' : 'J', 'k' : 'K', 'l' : 'L', 'm' : 'M', 
	'n' : 'N', 'o' : 'O', 'p' : 'P', 'q' : 'Q', 'r' : 'R', 's' : 'S', 't' : 'T', 
	'u' : 'U', 'v' : 'V', 'w' : 'W', 'x' : 'X', 'y' : 'Y', 'z' : 'Z'
}

COLOR_PATTERN: re.Pattern[str] = re.compile(r"\\color\((.*?)\)")
COLOR_PATTERN2: re.Pattern[str] = re.compile(r"\\color\(.*?\)")

def cache[T, **P](function: Callable[P, T]) -> Callable[P, T]:
	"""Decorator to cache the result of a function call."""

	previous: dict[tuple[Any, ...], T] = {}

	def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:

		key = (args, tuple(sorted(kwargs.items())))

		if key not in previous:
			previous[key] = function(*args, **kwargs)
	
		return previous[key]

	return wrapper

class Text:
	"""
	Text class that supports multiple colors, lines and alignments.
	Use \\color(color) to change the color of the text.
	Use \\n to create a new line.
	"""

	class Alignment(enum.Enum):
		LEFT = enum.auto()
		RIGHT = enum.auto()
		CENTER = enum.auto()

	value: Any
	string: str

	font: Font

	position: tuple[int, int]
	alignment: Alignment
	vertical_spacing: int

	surfaces: list[pg.Surface]
	rects: list[pg.Rect]

	def __init__(
		self, value: Any, position: Iterable[float], 
		size: int = 30, 
		alignnment: Alignment = Alignment.LEFT,
		vertical_spacing: int = 5
		) -> None:

		self.value = value
		self.string = str(value)

		self.font = get_font(default_font[0], size, default_font[2], default_font[3])

		self.position = tuple(map(int, position)) # type: ignore

		if len(self.position) != 2:
			raise ValueError("Position must be a tuple of (x, y)")

		self.alignment = alignnment
		self.vertical_spacing = vertical_spacing

		self._make_rects()

	def _parse_colors(self) -> list[pg.Color]:

		listed_colors = re.findall(COLOR_PATTERN, self.string)
		colors: list[pg.Color] = []

		for c in listed_colors:

			try: colors.append(pg.Color(c)); continue
			except ValueError: pass

			try: colors.append(pg.Color(eval(c))); continue
			except ValueError: pass

			try: colors.append(pg.Color(eval("(" + c + ")"))); continue
			except ValueError: pass

			raise ValueError(f"Invalid color value: {c}")

		return colors
		
	def _parse_text(self) -> list[list[tuple[ColorValue, str]]]:

		colors = [pg.Color("white")] + self._parse_colors()
		parts = [p for p in re.split(COLOR_PATTERN2, self.string)]
		
		result: list[list[tuple[ColorValue, str]]] = [[]]

		for color, part in zip(colors, parts):

			for line in part.split("\n"):
				result[-1].append((color, line))
				result.append([])
			
			result.pop()
		
		return result

	def _make_rects(self) -> None:

		parsed = self._parse_text()

		self.rects = []
		self.surfaces = []

		current_y = self.position[1]

		surfaces = [
			[self.font.render(string, True, color, None) for color, string in line] for line in parsed
		]

		for line in surfaces:

			width = sum(s.get_width() for s in line)
			current_width = 0

			for surface in line:
				
				rect = surface.get_rect()

				if self.alignment == self.Alignment.LEFT:
					rect.left = self.position[0] + current_width
				elif self.alignment == self.Alignment.RIGHT:
					rect.right = self.position[0] + current_width + rect.width - width
				elif self.alignment == self.Alignment.CENTER:
					rect.centerx = self.position[0] + current_width - (width - rect.width) // 2
				
				rect.top = current_y
				current_width += rect.width

				self.rects.append(rect)
				self.surfaces.append(surface)
			
			current_y += max(s.get_height() for s in line) + self.vertical_spacing

	def draw(self, surface: pg.Surface) -> None:
		"""Draw the text on the given surface."""
		for surf, rect in zip(self.surfaces, self.rects):
			surface.blit(surf, rect)

class PlainText:
	"""
	Simple text class with lower overhead than Text class.
	Does not support multiple colors, lines or alignments.
	Use this class when you only need to display a single line of text in a single color.
	"""

	value: Any
	string: str

	font: Font

	position: tuple[int, int]

	surface: pg.Surface
	rect: pg.Rect

	def __init__(self, value: Any, position: Iterable[float], color: ColorValue = "white", size: int = 30) -> None:

		self.value = value
		self.string = str(value)

		self.font = get_font(default_font[0], size, default_font[2], default_font[3])

		self.position = tuple(map(int, position)) # type: ignore

		if len(self.position) != 2:
			raise ValueError("Position must be a tuple of (x, y)")

		self.surface = self.font.render(self.string, True, color, None)

		self.rect = self.surface.get_rect()
		self.rect.topleft = self.position

	def draw(self, surface: pg.Surface) -> None:
		"""Draw the text on the given surface."""
		surface.blit(self.surface, self.rect)

def get_font(name: str | bytes, size: int = 30, bold: Hashable = False, italic: Hashable = False) -> Font:
	return _get_sys_font(str(name), int(size), hash(bold), hash(italic))

_get_sys_font = cache(pg.font.SysFont)
