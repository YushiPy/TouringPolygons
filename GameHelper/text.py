

from typing import Callable, Hashable, Iterable, Any

from pygame.font import Font
from pygame import Vector2
import pygame as pg

pg.font.init()

type FontInfo = tuple[str, int, bool, bool]
type FontInput = tuple[str, int, bool, bool] | tuple[str, int, bool] | tuple[str, int] | tuple[str] | str

fonts: dict[FontInfo, Font] = {}

default_font = 'ヒラキノ角コシックw0', 30, False, False

vertical_spacing = 1.5

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

	def __init__(self, value: Any, position: Iterable[float], color: Any) -> None:

		self.value = value
		self.string = str(value)

		self.position = Vector2(list(position))
		self.color = pg.color.Color(color)

		self.font = get_font(*default_font)
		self.surface = self.font.render(self.string, True, self.color)

	def draw(self, surface: pg.Surface) -> None:
		"""Draw the text on the given surface."""
		surface.blit(self.surface, self.position)


def get_font(name: str | bytes, size: int = 30, bold: Hashable = False, italic: Hashable = False) -> Font:
	return _get_sys_font(str(name), int(size), hash(bold), hash(italic))

_get_sys_font = cache(pg.font.SysFont)
