
from abc import ABC
from typing import Self, Sequence
import pygame as pg

pg.init()

# Default screen size for my personal computer
# TODO: Make it adaptable to any screen size
SIZE = WIDTH, HEIGHT = 1512, 945
CENTER = CENTERX, CENTERY = WIDTH // 2, HEIGHT // 2

class Game(ABC):

	all_events: list[pg.event.Event]

	down_keys: set[int]
	up_keys: set[int]
	events: set[int]

	held_keys: Sequence[bool]

	def __init__(self, __fps: int | float = 125) -> None:

		self.down_keys = set()
		self.up_keys = set()
		self.events = set()

		self.all_events: list[pg.event.Event] = []

		self.__all_events: list[set[int]] = [self.down_keys, self.up_keys, self.events]

		self.__base_surface: pg.Surface | None = None
		self.__display = pg.display.set_mode((WIDTH, HEIGHT), pg.FULLSCREEN | pg.SRCALPHA)
		self.surface = self.__display

		self.TIME_STEP = 1000 / __fps

		self.__accumulator = 0

		self.fps_tracker = 0
		self.game_loop_tracker = 0

		self.__run_game = True

		self.__times: list[float] = []
	
	@property
	def base_surface(self) -> pg.Surface | None:
		
		if self.__base_surface is not None: 
			return self.__base_surface
		
		self.__base_surface = self.set_base_surface(pg.Surface((WIDTH, HEIGHT)))

		return self.__base_surface
	
	def set_base_surface(self, surface: pg.Surface) -> pg.Surface | None:
		...

	def draw(self, surface: pg.Surface) -> None:
		...

	def fixed_update(self, down_keys: set[int], up_keys: set[int], events: set[int]) -> None | bool:
		...

	def update(self, down_keys: set[int], up_keys: set[int], events: set[int]) -> None | bool:
		...

	def rendering_update(self) -> None:
		self.__rendering_update()

	def __rendering_update(self) -> None:

		base = pg.Surface((WIDTH, HEIGHT)) if self.base_surface is None else self.base_surface.copy()            
		self.surface.blit(base, (0, 0))
		
		self.draw(self.surface)
		self.__display.blit(self.surface, (0, 0))
		
		pg.display.flip()
	
	def start(self) -> None: 
		...

	def end(self) -> None: 
		...

	def run(self) -> Self:

		self.start()
		self.start_time = __start = pg.time.get_ticks()

		while self.__run_game:

			self.ellapsed_time = pg.time.get_ticks() - self.start_time
			
			self.__times.append(__start)
			__start = self.__loop(__start)
			
		self.end_time = __start
		self.end()

		return self
	
	def __loop(self, __start: int) -> int:
		
		while self.__accumulator >= self.TIME_STEP:
		
			self.game_loop_tracker += 1
			self.__accumulator -= self.TIME_STEP
		
			self.__manage_events()
			self.held_keys = pg.key.get_pressed()
			
			if self.fixed_update(*self.__all_events) is not None: 
				self.__run_game = False

		if self.update(*self.__all_events) is not None: self.__run_game = False

		self.fps_tracker += 1
		self.__rendering_update()

		__end = pg.time.get_ticks()

		self.delta_time = __end - __start
		self.__accumulator += self.delta_time

		return __end

	def __manage_events(self) -> None:

		self.all_events = pg.event.get()

		self.down_keys.clear()
		self.up_keys.clear()
		self.events.clear()

		for i in self.all_events:
			if i.type == pg.KEYDOWN: self.down_keys.add(i.key)
			elif i.type == pg.KEYUP: self.up_keys.add(i.key)
			else: self.events.add(i.type)

		if pg.QUIT in self.events: self.__run_game = False
		if pg.K_ESCAPE in self.down_keys: self.__run_game = False

	def get_info(self, stringify: bool = True, precision: int = 3) -> str | tuple[float, float]:

		if not self.fps_tracker or not self.game_loop_tracker:
			if not stringify: return 0, 0

			return f"The game apparently hasn't been ran yet, " + \
					"rendering it impossible to determine any information about its behavior"

		fps = 1000 * self.fps_tracker / (self.end_time - self.start_time)
		game_fps = 1000 * self.game_loop_tracker / (self.end_time - self.start_time)

		if not stringify: return fps, game_fps

		fps, game_fps = round(fps, precision), round(game_fps, precision)

		expected_fps = round(1000 / self.TIME_STEP, 3)

		message_1 = f'The game ran, on average, at {fps} fps'
		message_2 = f'The game loop was called {game_fps} times per seconds'
		message_3 = f'Expected: {expected_fps}, Difference: {round(game_fps - expected_fps, 3)}'

		return f'{message_1}\n{message_2}\n{message_3}'
