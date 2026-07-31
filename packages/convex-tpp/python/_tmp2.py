
"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

from collections.abc import Sequence
from itertools import chain
import itertools
from math import isclose, isqrt
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from LegacySolutions.vector2 import Vector2
from LegacySolutions.polygon2 import Polygon2

from common import Solution

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
	raise ValueError("Ray does not intersect with bounding box.")

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


	corners = [
		Vector2(maxx, miny),
		Vector2(maxx, maxy),
		Vector2(minx, maxy),
		Vector2(minx, miny)
	]

	if w1 == w2 and direction1.cross(direction2) < 0:
		return [start1, p1] + corners[w1:] + corners[:w1] + [p2, start2]
	
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


class Drawing:

	def __init__(self, start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]) -> None:
		self.start = Vector2(*start)
		self.target = Vector2(*target)
		self.polygons = [[Vector2(*vertex) for vertex in polygon] for polygon in polygons]

		solution = Solution(start, target, polygons)

		self.cones = [[(Vector2(r1), Vector2(r2)) for r1, r2 in cones] for cones in solution.cones]
		self.first_contact = solution.first_contact
		self.path = solution.path
	
	def get_bbox(self, extra: float = 0.5) -> tuple[float, float, float, float]:
		"""
		Returns the bounding box of the drawing, which is the smallest rectangle
		that contains the start and end points, as well as all polygons.
		
		:param float extra: An optional parameter to expand the bounding box by a certain factor.
		:param bool square: If True, the bounding box will be square, expanding the smaller 
		side to match the larger one.
		
		:return: A tuple (minx, miny, maxx, maxy) representing the bounding box.
		"""

		points = list(chain([self.start, self.target], *self.polygons))
		bleft, tright = Polygon2.bbox(points, extra, True)

		minx, miny = bleft.x, bleft.y
		maxx, maxy = tright.x, tright.y

		return minx, miny, maxx, maxy

	def draw(self, scenes: list[int] | None = None, /, text: bool = True) -> None:

		n: int = len(self.polygons)

		if scenes is None:
			scenes = list(range(n + 1))

		if scenes == []:
			return

		count = len(scenes)

		height = isqrt(count)
		width = (count + height - 1) // height

		fig, axs = plt.subplots(height, width, figsize=(width * 5, height * 5), constrained_layout=True) # type: ignore
		flat = list(axs.flatten()) if count > 1 else [axs]

		for i, a in enumerate(scenes):

			if not (0 <= a <= n):
				continue
			
			self.draw_scene(flat[i], a - 1)
			
			if text:
				flat[i].set_title(f"Regions for polygon {a}", fontsize=14)

		if 0 in scenes and text:
			flat[0].set_title("Final Path", fontsize=14)

		for i in range(count, len(flat)):
			flat[i].set_axis_off()
		
		# Set title for the whole figure
		if text:
			fig.suptitle("Shortest path from Start to End touching every polygon", fontsize=16) # type: ignore

		for axis in flat:
			axis.legend()

		# plt.savefig(f"tmp.png", dpi=300)

		plt.show() # type: ignore

	def draw_scene(self, ax: Axes, index: int = 0) -> None:

		def fill(*args: Any, **kwargs: Any) -> None:

			original = kwargs.copy()

			kwargs.pop("label", None)
			kwargs["color"] = "white"
			kwargs["alpha"] = 1

			ax.fill(*args, **kwargs) # type: ignore
			ax.fill(*args, **original) # type: ignore
		
		def plot(*args: Any, **kwargs: Any) -> None:
			"""
			Plot a line with the given arguments.
			"""

			original = kwargs.copy()

			kwargs["color"] = "white"
			kwargs["linewidth"] = 4
			kwargs["linestyle"] = "solid"
			kwargs["markersize"] = 7
			kwargs.pop("label", None)

			ax.plot(*args, **kwargs) # type: ignore
			ax.plot(*args, **original) # type: ignore

		def draw_cones() -> None:

			for i in range(len(polygon)):

				vertex = polygon[i]

				#ray1, ray2 = self.get_cone(index, i)
				ray1, ray2 = self.cones[index][i]

				if ray1 == ray2:
					continue

				points = locate_cone(vertex, ray1, ray2, bbox)

				fill(*zip(*points), alpha=0.45, color="red")
				
				p1 = locate_ray(vertex, ray1, bbox)
				p2 = locate_ray(vertex, ray2, bbox)

				plot(*zip(vertex, p1), color="red", linewidth=2, linestyle='--')
				plot(*zip(vertex, p2), color="red", linewidth=2, linestyle='--')

			fill([minx], [miny], alpha=0.45, color="red", label="Cone Region")

		def draw_edges() -> None:

			for i in range(len(polygon)):

				#if self.is_blocked_edge(index, i): continue
				if not self.first_contact[index][i]: 
					continue

				v1 = polygon[i]
				v2 = polygon[(i + 1) % len(polygon)]

				#ray1 = self.get_cone(index, i)[1]
				#ray2 = self.get_cone(index, (i + 1) % len(polygon))[0]

				ray1 = self.cones[index][i][1]
				ray2 = self.cones[index][(i + 1) % len(polygon)][0]

				points = locate_edge(v1, ray1, v2, ray2, bbox)

				fill(*zip(*points), alpha=0.45, color="green")
			
			fill([minx], [miny], alpha=0.45, color="green", label="Edge Region")

		bbox = self.get_bbox()
		minx, miny, maxx, maxy = bbox

		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		# Setting labels
		plot(*zip(self.start), "o", color="green", label='Start' * (index == -1), markersize=4)
		plot(*zip(self.target), "o", color="red", label='End' * (index == -1), markersize=4)

		# Fill the background with a cyan color
		fill([minx, minx, maxx, maxx], [miny, maxy, maxy, miny], color="#6abdbe", alpha=0.7)
		
		fill([minx], [miny], color="#6abdbe", alpha=0.7, label="Pass Through Region" * (index != -1))

		if 0 <= index < len(self.polygons):
			polygon = self.polygons[index]
			draw_cones()
			draw_edges()

		for i, polygon in enumerate(self.polygons):
			fill(*zip(*polygon), alpha=0.8, label=f'Polygon {i + 1}' * (index == -1 or index == i))
			plot(*zip(*polygon, polygon[0]), linewidth=2)

		# Plot the final path
		plot(*zip(*self.path), color="purple")
		
		for p in self.path[1:-1]:
			plot(*zip(p), "o", color="purple", markersize=4)

		# Plot the start and end points
		plot(*zip(self.start), "o", color="green", markersize=4)
		plot(*zip(self.target), "o", color="red", markersize=4)

		# ax.legend() # type: ignore
		ax.grid() # type: ignore


import math
def regular_polygon(n: int, center: Vector2 = Vector2(0, 0), radius: float = 1) -> list[Vector2]:
	"""
	Generates the vertices of a regular polygon.

	:param int n: The number of sides of the polygon.
	:param Vector2 center: The center of the polygon.
	:param float radius: The radius of the circumscribed circle of the polygon.

	:return: A list of Vector2 points representing the vertices of the regular polygon.
	"""

	return [center + Vector2(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]


start = (-0.5, -0.5)
target = (1.5, 1.5)
polygons = [
	regular_polygon(20, Vector2(0.5, 0.5), 0.3),
	regular_polygon(20, Vector2(1.5, 0.8), 0.3),
	regular_polygon(100, Vector2(2.5, 2.0), 0.3)
]

start, target, polygons = (
	(-5.496255186600624, -2.4739893724210806), (16.0, 2.0), [[(0.6724334134187322, 3.6281230137969165), (0.6760235461407429, 3.5347795630246353), (0.6832038115847646, 3.4342558468083326), (0.6975643424728069, 3.2978308033719217), (0.7119248733608501, 3.1901268217115972), (0.7298755369709042, 3.0752425746072514), (0.7550064660249799, 2.9352273984488297), (0.7882792546899235, 2.8063615592619326), (0.8160387222991654, 2.705458904240138), (0.8519400495192736, 2.594164789857803), (0.8770709785733493, 2.518772002695576), (0.912972305793458, 2.4361989500893273), (0.9560538984575877, 2.353625897483078), (0.9991354911217176, 2.2818232430428624), (3.290625441865438, 0.3349085348910541), (3.5377707443025264, 0.30401537208641816), (3.8312557909465683, 0.3349085348910541), (4.495458791246241, 0.4121414419026441), (4.958856233315782, 0.59750041873046), (5.066982303132008, 0.659286744339732), (5.391360512580685, 0.9064320467768199), (5.561272908006183, 1.1072376050069537), (5.746631884833999, 1.3389363260417237), (5.901097698857178, 1.5551884656741755), (6.101903257087313, 1.8641200937205353), (6.132796419891949, 1.9722461635367612), (6.179136164098903, 2.157605140364577), (6.2100293269035385, 2.2966243729854394), (6.0246703500757235, 3.8721756760218744), (6.0092237686734045, 3.9339620016311465), (5.823864791845589, 4.505485513516913), (5.792971629040953, 4.5827184205285025), (5.638505815017773, 4.752630815953999), (5.576719489408501, 4.798970560160954), (5.453146838189957, 4.8762034671725445), (5.005195977522735, 5.077009025402678), (4.788943837890283, 5.154241932414267), (4.464565628441606, 5.2623680022304935), (4.171080581797564, 5.339600909242083), (4.109294256188292, 5.355047490644401), (3.6304502327164343, 5.355047490644401), (2.5337429531518563, 5.2623680022304935), (2.008559185473045, 5.107902188207314), (1.9776660226684086, 5.092455606804996), (1.8386467900475472, 5.015222699793406), (1.730520720231321, 4.953436374184134)], [(-20.34325721901824, -2.5750018699275867), (-20.343257219018234, -17.506030409663108), (-3.3920792226163297, -5.194814299291018)], [(-91.13610246413494, -4.072247662104436), (-90.7397925120122, -12.33796916026408), (-90.16496579025502, -19.270536608325006), (-88.53200121134685, -26.298045038714086), (-88.35009722560363, -26.996709758599327), (-86.76598743167528, -31.99242703615182), (-86.32897592842933, -33.2063478785017), (-85.16361191977344, -35.77986006428346), (-84.72660041652748, -36.31398523491741), (-83.22365675606905, -37.36850405009564), (-82.59060296512405, -37.779133536114), (-80.21081488298591, -38.98461108808716), (-78.31709836892009, -39.47017942502711), (-74.14121067123648, -40.48987293260102), (-73.70419916799052, -40.58698659998901), (-72.87873299519259, -40.684100267377005), (-69.43119780291892, -41.07255493692897), (-68.26583379426303, -40.92688443584698), (-66.37211728019722, -40.684100267377005), (-65.4495374400113, -40.53842976629502), (-62.584684252065564, -39.95574776196707), (-61.95344541404363, -39.81007726088508), (-60.15684256736579, -39.22739525655714), (-59.42849006195586, -38.88749742069917), (-58.16601238591198, -38.25625858267723), (-57.14631887833808, -37.527906077267296), (-54.91270452841429, -35.342848561037506), (-54.62136352625031, -35.05150755887353), (-54.524249858862326, -34.95439389148554), (-53.358885850206434, -33.35201837958369), (-51.80506717199858, -30.875619861189925), (-51.27094200136463, -29.904483187310014), (-50.10557799270874, -27.33097100152826), (-50.008464325320745, -27.08818683305828), (-48.31574367806979, -21.217972542587734), (-40.23887636510636, 22.730620146548908), (-47.5313340912486, 30.5281271576465), (-55.043262873022826, 37.30462235239388), (-61.19531492990321, 40.8917941834641), (-65.31559013398791, 41.36721055316619), (-69.31992264132003, 41.35373292922578), (-71.75134872780916, 41.15629745523924), (-72.7296499876163, 41.051320653155756), (-77.58692983795268, 38.792248507904525), (-82.4518961486751, 34.90736127199139), (-85.30928242085538, 31.034343098654254), (-87.10588526753321, 28.460830912872495), (-89.63237274514005, 21.671474732891447), (-90.11640895656093, 16.855747660007584), (-90.9146482646763, 8.26586702864848), (-90.94379089012031, 7.595586643436331)]])
polygons[2] = polygons[2][10:] + polygons[2][:10]

drawing = Drawing(start, target, polygons)
drawing.draw()


import u_tpp_tamc
import common

solution = common.Solution(start, target, polygons)
path = u_tpp_tamc.tpp_solve(start, target, polygons)

import math
import matplotlib.pyplot as plt


def cones_equal(c1: Sequence[tuple[Vector2, Vector2]], c2: Sequence[tuple[Vector2, Vector2]]) -> bool:
	
	if len(c1) != len(c2):
		return False
	
	for (r1_1, r1_2), (r2_1, r2_2) in zip(c1, c2):
		if not (Vector2(r1_1).is_close(Vector2(r2_1)) and Vector2(r1_2).is_close(Vector2(r2_2))):
			return False
	
	return True


points = solution.polygons[-1]
polygon_current = solution.polygons[-2]
cones = solution.cones[-2]
fc = solution.first_contact[-2]

def get_locations(points: Sequence[Vector2 | tuple[float, float]], index: int) -> list[int]:

	polygon = solution.polygons[index]
	cones = solution.cones[index]

	return [common.locate_point_binary_search2(Vector2(point), polygon, cones.__getitem__) for point in points] # type: ignore

def separate_points(points: Sequence[Vector2 | tuple[float, float]], index: int) -> tuple[list[Vector2], list[Vector2]]:

	polygon = solution.polygons[index]
	cones = solution.cones[index]

	locations = get_locations(points, index)

	pass_through = []
	reflected = []

	for point, location in zip(points, locations):

		if location % 2 == 0:
			pass_through.append(Vector2(point))
		else:
			v1 = Vector2(polygon[location // 2])
			v2 = Vector2(polygon[(location // 2 + 1) % len(polygon)])
			ray1, ray2 = cones[location // 2]

			if ray1 == ray2:
				pass_through.append(Vector2(point))
			else:
				r = Vector2(point).reflect_segment(v1, v2)
				reflected.append(r)

	return reflected, pass_through

reflected, pass_through = separate_points(points, -2)
points = reflected + pass_through

reflected2, pass_through2 = separate_points(points, -3)

print(len(reflected2), len(pass_through2))

plt.scatter(*zip(*reflected2), color="red", label="Reflected")
#locations = get_locations(reflected, -3)

# plt.plot(locations)