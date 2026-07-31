from itertools import chain
import json
import sys
import base64
from PIL import Image

from sqlmodel import Session, select, create_engine
from models import User, Drawing

engine = create_engine("sqlite:///tpp.db")

def make_blank_png() -> str:
	png = base64.b64encode(bytes([
		137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,
		0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,0,0,0,
		11,73,68,65,84,120,156,98,0,1,0,0,5,0,1,13,10,
		45,180,0,0,0,0,73,69,78,68,174,66,96,130
	])).decode()
	return f"data:image/png;base64,{png}"

def image_to_polygons(
	image_path: str,
	alpha_threshold: int = 10,
	gap: float = 0.1,
) -> tuple[list[list[tuple[float, float]]], list[str]]:
	img = Image.open(image_path).convert("RGBA")
	w, h = img.size

	polygons = []
	colors = []

	g = gap / 2

	for row in range(h):
		cols = range(w) if row % 2 == 0 else range(w - 1, -1, -1)
		for col in cols:
			r, g_val, b, a = img.getpixel((col, row))  # type: ignore

			if a < alpha_threshold:
				continue

			x = float(col)
			y = float(-row)

			square = [
				(x + g,       y + g),
				(x + 1.0 - g, y + g),
				(x + 1.0 - g, y + 1.0 - g),
				(x + g,       y + 1.0 - g),
			]

			polygons.append(square)
			colors.append(f"#{r:02x}{g_val:02x}{b:02x}")

	return polygons, colors

def inject_drawing(
	username: str,
	drawing_name: str,
	start: tuple[float, float],
	target: tuple[float, float],
	polygons: list[list[tuple[float, float]]],
	colors: list[str] | None = None,
):
	if not polygons:
		print("No polygons to inject.")
		sys.exit(1)

	minx = min(chain((p[0] for poly in polygons for p in poly), [start[0], target[0]]))
	maxx = max(chain((p[0] for poly in polygons for p in poly), [start[0], target[0]]))
	miny = min(chain((p[1] for poly in polygons for p in poly), [start[1], target[1]]))
	maxy = max(chain((p[1] for poly in polygons for p in poly), [start[1], target[1]]))

	width  = maxx - minx
	height = maxy - miny

	SCREEN_WIDTH  = 800
	SCREEN_HEIGHT = 600

	units_to_pixels = min(SCREEN_WIDTH / width, SCREEN_HEIGHT / height) * 0.9
	center = ((minx + maxx) / 2, (miny + maxy) / 2)

	if colors is None:
		colors = ["#2f72dc", "#dc2f72", "#2fdc72", "#722fdc", "#dc2fce", "#ffd902", "#dc722f", "#72dc2f", "#2f72ce", "#ce2f72"]

	data = {
		"startPoint": list(start),
		"targetPoint": list(target),
		"polygons": [[list(p) for p in poly] for poly in polygons],
		"polygonColors": colors,

		"currentPolygon": 0,
		"currentPolygonVertex": 0,
		"scrollSensitivity": 0.001,
		"snapping": False,
		"showVertexLine": False,

		"camera": {
			"position": list(center),
			"unitsToPixels": units_to_pixels,
		},

		"dataURL": make_blank_png(),
		"width": 800,
		"height": 600,

		"drawingName": drawing_name,
	}

	with Session(engine) as db:
		user = db.exec(select(User).where(User.username == username)).first()

		if not user:
			print(f"User '{username}' not found.")
			sys.exit(1)

		drawing = Drawing(user_id=user.id, data=json.dumps(data)) # type: ignore
		db.add(drawing)
		db.commit()
		db.refresh(drawing)

		print(f"Injected drawing '{drawing_name}' (id={drawing.id}) for user '{username}' with {len(polygons)} polygons.")

def inject_from_image(
	username: str,
	drawing_name: str,
	image_path: str,
	alpha_threshold: int = 10,
	gap: float = 0.1,
):
	polygons, colors = image_to_polygons(image_path, alpha_threshold=alpha_threshold, gap=gap)

	if not polygons:
		print("No pixels found after filtering.")
		sys.exit(1)

	img = Image.open(image_path)
	w, h = img.size

	start  = (0.0,       0.0)
	target = (float(w), -float(h))

	inject_drawing(username, drawing_name, start, target, polygons, colors)

def inject_dir(
	username: str,
	input_dir: str,
	alpha_threshold: int = 10,
	gap: float = 0.1,
):
	import os

	for filename in os.listdir(input_dir):
		if filename.lower().endswith((".png", ".jpg", ".jpeg")):
			image_path = os.path.join(input_dir, filename)
			drawing_name = os.path.splitext(filename)[0].lstrip("0123456789-_ ")

			inject_from_image(username, drawing_name, image_path, alpha_threshold=alpha_threshold, gap=gap)


if __name__ == "__main__":
	# --- Inject from image ---

	path = "output.png"

	#inject_dir(
	#	username="Gabriel",
	#	input_dir="_output_images",
	#	alpha_threshold=10,
	#	gap=0.1,
	#)

	instance = ((-2.0, 0.177198062845644), (2.0, 1.0), [[(0.4497354497354502, 4.761904761904762), (-1.851851851851852, 2.5793650793650795), (-0.5423280423280419, 3.32010582010582), (0.3306878306878307, 3.373015873015873), (0.846560846560847, 2.552910052910053), (-1.0, 1.0), (2.0, 2.0)], [(4.969135802469138, -3.1481481481481484), (3.9021164021164005, -2.2751322751322762), (3.505291005291004, -3.240740740740742), (1.970899470899469, -4.232804232804234), (0.6084656084656066, -4.034391534391536), (1.7217813051146398, -5.02300914134991)], [(-1.7900939934870728, -5.1503424152681445), (-4.797306307390922, -6.105278916062125), (-6.405620413991311, -6.281188271471543), (-5.928152163594321, -4.522094717377367), (-3.2895118324530586, -3.7681974799084355), (-7.9594863867745005, -1.588442056267219), (-7.959486386774497, -8.712242774269072)], [(-4.353344600881444, 4.926750658899912), (-9.521728995529566, 3.1425271968901045), (-9.722768258854616, 4.273373053093503), (-8.46627286307306, 5.705777804284474), (-6.80769894064141, 5.152919830140591), (-10.522736994168872, 8.488651017900835), (-10.522736994168868, 1.3648502998989844)], [(17.65625045019001, 7.99160221179721), (16.13829923570195, 10.530476367002898), (19.64436354527171, 10.973771854419763), (16.702493492414327, 13.149949701738926), (14.16361933720864, 10.40957759770739), (15.493505799459237, 7.508007134615176), (10.053061181161336, 5.573626825887033), (8.844073488206247, 7.185610416493819), (8.118680872433194, 9.522986622873656), (9.126170616562435, 12.102160367844515), (7.762701162840861, 13.703645556089068), (7.762701162840866, 2.2795588675053517)], [(23.016095888957572, -6.717748052489705), (19.201068057854844, -7.32224189896725), (15.735303338050254, -9.65961810534709), (14.16361933720864, -7.765537386384116), (13.96212138838279, -5.6699587185952955), (15.856202107345764, -3.372882101980626), (13.122546601608423, -1.0057047081978485), (13.122546601608427, -12.429791396781566)]])

	inject_drawing(
		username="Gabriel",
		drawing_name="Example1",
		start=instance[0],
		target=instance[1],
		polygons=instance[2],
	)

	# --- Or inject manually ---
	# inject_drawing(
	# 	username="Gabriel",
	# 	drawing_name="Example triangle",
	# 	start=(-3.0, 0.0),
	# 	target=(3.0, 0.0),
	# 	polygons=[
	# 		[(0.0, 2.0), (-1.0, 0.0), (1.0, 0.0)],
	# 		[(2.0, 1.0), (1.5, -1.0), (3.0, -1.0)],
	# 	],
	# )