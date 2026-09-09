from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

import main
from dashboard.dashboard_event import event_context, event_data, inline_event_assets, visual_data
from dashboard.dashboard_free_order import solve_free_editor


class EventTests(unittest.TestCase):
	def test_combined_challenge_all_orders_and_pieces_are_feasible(self):
		from itertools import permutations, product

		from shapely.geometry import LineString, Point, Polygon
		from shapely.ops import nearest_points, unary_union

		data = event_context()["challenge"]["combined_challenge"]
		self.assertEqual(len(data["solutions"]), 1944)
		self.assertEqual({(tuple(item["order"]), tuple(item["choices"])) for item in data["solutions"]}, set(product(permutations(range(4)), product(range(3), repeat=4))))
		for polygon, pieces in zip(data["geometry"]["polygons"], data["pieces"]):
			shapes = [Polygon(piece) for piece in pieces]
			self.assertLess(unary_union(shapes).symmetric_difference(Polygon(polygon)).area, 1e-7)
			for shape in shapes:
				self.assertAlmostEqual(shape.area, shape.convex_hull.area)
		for item in data["solutions"]:
			self.assertEqual(item["path"][0], data["geometry"]["start"])
			self.assertEqual(item["path"][-1], data["geometry"]["target"])
			self.assertAlmostEqual(LineString(item["path"]).length, item["length"])
			remaining = item["path"]
			for region in item["order"]:
				shape = Polygon(data["pieces"][region][item["choices"][region]]).buffer(1e-7)
				found = False
				for index, (a, b) in enumerate(zip(remaining, remaining[1:])):
					intersection = LineString([a, b]).intersection(shape)
					if intersection.is_empty:
						continue
					point = nearest_points(Point(a), intersection)[1]
					remaining = [[point.x, point.y], *remaining[index + 1:]]
					found = True
					break
				self.assertTrue(found)
		self.assertEqual(data["solutions"][data["reference"]]["length"], min(item["length"] for item in data["solutions"]))

	def test_piece_challenge_partitions_and_ordered_solutions(self):
		from itertools import product

		from shapely.geometry import Point, Polygon
		from shapely.ops import unary_union

		data = event_context()["challenge"]["piece_challenge"]
		for polygon, pieces in zip(data["geometry"]["polygons"], data["pieces"]):
			shapes = [Polygon(piece) for piece in pieces]
			self.assertLess(unary_union(shapes).symmetric_difference(Polygon(polygon)).area, 1e-7)
			self.assertAlmostEqual(sum(shape.area for shape in shapes), Polygon(polygon).area)
			for shape in shapes:
				self.assertAlmostEqual(shape.area, shape.convex_hull.area)
		self.assertEqual(len(data["solutions"]), 27)
		self.assertEqual({tuple(item["choices"]) for item in data["solutions"]}, set(product(range(3), repeat=3)))
		for item in data["solutions"]:
			with self.subTest(choices=item["choices"]):
				self.assertEqual(item["path"][0], data["geometry"]["start"])
				self.assertEqual(item["path"][-1], data["geometry"]["target"])
				self.assertEqual(len(item["path"]), 5)
				for region, (piece, point) in enumerate(zip(item["choices"], item["path"][1:-1])):
					self.assertLessEqual(Point(point).distance(Polygon(data["pieces"][region][piece])), 1e-7)
				self.assertAlmostEqual(item["length"], sum(math.dist(a, b) for a, b in zip(item["path"], item["path"][1:])))
		self.assertEqual(data["solutions"][data["reference"]]["length"], min(item["length"] for item in data["solutions"]))

	def test_challenge_covers_all_orders_with_ordered_feasible_paths(self):
		from itertools import permutations

		from shapely.geometry import Point, Polygon

		challenge = event_context()["challenge"]
		geometry = challenge["geometry"]
		self.assertEqual({tuple(item["order"]) for item in challenge["solutions"]}, set(permutations(range(4))))
		self.assertEqual(len(challenge["solutions"]), 24)
		for item in challenge["solutions"]:
			with self.subTest(order=item["order"]):
				self.assertEqual(item["path"][0], geometry["start"])
				self.assertEqual(item["path"][-1], geometry["target"])
				self.assertEqual(len(item["path"]), 6)
				for point, index in zip(item["path"][1:-1], item["order"]):
					self.assertLessEqual(Point(point).distance(Polygon(geometry["polygons"][index])), 1e-7)
				self.assertAlmostEqual(item["length"], sum(math.dist(a, b) for a, b in zip(item["path"], item["path"][1:])))
		self.assertEqual(challenge["solutions"][challenge["reference"]]["length"], min(item["length"] for item in challenge["solutions"]))

	def test_snapshot_matches_claims_and_independent_geometry(self):
		from shapely.geometry import LineString, Polygon

		data = event_data()
		rows = data["rows"]
		self.assertEqual({row["case"] for row in rows}, set(range(60)))
		self.assertEqual(len(rows), 60)
		self.assertEqual(sum(row["exact"] for row in rows), 43)
		self.assertEqual(sum(row["termination"] == "time_limit" for row in rows), 17)
		self.assertAlmostEqual(sum(row["seconds"] for row in rows), 44.269529457)
		for row in rows:
			with self.subTest(case=row["case"]):
				line = LineString(row["path"])
				self.assertTrue(all(line.distance(Polygon(polygon)) <= 1e-7 for polygon in row["geometry"]["polygons"]))
				self.assertEqual(row["path"][0], row["geometry"]["start"])
				self.assertEqual(row["path"][-1], row["geometry"]["target"])
				self.assertLessEqual(abs(line.length - row["upper_bound"]), 1e-7 + 1e-9 * row["upper_bound"])
				self.assertLessEqual(row["lower_bound"], row["upper_bound"])
				self.assertTrue(math.isfinite(row["lower_bound"]))
				self.assertEqual(row["exact"], row["upper_bound"] - row["lower_bound"] <= 1e-7 + 1e-9 * row["upper_bound"])
				self.assertEqual(sorted(row["order"]), list(range(row["polygons"])))

	def test_event_route_renders_without_solver_or_historical_files(self):
		request = Request({"type": "http", "method": "GET", "path": "/evento", "headers": []})
		with patch("dashboard.dashboard_free_order.ensure_binary", side_effect=AssertionError("Unexpected build")):
			response = asyncio.run(main.event_page(request))
		self.assertEqual(response.status_code, 200)
		html = response.body.decode()
		self.assertIn('lang="pt-BR"', html)
		self.assertIn('id="map-content"', html)
		self.assertIn("Quatro regiões", html)
		self.assertIn('href="/evento/offline"', html)
		self.assertIn('id="zoom-in"', html)
		self.assertIn('aria-label="Redefinir zoom e posição"', html)
		self.assertIn('id="context-return"', html)
		self.assertIn('id="challenge-dialog"', html)
		self.assertIn('data-challenge-tab="2"', html)
		self.assertIn('class="challenge-method-note"', html)
		self.assertGreater(html.index('class="challenge-method-note"'), html.index('data-challenge-step="2"'))
		self.assertIn('href="#icon-undo"', html)
		self.assertIn('href="#icon-trash"', html)
		self.assertIn('data-layer-note="show-decomposition"', html)
		self.assertIn("Não alteram o caminho calculado", html)
		self.assertIn("Dados e reprodutibilidade", html)
		self.assertNotIn('id="case-search"', html)
		self.assertNotIn('id="show-all-results"', html)
		self.assertNotIn("Você está vendo uma solução já calculada", html)
		self.assertNotIn('class="challenge-details"', html)
		data = json.loads(re.search(r'<script id="event-data" type="application/json">(.*?)</script>', html, re.S)[1])
		self.assertEqual(data, json.loads(json.dumps(visual_data())))

	def test_offline_document_contains_all_assets_and_valid_javascript(self):
		request = Request({"type": "http", "method": "GET", "path": "/evento/offline", "headers": []})
		response = asyncio.run(main.event_offline(request))
		html = response.body.decode()
		self.assertIn('attachment; filename="tpp-siicusp34.html"', response.headers["content-disposition"])
		self.assertNotRegex(html, r'(?:src|href)="/(?:static|evento)')
		self.assertNotIn('href="/"', html)
		self.assertIn("<style>", html)
		challenge = json.loads(re.search(r'<script id="challenge-data" type="application/json">(.*?)</script>', html, re.S)[1])
		self.assertEqual(challenge, event_context()["challenge"])
		javascript = re.search(r'<script type="module">(.*?)</script>', html, re.S)[1]
		self.assertNotRegex(javascript, r"(?m)^(import|export) ")
		result = subprocess.run(["node", "--input-type=module", "--check"], input=javascript, text=True, capture_output=True)
		self.assertEqual(result.returncode, 0, result.stderr)
		encoded = re.search(r'href="data:application/json;base64,([^"]+)"', html)[1]
		self.assertEqual(json.loads(base64.b64decode(encoded)), event_data())

	def test_offline_export_is_deterministic(self):
		html = main.templates.get_template("event.html").render(request=object(), offline=True, **event_context())
		self.assertEqual(inline_event_assets(html), inline_event_assets(html))

	def test_live_solver_uses_shared_runner_and_built_binary(self):
		case = ((0, 0), (2, 0), [[(1, 0), (2, 1), (1, 1)]])
		result = {"path": [[0, 0], [2, 0]], "exact": True}
		with patch("dashboard.dashboard_free_order.ensure_binary", return_value=Path("/tmp/test-tpp")), patch("dashboard.dashboard_free_order.run_unordered_solver", return_value=result) as runner:
			self.assertEqual(asyncio.run(solve_free_editor(case)), result)
			runner.assert_called_once_with(Path("/tmp/test-tpp"), *case, 200000, 3)

	def test_context_projection_stays_inside_viewport(self):
		context = event_context()
		for polygon in context["polygons"] + [context["path"]]:
			for point in polygon.split():
				x, y = map(float, point.split(","))
				self.assertTrue(0 <= x <= 840 and 0 <= y <= 480)

	def test_visual_contacts_and_convex_pieces_cover_the_original_regions(self):
		from shapely.geometry import LineString, Point, Polygon
		from shapely.ops import unary_union

		for row in visual_data()["rows"]:
			line = LineString(row["path"])
			for vertices, contact, pieces in zip(row["geometry"]["polygons"], row["visualization"]["contacts"], row["visualization"]["decomposition"]):
				with self.subTest(case=row["case"]):
					polygon = Polygon(vertices)
					self.assertIsNotNone(contact)
					self.assertGreaterEqual(contact["fraction"], 0)
					self.assertLessEqual(contact["fraction"], 1 + 1e-12)
					self.assertLessEqual(Point(contact["point"]).distance(polygon), 1e-7)
					self.assertLessEqual(Point(contact["point"]).distance(line), 1e-7)
					triangles = [Polygon(piece) for piece in pieces]
					self.assertTrue(all(piece.is_valid and piece.area > 0 and piece.convex_hull.symmetric_difference(piece).area <= max(1e-8, piece.area * 1e-12) for piece in triangles))
					self.assertLessEqual(unary_union(triangles).symmetric_difference(polygon).area, max(1e-8, polygon.area * 1e-12))

	def test_visual_enrichment_does_not_modify_the_recorded_evidence(self):
		for original, enriched in zip(event_data()["rows"], visual_data()["rows"]):
			self.assertEqual(original, {key: value for key, value in enriched.items() if key != "visualization"})

	def test_phone_preview_has_no_campaign_or_solver_routes(self):
		import event_server

		paths = {getattr(route, "path", "") for route in event_server.app.routes}
		self.assertEqual(paths, {"/", "/evento", "/evento/offline", "/static"})
		request = Request({"type": "http", "method": "GET", "path": "/evento", "headers": []})
		html = asyncio.run(event_server.event(request)).body.decode()
		self.assertNotIn('href="/"', html)
		self.assertIn('href="/evento/offline"', html)
