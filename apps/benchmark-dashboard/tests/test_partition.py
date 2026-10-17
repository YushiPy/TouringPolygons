import unittest

from fastapi import HTTPException

from dashboard.dashboard_partition import PartitionRequest, native_partition, partition_router


class PartitionTests(unittest.TestCase):
    def test_native_partition_covers_manual_regions_with_convex_pieces(self):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polygons = [
            [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]],
            [[5, 0], [6, 0], [6, 1], [5, 1]],
        ]
        decomposed = native_partition(polygons)
        self.assertEqual([len(pieces) for pieces in decomposed], [2, 1])
        for vertices, pieces in zip(polygons, decomposed, strict=True):
            shapes = [Polygon(piece) for piece in pieces]
            self.assertTrue(
                all(shape.is_valid and shape.area > 0 and shape.equals(shape.convex_hull) for shape in shapes)
            )
            self.assertLessEqual(unary_union(shapes).symmetric_difference(Polygon(vertices)).area, 1e-12)

    def test_dashboard_route_uses_native_partition(self):
        endpoint = partition_router().routes[0].endpoint
        polygon = [[0, 0], [3, 0], [3, 3], [1, 1], [0, 3]]
        response = endpoint(PartitionRequest(polygons=[polygon]))
        self.assertEqual(response["pieces"], native_partition([polygon]))
        with self.assertRaises(HTTPException) as error:
            endpoint(PartitionRequest(polygons=[[[0, 0], [2, 2], [2, 0], [0, 2]]]))
        self.assertEqual(error.exception.status_code, 422)
