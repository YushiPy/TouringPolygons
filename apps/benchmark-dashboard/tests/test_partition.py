import hashlib
import json
import unittest

from fastapi import HTTPException

from dashboard.dashboard_event import DATA_PATH, event_data
from dashboard.dashboard_partition import PartitionRequest, native_partition, partition_router


class PartitionTests(unittest.TestCase):
	def test_frozen_partitions_match_solver_library(self):
		frozen = json.loads(DATA_PATH.with_name("siicusp34-partitions.json").read_text())
		self.assertEqual(frozen["source_sha256"], hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
		for row in event_data()["rows"]:
			self.assertEqual(native_partition(row["geometry"]["polygons"]), frozen["cases"][str(row["case"])] )

	def test_dashboard_route_uses_native_partition(self):
		endpoint = partition_router().routes[0].endpoint
		polygon = [[0, 0], [3, 0], [3, 3], [1, 1], [0, 3]]
		response = endpoint(PartitionRequest(polygons=[polygon]))
		self.assertEqual(response["pieces"], native_partition([polygon]))
		with self.assertRaises(HTTPException) as error:
			endpoint(PartitionRequest(polygons=[[[0, 0], [2, 2], [2, 0], [0, 2]]]))
		self.assertEqual(error.exception.status_code, 422)
