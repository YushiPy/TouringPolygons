import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))
from benchmark_cases import read_encoded_cases
from convert_instances import write_binary_cases
from convert_paula import convert_file, parse_dat
from unordered_validation import validate_path


class PaulaInstanceTests(unittest.TestCase):
	def test_preserves_dimensions_coordinates_and_bbox_center_in_binary(self):
		text = "3 4\n1\n-2 3\n2\n4 -1\n4 5\n4\n0 0\n2 0\n2 2\n0 2\n"
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			source = root / "example.dat"
			source.write_text(text)
			case = convert_file(source, root)
			self.assertEqual(case.start, (1.0, 2.0))
			self.assertEqual(case.target, case.start)
			self.assertEqual(case.polygons, parse_dat(text))
			self.assertEqual((case.meta["points"], case.meta["segments"]), (1, 1))
			output = root / "cases.bin"
			write_binary_cases([case], output)
			(encoded,) = read_encoded_cases(output)
			self.assertEqual(
				struct.unpack_from("<dddd", encoded.data), (1.0, 2.0, 1.0, 2.0)
			)
			self.assertEqual([len(p) for p in encoded.polygons], [1, 2, 4])
			self.assertEqual([list(p) for p in encoded.polygons], case.polygons)

	def test_rejects_malformed_or_nonfinite_files(self):
		for text in [
			"0 2",
			"1 0",
			"1 2 0",
			"1 2 3 0 0 1 0 2 0",
			"1 2 2 0 0 1",
			"1 1 1 inf 0",
			"1 1 1 0 0 extra",
		]:
			with self.subTest(text=text), self.assertRaises(ValueError):
				parse_dat(text)

	def test_validator_checks_finite_segment_instead_of_supporting_line(self):
		regions = [[(1.0, 0.0)], [(2.0, -1.0), (2.0, 1.0)]]
		self.assertTrue(
			validate_path((0, 0), (3, 0), regions, [(0, 0), (3, 0)], 1e-9)["valid"]
		)
		self.assertFalse(
			validate_path((0, 2), (3, 2), regions[1:], [(0, 2), (3, 2)], 1e-9)["valid"]
		)
		self.assertTrue(
			validate_path((0, 0), (0, 0), [[(0, 0)]], [(0, 0), (0, 0)], 1e-9)["valid"]
		)


if __name__ == "__main__":
	unittest.main()
