import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_internal'))

import free_order_history
from free_order_history import apply_compatibility, order_trace_initializers, saved_times, select_cases

FIELDS = ['kind', 'node', 'polygon', 'sequence', 'path', 'source', 'reason']


class TraceInitializerOrderTest(unittest.TestCase):
	def test_reorders_single_and_multiline_calls(self):
		source = (
			'trace_event({.kind = "a", .source = f(x, y), .path = initial});\n'
			'trace_event({\n\t.kind = "b",\n\t.sequence = [&] { return std::vector<size_t>{1, 2}; }(),\n'
			'\t.polygon = chosen,\n\t.reason = ok ? "x, y" : "z",\n});\n'
		)
		updated, changed = order_trace_initializers(source, FIELDS)
		self.assertEqual(changed, 2)
		self.assertEqual(updated, (
			'trace_event({.kind = "a", .path = initial, .source = f(x, y)});\n'
			'trace_event({\n\t.kind = "b",\n\t.polygon = chosen,\n'
			'\t.sequence = [&] { return std::vector<size_t>{1, 2}; }(),\n\t.reason = ok ? "x, y" : "z",\n});\n'
		))
		self.assertEqual(order_trace_initializers(updated, FIELDS), (updated, 0))

	def test_leaves_unknown_designators_alone(self):
		source = 'trace_event({.other = 1, .kind = "a"});'
		self.assertEqual(order_trace_initializers(source, FIELDS), (source, 0))

	def test_lowers_the_cmake_standard_only_when_needed(self):
		with tempfile.TemporaryDirectory() as directory:
			cmake = Path(directory) / 'packages/demo/cpp/CMakeLists.txt'
			cmake.parent.mkdir(parents=True)
			cmake.write_text('set(CMAKE_CXX_STANDARD 26)\ntarget_compile_features(x PUBLIC cxx_std_26)\n')
			self.assertEqual(apply_compatibility(Path(directory), '26'), [])
			self.assertEqual(len(apply_compatibility(Path(directory), '23')), 1)
			self.assertEqual(cmake.read_text(), 'set(CMAKE_CXX_STANDARD 23)\ntarget_compile_features(x PUBLIC cxx_std_23)\n')


class CaseSelectionTest(unittest.TestCase):
	def test_sample_is_stratified_and_within_the_caps(self):
		times = saved_times()
		cases = select_cases(times)
		expected = sum(per_source for _, _, per_source in free_order_history.BANDS) * len(free_order_history.SOURCES)
		self.assertEqual(len(cases), expected)
		self.assertEqual(len(set(cases)), expected)
		for case in cases:
			self.assertLessEqual(times[case]['old_seconds'], free_order_history.OLD_SECONDS_CAP)
			self.assertGreaterEqual(times[case]['new_seconds'], free_order_history.NEW_SECONDS_FLOOR)
		self.assertEqual(cases, select_cases(times))


if __name__ == '__main__':
	unittest.main()
