"""The method figures must use a real incumbent and validate their drawn targets."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_tspn_cards as cards


class CardGeometryTests(unittest.TestCase):
    def setUp(self):
        self.polygons = [[[0, 0], [2, 0], [2, 2], [0, 2]]]

    def test_unvisited_and_open_targets_are_rejected(self):
        for path in ([[3, 3], [4, 3], [3, 3]], [[0, 0], [2, 0], [2, 2]]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cards.validate_tour(path, self.polygons)

    def test_real_initial_incumbent_is_selected_instead_of_a_later_solution(self):
        path = [[0, 0], [2, 0], [0, 0]]
        result = {"initial_length": 4, "trace": [
            {"kind": "incumbent", "source": "heuristic", "length": 4, "path": path},
            {"kind": "incumbent", "source": "oracle", "length": 0, "path": [[0, 0]]},
        ]}
        self.assertEqual(cards.initial_heuristic(result, self.polygons), path)

    def test_missing_trace_and_mismatched_incumbent_are_rejected(self):
        for trace in ([], [{"kind": "incumbent", "source": "heuristic", "length": 5}]):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                cards.initial_heuristic({"initial_length": 4, "trace": trace}, self.polygons)

    def test_a_centroid_bend_cannot_replace_the_native_heuristic(self):
        path = [[0, 0], [1, 1], [2, 0], [0, 0]]
        result = {"initial_length": cards.tour_length(path), "trace": [{
            "kind": "incumbent", "source": "heuristic", "length": cards.tour_length(path), "path": path,
        }]}
        with self.assertRaisesRegex(ValueError, "bends strictly inside"):
            cards.initial_heuristic(result, self.polygons)

    def test_a_hull_tour_is_validated_against_the_hull_it_actually_solves(self):
        nonconvex = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
        path = [[2, 2], [2.2, 1.8], [2, 2]]
        cards.validate_tour(path, [cards.hull(nonconvex)])
        with self.assertRaises(ValueError):
            cards.validate_tour(path, [nonconvex])

    def test_budget_limited_result_cannot_be_called_optimal_in_a_card(self):
        with patch.object(cards, "run_unordered_solver", return_value={
            "termination": "optimal", "path": [[0, 0], [2, 0], [0, 0]], "lower_bound": 3, "upper_bound": 4,
        }), self.assertRaisesRegex(ValueError, "unresolved gap"):
            cards.closed_tour(Path("unused"), self.polygons)


if __name__ == "__main__":
    unittest.main()
