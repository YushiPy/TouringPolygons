"""The compact files of paper 1 keep Gurobi's total and add its two parts."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

INTERNAL = Path(__file__).resolve().parents[1] / "_internal"
sys.path.insert(0, str(INTERNAL))

import paper1_cycles
import paper1_experiments


def cycle_row(name: str, repeat: int, build: float, optimize: float, total: float) -> dict:
	return {"name": name, "k": 4, "vertices": 16, "largest": 4, "repeat": repeat, "config": "exact", "gap": 0,
		"ours": {"seconds": 1e-4, "mean": 1e-4, "calls": 10, "status": 3, "step": 2, "exact": True,
			"lower": 1.0, "upper": 1.0, "exact_lower": 1.0, "exact_upper": 1.0, "anchor": 0, "anchored": 3,
			"probes": 0, "closed_form": 1, "rebuilds": 1, "rational_anchor_recoveries": 0,
			"rational_cycle_recoveries": 0, "rational_anchored": 0, "rational_probes": 0},
		"gurobi": {"status": 2, "objective": 1.0, "build": build, "optimize": optimize, "total": total}}


class Paper1SavedTests(unittest.TestCase):
	def test_cycles_per_instance_has_build_and_optimize(self):
		with tempfile.TemporaryDirectory() as tmp:
			out = Path(tmp)
			results = out / "cycles" / "results"
			results.mkdir(parents=True)
			rows = [cycle_row("a-k4", 0, 1e-4, 9e-4, 1.1e-3), cycle_row("a-k4", 1, 2e-4, 8e-4, 1.2e-3),
				cycle_row("a-k4", 2, 3e-4, 7e-4, 1.3e-3)]
			(results / "a-k4.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
			target = out / "cycles-per-instance.csv"
			self.assertEqual(paper1_cycles.write_per_instance(out, target), 1)
			[line] = list(csv.DictReader(target.open()))
			# The total is the median of the totals, as before; each part is
			# the median of that part.
			self.assertEqual(line["gurobi_ms"], "1.2")
			self.assertEqual(line["gurobi_build_ms"], "0.2")
			self.assertEqual(line["gurobi_optimize_ms"], "0.8")
			summary = paper1_cycles.summarize_cycles(rows)["all"]
			self.assertAlmostEqual(summary["gurobi_median_s"], 1.2e-3)
			self.assertAlmostEqual(summary["gurobi_build_median_s"], 2e-4)
			self.assertAlmostEqual(summary["gurobi_optimize_median_s"], 8e-4)

	def test_gurobi_part_ignores_rows_without_it(self):
		reps = [{"gurobi": {"total": 1.0, "build": 0.25}}, {"gurobi": {"total": 2.0}}, {"other": {}}]
		self.assertEqual(paper1_experiments.gurobi_part(reps, "build"), 0.25)
		self.assertIsNone(paper1_experiments.gurobi_part(reps, "optimize"))


if __name__ == "__main__":
	unittest.main()
