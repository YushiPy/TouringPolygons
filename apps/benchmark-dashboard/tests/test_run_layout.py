from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

import run_layout

from dashboard.dashboard_reports import completed_instance_count, summary_files


def write_run(directory: Path, cases: int, age: int) -> Path:
    directory.mkdir(parents=True)
    results = directory / "synthetic.csv"
    results.write_text("case_index;status\n" + "".join(f"{index};ok\n" for index in range(cases)))
    (directory / "synthetic.md").write_text("# summary\n")
    index = directory / "run-index.csv"
    with index.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["input_file", "status", "csv_output"])
        writer.writerow(["in.bin", "completed", str(results)])
    for path in (index, directory):
        os.utime(path, (1_700_000_000 + age, 1_700_000_000 + age))
    return directory


class FixedOrderRunDirectoryTests(unittest.TestCase):
    def test_the_dashboard_follows_the_newest_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            write_run(campaign / "results/20260101-000000-aaaaaa", cases=2, age=1)
            newest = write_run(campaign / "results/20260102-000000-bbbbbb", cases=5, age=2)
            self.assertEqual(run_layout.latest_fixed_order_directory(campaign), newest)
            self.assertEqual(completed_instance_count(campaign), 5)
            self.assertEqual([path.parent for path in summary_files(campaign)], [newest])

    def test_a_campaign_with_the_flat_legacy_layout_still_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            write_run(campaign / "results", cases=3, age=1)
            self.assertEqual(completed_instance_count(campaign), 3)
            self.assertEqual(len(summary_files(campaign)), 1)

    def test_free_order_runs_do_not_count_as_fixed_order_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "results/20260101-000000-cccccc").mkdir(parents=True)
            (campaign / "results/20260101-000000-cccccc/report.json").write_text("{}")
            self.assertEqual(completed_instance_count(campaign), 0)
            self.assertEqual(summary_files(campaign), [])


if __name__ == "__main__":
    unittest.main()
