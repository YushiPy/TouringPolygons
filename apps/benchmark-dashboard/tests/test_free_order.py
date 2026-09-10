from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import free_order_campaign
from fastapi import HTTPException
from test_api_integration import endpoint

import main
from dashboard.dashboard_free_order import free_command, free_result_case, free_results
from dashboard.dashboard_models import CompareSolversRequest, LiveSolveRequest, RunCampaignRequest


class FreeOrderTests(unittest.TestCase):
    def test_fixed_order_remains_the_default(self):
        self.assertEqual(RunCampaignRequest(name="sample").visit_order, "fixed")
        self.assertEqual(LiveSolveRequest().visit_order, "fixed")

    def test_free_command_uses_dedicated_runner_and_limits(self):
        request = RunCampaignRequest(
            name="sample",
            visit_order="free",
            solver="unordered",
            threads=4,
            max_calls="10",
            max_seconds="2",
            no_build=True,
            force=True,
        )
        command = free_command(request, Path("/tmp/sample"), Path("/tmp/tpp.py"))
        self.assertEqual(command[2:4], ["free-order", "/tmp/sample"])
        self.assertIn("--no-build", command)
        self.assertIn("--force", command)
        self.assertEqual(command[command.index("--solver") + 1], "unordered")
        self.assertEqual(command[command.index("--threads") + 1], "4")

    def test_rejects_wrong_solver_and_invalid_limits(self):
        for values in ({"solver": "binary"}, {"max_seconds": "nan"}, {"max_calls": "1,2"}):
            with self.subTest(values=values), self.assertRaises(HTTPException):
                free_command(
                    RunCampaignRequest(name="sample", visit_order="free", **values), Path("/tmp/a"), Path("/tmp/tpp.py")
                )
        with self.assertRaises(HTTPException):
            free_command(
                CompareSolversRequest(name="a", solvers=["unordered", "tspn"], max_seconds="0.5"),
                Path("/tmp/a"),
                Path("/tmp/tpp.py"),
                comparison=True,
            )

    def test_reports_do_not_load_fixed_order_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/report.json").write_text('{"rows":[{"solver":"binary"}]}')
            self.assertEqual(free_results(root)["rows"], [])
            run = root / "results/free-order/test"
            run.mkdir(parents=True)
            (run / "report.json").write_text('{"visit_order":"free","rows":[{"solver":"unordered"}]}')
            self.assertEqual(free_results(root)["rows"][0]["solver"], "unordered")

    def test_free_report_defers_and_hydrates_visualization_data(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            run = campaign / "results/free-order/test"
            run.mkdir(parents=True)
            geometry = {"start": [0, 0], "target": [2, 0], "polygons": []}
            (run / "geometry.json").write_text(json.dumps({"cases": {"digest": geometry}}))
            (run / "report.json").write_text(
                json.dumps(
                    {
                        "visit_order": "free",
                        "rows": [
                            {
                                "case": 0,
                                "solver": "unordered",
                                "sha256": "digest",
                                "geometry_sha256": "digest",
                                "path": [[0, 0], [2, 0]],
                            }
                        ],
                    }
                )
            )
            summary = free_results(campaign, endpoint="/detail")
            self.assertNotIn("geometry", summary["rows"][0])
            self.assertNotIn("path", summary["rows"][0])
            self.assertTrue(summary["rows"][0]["visualization_available"])
            detail = free_result_case(campaign, 0)
            self.assertEqual(detail["rows"][0]["geometry"], geometry)
            self.assertEqual(detail["rows"][0]["path"], [[0, 0], [2, 0]])

    def test_free_campaign_resumes_missing_checkpoint_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            inputs = campaign / "inputs/cases.bin"
            main.write_binary_cases(inputs, [((0, 0), (1, 0), []), ((0, 0), (2, 0), [])])
            (campaign / "campaign.json").write_text(
                json.dumps({"name": "resume", "inputs": [{"file": "inputs/cases.bin"}]})
            )

            def solve(_binary, start, target, _polygons, _calls, _seconds):
                return {
                    "path": [list(start), list(target)],
                    "lower_bound": target[0],
                    "upper_bound": target[0],
                    "exact": True,
                    "seconds": 0.001,
                    "calls": 1,
                }

            with (
                patch.object(free_order_campaign, "ensure_binary", return_value=Path("/tmp/fake")),
                patch.object(free_order_campaign, "run_unordered_solver", side_effect=solve) as solver,
            ):
                self.assertEqual(free_order_campaign.main([str(campaign), "--threads", "2"]), 0)
                report_path = next((campaign / "results/free-order").glob("*/report.json"))
                report = json.loads(report_path.read_text())
                report["rows"] = [row for row in report["rows"] if row["case"] == 0]
                report["status"] = "failed"
                report_path.write_text(json.dumps(report))
                self.assertEqual(free_order_campaign.main([str(campaign), "--threads", "2"]), 0)
                self.assertEqual(solver.call_count, 3)

            resumed = json.loads(report_path.read_text())
            self.assertEqual(len(resumed["rows"]), 2)
            self.assertEqual(resumed["checkpoint"]["completed_pairs"], 2)
            self.assertNotIn("geometry", resumed["rows"][0])
            self.assertTrue((report_path.parent / "geometry.json").exists())
            self.assertEqual(len(list((report_path.parent / "geometry").glob("*.json"))), 2)
            self.assertEqual(free_result_case(campaign, 1)["rows"][0]["geometry"]["target"], [2.0, 0.0])

    def test_editor_dispatches_free_order_without_fixed_solver(self):
        solve = endpoint("/api/editor/solve", "POST")
        expected = {"path": [[0, 0], [2, 0]], "order": [0], "exact": True}

        async def fake_solve(case):
            self.assertEqual(case[0], (0.0, 0.0))
            return expected

        with (
            patch("dashboard.dashboard_free_order.solve_free_editor", fake_solve),
            patch.object(main, "ensure_live_solver_binary", side_effect=AssertionError("fixed solver called")),
        ):
            result = asyncio.run(
                solve(LiveSolveRequest(visit_order="free", target=(2, 0), polygons=[[(1, -1), (2, -1), (1, 1)]]))
            )
        self.assertEqual(result, expected)

    def test_recorded_report_has_both_solvers_and_matched_hashes(self):
        result = asyncio.run(main.get_free_reference())
        if not result["rows"]:
            self.skipTest("Local recorded benchmark is absent.")
        ours = {r["case"]: r for r in result["rows"] if r["solver"] == "unordered"}
        for row in result["rows"]:
            self.assertEqual(row["sha256"], ours[row["case"]]["sha256"])
        self.assertEqual(len(ours), 60)
        self.assertTrue(all(r["valid"] for r in ours.values()))

    def test_free_api_builds_job_command(self):
        async def noop(job):
            pass

        async def run():
            return await main.run_campaign(
                RunCampaignRequest(name="sample", visit_order="free", solver="unordered", threads=4, max_seconds="2")
            )

        with (
            patch.object(main, "ensure_manual_binary_cache"),
            patch.object(main, "campaign_path", return_value=Path("/tmp/sample")),
            patch.object(main, "active_job", return_value=None),
            patch.object(main, "persist_jobs"),
            patch.object(main, "run_job", noop),
            patch.dict(main.jobs, clear=True),
        ):
            result = asyncio.run(run())
            self.assertIn("free-order", result["command"])
            self.assertNotIn("binary_search_lazy", result["command"])
            self.assertEqual(result["command"][result["command"].index("--threads") + 1], "4")
        json.dumps(result)
