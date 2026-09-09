from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from test_api_integration import endpoint

import main
from dashboard.dashboard_free_order import free_command, free_results
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
            threads=1,
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

    def test_rejects_wrong_solver_and_invalid_limits(self):
        for values in ({"solver": "binary"}, {"max_seconds": "nan"}, {"max_calls": "1,2"}, {"threads": 2}):
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
                RunCampaignRequest(name="sample", visit_order="free", solver="unordered", threads=1, max_seconds="2")
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
        json.dumps(result)
