from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from dashboard.dashboard_models import (
    CompareSolversRequest,
    ManualCampaignRequest,
    ManualCaseRequest,
    ManualCasesRequest,
    RunCampaignRequest,
)


def endpoint(path: str, method: str):
    def routes():
        for route in main.app.routes:
            router = getattr(route, "original_router", None)
            if router is None:
                yield route
            else:
                yield from router.routes

    for route in routes():
        if getattr(route, "path", None) == path and method in (route.methods or set()):
            return route.endpoint
    raise AssertionError(f"Route not registered: {method} {path}")


class DashboardApiIntegrationTests(unittest.TestCase):
    def binary_case_count(self, path: Path) -> int:
        return main.binary_case_count(path)

    def test_manual_campaign_mutation_and_preview_regenerate_missing_binary(self) -> None:
        create_manual = endpoint("/api/campaigns/manual", "POST")
        replace_cases = endpoint("/api/campaigns/{name}/cases", "PUT")
        preview = endpoint("/api/campaigns/{name}/preview/{kind}", "GET")
        case = ManualCaseRequest(
            start=(0.0, 0.0),
            target=(2.0, 0.0),
            polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
        )

        with tempfile.TemporaryDirectory() as directory:
            campaigns_root = Path(directory)
            with patch.object(main, "CAMPAIGNS_ROOT", campaigns_root):
                created = asyncio.run(create_manual(ManualCampaignRequest(name="integration")))
                self.assertTrue(created["ok"])
                updated = asyncio.run(
                    replace_cases(
                        "integration",
                        ManualCasesRequest(cases=[case]),
                    )
                )
                self.assertEqual(tuple(updated["cases"][0]["target"]), (2.0, 0.0))

                campaign_path = campaigns_root / "integration"
                binary_path = campaign_path / "inputs/manual.bin"
                self.assertFalse(binary_path.exists())

                response = asyncio.run(preview("integration", "instance-0"))
                self.assertEqual(response.path, campaign_path / "previews/instances/case-0000.svg")
                self.assertTrue(binary_path.exists())
                self.assertTrue(Path(response.path).exists())

    def test_comparison_regenerates_missing_manual_binary_before_suite_lookup(
        self,
    ) -> None:
        create_manual = endpoint("/api/campaigns/manual", "POST")
        replace_cases = endpoint("/api/campaigns/{name}/cases", "PUT")
        compare = endpoint("/api/comparisons", "POST")
        case = ManualCaseRequest(
            start=(0.0, 0.0),
            target=(2.0, 0.0),
            polygons=[[(0.5, 0.5), (1.5, 0.5), (1.0, 1.25)]],
        )

        async def noop_run_job(_job):
            return None

        with tempfile.TemporaryDirectory() as directory:
            campaigns_root = Path(directory)
            with (
                patch.object(main, "CAMPAIGNS_ROOT", campaigns_root),
                patch.object(main, "run_job", noop_run_job),
            ):
                main.jobs.clear()
                self.addCleanup(main.jobs.clear)
                asyncio.run(create_manual(ManualCampaignRequest(name="integration")))
                asyncio.run(
                    replace_cases(
                        "integration",
                        ManualCasesRequest(cases=[case]),
                    )
                )
                binary_path = campaigns_root / "integration" / "inputs/manual.bin"
                self.assertFalse(binary_path.exists())

                response = asyncio.run(
                    compare(
                        CompareSolversRequest(
                            name="integration",
                            solvers=["linear"],
                            no_build=True,
                        )
                    )
                )

                self.assertTrue(binary_path.exists())
                self.assertEqual(
                    response["command"][response["command"].index("--suite") + 1],
                    str(binary_path),
                )

    def test_run_regenerates_stale_manual_binary_before_dispatch(self) -> None:
        create_manual = endpoint("/api/campaigns/manual", "POST")
        replace_cases = endpoint("/api/campaigns/{name}/cases", "PUT")
        run_campaign = endpoint("/api/runs", "POST")
        first = ManualCaseRequest(
            start=(0.0, 0.0),
            target=(1.0, 0.0),
            polygons=[[(0.0, 1.0), (1.0, 1.0), (0.5, 2.0)]],
        )
        cases = [
            first,
            ManualCaseRequest(
                start=(2.0, 0.0),
                target=(3.0, 0.0),
                polygons=[[(2.0, 1.0), (3.0, 1.0), (2.5, 2.0)]],
            ),
            ManualCaseRequest(
                start=(4.0, 0.0),
                target=(5.0, 0.0),
                polygons=[[(4.0, 1.0), (5.0, 1.0), (4.5, 2.0)]],
            ),
            ManualCaseRequest(
                start=(6.0, 0.0),
                target=(7.0, 0.0),
                polygons=[[(6.0, 1.0), (7.0, 1.0), (6.5, 2.0)]],
            ),
            ManualCaseRequest(
                start=(8.0, 0.0),
                target=(9.0, 0.0),
                polygons=[[(8.0, 1.0), (9.0, 1.0), (8.5, 2.0)]],
            ),
        ]

        async def noop_run_job(_job):
            return None

        with tempfile.TemporaryDirectory() as directory:
            campaigns_root = Path(directory)
            with (
                patch.object(main, "CAMPAIGNS_ROOT", campaigns_root),
                patch.object(main, "run_job", noop_run_job),
                patch.object(main, "persist_jobs", lambda: None),
            ):
                main.jobs.clear()
                self.addCleanup(main.jobs.clear)
                asyncio.run(create_manual(ManualCampaignRequest(name="integration")))
                campaign_path = campaigns_root / "integration"
                binary_path = campaign_path / "inputs/manual.bin"
                main.write_binary_cases(binary_path, [main.manual_case_from_request(first)])
                self.assertEqual(self.binary_case_count(binary_path), 1)

                asyncio.run(replace_cases("integration", ManualCasesRequest(cases=cases), refresh_previews=False))
                response = asyncio.run(
                    run_campaign(
                        RunCampaignRequest(
                            name="integration",
                            threads=12,
                            solver="binary",
                            max_instances=5,
                            max_calls="1000000",
                            no_build=True,
                        )
                    )
                )

                self.assertEqual(self.binary_case_count(binary_path), 5)
                self.assertIn("--max-instances", response["command"])
                self.assertEqual(response["command"][response["command"].index("--max-instances") + 1], "5")

    def test_manual_edit_makes_existing_completion_marker_stale(self) -> None:
        create_manual = endpoint("/api/campaigns/manual", "POST")
        replace_cases = endpoint("/api/campaigns/{name}/cases", "PUT")
        first = ManualCaseRequest(
            start=(0.0, 0.0),
            target=(1.0, 0.0),
            polygons=[[(0.0, 1.0), (1.0, 1.0), (0.5, 2.0)]],
        )
        second = ManualCaseRequest(
            start=(2.0, 0.0),
            target=(3.0, 0.0),
            polygons=[[(2.0, 1.0), (3.0, 1.0), (2.5, 2.0)]],
        )

        with tempfile.TemporaryDirectory() as directory:
            campaigns_root = Path(directory)
            with patch.object(main, "CAMPAIGNS_ROOT", campaigns_root):
                asyncio.run(create_manual(ManualCampaignRequest(name="integration")))
                campaign_path = campaigns_root / "integration"
                binary_path = campaign_path / "inputs/manual.bin"
                marker = campaign_path / "results/manual.done"
                asyncio.run(replace_cases("integration", ManualCasesRequest(cases=[first]), refresh_previews=False))
                main.ensure_manual_binary_cache(campaign_path)
                marker.parent.mkdir(parents=True)
                old_signature = {
                    "input_size": binary_path.stat().st_size,
                    "input_mtime_ns": binary_path.stat().st_mtime_ns,
                    "max_instances": "1",
                }
                marker.write_text(json.dumps(old_signature, sort_keys=True) + "\n")

                asyncio.run(replace_cases("integration", ManualCasesRequest(cases=[first, second]), refresh_previews=False))
                main.ensure_manual_binary_cache(campaign_path)

                self.assertFalse(marker.exists())
                self.assertEqual(self.binary_case_count(binary_path), 2)


if __name__ == "__main__":
    unittest.main()
