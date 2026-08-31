from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from dashboard_models import (
    CompareSolversRequest,
    ManualCampaignRequest,
    ManualCaseRequest,
    ManualCasesRequest,
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


if __name__ == "__main__":
    unittest.main()
