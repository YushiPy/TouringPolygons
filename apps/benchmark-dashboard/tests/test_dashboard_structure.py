import re
import unittest
from pathlib import Path

import main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class DashboardStructureTests(unittest.TestCase):
    def test_python_support_modules_live_in_dashboard_package(self) -> None:
        self.assertTrue((ROOT / "dashboard/__init__.py").exists())
        self.assertEqual(
            sorted(path.name for path in ROOT.glob("dashboard_*.py")),
            [],
        )

    def test_campaign_routes_are_registered(self) -> None:
        paths_data = main.app.openapi()["paths"]
        paths = set(paths_data)
        self.assertTrue(
            {
                "/editor/offline",
                "/api/campaigns",
                "/api/campaigns/{name}",
                "/api/campaigns/{name}/preview",
                "/api/campaigns/{name}/cases",
                "/api/editor/solve",
            }.issubset(paths)
        )
        self.assertIn("get", paths_data["/api/campaigns"])
        self.assertIn("put", paths_data["/api/campaigns/{name}/cases"])
        self.assertIn("post", paths_data["/api/editor/solve"])

    def test_offline_editor_is_self_contained(self) -> None:
        page = (ROOT / "offline-editor/index.html").read_text()
        script = (ROOT / "static/offline-editor.js").read_text()

        self.assertIn("/static/offline-editor.js", page)
        self.assertIn('id="manual-case-canvas"', page)
        self.assertIn('id="offline-case-list"', page)
        self.assertIn('id="offline-export-all"', page)
        self.assertIn('data-editor-layer="lastStepMap"', page)
        self.assertIn("partitionProvider: localPartition", script)
        self.assertIn("solveWasmProvider: solveEditorWasmAsync", script)
        self.assertIn("lastStepMapProvider: solveEditorWasmMaps", script)
        self.assertIn("source.startPoint", script)
        self.assertIn("tpp-offline-editor-library-v2", script)
        self.assertNotIn("fetch(", script)

    def test_siicusp_event_is_self_contained_and_uses_its_local_solver(self) -> None:
        event_root = ROOT.parent / "siicusp34"
        index = (event_root / "index.html").read_text()
        app = (event_root / "app.js").read_text()
        solver = (event_root / "tpp-solver.js").read_text()

        self.assertIn('<script type="module" src="app.js"></script>', index)
        self.assertIn('from "./tpp-solver.js"', app)
        self.assertIn("solveChallengeRoute", app)
        self.assertIn("export function tppSolveConvex", solver)
        self.assertNotIn("visualizer-local", index + app + solver)
        for filename in ("tpp-solver.js", "tpp-vector2.js"):
            self.assertTrue((event_root / filename).exists(), filename)

    def test_index_template_contains_required_hooks(self) -> None:
        template = main.templates.get_template("index.html").render(request=object())
        for element_id in (
            "append-target",
            "create-form",
            "manual-case-canvas",
            "manual-background-input",
            "manual-satellite-button",
            "satellite-map",
            "run-form",
            "compare-form",
            "campaign-modal",
            "confirm-modal",
            "keybind-modal",
            "job-dock",
        ):
            self.assertRegex(template, rf'id=["\']{element_id}["\']')

    def test_style_manifest_imports_existing_files(self) -> None:
        manifest = (ROOT / "static/style.css").read_text()
        imports = re.findall(r'@import\s+url\(["\']([^"\']+)["\']\)', manifest)
        imports = [Path(imported).name for imported in imports]
        self.assertTrue(imports)
        for imported in imports:
            self.assertTrue((ROOT / "static" / imported).exists(), imported)

    def test_template_css_links_exist(self) -> None:
        template = (ROOT / "templates/index.html").read_text()
        links = re.findall(r'href=["\']/static/([^"\']+\.css)', template)
        self.assertTrue(links)
        for linked in links:
            self.assertTrue((ROOT / "static" / linked).exists(), linked)

    def test_manual_instance_actions_use_original_controls(self) -> None:
        manual_cases = (ROOT / "static/manual-cases.js").read_text()

        self.assertIn('duplicateButton.textContent = "⧉"', manual_cases)
        self.assertIn("setCloseIcon(deleteButton)", manual_cases)
        self.assertNotIn("setDuplicateIcon(duplicateButton)", manual_cases)

    def test_dashboard_test_runner_exists(self) -> None:
        script = ROOT / "scripts/run-tests.sh"

        self.assertTrue(script.exists())
        self.assertIn("RUN_BROWSER", script.read_text())

    def test_dashboard_owns_browser_solver_assets(self) -> None:
        self.assertEqual(main.WASM_STATIC_ROOT, ROOT / "static/wasm")
        self.assertTrue((ROOT / "wasm/build.sh").exists())
        self.assertTrue((ROOT / "wasm/test-intersections.mjs").exists())

        solver = (ROOT / "static/editor-solver.js").read_text()
        self.assertIn("/static/wasm/", solver)
        self.assertIn("solveEditorWasmMaps", solver)
        self.assertIn("tpp_solve_convex_maps", (ROOT / "wasm/tpp_convex_wasm.cpp").read_text())
        self.assertNotIn("visualizer-static", solver)

    def test_generated_preview_images_use_lazy_loading(self) -> None:
        for path in (ROOT / "static/app.js", ROOT / "static/campaign-rendering.js"):
            self.assertIn('loading="lazy"', path.read_text(), path.name)

    def test_benchmark_solution_preview_uses_dashboard_style(self) -> None:
        source = (ROOT.parents[1] / "packages/nonconvex-tpp/cpp/src/main-bnb_workload_benchmark.cpp").read_text()

        self.assertIn('data-preview-version=\\"7\\"', source)
        self.assertIn('fill=\\"#121417\\"', source)
        self.assertIn('stroke=\\"#facc15\\"', source)

    def test_large_preview_grids_start_with_a_sample(self) -> None:
        module = (ROOT / "static/preview-panels.js").read_text()
        self.assertIn("PREVIEW_SAMPLE_THRESHOLD = 100", module)
        self.assertIn("Load all ${previewCount} instances", module)

    def test_local_javascript_imports_resolve(self) -> None:
        for source in (ROOT / "static").glob("*.js"):
            for imported in re.findall(r'from ["\'](\./[^"\']+)["\']', source.read_text()):
                self.assertTrue(
                    (source.parent / imported.removeprefix("./").split("?", 1)[0]).exists(),
                    f"{source.name}: {imported}",
                )


if __name__ == "__main__":
    unittest.main()
