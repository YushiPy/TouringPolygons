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

    def test_index_template_contains_required_hooks(self) -> None:
        template = main.templates.get_template("index.html").render(request=object())
        for element_id in (
            "create-form",
            "manual-case-canvas",
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

    def test_generated_preview_images_use_lazy_loading(self) -> None:
        for path in (ROOT / "static/app.js", ROOT / "static/campaign-rendering.js"):
            self.assertIn('loading="lazy"', path.read_text(), path.name)

    def test_benchmark_solution_preview_uses_dashboard_style(self) -> None:
        source = (
            ROOT.parents[1]
            / "packages/nonconvex-tpp/cpp/src/main-bnb_workload_benchmark.cpp"
        ).read_text()

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
                self.assertTrue((source.parent / imported.removeprefix("./")).exists(), f"{source.name}: {imported}")


if __name__ == "__main__":
    unittest.main()
