import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button, Checkbox, Input, Select

from LidarProcessing import LidarProcessing, configuration


class ConfigurationEditorTests(unittest.TestCase):
    def test_configuration_editor(self):
        asyncio.run(self._check_configuration_editor())

    async def _check_configuration_editor(self):
        cfg = configuration.Configuration()
        cfg.new_setting = 42
        with patch.object(configuration, "Configuration", return_value=cfg):
            app = LidarProcessing()
        with patch.object(LidarProcessing, "on_mount"):
            async with app.run_test(size=(140, 50)) as pilot:
                products = {
                    "create_DSM": "btn-prod-dsm",
                    "create_DEM": "btn-prod-dem",
                    "create_CHM": "btn-prod-chm",
                }
                stages = {"run_preprocessing", "run_processing", "run_validation"}
                for name in vars(app.cfg):
                    if name in stages:
                        continue
                    if name in products:
                        self.assertIsNotNone(app.query_one(f"#{products[name]}", Button))
                    else:
                        self.assertIsNotNone(app.query_one(f"#edit-{name}"), name)

                sor = app.query_one("#edit-sor_passes", Input)
                sor.value = "4"
                sor.post_message(Input.Submitted(sor, sor.value))
                cell = app.query_one("#edit-elm_cell", Input)
                cell.value = "2.5"
                cell.post_message(Input.Blurred(cell, cell.value))
                app.query_one("#edit-elm_filter", Checkbox).value = True
                await pilot.pause()
                self.assertEqual(app.cfg.sor_passes, 4)
                self.assertEqual(app.cfg.elm_cell, 2.5)
                self.assertTrue(app.cfg.elm_filter)
                app.query_one("#edit-point_density_method", Select).value = "sampling"
                await pilot.pause()
                self.assertEqual(app.cfg.point_density_method, "sampling")

                editor = app.query_one("#edit-preprocess_chunk_size", Input)
                editor.value = "not-a-number"
                app._start_run()
                self.assertIsNone(app._runner)
                self.assertEqual(app.cfg.preprocess_chunk_size, 500)
                self.assertIn("preprocess_chunk_size", app._invalid_inputs)

                editor.value = "750"
                app._set_config_value("preprocess_chunk_size", editor.value)
                self.assertEqual(app.cfg.preprocess_chunk_size, 750)
                self.assertNotIn("preprocess_chunk_size", app._invalid_inputs)

                app.query_one("#btn-prod-chm", Button).press()
                await pilot.pause()
                self.assertTrue(app.cfg.create_CHM)

                editor.value = "900"
                with patch("LidarProcessing.threading.Thread") as thread:
                    app._start_run()
                    thread.return_value.start.assert_called_once()
                    self.assertEqual(app.cfg.preprocess_chunk_size, 900)

    def test_resolution_retains_numeric_and_auto_support(self):
        app = LidarProcessing()
        for raw, expected in (("0.5", 0.5), ("Auto", "Auto"), ("2", 2)):
            app._set_config_value("resolution", raw)
            self.assertEqual(app.cfg.resolution, expected)

    def test_save_and_load(self):
        asyncio.run(self._check_save_and_load())

    async def _check_save_and_load(self):
        app = LidarProcessing()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            with patch.object(LidarProcessing, "on_mount"):
                async with app.run_test(size=(140, 50)) as pilot:
                    app.query_one("#config-file", Input).value = str(path)
                    app.query_one("#edit-year", Input).value = "2021"
                    app.query_one("#edit-elm_filter", Checkbox).value = True
                    app.query_one("#edit-point_density_method", Select).value = "sampling"
                    app.query_one("#btn-prod-chm", Button).press()
                    await pilot.pause()
                    app.query_one("#btn-process", Button).press()
                    await pilot.pause()
                    app.query_one("#btn-save", Button).press()
                    await pilot.pause()
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(saved["settings"]["year"], 2021)
                    self.assertTrue(saved["settings"]["elm_filter"])
                    self.assertEqual(saved["settings"]["point_density_method"], "sampling")
                    self.assertFalse(saved["settings"]["run_processing"])
                    self.assertTrue(saved["settings"]["create_CHM"])

                    app.query_one("#edit-year", Input).value = "2030"
                    app.query_one("#edit-elm_filter", Checkbox).value = False
                    app.query_one("#edit-point_density_method", Select).value = "density"
                    app.query_one("#btn-process", Button).press()
                    await pilot.pause()
                    app.query_one("#btn-load", Button).press()
                    await pilot.pause()
                    self.assertEqual(app.cfg.year, 2021)
                    self.assertEqual(app.query_one("#edit-year", Input).value, "2021")
                    self.assertTrue(app.query_one("#edit-elm_filter", Checkbox).value)
                    self.assertEqual(app.query_one("#edit-point_density_method", Select).value, "sampling")
                    self.assertEqual(app.query_one("#btn-process", Button).variant, "default")
                    self.assertTrue(app.query_one("#btn-prod-chm", Button).disabled)
                    self.assertEqual(app.cfg.create_CHM, saved["settings"]["create_CHM"])

                    app.query_one("#edit-year", Input).value = "2040"
                    app.query_one("#btn-save", Button).press()
                    await pilot.pause()
                    app.screen.query_one("#cancel-save", Button).press()
                    await pilot.pause()
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)
                    app.query_one("#btn-save", Button).press()
                    await pilot.pause()
                    app.screen.query_one("#confirm-save", Button).press()
                    await pilot.pause()
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["settings"]["year"], 2040)

                    before = vars(app.cfg).copy()
                    for document in (
                        {"version": 1, "settings": {"year": 2000, "elm_filter": "false"}},
                        {"version": 1, "settings": {"unexpected": 3}},
                        {"version": 2, "settings": {}},
                    ):
                        path.write_text(json.dumps(document), encoding="utf-8")
                        app._load_config()
                        self.assertEqual(vars(app.cfg), before)
                    path.write_text("{", encoding="utf-8")
                    app._load_config()
                    self.assertEqual(vars(app.cfg), before)

                    path.write_text(json.dumps({"version": 1, "settings": {"year": 2022}}), encoding="utf-8")
                    app._load_config()
                    await pilot.pause()
                    self.assertEqual(app.cfg.year, 2022)
                    self.assertEqual(app.cfg.elm_filter, before["elm_filter"])
                    with patch.object(app, "_runner") as runner:
                        runner.is_alive.return_value = True
                        app._load_config()
                        app._request_save()
                        self.assertEqual(app.cfg.year, 2022)
                    invalid_path = Path(directory) / "invalid.json"
                    app.query_one("#config-file", Input).value = str(invalid_path)
                    app.query_one("#edit-year", Input).value = "not-a-number"
                    app._request_save()
                    self.assertFalse(invalid_path.exists())
                    app.query_one("#edit-year", Input).value = "2022"
                    app.query_one("#config-file", Input).value = str(Path(directory) / "config.py")
                    with patch("builtins.print") as output:
                        app._request_save()
                        self.assertIn(".json extension", str(output.call_args))


if __name__ == "__main__":
    unittest.main()
