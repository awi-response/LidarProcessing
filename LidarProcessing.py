from __future__ import annotations
from typing import List, Any, Callable
from collections import deque
import io
import json
import math
import sys
import os
from pathlib import Path
import tempfile
import threading
import traceback

from config import config as configuration

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Collapsible,
    Input, Select, Checkbox, Button, Log, ProgressBar, Sparkline
)
from textual.containers import VerticalScroll, Horizontal, Container, Vertical
from textual.screen import ModalScreen


class ConfirmOverwrite(ModalScreen[bool]):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-overwrite"):
            yield Static(f"Overwrite existing configuration?\n{self.path}")
            with Horizontal():
                yield Button("Overwrite", id="confirm-save", variant="primary")
                yield Button("Cancel", id="cancel-save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-save")


class _IOToLog(io.TextIOBase):
    """Redirect text writes into a Textual Log widget."""
    def __init__(self, get_log: Callable[[], Log]):
        self._get_log = get_log

    def write(self, s: str) -> int:
        if s:
            self._get_log().write(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover
        pass


class LidarProcessing(App):
    """Config editor (left) + controls, output, RAM bar + Run + bottom per-core CPU sparklines."""
    CSS_PATH = "hanna.css"

    # Enumerated option fields (use the config as reference for allowed values)
    OPTIONS: dict[str, list[str]] = {
        "point_density_method": ["sampling", "density"],
        "data_type": ["raster", "vector"],
        "validation_target": ["DSM", "DEM", "CHM"],
        # "resolution": ["Auto", "0.5", "1", "2", "5"],  # optional
    }

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("pageup", "scroll_up", "Scroll up"),
        ("pagedown", "scroll_down", "Scroll down"),
        ("up", "scroll_up", "Scroll up"),
        ("down", "scroll_down", "Scroll down"),
        ("home", "scroll_top", "Top"),
        ("end", "scroll_bottom", "Bottom"),
        ("c", "collapse_or_expand(True)", "Collapse all"),
        ("e", "collapse_or_expand(False)", "Expand all"),
        ("r", "run_now", "Run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = configuration.Configuration()
        self._config_types = {name: type(value) for name, value in vars(self.cfg).items()}
        self._invalid_inputs: set[str] = set()
        # help texts extracted from inline comments in Configuration.__init__()
        self._help_map: dict[str, str] = {}
        if hasattr(configuration, "get_help_map"):
            try:
                self._help_map = configuration.get_help_map() or {}
            except Exception:
                self._help_map = {}

        # Which pipeline parts to run
        self.cfg.run_preprocessing = True
        self.cfg.run_processing = True
        self.cfg.run_validation = True
        self._config_types.update({
            "run_preprocessing": bool,
            "run_processing": bool,
            "run_validation": bool,
        })

        # stdout/stderr redirection
        self._orig_stdout = None
        self._orig_stderr = None

        # psutil (lazy)
        self._psutil_ok: bool | None = None
        self._psutil = None

        # CPU sparkline data (filled on mount)
        self._cpu_values: list[deque[float]] = []

        # Run state
        self._runner: threading.Thread | None = None

    # ---------- Compose ----------
    def compose(self) -> ComposeResult:
        yield Header()

        # Middle: left config + right controls/output/ram+run
        yield Horizontal(
            # LEFT: Config panel
            Container(
                VerticalScroll(
                    Static("Configuration Variables", classes="header"),
                    *self._build_collapsible_groups(),
                    id="config_scroll",
                ),
                id="config_box",
            ),

            # RIGHT: Actions, Products, Output, RAM bar + Run button
            Container(
                Vertical(
                    Static("Process", classes="section-header"),
                    Container(
                        self._make_action_button("btn-preprocess", "Preprocess", self.cfg.run_preprocessing),
                        self._make_action_button("btn-process", "Process", self.cfg.run_processing),
                        self._make_action_button("btn-validate", "Validate", self.cfg.run_validation),
                        id="actions_row", classes="btn-row",
                    ),
                    Static("Products", classes="section-header", id="products_header"),
                    Container(
                        self._make_product_button("btn-prod-dsm", "DSM", self.cfg.create_DSM, enabled=self.cfg.run_processing),
                        self._make_product_button("btn-prod-dem", "DEM/DTM", self.cfg.create_DEM, enabled=self.cfg.run_processing),
                        self._make_product_button("btn-prod-chm", "CHM", self.cfg.create_CHM, enabled=self.cfg.run_processing),
                        id="products_row", classes="btn-row",
                    ),
                    Static("Configuration file (JSON)", classes="section-header"),
                    Input(value=os.path.join("config", "tui_config.json"), id="config-file"),
                    Container(
                        Button("Load", id="btn-load"),
                        Button("Save", id="btn-save"),
                        id="config_file_actions",
                    ),
                    Static("Output", id="output_header"),
                    Container(
                        Log(id="output_log"),
                        id="output_box",
                    ),
                    Static("RAM usage", id="ram_header"),
                    Container(
                        ProgressBar(total=100, show_eta=False, id="ram_bar"),
                        Button("Run", id="btn-run", variant="primary"),
                        id="ram_box",
                    ),
                    id="right_stack",
                ),
                id="right_box",
            ),
            id="config_row",
        )

        # Bottom full-width per-core CPU sparklines (added at very bottom)
        yield Container(id="cpu_row")

        yield Footer()

    # ---------- Lifecycle ----------
    def on_mount(self) -> None:

        self.add_class(
            "theme-terminal"
            if (os.getenv("COLORTERM") or os.getenv("TERM_PROGRAM") or os.getenv("WT_SESSION") or os.getenv("TERM"))
            else "theme-default"
        )

        # Focus the left scroll so keyboard scrolling works immediately
        self.query_one("#config_scroll").focus()
        self._update_products_enabled()

        # Redirect stdout/stderr into the on-screen Log
        log_getter = lambda: self.query_one("#output_log", Log)
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _IOToLog(log_getter)
        sys.stderr = _IOToLog(log_getter)

        # Start periodic updates
        self.set_interval(1.0, self._update_ram)
        self._init_cpu_sparklines()
        self.set_interval(1.0, self._update_cpu)

        # Make the log auto-scroll as lines arrive
        self.query_one("#output_log", Log).auto_scroll = True

        print("Processing Shell started. Output will appear here.\n")

    def on_unmount(self) -> None:
        # Restore stdio
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

    # ---------- Builders ----------
    def _build_collapsible_groups(self) -> List[Collapsible]:
        groups: dict[str, list[str]] = {
            "General Settings": ["run_name", "year"],
            "Paths": [
                "target_area_dir", "las_files_dir", "las_footprints_dir",
                "preprocessed_dir", "results_dir", "validation_dir",
            ],
            "Preprocessing": [
                "multiple_targets", "target_name_field",
                "max_elevation_threshold", "knn", "multiplier", "sor_passes",
                "elm_filter", "elm_cell", "elm_threshold",
                "radius_filter", "radius_filter_radius", "radius_filter_min_count",
            ],
            "Processing": [
                "fill_gaps", "resolution", "point_density_method",
            ],
            "Ground Filtering": [
                "smrf_filter", "csf_filter", "threshold",
                "smrf_window_size", "smrf_slope", "smrf_scalar",
                "csf_rigidness", "csf_iterations", "csf_time_step",
                "csf_cloth_resolution",
            ],
            "Validation": [
                "data_type", "validation_target", "val_column_point",
                "val_band_raster", "sample_size",
            ],
            "Advanced Settings": [
                "overlap", "filter_date", "automatic_date_parser", "start_date", "end_date",
                "chunk_size", "chunk_overlap", "num_workers",
                "preprocess_chunk_size", "preprocess_chunk_overlap",
                "preprocess_reproject_vertical", "preprocess_vertical_target_epsg",
            ],
        }

        represented = {name for names in groups.values() for name in names}
        represented.update({
            "create_DSM", "create_DEM", "create_CHM",
            "run_preprocessing", "run_processing", "run_validation",
        })
        remaining = [name for name in vars(self.cfg) if not name.startswith("_") and name not in represented]
        if remaining:
            groups["Other Settings"] = remaining

        collapsibles: list[Collapsible] = []
        for group_name, var_names in groups.items():
            rows: list[Horizontal] = []

            for var_name in var_names:
                value = getattr(self.cfg, var_name)

                # Label
                name = Static(f"{var_name}:", classes="var-name")
                name.styles.width = 33
                name.styles.min_width = 12
                # Tooltip from inline comments (hover over the label)
                help_text = self._help_map.get(var_name)
                if help_text:
                    name.tooltip = help_text

                # Editor
                editor = self._make_editor(var_name, value)
                editor.styles.width = "1fr"
                editor.styles.min_width = 0

                rows.append(Horizontal(name, editor, classes="row"))

            # Wrap rows in a scrollable body so expanded groups don't fill the whole screen
            body = VerticalScroll(*rows, classes="group-body")

            # Collapsible group (collapsed by default)
            collapsibles.append(
                Collapsible(
                    body,
                    title=group_name,
                    collapsed=True,
                    classes="config-group",
                )
            )

        return collapsibles

    def _make_editor(self, var_name: str, value: Any):
        editor_id = f"edit-{var_name}"

        # Enumerated options -> Select
        if var_name in self.OPTIONS:
            options = [(opt, opt) for opt in self.OPTIONS[var_name]]
            v = str(value)
            if v not in [o[1] for o in options]:
                options = [(v, v)] + options  # keep current unknown value selectable
            return Select(options=options, value=v, id=editor_id, classes="var-editor")

        # Booleans -> Checkbox
        if isinstance(value, bool):
            return Checkbox(value=value, id=editor_id, classes="var-editor")

        # Everything else -> Input (string/numeric/dates/paths)
        return Input(value=str(value), id=editor_id, classes="var-editor")

    # ---------- Events: push edits into self.cfg ----------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if (event.input.id or "").startswith("edit-"):
            self._set_config_value(event.input.id.removeprefix("edit-"), event.value)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        if (event.input.id or "").startswith("edit-"):
            self._set_config_value(event.input.id.removeprefix("edit-"), event.input.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        var = (event.select.id or "").removeprefix("edit-")
        if event.select.value is not Select.BLANK:
            self._set_config_value(var, event.select.value)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        var = (event.checkbox.id or "").removeprefix("edit-")
        self._set_config_value(var, event.checkbox.value)

    # ---------- Action & product buttons ----------
    def _make_action_button(self, btn_id: str, label: str, active: bool) -> Button:
        # Active -> primary; inactive -> default + dim style (still clickable)
        btn = Button(label, id=btn_id, classes="action-btn",
                     variant="primary" if active else "default")
        if not active:
            btn.add_class("as-disabled")
        return btn

    def _make_product_button(self, btn_id: str, label: str, active: bool, enabled: bool) -> Button:
        return Button(
            label, id=btn_id, classes="product-btn",
            variant="primary" if active else "default",
            disabled=not enabled,
        )

    def _update_products_enabled(self) -> None:
        processing_on = bool(getattr(self.cfg, "run_processing", True))
        for btn_id, cfg_attr in [
            ("#btn-prod-dsm", "create_DSM"),
            ("#btn-prod-dem", "create_DEM"),
            ("#btn-prod-chm", "create_CHM"),
        ]:
            btn = self.query_one(btn_id, Button)
            btn.disabled = not processing_on
            # keep variant in sync with config value
            btn.variant = "primary" if getattr(self.cfg, cfg_attr) else "default"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button
        # Action buttons
        if btn.id == "btn-preprocess":
            self._toggle_action(btn, "run_preprocessing", "Preprocess")
        elif btn.id == "btn-process":
            self._toggle_action(btn, "run_processing", "Process")
            self._update_products_enabled()
        elif btn.id == "btn-validate":
            self._toggle_action(btn, "run_validation", "Validate")
        # Product buttons (won't fire if disabled)
        elif btn.id == "btn-prod-dsm":
            self._toggle_product(btn, "create_DSM")
        elif btn.id == "btn-prod-dem":
            self._toggle_product(btn, "create_DEM")
        elif btn.id == "btn-prod-chm":
            self._toggle_product(btn, "create_CHM")
        elif btn.id == "btn-save":
            self._request_save()
        elif btn.id == "btn-load":
            self._load_config()
        # Run button
        elif btn.id == "btn-run":
            self._start_run()

    # quick keybinding
    def action_run_now(self) -> None:
        self._start_run()

    def _toggle_action(self, btn: Button, cfg_attr: str, label_text: str) -> None:
        if btn.variant == "primary":
            btn.variant = "default"
            btn.add_class("as-disabled")
            btn.label = label_text
            setattr(self.cfg, cfg_attr, False)
            print(f"{label_text}: OFF\n")
        else:
            btn.variant = "primary"
            btn.remove_class("as-disabled")
            btn.label = label_text
            setattr(self.cfg, cfg_attr, True)
            print(f"{label_text}: ON\n")

    def _toggle_product(self, btn: Button, cfg_attr: str) -> None:
        new_val = not bool(getattr(self.cfg, cfg_attr))
        setattr(self.cfg, cfg_attr, new_val)
        btn.variant = "primary" if new_val else "default"
        print(f"Product {cfg_attr} -> {new_val}\n")

    def _config_path(self) -> Path:
        text = self.query_one("#config-file", Input).value.strip()
        if not text:
            raise ValueError("Enter a JSON configuration file path.")
        path = Path(text).expanduser()
        if path.suffix.lower() != ".json":
            raise ValueError("The configuration file must have a .json extension.")
        return path

    def _sync_inputs(self) -> bool:
        for editor in self.query("Input.var-editor"):
            self._set_config_value((editor.id or "").removeprefix("edit-"), editor.value)
        if self._invalid_inputs:
            print(f"[CONFIG ERROR] Correct invalid values: {', '.join(sorted(self._invalid_inputs))}\n")
            return False
        return True

    def _request_save(self) -> None:
        if self._runner and self._runner.is_alive():
            print("[CONFIG ERROR] Cannot save while a run is in progress.\n")
            return
        if not self._sync_inputs():
            return
        try:
            path = self._config_path()
            payload = json.dumps(
                {"version": 1, "settings": {name: getattr(self.cfg, name) for name in self._config_types}},
                indent=2, allow_nan=False,
            ) + "\n"
            if path.exists():
                self.push_screen(ConfirmOverwrite(path), lambda confirmed: self._write_config(path, payload, True) if confirmed else None)
            else:
                self._write_config(path, payload, False)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[CONFIG ERROR] Could not save configuration: {exc}\n")

    def _write_config(self, path: Path, payload: str, overwrite: bool) -> None:
        temporary = None
        try:
            if self._runner and self._runner.is_alive():
                raise ValueError("Cannot save while a run is in progress.")
            if overwrite:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
                    temporary = Path(file.name)
                    file.write(payload)
                os.replace(temporary, path)
                temporary = None
            else:
                with path.open("x", encoding="utf-8") as file:
                    file.write(payload)
            print(f"Configuration saved to {path}\n")
        except (OSError, ValueError) as exc:
            print(f"[CONFIG ERROR] Could not save configuration: {exc}\n")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError as exc:
                    print(f"[CONFIG ERROR] Could not remove temporary file {temporary}: {exc}\n")

    def _load_config(self) -> None:
        if self._runner and self._runner.is_alive():
            print("[CONFIG ERROR] Cannot load while a run is in progress.\n")
            return
        try:
            path = self._config_path()
            with path.open(encoding="utf-8") as file:
                document = json.load(file)
            if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1:
                raise ValueError("Unsupported or missing configuration version.")
            settings = document.get("settings")
            if not isinstance(settings, dict):
                raise ValueError("Configuration settings must be an object.")
            unknown = settings.keys() - self._config_types.keys()
            if unknown:
                raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
            for name, value in settings.items():
                expected = self._config_types[name]
                valid = type(value) is expected
                if expected is float:
                    valid = type(value) in (int, float) and math.isfinite(value)
                elif name == "resolution":
                    valid = (type(value) in (int, float) and math.isfinite(value)) or value == "Auto"
                if not valid:
                    raise ValueError(f"Invalid value for {name}: {value!r}")

            for name, value in settings.items():
                setattr(self.cfg, name, value)
            self._invalid_inputs.clear()
            self._refresh_config_controls()
            missing = self._config_types.keys() - settings.keys()
            if missing:
                print(f"Configuration kept current values for missing settings: {', '.join(sorted(missing))}\n")
            print(f"Configuration loaded from {path}\n")
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            print(f"[CONFIG ERROR] Could not load configuration: {exc}\n")

    def _refresh_config_controls(self) -> None:
        for editor in self.query(".var-editor"):
            name = (editor.id or "").removeprefix("edit-")
            value = getattr(self.cfg, name)
            if isinstance(editor, Input):
                editor.value = str(value)
            elif isinstance(editor, Checkbox):
                editor.value = value
            elif isinstance(editor, Select):
                options = self.OPTIONS[name]
                if value not in options:
                    options = [value, *options]
                editor.set_options([(option, option) for option in options])
                editor.value = value
        for btn_id, name in [
            ("btn-preprocess", "run_preprocessing"),
            ("btn-process", "run_processing"),
            ("btn-validate", "run_validation"),
        ]:
            button = self.query_one(f"#{btn_id}", Button)
            active = getattr(self.cfg, name)
            button.variant = "primary" if active else "default"
            button.set_class(not active, "as-disabled")
        self._update_products_enabled()

    # ---------- Run pipeline ----------
    def _start_run(self) -> None:
        # Already running?
        if self._runner and self._runner.is_alive():
            print("A run is already in progress.\n")
            return

        if not self._sync_inputs():
            return

        # Update button UI
        run_btn = self.query_one("#btn-run", Button)
        run_btn.disabled = True
        run_btn.label = "Running…"

        print("Starting run with current configuration...\n")

        # Launch background thread
        self._runner = threading.Thread(target=self._run_pipeline, daemon=True)
        self._runner.start()

    def _run_pipeline(self) -> None:
        try:
            # Validate config
            try:
                self.cfg.validate()
            except Exception as e:
                print(f"[CONFIG ERROR] {e}\n")
                return

            # Import pipeline modules lazily
            try:
                import preprocessing  # type: ignore
            except Exception:
                preprocessing = None  # type: ignore
            try:
                import processing  # type: ignore
            except Exception:
                processing = None  # type: ignore
            try:
                import validation  # type: ignore
            except Exception:
                validation = None  # type: ignore

            # Run selected stages
            if getattr(self.cfg, "run_preprocessing", False):
                if preprocessing and hasattr(preprocessing, "preprocess_all"):
                    print("==> Preprocessing...\n")
                    preprocessing.preprocess_all(self.cfg)  # type: ignore[attr-defined]
                    print("Preprocessing done.\n")
                else:
                    print("Preprocessing module/function not available.\n")

            if getattr(self.cfg, "run_processing", False):
                if processing and hasattr(processing, "process_all"):
                    print("==> Processing...\n")
                    processing.process_all(self.cfg)  # type: ignore[attr-defined]
                    print("Processing done.\n")
                else:
                    print("Processing module/function not available.\n")

            if getattr(self.cfg, "run_validation", False):
                if validation and hasattr(validation, "validate_all"):
                    print("==> Validation...\n")
                    validation.validate_all(self.cfg)  # type: ignore[attr-defined]
                    print("Validation done.\n")
                else:
                    print("Validation module/function not available.\n")

            print("Run complete.\n")

        except Exception:
            tb = traceback.format_exc()
            print(f"[RUN ERROR]\n{tb}\n")

        finally:
            # Re-enable Run button on the UI thread
            def _finish():
                btn = self.query_one("#btn-run", Button)
                btn.disabled = False
                btn.label = "Run"
            self.call_from_thread(_finish)

    # ---------- Scroll actions ----------
    def action_scroll_up(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_relative(y=-5)

    def action_scroll_down(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_relative(y=5)

    def action_scroll_top(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_home()

    def action_scroll_bottom(self) -> None:
        self.query_one("#config_scroll", VerticalScroll).scroll_end()

    def action_collapse_or_expand(self, collapse: bool) -> None:
        for coll in self.walk_children(Collapsible):
            coll.collapsed = collapse

    # ---------- Casting & assignment ----------
    def _set_config_value(self, var_name: str, raw_value: Any) -> None:
        if not var_name:
            return

        if var_name in self.OPTIONS:
            setattr(self.cfg, var_name, str(raw_value))
            return

        value_type = self._config_types[var_name]
        try:
            if var_name == "resolution":
                text = str(raw_value).strip()
                if text.lower() == "auto":
                    value = "Auto"
                else:
                    try:
                        value = int(text)
                    except ValueError:
                        value = float(text)
            elif value_type is bool:
                value = bool(raw_value)
            elif value_type is int:
                value = int(str(raw_value).strip())
            elif value_type is float:
                value = float(str(raw_value).strip())
            else:
                value = str(raw_value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite number")
        except ValueError:
            self._invalid_inputs.add(var_name)
            print(f"[CONFIG ERROR] Invalid value for {var_name}: {raw_value!r}\n")
            return

        self._invalid_inputs.discard(var_name)
        setattr(self.cfg, var_name, value)

    # ---------- RAM polling ----------
    def _update_ram(self) -> None:
        """Poll system RAM % and push into the progress bar once per second."""
        if self._psutil_ok is None:
            if not self._ensure_psutil():
                print("psutil not available; RAM bar disabled.\n")
                return
        if not self._psutil_ok:
            return

        try:
            mem = self._psutil.virtual_memory()
            pct = float(mem.percent)  # 0..100
        except Exception:
            return

        bar = self.query_one("#ram_bar", ProgressBar)
        bar.progress = max(0, min(100, int(round(pct))))

    # ---------- CPU sparklines ----------
    def _ensure_psutil(self) -> bool:
        """Try to import psutil once."""
        try:
            import psutil  # type: ignore
            self._psutil_ok = True
            self._psutil = psutil
        except Exception:
            self._psutil_ok = False
        return bool(self._psutil_ok)

    def _init_cpu_sparklines(self) -> None:
        """Create one sparkline per logical CPU, full-width at bottom."""
        if self._psutil_ok is None:
            if not self._ensure_psutil():
                print("psutil not available; CPU sparklines disabled.\n")
                return
        if not self._psutil_ok:
            return

        # Determine core count, warm up cpu_percent, and build deques
        ncores = self._psutil.cpu_count(logical=True) or 1
        self._psutil.cpu_percent(percpu=True)  # prime measurement
        self._cpu_values = [deque(maxlen=120) for _ in range(ncores)]  # 2 minutes @ 1Hz

        row = self.query_one("#cpu_row", Container)
        widgets: list[Sparkline] = []
        for i in range(ncores):
            sp = Sparkline(id=f"cpu_sp_{i}")
            sp.add_class("cpu-spark")
            sp.max_value = 100.0
            widgets.append(sp)

        row.mount(*widgets)

    def _update_cpu(self) -> None:
        """Update per-core sparkline data every second."""
        if not self._psutil_ok:
            return
        try:
            per = self._psutil.cpu_percent(percpu=True)
        except Exception:
            return
        if not per:
            return

        # Append data and refresh widgets
        for i, pct in enumerate(per):
            if i >= len(self._cpu_values):
                break
            self._cpu_values[i].append(float(pct))
            try:
                sp = self.query_one(f"#cpu_sp_{i}", Sparkline)
                sp.data = list(self._cpu_values[i])
            except Exception:
                pass


if __name__ == "__main__":
    LidarProcessing().run()
