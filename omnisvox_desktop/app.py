"""OmnisVox CFD desktop application (PySide6 + pyqtgraph).

    python -m omnisvox_desktop
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from omnisvox_cfd import CASES, Simulation, __version__, get_backend, gpu_status, json_safe
from omnisvox_cfd.io import encode_png, field_limits, field_to_rgb

# ---- palette (light "drafting paper" theme) ---------------------------------
PAPER, PANEL, INK, MUTED, RULE = "#f1f4f7", "#ffffff", "#15202b", "#5b6875", "#d3dae1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SIGNED_FIELDS = ("vorticity", "vorticity_z", "u", "v", "w", "streamfunction")
ACCURACY = {"Draft": 4.0, "Standard": 1.0, "Strict": 0.3}

STYLE = f"""
QWidget {{ background: {PAPER}; color: {INK}; font-size: 13px; }}
QMainWindow, QScrollArea {{ background: {PAPER}; border: none; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 6px;
            margin-top: 18px; padding: 10px 10px 8px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; top: 2px; padding: 0 2px;
                   background: transparent; color: {INK}; }}
QGroupBox QWidget {{ background: {PANEL}; }}
QLabel[role="hint"] {{ color: {MUTED}; font-weight: 400; }}
QLabel[role="readout"] {{ font-size: 15px; font-weight: 600; }}
QComboBox, QDoubleSpinBox, QSpinBox {{ background: {PANEL}; border: 1px solid {RULE};
            border-radius: 4px; padding: 3px 6px; min-height: 22px; font-weight: 400; }}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: {BLUE}; }}
QCheckBox {{ font-weight: 400; }}
QPushButton {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 5px;
              padding: 6px 14px; font-weight: 600; }}
QPushButton:hover {{ border-color: {INK}; }}
QPushButton:disabled {{ color: #9aa4ae; }}
QPushButton[role="primary"] {{ background: {INK}; color: #ffffff; border-color: {INK}; }}
QPushButton[role="primary"]:disabled {{ background: #8b96a1; border-color: #8b96a1; }}
QListWidget {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 6px;
              font-weight: 400; }}
QProgressBar {{ border: 1px solid {RULE}; border-radius: 4px; background: {PANEL};
               height: 10px; text-align: center; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}
QStatusBar {{ color: {MUTED}; }}
"""


def describe_event(ev: dict, tiers=None) -> str:
    kind = ev["type"]
    t = f"t = {ev['t']:.3f}"

    def grid(i):
        return " × ".join(map(str, tiers[i])) if tiers and i < len(tiers) else f"tier {i}"

    if kind == "rollback_promote":
        return (f"t = {ev['t_detect']:.3f}  Rewound to t = {ev['t_restore']:.3f} and continued on "
                f"{grid(ev['to_tier'])} ({ev['reason']}); {ev['steps_discarded']} steps redone")
    if kind == "promote":
        return f"{t}  Refined to {grid(ev['to_tier'])} in place"
    if kind == "initial_promote":
        return (f"{t}  The initial field is too sharp for {grid(ev['from_tier'])}: "
                f"started on {grid(ev['to_tier'])} instead")
    if kind == "demote":
        return f"{t}  Coarsened to {grid(ev['to_tier'])}: the fine scales had faded"
    if kind == "saturated":
        return f"{t}  From here the finest grid is under-resolved (results are flagged)"
    if kind == "resolved":
        return f"{t}  The finest grid is resolving the flow again"
    if kind == "sgs_on":
        return f"{t}  Sub-grid model switched on for the finest grid"
    if kind == "rollback_retry":
        return f"{t}  Fault on the finest grid: rewound and retried with CFL {ev['cfl']:.3g}"
    if kind == "rollback_local_refine":
        vol = (ev.get("region") or {}).get("volume_fraction")
        where = f"{vol:.0%} of the box" if vol is not None else "a local region"
        return (f"t = {ev['t_detect']:.3f}  Rewound to t = {ev['t_restore']:.3f} and replayed "
                f"{where} on a {ev.get('refine', 2)}× finer local patch (experimental)")
    if kind == "local_retire":
        return f"{t}  Local patch retired: the fine scales had faded"
    if kind == "local_fallback":
        return f"{t}  Local patch not used ({ev.get('reason', 'unsuitable')}); refining the whole box"
    return f"{t}  {kind}"


class SimWorker(QtCore.QObject):
    """Owns the Simulation and advances it on a background thread."""

    frame = QtCore.Signal(object)       # dict: field, snapshot, history, events
    failed = QtCore.Signal(str)
    finished = QtCore.Signal(object)    # summary dict

    def __init__(self, cfg: dict, field_name: str):
        super().__init__()
        self.cfg = cfg
        self.field_name = field_name
        self.plane_index = None
        self._pause = False
        self._stop = False
        self.sim: Simulation | None = None
        self.frame_interval = 0.06

    @QtCore.Slot()
    def run(self):
        try:
            self.sim = Simulation(self.cfg)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        sim = self.sim
        self._emit_frame()
        while not self._stop and not sim.done:
            if self._pause:
                time.sleep(0.03)
                continue
            try:
                sim.advance(max_wall=self.frame_interval)
            except Exception as exc:        # programming error, not a solver fault
                self.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            self._emit_frame()
        if sim.status == "failed":
            self.failed.emit(sim.error or "simulation failed")
        self._emit_frame()
        self.finished.emit(sim.summary())

    def _emit_frame(self):
        sim = self.sim
        try:
            fld = sim.field(self.field_name)
        except Exception:
            fld = sim.field()
        mask = getattr(sim.solver, "body_mask", None)
        h = sim.history
        self.frame.emit({
            "field": fld, "field_name": self.field_name,
            "mask": None if mask is None else sim.be.to_numpy(mask),
            "snapshot": sim.snapshot(),
            "history": {k: np.asarray(h[k], dtype=float)
                        for k in ("t", "tier", "pressure", "dissipation", "energy", "enstrophy")},
            "events": list(sim.events),
            "cost": sim.ctrl.cost_summary(),
            "lengths": sim.solver.grid.lengths,
        })

    def set_paused(self, paused: bool):
        self._pause = paused

    def stop(self):
        self._stop = True


class ValidationWorker(QtCore.QObject):
    result = QtCore.Signal(object)
    done = QtCore.Signal()

    def __init__(self, device, precision):
        super().__init__()
        self.device, self.precision = device, precision

    @QtCore.Slot()
    def run(self):
        from omnisvox_cfd.validation import run_validation

        try:
            run_validation(get_backend(self.device, self.precision), on_result=self.result.emit)
        except Exception as exc:
            self.result.emit({"name": "Validation could not start", "passed": False,
                              "value": float("nan"), "tolerance": None, "seconds": 0,
                              "note": str(exc)})
        self.done.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"OmnisVox CFD {__version__}")
        self.resize(1360, 860)
        self.thread: QtCore.QThread | None = None
        self.worker: SimWorker | None = None
        self.last = None            # last frame payload
        self.limits = None
        self.summary = None
        self._param_widgets: dict[str, QtWidgets.QWidget] = {}
        self._build_ui()
        self._build_menu()
        self._on_case_changed()

    # ---- construction ------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 6)
        root.setSpacing(12)

        # -- left: setup ------------------------------------------------
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 10, 0)
        lv.setSpacing(4)

        box = QtWidgets.QGroupBox("Flow")
        f = QtWidgets.QFormLayout(box)
        self.case_combo = QtWidgets.QComboBox()
        for key, case in CASES.items():
            self.case_combo.addItem(case.title, key)
        self.case_combo.currentIndexChanged.connect(self._on_case_changed)
        f.addRow(self.case_combo)
        self.case_hint = QtWidgets.QLabel()
        self.case_hint.setWordWrap(True)
        self.case_hint.setProperty("role", "hint")
        f.addRow(self.case_hint)
        self.param_form = QtWidgets.QFormLayout()
        f.addRow(self.param_form)
        self.t_end = QtWidgets.QDoubleSpinBox(decimals=3, minimum=1e-3, maximum=1e6)
        f.addRow("End time", self.t_end)
        lv.addWidget(box)

        box = QtWidgets.QGroupBox("Resolution")
        f = QtWidgets.QFormLayout(box)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._refresh_tier_combo)
        f.addRow("Grid tiers", self.preset_combo)
        self.adaptive = QtWidgets.QCheckBox("Adapt resolution with rollback")
        self.adaptive.setChecked(True)
        self.adaptive.toggled.connect(self._on_adaptive_toggled)
        f.addRow(self.adaptive)
        self.tier_combo = QtWidgets.QComboBox()
        self.tier_label = QtWidgets.QLabel("Start on")
        f.addRow(self.tier_label, self.tier_combo)
        self.accuracy = QtWidgets.QComboBox()
        self.accuracy.addItems(list(ACCURACY))
        self.accuracy.setCurrentText("Standard")
        self.accuracy.setToolTip("How early the grid refines. Draft tolerates about 4x the "
                                 "unresolved share of Standard; Strict about a third.")
        f.addRow("Accuracy", self.accuracy)
        self.rollback = QtWidgets.QCheckBox("Rewind and redo when refining")
        self.rollback.setChecked(True)
        self.rollback.setToolTip("Off: refine in place and keep the under-resolved stretch.")
        f.addRow(self.rollback)
        self.saturation = QtWidgets.QComboBox()
        self.saturation.addItem("Keep going and flag it", "continue")
        self.saturation.addItem("Switch on sub-grid model (3-D)", "sgs")
        self.saturation.addItem("Stop the run", "stop")
        self.saturation.setToolTip("The sub-grid model (plain Smagorinsky) keeps a run smooth but is "
                                   "too dissipative for transitional flows. See docs/METHOD.md.")
        f.addRow("If finest grid\nis not enough", self.saturation)
        lv.addWidget(box)

        box = QtWidgets.QGroupBox("Compute")
        f = QtWidgets.QFormLayout(box)
        self.device = QtWidgets.QComboBox()
        st = gpu_status()
        self.device.addItem("CPU", "cpu")
        if st["available"]:
            self.device.addItem(f"GPU: {st['name']}", "gpu")
            self.device.setCurrentIndex(1)
        else:
            self.device.addItem("GPU (not available)", "gpu")
            self.device.model().item(1).setEnabled(False)
            self.device.setToolTip(st["reason"])
        f.addRow("Device", self.device)
        self.precision = QtWidgets.QComboBox()
        self.precision.addItem("Double", "double")
        self.precision.addItem("Single (faster on GPU)", "single")
        f.addRow("Precision", self.precision)
        lv.addWidget(box)

        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start run")
        self.start_btn.setProperty("role", "primary")
        self.start_btn.clicked.connect(self.start_run)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_run)
        for b in (self.start_btn, self.pause_btn, self.stop_btn):
            row.addWidget(b)
        lv.addSpacing(8)
        lv.addLayout(row)
        lv.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        # -- centre: field ---------------------------------------------
        mid = QtWidgets.QVBoxLayout()
        mid.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        self.readout = QtWidgets.QLabel("Set up a flow and start the run")
        self.readout.setProperty("role", "readout")
        top.addWidget(self.readout, 1)
        top.addWidget(QtWidgets.QLabel("Show"))
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        top.addWidget(self.field_combo)
        mid.addLayout(top)

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background=PANEL,
                            foreground=INK)
        self.view = pg.GraphicsLayoutWidget()
        self.vb = self.view.addViewBox(lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.image = pg.ImageItem()
        self.vb.addItem(self.image)
        mid.addWidget(self.view, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        mid.addWidget(self.progress)
        self.subreadout = QtWidgets.QLabel("")
        self.subreadout.setProperty("role", "hint")
        mid.addWidget(self.subreadout)
        root.addLayout(mid, 5)

        # -- right: adaptive-resolution record -------------------------
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setMinimumWidth(380)

        def make_plot(row, title, log_y=False):
            p = self.plots.addPlot(row=row, col=0)
            p.setTitle(title, color=INK, size="10pt")
            p.showGrid(x=False, y=True, alpha=0.15)
            p.setMenuEnabled(False)
            p.setLogMode(y=log_y)
            for ax in ("left", "bottom"):
                p.getAxis(ax).setPen(pg.mkPen(RULE))
                p.getAxis(ax).setTextPen(pg.mkPen(MUTED))
            return p

        self.p_tier = make_plot(0, "Grid in use")
        self.p_press = make_plot(1, "Unresolved share (saturation pressure)", log_y=True)
        self.p_diss = make_plot(2, "Dissipation rate")
        self.p_diss.setLabel("bottom", "time")
        self.p_press.setXLink(self.p_tier)
        self.p_diss.setXLink(self.p_tier)
        self.c_tier = self.p_tier.plot(pen=pg.mkPen(BLUE, width=2), stepMode="right")
        self.s_roll = pg.ScatterPlotItem(size=10, symbol="t", brush=pg.mkBrush(ORANGE),
                                         pen=pg.mkPen(PANEL, width=1.5))
        self.s_down = pg.ScatterPlotItem(size=10, symbol="t1", brush=pg.mkBrush(AQUA),
                                         pen=pg.mkPen(PANEL, width=1.5))
        self.p_tier.addItem(self.s_roll)
        self.p_tier.addItem(self.s_down)
        self.c_press = self.p_press.plot(pen=pg.mkPen(BLUE, width=2))
        self.l_up = pg.InfiniteLine(angle=0, pen=pg.mkPen(ORANGE, width=1, style=QtCore.Qt.DashLine))
        self.l_dn = pg.InfiniteLine(angle=0, pen=pg.mkPen(AQUA, width=1, style=QtCore.Qt.DashLine))
        self.p_press.addItem(self.l_up)
        self.p_press.addItem(self.l_dn)
        self.c_diss = self.p_diss.plot(pen=pg.mkPen(BLUE, width=2))
        right.addWidget(self.plots, 3)

        legend = QtWidgets.QLabel(
            f"<span style='color:{ORANGE}'>&#9650;</span> rolled back and refined &nbsp; "
            f"<span style='color:{AQUA}'>&#9660;</span> coarsened &nbsp; "
            f"<span style='color:{ORANGE}'>- -</span> refine above &nbsp; "
            f"<span style='color:{AQUA}'>- -</span> coarsen below")
        legend.setProperty("role", "hint")
        right.addWidget(legend)
        self.cost_label = QtWidgets.QLabel("")
        self.cost_label.setWordWrap(True)
        right.addWidget(self.cost_label)
        self.events = QtWidgets.QListWidget()
        self.events.setWordWrap(True)
        self.events.setMinimumHeight(120)
        right.addWidget(self.events, 2)
        root.addLayout(right, 4)

        self.statusBar().showMessage(get_backend("cpu").describe())

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for label, fn in (("Save image (PNG)…", self.save_png),
                          ("Export fields (NPZ)…", lambda: self.export("npz")),
                          ("Export history (CSV)…", lambda: self.export("csv")),
                          ("Export volume (VTK)…", lambda: self.export("vtk")),
                          ("Save run summary (JSON)…", lambda: self.export("json")),
                          (None, None),
                          ("Save setup…", self.save_setup), ("Load setup…", self.load_setup),
                          (None, None), ("Quit", self.close)):
            if label is None:
                m.addSeparator()
            else:
                m.addAction(label, fn)
        t = self.menuBar().addMenu("&Tools")
        t.addAction("Run validation suite…", self.run_validation)
        h = self.menuBar().addMenu("&Help")
        h.addAction("About", self.about)

    # ---- setup panel -------------------------------------------------
    def current_case(self):
        return CASES[self.case_combo.currentData()]

    def _on_case_changed(self):
        case = self.current_case()
        self.case_hint.setText(case.summary)
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._param_widgets.clear()
        for p in case.params:
            if p.choices is not None:
                w = QtWidgets.QComboBox()
                w.addItems([str(c) for c in p.choices])
                w.setCurrentText(str(p.default))
            elif isinstance(p.default, int):
                w = QtWidgets.QSpinBox()
                w.setRange(int(p.min), int(p.max))
                w.setValue(p.default)
            else:
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(1 if abs(p.default) >= 10 else 3)
                w.setRange(p.min, p.max)
                w.setValue(p.default)
                w.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
            if p.help:
                w.setToolTip(p.help)
            self._param_widgets[p.name] = w
            self.param_form.addRow(p.label.split(" (")[0], w)
        self.t_end.setValue(case.t_end)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for name in case.tier_presets:
            self.preset_combo.addItem(name.replace("^2", "²").replace("^3", "³"), name)
        self.preset_combo.setCurrentIndex(list(case.tier_presets).index(case.default_preset))
        self.preset_combo.blockSignals(False)
        self._refresh_tier_combo()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for name in case.describe()["fields"]:
            self.field_combo.addItem(name.replace("_", " "), name)
        self.field_combo.setCurrentIndex(self.field_combo.findData(case.default_field))
        self.field_combo.blockSignals(False)
        self.saturation.model().item(1).setEnabled(case.ndim == 3)
        if case.ndim != 3 and self.saturation.currentData() == "sgs":
            self.saturation.setCurrentIndex(0)

    def _refresh_tier_combo(self):
        case = self.current_case()
        tiers = case.tier_presets.get(self.preset_combo.currentData(), [])
        self.tier_combo.clear()
        for i, s in enumerate(tiers):
            self.tier_combo.addItem(" × ".join(map(str, s)), i)
        # Turbulent starts begin on the finest grid and coarsen as they decay (engine 1.1.2).
        start_fine = getattr(case, "start_fine", False)
        self.tier_combo.setCurrentIndex(len(tiers) - 1 if (start_fine or not self.adaptive.isChecked()) else 0)

    def _on_adaptive_toggled(self, on):
        self.tier_label.setText("Start on" if on else "Fixed grid")
        for w in (self.accuracy, self.rollback, self.saturation):
            w.setEnabled(on)
        self._refresh_tier_combo()

    def build_config(self) -> dict:
        case = self.current_case()
        params = {}
        for p in case.params:
            w = self._param_widgets[p.name]
            params[p.name] = (w.currentText() if isinstance(w, QtWidgets.QComboBox) else w.value())
        arr = {"enabled": self.adaptive.isChecked(), "start_tier": self.tier_combo.currentData() or 0,
               "press_up": case.press_up * ACCURACY[self.accuracy.currentText()],
               "rollback": self.rollback.isChecked(),
               "on_saturation": self.saturation.currentData()}
        return {"case": case.key, "params": params, "preset": self.preset_combo.currentData(),
                "t_end": self.t_end.value(), "device": self.device.currentData(),
                "precision": self.precision.currentData(), "arr": arr}

    def apply_config(self, cfg: dict):
        idx = self.case_combo.findData(cfg.get("case"))
        if idx < 0:
            raise ValueError(f"unknown case {cfg.get('case')!r}")
        self.case_combo.setCurrentIndex(idx)
        self._on_case_changed()
        for name, value in (cfg.get("params") or {}).items():
            w = self._param_widgets.get(name)
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentText(str(value))
            elif w is not None:
                w.setValue(value)
        if cfg.get("preset"):
            self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(cfg["preset"])))
        if cfg.get("t_end"):
            self.t_end.setValue(cfg["t_end"])
        arr = cfg.get("arr") or {}
        self.adaptive.setChecked(arr.get("enabled", True))
        self.rollback.setChecked(arr.get("rollback", True))
        if "start_tier" in arr:
            self.tier_combo.setCurrentIndex(arr["start_tier"])
        if "on_saturation" in arr:
            self.saturation.setCurrentIndex(max(0, self.saturation.findData(arr["on_saturation"])))
        if "press_up" in arr:
            ratio = arr["press_up"] / self.current_case().press_up
            best = min(ACCURACY, key=lambda k: abs(np.log(ACCURACY[k] / ratio)))
            self.accuracy.setCurrentText(best)
        self.precision.setCurrentIndex(max(0, self.precision.findData(cfg.get("precision", "double"))))

    # ---- run control -------------------------------------------------
    def start_run(self):
        self.stop_run()
        cfg = self.build_config()
        self.limits = None
        self.summary = None
        self.events.clear()
        self._n_events = 0
        self.cost_label.setText("")
        self.readout.setText("Preparing the grid…")
        self.l_up.setValue(np.log10(cfg["arr"]["press_up"]))
        self.l_dn.setValue(np.log10(cfg["arr"]["press_up"] * 0.2))
        self.l_up.setVisible(cfg["arr"]["enabled"])
        self.l_dn.setVisible(cfg["arr"]["enabled"])
        self.thread = QtCore.QThread(self)
        self.worker = SimWorker(cfg, self.field_combo.currentData())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame.connect(self.on_frame)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()
        self.start_btn.setText("Restart")
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)

    def toggle_pause(self):
        if not self.worker:
            return
        paused = self.pause_btn.text() == "Pause"
        self.worker.set_paused(paused)
        self.pause_btn.setText("Resume" if paused else "Pause")

    def stop_run(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait(10000)
        self.thread = None
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _on_field_changed(self):
        self.limits = None
        if self.worker:
            self.worker.field_name = self.field_combo.currentData()

    # ---- incoming data -----------------------------------------------
    @QtCore.Slot(object)
    def on_frame(self, payload):
        self.last = payload
        snap, h = payload["snapshot"], payload["history"]
        fld = payload["field"]
        symmetric = payload["field_name"] in SIGNED_FIELDS
        lo, hi = field_limits(fld, symmetric)
        if self.limits is None:
            self.limits = (lo, hi)
        else:                               # ease the colour scale to avoid flicker
            a = 0.15
            self.limits = (self.limits[0] + a * (lo - self.limits[0]),
                           self.limits[1] + a * (hi - self.limits[1]))
        rgb = field_to_rgb(fld, symmetric, self.limits, payload["mask"])
        lx, ly = payload["lengths"][0], payload["lengths"][1]
        self.image.setImage(rgb, autoLevels=False)
        self.image.setRect(QtCore.QRectF(0, 0, lx, ly))
        self.vb.setRange(QtCore.QRectF(0, 0, lx, ly), padding=0.01)

        shape = " × ".join(map(str, snap["shape"]))
        self.readout.setText(f"t = {snap['t']:.3f} of {snap['t_end']:g}    grid {shape}")
        rate = snap["steps"] / snap["wall"] if snap["wall"] > 0 else 0.0
        flag = "   finest grid under-resolved" if snap["saturated"] else ""
        self.subreadout.setText(f"{snap['steps']} steps, {rate:.1f} steps/s, "
                                f"{snap['wall']:.0f} s elapsed{flag}")
        self.progress.setValue(int(1000 * snap["progress"]))

        t = h["t"]
        if t.size:
            self.c_tier.setData(t, h["tier"])
            p = np.maximum(h["pressure"], 1e-6)
            self.c_press.setData(t, p)
            self.c_diss.setData(t, h["dissipation"])
            n = int(np.nanmax(h["tier"])) if h["tier"].size else 0
            tiers = self.worker.sim.tiers if self.worker and self.worker.sim else []
            ticks = [(i, "×".join(map(str, s))) for i, s in enumerate(tiers)]
            self.p_tier.getAxis("left").setTicks([ticks])
            self.p_tier.setYRange(-0.3, max(len(tiers) - 1, n) + 0.3, padding=0)
        evs = payload["events"]
        ups = [(e["t"], e["to_tier"]) for e in evs if e["type"] in ("rollback_promote", "promote")]
        downs = [(e["t"], e["to_tier"]) for e in evs if e["type"] == "demote"]
        self.s_roll.setData(pos=np.array(ups).reshape(-1, 2))
        self.s_down.setData(pos=np.array(downs).reshape(-1, 2))
        for e in evs[self._n_events:]:
            self.events.addItem(describe_event(e, self.worker.sim.tiers if self.worker and self.worker.sim else None))
        if len(evs) != self._n_events:
            self.events.scrollToBottom()
        self._n_events = len(evs)
        c = payload["cost"]
        self.cost_label.setText(
            f"{c['cell_step_saving_est']:.2f}× fewer cell updates than the finest grid throughout "
            f"(estimate).  {c['rollbacks']} rollback(s), {c['wasted']['steps']} steps redone, "
            f"safety limiter fired {c['guard_hits']}×.")

    @QtCore.Slot(str)
    def on_failed(self, message):
        self.readout.setText("Run stopped")
        self.events.addItem("Stopped: " + message)
        self.events.scrollToBottom()
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    @QtCore.Slot(object)
    def on_finished(self, summary):
        self.summary = summary
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        if summary["status"] == "done":
            sat = summary["time_saturated"]
            note = (f"; {sat:.3g} time units flagged under-resolved on the finest grid"
                    if sat > 0 else "")
            self.events.addItem(f"Finished in {summary['wall_seconds']:.1f} s{note}")
            self.events.scrollToBottom()

    # ---- exports -----------------------------------------------------
    def _need_run(self):
        if not (self.worker and self.worker.sim):
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Start a run first.")
            return None
        if self.thread and self.thread.isRunning() and self.pause_btn.text() == "Pause":
            self.toggle_pause()
            time.sleep(0.15)
        return self.worker.sim

    def save_png(self):
        if self.last is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Start a run first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "flow.png", "PNG (*.png)")
        if path:
            p = self.last
            rgb = field_to_rgb(p["field"], p["field_name"] in SIGNED_FIELDS, self.limits, p["mask"])
            Path(path).write_bytes(encode_png(rgb))

    def export(self, kind):
        sim = self._need_run()
        if sim is None:
            return
        flt = {"npz": "NumPy archive (*.npz)", "csv": "CSV (*.csv)", "vtk": "VTK (*.vtk)",
               "json": "JSON (*.json)"}[kind]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", f"run.{kind}", flt)
        if not path:
            return
        try:
            if kind == "npz":
                sim.export_npz(path)
            elif kind == "csv":
                sim.export_csv(path)
            elif kind == "vtk":
                sim.export_vtk(path)
            else:
                Path(path).write_text(json.dumps(json_safe(sim.summary()), indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    def save_setup(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save setup", "setup.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.build_config(), indent=2))

    def load_setup(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load setup", "", "JSON (*.json)")
        if path:
            try:
                self.apply_config(json.loads(Path(path).read_text()))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not load setup", str(exc))

    # ---- tools ---------------------------------------------------------
    def run_validation(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Validation suite")
        dlg.resize(720, 480)
        v = QtWidgets.QVBoxLayout(dlg)
        head = QtWidgets.QLabel(f"Running on {self.device.currentText()}, "
                                f"{self.precision.currentData()} precision…")
        v.addWidget(head)
        lst = QtWidgets.QListWidget()
        lst.setWordWrap(True)
        v.addWidget(lst)
        counts = {"ok": 0, "n": 0}
        thread = QtCore.QThread(dlg)
        worker = ValidationWorker(self.device.currentData(), self.precision.currentData())
        worker.moveToThread(thread)

        def on_result(r):
            counts["n"] += 1
            counts["ok"] += bool(r["passed"])
            item = QtWidgets.QListWidgetItem(
                f"{'Passed' if r['passed'] else 'FAILED'}   {r['name']}\n"
                f"      value {r['value']:.3g}"
                + (f", tolerance {r['tolerance']:g}" if r.get("tolerance") else "")
                + (f".  {r['note']}" if r.get("note") else ""))
            if not r["passed"]:
                item.setForeground(QtGui.QBrush(QtGui.QColor("#b3261e")))
            lst.addItem(item)
            lst.scrollToBottom()

        worker.result.connect(on_result)
        worker.done.connect(lambda: head.setText(f"{counts['ok']} of {counts['n']} checks passed"))
        worker.done.connect(thread.quit)
        thread.started.connect(worker.run)
        dlg._keep = (thread, worker)
        thread.start()
        dlg.finished.connect(lambda *_: thread.wait(60000))
        dlg.exec()

    def about(self):
        st = gpu_status()
        gpu = f"{st['name']} ({st['memory_gb']} GB)" if st["available"] else f"not available: {st['reason']}"
        QtWidgets.QMessageBox.about(
            self, "About OmnisVox CFD",
            f"<b>OmnisVox CFD {__version__}</b><br>Pseudo-spectral Navier–Stokes solver with "
            f"adaptive rollback-refinement.<br><br>CPU: {get_backend('cpu').describe()}<br>GPU: {gpu}")

    def closeEvent(self, ev):
        self.stop_run()
        super().closeEvent(ev)


def main(argv=None):
    app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("OmnisVox CFD")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
