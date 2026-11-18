"""Headless smoke test of the desktop UI (needs PySide6 + pyqtgraph)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")


@pytest.fixture(scope="module")
def app():
    from omnisvox_desktop.app import STYLE

    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    a.setStyleSheet(STYLE)
    return a


def _pump(app, seconds, until=None):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        if until and until():
            return True
        time.sleep(0.01)
    return False


def test_run_to_completion_and_export(app, tmp_path):
    from omnisvox_desktop.app import MainWindow

    win = MainWindow()
    win.show()
    win.apply_config({"case": "taylor_green_3d", "params": {"Re": 400.0},
                      "preset": "16^3 / 32^3 / 64^3", "t_end": 3.5})
    cfg = win.build_config()
    assert cfg["case"] == "taylor_green_3d" and cfg["t_end"] == 3.5
    win.start_run()
    assert _pump(app, 120, lambda: win.summary is not None), "run did not finish"
    assert win.summary["status"] == "done"
    assert win.events.count() >= 2                    # at least one adaptation + "Finished"
    assert win.last["field"].ndim == 2
    sim = win.worker.sim
    sim.export_csv(tmp_path / "h.csv")
    assert (tmp_path / "h.csv").stat().st_size > 100
    shot = os.environ.get("OV_SCREENSHOT")
    if shot:
        win.grab().save(shot)
    win.close()


def test_every_case_builds_a_valid_config(app):
    from omnisvox_cfd import CASES, resolve_config
    from omnisvox_desktop.app import MainWindow

    win = MainWindow()
    for i, key in enumerate(CASES):
        win.case_combo.setCurrentIndex(i)
        cfg = win.build_config()
        assert cfg["case"] == key
        resolve_config(cfg)
        win.adaptive.setChecked(False)
        resolve_config(win.build_config())
        win.adaptive.setChecked(True)
    win.close()


def test_bad_run_reports_instead_of_crashing(app):
    from omnisvox_desktop.app import MainWindow

    win = MainWindow()
    win.device.setCurrentIndex(1)        # GPU entry; unavailable on CI machines
    if win.device.currentData() == "gpu":
        from omnisvox_cfd import gpu_status
        if not gpu_status()["available"]:
            win.start_run()
            assert _pump(app, 20, lambda: win.readout.text() == "Run stopped")
            assert "GPU" in win.events.item(win.events.count() - 1).text()
    win.close()
