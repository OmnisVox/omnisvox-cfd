"""Entry point for the frozen OmnisVox.exe (PyInstaller).

    OmnisVox.exe              opens the app
    OmnisVox.exe --selftest   builds the main window, runs a tiny simulation, exits
"""
import os
import sys


def selftest() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from omnisvox_cfd import Simulation
    from omnisvox_desktop.app import MainWindow, STYLE

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.close()
    sim = Simulation({"case": "taylor_green_3d", "params": {"Re": 400.0},
                      "preset": "16^3 / 32^3 / 64^3", "t_end": 0.5, "device": "cpu"}).run()
    ok = sim.status == "done"
    print("OmnisVox selftest:", "OK" if ok else f"FAILED ({sim.error})")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from omnisvox_desktop.app import main
    sys.exit(main())
