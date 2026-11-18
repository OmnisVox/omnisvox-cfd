# Changelog (desktop edition)

## 1.1.2 — 2026-11-17

Engine level with server edition 1.1.2.

- **Fixed:** ARR no longer steps down while a cascade is still building
  (it demoted 256² → 128² at t = 0.057 in 2-D turbulence and had to come back).
- **Changed:** decaying-turbulence cases (2-D and 3-D) start on the finest grid
  and coarsen as they decay; the app's "Start on" default follows. On the A100
  benchmark this turned 2-D turbulence from a loss into plain-512² accuracy at
  about half its cost. See `docs/METHOD.md`, "Demotion" and "Flows that start
  turbulent".

## 1.1.1 — 2026-11-17

Engine brought level with the server edition 1.1.1.

### Added
- `OmnisVox.bat`: double-click launcher; first run sets up a private Python
  environment in `.venv`, later runs open the app directly.
- `Build-OmnisVox-EXE.bat` + `packaging/launcher.py`: builds a standalone
  `OmnisVox.exe` (PyInstaller, CPU) and verifies it with `--selftest`.
- Experimental spatial 3-D adaptive refinement (`arr.local_3d`, default **off**;
  `arr.local_scheme` = `"spectral"` default, `"fd"` legacy). See
  `docs/METHOD.md` § "Spatial 3-D rollback path" and
  `benchmarks/LOCAL_ARR_SMOKE.md`.
- Event descriptions in the app for `rollback_local_refine`, `local_retire`,
  `local_fallback`.
- Benchmarks: `local_arr_smoke.py`, `local_arr_localized.py`.
- Engine tests for local refinement (51 engine tests).

### Unchanged
- The app does not expose spatial refinement; with `local_3d` off (default),
  runs follow the 1.0 whole-domain path.
- Solvers, cases, exports, CLI, GPU path.

### Note
- The cost summary now counts local-patch rollbacks/retirements in
  `rollbacks` / `promotions` / `demotions` (zero when local refinement is off).
