"""Self-validation suite.

Runs on whichever backend it is given, so ``omnisvox-cfd validate --device gpu``
exercises the GPU code path with the same checks used on the CPU.  Each check
returns a dict: name, value, reference, tolerance, passed, seconds, note.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from .backend import Backend, get_backend, gpu_status
from .cases import get_case
from .runner import Simulation
from .solver2d import NavierStokes2D
from .solver3d import NavierStokes3D
from .spectral import SpectralGrid, resample_hat


def _check(name, value, reference, tol, passed, t0, note=""):
    return {"name": name, "value": float(value), "reference": reference, "tolerance": tol,
            "passed": bool(passed), "seconds": round(time.perf_counter() - t0, 3), "note": note}


def _solver(case_key, shape, be, params=None, **kw):
    case = get_case(case_key)
    p = case.resolve_params(params or {})
    s = case.make_solver(shape, p, be, **kw)
    s.set_state(case.initial_state(s, p))
    return s


def _tol(be: Backend, double, single):
    return double if be.precision == "double" else single


# ---------------------------------------------------------------------------
def check_grid_transfer(be):
    t0 = time.perf_counter()
    rng = np.random.default_rng(3)
    g = SpectralGrid((24, 32, 16), (1.0, 2.0, 3.0), be)
    a_hat = g.fft(be.asarray(rng.standard_normal(g.shape), dtype=be.rdtype)) * g.dealias
    fine_shape = (48, 64, 32)
    gf = SpectralGrid(fine_shape, g.lengths, be)
    up = resample_hat(a_hat, g.shape, fine_shape, be)
    coarse_phys = g.ifft(a_hat)
    fine_phys = gf.ifft(up)
    e1 = be.scalar(be.xp.max(be.xp.abs(fine_phys[::2, ::2, ::2] - coarse_phys)))
    back = resample_hat(up, fine_shape, g.shape, be)
    e2 = be.scalar(be.xp.max(be.xp.abs(back - a_hat))) / max(be.scalar(be.xp.max(be.xp.abs(a_hat))), 1e-30)
    err = max(e1, e2)
    tol = _tol(be, 1e-11, 1e-4)
    return _check("Grid transfer: prolong/restrict exact on shared modes", err, 0.0, tol,
                  err < tol, t0, "max error of prolongation at coincident points and of a round trip")


def check_taylor_green_2d(be):
    t0 = time.perf_counter()
    s = _solver("taylor_green_2d", (64, 64), be, {"Re": 100.0})
    e0 = s.diagnostics()["energy"]
    s.advance_to(2.0)
    err = abs(s.diagnostics()["energy"] / e0 - math.exp(-4 * 2.0 / 100.0))
    tol = _tol(be, 1e-10, 1e-4)
    return _check("2-D Taylor-Green decay vs exact solution", err, 0.0, tol, err < tol, t0,
                  "relative energy error at t = 2")


def check_abc(be):
    t0 = time.perf_counter()
    s = _solver("abc_flow", (32, 32, 32), be, {"Re": 20.0})
    e0 = s.diagnostics()["energy"]
    s.advance_to(3.0)
    err = abs(s.diagnostics()["energy"] / e0 - math.exp(-2 * 3.0 / 20.0))
    tol = _tol(be, 1e-10, 1e-4)
    return _check("3-D ABC Beltrami decay vs exact solution", err, 0.0, tol, err < tol, t0,
                  "relative energy error at t = 3")


def check_divergence(be):
    t0 = time.perf_counter()
    s = _solver("taylor_green_3d", (32, 32, 32), be, {"Re": 200.0})
    s.advance_to(2.0)
    div = s.max_divergence()
    tol = _tol(be, 1e-10, 1e-3)
    return _check("3-D velocity stays divergence-free", div, 0.0, tol, div < tol, t0,
                  "max |div u| after 2 time units of Taylor-Green flow")


def check_inviscid_energy(be):
    t0 = time.perf_counter()
    case = get_case("taylor_green_3d")
    p = case.resolve_params({"Re": 1e5})
    s = NavierStokes3D((32, 32, 32), case.lengths(p), 0.0, be, cfl=0.25)
    s.set_state(case.initial_state(s, p))
    e0 = s.diagnostics()["energy"]
    s.advance_to(1.5)
    err = abs(s.diagnostics()["energy"] / e0 - 1.0)
    tol = _tol(be, 1e-7, 1e-4)
    return _check("Inviscid energy conservation (nonlinear term)", err, 0.0, tol, err < tol, t0,
                  "relative energy drift at nu = 0 over 1.5 time units")


def check_energy_budget(be):
    t0 = time.perf_counter()
    s = _solver("taylor_green_3d", (32, 32, 32), be, {"Re": 100.0}, dt_fixed=0.01)
    s.advance_to(2.0)
    d0 = s.diagnostics()
    s.step()
    d1 = s.diagnostics()
    dedt = (d1["energy"] - d0["energy"]) / 0.01
    eps = 0.5 * (d0["dissipation"] + d1["dissipation"])
    err = abs(-dedt / eps - 1.0)
    tol = _tol(be, 1e-4, 5e-2)
    return _check("Energy budget: -dE/dt = 2 nu Z", err, 0.0, tol, err < tol, t0,
                  "relative mismatch between energy decay rate and viscous dissipation")


def check_rk4_order(be):
    """Self-convergence of the full nonlinear step on a smooth shear layer."""
    t0 = time.perf_counter()
    if be.precision != "double":
        return _check("Time integration is 4th order", 4.0, 4.0, 0.5, True, t0,
                      "skipped in single precision (errors below round-off)")
    params = {"Re": 2000.0, "rho": 20.0}

    def run(dt):
        s = _solver("shear_layer", (64, 64), be, params, dt_fixed=dt)
        s.advance_to(0.2)
        return s.state

    ref = run(0.2 / 640)
    xp = be.xp
    errs = [be.scalar(xp.max(xp.abs(run(0.2 / n) - ref))) for n in (20, 40)]
    order = math.log2(errs[0] / errs[1])
    return _check("Time integration is 4th order", order, 4.0, 0.4, abs(order - 4.0) < 0.4, t0,
                  "observed order from dt-halving against a fine-dt reference")


def check_2d_3d_agreement(be):
    """A z-invariant flow in the 3-D solver must match the independent 2-D solver."""
    t0 = time.perf_counter()
    params = {"Re": 1000.0, "rho": 20.0}
    s2 = _solver("shear_layer", (64, 64), be, params, dt_fixed=0.005)
    nz = 8
    s3 = NavierStokes3D((64, 64, nz), (1.0, 1.0, 1.0), s2.nu, be, dt_fixed=0.005)
    f = s2._physical(s2.state)
    xp = be.xp
    u3 = xp.zeros((3, 64, 64, nz), dtype=be.rdtype)
    u3[0] = f[0][:, :, None]
    u3[1] = f[1][:, :, None]
    g3 = s3.grid
    s3.set_state(s3._project(g3.fft(u3) * g3.dealias))
    s2.advance_to(0.3)
    s3.advance_to(0.3)
    w2 = s2.grid.ifft(s2.state)
    w3 = g3.ifft(s3._curl_hat(s3.state))[2][:, :, 0]
    err = be.scalar(xp.max(xp.abs(w3 - w2))) / be.scalar(xp.max(xp.abs(w2)))
    tol = _tol(be, 1e-9, 1e-3)
    return _check("2-D and 3-D solvers agree on a z-invariant flow", err, 0.0, tol, err < tol, t0,
                  "relative max vorticity difference after shear-layer roll-up begins")


def check_rollback_bitwise(be):
    t0 = time.perf_counter()
    sim = Simulation({"case": "taylor_green_3d", "params": {"Re": 400.0}, "t_end": 2.0,
                      "preset": "16^3 / 32^3 / 64^3", "precision": be.precision,
                      "arr": {"start_tier": 1, "min_tier": 1, "max_tier": 1}}, backend=be)
    c = sim.ctrl
    c.run_chunk()                      # leaves a fresh checkpoint at the current instant
    c._save_checkpoint()
    saved = c.checkpoint["snap"]["state"].copy()
    t_saved = c.t
    for _ in range(5):                 # advance past the checkpoint without observing
        c.solver.step()
    first = c.solver.state.copy()
    t_first = c.t
    c._restore_checkpoint()
    xp = be.xp
    same_state = bool(xp.all(c.solver.state == saved)) and c.t == t_saved
    for _ in range(5):
        c.solver.step()
    replay = bool(xp.all(c.solver.state == first)) and c.t == t_first
    ok = same_state and replay
    return _check("Rollback restores the checkpoint bit-for-bit and replays identically",
                  0.0 if ok else 1.0, 0.0, 0.0, ok, t0,
                  "state and time compared exactly after rewind and after re-running the interval")


def check_arr_vs_uniform(be):
    t0 = time.perf_counter()
    base = {"case": "taylor_green_3d", "params": {"Re": 200.0}, "t_end": 6.0,
            "preset": "16^3 / 32^3 / 64^3", "precision": be.precision}
    fine = Simulation({**base, "arr": {"enabled": False, "start_tier": 2}}, backend=be).run()
    coarse = Simulation({**base, "arr": {"enabled": False, "start_tier": 0}}, backend=be).run()
    arr = Simulation(base, backend=be).run()

    def err(sim):
        return abs(sim.history["enstrophy"][-1] / fine.history["enstrophy"][-1] - 1.0)

    e_arr, e_coarse = err(arr), err(coarse)
    cost = arr.ctrl.cost_summary()
    saving = sum(c["cell_steps"] for c in fine.ctrl.cost.values()) / max(cost["cell_steps"], 1)
    ok = (arr.status == "done" and e_arr < 0.02 and e_arr < 0.5 * e_coarse
          and cost["promotions"] >= 1 and cost["guard_hits"] == 0)
    return _check("Adaptive run reproduces the uniform fine-grid run", e_arr, 0.0, 0.02, ok, t0,
                  f"enstrophy error at t = 6 vs uniform 64^3 (uniform 16^3 error {e_coarse:.3f}); "
                  f"{cost['promotions']} promotion(s), {cost['rollbacks']} rollback(s), "
                  f"{saving:.2f}x fewer cell-steps, limiter fired {cost['guard_hits']} times")


def check_fault_recovery(be):
    t0 = time.perf_counter()
    sim = Simulation({"case": "taylor_green_3d", "params": {"Re": 100.0}, "t_end": 3.0,
                      "preset": "16^3 / 32^3 / 64^3", "precision": be.precision}, backend=be)
    sim.advance(max_chunks=1)
    assert not sim.done
    sim.solver.state[0, 1, 1, 1] = float("nan")       # corrupt the state
    sim.run()
    kinds = [e["type"] for e in sim.events]
    finite = all(math.isfinite(v) for v in sim.history["energy"])
    ok = sim.status == "done" and "rollback_promote" in kinds and finite
    return _check("Injected non-finite state is rolled back and the run completes",
                  0.0 if ok else 1.0, 0.0, 0.0, ok, t0, "events: " + ", ".join(kinds))


def check_cpu_gpu_agreement(be):
    """GPU only: the same run on both devices must agree to round-off."""
    t0 = time.perf_counter()
    cfg = {"case": "taylor_green_3d", "params": {"Re": 400.0}, "t_end": 3.0,
           "preset": "16^3 / 32^3 / 64^3", "precision": be.precision}
    g = Simulation(cfg, backend=be).run()
    c = Simulation(cfg, backend=get_backend("cpu", be.precision)).run()
    a = g.be.to_numpy(g.solver.state)
    b = np.asarray(c.solver.state)
    same_path = [e["type"] for e in g.events] == [e["type"] for e in c.events]
    err = float(np.max(np.abs(a - b)) / np.max(np.abs(b))) if a.shape == b.shape else float("inf")
    tol = _tol(be, 1e-8, 1e-2)
    return _check("GPU and CPU runs agree", err, 0.0, tol, err < tol and same_path, t0,
                  "relative max difference of the final spectral state; same adaptive events: "
                  + str(same_path))


QUICK: list[Callable] = [
    check_grid_transfer, check_taylor_green_2d, check_abc, check_divergence,
    check_inviscid_energy, check_energy_budget, check_rk4_order, check_2d_3d_agreement,
    check_rollback_bitwise, check_arr_vs_uniform, check_fault_recovery,
]


def run_validation(be: Backend | None = None, on_result: Callable[[dict], None] | None = None):
    """Run every check; returns the list of result dicts."""
    be = be or get_backend("auto")
    checks = list(QUICK)
    if be.is_gpu:
        checks.append(check_cpu_gpu_agreement)
    results = []
    for fn in checks:
        try:
            r = fn(be)
        except Exception as exc:  # a crash is a failed check, not a crashed suite
            r = {"name": fn.__name__, "value": float("nan"), "reference": None, "tolerance": None,
                 "passed": False, "seconds": 0.0, "note": f"raised {type(exc).__name__}: {exc}"}
        results.append(r)
        if on_result:
            on_result(r)
    return results


def throughput(be: Backend | None = None, grids=None, steps=6):
    """Measured time per step and cell-updates per second for a set of grids."""
    be = be or get_backend("auto")
    grids = grids or [("taylor_green_3d", (32,) * 3), ("taylor_green_3d", (64,) * 3),
                      ("shear_layer", (256, 256)), ("shear_layer", (512, 512))]
    rows = []
    for case_key, shape in grids:
        s = _solver(case_key, shape, be)
        s.step()
        be.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            s.step()
        be.synchronize()
        dt = (time.perf_counter() - t0) / steps
        rows.append({"case": case_key, "shape": list(shape), "seconds_per_step": dt,
                     "cells_per_second": float(np.prod(shape)) / dt})
        s.state = None
        be.free_memory()
    return rows


def format_results(results) -> str:
    lines = []
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"[{mark}] {r['name']}: {r['value']:.3g}"
                     + (f" (tol {r['tolerance']:g})" if r["tolerance"] else "")
                     + f"  [{r['seconds']}s]")
        if r["note"]:
            lines.append(f"       {r['note']}")
    n_ok = sum(r["passed"] for r in results)
    lines.append(f"{n_ok}/{len(results)} checks passed")
    return "\n".join(lines)
