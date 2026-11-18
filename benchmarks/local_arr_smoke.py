"""Small deterministic smoke benchmark for v1.1 spatial 3-D ARR.

This is not a production performance benchmark.  It exists to verify three
things cheaply: a local cascade is detected, the controller stays on the coarse
global tier while the patch is active, and the restricted result moves toward
a uniform fine-grid reference with less cell-step work.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnisvox_cfd.arr import ARRConfig, ARRController
from omnisvox_cfd.backend import get_backend
from omnisvox_cfd.solver3d import NavierStokes3D
from omnisvox_cfd.spectral import resample_hat

L = 2 * np.pi
RE = 1000.0
T_END = 0.08
COARSE = (24, 24, 24)
FINE = (48, 48, 48)


def initial_state(s):
    x, y, z = s.grid.coords()
    pd = lambda a, c: (a - c + np.pi) % (2 * np.pi) - np.pi
    X, Y, Z = pd(x, np.pi), pd(y, np.pi), pd(z, np.pi)
    env = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)
    u = np.stack((np.sin(2*x)*np.cos(2*y)*np.cos(2*z),
                  -np.cos(2*x)*np.sin(2*y)*np.cos(2*z),
                  np.zeros_like(x))) * env
    uh = s._project(s.grid.fft(s.be.asarray(u)) * s.grid.dealias)
    up = s.grid.ifft(uh)
    rms = np.sqrt(s.be.scalar(s.be.xp.mean(s.be.xp.sum(up**2, axis=0))))
    return uh / max(rms, 1e-12)


def relerr(a, b):
    return float(np.sqrt(np.mean((a-b)**2)) / max(np.sqrt(np.mean(b**2)), 1e-30))


def run():
    be = get_backend("cpu")
    nu = 1.0 / RE

    fine = NavierStokes3D(FINE, (L,)*3, nu, be, cfl=0.3)
    fine.set_state(initial_state(fine))
    t = time.perf_counter(); fine.advance_to(T_END); fine_wall = time.perf_counter() - t
    ref_hat = resample_hat(fine.state, FINE, COARSE, be)
    ref = NavierStokes3D(COARSE, (L,)*3, nu, be)
    ref.set_state(ref_hat * ref.grid.dealias, T_END)
    uref = ref.volume("velocity")

    coarse = NavierStokes3D(COARSE, (L,)*3, nu, be, cfl=0.3)
    coarse.set_state(initial_state(coarse))
    t = time.perf_counter(); coarse.advance_to(T_END); coarse_wall = time.perf_counter() - t
    coarse_err = relerr(coarse.volume("velocity"), uref)

    tiers = [COARSE, FINE]
    def make(i):
        return NavierStokes3D(tiers[i], (L,)*3, nu, be, cfl=0.3)

    cfg = ARRConfig(
        press_up=0.03, obs_every=1, hold=2,
        local_3d=True, local_target_fraction=0.70,
        local_padding=2, local_blend=2, local_max_volume=0.70,
        local_min_age=0.03, local_fail_factor=20.0,
    )
    ctrl = ARRController(make, tiers, initial_state, cfg, energy_monotone=False)
    t = time.perf_counter()
    while ctrl.t < T_END - 1e-12:
        ctrl.run_chunk(t_limit=T_END)
    local_wall = time.perf_counter() - t
    local_err = relerr(ctrl.solver.volume("velocity"), uref)
    cost = ctrl.cost_summary()

    return {
        "coarse_relative_velocity_error": coarse_err,
        "local_arr_relative_velocity_error": local_err,
        "error_improvement_percent": 100.0 * (coarse_err - local_err) / coarse_err,
        "coarse_wall_seconds": coarse_wall,
        "local_arr_wall_seconds": local_wall,
        "uniform_fine_wall_seconds": fine_wall,
        "cell_step_saving_est": cost["cell_step_saving_est"],
        "global_divergence_after_merge": ctrl.solver.max_divergence(),
        "events": ctrl.events,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
