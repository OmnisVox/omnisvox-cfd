"""Localized 3-D benchmark for spatial ARR, with honest baselines.

Compact vortex packet in a 32^3 periodic box (Re = 1000).  Every run is
scored against a 128^3 uniform reference restricted to the 32^3 band.

Baselines that a local scheme has to beat:
  * uniform grids from t = 0 (cheapest grid that reaches a given accuracy);
  * whole-domain 64^3 promotion from the *same rollback point* the local
    path uses (same trigger, same refinement ratio, no localisation).

Run:  python benchmarks/local_arr_localized.py            (~3-5 min on CPU)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnisvox_cfd.arr import ARRConfig, ARRController
from omnisvox_cfd.backend import get_backend
from omnisvox_cfd.solver3d import NavierStokes3D
from omnisvox_cfd.spectral import resample_hat

L = 2 * np.pi
NU = 1.0 / 1000.0
T_END = 0.12
C, F, REF = (32,) * 3, (64,) * 3, (128,) * 3
SIG, KW, AMP = 0.6, 3, 0.25
be = get_backend("cpu")


def sol(shape):
    return NavierStokes3D(shape, (L,) * 3, NU, be, cfl=0.3)


def init(s):
    x, y, z = s.grid.coords()
    pd = lambda a, c: (a - c + np.pi) % (2 * np.pi) - np.pi
    X, Y, Z = pd(x, 2.2), pd(y, 3.4), pd(z, 2.9)
    env = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * SIG ** 2))
    u = np.stack((np.sin(KW * x) * np.cos(KW * y) * np.cos(KW * z),
                  -np.cos(KW * x) * np.sin(KW * y) * np.cos(KW * z), np.zeros_like(x))) * env
    uh = s._project(s.grid.fft(s.be.asarray(u)) * s.grid.dealias)
    up = s.grid.ifft(uh)
    return uh / max(float(np.sqrt(np.mean(np.sum(up ** 2, axis=0)))), 1e-12) * AMP


def restrict(s):
    r = sol(C)
    r.set_state(resample_hat(s.state, s.shape, C, be) * r.grid.dealias, s.t)
    return r.volume("velocity")


def relerr(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def uniform(N, start=None):
    s = sol((N,) * 3)
    if start is None:
        s.set_state(init(s))
    else:
        s.set_state(resample_hat(start.state, start.shape, (N,) * 3, be) * s.grid.dealias, start.t)
    tic = time.perf_counter()
    s.advance_to(T_END)
    return s, time.perf_counter() - tic


def local(scheme, target_fraction):
    tiers = [C, F]
    cfg = ARRConfig(press_up=0.03, obs_every=1, hold=2, local_3d=True, local_scheme=scheme,
                    local_target_fraction=target_fraction, local_padding=3, local_blend=2,
                    local_max_volume=0.6, local_min_age=0.03, local_fail_factor=1e9)
    c = ARRController(lambda i: sol(tiers[i]), tiers, init, cfg, energy_monotone=False)
    tic = time.perf_counter()
    while c.t < T_END - 1e-12:
        c.run_chunk(t_limit=T_END)
    wall = time.perf_counter() - tic
    ev = [e for e in c.events if e.get("region")]
    return c, ev, wall


def run():
    cache = Path(__file__).with_name("results") / "localized_ref128.npy"
    if cache.exists():
        uref = np.load(cache)
    else:
        s, _ = uniform(REF[0])
        uref = restrict(s)
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, uref)
    rows = []
    for N in (32, 40, 48, 64):
        s, w = uniform(N)
        rows.append({"run": f"uniform {N}^3", "error": relerr(restrict(s), uref), "wall_s": w})
    t_restore = None
    for tf in (0.7, 0.9):
        for scheme in ("fd", "spectral"):
            c, ev, w = local(scheme, tf)
            local_ok = c.solver.shape == C and ev
            rows.append({"run": f"local {scheme}, target {tf}",
                         "error": relerr(restrict(c.solver), uref), "wall_s": w,
                         "patch_volume": ev[0]["region"]["volume_fraction"] if local_ok else None,
                         "fell_back": not local_ok})
            if local_ok:
                t_restore = ev[0]["t_restore"]
    if t_restore is not None:
        s0 = sol(C)
        s0.set_state(init(s0))
        s0.advance_to(t_restore)
        g, w = uniform(F[0], start=s0)
        rows.append({"run": f"global {F[0]}^3 from same rollback (t={t_restore:.4f})",
                     "error": relerr(restrict(g), uref), "wall_s": w})
    return rows


if __name__ == "__main__":
    rows = run()
    print(json.dumps(rows, indent=1))
    for r in rows:
        vol = f"  patch {r['patch_volume']:.0%}" if r.get("patch_volume") else ""
        print(f"{r['run']:44s} err={r['error']:.4f}  wall={r['wall_s']:.2f}s{vol}")
