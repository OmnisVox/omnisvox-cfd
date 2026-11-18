"""Engine tests.  Run with:  python -m pytest -q"""
import json
import math

import numpy as np
import pytest

from omnisvox_cfd import (ARRConfig, Simulation, SpectralGrid, get_backend, get_case,
                          json_safe, resample_hat, resolve_config)
from omnisvox_cfd import validation as V
from omnisvox_cfd.io import encode_png, field_to_rgb, frame_png
from omnisvox_cfd.solver3d import NavierStokes3D
from omnisvox_cfd.local_arr import LocalPatch3D, PatchRegion, locate_hot_region
from omnisvox_cfd.local_spectral import SpectralPatch3D


@pytest.fixture(scope="module")
def be():
    return get_backend("cpu")


# ---- numerical checks shared with `omnisvox-cfd validate` -------------------
@pytest.mark.parametrize("check", V.QUICK, ids=lambda f: f.__name__)
def test_validation_check(check, be):
    r = check(be)
    assert r["passed"], f"{r['name']}: value {r['value']} tol {r['tolerance']} ({r['note']})"


def test_single_precision_checks():
    be32 = get_backend("cpu", "single")
    for check in (V.check_taylor_green_2d, V.check_abc, V.check_grid_transfer):
        r = check(be32)
        assert r["passed"], r


# ---- spectral building blocks ------------------------------------------------
def test_parseval(be):
    rng = np.random.default_rng(0)
    for shape, lengths in (((16, 24), (1.0, 2.0)), ((16, 12, 20), (1.0, 2.0, 3.0))):
        g = SpectralGrid(shape, lengths, be)
        a = rng.standard_normal(shape)
        assert g.mean_sq(g.fft(a)) == pytest.approx(np.mean(a ** 2), rel=1e-12)


def test_restriction_removes_only_unrepresentable_modes(be):
    g = SpectralGrid((64, 64), (1.0, 1.0), be)
    x, y = g.coords()
    a = np.sin(2 * np.pi * 3 * x) + 0.5 * np.cos(2 * np.pi * 18 * y)     # mode 18 > 32/3
    coarse = resample_hat(g.fft(a), (64, 64), (32, 32), be)
    gc = SpectralGrid((32, 32), (1.0, 1.0), be)
    back = gc.ifft(coarse * gc.dealias)
    xc, _ = gc.coords()
    assert np.allclose(back, np.sin(2 * np.pi * 3 * xc), atol=1e-12)


def test_guard_is_identity_below_knee(be):
    s = NavierStokes3D((16, 16, 16), (1, 1, 1), 0.01, be, guard_cap=2.0)
    speed = np.array([0.0, 0.5, 1.49, 1.5, 1.8, 5.0, 100.0])
    fac = s._limit_speed(speed)
    assert np.all(fac[:4] == 1.0)
    assert np.all(fac[4:] < 1.0)
    assert np.all(speed * fac <= 2.0 + 1e-12)
    assert np.all(np.diff(speed * fac) >= 0)          # monotone


def test_smagorinsky_adds_dissipation(be):
    case = get_case("taylor_green_3d")
    p = case.resolve_params({"Re": 1600.0})
    out = {}
    for sgs in ("none", "smagorinsky"):
        s = case.make_solver((24, 24, 24), p, be, sgs=sgs)
        s.set_state(case.initial_state(s, p))
        s.advance_to(3.0)
        out[sgs] = s.diagnostics()["energy"]
        assert s.max_divergence() < 1e-10
    assert out["smagorinsky"] < out["none"]



def _localized_vortex_state(s, sigma=0.23):
    """Divergence-free compact vortex used to exercise spatial ARR."""
    x, y, z = s.grid.coords()
    L = 2 * np.pi
    pd = lambda a, c: (a - c + np.pi) % L - np.pi
    X, Y, Z = pd(x, np.pi), pd(y, np.pi), pd(z, np.pi)
    psi = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sigma ** 2))
    u = np.stack((-Y / sigma ** 2 * psi, X / sigma ** 2 * psi, np.zeros_like(psi)))
    return s._project(s.grid.fft(s.be.asarray(u)) * s.grid.dealias)


def test_local_patch_merge_uses_global_projection(be):
    s = NavierStokes3D((24, 24, 24), (2*np.pi,)*3, 1/400, be, cfl=0.35)
    s.set_state(_localized_vortex_state(s))
    region, info = locate_hot_region(s, 0.7, target_fraction=0.5, padding=2, min_size=4)
    assert region.volume_fraction < 0.5
    u0, t0 = s.volume("velocity"), s.t
    patch = LocalPatch3D(region, 2, s.grid.lengths, s.nu, s.cfl, u0, t0, blend_coarse=2)
    patch.mark_born()
    s.step(t_limit=0.01)
    patch.advance_interval(t0, s.t, u0, s.volume("velocity"))
    m = patch.merge_into(s)
    assert math.isfinite(m["correction_rms"]) and m["correction_rms"] > 0
    assert s.max_divergence() < 1e-10


def _spectral_patch_setup(be, t_run=0.02):
    s = NavierStokes3D((24, 24, 24), (2*np.pi,)*3, 1/400, be, cfl=0.35)
    s.set_state(_localized_vortex_state(s, sigma=0.5))
    s.advance_to(t_run)
    region = PatchRegion((5, 6, 4), (14, 12, 16), (24, 24, 24))
    return s, region


def test_spectral_patch_interpolation_is_exact(be):
    s, region = _spectral_patch_setup(be)
    p = SpectralPatch3D(region, 2, s.grid.lengths, s.nu, s.cfl, s.volume("velocity"), s.t)
    f = NavierStokes3D((48, 48, 48), (2*np.pi,)*3, s.nu, be)
    f.set_state(resample_hat(s.state, (24,)*3, (48,)*3, be), s.t)
    ix = [(2*a + np.arange(2*n)) % 48 for a, n in zip(region.start, region.size)]
    uf = f.volume("velocity")[np.ix_(np.arange(3), *ix)]
    assert np.max(np.abs(p.u - uf)) < 1e-11 * max(1.0, np.max(np.abs(uf)))


def test_spectral_patch_detail_is_solenoidal_and_compact(be):
    s, region = _spectral_patch_setup(be)
    u0, t0 = s.volume("velocity"), s.t
    p = SpectralPatch3D(region, 2, s.grid.lengths, s.nu, s.cfl, u0, t0, blend_coarse=2)
    s.step(t_limit=t0 + 0.01)
    p.advance_interval(t0, s.t, u0, s.volume("velocity"))
    d = p.delta
    assert np.max(np.abs(d)) > 0
    assert p.max_divergence < 1e-8 * max(1.0, np.max(np.abs(d)) * 50)
    face = max(np.abs(d[:, 0]).max(), np.abs(d[:, -1]).max(), np.abs(d[:, :, 0]).max(),
               np.abs(d[:, :, -1]).max(), np.abs(d[:, :, :, 0]).max(), np.abs(d[:, :, :, -1]).max())
    # spectral derivatives of the narrow taper leave a small seam residue
    assert face < 5e-2 * np.max(np.abs(d))


def test_spectral_merge_conserves_patch_field_and_divergence(be):
    s, region = _spectral_patch_setup(be)
    u0, t0 = s.volume("velocity"), s.t
    p = SpectralPatch3D(region, 2, s.grid.lengths, s.nu, s.cfl, u0, t0, blend_coarse=2)
    s.step(t_limit=t0 + 0.01)
    u1 = s.volume("velocity")
    p.advance_interval(t0, s.t, u0, u1)
    total_before = p.last_target + p.delta
    m = p.merge_into(s)
    T_after = p._target(s.volume("velocity"))[0]
    total_after = T_after + p.delta
    core = (slice(None),) + tuple(slice(6, n - 6) for n in p.fine_shape)
    scale = np.max(np.abs(total_before[core]))
    # ownership moves between levels; only the re-taper of the remainder changes the total
    assert np.max(np.abs(total_after - total_before)[core]) < 1e-4 * scale
    assert s.max_divergence() < 1e-10
    assert math.isfinite(m["energy_rel_change"])


def test_localiser_ignores_scattered_background(be):
    s = NavierStokes3D((32, 32, 32), (2*np.pi,)*3, 1/400, be)
    st = _localized_vortex_state(s, sigma=0.3)
    rng = np.random.default_rng(1)
    bg = s.grid.ifft(s.grid.fft(s.be.asarray(rng.standard_normal((3, 32, 32, 32)))) * s.grid.dealias)
    bg *= 0.005 * np.abs(s.grid.ifft(st)).max() / np.abs(bg).max()
    s.set_state(s._project(st + s.grid.fft(s.be.asarray(bg)) * s.grid.dealias))
    region, info = locate_hot_region(s, 0.7, target_fraction=0.7, padding=2, min_size=4)
    # 1.1.0's top-cell bounding box returned 38% of the domain here
    assert region.volume_fraction < 0.25
    assert info["selected_fraction"] > 0.7


def test_local_arr_spectral_end_to_end(be):
    tiers = [(24,)*3, (48,)*3]
    mk = lambda i: NavierStokes3D(tiers[i], (2*np.pi,)*3, 1/400, be, cfl=0.35)
    init = lambda s: _localized_vortex_state(s, sigma=0.23)
    from omnisvox_cfd.arr import ARRController
    cfg = ARRConfig(press_up=0.03, obs_every=1, hold=2, local_3d=True, local_scheme="spectral",
                    local_target_fraction=0.7, local_padding=2, local_blend=2,
                    local_max_volume=0.6, local_min_age=0.01, local_fail_factor=1e6)
    c = ARRController(mk, tiers, init, cfg, energy_monotone=False)
    while c.t < 0.06 - 1e-12:
        c.run_chunk(t_limit=0.06)
    kinds = [e["type"] for e in c.events]
    assert "rollback_local_refine" in kinds
    assert c.solver.max_divergence() < 1e-10
    assert np.all(np.isfinite(c.solver.volume("velocity")))


def test_local_3d_is_off_by_default_and_scheme_validated():
    assert ARRConfig().local_3d is False
    assert ARRConfig().local_scheme == "spectral"
    with pytest.raises(ValueError):
        ARRConfig(local_scheme="bogus").validate(3)


# ---- adaptive controller -----------------------------------------------------
def _tgv(**arr):
    return {"case": "taylor_green_3d", "params": {"Re": 400.0}, "t_end": 4.0,
            "preset": "16^3 / 32^3 / 64^3", "arr": arr}


def test_arr_promotes_with_rollback_and_history_is_consistent():
    sim = Simulation(_tgv()).run()
    kinds = [e["type"] for e in sim.events]
    assert sim.status == "done" and "rollback_promote" in kinds
    h = sim.history
    assert all(b >= a for a, b in zip(h["t"], h["t"][1:])), "history time must not go backwards"
    assert h["t"][-1] == pytest.approx(4.0)
    assert all(math.isfinite(v) for v in h["energy"])
    assert all(b <= a * (1 + 1e-6) for a, b in zip(h["energy"], h["energy"][1:]))
    for e in sim.events:
        if e["type"] == "rollback_promote":
            assert e["t_restore"] <= e["t_detect"] and e["to_tier"] == e["from_tier"] + 1
    cost = sim.summary()["cost"]
    assert cost["wasted"]["steps"] == sum(e.get("steps_discarded", 0) for e in sim.events)
    assert cost["guard_hits"] == 0


def test_promote_in_place_mode_never_discards():
    sim = Simulation(_tgv(rollback=False)).run()
    assert any(e["type"] == "promote" for e in sim.events)
    assert sim.summary()["cost"]["wasted"]["steps"] == 0


def test_disabled_arr_stays_on_one_grid():
    sim = Simulation(_tgv(enabled=False, start_tier=1)).run()
    assert set(sim.history["tier"]) == {1}
    assert not [e for e in sim.events if e["type"] != "saturated"]


def test_demotion_and_no_chatter():
    sim = Simulation({"case": "turbulence_2d", "params": {"Re": 200.0, "k0": 6},
                      "t_end": 12.0, "preset": "64^2 / 128^2 / 256^2"}).run()
    kinds = [e["type"] for e in sim.events]
    assert sim.status == "done"
    assert "demote" in kinds, kinds
    # hysteresis: a demotion is never immediately undone
    for a, b in zip(sim.events, sim.events[1:]):
        if a["type"] == "demote" and b["type"] in ("rollback_promote", "promote"):
            assert b["t"] - a["t"] > 0.5
    assert len(kinds) <= 8, kinds


def test_turbulent_cases_start_fine_and_only_coarsen():
    from omnisvox_cfd import resolve_config
    _, _, _, tiers, arr = resolve_config({"case": "turbulence_2d", "preset": "64^2 / 128^2 / 256^2"})
    assert arr.start_tier == len(tiers) - 1
    _, _, _, _, arr = resolve_config({"case": "turbulence_2d", "preset": "64^2 / 128^2 / 256^2",
                                      "arr": {"start_tier": 0}})
    assert arr.start_tier == 0                       # an explicit choice wins
    _, _, _, _, arr = resolve_config({"case": "turbulence_2d", "preset": "64^2 / 128^2 / 256^2",
                                      "arr": {"enabled": False}})
    assert arr.start_tier == 0                       # fixed-grid runs are untouched
    _, _, _, _, arr = resolve_config({"case": "taylor_green_3d"})
    assert arr.start_tier == 0                       # smooth starts still begin coarse
    sim = Simulation({"case": "turbulence_2d", "params": {"Re": 200.0, "k0": 6},
                      "t_end": 12.0, "preset": "64^2 / 128^2 / 256^2"}).run()
    kinds = [e["type"] for e in sim.events]
    assert "demote" in kinds and not any(k in ("promote", "rollback_promote") for k in kinds), kinds


def test_no_demotion_while_the_cascade_is_still_building():
    # Started on the finest grid with the cascade not yet begun, the coarse-grid
    # pressure is ~0 but rising: ARR must not step down and then have to come back.
    sim = Simulation({"case": "turbulence_2d", "params": {"Re": 1000.0, "k0": 6}, "t_end": 0.4,
                      "preset": "128^2 / 256^2 / 512^2", "arr": {"start_tier": 1}}).run()
    assert not [e for e in sim.events if e["type"] == "demote"], sim.events   # 1.1.1 demoted at t=0.057


def test_initial_condition_gets_a_finer_grid_when_needed():
    sim = Simulation({"case": "shear_layer", "params": {"rho": 100.0}, "t_end": 0.01,
                      "preset": "64^2 / 128^2 / 256^2"})
    assert sim.events and sim.events[0]["type"] == "initial_promote"
    assert sim.tier >= 1 and sim.t == 0.0


def test_fault_on_finest_tier_rewinds_and_retries():
    sim = Simulation(_tgv(enabled=False, start_tier=1))
    sim.advance(max_chunks=4)
    sim.solver.state[1, 2, 3, 1] = float("inf")
    sim.run()
    assert sim.status == "done"
    assert any(e["type"] == "rollback_retry" for e in sim.events)
    assert all(math.isfinite(v) for v in sim.history["energy"])


def test_unrecoverable_run_fails_cleanly():
    sim = Simulation(_tgv(enabled=False, start_tier=0, max_retries=1))
    sim.advance(max_chunks=2)
    from omnisvox_cfd.base import SolverFault

    def bad_step(t_limit=None):
        raise SolverFault("synthetic fault")

    sim.solver.step = bad_step
    sim.run()
    assert sim.status == "failed" and "synthetic fault" in sim.error


def test_saturation_is_flagged_and_stop_policy_fails():
    cfg = {"case": "taylor_green_3d", "params": {"Re": 1600.0}, "t_end": 6.0,
           "tiers": [[16, 16, 16], [24, 24, 24]]}
    sim = Simulation(cfg).run()
    assert sim.status == "done"
    assert any(e["type"] == "saturated" for e in sim.events)
    assert sim.summary()["time_saturated"] > 0
    sim2 = Simulation({**cfg, "arr": {"on_saturation": "stop"}}).run()
    assert sim2.status == "failed" and "under-resolved" in sim2.error


def test_sgs_fallback_switches_on_at_saturation():
    cfg = {"case": "taylor_green_3d", "params": {"Re": 1600.0}, "t_end": 5.0,
           "tiers": [[16, 16, 16], [24, 24, 24]], "arr": {"on_saturation": "sgs"}}
    sim = Simulation(cfg).run()
    assert any(e["type"] == "sgs_on" for e in sim.events)
    assert sim.solver.sgs == "smagorinsky"


# ---- obstacle flow -----------------------------------------------------------
def test_obstacle_body_is_impermeable_and_sheds_drag():
    sim = Simulation({"case": "obstacle", "params": {"Re": 100.0}, "t_end": 3.0,
                      "preset": "128x64 / 256x128 / 512x256",
                      "arr": {"enabled": False, "start_tier": 1}}).run()
    assert sim.status == "done"
    speed = sim.field("speed")
    mask = np.asarray(sim.solver.body_mask)
    assert speed[mask > 0.99].max() < 0.1          # essentially at rest inside the body
    assert 0.5 < sim.history["cd"][-1] < 4.0
    far = speed[5, :]                               # upstream column: undisturbed stream
    assert np.allclose(far, 1.0, atol=0.05)


@pytest.mark.parametrize("body", ["square", "ellipse", "naca"])
def test_other_bodies_run(body):
    sim = Simulation({"case": "obstacle", "params": {"Re": 200.0, "body": body, "angle": 10.0},
                      "t_end": 0.5, "preset": "128x64 / 256x128 / 512x256",
                      "arr": {"enabled": False, "start_tier": 0}}).run()
    assert sim.status == "done" and math.isfinite(sim.history["energy"][-1])


# ---- configuration -----------------------------------------------------------
@pytest.mark.parametrize("bad, msg", [
    ({"case": "nope"}, "unknown case"),
    ({"case": "shear_layer", "params": {"Re": -5}}, "Re must lie"),
    ({"case": "shear_layer", "params": {"bogus": 1}}, "unknown parameter"),
    ({"case": "shear_layer", "tiers": [[64, 64], [32, 32]]}, "strictly increasing"),
    ({"case": "shear_layer", "tiers": [[63, 64]]}, "even"),
    ({"case": "shear_layer", "tiers": [[64, 64, 64]]}, "2-D"),
    ({"case": "shear_layer", "sgs": "smagorinsky"}, "3-D cases only"),
    ({"case": "shear_layer", "arr": {"press_up": 2}}, "press_up"),
    ({"case": "shear_layer", "arr": {"whatever": 2}}, "unknown arr option"),
    ({"case": "shear_layer", "t_end": -1}, "t_end"),
    ({"case": "shear_layer", "surprise": 1}, "unknown config key"),
    ({"case": "shear_layer", "arr": {"start_tier": 7}}, "start_tier"),
])
def test_bad_configs_are_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        resolve_config(bad)


def test_all_cases_start(be):
    for key, case in __import__("omnisvox_cfd").CASES.items():
        preset = min(case.tier_presets, key=lambda k: np.prod(case.tier_presets[k][0]))
        sim = Simulation({"case": key, "preset": preset, "t_end": 1e-3,
                          "arr": {"start_tier": 0}})
        sim.run()
        assert sim.status == "done", key
        assert sim.field().ndim == 2
        json.dumps(json_safe(sim.summary()))      # must be valid JSON
        json.dumps(json_safe(case.describe()))


# ---- exports -----------------------------------------------------------------
def test_exports(tmp_path):
    sim = Simulation({"case": "shear_layer", "t_end": 0.02, "preset": "64^2 / 128^2 / 256^2"}).run()
    png = frame_png(sim)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    sim.export_csv(tmp_path / "h.csv")
    rows = (tmp_path / "h.csv").read_text().strip().splitlines()
    assert rows[0].startswith("t,step,tier") and len(rows) == len(sim.history["t"]) + 1
    sim.export_npz(tmp_path / "f.npz")
    z = np.load(tmp_path / "f.npz")
    assert z["vorticity"].shape == tuple(sim.shape)
    sim.export_vtk(tmp_path / "f.vtk")
    head = (tmp_path / "f.vtk").read_bytes()[:200]
    assert b"STRUCTURED_POINTS" in head
    sim3 = Simulation({"case": "abc_flow", "t_end": 0.01}).run()
    sim3.export_vtk(tmp_path / "g.vtk")
    sim3.export_npz(tmp_path / "g.npz")
    assert np.load(tmp_path / "g.npz")["velocity"].shape == (3,) + tuple(sim3.shape)


def test_png_roundtrip_dimensions():
    a = np.linspace(-1, 1, 12 * 8).reshape(12, 8)       # (Nx, Ny)
    rgb = field_to_rgb(a, symmetric=True)
    assert rgb.shape == (8, 12, 3)
    png = encode_png(rgb)
    import struct
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (12, 8)
