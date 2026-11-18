"""Benchmark suite.  Results land in benchmarks/results/*.json.

    python benchmarks/run_all.py                 # everything (about 1 h on 2 CPU cores)
    python benchmarks/run_all.py --only tgv      # one group: exact, throughput, tgv, shear, turb2d, cylinder,
                                                 # cylinder_refined, timing
    python benchmarks/run_all.py --device gpu    # same suite on the GPU
    python benchmarks/run_all.py --force         # re-run even if a result file exists

Then:  python benchmarks/make_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnisvox_cfd import Simulation, get_backend, json_safe  # noqa: E402
from omnisvox_cfd.validation import run_validation, throughput  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def uniform(tier):
    return {"enabled": False, "start_tier": tier}


def tgv(Re, t_end, **kw):
    return {"case": "taylor_green_3d", "params": {"Re": Re}, "t_end": t_end, **kw}


SHEAR = {"case": "shear_layer", "params": {"Re": 30000.0, "rho": 100.0}, "t_end": 1.8}
TURB = {"case": "turbulence_2d", "params": {"Re": 1000.0, "k0": 6}, "t_end": 30.0,
        "preset": "64^2 / 128^2 / 256^2"}
CYL = {"case": "obstacle", "params": {"Re": 100.0}}

# name -> (config, save final field?)
GROUPS = {
    "tgv": {
        "tgv1600_uniform128": (tgv(1600.0, 12.0, arr=uniform(2)), False),
        "tgv1600_uniform64": (tgv(1600.0, 12.0, arr=uniform(1)), False),
        "tgv1600_uniform32": (tgv(1600.0, 12.0, arr=uniform(0)), False),
        "tgv1600_arr": (tgv(1600.0, 12.0), False),
        "tgv1600_arr_sgs64": (tgv(1600.0, 12.0, preset="16^3 / 32^3 / 64^3",
                                  arr={"on_saturation": "sgs"}), False),
        "tgv1600_les64": (tgv(1600.0, 12.0, sgs="smagorinsky", arr=uniform(1)), False),
        "tgv400_uniform128": (tgv(400.0, 12.0, arr=uniform(2)), False),
        "tgv400_uniform64": (tgv(400.0, 12.0, arr=uniform(1)), False),
        "tgv400_arr": (tgv(400.0, 12.0), False),
    },
    "shear": {
        "shear_uniform512": ({**SHEAR, "arr": uniform(2)}, True),
        "shear_uniform256": ({**SHEAR, "arr": uniform(1)}, True),
        "shear_uniform128": ({**SHEAR, "arr": uniform(0)}, True),
        "shear_arr": (SHEAR, True),
        "shear_arr_norollback": ({**SHEAR, "arr": {"rollback": False}}, True),
        # deeper rollback: freeze the checkpoint much earlier, so more is redone
        "shear_arr_deep": ({**SHEAR, "arr": {"warn_ratio": 0.1, "down_ratio": 0.05,
                                             "max_spec_obs": 400}}, True),
        "shear_arr_strict": ({**SHEAR, "arr": {"press_up": 0.0015}}, True),
    },
    "turb2d": {
        "turb2d_uniform256": ({**TURB, "arr": uniform(2)}, True),
        "turb2d_uniform64": ({**TURB, "arr": uniform(0)}, True),
        "turb2d_arr": (TURB, True),
    },
    "cylinder": {
        "cyl_512x256": ({**CYL, "t_end": 120.0, "arr": uniform(1)}, True),
        "cyl_256x128": ({**CYL, "t_end": 120.0, "arr": uniform(0)}, False),
        "cyl_arr": ({**CYL, "t_end": 120.0}, True),
        "cyl_512x512_tall": ({**CYL, "t_end": 100.0, "tiers": [[512, 512]]}, False),
    },
    # run this group alone on an idle machine: it is the wall-clock comparison
    "timing": {
        "timing_tgv400_uniform64": (tgv(400.0, 10.0, preset="16^3 / 32^3 / 64^3",
                                        arr=uniform(2)), False),
        "timing_tgv400_arr": (tgv(400.0, 10.0, preset="16^3 / 32^3 / 64^3"), False),
        "timing_turb2d_uniform256": ({**TURB, "arr": uniform(2)}, False),
        "timing_turb2d_arr": (TURB, False),
    },
}


def run(name, cfg, save_field, device, precision, force=False):
    path = OUT / f"{name}.json"
    if path.exists() and not force:
        print(f"[{name}] exists, skipping", flush=True)
        return
    cfg = {**cfg, "device": device, "precision": precision}
    sim = Simulation(cfg)
    t0 = last = time.perf_counter()
    while not sim.done:
        sim.advance(max_wall=5.0)
        if time.perf_counter() - last > 60:
            last = time.perf_counter()
            print(f"[{name}] t={sim.t:.3f}/{sim.t_end:g} grid={'x'.join(map(str, sim.shape))} "
                  f"wall={last - t0:.0f}s", flush=True)
    out = {"name": name, "config": cfg, "summary": sim.summary(), "history": sim.history}
    path.write_text(json.dumps(json_safe(out)))
    if save_field:
        np.save(OUT / f"{name}_field.npy", sim.field())
    c = out["summary"]["cost"]
    print(f"[{name}] {sim.status} wall={sim.wall:.1f}s steps={sim.ctrl.total_steps} "
          f"promotions={c['promotions']} rollbacks={c['rollbacks']} demotions={c['demotions']} "
          f"limiter={c['guard_hits']}", flush=True)


def run_cylinder_refined(device, precision, force=False, t_switch=40.0, t_end=80.0):
    """Grid-convergence point for the cylinder at 32 cells per diameter.

    Develops the wake on 512x256, then prolongs the flow onto 1024x512 and
    continues there, which is much cheaper than starting the fine grid from rest.
    """
    name = "cyl_1024x512_restart"
    path = OUT / f"{name}.json"
    if path.exists() and not force:
        print(f"[{name}] exists, skipping", flush=True)
        return
    cfg = {**CYL, "t_end": t_end, "tiers": [[512, 256], [1024, 512]], "device": device,
           "precision": precision, "arr": {"press_up": 0.99, "start_tier": 0}}
    sim = Simulation(cfg)
    sim.advance(until=t_switch)
    sim.ctrl._promote(reason="manual refinement for the convergence study")
    sim.ctrl.min_tier = 1                      # stay on the fine grid from here on
    last = time.perf_counter()
    while not sim.done:
        sim.advance(max_wall=5.0)
        if time.perf_counter() - last > 60:
            last = time.perf_counter()
            print(f"[{name}] t={sim.t:.3f}/{sim.t_end:g} wall={sim.wall:.0f}s", flush=True)
    out = {"name": name, "config": cfg, "summary": sim.summary(), "history": sim.history,
           "t_switch": t_switch}
    path.write_text(json.dumps(json_safe(out)))
    print(f"[{name}] {sim.status} wall={sim.wall:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="groups and/or run names")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="double")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    be = get_backend(args.device, args.precision)
    print("backend:", be.describe(), flush=True)
    want = set(args.only or [])

    if not want or "exact" in want:
        res = run_validation(be)
        (OUT / "validation.json").write_text(json.dumps(json_safe(
            {"backend": be.describe(), "results": res}), indent=1))
        print(f"[validation] {sum(r['passed'] for r in res)}/{len(res)} passed", flush=True)
    if not want or "throughput" in want:
        grids = [("taylor_green_3d", (32,) * 3), ("taylor_green_3d", (64,) * 3),
                 ("taylor_green_3d", (128,) * 3), ("shear_layer", (256, 256)),
                 ("shear_layer", (512, 512)), ("shear_layer", (1024, 1024))]
        if be.is_gpu:
            grids += [("taylor_green_3d", (256,) * 3), ("shear_layer", (2048, 2048))]
        rows = throughput(be, grids)
        (OUT / "throughput.json").write_text(json.dumps(json_safe(
            {"backend": be.describe(), "rows": rows}), indent=1))
        print("[throughput] done", flush=True)
    for group, runs in GROUPS.items():
        for name, (cfg, save_field) in runs.items():
            if want and group not in want and name not in want:
                continue
            if not want and group == "timing":
                continue
            run(name, cfg, save_field, args.device, args.precision, args.force)
    if "cylinder_refined" in want:
        run_cylinder_refined(args.device, args.precision, args.force)


if __name__ == "__main__":
    main()
