"""Command-line interface.

    omnisvox-cfd info
    omnisvox-cfd cases
    omnisvox-cfd run --case taylor_green_3d --param Re=1600 --out runs/tgv
    omnisvox-cfd validate [--device gpu] [--precision single]
    omnisvox-cfd bench [--device gpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .backend import BackendError, get_backend, gpu_status
from .cases import CASES, get_case
from .runner import Simulation, json_safe


def _parse_kv(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected NAME=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_info(args):
    st = gpu_status()
    print(f"OmnisVox CFD {__version__}")
    print("CPU backend:", get_backend("cpu").describe())
    if st["available"]:
        print(f"GPU backend: available - {st['name']} ({st['memory_gb']} GB)")
    else:
        print("GPU backend: not available -", st["reason"])
    return 0


def cmd_cases(args):
    for c in CASES.values():
        print(f"{c.key}  ({c.ndim}-D)  {c.title}")
        print(f"    {c.summary}")
        for p in c.params:
            rng = f"choices {p.choices}" if p.choices else f"range [{p.min}, {p.max}]"
            print(f"    --param {p.name}={p.default!r}   {p.label}; {rng}")
        print(f"    presets: {', '.join(repr(k) for k in c.tier_presets)} "
              f"(default {c.default_preset!r}); t_end default {c.t_end}")
    return 0


def build_config(args) -> dict:
    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    if args.case:
        cfg["case"] = args.case
    cfg.setdefault("case", "taylor_green_3d")
    if args.param:
        cfg.setdefault("params", {}).update(_parse_kv(args.param))
    if args.preset:
        cfg["preset"] = args.preset
    if args.tiers:
        cfg["tiers"] = [[int(n) for n in t.lower().split("x")] for t in args.tiers.split(",")]
    for key in ("t_end", "device", "precision", "cfl", "sgs"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    arr = cfg.setdefault("arr", {})
    arr.update(_parse_kv(args.arr))
    if args.no_arr:
        arr["enabled"] = False
    if args.start_tier is not None:
        arr["start_tier"] = args.start_tier
    return cfg


def cmd_run(args):
    from .io import frame_png

    cfg = build_config(args)
    try:
        sim = Simulation(cfg, on_event=None)
    except (ValueError, BackendError) as exc:
        print("error:", exc, file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    print(f"case {sim.case.key}  tiers {' / '.join('x'.join(map(str, s)) for s in sim.tiers)}  "
          f"backend {sim.be.describe()}")
    n_frames = max(0, args.frames)
    next_frame, frame_no, seen = 0.0, 0, 0
    t0 = last = time.perf_counter()
    while not sim.done:
        until = min(sim.t_end, next_frame) if n_frames and out else None
        sim.advance(max_wall=1.0, until=until if until and until > sim.t else None)
        for ev in sim.events[seen:]:
            extra = {k: v for k, v in ev.items() if k not in ("type", "t", "step")}
            print(f"  [event] t={ev['t']:.4f} {ev['type']} "
                  + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in extra.items()))
        seen = len(sim.events)
        if n_frames and out and sim.t >= next_frame - 1e-9:
            (out / f"frame_{frame_no:04d}.png").write_bytes(frame_png(sim))
            frame_no += 1
            next_frame = frame_no * sim.t_end / max(n_frames - 1, 1)
        now = time.perf_counter()
        if now - last > 2.0 or sim.done:
            last = now
            h = sim.history
            print(f"  t={sim.t:9.4f}/{sim.t_end:g}  grid {'x'.join(map(str, sim.shape))}  "
                  f"E={h['energy'][-1]:.5g}  eps={h['dissipation'][-1]:.4g}  "
                  f"P={h['pressure'][-1]:.2e}  wall={now - t0:.0f}s", flush=True)
    summ = sim.summary()
    c = summ["cost"]
    print(f"{sim.status}: {sim.ctrl.total_steps} steps, {sim.wall:.1f}s wall, "
          f"{c['promotions']} promotion(s), {c['rollbacks']} rollback(s), "
          f"{c['demotions']} demotion(s), limiter fired {c['guard_hits']}x, "
          f"est. {c['cell_step_saving_est']:.2f}x fewer cell-steps than uniform finest")
    if summ["time_saturated"] > 0:
        print(f"note: {summ['time_saturated']:.3g} time units were flagged under-resolved "
              "on the finest tier; add a finer tier for a fully resolved result")
    if sim.error:
        print("error:", sim.error, file=sys.stderr)
    if out:
        (out / "summary.json").write_text(json.dumps(json_safe(summ), indent=2))
        sim.export_csv(out / "history.csv")
        sim.export_npz(out / "final.npz")
        if args.vtk:
            sim.export_vtk(out / "final.vtk")
        (out / "final.png").write_bytes(frame_png(sim))
        print("wrote", out)
    return 0 if sim.status == "done" else 1


def cmd_validate(args):
    from .validation import format_results, run_validation

    try:
        be = get_backend(args.device or "auto", args.precision or "double")
    except BackendError as exc:
        print("error:", exc, file=sys.stderr)
        return 2
    print("Validating on", be.describe())
    res = run_validation(be, on_result=lambda r: print(format_results([r]).rsplit("\n", 1)[0],
                                                        flush=True))
    ok = sum(r["passed"] for r in res)
    print(f"{ok}/{len(res)} checks passed")
    if args.json:
        Path(args.json).write_text(json.dumps(json_safe(res), indent=2))
    return 0 if ok == len(res) else 1


def cmd_bench(args):
    from .validation import throughput

    try:
        be = get_backend(args.device or "auto", args.precision or "double")
    except BackendError as exc:
        print("error:", exc, file=sys.stderr)
        return 2
    grids = None
    if args.large:
        grids = [("taylor_green_3d", (64,) * 3), ("taylor_green_3d", (128,) * 3),
                 ("taylor_green_3d", (256,) * 3), ("shear_layer", (1024, 1024)),
                 ("shear_layer", (2048, 2048))]
    print("Throughput on", be.describe())
    print(f"{'grid':>16} {'s/step':>10} {'Mcells/s':>10}")
    rows = throughput(be, grids)
    for r in rows:
        print(f"{'x'.join(map(str, r['shape'])):>16} {r['seconds_per_step']:10.4f} "
              f"{r['cells_per_second'] / 1e6:10.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(json_safe({"backend": be.describe(), "rows": rows}),
                                              indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omnisvox-cfd", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="show version and available backends").set_defaults(fn=cmd_info)
    sub.add_parser("cases", help="list cases and their parameters").set_defaults(fn=cmd_cases)

    r = sub.add_parser("run", help="run a simulation")
    r.add_argument("--config", help="JSON config file (flags below override it)")
    r.add_argument("--case", choices=list(CASES))
    r.add_argument("--param", action="append", metavar="NAME=VALUE")
    r.add_argument("--preset", help="tier preset name (see 'cases')")
    r.add_argument("--tiers", help="explicit tiers, e.g. 32x32x32,64x64x64")
    r.add_argument("--t-end", dest="t_end", type=float)
    r.add_argument("--device", choices=["auto", "cpu", "gpu"])
    r.add_argument("--precision", choices=["double", "single"])
    r.add_argument("--cfl", type=float)
    r.add_argument("--sgs", choices=["none", "smagorinsky"])
    r.add_argument("--no-arr", action="store_true", help="fixed grid (use --start-tier)")
    r.add_argument("--start-tier", type=int)
    r.add_argument("--arr", action="append", metavar="NAME=VALUE",
                   help="adaptive-controller option, e.g. press_up=0.03")
    r.add_argument("--out", help="output directory")
    r.add_argument("--frames", type=int, default=0, help="number of PNG frames to save")
    r.add_argument("--vtk", action="store_true", help="also write final.vtk")
    r.set_defaults(fn=cmd_run)

    for name, fn, hlp in (("validate", cmd_validate, "run the self-validation suite"),
                          ("bench", cmd_bench, "measure throughput")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--device", choices=["auto", "cpu", "gpu"])
        s.add_argument("--precision", choices=["double", "single"])
        s.add_argument("--json", help="write results to this file")
        if name == "bench":
            s.add_argument("--large", action="store_true", help="GPU-sized grids")
        s.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
