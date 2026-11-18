"""Turn benchmarks/results/*.json into figures and benchmarks/REPORT.md.

    python benchmarks/make_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnisvox_cfd import get_backend, resample_hat, SpectralGrid  # noqa: E402
from omnisvox_cfd.io import field_to_rgb  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, RULE = "#15202b", "#5b6875", "#d3dae1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]      # fixed order
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "figure.dpi": 130, "savefig.bbox": "tight",
    "legend.frameon": False,
})
BE = get_backend("cpu")


def load(name):
    p = RES / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("history") else None


def H(d, key):
    return np.array([np.nan if v is None else v for v in d["history"][key]], dtype=float)


def cell_steps(d):
    return d["summary"]["cost"]["cell_steps"]


def uses_sgs(d):
    return d["config"].get("sgs") == "smagorinsky" or any(
        e["type"] == "sgs_on" for e in d["summary"]["events"])


def total_dissipation(d):
    """(t, eps).  Without a sub-grid model eps = 2 nu Z exactly (checked by the energy-budget
    test).  With one, the model removes energy too, so eps = -dE/dt from the energy history."""
    t, e = H(d, "t"), H(d, "energy")
    if not uses_sgs(d):
        return t, H(d, "dissipation")
    tg = np.linspace(t[0], t[-1], 241)
    eg = np.interp(tg, t, e)
    return tg, -np.gradient(eg, tg)


def peak(d):
    t, e = total_dissipation(d)
    i = int(np.nanargmax(e))
    return e[i], t[i]


def curve_diff(d, ref):
    """RMS difference of the dissipation history from a reference, relative to its peak."""
    (ta, ea), (tb, eb) = total_dissipation(d), total_dissipation(ref)
    t = np.linspace(0, min(ta[-1], tb[-1]), 600)
    a, b = np.interp(t, ta, ea), np.interp(t, tb, eb)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.max(np.abs(b)))


def tier_axis(ax, d, color=SERIES[0]):
    t, tier = H(d, "t"), H(d, "tier")
    ax.step(t, tier, where="post", color=color)
    tiers = d["summary"]["tiers"]
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(["×".join(map(str, s)) for s in tiers])
    ax.set_ylim(-0.4, len(tiers) - 0.6)
    for e in d["summary"]["events"]:
        if e["type"] == "rollback_promote":
            ax.plot(e["t_restore"], e["to_tier"], marker="^", color=SERIES[1], ms=7,
                    mec="white", mew=1, ls="none")
        elif e["type"] == "demote":
            ax.plot(e["t"], e["to_tier"], marker="v", color=SERIES[2], ms=7, mec="white",
                    mew=1, ls="none")
    ax.set_ylabel("grid in use")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
def section_validation():
    p = RES / "validation.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    rows = [[("pass" if r["passed"] else "**FAIL**"), r["name"], f"{r['value']:.3g}",
             "" if r["tolerance"] in (None, 0) else f"{r['tolerance']:g}", r["note"]]
            for r in d["results"]]
    n = sum(r["passed"] for r in d["results"])
    return (f"## 1. Correctness checks\n\n{n} of {len(d['results'])} passed on {d['backend']}. "
            "Re-run any time with `omnisvox-cfd validate`.\n\n"
            + md_table(["", "check", "value", "tolerance", "what is measured"], rows) + "\n")


def section_tgv(Re, title_no):
    names = {"uniform32": f"tgv{Re}_uniform32", "uniform64": f"tgv{Re}_uniform64",
             "uniform128": f"tgv{Re}_uniform128", "arr": f"tgv{Re}_arr",
             "draft": f"tgv{Re}_arr_draft", "les64": f"tgv{Re}_les64",
             "sgs64": f"tgv{Re}_arr_sgs64"}
    runs = {k: load(v) for k, v in names.items()}
    ref = runs["uniform128"]
    if ref is None:
        return ""
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.1]})
    plot = [("uniform128", "uniform 128³", SERIES[0], "-"), ("uniform64", "uniform 64³", SERIES[2], "-"),
            ("uniform32", "uniform 32³", SERIES[3], "-"), ("arr", "adaptive 32³/64³/128³", SERIES[1], "--")]
    for key, label, color, ls in plot:
        d = runs.get(key)
        if d:
            a0.plot(*total_dissipation(d), color=color, ls=ls, label=label)
    if Re == 1600:
        a0.plot([9.0], [0.0129], marker="o", color=INK, ms=6, ls="none",
                label="512³ DNS peak (about 0.013 near t = 9)")
    a0.set_ylabel("dissipation rate")
    a0.legend(loc="upper left", fontsize=9)
    a0.set_title(f"Taylor–Green vortex, Re = {Re}", loc="left", fontsize=11, color=INK)
    if runs["arr"]:
        tier_axis(a1, runs["arr"], SERIES[1])
    a1.set_xlabel("time")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"tgv{Re}.png")
    plt.close(fig)

    rows = []
    labels = [("uniform128", "uniform 128³"), ("uniform64", "uniform 64³"), ("uniform32", "uniform 32³"),
              ("arr", "adaptive 32/64/128, standard"), ("draft", "adaptive 32/64/128, draft"),
              ("les64", "uniform 64³ + Smagorinsky"), ("sgs64", "adaptive 16/32/64, sub-grid fallback")]
    for key, label in labels:
        d = runs.get(key)
        if not d:
            continue
        pk, tp = peak(d)
        c = d["summary"]["cost"]
        e_err = abs(H(d, "energy")[-1] / H(ref, "energy")[-1] - 1)
        rows.append([label, f"{pk:.5f}", f"{tp:.2f}", f"{100 * curve_diff(d, ref):.2f} %",
                     f"{100 * e_err:.2f} %",
                     f"{cell_steps(d) / cell_steps(ref):.3f}", c["rollbacks"],
                     f"{d['summary']['time_saturated']:.1f}", c["guard_hits"]])
    txt = (f"## {title_no}. Taylor–Green vortex, Re = {Re}\n\n![TGV Re={Re}](figures/tgv{Re}.png)\n\n"
           + "Dissipation is 2νZ, except for runs with the sub-grid model, where it is −dE/dt "
             "(the model removes energy too).\n\n"
           + md_table(["run", "peak dissipation", "at t", "RMS diff. of dissipation history",
                       "final energy diff.", "cell-steps vs uniform 128³", "rollbacks", "time flagged under-resolved",
                       "limiter firings"], rows) + "\n")
    return txt, runs


def field_error(a, b):
    """Relative L2 difference of two periodic 2-D fields on possibly different grids."""
    ga = SpectralGrid(a.shape, (1.0, 1.0), BE)
    gb = SpectralGrid(b.shape, (1.0, 1.0), BE)
    diff = resample_hat(ga.fft(a), a.shape, b.shape, BE) - gb.fft(b)
    return float(np.sqrt(gb.mean_sq(diff) / gb.mean_sq(gb.fft(b))))


def section_shear(no):
    names = ["shear_uniform128", "shear_uniform256", "shear_uniform512", "shear_arr",
             "shear_arr_norollback", "shear_arr_deep", "shear_arr_strict"]
    runs = {n: load(n) for n in names}
    fields = {n: np.load(RES / f"{n}_field.npy") for n in names if (RES / f"{n}_field.npy").exists()}
    ref_name = "shear_ref1024" if (RES / "shear_ref1024_field.npy").exists() else "shear_uniform512"
    if ref_name not in fields and ref_name != "shear_ref1024":
        return ""
    ref = np.load(RES / f"{ref_name}_field.npy")
    ref_run = runs.get("shear_uniform512")
    labels = {"shear_uniform128": "uniform 128²", "shear_uniform256": "uniform 256²",
              "shear_uniform512": "uniform 512²", "shear_arr": "adaptive, rollback on",
              "shear_arr_norollback": "adaptive, refine in place",
              "shear_arr_deep": "adaptive, deep rollback",
              "shear_arr_strict": "adaptive, strict"}
    show = [n for n in names if n in fields]
    pics = show[:4]
    fig, axes = plt.subplots(1, len(pics), figsize=(2.6 * len(pics), 2.9))
    lim = float(np.percentile(np.abs(ref), 99.5))
    for ax, n in zip(np.atleast_1d(axes), pics):
        ax.imshow(field_to_rgb(fields[n], True, (-lim, lim)), interpolation="nearest")
        ax.set_title(labels[n], fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Double shear layer, vorticity at t = 1.8 (Re = 30 000, ρ = 100)", x=0.01, ha="left",
                 fontsize=11)
    fig.savefig(FIG / "shear.png")
    plt.close(fig)
    rows = []
    for n in show:
        d = runs[n]
        c = d["summary"]["cost"]
        err = field_error(fields[n], ref) if n != ref_name else 0.0
        rows.append([labels[n], "reference" if n == ref_name else f"{100 * err:.2f} %",
                     f"{cell_steps(d) / cell_steps(ref_run):.3f}" if ref_run else "",
                     c["rollbacks"], c["wasted"]["steps"], f"{d['summary']['time_saturated']:.2f}",
                     c["guard_hits"]])
    ref_label = "a uniform 1024² run" if ref_name == "shear_ref1024" else "the uniform 512² run"
    return (f"## {no}. Double shear layer (2-D)\n\n![shear layer](figures/shear.png)\n\n"
            f"Vorticity error is the relative L2 difference from {ref_label} at t = 1.8.\n\n"
            + md_table(["run", "vorticity error", "cell-steps vs uniform 512²", "rollbacks",
                        "steps redone", "time flagged", "limiter firings"], rows) + "\n")


def section_turb(no):
    arr, fine, coarse = load("turb2d_arr"), load("turb2d_uniform256"), load("turb2d_uniform64")
    if not (arr and fine):
        return ""
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.1]})
    a0.plot(H(fine, "t"), H(fine, "enstrophy"), color=SERIES[0], label="uniform 256²")
    if coarse:
        a0.plot(H(coarse, "t"), H(coarse, "enstrophy"), color=SERIES[3], label="uniform 64²")
    a0.plot(H(arr, "t"), H(arr, "enstrophy"), color=SERIES[1], ls="--", label="adaptive 64²/128²/256²")
    a0.set_yscale("log")
    a0.set_ylabel("enstrophy")
    a0.legend(fontsize=9)
    a0.set_title("Decaying 2-D turbulence, Re = 1000", loc="left", fontsize=11, color=INK)
    tier_axis(a1, arr, SERIES[1])
    a1.set_xlabel("time")
    fig.savefig(FIG / "turb2d.png")
    plt.close(fig)
    rows = []
    for d, label in ((fine, "uniform 256²"), (coarse, "uniform 64²"), (arr, "adaptive 64/128/256")):
        if not d:
            continue
        c = d["summary"]["cost"]
        rows.append([label, f"{100 * abs(H(d, 'enstrophy')[-1] / H(fine, 'enstrophy')[-1] - 1):.2f} %",
                     f"{100 * abs(H(d, 'energy')[-1] / H(fine, 'energy')[-1] - 1):.2f} %",
                     f"{cell_steps(d) / cell_steps(fine):.3f}",
                     c["rollbacks"], c["demotions"], c["guard_hits"]])
    return (f"## {no}. Decaying 2-D turbulence\n\n![2-D turbulence](figures/turb2d.png)\n\n"
            + md_table(["run", "final enstrophy diff.", "final energy diff.",
                        "cell-steps vs uniform 256²", "rollbacks", "coarsenings",
                        "limiter firings"], rows)
            + "\n\nThe flow is chaotic, so fields are compared through their statistics, not "
              "point by point.\n")


def strouhal(d, t_from):
    t, cl = H(d, "t"), H(d, "cl")
    m = t > t_from
    ti = np.linspace(t[m][0], t[m][-1], 8192)
    ci = np.interp(ti, t[m], cl[m])
    ci -= ci.mean()
    up = ti[1:][(ci[:-1] < 0) & (ci[1:] >= 0)]
    # refine crossings by linear interpolation
    idx = np.where((ci[:-1] < 0) & (ci[1:] >= 0))[0]
    zc = ti[idx] - ci[idx] * (ti[idx + 1] - ti[idx]) / (ci[idx + 1] - ci[idx])
    if len(zc) < 3:
        return None
    periods = np.diff(zc)
    return {"St": float(1.0 / periods.mean()), "periods": len(periods),
            "spread": float(periods.std() / periods.mean()),
            "cl_rms": float(np.sqrt(np.mean(ci ** 2))), "cd": float(np.nanmean(H(d, "cd")[m]))}


def section_cylinder(no):
    runs = [("cyl_256x128", "256×128, domain 32 D × 16 D (8 cells per D)"),
            ("cyl_512x256", "512×256, domain 32 D × 16 D (16 cells per D)"),
            ("cyl_512x512_tall", "512×512, domain 32 D × 32 D (16 cells per D)"),
            ("cyl_1024x512_restart", "1024×512, domain 32 D × 16 D (32 cells per D), continued from the 512×256 flow at t = 40"),
            ("cyl_arr", "adaptive 256×128 / 512×256 / 1024×512")]
    rows, first = [], None
    for name, label in runs:
        d = load(name)
        if not d:
            continue
        if name == "cyl_512x256" or first is None:
            first = d
        s = strouhal(d, d["t_switch"] + 15.0 if "t_switch" in d else 0.5 * H(d, "t")[-1])
        if not s:
            continue
        late = H(d, "t") > 0.5 * H(d, "t")[-1]
        grids = sorted({int(v) for v in H(d, "tier")[late]})
        tiers = d["summary"]["tiers"]
        rows.append([label, f"{s['St']:.4f}", f"{100 * (s['St'] / 0.164 - 1):+.1f} %", s["periods"],
                     f"{s['cd']:.2f}", f"{s['cl_rms']:.3f}",
                     ", ".join("×".join(map(str, tiers[g])) for g in grids),
                     d["summary"]["cost"]["guard_hits"]])
    if not rows:
        return ""
    d = first
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(H(d, "t"), H(d, "cl"), color=SERIES[0])
    ax.set_xlabel("time  (D/U)")
    ax.set_ylabel("lift coefficient")
    ax.set_title("Cylinder at Re = 100: lift history, 512×256", loc="left", fontsize=11, color=INK)
    fig.savefig(FIG / "cylinder_lift.png")
    plt.close(fig)
    img = ""
    if (RES / "cyl_512x256_field.npy").exists():
        f = np.load(RES / "cyl_512x256_field.npy")
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.imshow(field_to_rgb(f, True, (-2.0, 2.0)), interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title("Vorticity, end of run (fringe zone at the right edge)", loc="left",
                     fontsize=11, color=INK)
        fig.savefig(FIG / "cylinder_vorticity.png")
        plt.close(fig)
        img = "![cylinder wake](figures/cylinder_vorticity.png)\n\n"
    return (f"## {no}. Circular cylinder, Re = 100\n\n{img}![lift](figures/cylinder_lift.png)\n\n"
            "Strouhal number from the mean period between upward zero crossings of the lift over "
            "the second half of each run. Reference: St = 0.164 (Williamson 1989, isolated "
            "cylinder); published C_d is about 1.34 and rms C_l about 0.23.\n\n"
            + md_table(["run", "St", "vs 0.164", "periods", "mean C_d", "rms C_l", "grid during measurement",
                        "limiter firings"], rows) + "\n")


def section_timing(no):
    pairs = [("timing_tgv400_uniform64", "timing_tgv400_arr", "Taylor–Green Re = 400 to t = 10, 64³ finest"),
             ("timing_turb2d_uniform256", "timing_turb2d_arr", "2-D turbulence Re = 1000 to t = 30, 256² finest")]
    rows = []
    for u, a, label in pairs:
        du, da = load(u), load(a)
        if not (du and da):
            continue
        rows.append([label, f"{du['summary']['wall_seconds']:.1f} s", f"{da['summary']['wall_seconds']:.1f} s",
                     f"{du['summary']['wall_seconds'] / da['summary']['wall_seconds']:.2f}×",
                     f"{cell_steps(du) / cell_steps(da):.2f}×",
                     f"{100 * abs(H(da, 'enstrophy')[-1] / H(du, 'enstrophy')[-1] - 1):.2f} %"])
    thr = ""
    p = RES / "throughput.json"
    if p.exists():
        d = json.loads(p.read_text())
        trow = [["×".join(map(str, r["shape"])), f"{r['seconds_per_step']:.4f}",
                 f"{r['cells_per_second'] / 1e6:.2f}"] for r in d["rows"]]
        thr = (f"\nThroughput on {d['backend']}:\n\n"
               + md_table(["grid", "seconds per step", "million cell-updates per second"], trow) + "\n")
    if not rows and not thr:
        return ""
    body = ""
    if rows:
        body = ("Head-to-head wall-clock, run back to back on an otherwise idle machine:\n\n"
                + md_table(["case", "uniform finest grid", "adaptive", "wall-clock ratio",
                            "cell-step ratio", "final enstrophy difference"], rows) + "\n")
    return f"## {no}. Speed\n\n{body}{thr}"


def main():
    FIG.mkdir(exist_ok=True)
    parts = [section_validation()]
    n = 2
    for Re in (1600, 400):
        out = section_tgv(Re, n)
        if out:
            parts.append(out[0])
            n += 1
    for fn in (section_shear, section_turb, section_cylinder, section_timing):
        s = fn(n)
        if s:
            parts.append(s)
            n += 1
    head = (HERE / "REPORT_head.md").read_text() if (HERE / "REPORT_head.md").exists() else "# Benchmark report\n"
    (HERE / "REPORT.md").write_text(head + "\n" + "\n".join(p for p in parts if p))
    print("wrote", HERE / "REPORT.md", "with", len([p for p in parts if p]), "sections")


if __name__ == "__main__":
    main()
