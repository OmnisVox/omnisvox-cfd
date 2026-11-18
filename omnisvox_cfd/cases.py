"""Case library: geometry, initial conditions and per-case defaults.

Every case can build a solver at any tier resolution, so the adaptive
controller can move a run between grids.  Analytic initial conditions are
sampled directly at the requested resolution; random ones are generated on a
fixed reference grid and transferred spectrally so that every tier starts from
the *same* field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backend import Backend, get_backend
from .solver2d import NavierStokes2D
from .solver3d import NavierStokes3D
from .spectral import resample_hat


@dataclass
class Param:
    name: str
    label: str
    default: Any
    min: float | None = None
    max: float | None = None
    choices: list | None = None
    help: str = ""

    def coerce(self, value):
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(f"{self.name} must be one of {self.choices}")
            return value
        try:
            v = type(self.default)(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: cannot interpret {value!r}") from None
        if not np.isfinite(v):
            raise ValueError(f"{self.name} must be finite")
        if self.min is not None and v < self.min or self.max is not None and v > self.max:
            raise ValueError(f"{self.name} must lie in [{self.min}, {self.max}]")
        return v

    def to_dict(self):
        return {k: getattr(self, k) for k in ("name", "label", "default", "min", "max",
                                              "choices", "help")}


class Case:
    key = ""
    title = ""
    ndim = 2
    summary = ""
    validation = ""               # what this case is checked against
    params: list[Param] = []
    tier_presets: dict[str, list[tuple]] = {}
    default_preset = ""
    t_end = 1.0
    default_field = "vorticity"
    energy_monotone = False       # unforced, body-free: kinetic energy can only decay
    # Promotion threshold ("standard" accuracy).  Measured against fine-grid
    # references, the pointwise vorticity error is roughly 3-7x the pressure,
    # the velocity error roughly 0.7x, integrated statistics roughly 0.1x.
    press_up = 0.005              # 2-D default; 3-D cases override
    # Flows that start already turbulent: under-resolution during the initial
    # cascade leaves a lasting error that the saturation trigger cannot see in
    # time, so ARR starts them on the finest tier and coarsens as they decay.
    start_fine = False
    symmetric_cmap = True

    # -- parameters ----------------------------------------------------
    def resolve_params(self, given: dict | None) -> dict:
        given = dict(given or {})
        out = {}
        for p in self.params:
            out[p.name] = p.coerce(given.pop(p.name)) if p.name in given else p.default
        if given:
            raise ValueError(f"unknown parameter(s) for case '{self.key}': {sorted(given)}")
        return out

    def lengths(self, p, shape=None):
        return (2 * np.pi,) * self.ndim

    def nu(self, p):
        return 1.0 / p["Re"]

    def solver_kwargs(self, shape, p) -> dict:
        return {}

    def make_solver(self, shape, p, be: Backend, **kw):
        cls = NavierStokes2D if self.ndim == 2 else NavierStokes3D
        args = self.solver_kwargs(shape, p)
        args.update(kw)
        return cls(shape, self.lengths(p, shape), self.nu(p), be, **args)

    def initial_state(self, solver, p):
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "key": self.key, "title": self.title, "ndim": self.ndim, "summary": self.summary,
            "validation": self.validation, "params": [q.to_dict() for q in self.params],
            "tier_presets": {k: [list(s) for s in v] for k, v in self.tier_presets.items()},
            "default_preset": self.default_preset, "t_end": self.t_end,
            "fields": list((NavierStokes2D if self.ndim == 2 else NavierStokes3D).field_names),
            "default_field": self.default_field, "press_up": self.press_up,
            "start_fine": self.start_fine,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _square_presets(sizes, ndim):
    out = {}
    for group in sizes:
        out[" / ".join(f"{n}^{ndim}" for n in group)] = [(n,) * ndim for n in group]
    return out


def _vorticity_state(solver, omega: np.ndarray):
    g = solver.grid
    w_hat = g.fft(solver.be.asarray(omega, dtype=solver.be.rdtype)) * g.dealias
    w_hat[0, 0] = 0.0
    return w_hat


def _velocity_state(solver, u: np.ndarray):
    g = solver.grid
    u_hat = g.fft(solver.be.asarray(u, dtype=solver.be.rdtype)) * g.dealias
    return solver._project(u_hat)


def _from_reference(solver, ref_hat: np.ndarray, ref_shape):
    """Transfer a host spectrum from the reference grid onto the solver's grid."""
    host = get_backend("cpu")
    out = resample_hat(ref_hat, ref_shape, solver.shape, host)
    out = solver.be.asarray(out, dtype=solver.be.cdtype)
    return out * solver.grid.dealias


# ---------------------------------------------------------------------------
# 2-D cases
# ---------------------------------------------------------------------------
class TaylorGreen2D(Case):
    key = "taylor_green_2d"
    title = "Taylor-Green vortex (2-D, exact)"
    summary = "Steady vortex array that decays as exp(-2 nu t). An exact Navier-Stokes solution."
    validation = "Exact solution: energy(t) = energy(0) exp(-4 t / Re)."
    params = [Param("Re", "Reynolds number (1/nu)", 100.0, 1.0, 1e6)]
    tier_presets = _square_presets([(32, 64, 128), (64, 128, 256)], 2)
    default_preset = "32^2 / 64^2 / 128^2"
    t_end = 10.0
    energy_monotone = True

    def initial_state(self, solver, p):
        x, y = solver.grid.coords()
        return _vorticity_state(solver, 2.0 * np.sin(x) * np.sin(y))


class ShearLayer(Case):
    key = "shear_layer"
    title = "Double shear layer (2-D)"
    summary = ("Two perturbed shear layers roll up into vortices. Under-resolved grids "
               "produce spurious extra vortices - a classic resolution test.")
    validation = "Brown & Minion (1995) thin shear layer; compared against a fine uniform grid."
    params = [
        Param("Re", "Reynolds number (1/nu)", 10000.0, 100.0, 1e6),
        Param("rho", "Shear-layer sharpness", 80.0, 10.0, 400.0,
              help="Layer thickness is about 1/rho of the box."),
        Param("delta", "Perturbation amplitude", 0.05, 0.0, 0.5),
    ]
    tier_presets = _square_presets([(128, 256, 512), (64, 128, 256), (256, 512, 1024)], 2)
    default_preset = "128^2 / 256^2 / 512^2"
    t_end = 1.8
    energy_monotone = True

    def lengths(self, p, shape=None):
        return (1.0, 1.0)

    def initial_state(self, solver, p):
        x, y = solver.grid.coords()
        rho, delta = p["rho"], p["delta"]
        lower = y <= 0.5
        arg = np.where(lower, rho * (y - 0.25), rho * (0.75 - y))
        dudy = np.where(lower, rho, -rho) / np.cosh(arg) ** 2
        omega = 2 * np.pi * delta * np.cos(2 * np.pi * x) - dudy
        return _vorticity_state(solver, omega)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        s = 1.0 / (1.0 + np.exp(1.0 / (x - 1.0) + 1.0 / x))
    return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, s))


class Obstacle(Case):
    key = "obstacle"
    title = "Flow past a body (2-D)"
    summary = ("Uniform stream past a cylinder, square, ellipse or NACA aerofoil with vortex "
               "shedding. Immersed-boundary body, fringe-zone outflow.")
    validation = ("Circular cylinder, Re = 100: Strouhal number against Williamson's "
                  "universal curve (St = 0.164).")
    params = [
        Param("Re", "Reynolds number (U D / nu)", 100.0, 10.0, 20000.0),
        Param("body", "Body shape", "cylinder", choices=["cylinder", "square", "ellipse", "naca"]),
        Param("angle", "Angle of attack (deg)", 0.0, -45.0, 45.0,
              help="Used by square, ellipse and NACA bodies."),
        Param("thickness", "Thickness ratio", 0.12, 0.04, 1.0,
              help="Ellipse: minor/major axis. NACA: thickness/chord (0.12 = NACA 0012)."),
    ]
    tier_presets = {
        "256x128 / 512x256 / 1024x512": [(256, 128), (512, 256), (1024, 512)],
        "512x256 / 1024x512": [(512, 256), (1024, 512)],
        "128x64 / 256x128 / 512x256": [(128, 64), (256, 128), (512, 256)],
    }
    default_preset = "256x128 / 512x256 / 1024x512"
    t_end = 120.0
    LX, XC = 32.0, 8.0

    def lengths(self, p, shape=None):
        """Square cells: the domain is 32 D long and 32 D * Ny/Nx tall."""
        if shape is None:
            shape = self.tier_presets[self.default_preset][0]
        return (self.LX, self.LX * shape[1] / shape[0])

    def signed_distance(self, x, y, p, yc=8.0):
        """Approximate signed distance to the body surface (positive outside)."""
        rx, ry = x - self.XC, y - yc
        a = np.deg2rad(p["angle"])
        xb = rx * np.cos(a) - ry * np.sin(a)        # along the chord (nose-up for a > 0)
        yb = rx * np.sin(a) + ry * np.cos(a)
        body = p["body"]
        if body == "cylinder":
            return np.hypot(rx, ry) - 0.5
        if body == "square":
            return np.maximum(np.abs(xb), np.abs(yb)) - 0.5
        if body == "ellipse":
            sa, sb = 0.5, 0.5 * p["thickness"]
            return (np.sqrt((xb / sa) ** 2 + (yb / sb) ** 2) - 1.0) * sb
        # symmetric NACA 4-digit section, unit chord centred on (XC, YC)
        xc = np.clip(xb + 0.5, 0.0, 1.0)
        tt = p["thickness"]
        yt = 5 * tt * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                       + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
        return np.maximum(np.abs(yb) - yt, np.maximum(-(xb + 0.5), xb - 0.5))

    def solver_kwargs(self, shape, p):
        nx, ny = shape
        dx = self.LX / nx
        x = (np.arange(nx) * dx)[:, None]
        y = (np.arange(ny) * dx)[None, :]
        sd = self.signed_distance(x + 0 * y, y + 0 * x, p, yc=0.5 * ny * dx)
        # Smooth mask (keeps the spectral representation clean).  A smoothed
        # Brinkman body acts slightly larger than its mask, so the mask is drawn
        # one cell inside the surface; the residual error shrinks with dx.
        mask = 0.5 * (1.0 - np.tanh((sd + dx) / (0.75 * dx)))
        # fringe: rises over x in [26, 28], holds, falls over [31, 32]
        prof = _smoothstep((x - 26.0) / 2.0) - _smoothstep((x - 31.0) / 1.0)
        fringe = 4.0 * prof + 0 * y
        # resolution indicator ignores the body's own (always sharp) surface layer
        window = (0.5 * (1.0 + np.tanh((sd - 0.35) / 0.1))) * (1.0 - np.clip(prof, 0, 1) + 0 * y)
        return {"mean_flow": (1.0, 0.0), "body_mask": mask, "fringe_rate": fringe,
                "indicator_window": window, "ref_length": 1.0}

    def initial_state(self, solver, p):
        x, y = solver.grid.coords()
        yc = 0.5 * solver.grid.lengths[1]
        # off-axis vortex just behind the body to break symmetry
        blob = 2.0 * np.exp(-((x - self.XC - 1.5) ** 2 + (y - yc - 0.5) ** 2) / 0.25)
        return _vorticity_state(solver, blob)


class Turbulence2D(Case):
    key = "turbulence_2d"
    title = "Decaying turbulence (2-D)"
    summary = ("Random vortices merge and filament. Fine scales fade as the flow decays, "
               "so the adaptive grid can step back down.")
    validation = "Adaptive run compared against a uniform finest-grid run."
    params = [
        Param("Re", "Reynolds number (1/nu, unit rms velocity)", 5000.0, 100.0, 1e6),
        Param("k0", "Initial energy peak wavenumber", 8, 2, 24),
        Param("seed", "Random seed", 1, 0, 2 ** 31 - 1),
    ]
    tier_presets = _square_presets([(128, 256, 512), (64, 128, 256), (256, 512, 1024)], 2)
    default_preset = "128^2 / 256^2 / 512^2"
    t_end = 20.0
    energy_monotone = True
    start_fine = True
    REF = (128, 128)

    def initial_state(self, solver, p):
        n = self.REF[0]
        rng = np.random.default_rng(int(p["seed"]))
        kx = np.fft.fftfreq(n, 1.0 / n)[:, None]
        ky = np.arange(n // 2 + 1)[None, :]
        k = np.hypot(kx, ky)
        k0 = float(p["k0"])
        amp = k ** 2.5 * np.exp(-0.5 * (k / k0) ** 2)      # vorticity amplitude
        amp[k > n / 3.0 - 1] = 0.0
        w_hat = amp * np.exp(2j * np.pi * rng.random(k.shape))
        omega = np.fft.irfft2(w_hat, s=self.REF)
        w_hat = np.fft.rfft2(omega)
        k2 = k ** 2
        k2[0, 0] = 1.0
        wgt = np.full(k.shape, 2.0)
        wgt[:, 0] = 1.0
        wgt[:, -1] = 1.0
        urms2 = np.sum(wgt * np.abs(w_hat) ** 2 / k2) / n ** 4     # <u^2 + v^2>
        w_hat *= 1.0 / np.sqrt(urms2)
        w_hat[0, 0] = 0.0
        return _from_reference(solver, w_hat, self.REF)


# ---------------------------------------------------------------------------
# 3-D cases
# ---------------------------------------------------------------------------
class TaylorGreen3D(Case):
    key = "taylor_green_3d"
    title = "Taylor-Green vortex (3-D)"
    ndim = 3
    press_up = 0.03
    summary = ("Smooth vortex that stretches, breaks down into turbulence near t = 9 and "
               "then decays. The standard transition benchmark.")
    validation = ("Re = 1600: dissipation-rate history against the 512^3 pseudo-spectral "
                  "reference of van Rees et al. (2011).")
    params = [Param("Re", "Reynolds number", 1600.0, 10.0, 1e5)]
    tier_presets = _square_presets([(32, 64, 128), (16, 32, 64), (64, 128, 256)], 3)
    default_preset = "32^3 / 64^3 / 128^3"
    t_end = 20.0
    default_field = "vorticity_mag"
    energy_monotone = True
    symmetric_cmap = False

    def initial_state(self, solver, p):
        x, y, z = solver.grid.coords()
        u = np.stack((np.sin(x) * np.cos(y) * np.cos(z),
                      -np.cos(x) * np.sin(y) * np.cos(z),
                      np.zeros_like(x)))
        return _velocity_state(solver, u)


class BeltramiABC(Case):
    key = "abc_flow"
    title = "ABC Beltrami flow (3-D, exact)"
    ndim = 3
    press_up = 0.03
    summary = "Arnold-Beltrami-Childress flow: an exact 3-D solution that decays as exp(-nu t)."
    validation = "Exact solution: energy(t) = energy(0) exp(-2 t / Re)."
    params = [Param("Re", "Reynolds number (1/nu)", 50.0, 1.0, 1e5)]
    tier_presets = _square_presets([(16, 32, 64), (32, 64, 128)], 3)
    default_preset = "16^3 / 32^3 / 64^3"
    t_end = 10.0
    default_field = "speed"
    energy_monotone = True
    symmetric_cmap = False

    def initial_state(self, solver, p):
        x, y, z = solver.grid.coords()
        a, b, c = 1.0, 0.8, 0.6
        u = np.stack((a * np.sin(z) + c * np.cos(y),
                      b * np.sin(x) + a * np.cos(z),
                      c * np.sin(y) + b * np.cos(x)))
        return _velocity_state(solver, u)


class Turbulence3D(Case):
    key = "turbulence_3d"
    title = "Decaying isotropic turbulence (3-D)"
    ndim = 3
    press_up = 0.03
    start_fine = True
    summary = "Random solenoidal velocity field that cascades to small scales and decays."
    validation = "Adaptive run compared against a uniform finest-grid run."
    params = [
        Param("Re", "Reynolds number (1/nu, unit rms velocity)", 300.0, 10.0, 1e5),
        Param("seed", "Random seed", 1, 0, 2 ** 31 - 1),
    ]
    tier_presets = _square_presets([(32, 64, 128), (16, 32, 64), (64, 128, 256)], 3)
    default_preset = "32^3 / 64^3 / 128^3"
    t_end = 10.0
    default_field = "vorticity_mag"
    energy_monotone = True
    symmetric_cmap = False
    REF = (32, 32, 32)
    K0 = 3.0

    def initial_state(self, solver, p):
        n = self.REF[0]
        rng = np.random.default_rng(int(p["seed"]))
        u = rng.standard_normal((3, n, n, n))
        u_hat = np.fft.rfftn(u, axes=(1, 2, 3))
        kx = np.fft.fftfreq(n, 1.0 / n)[:, None, None]
        ky = np.fft.fftfreq(n, 1.0 / n)[None, :, None]
        kz = np.arange(n // 2 + 1)[None, None, :]
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k = np.sqrt(k2)
        shape_fn = k * np.exp(-(k / self.K0) ** 2)      # E(k) ~ k^4 exp(-2 (k/k0)^2)
        shape_fn[np.maximum(np.maximum(np.abs(kx), np.abs(ky)), kz) > n / 3.0 - 1] = 0.0
        u_hat *= shape_fn
        k2s = k2.copy()
        k2s[0, 0, 0] = 1.0
        div = (kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]) / k2s
        u_hat[0] -= kx * div
        u_hat[1] -= ky * div
        u_hat[2] -= kz * div
        u_hat[:, 0, 0, 0] = 0.0
        u = np.fft.irfftn(u_hat, s=self.REF, axes=(1, 2, 3))
        u /= np.sqrt(np.mean(np.sum(u ** 2, axis=0)) / 3.0)     # unit rms per component
        u_hat = np.fft.rfftn(u, axes=(1, 2, 3))
        return _from_reference(solver, u_hat, self.REF)


CASES: dict[str, Case] = {c.key: c for c in (
    TaylorGreen3D(), ShearLayer(), Obstacle(), Turbulence2D(), Turbulence3D(),
    TaylorGreen2D(), BeltramiABC(),
)}


def get_case(key: str) -> Case:
    try:
        return CASES[key]
    except KeyError:
        raise ValueError(f"unknown case '{key}'. Available: {', '.join(CASES)}") from None
