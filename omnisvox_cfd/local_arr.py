"""Local 3-D refinement support for ARR.

The global pseudo-spectral solver remains the authoritative coarse solution and
incompressibility projector.  A :class:`LocalPatch3D` is a temporary, padded
fine-grid finite-difference patch that follows a spatially concentrated burst of
high-wavenumber activity.  The patch is boundary-driven by the coarse solution,
subcycles at the fine-grid CFL limit, and feeds only its tapered correction back
into the coarse field.  Every merge is followed by the global spectral
projection, so ``div u = 0`` is restored on the periodic domain.

This module deliberately keeps the local solver independent of the global FFT
basis.  A cropped Fourier basis would make opposite patch faces periodic, which
is generally the wrong boundary condition for an interior AMR patch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np
from scipy import fft as spfft
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class PatchRegion:
    """Periodic coarse-grid box occupied by one local patch."""

    start: tuple[int, int, int]
    size: tuple[int, int, int]
    global_shape: tuple[int, int, int]

    @property
    def coarse_cells(self) -> int:
        return int(np.prod(self.size))

    @property
    def volume_fraction(self) -> float:
        return self.coarse_cells / float(np.prod(self.global_shape))

    def indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple((s + np.arange(n, dtype=int)) % N
                     for s, n, N in zip(self.start, self.size, self.global_shape))

    def to_dict(self) -> dict:
        return {"start": list(self.start), "size": list(self.size),
                "global_shape": list(self.global_shape),
                "volume_fraction": self.volume_fraction}


# ---------------------------------------------------------------------------
# Spatial localisation
# ---------------------------------------------------------------------------
def _minimum_periodic_interval(indices: np.ndarray, n: int, padding: int) -> tuple[int, int]:
    """Smallest circular interval containing ``indices``, expanded by padding."""
    idx = np.unique(np.asarray(indices, dtype=int) % n)
    if idx.size == 0:
        return 0, n
    if idx.size == n:
        return 0, n
    s = np.sort(idx)
    # Gaps are numbers of *unselected* cells between consecutive selected cells.
    nxt = np.r_[s[1:], s[0] + n]
    gaps = nxt - s - 1
    j = int(np.argmax(gaps))
    # Complement of the largest empty gap is the minimum occupied arc.
    start = int(nxt[j] % n)
    size = int(n - gaps[j])
    start = (start - padding) % n
    size = min(n, size + 2 * padding)
    return start, size


def _make_even_interval(start: int, size: int, n: int, minimum: int = 4) -> tuple[int, int]:
    """Patch coarse sizes are kept even so 2x refinement stays FFT/export friendly."""
    size = max(minimum, min(n, int(size)))
    if size < n and size % 2:
        size += 1
    size = min(size, n)
    return int(start) % n, size


def local_high_k_activity(solver, theta: float) -> np.ndarray:
    """Return a host-space map of high-k enstrophy activity.

    We high-pass the *vorticity* rather than velocity.  The resulting physical
    density localises where the enstrophy cascade is pressing against the
    current spectral cut-off.  It is a localisation diagnostic; the controller's
    existing global ``tail_fractions`` remains the promotion criterion.
    """
    g = solver.grid
    mask = g.rbox > float(theta)
    w_hat = solver._curl_hat(solver.state) * mask
    w = g.ifft(w_hat)
    a = (w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    return solver.be.to_numpy(a)


def _min_arc_with_mass(profile: np.ndarray, frac: float) -> tuple[int, int]:
    """Shortest circular window holding at least ``frac`` of a 1-D profile's mass."""
    n = profile.size
    tot = float(profile.sum())
    if tot <= 0.0:
        return 0, n
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([profile, profile]))])
    for length in range(1, n + 1):
        sums = cs[length:length + n] - cs[:n]
        j = int(np.argmax(sums))
        if sums[j] >= frac * tot:
            return j, length
    return 0, n


def locate_hot_region(solver, theta: float, target_fraction: float = 0.80,
                      padding: int = 3, min_size: int = 4) -> tuple[PatchRegion, dict]:
    """Find a padded periodic box holding about ``target_fraction`` of local activity.

    Per-axis marginal localisation: on each axis, the shortest circular window
    holding ``target_fraction**(1/3)`` of that axis' activity profile.  (1.1.0
    took the bounding box of the top cells, which a thin scattered background
    inflated to the whole domain.)
    """
    a = np.asarray(local_high_k_activity(solver, theta), dtype=float)
    total = float(np.sum(a))
    shape = tuple(int(n) for n in solver.shape)
    if not math.isfinite(total) or total <= 0.0:
        reg = PatchRegion((0, 0, 0), shape, shape)
        return reg, {"activity_total": total, "selected_fraction": 1.0,
                     "region_volume_fraction": 1.0}
    per_axis = float(target_fraction) ** (1.0 / 3.0)
    starts, sizes = [], []
    for ax, n in enumerate(shape):
        prof = a.sum(axis=tuple(i for i in range(3) if i != ax))
        st, ln = _min_arc_with_mass(prof, per_axis)
        st, ln = (st - int(padding)) % n, min(n, ln + 2 * int(padding))
        st, ln = _make_even_interval(st, ln, n, min_size)
        starts.append(st); sizes.append(ln)
    reg = PatchRegion(tuple(starts), tuple(sizes), shape)
    captured = float(a[np.ix_(*reg.indices())].sum()) / total
    return reg, {"activity_total": total, "selected_fraction": captured,
                 "region_volume_fraction": reg.volume_fraction}


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def _fine_coords(region: PatchRegion, refine: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = []
    for st, nc, N in zip(region.start, region.size, region.global_shape):
        coords.append((st + np.arange(nc * refine, dtype=float) / refine) % N)
    return tuple(coords)


def sample_global_to_patch(global_u: np.ndarray, region: PatchRegion, refine: int) -> np.ndarray:
    """Periodic cubic interpolation of global velocity onto the aligned fine patch."""
    global_u = np.asarray(global_u)
    xs = _fine_coords(region, refine)
    mesh = np.meshgrid(*xs, indexing="ij", sparse=False)
    coord = np.asarray(mesh)
    out = np.empty((3,) + tuple(n * refine for n in region.size), dtype=global_u.dtype)
    for c in range(3):
        out[c] = map_coordinates(global_u[c], coord, order=3, mode="wrap", prefilter=True)
    return out


def _smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _blend_weight(shape: Sequence[int], width: int) -> np.ndarray:
    """0 on patch faces, 1 in the interior, smooth across ``width`` fine cells."""
    width = max(1, int(width))
    ds = []
    for n in shape:
        i = np.arange(n)
        ds.append(np.minimum(i, n - 1 - i))
    d = np.minimum.reduce(np.meshgrid(*ds, indexing="ij"))
    return _smoothstep01(d / float(width))


# ---------------------------------------------------------------------------
# Local projection patch
# ---------------------------------------------------------------------------
class LocalPatch3D:
    """Legacy (1.1.0) 2x/4x finite-difference refinement patch (``local_scheme="fd"``).

    Known limitation: its collocated 2nd-order projection enforces a discrete
    constraint that the spectral solution does not satisfy, so it distorts
    resolved fine scales.  Kept for comparison only; the default is
    :class:`omnisvox_cfd.local_spectral.SpectralPatch3D`.

    The patch uses a standard velocity-projection method with second-order
    centred differences.  Its outer buffer is continuously driven toward the
    coarse solution, so the correction is zero at patch faces.  This makes the
    correction safe to restrict and blend back into the global field.
    """

    def __init__(self, region: PatchRegion, refine: int, lengths: Sequence[float],
                 nu: float, cfl: float, coarse_u: np.ndarray, t: float, *,
                 blend_coarse: int = 3):
        self.region = region
        self.refine = int(refine)
        if self.refine not in (2, 4):
            raise ValueError("local refinement factor must be 2 or 4")
        self.nu = float(nu)
        self.cfl = min(float(cfl), 0.45)
        self.t = float(t)
        self.fine_shape = tuple(n * self.refine for n in region.size)
        self.global_lengths = tuple(float(L) for L in lengths)
        self.global_spacing = tuple(L / N for L, N in zip(self.global_lengths,
                                                           region.global_shape))
        self.spacing = tuple(dx / self.refine for dx in self.global_spacing)
        self.patch_lengths = tuple(n * dx for n, dx in zip(region.size, self.global_spacing))
        self.blend_width = max(2, int(blend_coarse) * self.refine)
        self.weight = _blend_weight(self.fine_shape, self.blend_width)
        xs = _fine_coords(region, self.refine)
        self._coords = np.asarray(np.meshgrid(*xs, indexing="ij", sparse=False))
        self.u = self._sample(coarse_u)
        self.last_target = self.u.copy()
        self.steps = 0
        self.wall = 0.0
        self.max_divergence = 0.0
        self.last_pressure = 0.0
        self._lam = self._poisson_eigs()

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.fine_shape))

    @property
    def age(self) -> float:
        return 0.0 if not hasattr(self, "born_t") else self.t - self.born_t

    def mark_born(self):
        self.born_t = self.t

    def _sample(self, global_u):
        global_u = np.asarray(global_u)
        out = np.empty((3,) + self.fine_shape, dtype=global_u.dtype)
        for c in range(3):
            out[c] = map_coordinates(global_u[c], self._coords, order=3, mode="wrap", prefilter=True)
        return out

    def _poisson_eigs(self):
        """Eigenvalues of the central-divergence/central-gradient Poisson operator."""
        vals = []
        for n, h in zip(self.fine_shape, self.spacing):
            m = n - 2
            k = np.arange(1, m + 1, dtype=float)
            vals.append(-(np.sin(np.pi * k / (m + 1.0)) ** 2) / (h * h))
        return (vals[0][:, None, None] + vals[1][None, :, None] + vals[2][None, None, :])

    def _anchor(self, u, target):
        w = self.weight[None, ...]
        return target + w * (u - target)

    def _rhs(self, u):
        dx, dy, dz = self.spacing
        v = u[:, 1:-1, 1:-1, 1:-1]
        out = np.empty_like(v)
        for c in range(3):
            f = u[c]
            fx = (f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (2.0 * dx)
            fy = (f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]) / (2.0 * dy)
            fz = (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / (2.0 * dz)
            lap = ((f[2:, 1:-1, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1]
                    + f[:-2, 1:-1, 1:-1]) / dx ** 2
                   + (f[1:-1, 2:, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1]
                      + f[1:-1, :-2, 1:-1]) / dy ** 2
                   + (f[1:-1, 1:-1, 2:] - 2.0 * f[1:-1, 1:-1, 1:-1]
                      + f[1:-1, 1:-1, :-2]) / dz ** 2)
            out[c] = -(v[0] * fx + v[1] * fy + v[2] * fz) + self.nu * lap
        return out

    def _divergence(self, u):
        dx, dy, dz = self.spacing
        return ((u[0, 2:, 1:-1, 1:-1] - u[0, :-2, 1:-1, 1:-1]) / (2.0 * dx)
                + (u[1, 1:-1, 2:, 1:-1] - u[1, 1:-1, :-2, 1:-1]) / (2.0 * dy)
                + (u[2, 1:-1, 1:-1, 2:] - u[2, 1:-1, 1:-1, :-2]) / (2.0 * dz))

    def _project(self, u, dt, target):
        u = self._anchor(u, target)
        dx, dy, dz = self.spacing
        # Central div/grad are not an exact algebraic pair on a collocated grid.
        # A compatible central-difference projection removes the bulk residual while the
        # outer sponge keeps the supplied coarse boundary values fixed.
        for _ in range(1):
            div = self._divergence(u)
            rhs = div / max(dt, 1e-30)
            rhat = spfft.dstn(rhs, type=1, norm="ortho", workers=-1)
            phi = spfft.idstn(rhat / self._lam, type=1, norm="ortho", workers=-1)
            pad = np.pad(phi, 1, mode="constant")
            u[0, 1:-1, 1:-1, 1:-1] -= dt * (pad[2:, 1:-1, 1:-1] - pad[:-2, 1:-1, 1:-1]) / (2.0 * dx)
            u[1, 1:-1, 1:-1, 1:-1] -= dt * (pad[1:-1, 2:, 1:-1] - pad[1:-1, :-2, 1:-1]) / (2.0 * dy)
            u[2, 1:-1, 1:-1, 1:-1] -= dt * (pad[1:-1, 1:-1, 2:] - pad[1:-1, 1:-1, :-2]) / (2.0 * dz)
            u = self._anchor(u, target)
        md = float(np.max(np.abs(self._divergence(u))))
        self.max_divergence = max(self.max_divergence, md)
        return u

    def _pick_dt(self, u, remaining):
        dx, dy, dz = self.spacing
        speed_rate = np.max(np.abs(u[0]) / dx + np.abs(u[1]) / dy + np.abs(u[2]) / dz)
        dt_adv = self.cfl / max(float(speed_rate), 1e-30)
        if self.nu > 0:
            dt_diff = 0.15 / (self.nu * (1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))
        else:
            dt_diff = float("inf")
        return min(float(remaining), dt_adv, dt_diff)

    def advance_interval(self, t0: float, t1: float, coarse_u0: np.ndarray,
                         coarse_u1: np.ndarray) -> int:
        """Subcycle from t0 to t1 while coarse boundaries interpolate in time."""
        if t1 <= t0:
            return 0
        target0 = self._sample(coarse_u0)
        target1 = self._sample(coarse_u1)
        # Re-anchor to the actual coarse solution at the start of every coarse interval.
        self.u = self._anchor(self.u, target0)
        t = float(t0)
        n = 0
        tic = time.perf_counter()
        while t < t1 - 1e-13 * max(1.0, abs(t1)):
            a0 = (t - t0) / (t1 - t0)
            rem = t1 - t
            target = (1.0 - a0) * target0 + a0 * target1
            dt = self._pick_dt(self.u, rem)
            if not math.isfinite(dt) or dt <= 1e-14:
                raise RuntimeError("local patch time step collapsed")
            a1 = min(1.0, (t + dt - t0) / (t1 - t0))
            target_end = (1.0 - a1) * target0 + a1 * target1
            # SSP-RK2 with a projection after each stage.
            rhs0 = self._rhs(self.u)
            u1 = self.u.copy()
            u1[:, 1:-1, 1:-1, 1:-1] += dt * rhs0
            u1 = self._project(u1, dt, target_end)
            rhs1 = self._rhs(u1)
            u2 = self.u.copy()
            u2[:, 1:-1, 1:-1, 1:-1] += 0.5 * dt * (rhs0 + rhs1)
            self.u = self._project(u2, dt, target_end)
            t += dt
            n += 1
        self.t = float(t1)
        self.steps += n
        self.wall += time.perf_counter() - tic
        self.last_target = target1
        return n

    def correction_coarse(self, coarse_u: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Volume-average the tapered fine correction back to coarse patch cells."""
        target = self._sample(coarse_u)
        corr = (self.u - target) * self.weight[None, ...]
        r = self.refine
        nx, ny, nz = self.region.size
        # conservative block restriction (one average per parent coarse cell)
        cc = corr.reshape(3, nx, r, ny, r, nz, r).mean(axis=(2, 4, 6))
        return cc, self.region.indices()

    def merge_into(self, solver) -> dict:
        """Add local correction to global field, then use the existing global projector."""
        u = solver.volume("velocity")
        before_e = float(np.mean(np.sum(u ** 2, axis=0)))
        corr, idx = self.correction_coarse(u)
        # np.ix_ for the three spatial axes, component axis kept whole.
        ix = np.ix_(np.arange(3), idx[0], idx[1], idx[2])
        u[ix] += corr
        g = solver.grid
        state = g.fft(solver.be.asarray(u, dtype=solver.be.rdtype)) * g.dealias
        solver.state = solver._project(state)
        solver._prime()
        div = solver.max_divergence()
        after_u = solver.volume("velocity")
        after_e = float(np.mean(np.sum(after_u ** 2, axis=0)))
        return {"divergence": div, "energy_rel_change": (after_e - before_e) / max(before_e, 1e-30),
                "correction_rms": float(np.sqrt(np.mean(corr ** 2)))}

    def tail_fraction(self, theta: float) -> float:
        """Windowed high-k enstrophy share used only to detect patch saturation."""
        u = self.u - self.last_target
        # The correction already vanishes at the boundary; a second soft window
        # suppresses any remaining edge leakage before the local FFT diagnostic.
        u = u * self.weight[None, ...]
        uh = np.fft.rfftn(u, axes=(1, 2, 3))
        nx, ny, nz = self.fine_shape
        kx = np.fft.fftfreq(nx, 1.0 / nx)[:, None, None]
        ky = np.fft.fftfreq(ny, 1.0 / ny)[None, :, None]
        kz = np.arange(nz // 2 + 1)[None, None, :]
        # curl magnitude in mode-number units; scale cancels in the fraction.
        wx = 1j * (ky * uh[2] - kz * uh[1])
        wy = 1j * (kz * uh[0] - kx * uh[2])
        wz = 1j * (kx * uh[1] - ky * uh[0])
        d = np.abs(wx) ** 2 + np.abs(wy) ** 2 + np.abs(wz) ** 2
        total = float(np.sum(d))
        if not math.isfinite(total) or total <= 1e-30:
            self.last_pressure = 0.0
            return 0.0
        rbox = np.maximum(np.maximum(np.abs(kx) / (nx / 3.0), np.abs(ky) / (ny / 3.0)),
                          kz / (nz / 3.0))
        p = float(np.sum(d[rbox > float(theta)]) / total)
        self.last_pressure = p
        return p

    def snapshot(self) -> dict:
        return {"scheme": "fd", "region": self.region.to_dict(), "refine": self.refine, "nu": self.nu,
                "cfl": self.cfl, "t": self.t, "u": self.u.copy(),
                "last_target": self.last_target.copy(), "steps": self.steps,
                "wall": self.wall, "max_divergence": self.max_divergence,
                "last_pressure": self.last_pressure,
                "blend_width": self.blend_width,
                "global_lengths": list(self.global_lengths),
                "born_t": getattr(self, "born_t", self.t)}

    @classmethod
    def from_snapshot(cls, snap: dict, coarse_u: np.ndarray):
        rd = snap["region"]
        region = PatchRegion(tuple(rd["start"]), tuple(rd["size"]), tuple(rd["global_shape"]))
        # blend_width is in fine cells in the snapshot; convert approximately back to coarse.
        refine = int(snap["refine"])
        obj = cls(region, refine, snap["global_lengths"], snap["nu"], snap["cfl"], coarse_u,
                  snap["t"], blend_coarse=max(1, int(round(snap["blend_width"] / refine))))
        obj.u = np.asarray(snap["u"]).copy()
        obj.last_target = np.asarray(snap["last_target"]).copy()
        obj.steps = int(snap["steps"]); obj.wall = float(snap["wall"])
        obj.max_divergence = float(snap["max_divergence"])
        obj.last_pressure = float(snap["last_pressure"])
        obj.born_t = float(snap.get("born_t", snap["t"]))
        return obj
