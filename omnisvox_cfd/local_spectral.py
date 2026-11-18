"""Spectral superposition patch for local 3-D ARR (default local scheme since 1.1.1).

Idea
----
u = T + delta inside the patch.

* T is the coarse global solution, interpolated *spectrally* (exact for the
  band-limited coarse field) onto the aligned fine patch points, together with
  its gradient and Laplacian.  T is divergence-free to machine precision.
* delta is the local fine-scale detail.  It is kept compactly supported by
  tapering its *vector potential* (delta = curl(w A)), so it is exactly
  divergence-free, vanishes at the patch faces and is therefore periodic-
  friendly: all patch derivatives and the patch projection are plain FFTs.
* Scale ownership: the coarse solver owns the coarse band, the patch owns the
  rest.  At each merge the coarse-band part of delta is handed to the global
  field (exact point samples of a band-limited field, no block averaging) and
  removed from delta, so nothing is double counted.

Interface matches LocalPatch3D so ARRController can use it unchanged.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from .local_arr import PatchRegion


def _blend_weight(shape, width):
    """0 on patch faces, 1 inside; C2 smootherstep across ``width`` fine cells."""
    ds = [np.minimum(np.arange(n), n - 1 - np.arange(n)) for n in shape]
    x = np.clip(np.minimum.reduce(np.meshgrid(*ds, indexing="ij")) / float(max(1, width)), 0.0, 1.0)
    return x ** 3 * (x * (6.0 * x - 15.0) + 10.0)


def _kvec(n, length):
    return 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)


class SpectralPatch3D:
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
        self.global_lengths = tuple(float(x) for x in lengths)
        Nc = region.global_shape
        self.Nc = tuple(int(n) for n in Nc)
        self.Mf = tuple(self.refine * n for n in self.Nc)
        self.global_spacing = tuple(Lg / n for Lg, n in zip(self.global_lengths, self.Nc))
        self.spacing = tuple(dx / self.refine for dx in self.global_spacing)
        self.fine_shape = tuple(self.refine * n for n in region.size)
        self.patch_lengths = tuple(n * h for n, h in zip(self.fine_shape, self.spacing))
        self.blend_width = max(2, int(blend_coarse) * self.refine)
        self.weight = _blend_weight(self.fine_shape, self.blend_width)
        # fine global indices covered by the patch (aligned: fine point 2i == coarse point i)
        self._fix = [(self.refine * s + np.arange(m)) % M
                     for s, m, M in zip(region.start, self.fine_shape, self.Mf)]
        # global wavenumbers (coarse) and patch wavenumbers
        self._kc = [_kvec(n, Lg) for n, Lg in zip(self.Nc, self.global_lengths)]
        kp = [_kvec(n, Lp) for n, Lp in zip(self.fine_shape, self.patch_lengths)]
        self.KX, self.KY, self.KZ = np.meshgrid(*kp, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2
        self._K2safe = np.where(self.K2 == 0, 1.0, self.K2)
        # patch 2/3 dealias mask
        dmask = [np.abs(np.fft.fftfreq(n, 1.0 / n)) < n / 3.0 for n in self.fine_shape]
        self.dealias = dmask[0][:, None, None] & dmask[1][None, :, None] & dmask[2][None, None, :]
        # coarse band in patch wavenumbers: the global solver keeps |n| < N/3 per axis
        kcut = [(n / 3.0) * 2.0 * np.pi / Lg for n, Lg in zip(self.Nc, self.global_lengths)]
        self.low = ((np.abs(self.KX) < kcut[0]) & (np.abs(self.KY) < kcut[1])
                    & (np.abs(self.KZ) < kcut[2]))
        self.delta = np.zeros((3,) + self.fine_shape)
        self.last_target = None
        self.steps = 0
        self.wall = 0.0
        self.max_divergence = 0.0
        self.last_pressure = 0.0
        self._T_cache = {}
        self.u = self._target(coarse_u)[0].copy()   # for API parity / diagnostics

    # ------------------------------------------------------------------ API
    @property
    def cell_count(self) -> int:
        return int(np.prod(self.fine_shape))

    @property
    def age(self) -> float:
        return 0.0 if not hasattr(self, "born_t") else self.t - self.born_t

    def mark_born(self):
        self.born_t = self.t

    # ------------------------------------------------------- spectral tools
    def _upsample_on_patch(self, F_hat_coarse):
        """Exact band-limited evaluation of a coarse spectral field on the patch points.

        Separable partial inverse DFT: only the patch rows/cols/planes are
        evaluated, so cost is O(Nc^3 * patch_size_along_axis) rather than a full
        fine-grid transform.
        """
        out = F_hat_coarse
        for ax, (n, Lg, fix, M) in enumerate(zip(self.Nc, self.global_lengths, self._fix, self.Mf)):
            x = fix * (Lg / M)
            E = np.exp(1j * np.outer(x, self._kc[ax])) / n      # (patch_len, n)
            out = np.moveaxis(np.tensordot(E, np.moveaxis(out, ax, 0), axes=(1, 0)), 0, ax)
        return np.real(out)

    def _target(self, coarse_u):
        """T, grad T (3x3), lap T on the patch from the coarse velocity."""
        key = id(coarse_u)
        Uh = np.fft.fftn(np.asarray(coarse_u, dtype=float), axes=(1, 2, 3))
        k = np.meshgrid(*self._kc, indexing="ij")
        k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
        T = np.stack([self._upsample_on_patch(Uh[c]) for c in range(3)])
        G = np.empty((3, 3) + self.fine_shape)
        for c in range(3):
            for d in range(3):
                G[c, d] = self._upsample_on_patch(1j * k[d] * Uh[c])
        Lp = np.stack([self._upsample_on_patch(-k2 * Uh[c]) for c in range(3)])
        return T, G, Lp

    def _fft(self, a):
        return np.fft.fftn(a, axes=(-3, -2, -1))

    def _ifft(self, a):
        return np.real(np.fft.ifftn(a, axes=(-3, -2, -1)))

    def _curl_hat(self, ah):
        return 1j * np.stack([self.KY * ah[2] - self.KZ * ah[1],
                              self.KZ * ah[0] - self.KX * ah[2],
                              self.KX * ah[1] - self.KY * ah[0]])

    def _confine(self, d):
        """Exactly divergence-free, compactly supported version of d: curl(w * curl^-1 d)."""
        dh = self._fft(d)
        Ah = self._curl_hat(dh) / self._K2safe
        Ah[:, 0, 0, 0] = 0.0
        A = self._ifft(Ah)
        out = self._ifft(self._curl_hat(self._fft(self.weight[None] * A)))
        return out

    # -------------------------------------------------------------- physics
    def _rhs_delta(self, delta, T, G, Lp, dTdt):
        dh = self._fft(delta)
        gd = np.empty((3, 3) + self.fine_shape)
        K = (self.KX, self.KY, self.KZ)
        for c in range(3):
            for d in range(3):
                gd[c, d] = self._ifft(1j * K[d] * dh[c])
        lap_d = self._ifft(-self.K2 * dh)
        u = T + delta
        grad = G + gd
        adv = np.einsum("dxyz,cdxyz->cxyz", u, grad)
        rhs = -adv + self.nu * (Lp + lap_d) - dTdt
        # The tendency contains T.grad(T), which is not periodic on the patch.
        # Taper it first so the FFT filter sees a compact field (no Gibbs ringing).
        rh = self._fft(rhs * self.weight[None]) * self.dealias
        # Leray-project every stage tendency (pressure), as the global solver does
        kd = (self.KX * rh[0] + self.KY * rh[1] + self.KZ * rh[2]) / self._K2safe
        rh = rh - np.stack([self.KX * kd, self.KY * kd, self.KZ * kd])
        return self._ifft(rh)

    def _pick_dt(self, T, remaining):
        u = T + self.delta
        dx, dy, dz = self.spacing
        rate = float(np.max(np.abs(u[0]) / dx + np.abs(u[1]) / dy + np.abs(u[2]) / dz))
        dt_adv = self.cfl / max(rate, 1e-30)
        dt_diff = 0.3 / (self.nu * (1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2)) if self.nu > 0 else math.inf
        return min(float(remaining), dt_adv, dt_diff)

    def advance_interval(self, t0, t1, coarse_u0, coarse_u1) -> int:
        if t1 <= t0:
            return 0
        tic = time.perf_counter()
        T0, G0, L0 = self._target(coarse_u0)
        T1, G1, L1 = self._target(coarse_u1)
        dTdt = (T1 - T0) / (t1 - t0)
        lerp = lambda a, b, s: (1.0 - s) * a + s * b
        t, n = float(t0), 0
        while t < t1 - 1e-13 * max(1.0, abs(t1)):
            s0 = (t - t0) / (t1 - t0)
            Ta = lerp(T0, T1, s0)
            dt = self._pick_dt(Ta, t1 - t)
            if not math.isfinite(dt) or dt <= 1e-14:
                raise RuntimeError("spectral patch time step collapsed")
            s1 = min(1.0, (t + dt - t0) / (t1 - t0)); sm = 0.5 * (s0 + s1)
            # classical RK4 on delta, coarse target linear in time
            args = lambda s: (lerp(T0, T1, s), lerp(G0, G1, s), lerp(L0, L1, s), dTdt)
            k1 = self._rhs_delta(self.delta, *args(s0))
            k2 = self._rhs_delta(self.delta + 0.5 * dt * k1, *args(sm))
            k3 = self._rhs_delta(self.delta + 0.5 * dt * k2, *args(sm))
            k4 = self._rhs_delta(self.delta + dt * k3, *args(s1))
            d = self.delta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.delta = self._confine(d)
            t += dt
            n += 1
        dmax = float(np.max(np.abs(self._ifft(1j * (self.KX * self._fft(self.delta[0])
                                                     + self.KY * self._fft(self.delta[1])
                                                     + self.KZ * self._fft(self.delta[2]))))))
        self.max_divergence = max(self.max_divergence, dmax)
        self.t = float(t1)
        self.steps += n
        self.wall += time.perf_counter() - tic
        self.last_target = T1
        self.u = T1 + self.delta
        return n

    def _split_low(self):
        dh = self._fft(self.delta)
        low = self._ifft(dh * self.low[None])
        return low

    def correction_coarse(self, coarse_u):
        low = self._split_low()
        r = self.refine
        cc = low[:, ::r, ::r, ::r]          # exact samples of a band-limited field at coarse points
        return cc, self.region.indices()

    def merge_into(self, solver) -> dict:
        """Hand the coarse band of delta to the global field, conserving u = T + delta.

        Whatever the global field actually absorbs (after its own dealiasing and
        projection) is interpolated back exactly and removed from delta, so a
        merge only moves ownership between the two levels inside the patch.
        """
        u = solver.volume("velocity")
        before_e = float(np.mean(np.sum(u ** 2, axis=0)))
        low = self._split_low() * self.weight[None]
        r = self.refine
        corr = low[:, ::r, ::r, ::r]
        idx = self.region.indices()
        u_new = u.copy()
        u_new[np.ix_(np.arange(3), *idx)] += corr
        g = solver.grid
        state = g.fft(solver.be.asarray(u_new, dtype=solver.be.rdtype)) * g.dealias
        solver.state = solver._project(state)
        solver._prime()
        after = solver.volume("velocity")
        absorbed = np.fft.fftn(after - u, axes=(1, 2, 3))
        self.delta = self._confine(self.delta - np.stack(
            [self._upsample_on_patch(absorbed[c]) for c in range(3)]))
        div = solver.max_divergence()
        after_e = float(np.mean(np.sum(after ** 2, axis=0)))
        return {"divergence": div, "energy_rel_change": (after_e - before_e) / max(before_e, 1e-30),
                "correction_rms": float(np.sqrt(np.mean(corr ** 2)))}

    def tail_fraction(self, theta: float) -> float:
        """Share of patch enstrophy (coarse+detail) sitting in the patch's top band.

        This is the patch analogue of the global tail fraction: it rises only
        when the patch itself is running out of resolution.
        """
        if self.last_target is None:
            return 0.0
        dh = self._fft(self.delta)
        wd = self._curl_hat(dh)
        n = [np.fft.fftfreq(m, 1.0 / m) for m in self.fine_shape]
        rbox = np.maximum(np.maximum(np.abs(n[0])[:, None, None] / (self.fine_shape[0] / 3.0),
                                     np.abs(n[1])[None, :, None] / (self.fine_shape[1] / 3.0)),
                          np.abs(n[2])[None, None, :] / (self.fine_shape[2] / 3.0))
        top = float(np.sum(np.abs(wd[:, rbox > float(theta)]) ** 2)) / np.prod(self.fine_shape) ** 2
        Th = self._fft(self.last_target + self.delta)
        tot = float(np.sum(np.abs(self._curl_hat(Th)) ** 2)) / np.prod(self.fine_shape) ** 2
        p = top / tot if tot > 1e-30 and math.isfinite(tot) else 0.0
        self.last_pressure = p
        return p

    def snapshot(self) -> dict:
        return {"scheme": "spectral", "region": self.region.to_dict(), "refine": self.refine,
                "nu": self.nu, "cfl": self.cfl, "t": self.t, "delta": self.delta.copy(),
                "last_target": None if self.last_target is None else self.last_target.copy(),
                "steps": self.steps, "wall": self.wall, "max_divergence": self.max_divergence,
                "last_pressure": self.last_pressure, "blend_width": self.blend_width,
                "global_lengths": list(self.global_lengths),
                "born_t": getattr(self, "born_t", self.t)}

    @classmethod
    def from_snapshot(cls, snap: dict, coarse_u: np.ndarray):
        rd = snap["region"]
        region = PatchRegion(tuple(rd["start"]), tuple(rd["size"]), tuple(rd["global_shape"]))
        refine = int(snap["refine"])
        obj = cls(region, refine, snap["global_lengths"], snap["nu"], snap["cfl"], coarse_u,
                  snap["t"], blend_coarse=max(1, int(round(snap["blend_width"] / refine))))
        obj.delta = np.asarray(snap["delta"]).copy()
        lt = snap.get("last_target")
        obj.last_target = None if lt is None else np.asarray(lt).copy()
        obj.steps = int(snap["steps"]); obj.wall = float(snap["wall"])
        obj.max_divergence = float(snap["max_divergence"])
        obj.last_pressure = float(snap["last_pressure"])
        obj.born_t = float(snap.get("born_t", snap["t"]))
        return obj
