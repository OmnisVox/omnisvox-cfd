"""3-D incompressible Navier-Stokes, triply periodic, pseudo-spectral.

    du/dt = P[ u x omega  - div(tau_sgs) + f ] + nu lap(u),   div u = 0

Rotational form of the nonlinear term, 2/3-rule dealiasing, exact projection
``P = I - k k^T / k^2`` every stage, integrating-factor RK4 for viscosity.
Optional Smagorinsky sub-grid model for LES.
"""
from __future__ import annotations

import math

import numpy as np

from .base import SpectralSolver


class NavierStokes3D(SpectralSolver):
    ndim = 3

    def __init__(self, shape, lengths, nu, be, *, sgs="none", cs=0.17, forcing_hat=None, **kw):
        super().__init__(shape, lengths, nu, be, **kw)
        if sgs not in ("none", "smagorinsky"):
            raise ValueError("sgs must be 'none' or 'smagorinsky'")
        self.sgs = sgs
        self.cs = float(cs)
        self.forcing_hat = forcing_hat
        # filter width = pi / k_cut = 1.5 dx for the 2/3-truncated grid
        dx = self.grid.spacing
        self.delta = 1.5 * float(np.prod(dx)) ** (1.0 / 3.0)
        self.ffts_per_stage = 9 + (12 if sgs == "smagorinsky" else 0)
        self.nut_max = 0.0

    # ---- helpers -----------------------------------------------------
    def _project(self, f_hat):
        g = self.grid
        kx, ky, kz = g.k
        div = (kx * f_hat[0] + ky * f_hat[1] + kz * f_hat[2]) * g.inv_k2
        f_hat[0] -= kx * div
        f_hat[1] -= ky * div
        f_hat[2] -= kz * div
        return f_hat

    def _curl_hat(self, u_hat):
        xp = self.be.xp
        kx, ky, kz = self.grid.k
        w = xp.empty_like(u_hat)
        w[0] = 1j * (ky * u_hat[2] - kz * u_hat[1])
        w[1] = 1j * (kz * u_hat[0] - kx * u_hat[2])
        w[2] = 1j * (kx * u_hat[1] - ky * u_hat[0])
        return w

    def _measure(self, u):
        xp, be = self.be.xp, self.be
        dx, dy, dz = self.grid.spacing
        rate = xp.max(xp.abs(u[0]) / dx + xp.abs(u[1]) / dy + xp.abs(u[2]) / dz)
        self._cfl_rate = be.scalar(rate)
        self.umax = math.sqrt(be.scalar(xp.max(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))) \
            if math.isfinite(self._cfl_rate) else float("nan")

    def _prime(self):
        self._measure(self.grid.ifft(self.state))

    # ---- right-hand side --------------------------------------------
    def _rhs(self, u_hat, measure=False):
        xp, g = self.be.xp, self.grid
        u = g.ifft(u_hat)
        if measure:
            self._measure(u)
        w = g.ifft(self._curl_hat(u_hat))
        c = xp.empty_like(u)
        c[0] = u[1] * w[2] - u[2] * w[1]
        c[1] = u[2] * w[0] - u[0] * w[2]
        c[2] = u[0] * w[1] - u[1] * w[0]
        del u, w
        rhs = g.fft(c)
        del c
        if self.sgs == "smagorinsky":
            rhs += self._sgs_term(u_hat, measure)
        if self.forcing_hat is not None:
            rhs += self.forcing_hat
        rhs *= g.dealias
        return self._project(rhs)

    def _sgs_term(self, u_hat, measure=False):
        """div(2 nu_t S) with nu_t = (Cs Delta)^2 |S|, |S| = sqrt(2 S_ij S_ij)."""
        xp, g = self.be.xp, self.grid
        k = g.k
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        s_hat = xp.empty((6,) + g.hshape, dtype=u_hat.dtype)
        for n, (i, j) in enumerate(pairs):
            s_hat[n] = 0.5j * (k[i] * u_hat[j] + k[j] * u_hat[i])
        s = g.ifft(s_hat)
        del s_hat
        mag = xp.sqrt(2.0 * (s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
                      + 4.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
        nut = (self.cs * self.delta) ** 2 * mag
        if measure:
            self.nut_max = self.be.scalar(xp.max(nut))
        s *= 2.0 * nut
        t_hat = g.fft(s)
        del s, mag, nut
        out = xp.empty((3,) + g.hshape, dtype=u_hat.dtype)
        out[0] = 1j * (k[0] * t_hat[0] + k[1] * t_hat[3] + k[2] * t_hat[4])
        out[1] = 1j * (k[0] * t_hat[3] + k[1] * t_hat[1] + k[2] * t_hat[5])
        out[2] = 1j * (k[0] * t_hat[4] + k[1] * t_hat[5] + k[2] * t_hat[2])
        return out

    def _apply_guard(self):
        g = self.grid
        u = g.ifft(self.state)
        speed = self.be.xp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        u *= self._limit_speed(speed)
        self.state = self._project(g.fft(u) * g.dealias)
        self.guard_hits += 1
        self._measure(g.ifft(self.state))

    # ---- diagnostics -------------------------------------------------
    def dissipation_density(self):
        u = self.state
        g = self.grid
        return (u.real ** 2 + u.imag ** 2).sum(axis=0) * (g.k2 * g.pw)

    def diagnostics(self) -> dict:
        g, xp, be = self.grid, self.be.xp, self.be
        e_density = (self.state.real ** 2 + self.state.imag ** 2).sum(axis=0) * g.pw
        energy = 0.5 * be.scalar(xp.sum(e_density))
        enstrophy = 0.5 * be.scalar(xp.sum(e_density * g.k2))
        return {
            "energy": energy,
            "enstrophy": enstrophy,
            "dissipation": 2.0 * self.nu * enstrophy,
            "umax": self.umax,
        }

    def spectrum(self):
        g = self.grid
        d = 0.5 * (self.state.real ** 2 + self.state.imag ** 2).sum(axis=0) * g.pw
        return g.shell_spectrum(d)

    def max_divergence(self) -> float:
        g, xp = self.grid, self.be.xp
        kx, ky, kz = g.k
        div = g.ifft(1j * (kx * self.state[0] + ky * self.state[1] + kz * self.state[2]))
        return self.be.scalar(xp.max(xp.abs(div)))

    # ---- fields for display / export --------------------------------
    field_names = ("vorticity_mag", "vorticity_z", "speed", "u", "v", "w")

    def field(self, name="vorticity_mag", plane="z", index=None) -> np.ndarray:
        """A 2-D slice of a physical field as a host array."""
        g, xp = self.grid, self.be.xp
        if name in ("vorticity_mag", "vorticity_z"):
            w = g.ifft(self._curl_hat(self.state))
            f = xp.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2) if name == "vorticity_mag" else w[2]
        else:
            u = g.ifft(self.state)
            if name == "speed":
                f = xp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            else:
                f = u["uvw".index(name)]
        ax = "xyz".index(plane)
        idx = self.shape[ax] // 8 if index is None else int(index) % self.shape[ax]
        sl = [slice(None)] * 3
        sl[ax] = idx
        return self.be.to_numpy(f[tuple(sl)])

    def volume(self, name="velocity") -> np.ndarray:
        g = self.grid
        if name == "velocity":
            return self.be.to_numpy(g.ifft(self.state))
        if name == "vorticity":
            return self.be.to_numpy(g.ifft(self._curl_hat(self.state)))
        raise KeyError(name)
