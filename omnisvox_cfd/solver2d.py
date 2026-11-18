"""2-D incompressible Navier-Stokes in vorticity-streamfunction form.

    d(omega)/dt + div(u omega) = nu lap(omega) + f,   lap(psi) = -omega,
    u = U0 + d(psi)/dy,  v = V0 - d(psi)/dx

Pseudo-spectral with 2/3 dealiasing and integrating-factor RK4.  Solid bodies
are imposed by Brinkman volume penalisation, integrated exactly after every
step (u -> u exp(-chi dt / eta)), and an optional fringe zone relaxes the flow back to the free
stream so a wake does not re-enter through the periodic boundary.
"""
from __future__ import annotations

import math

import numpy as np

from .base import SpectralSolver


class NavierStokes2D(SpectralSolver):
    ndim = 2
    ffts_per_stage = 5

    def __init__(self, shape, lengths, nu, be, *, mean_flow=(0.0, 0.0), body_mask=None,
                 fringe_rate=None, indicator_window=None, forcing_hat=None,
                 ref_length=1.0, penal_eta=None, **kw):
        super().__init__(shape, lengths, nu, be, **kw)
        self.U0, self.V0 = float(mean_flow[0]), float(mean_flow[1])
        rd = be.rdtype
        self.body_mask = None if body_mask is None else be.asarray(body_mask, dtype=rd)
        self.fringe_rate = None if fringe_rate is None else be.asarray(fringe_rate, dtype=rd)
        self.indicator_window = (None if indicator_window is None
                                 else be.asarray(indicator_window, dtype=rd))
        self.forcing_hat = forcing_hat
        self.ref_length = float(ref_length)
        # Brinkman penalisation time: velocity inside a body decays as exp(-t/eta).
        # Tied to the grid (not to dt) so the result does not depend on the step size.
        if penal_eta is None:
            penal_eta = 0.25 * min(self.grid.spacing) / max(abs(self.U0) + abs(self.V0), 1e-12)
        self.penal_eta = float(penal_eta)
        self.force = (0.0, 0.0)            # force on the body per unit density
        self._has_post = self.body_mask is not None or self.fringe_rate is not None

    # ---- helpers -----------------------------------------------------
    def _velocity_hat(self, w_hat):
        kx, ky = self.grid.k
        psi = w_hat * self.grid.inv_k2
        return 1j * ky * psi, -1j * kx * psi

    def _physical(self, w_hat):
        xp, g = self.be.xp, self.grid
        stack = xp.empty((3,) + g.hshape, dtype=w_hat.dtype)
        stack[0], stack[1] = self._velocity_hat(w_hat)
        stack[2] = w_hat
        f = g.ifft(stack)
        f[0] += self.U0
        f[1] += self.V0
        return f

    def _measure(self, f):
        xp, be = self.be.xp, self.be
        dx, dy = self.grid.spacing
        self._cfl_rate = be.scalar(xp.max(xp.abs(f[0]) / dx + xp.abs(f[1]) / dy))
        self.umax = math.sqrt(be.scalar(xp.max(f[0] ** 2 + f[1] ** 2))) \
            if math.isfinite(self._cfl_rate) else float("nan")

    def _prime(self):
        self._measure(self._physical(self.state))

    # ---- right-hand side --------------------------------------------
    def _rhs(self, w_hat, measure=False):
        g = self.grid
        f = self._physical(w_hat)
        if measure:
            self._measure(f)
        f[0] *= f[2]
        f[1] *= f[2]
        flux = g.fft(f[:2])
        kx, ky = g.k
        rhs = -1j * (kx * flux[0] + ky * flux[1])
        if self.forcing_hat is not None:
            rhs += self.forcing_hat
        rhs *= g.dealias
        return rhs

    def _from_velocity(self, u, v):
        """Vorticity spectrum of a (not necessarily solenoidal) velocity field."""
        g, xp = self.grid, self.be.xp
        uv = xp.stack((u - self.U0, v - self.V0))
        uv_hat = g.fft(uv)
        kx, ky = g.k
        return (1j * (kx * uv_hat[1] - ky * uv_hat[0])) * g.dealias

    def _post_step(self, dt):
        if not self._has_post:
            return
        xp = self.be.xp
        f = self._physical(self.state)
        u, v = f[0], f[1]
        if self.fringe_rate is not None:
            lam = 1.0 - xp.exp(-self.fringe_rate * dt)
            u = u + lam * (self.U0 - u)
            v = v + lam * (self.V0 - v)
        if self.body_mask is not None:
            keep = xp.exp(self.body_mask * (-dt / self.penal_eta))
            lost = 1.0 - keep
            da = self.grid.spacing[0] * self.grid.spacing[1]
            fx = self.be.scalar(xp.sum(lost * u)) * da / dt
            fy = self.be.scalar(xp.sum(lost * v)) * da / dt
            self.force = (fx, fy)
            u = u * keep
            v = v * keep
        self.state = self._from_velocity(u, v)

    def _apply_guard(self):
        f = self._physical(self.state)
        speed = self.be.xp.sqrt(f[0] ** 2 + f[1] ** 2)
        fac = self._limit_speed(speed)
        self.state = self._from_velocity(f[0] * fac, f[1] * fac)
        self.guard_hits += 1
        self._prime()

    # ---- diagnostics -------------------------------------------------
    def dissipation_density(self):
        g = self.grid
        w_hat = self.state
        if self.indicator_window is not None:
            w_hat = g.fft(g.ifft(w_hat) * self.indicator_window)
        return (w_hat.real ** 2 + w_hat.imag ** 2) * g.pw

    def diagnostics(self) -> dict:
        g, xp, be = self.grid, self.be.xp, self.be
        z_density = (self.state.real ** 2 + self.state.imag ** 2) * g.pw
        enstrophy = 0.5 * be.scalar(xp.sum(z_density))
        energy = 0.5 * be.scalar(xp.sum(z_density * g.inv_k2))
        palinstrophy = 0.5 * be.scalar(xp.sum(z_density * g.k2))
        out = {
            "energy": energy,                 # of the fluctuation about the mean flow
            "enstrophy": enstrophy,
            "palinstrophy": palinstrophy,
            "dissipation": 2.0 * self.nu * enstrophy,
            "umax": self.umax,
        }
        if self.body_mask is not None:
            q = 0.5 * (self.U0 ** 2 + self.V0 ** 2) * self.ref_length
            if q > 0:
                out["cd"] = self.force[0] / q
                out["cl"] = self.force[1] / q
        return out

    def spectrum(self):
        g = self.grid
        d = 0.5 * (self.state.real ** 2 + self.state.imag ** 2) * g.pw * g.inv_k2
        return g.shell_spectrum(d)

    # ---- fields for display / export --------------------------------
    field_names = ("vorticity", "speed", "u", "v", "streamfunction")

    def field(self, name="vorticity", **_) -> np.ndarray:
        g, xp = self.grid, self.be.xp
        if name == "vorticity":
            f = g.ifft(self.state)
        elif name == "streamfunction":
            f = g.ifft(self.state * g.inv_k2)
        else:
            p = self._physical(self.state)
            f = {"u": p[0], "v": p[1]}.get(name)
            if f is None:
                if name != "speed":
                    raise KeyError(name)
                f = xp.sqrt(p[0] ** 2 + p[1] ** 2)
        return self.be.to_numpy(f)

    def probe(self, ix, iy):
        """(u, v) at one grid point - cheap enough only for occasional use."""
        p = self._physical(self.state)
        return self.be.scalar(p[0][ix, iy]), self.be.scalar(p[1][ix, iy])
