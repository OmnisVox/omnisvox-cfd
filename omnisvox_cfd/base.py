"""Shared machinery for the pseudo-spectral solvers.

Time integration is integrating-factor RK4: the viscous term is integrated
exactly through ``E = exp(-nu k^2 dt)``, so viscosity never limits the time
step; only the advective CFL condition does.
"""
from __future__ import annotations

import math
from collections import OrderedDict

from .backend import Backend
from .spectral import SpectralGrid


class SolverFault(RuntimeError):
    """The state became unusable (non-finite values, time-step collapse)."""


class SpectralSolver:
    ndim = 0
    #: number of forward+inverse FFTs per RK stage (for cost reporting)
    ffts_per_stage = 0

    def __init__(self, shape, lengths, nu, be: Backend, *, cfl=0.5, dt_max=None,
                 dt_fixed=None, guard_cap=None):
        self.be = be
        self.grid = SpectralGrid(shape, lengths, be)
        self.shape = self.grid.shape
        self.nu = float(nu)
        self.cfl = float(cfl)
        self.dt_max = dt_max
        self.dt_fixed = dt_fixed
        self.guard_cap = guard_cap          # None disables the last-resort limiter
        self.guard_hits = 0
        self.t = 0.0
        self.step_count = 0
        self.dt = None
        self.umax = 0.0                     # max |u| seen at the last stage-1 evaluation
        self._cfl_rate = 0.0                # max sum_i |u_i|/dx_i
        self._if_cache: OrderedDict = OrderedDict()
        self.state = None                   # spectral state array

    # ---- state -------------------------------------------------------
    def set_state(self, state_hat, t=0.0):
        self.state = state_hat
        self.t = float(t)
        self.dt = None
        self._prime()

    def get_state(self):
        return self.state.copy()

    def _prime(self):
        """Measure velocity scales for the first CFL estimate."""
        raise NotImplementedError

    def snapshot(self) -> dict:
        """Everything needed to resume bit-for-bit from this instant."""
        return {"state": self.state.copy(), "t": self.t, "dt": self.dt,
                "cfl_rate": self._cfl_rate, "umax": self.umax}

    def restore(self, snap: dict):
        self.state = snap["state"].copy()
        self.t = snap["t"]
        self.dt = snap["dt"]
        self._cfl_rate = snap["cfl_rate"]
        self.umax = snap["umax"]

    # ---- time step ---------------------------------------------------
    def _integrating_factors(self, dt):
        key = float(dt)
        hit = self._if_cache.get(key)
        if hit is not None:
            self._if_cache.move_to_end(key)
            return hit
        xp = self.be.xp
        E = xp.exp((-0.5 * self.nu * dt) * self.grid.k2).astype(self.be.rdtype)
        E2 = E * E
        self._if_cache[key] = (E, E2)
        while len(self._if_cache) > 3:
            self._if_cache.popitem(last=False)
        return E, E2

    def _pick_dt(self, t_limit=None):
        if not math.isfinite(self._cfl_rate):
            raise SolverFault("non-finite velocity field")
        if self.dt_fixed is not None:
            dt = float(self.dt_fixed)
        else:
            target = self.cfl / max(self._cfl_rate, 1e-30)
            if self.dt_max is not None:
                target = min(target, float(self.dt_max))
            # keep dt while the actual CFL number stays in a band around the
            # target: this avoids rebuilding the integrating factor every step
            if self.dt is None or not (0.6 * target <= self.dt <= 1.25 * target):
                self.dt = target
            dt = self.dt
        if t_limit is not None and self.t + dt > t_limit - 1e-12 * max(1.0, abs(t_limit)):
            dt = t_limit - self.t
            if dt <= 0.0:
                return 0.0
        if dt < 1e-12:
            raise SolverFault("time step collapsed")
        return dt

    def step(self, t_limit=None):
        """Advance one RK4 step (clipped so as not to pass ``t_limit``)."""
        if self.guard_cap is not None and self.umax > 0.75 * self.guard_cap:
            self._apply_guard()
        dt = self._pick_dt(t_limit)
        if dt == 0.0:
            return 0.0
        E, E2 = self._integrating_factors(dt)
        u0 = self.state
        k1 = self._rhs(u0, measure=True)
        if not math.isfinite(self._cfl_rate):
            raise SolverFault("non-finite velocity field")
        acc = E2 * k1
        k = self._rhs(E * (u0 + (0.5 * dt) * k1))
        del k1
        acc += (2.0 * E) * k
        Eu0 = E * u0
        k = self._rhs(Eu0 + (0.5 * dt) * k)
        acc += (2.0 * E) * k
        E2u0 = E * Eu0
        del Eu0
        k = self._rhs(E2u0 + (dt * E) * k)
        acc += k
        self.state = E2u0 + (dt / 6.0) * acc
        self.t += dt
        self.step_count += 1
        self._post_step(dt)
        return dt

    def advance_to(self, t_target, max_steps=None):
        n = 0
        while self.t < t_target - 1e-12 * max(1.0, abs(t_target)):
            if self.step(t_limit=t_target) == 0.0:
                break
            n += 1
            if max_steps is not None and n >= max_steps:
                break
        return n

    # ---- hooks -------------------------------------------------------
    def _rhs(self, state_hat, measure=False):
        raise NotImplementedError

    def _post_step(self, dt):
        pass

    def _apply_guard(self):
        raise NotImplementedError

    def _limit_speed(self, speed):
        """Last-resort limiter factor for a speed field.

        Identity below the knee (0.75 cap), then a C^1 tanh roll-off that
        saturates at ``cap``.  Because it is exactly the identity in normal
        operation it does not alter the momentum balance unless it fires, and
        every firing is counted in ``guard_hits``.
        """
        xp = self.be.xp
        cap = self.guard_cap
        knee = 0.75 * cap
        over = xp.maximum(speed - knee, 0.0)
        limited = knee + (cap - knee) * xp.tanh(over / (cap - knee))
        return xp.where(speed > knee, limited / xp.maximum(speed, 1e-30), 1.0)

    # ---- resolution indicator ---------------------------------------
    def dissipation_density(self):
        """Per-mode density of the cascading invariant's dissipation."""
        raise NotImplementedError

    def tail_fractions(self, thresholds):
        """Share of the dissipation carried by modes with rbox > threshold.

        ``rbox`` is the L-inf mode number normalised by the dealiasing cut-off,
        so ``threshold = theta`` probes the band just under this grid's cut-off
        and ``threshold = theta * (Nc / N)`` probes what a coarser grid of size
        ``Nc`` would be unable to carry.
        """
        xp = self.be.xp
        d = self.dissipation_density()
        total = self.be.scalar(xp.sum(d))
        if not math.isfinite(total):
            raise SolverFault("non-finite spectrum")
        if total <= 0.0:
            return [0.0 for _ in thresholds]
        out = []
        for th in thresholds:
            out.append(self.be.scalar(xp.sum(d * (self.grid.rbox > th))) / total)
        return out
