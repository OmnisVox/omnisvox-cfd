"""Adaptive Rollback-Refinement (ARR).

The whole periodic domain runs on one of several resolution tiers.  A cheap
spectral observable - the *saturation pressure* - is sampled every few steps:

    P = share of the dissipation carried by modes above ``theta * k_cut``

A resolved flow has almost nothing there; when the cascade reaches the grid
cut-off the share climbs.  The controller reacts in three layers:

1. **Early warning** (``P > warn``): stop refreshing the checkpoint, so the last
   checkpoint is a state that was still clean.
2. **Rollback + promote** (``P > press_up`` or a hard fault such as non-finite
   values, a collapsing time step, or energy growth in an unforced flow):
   discard everything since that checkpoint, prolong the checkpoint spectrally
   onto the next finer tier and redo the interval there.  The run never
   commits to an under-resolved stretch.
3. **Last resort** on the finest tier only: retry from the checkpoint with a
   smaller CFL number, optionally switch on a sub-grid model, and - only if a
   velocity cap is configured - a counted tanh limiter.

When the finer grid is no longer needed (the predicted pressure on the next
coarser tier stays below ``press_dn`` for ``hold`` observations) the state is
restricted by spectral truncation.  ``press_dn < warn < press_up`` gives the
hysteresis that prevents tier chatter.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Sequence

import numpy as np

from .base import SolverFault
from .spectral import resample_hat
from .local_arr import LocalPatch3D, locate_hot_region
from .local_spectral import SpectralPatch3D

_PATCH_SCHEMES = {"spectral": SpectralPatch3D, "fd": LocalPatch3D}


def _patch_from_snapshot(snap, coarse_u):
    return _PATCH_SCHEMES[snap.get("scheme", "fd")].from_snapshot(snap, coarse_u)


class SimulationFailed(RuntimeError):
    """The run could not be continued even on the finest tier."""


@dataclass
class ARRConfig:
    enabled: bool = True
    theta: float = 0.7             # probe band starts at theta * k_cut
    press_up: float = 0.03         # rollback + promote above this (cases set their own)
    warn_ratio: float = 0.5        # warn      = warn_ratio * press_up
    down_ratio: float = 0.2        # press_dn  = down_ratio * press_up
    hold: int = 5                  # calm observations needed before demoting
    obs_every: int = 5             # steps between observations
    max_spec_obs: int = 20         # speculative observations before the state is accepted
    rollback: bool = True          # False = promote in place (for ablation studies)
    start_tier: int = 0
    min_tier: int = 0
    max_tier: int | None = None
    on_saturation: str = "continue"   # at the finest tier: continue | sgs | stop
    guard_factor: float | None = 4.0  # limiter cap = factor * initial max speed; None = off
    energy_tol: float = 1e-4       # allowed relative energy growth per observation (unforced)
    max_retries: int = 3

    # 3-D spatial ARR (experimental).  The global spectral solver remains the
    # backbone; a temporary local patch carries fine scales only where high-k
    # activity is concentrated.  Unsupported/unsafe patches fall back to the
    # original whole-domain tier promotion.  Off by default: on the benchmarks
    # so far whole-domain promotion is still faster and more accurate.
    local_3d: bool = False
    local_scheme: str = "spectral"    # "spectral" (1.1.1) | "fd" (1.1.0 legacy)
    local_refine: int = 2
    local_target_fraction: float = 0.80
    local_padding: int = 3            # coarse cells around the detected hot region
    local_blend: int = 3              # coarse-cell width of the patch sponge
    local_max_volume: float = 0.45    # fall back to global refinement above this
    local_min_age: float = 0.25       # simulated time before a patch may retire
    local_fail_factor: float = 10.0   # local-patch pressure multiple that forces fallback
    local_merge_energy_tol: float = 0.02
    local_allow_gpu: bool = False     # host patch would otherwise cause PCIe traffic each step

    def validate(self, n_tiers: int):
        if not 0.3 <= self.theta <= 0.95:
            raise ValueError("arr.theta must lie in [0.3, 0.95]")
        if not 0.0 < self.press_up < 1.0:
            raise ValueError("arr.press_up must lie in (0, 1)")
        if not 0.0 < self.down_ratio < self.warn_ratio <= 1.0:
            raise ValueError("need 0 < arr.down_ratio < arr.warn_ratio <= 1")
        if self.obs_every < 1 or self.hold < 1 or self.max_spec_obs < 1:
            raise ValueError("arr.obs_every, arr.hold and arr.max_spec_obs must be >= 1")
        if self.on_saturation not in ("continue", "sgs", "stop"):
            raise ValueError("arr.on_saturation must be continue, sgs or stop")
        if self.local_scheme not in _PATCH_SCHEMES:
            raise ValueError(f"arr.local_scheme must be one of {sorted(_PATCH_SCHEMES)}")
        if self.local_refine not in (2, 4):
            raise ValueError("arr.local_refine must be 2 or 4")
        if not 0.5 <= self.local_target_fraction <= 0.99:
            raise ValueError("arr.local_target_fraction must lie in [0.5, 0.99]")
        if self.local_padding < 1 or self.local_blend < 1:
            raise ValueError("arr.local_padding and arr.local_blend must be >= 1")
        if not 0.05 <= self.local_max_volume <= 1.0:
            raise ValueError("arr.local_max_volume must lie in [0.05, 1]")
        if self.local_min_age < 0 or self.local_fail_factor <= 1.0:
            raise ValueError("arr.local_min_age must be >= 0 and local_fail_factor > 1")
        if not 0.0 <= self.local_merge_energy_tol <= 0.5:
            raise ValueError("arr.local_merge_energy_tol must lie in [0, 0.5]")
        hi = n_tiers - 1 if self.max_tier is None else self.max_tier
        if not (0 <= self.min_tier <= self.start_tier <= hi <= n_tiers - 1):
            raise ValueError("need 0 <= min_tier <= start_tier <= max_tier < number of tiers")
        return hi


HISTORY_KEYS = ("t", "step", "tier", "dt", "energy", "enstrophy", "dissipation", "umax",
                "pressure", "pressure_coarse", "saturated", "local_active",
                "local_volume", "local_pressure")


class ARRController:
    """Drives a family of per-tier solvers through one simulation."""

    def __init__(self, make_solver: Callable[[int], object], tier_shapes: Sequence[Sequence[int]],
                 initial_state: Callable[[object], object], cfg: ARRConfig, *,
                 energy_monotone: bool = False, on_event: Callable[[dict], None] | None = None):
        self.cfg = cfg
        self.tier_shapes = [tuple(int(n) for n in s) for s in tier_shapes]
        self.n_tiers = len(self.tier_shapes)
        self.max_tier = cfg.validate(self.n_tiers)
        if not cfg.enabled:
            self.min_tier = self.max_tier_eff = cfg.start_tier
        else:
            self.min_tier, self.max_tier_eff = cfg.min_tier, self.max_tier
        self._make = make_solver
        self._solvers: dict[int, object] = {}
        self.energy_monotone = energy_monotone
        self.on_event = on_event

        self.tier = cfg.start_tier
        self.solver = self._get(self.tier)
        self.solver.set_state(initial_state(self.solver), 0.0)
        self.be = self.solver.be
        self.u_ref = max(self.solver.umax, 1e-12)
        self._sync_guard()

        self.history = {k: [] for k in HISTORY_KEYS}
        self.extra_keys: list[str] = []
        self.events: list[dict] = []
        self.cost = {i: {"steps": 0, "cell_steps": 0, "wall": 0.0} for i in range(self.n_tiers)}
        self.wasted = {"steps": 0, "cell_steps": 0, "wall": 0.0}
        self.local_cost = {"steps": 0, "cell_steps": 0, "wall": 0.0}
        self.local_wasted = {"steps": 0, "cell_steps": 0, "wall": 0.0}
        self.local_patch = None
        self._local_calm = 0
        self.uniform_equiv_cell_steps = 0.0
        self.total_steps = 0
        self.guard_hits = 0
        self.saturated = False
        self._calm = 0
        self._calm_q0 = 0.0
        self._spec = 0
        self._retries = 0
        self._chunk_log: list[tuple] = []   # (t_end, tier, steps, wall, equiv) since checkpoint
        self._last_energy = None
        self.checkpoint = None
        self._observe(initial=True)

    # ---- solver management ------------------------------------------
    def _get(self, tier):
        s = self._solvers.get(tier)
        if s is None:
            s = self._make(tier)
            self._solvers[tier] = s
        return s

    def _sync_guard(self):
        """The limiter is armed on the finest tier only."""
        for i, s in self._solvers.items():
            armed = self.cfg.guard_factor is not None and i == self.max_tier_eff
            s.guard_cap = self.cfg.guard_factor * self.u_ref if armed else None

    @property
    def t(self):
        return self.solver.t

    @property
    def shape(self):
        return self.tier_shapes[self.tier]

    def _emit(self, kind, **kw):
        ev = {"type": kind, "t": self.solver.t, "step": self.total_steps, **kw}
        self.events.append(ev)
        if self.on_event:
            self.on_event(ev)
        return ev

    def _switch(self, new_tier, state_hat, t):
        old_shape = self.tier_shapes[self.tier]
        new = self._get(new_tier)
        self._sync_guard()
        new.set_state(resample_hat(state_hat, old_shape, self.tier_shapes[new_tier], self.be)
                      * (new.grid.dealias if new_tier < self.tier else 1.0), t)
        self.solver.state = None            # release the old tier's state
        self.be.free_memory()
        self.tier, self.solver = new_tier, new

    # ---- checkpointing ----------------------------------------------
    def _save_checkpoint(self):
        s = self.solver
        self.checkpoint = {
            "tier": self.tier, "t": s.t, "snap": s.snapshot(),
            "rows": len(self.history["t"]), "energy": self._last_energy,
            "total_steps": self.total_steps,
            "local_patch": self.local_patch.snapshot() if self.local_patch is not None else None,
        }
        self._chunk_log.clear()
        self._spec = 0

    def _restore_checkpoint(self):
        """Rewind to the checkpoint; returns the number of discarded steps."""
        ck = self.checkpoint
        discarded = 0
        for item in self._chunk_log:
            # v1.0 chunks had five fields; local-ARR chunks add local work.
            _, tier, steps, wall, equiv, *local = item
            lp_steps, lp_cells, lp_wall = (local + [0, 0, 0.0])[:3]
            cells = int(np.prod(self.tier_shapes[tier]))
            self.cost[tier]["steps"] -= steps
            self.cost[tier]["cell_steps"] -= steps * cells
            self.cost[tier]["wall"] -= wall
            self.wasted["steps"] += steps
            self.wasted["cell_steps"] += steps * cells
            self.wasted["wall"] += wall
            self.local_cost["steps"] -= lp_steps
            self.local_cost["cell_steps"] -= lp_cells
            self.local_cost["wall"] -= lp_wall
            self.local_wasted["steps"] += lp_steps
            self.local_wasted["cell_steps"] += lp_cells
            self.local_wasted["wall"] += lp_wall
            self.uniform_equiv_cell_steps -= equiv
            discarded += steps
        self._chunk_log.clear()
        for k in list(self.history):
            del self.history[k][ck["rows"]:]
        if ck["tier"] != self.tier:
            self.solver.state = None
            self.tier, self.solver = ck["tier"], self._get(ck["tier"])
        self.solver.restore(ck["snap"])
        lp = ck.get("local_patch")
        if lp is None:
            self.local_patch = None
        else:
            self.local_patch = _patch_from_snapshot(lp, self.solver.volume("velocity"))
        self._last_energy = ck["energy"]
        self._calm = 0
        self._local_calm = 0
        self._spec = 0
        return discarded

    # ---- observation -------------------------------------------------
    def _coarse_ratio(self):
        """Mode-number ratio between this tier and the next coarser one."""
        if self.tier <= self.min_tier:
            return None
        return min(c / f for c, f in zip(self.tier_shapes[self.tier - 1], self.shape))

    def _can_local_refine(self):
        return (self.cfg.enabled and self.cfg.local_3d
                and getattr(self.solver, "ndim", 0) == 3
                and (self.cfg.local_allow_gpu or not self.be.is_gpu))

    def _local_pressure(self):
        if self.local_patch is None:
            return float("nan")
        try:
            return self.local_patch.tail_fraction(self.cfg.theta)
        except Exception:
            return float("inf")

    def _observe(self, initial=False):
        cfg, s = self.cfg, self.solver
        ratio = self._coarse_ratio()
        ths = [cfg.theta] + ([cfg.theta * ratio] if ratio else [])
        fr = s.tail_fractions(ths)                       # may raise SolverFault
        p = fr[0]
        q = fr[1] if ratio else float("nan")
        d = s.diagnostics()
        if not all(math.isfinite(v) for v in d.values()):
            raise SolverFault("non-finite diagnostics")
        if (self.energy_monotone and self._last_energy is not None
                and d["energy"] > self._last_energy * (1.0 + cfg.energy_tol) + 1e-300):
            raise SolverFault("kinetic energy grew in an unforced flow")
        self._last_energy = d["energy"]
        self.guard_hits = sum(sv.guard_hits for sv in self._solvers.values())

        press_up = cfg.press_up
        can_promote = cfg.enabled and self.tier < self.max_tier_eff
        can_local = self._can_local_refine()
        local_p = self._local_pressure()
        if p > press_up and not initial:
            if self.local_patch is not None:
                # A healthy local patch owns this hot region.  Escalate only if
                # either the coarse domain or the fine correction is itself far
                # beyond the calibrated pressure threshold.
                if (math.isfinite(local_p)
                        and local_p > cfg.local_fail_factor * press_up):
                    if can_promote:
                        self._global_promote(reason="local patch saturated", pressure=max(p, local_p))
                        return
                    # no global tier left: let the finest-tier saturation policy handle it
                # otherwise keep the patch alive and continue on the current global tier
            elif can_local and cfg.rollback:
                if self._local_promote(reason="saturation pressure", pressure=p):
                    return
                if can_promote:
                    self._global_promote(reason="local refinement fallback", pressure=p)
                    return
            elif can_promote:
                self._global_promote(reason="saturation pressure", pressure=p)
                return
        if (p > press_up and not can_promote
                and (self.local_patch is None
                     or (math.isfinite(local_p)
                         and local_p > cfg.local_fail_factor * press_up))):
            if not self.saturated:
                self.saturated = True
                self._emit("saturated", tier=self.tier, pressure=p)
                if cfg.on_saturation == "sgs" and hasattr(s, "sgs") and s.sgs == "none":
                    s.sgs = "smagorinsky"
                    s.ffts_per_stage += 12
                    self._emit("sgs_on", tier=self.tier)
                if cfg.on_saturation == "stop":
                    raise SimulationFailed(
                        f"under-resolved on the finest tier {self.shape} (P={p:.3g}); "
                        "add a finer tier or lower the Reynolds number")
        elif self.saturated and p < cfg.warn_ratio * press_up:
            self.saturated = False
            self._emit("resolved", tier=self.tier, pressure=p)

        row = {"t": s.t, "step": self.total_steps, "tier": self.tier,
               "dt": s.dt if s.dt else 0.0, "energy": d["energy"], "enstrophy": d["enstrophy"],
               "dissipation": d["dissipation"], "umax": d["umax"], "pressure": p,
               "pressure_coarse": q, "saturated": bool(self.saturated),
               "local_active": bool(self.local_patch is not None),
               "local_volume": (self.local_patch.region.volume_fraction
                                if self.local_patch is not None else 0.0),
               "local_pressure": local_p}
        for k, v in d.items():
            if k not in row:
                if k not in self.history:
                    self.history[k] = [float("nan")] * len(self.history["t"])
                    self.extra_keys.append(k)
                row[k] = v
        for k in self.history:
            self.history[k].append(row.get(k, float("nan")))

        # checkpoint policy: refresh while clean, freeze while warned
        clean = p < cfg.warn_ratio * press_up or not (can_promote or can_local)
        if initial or self.checkpoint is None or clean:
            self._save_checkpoint()
        else:
            self._spec += 1
            if self._spec >= cfg.max_spec_obs:
                self._save_checkpoint()         # hovering below press_up: accept it

        # Local patch retirement and global-tier demotion have separate hysteresis.
        if self.local_patch is not None:
            if (p < cfg.down_ratio * press_up and self.local_patch.age >= cfg.local_min_age
                    and not self.saturated):
                self._local_calm += 1
                if self._local_calm >= cfg.hold and not initial:
                    self._retire_local(p)
            else:
                self._local_calm = 0
            self._calm = 0
        elif cfg.enabled and ratio and q < cfg.down_ratio * press_up and not self.saturated:
            # Demote only while the coarse-grid pressure is genuinely falling.  A flow
            # that has not started its cascade yet (pressure ~0 and flat, or rising)
            # looks "calm" but is about to need the resolution; demoting there and
            # re-promoting later leaves a lasting error (see METHOD.md, ARR, demotion).
            if self._calm == 0:
                self._calm_q0 = q
            self._calm += 1
            if self._calm >= cfg.hold and not initial and q < self._calm_q0:
                self._demote(q)
        else:
            self._calm = 0

    def _promote(self, reason, pressure=None):
        # Fault recovery always uses the proven whole-domain path.  Spatial ARR
        # is attempted only for a measured saturation-pressure event.
        if (reason == "saturation pressure" and self.local_patch is None
                and self._can_local_refine() and self.cfg.rollback):
            if self._local_promote(reason, pressure):
                return
        self._global_promote(reason, pressure)

    def _global_promote(self, reason, pressure=None):
        cfg = self.cfg
        t_detect, from_tier = self.solver.t, self.tier
        if cfg.rollback:
            discarded = self._restore_checkpoint()
        else:
            discarded = 0
        t_restore = self.solver.t
        # A whole-domain tier supersedes any local patch snapshot restored with
        # the checkpoint.
        self.local_patch = None
        state = self.solver.state
        self._switch(self.tier + 1, state, t_restore)
        self._emit("rollback_promote" if cfg.rollback else "promote",
                   reason=reason, from_tier=from_tier, to_tier=self.tier,
                   t_detect=t_detect, t_restore=t_restore, steps_discarded=discarded,
                   pressure=pressure)
        self._calm = 0
        self._retries = 0
        self.saturated = False
        self._observe(initial=True)

    def _local_promote(self, reason, pressure=None):
        """Rollback and replay the dirty interval with a temporary fine 3-D patch.

        Returns True on success.  Any localisation/patch failure rewinds again
        and returns False so the caller can use the original global promotion.
        """
        cfg = self.cfg
        if self.local_patch is not None:
            return True
        t_detect = self.solver.t
        from_tier = self.tier
        try:
            region, loc = locate_hot_region(
                self.solver, cfg.theta, cfg.local_target_fraction,
                cfg.local_padding, min_size=max(4, cfg.local_blend + 1))
        except Exception as exc:
            self._emit("local_fallback", reason=f"localisation failed: {exc}", pressure=pressure)
            return False
        if region.volume_fraction > cfg.local_max_volume or any(
                n >= N for n, N in zip(region.size, region.global_shape)):
            self._emit("local_fallback", reason="hot region too large", pressure=pressure,
                       region=region.to_dict(), localisation=loc)
            return False

        discarded = self._restore_checkpoint()
        t_restore = self.solver.t
        try:
            coarse_u = self.solver.volume("velocity")
            patch = _PATCH_SCHEMES[cfg.local_scheme](
                region, cfg.local_refine, self.solver.grid.lengths, self.solver.nu,
                self.solver.cfl, coarse_u, t_restore, blend_coarse=cfg.local_blend)
            patch.mark_born()
            self.local_patch = patch
            replay = self._run_coupled_until(t_detect)
            self._emit("rollback_local_refine", reason=reason, from_tier=from_tier,
                       to_tier=from_tier, t_detect=t_detect, t_restore=t_restore,
                       steps_discarded=discarded, pressure=pressure,
                       region=region.to_dict(), localisation=loc,
                       refine=cfg.local_refine, replay=replay)
            self._calm = self._local_calm = 0
            self._retries = 0
            self.saturated = False
            self._observe(initial=True)
            return True
        except Exception as exc:
            # Throw away the failed replay and return exactly to the clean
            # checkpoint before the caller tries the proven global path.
            try:
                self._restore_checkpoint()
            except Exception:
                pass
            self.local_patch = None
            self._emit("local_fallback", reason=f"local replay failed: {exc}", pressure=pressure,
                       region=region.to_dict(), localisation=loc)
            return False

    def _retire_local(self, pressure):
        patch = self.local_patch
        if patch is None:
            return
        self._emit("local_retire", pressure=pressure, region=patch.region.to_dict(),
                   age=patch.age, local_pressure=patch.last_pressure,
                   fine_steps=patch.steps)
        self.local_patch = None
        self._local_calm = 0
        # The merged coarse state at retirement is now an ordinary clean ARR
        # checkpoint.  This prevents a later rollback from resurrecting a dead patch.
        self._save_checkpoint()

    def _run_coupled_until(self, target, max_steps=None):
        """Advance coarse global state and active local patch to ``target``."""
        if self.local_patch is None:
            return {"coarse_steps": 0, "fine_steps": 0}
        s, tier = self.solver, self.tier
        cells = int(np.prod(self.shape))
        fine_global = self.tier_shapes[self.max_tier_eff]
        equiv_per_step = float(np.prod(fine_global)) * max(
            f / c for f, c in zip(fine_global, self.shape))
        nc = nf = 0
        while s.t < target - 1e-12 * max(1.0, abs(target)):
            if max_steps is not None and nc >= max_steps:
                break
            u0 = s.volume("velocity")
            t0 = s.t
            tic = time.perf_counter()
            if s.step(t_limit=target) == 0.0:
                break
            self.be.synchronize()
            coarse_wall = time.perf_counter() - tic
            u1 = s.volume("velocity")
            p0_steps = self.local_patch.steps
            p0_wall = self.local_patch.wall
            self.local_patch.advance_interval(t0, s.t, u0, u1)
            merge = self.local_patch.merge_into(s)
            if (not math.isfinite(merge["divergence"])
                    or not math.isfinite(merge["energy_rel_change"])
                    or merge["energy_rel_change"] > self.cfg.local_merge_energy_tol):
                raise SolverFault(
                    "local merge validation failed "
                    f"(div={merge['divergence']:.3g}, dE={merge['energy_rel_change']:.3g})")
            lp_steps = self.local_patch.steps - p0_steps
            lp_wall = self.local_patch.wall - p0_wall
            lp_cells = lp_steps * self.local_patch.cell_count
            self._account_coupled(tier, 1, cells, coarse_wall, equiv_per_step,
                                  lp_steps, lp_cells, lp_wall)
            nc += 1
            nf += lp_steps
        return {"coarse_steps": nc, "fine_steps": nf,
                "patch_cells": self.local_patch.cell_count if self.local_patch else 0}

    def _demote(self, q):
        from_tier = self.tier
        self._switch(self.tier - 1, self.solver.state, self.solver.t)
        self._emit("demote", from_tier=from_tier, to_tier=self.tier, pressure_coarse=q)
        self._calm = 0
        self._observe(initial=True)

    def _handle_fault(self, exc):
        cfg = self.cfg
        if cfg.enabled and self.tier < self.max_tier_eff:
            self._promote(reason=str(exc))
            return
        # finest tier: rewind and retry with a smaller CFL number
        self._retries += 1
        if self._retries > cfg.max_retries or self.checkpoint is None:
            raise SimulationFailed(
                f"unrecoverable on the finest tier {self.shape}: {exc}") from exc
        discarded = self._restore_checkpoint()
        self.solver.cfl *= 0.5
        self.solver.dt = None
        self._emit("rollback_retry", reason=str(exc), tier=self.tier,
                   steps_discarded=discarded, cfl=self.solver.cfl)

    # ---- driving -----------------------------------------------------
    def run_chunk(self, t_limit=None):
        """Advance up to ``obs_every`` coarse steps, then observe and adapt.

        With an active local patch, each coarse step is coupled to fine patch
        subcycling and a globally projected correction merge.
        """
        if self.local_patch is not None:
            fault = None
            t_before = self.solver.t
            target = float("inf") if t_limit is None else float(t_limit)
            try:
                replay = self._run_coupled_until(target, max_steps=self.cfg.obs_every)
            except (SolverFault, RuntimeError, FloatingPointError) as exc:
                fault = exc
                replay = {"coarse_steps": 0}
            n = replay.get("coarse_steps", 0)
            if fault is None and self.solver.t > t_before:
                try:
                    self._observe()
                except SolverFault as exc:
                    fault = exc
            if fault is not None:
                try:
                    self._handle_fault(fault)
                except SolverFault as exc:
                    raise SimulationFailed(f"recovery failed: {exc}") from exc
            return n

        s, tier = self.solver, self.tier
        cells = int(np.prod(self.shape))
        fine = self.tier_shapes[self.max_tier_eff]
        equiv_per_step = float(np.prod(fine)) * max(f / c for f, c in zip(fine, self.shape))
        t0 = time.perf_counter()
        n = 0
        fault = None
        try:
            for _ in range(self.cfg.obs_every):
                if s.step(t_limit=t_limit) == 0.0:
                    break
                n += 1
        except SolverFault as exc:
            fault = exc
        self.be.synchronize()
        self._account(tier, n, cells, time.perf_counter() - t0, equiv_per_step)
        if fault is None and n:
            try:
                self._observe()
            except SolverFault as exc:
                fault = exc
        if fault is not None:
            try:
                self._handle_fault(fault)
            except SolverFault as exc:
                raise SimulationFailed(f"recovery failed: {exc}") from exc
        return n

    def _account(self, tier, n, cells, wall, equiv_per_step):
        self.cost[tier]["steps"] += n
        self.cost[tier]["cell_steps"] += n * cells
        self.cost[tier]["wall"] += wall
        self.uniform_equiv_cell_steps += n * equiv_per_step
        self.total_steps += n
        self._chunk_log.append((self.solver.t, tier, n, wall, n * equiv_per_step, 0, 0, 0.0))

    def _account_coupled(self, tier, n, cells, wall, equiv_per_step,
                         lp_steps, lp_cells, lp_wall):
        self.cost[tier]["steps"] += n
        self.cost[tier]["cell_steps"] += n * cells
        self.cost[tier]["wall"] += wall
        self.local_cost["steps"] += lp_steps
        self.local_cost["cell_steps"] += lp_cells
        self.local_cost["wall"] += lp_wall
        self.uniform_equiv_cell_steps += n * equiv_per_step
        self.total_steps += n
        self._chunk_log.append((self.solver.t, tier, n, wall, n * equiv_per_step,
                                lp_steps, lp_cells, lp_wall))

    # ---- reporting ---------------------------------------------------
    def cost_summary(self) -> dict:
        kept = sum(c["cell_steps"] for c in self.cost.values())
        total = (kept + self.wasted["cell_steps"] + self.local_cost["cell_steps"]
                 + self.local_wasted["cell_steps"])
        wall = (sum(c["wall"] for c in self.cost.values()) + self.wasted["wall"]
                + self.local_cost["wall"] + self.local_wasted["wall"])
        return {
            "tiers": [{"tier": i, "shape": list(self.tier_shapes[i]), **self.cost[i]}
                      for i in range(self.n_tiers)],
            "wasted": dict(self.wasted),
            "local_patch": dict(self.local_cost),
            "local_wasted": dict(self.local_wasted),
            "cell_steps": total,
            "wall_seconds": wall,
            "uniform_finest_cell_steps_est": self.uniform_equiv_cell_steps,
            "cell_step_saving_est": (self.uniform_equiv_cell_steps / total) if total else 1.0,
            "rollbacks": sum(1 for e in self.events if e["type"] in ("rollback_promote", "rollback_local_refine")),
            "promotions": sum(1 for e in self.events if e["type"] in ("rollback_promote", "promote", "rollback_local_refine")),
            "demotions": sum(1 for e in self.events if e["type"] in ("demote", "local_retire")),
            "guard_hits": self.guard_hits,
        }

    def config_dict(self):
        return asdict(self.cfg)
