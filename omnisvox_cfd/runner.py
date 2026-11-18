"""High-level simulation driver used by the CLI, the desktop app and the server."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np

from . import __version__
from .arr import ARRConfig, ARRController, SimulationFailed
from .backend import get_backend, Backend
from .cases import get_case

MAX_CELLS_HARD = 512 ** 3


@dataclass
class SimConfig:
    case: str = "taylor_green_3d"
    params: dict = field(default_factory=dict)
    tiers: list | None = None          # explicit list of grid shapes, coarse -> fine
    preset: str | None = None          # or the name of a case tier preset
    t_end: float | None = None
    device: str = "auto"               # auto | cpu | gpu
    precision: str = "double"          # double | single
    cfl: float = 0.5
    sgs: str = "none"                  # 3-D only: none | smagorinsky
    arr: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config key(s): {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_config(cfg: SimConfig | dict):
    """Validate a config; returns (cfg, case, params, tier_shapes, arr_cfg)."""
    if isinstance(cfg, dict):
        cfg = SimConfig.from_dict(cfg)
    case = get_case(cfg.case)
    params = case.resolve_params(cfg.params)
    if cfg.tiers:
        tiers = [tuple(int(n) for n in s) for s in cfg.tiers]
    else:
        preset = cfg.preset or case.default_preset
        if preset not in case.tier_presets:
            raise ValueError(f"unknown preset '{preset}' for case '{case.key}'. "
                             f"Available: {list(case.tier_presets)}")
        tiers = [tuple(s) for s in case.tier_presets[preset]]
    if not 1 <= len(tiers) <= 5:
        raise ValueError("between 1 and 5 resolution tiers are supported")
    for s in tiers:
        if len(s) != case.ndim:
            raise ValueError(f"case '{case.key}' is {case.ndim}-D but tier {s} is not")
        if any(n % 2 or n < 8 for n in s):
            raise ValueError(f"grid sizes must be even and >= 8, got {s}")
        if int(np.prod(s)) > MAX_CELLS_HARD:
            raise ValueError(f"tier {s} exceeds the supported maximum grid size")
    for a, b in zip(tiers, tiers[1:]):
        if any(nb <= na for na, nb in zip(a, b)):
            raise ValueError("tiers must be strictly increasing in every direction")
    t_end = case.t_end if cfg.t_end is None else float(cfg.t_end)
    if not (math.isfinite(t_end) and t_end > 0):
        raise ValueError("t_end must be a positive number")
    if not 0.05 <= cfg.cfl <= 1.0:
        raise ValueError("cfl must lie in [0.05, 1.0]")
    if cfg.sgs not in ("none", "smagorinsky"):
        raise ValueError("sgs must be 'none' or 'smagorinsky'")
    if cfg.sgs != "none" and case.ndim != 3:
        raise ValueError("the Smagorinsky model is available for 3-D cases only")
    if cfg.precision not in ("double", "single"):
        raise ValueError("precision must be 'double' or 'single'")
    if cfg.device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be auto, cpu or gpu")
    arr_d = dict(cfg.arr or {})
    unknown = set(arr_d) - set(ARRConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown arr option(s): {sorted(unknown)}")
    arr_d.setdefault("press_up", case.press_up)
    if case.start_fine and arr_d.get("enabled", True) and "start_tier" not in arr_d:
        arr_d["start_tier"] = len(tiers) - 1
    if len(tiers) == 1:
        arr_d["enabled"] = False
        arr_d["start_tier"] = 0
    arr = ARRConfig(**arr_d)
    arr.validate(len(tiers))
    return cfg, case, params, tiers, arr


class Simulation:
    """One run.  Call :meth:`advance` repeatedly; inspect ``history``/``events``."""

    def __init__(self, cfg: SimConfig | dict, backend: Backend | None = None,
                 on_event: Callable[[dict], None] | None = None):
        self.cfg, self.case, self.params, self.tiers, self.arr_cfg = resolve_config(cfg)
        self.be = backend or get_backend(self.cfg.device, self.cfg.precision)
        self.t_end = self.case.t_end if self.cfg.t_end is None else float(self.cfg.t_end)
        self.status = "ready"            # ready | running | done | failed
        self.error = None
        self.wall = 0.0

        def make(tier):
            kw = {"cfl": self.cfg.cfl}
            if self.case.ndim == 3:
                kw["sgs"] = self.cfg.sgs
            return self.case.make_solver(self.tiers[tier], self.params, self.be, **kw)

        def init(solver):
            return self.case.initial_state(solver, self.params)

        self._make, self._init = make, init
        self.ctrl = ARRController(make, self.tiers, init, self.arr_cfg,
                                  energy_monotone=self.case.energy_monotone, on_event=on_event)
        self._initial_promotion()

    def _initial_promotion(self):
        """If the initial condition is already under-resolved, start finer.

        The initial field is re-sampled from its definition on the finer grid
        rather than prolonged, so nothing is lost.
        """
        c = self.ctrl
        cfg = c.cfg
        while cfg.enabled and c.tier < c.max_tier_eff and c.history["pressure"][-1] > cfg.press_up:
            # In 3-D local ARR, an initially hot *localised* field is allowed one
            # speculative coarse chunk.  If it is genuinely local the controller
            # rolls back to t=0 and replays with a patch; if it is distributed,
            # localisation rejects it and the existing global promotion runs.
            if (self.case.ndim == 3 and cfg.local_3d
                    and (cfg.local_allow_gpu or not self.be.is_gpu)):
                break
            p = c.history["pressure"][-1]
            start = c.tier + 1
            arr = ARRConfig(**{**asdict(cfg), "start_tier": start})
            events = c.events
            on_event = c.on_event
            c.solver.state = None
            self.be.free_memory()
            self.ctrl = c = ARRController(self._make, self.tiers, self._init, arr,
                                          energy_monotone=self.case.energy_monotone,
                                          on_event=on_event)
            ev = {"type": "initial_promote", "t": 0.0, "step": 0, "from_tier": start - 1,
                  "to_tier": start, "pressure": p,
                  "reason": "initial condition under-resolved"}
            c.events[:0] = events + [ev]
            if on_event:
                on_event(ev)
            cfg = c.cfg

    # ---- convenience accessors --------------------------------------
    @property
    def t(self):
        return self.ctrl.t

    @property
    def tier(self):
        return self.ctrl.tier

    @property
    def shape(self):
        return self.ctrl.shape

    @property
    def history(self):
        return self.ctrl.history

    @property
    def events(self):
        return self.ctrl.events

    @property
    def solver(self):
        return self.ctrl.solver

    @property
    def done(self):
        return self.status in ("done", "failed")

    @property
    def progress(self):
        return min(1.0, self.t / self.t_end)

    # ---- stepping ----------------------------------------------------
    def advance(self, max_wall: float | None = None, max_chunks: int | None = None,
                until: float | None = None) -> str:
        """Run until ``until``/``t_end``, a wall-clock budget, or a chunk budget."""
        if self.done:
            return self.status
        self.status = "running"
        target = self.t_end if until is None else min(float(until), self.t_end)
        t0 = time.perf_counter()
        chunks = 0
        try:
            while self.t < target - 1e-12 * max(1.0, target):
                self.ctrl.run_chunk(t_limit=target)
                chunks += 1
                if max_chunks is not None and chunks >= max_chunks:
                    break
                if max_wall is not None and time.perf_counter() - t0 >= max_wall:
                    break
        except SimulationFailed as exc:
            self.status, self.error = "failed", str(exc)
        finally:
            self.wall += time.perf_counter() - t0
        if self.status != "failed" and self.t >= self.t_end - 1e-12 * max(1.0, self.t_end):
            self.status = "done"
        return self.status

    def run(self):
        while not self.done:
            self.advance()
        return self

    # ---- output ------------------------------------------------------
    def field(self, name: str | None = None, **kw) -> np.ndarray:
        return self.solver.field(name or self.case.default_field, **kw)

    def snapshot(self) -> dict:
        """Small live-status dict (cheap; no field data)."""
        h = self.history
        last = {k: (v[-1] if v else None) for k, v in h.items()}
        return {
            "status": self.status, "t": self.t, "t_end": self.t_end, "progress": self.progress,
            "tier": self.tier, "shape": list(self.shape), "steps": self.ctrl.total_steps,
            "wall": self.wall, "last": last, "n_events": len(self.events),
            "saturated": self.ctrl.saturated, "error": self.error,
        }

    def summary(self) -> dict:
        h = self.history
        cost = self.ctrl.cost_summary()
        out = {
            "version": __version__,
            "status": self.status, "error": self.error,
            "case": self.case.key, "params": self.params,
            "tiers": [list(s) for s in self.tiers],
            "arr": self.ctrl.config_dict(),
            "backend": self.be.describe(),
            "t_final": self.t, "t_end": self.t_end, "steps": self.ctrl.total_steps,
            "wall_seconds": self.wall,
            "cost": cost,
            "final": {k: (v[-1] if v else None) for k, v in h.items()},
            "time_saturated": _time_flagged(h["t"], h["saturated"]),
            "events": self.events,
        }
        if h["dissipation"]:
            i = int(np.argmax(h["dissipation"]))
            out["peak_dissipation"] = {"value": h["dissipation"][i], "t": h["t"][i]}
        return out

    def export_npz(self, path, fields=None):
        from .io import save_npz
        return save_npz(self, path, fields)

    def export_csv(self, path):
        from .io import save_history_csv
        return save_history_csv(self.history, path)

    def export_vtk(self, path):
        from .io import save_vtk
        return save_vtk(self, path)


def _time_flagged(t, flags):
    """Total simulated time spent with a flag raised."""
    total = 0.0
    for i in range(1, len(t)):
        if flags[i] and flags[i - 1] and t[i] > t[i - 1]:
            total += t[i] - t[i - 1]
    return total


def json_safe(obj):
    """Recursively convert numpy scalars/NaN so ``json.dumps`` produces valid JSON."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj
