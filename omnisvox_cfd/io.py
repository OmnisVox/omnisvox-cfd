"""Exports: NPZ snapshots, CSV histories, legacy-VTK volumes, PNG images."""
from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# colour maps (small anchor tables, linearly interpolated to 256 entries)
# ---------------------------------------------------------------------------
_ANCHORS = {
    # blue - near-white - red, for signed fields such as vorticity
    "diverging": [(0.00, (33, 62, 140)), (0.25, (72, 134, 201)), (0.45, (190, 214, 235)),
                  (0.50, (244, 243, 240)), (0.55, (240, 205, 190)), (0.75, (207, 98, 79)),
                  (1.00, (135, 26, 38))],
    # single-hue ramp (deep navy -> blue -> pale), for magnitudes
    "sequential": [(0.00, (10, 22, 40)), (0.25, (20, 54, 104)), (0.50, (42, 120, 214)),
                   (0.75, (134, 186, 240)), (1.00, (238, 246, 255))],
}


def colormap(name: str) -> np.ndarray:
    pts = _ANCHORS[name]
    xs = np.array([p[0] for p in pts])
    cols = np.array([p[1] for p in pts], dtype=float)
    x = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(x, xs, cols[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def field_limits(a: np.ndarray, symmetric: bool, pct: float = 99.5):
    """Robust colour limits for a field."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(a), pct)) or 1.0
        return -m, m
    lo, hi = float(np.percentile(a, 100 - pct)), float(np.percentile(a, pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def field_to_rgb(a: np.ndarray, symmetric: bool = True, limits=None, mask=None) -> np.ndarray:
    """Map a 2-D field (shape (Nx, Ny)) to an image (rows = y, top = max y)."""
    lo, hi = limits if limits is not None else field_limits(a, symmetric)
    lut = colormap("diverging" if symmetric else "sequential")
    idx = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255)
    idx = np.nan_to_num(idx, nan=0.0).astype(np.uint8)
    rgb = lut[idx]
    if mask is not None:
        solid = np.asarray(mask)[..., None]
        rgb = (rgb * (1 - solid) + np.array([58, 60, 66]) * solid).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(rgb, (1, 0, 2))[::-1])


def encode_png(rgb: np.ndarray) -> bytes:
    """Minimal dependency-free PNG encoder for an (H, W, 3) uint8 array."""
    h, w, _ = rgb.shape
    raw = np.empty((h, 1 + 3 * w), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rgb.reshape(h, 3 * w)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def upscale(rgb: np.ndarray, min_side: int = 384) -> np.ndarray:
    """Nearest-neighbour integer upscale so small grids are still viewable."""
    f = max(1, int(np.ceil(min_side / max(1, min(rgb.shape[:2])))))
    return rgb if f == 1 else np.repeat(np.repeat(rgb, f, axis=0), f, axis=1)


def frame_png(sim, name=None, limits=None, min_side=384) -> bytes:
    a = sim.field(name)
    name = name or sim.case.default_field
    symmetric = name in ("vorticity", "vorticity_z", "u", "v", "w", "streamfunction")
    mask = None
    bm = getattr(sim.solver, "body_mask", None)
    if bm is not None:
        mask = sim.be.to_numpy(bm)
    return encode_png(upscale(field_to_rgb(a, symmetric, limits, mask), min_side))


# ---------------------------------------------------------------------------
# data exports
# ---------------------------------------------------------------------------
def save_history_csv(history: dict, path):
    path = Path(path)
    keys = list(history)
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(keys)
        for row in zip(*(history[k] for k in keys)):
            wr.writerow(row)
    return path


def save_npz(sim, path, fields=None):
    """Final-state snapshot: 2-D fields, or velocity + vorticity volumes in 3-D."""
    from .runner import json_safe

    if not hasattr(path, "write"):          # a path; file-like objects pass through
        path = Path(path)
    out = {"t": sim.t, "shape": np.array(sim.shape), "lengths": np.array(sim.solver.grid.lengths),
           "summary_json": json.dumps(json_safe(sim.summary()))}
    for k, v in sim.history.items():
        out["history_" + k] = np.asarray(v, dtype=float)
    if sim.case.ndim == 2:
        for name in fields or ("vorticity", "u", "v"):
            out[name] = sim.field(name)
    else:
        out["velocity"] = sim.solver.volume("velocity")
        out["vorticity"] = sim.solver.volume("vorticity")
    np.savez_compressed(path, **out)
    return path


def save_vtk(sim, path):
    """Legacy binary VTK (STRUCTURED_POINTS) readable by ParaView/VisIt."""
    path = Path(path)
    g = sim.solver.grid
    if sim.case.ndim == 2:
        nx, ny = g.shape
        dims, spacing = (nx, ny, 1), (g.spacing[0], g.spacing[1], 1.0)
        scalars = {"vorticity": sim.field("vorticity")[..., None]}
        vec = np.stack((sim.field("u"), sim.field("v"), np.zeros((nx, ny))))[..., None]
    else:
        dims, spacing = g.shape, g.spacing
        vec = sim.solver.volume("velocity")
        w = sim.solver.volume("vorticity")
        scalars = {"vorticity_magnitude": np.sqrt((w ** 2).sum(axis=0))}
    n = int(np.prod(dims))
    with path.open("wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\nOmnisVox CFD\nBINARY\nDATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {dims[0]} {dims[1]} {dims[2]}\n".encode())
        fh.write(b"ORIGIN 0 0 0\n")
        fh.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n".encode())
        fh.write(f"POINT_DATA {n}\n".encode())
        for name, s in scalars.items():
            fh.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
            fh.write(np.asarray(s, dtype=">f4").transpose(2, 1, 0).tobytes())   # x fastest
            fh.write(b"\n")
        fh.write(b"VECTORS velocity float\n")
        v = np.asarray(vec, dtype=">f4").transpose(3, 2, 1, 0)                 # (z, y, x, comp)
        fh.write(v.tobytes())
        fh.write(b"\n")
    return path
