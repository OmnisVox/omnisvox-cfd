"""Array backend selection: NumPy/SciPy on CPU, CuPy on GPU.

Every solver in this package takes a ``Backend`` and uses only ``be.xp`` plus
the FFT wrappers below, so the same code path runs on both devices.  Rules the
solvers follow so that the GPU path stays correct:

* never call ``numpy`` functions on state arrays - always ``be.xp``;
* move data to host memory only through ``be.to_numpy`` / ``be.scalar``;
* build initial conditions on the host and upload them with ``be.asarray``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # SciPy's pocketfft is multi-threaded and keeps single precision.
    import scipy.fft as _cpu_fft

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dependency in practice
    _cpu_fft = np.fft
    _HAVE_SCIPY = False


class BackendError(RuntimeError):
    pass


def gpu_status() -> dict:
    """Describe GPU availability without raising."""
    info = {"available": False, "reason": "", "name": None, "memory_gb": None}
    try:
        import cupy as cp  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - depends on host
        info["reason"] = f"CuPy is not installed ({exc.__class__.__name__})"
        return info
    try:  # pragma: no cover - needs a GPU
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            info["reason"] = "no CUDA device found"
            return info
        dev = cp.cuda.Device(0)
        free, total = dev.mem_info
        try:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            name = "CUDA device 0"
        info.update(available=True, name=name, memory_gb=round(total / 2**30, 2),
                    free_gb=round(free / 2**30, 2))
    except Exception as exc:  # pragma: no cover
        info["reason"] = f"CUDA initialisation failed: {exc}"
    return info


@dataclass
class Backend:
    """A thin, explicit wrapper around the array module in use."""

    name: str                      # "numpy" or "cupy"
    xp: Any                        # array module
    precision: str = "double"      # "double" or "single"
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)

    def __post_init__(self):
        if self.precision not in ("double", "single"):
            raise BackendError("precision must be 'double' or 'single'")
        self.rdtype = np.float64 if self.precision == "double" else np.float32
        self.cdtype = np.complex128 if self.precision == "double" else np.complex64
        if self.name == "cupy":  # pragma: no cover - needs a GPU
            import cupy.fft as cfft

            self._fft = cfft
        else:
            self._fft = _cpu_fft

    # ---- device info -------------------------------------------------
    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def describe(self) -> str:
        if self.is_gpu:  # pragma: no cover
            st = gpu_status()
            return f"GPU: {st.get('name')} ({st.get('memory_gb')} GB), {self.precision} precision"
        return f"CPU: {self.workers} thread(s), {self.precision} precision"

    # ---- transfers ---------------------------------------------------
    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a) -> np.ndarray:
        if self.is_gpu:  # pragma: no cover
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def scalar(self, a) -> float:
        """Host float from a 0-d array (works for numpy and cupy)."""
        return float(a)

    def free_memory(self):
        if self.is_gpu:  # pragma: no cover
            self.xp.get_default_memory_pool().free_all_blocks()

    def synchronize(self):
        """Block until queued GPU work is done (for honest timings)."""
        if self.is_gpu:  # pragma: no cover
            self.xp.cuda.Stream.null.synchronize()

    # ---- FFTs (real <-> half-complex over all spatial axes) ---------
    def _kw(self):
        return {"workers": self.workers} if (not self.is_gpu and _HAVE_SCIPY) else {}

    def rfftn(self, a, ndim: int):
        """Forward real FFT over the last ``ndim`` axes."""
        return self._fft.rfftn(a, axes=tuple(range(-ndim, 0)), **self._kw())

    def irfftn(self, a, shape):
        """Inverse real FFT over the last ``len(shape)`` axes to ``shape``."""
        nd = len(shape)
        return self._fft.irfftn(a, s=tuple(shape), axes=tuple(range(-nd, 0)), **self._kw())


def get_backend(device: str = "auto", precision: str = "double", workers: int | None = None) -> Backend:
    """Return a backend.

    device: "cpu", "gpu", or "auto" (GPU if CuPy and a CUDA device are present).
    """
    device = (device or "auto").lower()
    if device not in ("auto", "cpu", "gpu"):
        raise BackendError("device must be 'auto', 'cpu' or 'gpu'")
    kw = {} if workers is None else {"workers": int(workers)}
    if device in ("auto", "gpu"):
        st = gpu_status()
        if st["available"]:  # pragma: no cover - needs a GPU
            import cupy as cp

            return Backend("cupy", cp, precision, **kw)
        if device == "gpu":
            raise BackendError(
                "GPU requested but not usable: " + st["reason"] +
                ". Install a CuPy wheel matching your CUDA version, e.g. "
                "'pip install cupy-cuda12x'."
            )
    return Backend("numpy", np, precision, **kw)
