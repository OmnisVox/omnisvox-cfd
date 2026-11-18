"""Fourier grids, dealiasing, spectral diagnostics and grid-transfer operators.

Conventions
-----------
* Physical fields have shape ``(Nx, Ny)`` or ``(Nx, Ny, Nz)`` (``indexing='ij'``).
* Spectral fields are the *unnormalised* real FFT over all spatial axes, so the
  last axis is half-length (``N//2 + 1``).
* All grid sizes are even.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .backend import Backend


class SpectralGrid:
    """Wavenumbers, masks and Parseval weights for one periodic grid."""

    def __init__(self, shape: Sequence[int], lengths: Sequence[float], be: Backend):
        shape = tuple(int(n) for n in shape)
        lengths = tuple(float(L) for L in lengths)
        if len(shape) not in (2, 3) or len(shape) != len(lengths):
            raise ValueError("shape/lengths must both be 2-D or 3-D")
        if any(n % 2 or n < 8 for n in shape):
            raise ValueError(f"grid sizes must be even and >= 8, got {shape}")
        self.shape, self.lengths, self.be = shape, lengths, be
        self.ndim = len(shape)
        self.ncells = int(np.prod(shape))
        self.spacing = tuple(L / n for L, n in zip(lengths, shape))
        self.hshape = shape[:-1] + (shape[-1] // 2 + 1,)
        xp, rd = be.xp, be.rdtype

        # integer mode numbers and physical wavenumbers, broadcastable
        modes, k = [], []
        for ax, (n, L) in enumerate(zip(shape, lengths)):
            if ax == self.ndim - 1:
                m = np.arange(n // 2 + 1)
            else:
                m = np.fft.fftfreq(n, 1.0 / n)
            bshape = [1] * self.ndim
            bshape[ax] = m.size
            modes.append(m.reshape(bshape))
            k.append((2.0 * np.pi / L) * m.reshape(bshape))
        self.k = [be.asarray(ki, dtype=rd) for ki in k]
        k2 = sum(ki ** 2 for ki in k)
        self.k2 = be.asarray(k2, dtype=rd)
        k2s = k2.copy()
        k2s[(0,) * self.ndim] = 1.0
        self.inv_k2 = be.asarray(1.0 / k2s, dtype=rd)
        self.inv_k2[(0,) * self.ndim] = 0.0

        # 2/3-rule dealiasing: keep |n_i| < N_i/3 on every axis (box truncation)
        rbox = np.zeros(self.hshape)
        for m, n in zip(modes, shape):
            rbox = np.maximum(rbox, np.abs(m) / (n / 3.0))
        self.rbox = be.asarray(rbox, dtype=rd)          # 0 .. (>1 outside the box)
        self.dealias = be.asarray(rbox < 1.0, dtype=rd)  # 1 inside, 0 outside
        self.kcut = min((n / 3.0) * (2.0 * np.pi / L) for n, L in zip(shape, lengths))

        # Parseval weights for the half-complex layout, already divided by N^2
        w = np.full(self.hshape, 2.0)
        w[..., 0] = 1.0
        w[..., -1] = 1.0
        self.pw = be.asarray(w / float(self.ncells) ** 2, dtype=rd)
        self._kmag = None

    # ---- transforms --------------------------------------------------
    def fft(self, a):
        return self.be.rfftn(a, self.ndim)

    def ifft(self, a_hat):
        return self.be.irfftn(a_hat, self.shape)

    # ---- integrals ---------------------------------------------------
    def mean_sq(self, a_hat, weight=None) -> float:
        """Volume mean of f^2 from its spectrum (optionally weighted per mode)."""
        xp = self.be.xp
        d = (a_hat.real ** 2 + a_hat.imag ** 2) * self.pw
        if weight is not None:
            d = d * weight
        return self.be.scalar(xp.sum(d))

    def coords(self):
        """Host-side cell coordinates (numpy, ``indexing='ij'``)."""
        axes = [np.arange(n) * (L / n) for n, L in zip(self.shape, self.lengths)]
        return np.meshgrid(*axes, indexing="ij")

    def shell_spectrum(self, density_hat):
        """Bin a per-mode density (already Parseval weighted) into unit shells.

        Returns (k_centres, E(k)) as numpy arrays; shells are in units of the
        smallest fundamental wavenumber.
        """
        xp, be = self.be.xp, self.be
        if self._kmag is None:
            k0 = min(2.0 * np.pi / L for L in self.lengths)
            self._k0 = k0
            self._kmag = xp.rint(xp.sqrt(self.k2) / k0).astype(xp.int64)
        nb = int(be.scalar(xp.max(self._kmag))) + 1
        spec = xp.bincount(self._kmag.ravel(), weights=density_hat.ravel().astype(xp.float64),
                           minlength=nb)
        return np.arange(nb) * self._k0, be.to_numpy(spec)


def resample_hat(a_hat, shape_from: Sequence[int], shape_to: Sequence[int], be: Backend):
    """Spectral grid transfer (prolongation by zero-padding, restriction by truncation).

    ``a_hat`` may carry leading component axes, e.g. ``(3, Nx, Ny, Nz//2+1)``.
    The transfer is exact for every Fourier mode both grids represent, drops
    the Nyquist modes (which are ambiguous), and rescales for the unnormalised
    FFT so that the physical field values are unchanged.
    """
    shape_from, shape_to = tuple(shape_from), tuple(shape_to)
    nd = len(shape_from)
    if shape_from == shape_to:
        return a_hat.copy()
    xp = be.xp
    lead = a_hat.shape[:-nd]
    hto = shape_to[:-1] + (shape_to[-1] // 2 + 1,)
    out = xp.zeros(lead + hto, dtype=a_hat.dtype)

    # per-axis (source slice, destination slice) blocks
    blocks = []
    for ax, (n, m) in enumerate(zip(shape_from, shape_to)):
        keep = min(n, m) // 2  # modes 0..keep-1 positive, -(keep-1)..-1 negative
        if ax == nd - 1:
            blocks.append([(slice(0, keep), slice(0, keep))])
        else:
            pos = (slice(0, keep), slice(0, keep))
            neg = (slice(n - keep + 1, n), slice(m - keep + 1, m))
            blocks.append([pos, neg])

    def _rec(ax, src, dst):
        if ax == nd:
            idx_s = (Ellipsis,) + tuple(src)
            idx_d = (Ellipsis,) + tuple(dst)
            out[idx_d] = a_hat[idx_s]
            return
        for s, d in blocks[ax]:
            _rec(ax + 1, src + [s], dst + [d])

    _rec(0, [], [])
    scale = float(np.prod(shape_to)) / float(np.prod(shape_from))
    out *= scale
    return out
