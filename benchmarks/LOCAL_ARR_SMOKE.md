# Local 3-D ARR benchmarks (1.1.1)

Two cases. Every number below was measured on this build (CPU, NumPy/SciPy).
Errors are relative velocity errors against a uniform fine reference restricted
to the coarse band. "Global from same rollback" is the whole-domain promotion a
local patch competes with: same trigger time, same 2× ratio, no localisation.

## Localized case — `python benchmarks/local_arr_localized.py`

Compact vortex packet in a 32³ box, Re = 1000, t = 0.12, reference 128³.

| run | error | wall | patch volume |
|---|---:|---:|---:|
| uniform 32³ | 0.0531 | 0.14 s | |
| uniform 40³ | 0.0105 | 0.31 s | |
| uniform 48³ | 0.0040 | 0.68 s | |
| uniform 64³ | 0.0003 | 2.26 s | |
| local, legacy FD patch, target 0.7 | 0.0691 | 1.38 s | 12% |
| local, spectral patch, target 0.7 | 0.0408 | 2.18 s | 12% |
| local, legacy FD patch, target 0.9 | 0.0530 | 3.38 s | 38% |
| local, spectral patch, target 0.9 | 0.0199 | 7.20 s | 38% |
| global 64³ from same rollback (t = 0.0301) | 0.0082 | 1.68 s | 100% |

## Original smoke case — `python benchmarks/local_arr_smoke.py`

Vortex packet in a 24³ box, Re = 1000, t = 0.08, reference 48³. This packet
fills most of the box, so it is not a good test of *local* refinement.

| run | error | wall |
|---|---:|---:|
| coarse 24³ | 0.0377 | 0.05 s |
| local, legacy FD patch (1.1.0) | 0.0350 | 1.2–1.6 s |
| local, spectral patch (1.1.1) | 0.0309 | 1.4 s |
| global 48³ from same rollback | 0.0106 | 0.49 s |
| uniform 28³ from t = 0 | 0.0047 | 0.10 s |

## What this shows

* The 1.1.0 finite-difference patch is **not more accurate than the coarse grid
  alone** on the localized case (0.069 / 0.053 vs 0.053). Its collocated
  second-order projection enforces a discrete constraint the true field does not
  satisfy, distorting resolved fine scales every substep; better stencils and 4×
  refinement do not help because the projection undoes them.
* The 1.1.1 spectral patch fixes that. Fed a perfect coarse band, it reproduces
  the fine scales in its core to 0.3% (the coarse band alone is off by 11.5%).
  Coupled, it beats the coarse grid by 2.7× at 38% patch volume, and the error
  outside the patch stays at or below the coarse run's.
* What remains is coverage. Activity outside the patch is unresolved, and fine
  structure leaving the patch is lost in the taper, so error falls as coverage
  rises (12% → 38% volume: 0.041 → 0.020).
* **Neither patch beats whole-domain promotion from the same rollback point**,
  in error or in wall time, on either case. Uniform 40³ from t = 0 is better
  still. Patch cost is dominated by Python-level work (partial DFTs, per-stage
  FFTs); it would need native/GPU kernels *and* a genuinely small hot region in
  a large domain before the local path can pay off.

`arr.local_3d` is therefore off by default. Enable it for research only.
