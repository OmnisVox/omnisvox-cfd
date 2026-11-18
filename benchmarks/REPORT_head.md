# OmnisVox CFD 1.0.0 — benchmark report

Everything below was run with `benchmarks/run_all.py` on **2 CPU threads, double
precision, NumPy/SciPy backend**. There was no GPU on the build machine, so no
GPU numbers appear here; run `python benchmarks/run_all.py --device gpu` to
produce them. Regenerate this file with `python benchmarks/make_report.py`.

## What the numbers say

**The solver is correct.** It reproduces the two exact solutions to round-off,
is 4th-order in time, conserves energy when it should, and its two independent
implementations (2-D and 3-D) agree. Section 1.

**Standard benchmarks.**

* *Taylor–Green, Re = 1600.* On 128³ the dissipation peak is 0.01376 at
  t = 8.93. The 512³ pseudo-spectral reference (van Rees et al. 2011) peaks at
  about 0.013 near t = 9 — I am quoting that value from memory and could not
  download the reference curve here, so overlay it before publishing. 128³ is
  known to be too coarse for this Reynolds number (k_max η ≈ 0.5), a small
  overshoot is the expected symptom, and the solver says so itself: it flags
  7.8 of the 12 time units as under-resolved. A resolved run needs 256³, which
  needs the GPU.
* *Cylinder, Re = 100.* Strouhal number 0.168 / 0.172 / 0.173 at 8 / 16 / 32
  cells per diameter, against Williamson's 0.164 for an isolated cylinder. The
  sequence is converging, to about 0.174: **6 % high**. Doubling the domain
  height takes off 1 %. So the offset is a property of the set-up (a periodic
  row of penalised cylinders with a fringe-zone outflow), not a resolution
  error that a finer grid will remove. Drag falls toward the published 1.34
  as the grid is refined (1.67, 1.58, 1.47) but is still 10 % high at 32 cells
  per diameter. Treat body flows as a 5–10 % tool. Section 6.

**Adaptive resolution with rollback works as a mechanism.** It promotes,
rewinds, demotes without chatter, picks a finer grid when the initial
condition needs one, recovers from injected faults, and its replay is
bit-for-bit reproducible. The tanh limiter fired **zero** times in every run in
this report.

**The savings are modest, not dramatic.** Against the finest grid run
throughout, on cell-updates:

| flow | adaptive cost | saving | result vs finest grid |
|---|---|---|---|
| Taylor–Green Re = 1600, 32³/64³/128³ | 0.84 | 1.2× | dissipation history within 1.2 %, final energy within 1.0 % |
| Double shear layer, 128²/256²/512² | 0.71 | 1.4× | vorticity within 3.5 % |
| Decaying 2-D turbulence, 64²/128²/256² | 0.81 | 1.2× | enstrophy within 0.4 % |
| Cylinder Re = 100, 256×128/512×256/1024×512 | 0.14 (estimate) | about 7× (estimate) | settles on 512×256; same Strouhal number as the fixed 512×256 run |

That is the honest size of the effect for flows that become and stay
turbulent: once the finest grid is needed, it stays needed, and only the calm
start (or a long decay) is saved. The earlier "about 28×" figure does not
survive a comparison at equal accuracy. Large savings appear only when the
demanding part of the flow is brief — the cylinder's impulsive start is the
one example here, and the 2-D turbulence case would keep gaining if run far
longer than t = 30. Note the cylinder saving is relative to a 1024×512 grid
that the flow, by the controller's own criterion, does not need; and that
grid gives a slightly different Strouhal number (0.173 against 0.172).

Wall-clock, back to back on an idle machine (section 7): 1.25× on
Taylor–Green Re = 400 with 64³ finest, 1.27× on the 2-D turbulence case.

**Rollback itself changes accuracy very little.** In the shear-layer test,
rewinding before refining gave 3.45 % error against 3.57 % for refining in
place; freezing the checkpoint much earlier ("deep rollback") gave 2.71 % at
19 % more cost. With the grid checked every 5 steps there is simply not much
contaminated history to throw away. Rollback earns its place as fault
recovery (a non-finite state or an energy rise is undone rather than
committed), not as an accuracy feature.

**An adaptive run is only as accurate as its threshold.** In the shear-layer
test the plain 256² grid already gets 4.3 % at one-eighth of the cost; the
adaptive run spends 5.6× more to reach 3.5 %, and to get under 1 % this flow
needs 512² from the first step, where there is nothing left to save.

**The Smagorinsky fallback hurts on this benchmark.** Plain Smagorinsky moves
the Taylor–Green dissipation peak from t ≈ 9 to t ≈ 6.5 (a known weakness in
transitional flow). The unfiltered 64³ run is closer to the truth. It is kept
as an option, off by default.

**Not run here:** anything on a GPU, 256³, the Re = 400 adaptive Taylor–Green
case (`--only tgv400_arr`), and wall-bounded or 3-D body flows (the solver
does not do those).
