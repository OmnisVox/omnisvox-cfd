# How OmnisVox CFD works

This is the reference for the engine in `omnisvox_cfd/`. It is the same engine
in the desktop app and the server.

## 1. What is solved

Incompressible Navier–Stokes on periodic boxes, pseudo-spectral in space.

**3-D** (`solver3d.py`), velocity form with the rotational nonlinear term:

    du/dt = P[ u × ω + ∇·(2 ν_t S) + f ] + ν ∇²u,     ∇·u = 0

`P = I − k kᵀ/k²` is applied every Runge–Kutta stage, so the field is
divergence-free to round-off. `ν_t` is the optional Smagorinsky eddy viscosity,
`ν_t = (C_s Δ)² |S|`, with `C_s = 0.17` and `Δ = 1.5 dx` (π over the cut-off).

**2-D** (`solver2d.py`), vorticity–streamfunction form with a prescribed mean
flow `U0`:

    dω/dt + ∇·(u ω) = ν ∇²ω,   ∇²ψ = −ω,   u = U0 + ∂ψ/∂y,  v = V0 − ∂ψ/∂x

Both use:

* **2/3-rule dealiasing** on the nonlinear term (box truncation, `|n_i| < N_i/3`).
* **Integrating-factor RK4.** Viscosity is integrated exactly through
  `E = exp(−ν k² dt)`, so it never limits the time step. Only the advective CFL
  number does (default 0.5). The step is 4th-order accurate; this is checked by
  the validation suite.
* **Adaptive time step** held constant while the CFL number stays in a band, so
  the integrating factor is not rebuilt every step.

### Solid bodies (2-D)

Bodies are imposed by **Brinkman volume penalisation**, integrated exactly after
each step: `u → u·exp(−χ dt/η)` with `η = 0.25 dx/U`. The mask `χ` is a smooth
tanh profile so the spectral representation stays clean. A smoothed penalised
body acts slightly larger than its mask, so the mask is drawn one cell inside
the true surface; the remaining error in effective diameter is a fraction of a
cell and shrinks with resolution. A **fringe zone** over the last 6 length
units of the domain relaxes the flow back to the free stream so the wake does
not re-enter through the periodic boundary.

Consequences you should know about:

* The flow is a periodic array of bodies. With the default 32 D × 16 D domain
  the blockage is 6 %, which raises the shedding frequency and drag by a few
  percent relative to an isolated body. Use a taller grid (for example
  512 × 512) to reduce it.
* `cd`/`cl` in the history are the momentum removed by the penalisation. The
  lift *frequency* is reliable (it is what the Strouhal validation uses); treat
  the force *magnitudes* as indicative, not validated.
* Body surfaces are resolved to about one cell. Thin bodies (NACA sections at
  low grid resolution) are demonstrations, not design-grade aerodynamics.

## 2. Adaptive Rollback-Refinement (ARR)

Code: `arr.py`. The global periodic solution runs on one of up to five
**spectral tiers** (for example 32³ / 64³ / 128³). In 2-D, and as the safety
fallback in 3-D, ARR adapts the whole domain in time exactly as in v1.0.

**3-D spatial ARR (v1.1).** On CPU, the controller can now keep the global
spectral tier coarse while a temporary finite-difference patch carries the
extra resolution only around a spatially concentrated high-k event. The global
FFT solver remains the authoritative long-range solution and pressure
projection; the patch is a short-lived local correction, not a replacement
solver. If localisation is not clearly cheaper or any merge check fails, ARR
uses the original whole-domain promotion automatically.

### The observable: saturation pressure

Every `obs_every` steps (default 5) the controller computes, from the spectrum
alone (no extra FFTs for periodic flows):

    P = (enstrophy in modes above θ·k_cut) / (total enstrophy),   θ = 0.7

A resolved flow has almost nothing just under the cut-off. When the cascade
reaches the grid, `P` climbs. For body flows the vorticity is first multiplied
by a window that excludes the body's own surface layer and the fringe.

Measured against fine-grid references (see `benchmarks/REPORT.md`):

| quantity | error relative to P |
|---|---|
| pointwise vorticity | about 3–7 × P |
| pointwise velocity (3-D) | about 0.7 × P |
| integrated statistics (energy, enstrophy) | about 0.1 × P |

These ratios hold in the marginal range (P around 10⁻³ to 10⁻²). Anything a
coarse tier cannot represent is lost for good, including fine detail in the
initial condition, so the accuracy of an adaptive run is set by the threshold,
not by the finest tier it visits.

So `press_up` is an accuracy dial. Defaults ("Standard"): `0.005` for 2-D
cases, `0.03` for 3-D cases. "Draft" is 4× looser, "Strict" is 0.3×.

### The three layers

1. **Early warning** — `P > warn` (= `0.5·press_up`). The checkpoint stops
   being refreshed, so the last checkpoint is a state that was still clean.
   If `P` hovers in the warning band for `max_spec_obs` observations the state
   is accepted and checkpointing resumes, which bounds the cost of a rollback.
2. **Rollback and promote** — `P > press_up`, or a hard fault (non-finite
   values, collapsing time step, kinetic energy *growing* in an unforced
   flow). Everything since the checkpoint is discarded, the checkpoint is
   prolonged onto the next finer tier, and the interval is recomputed there.
   Discarded steps are counted and reported as "redone".
3. **Last resort, finest tier only** — a hard fault rewinds and retries with
   half the CFL number (up to `max_retries`); if `P > press_up` there the
   stretch is **flagged** as under-resolved (`saturated` in the history) and,
   depending on `on_saturation`, the run continues, switches on the
   Smagorinsky model, or stops. (Plain Smagorinsky is too dissipative for
   transitional flows: in the Taylor–Green benchmark it made the result worse
   than leaving the grid under-resolved. "Continue and flag" is the default
   for that reason.) If nothing works the run ends as `failed` with
   a message. It never silently returns garbage.

**Demotion.** On a fine tier the controller also measures `Q`, the share of
enstrophy above `θ` times the *next coarser* tier's cut-off — what that grid
would be unable to carry. When `Q < press_dn` (= `0.2·press_up`) for `hold`
consecutive observations **and `Q` is lower at the end of that stretch than at
its start**, the state is restricted by spectral truncation.
`press_dn < warn < press_up` is the hysteresis that prevents chatter. The
falling-`Q` condition (1.1.2) stops ARR stepping down while a cascade is still
building: before a turbulent flow's first cascade, `Q` is near zero and flat
or rising, which used to look "calm".

**Flows that start turbulent.** Cases with `start_fine` (2-D and 3-D decaying
turbulence) begin on the finest tier unless `arr.start_tier` is given, and ARR
coarsens them as they decay. Reason, measured on 2-D decaying turbulence
(Re 1000, 128²/256²/512², reference 1024²): spending only the first 0.3% of
the run on 128² left a dissipation-history error 12× larger than starting on
512², because early under-resolution shifts the large vortices and therefore
the timing of later mergers. The saturation trigger only sees the present
spectral tail, so it cannot catch damage that is already done. Starting fine
and coarsening reached uniform-512² accuracy (error 1e-4) with about 0.56× of
the uniform-512² step cost; starting coarse was 4× *less* accurate than plain
256².

**Initial condition.** If the initial field is already above `press_up` on the
starting tier, the run starts on a finer tier and the initial condition is
re-sampled from its definition there (not prolonged).


### Spatial 3-D rollback path (experimental, off by default)

The global trigger is still saturation pressure `P`; the spatial map changes
**where the replay is spent**, not the accuracy criterion. With `local_3d`
enabled, when `P` crosses `press_up` on a 3-D CPU run:

1. The controller computes a high-pass vorticity field (`rbox > theta`) and
   transforms it to physical space. On each axis it takes the shortest
   periodic window holding `local_target_fraction^(1/3)` of that axis'
   activity profile, then pads it. (1.1.0 used the bounding box of the top
   cells, which scattered low-level background inflated to the whole domain.)
2. If that region is smaller than `local_max_volume`, ARR restores the last
   clean checkpoint and creates a `local_refine`× patch. Otherwise it falls
   back to the next global spectral tier.
3. **Spectral patch (`local_scheme="spectral"`, default).** Inside the patch
   `u = T + delta`. `T` is the coarse global field evaluated *exactly* on the
   aligned fine points by a separable partial inverse DFT (with its gradient
   and Laplacian), so `T` is divergence-free to round-off. `delta` is the
   local detail. It is kept compactly supported and exactly solenoidal by
   tapering its vector potential, `delta = curl(w A)`, so it is
   periodic-friendly and all patch derivatives and the patch pressure
   projection are FFTs, consistent with the global solver. `delta` is
   advanced with RK4, Leray-projecting every stage tendency, while `T` is
   interpolated in time across the coarse step.
4. **Merge.** The coarse band of `delta` is handed to the global field. Exactly
   what the global field absorbs (after its own dealiasing and projection) is
   interpolated back and removed from `delta`, so a merge only moves ownership
   between levels and the total field inside the patch is unchanged. The
   global Fourier projector then holds incompressibility to round-off.
5. The patch remains alive while the event persists, retiring after `hold`
   calm observations past `local_min_age`. The patch saturation measure is the
   share of total patch enstrophy in the patch's own top band, i.e. it rises
   only when the patch itself runs out of resolution.

The legacy 1.1.0 finite-difference patch (`local_scheme="fd"`) is kept for
comparison. Its collocated second-order projection enforces a discrete
constraint the true field does not satisfy (the 48³ reference measures a
"divergence" of 3.0 under that stencil), so every substep distorts resolved
fine scales; on a localized benchmark it ends up *less* accurate than the
coarse grid alone.

This implementation supports **one active patch**. Homogeneous turbulence and
symmetric multi-hotspot fields such as Taylor-Green legitimately produce a
region as large as the box and keep global refinement. See
`benchmarks/LOCAL_ARR_SMOKE.md` for measured accuracy and cost: on every case
measured so far, whole-domain promotion from the same rollback point is
still faster and more accurate, which is why `local_3d` is off by default.

### Grid transfer

`resample_hat` in `spectral.py`. Prolongation is zero-padding in Fourier space,
restriction is truncation. Both are exact for every mode the two grids share,
drop the ambiguous Nyquist modes, and rescale for the unnormalised FFT.
Prolongation conserves energy exactly; restriction loses exactly the energy in
the discarded modes, which is what `Q` bounds.

### The tanh limiter

`guard_factor` (default 4) arms a speed limiter on the finest tier only:
exactly the identity below `0.75·cap`, a smooth tanh roll-off to
`cap = guard_factor × initial max speed` above. It is applied to the state
between steps, never inside the Runge–Kutta stages, and **every firing is
counted** (`guard_hits`). In every benchmark in the report it fired zero times.
If it ever fires in your runs, that stretch is no longer a Navier–Stokes
solution and you should treat it that way. Set `guard_factor: null` to remove
it entirely.

### Cost accounting

The summary reports, per tier, steps, cell-steps and wall time; the steps
redone after rollbacks; and `cell_step_saving_est`, an *estimate* of how many
cell updates a run on the finest tier throughout would have needed (it assumes
the finest grid's time step scales with its spacing). The benchmark report
contains measured comparisons against real uniform runs.

## 3. Configuration reference

```jsonc
{
  "case": "taylor_green_3d",          // see `omnisvox-cfd cases`
  "params": {"Re": 1600},
  "preset": "32^3 / 64^3 / 128^3",    // or "tiers": [[32,32,32],[64,64,64]]
  "t_end": 12,
  "device": "auto",                   // auto | cpu | gpu
  "precision": "double",              // double | single
  "cfl": 0.5,
  "sgs": "none",                      // 3-D: none | smagorinsky
  "arr": {
    "enabled": true,                  // false = fixed grid at start_tier
    "start_tier": 0, "min_tier": 0, "max_tier": null,
    "press_up": 0.03,                 // default depends on the case
    "warn_ratio": 0.5, "down_ratio": 0.2, "theta": 0.7,
    "hold": 5, "obs_every": 5, "max_spec_obs": 20,
    "rollback": true,                 // false = refine in place (for comparison)
    "on_saturation": "continue",      // continue | sgs | stop
    "guard_factor": 4.0,              // null disables the limiter
    "max_retries": 3,

    // 3-D spatial ARR (experimental, CPU; automatic fallback to global tiers)
    "local_3d": false,               // off by default since 1.1.1
    "local_scheme": "spectral",      // spectral (1.1.1) | fd (1.1.0 legacy)
    "local_refine": 2,               // 2 or 4
    "local_target_fraction": 0.80,   // activity the patch must contain
    "local_padding": 3,              // coarse cells
    "local_blend": 3,                // coarse-cell sponge width
    "local_max_volume": 0.45,        // otherwise refine the whole domain
    "local_min_age": 0.25,
    "local_fail_factor": 10.0,
    "local_merge_energy_tol": 0.02,
    "local_allow_gpu": false
  }
}
```

## 4. GPU

`backend.py` selects NumPy/SciPy (CPU, multi-threaded FFTs) or CuPy (CUDA).
The solvers use only the array module the backend hands them, so the CPU and
GPU paths are the same code. Double precision is the default; single precision
roughly halves memory and is usually faster on consumer GPUs.

Memory guide: counting the arrays alive during a step, a 256³ run needs about
5 GB in double precision and half that in single (an estimate from array
counts, not a measurement). Confirm on your card with
`omnisvox-cfd bench --device gpu --large` before committing to a grid.

**The GPU path was written but not executed in the build environment (no GPU
there).** Before relying on it run:

    pip install cupy-cuda12x          # match your CUDA version
    omnisvox-cfd validate --device gpu

That runs the same checks as on the CPU plus a CPU-vs-GPU agreement test.

## 5. What is and is not validated

Validated (numbers in `benchmarks/REPORT.md`, checks in `omnisvox-cfd validate`):

* exact decay of 2-D Taylor–Green and 3-D ABC flows; 4th-order time accuracy;
  inviscid energy conservation; the energy budget; divergence-free velocity;
* agreement of the independent 2-D and 3-D solvers on a z-invariant flow;
* grid transfer exactness; bit-for-bit rollback and replay; fault recovery;
* Taylor–Green transition at Re = 1600 and Re = 400;
* Strouhal number of a circular cylinder at Re = 100.

Not validated / known limits:

* GPU execution (see above) and the Modal worker (written to Modal's
  documented API, never run).
* Force magnitudes on immersed bodies; non-circular bodies in general.
* Wall-bounded and inflow/outflow problems beyond the fringe approach; complex
  3-D geometry; compressibility. This is a periodic-box solver.
* The adaptive saving depends entirely on the flow. A spatially distributed
  cascade still needs global refinement; v1.1 intentionally falls back rather
  than claiming a false local speedup.
* Spatial 3-D ARR is CPU-only in v1.1 and supports one active patch. Small-grid
  wall-clock performance is dominated by patch interpolation/projection
  overhead even when cell-step work is reduced. GPU patch kernels and multiple
  simultaneous patches remain future optimisation work.
