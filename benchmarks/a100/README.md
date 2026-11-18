# A100 benchmark

Produced with `omnisvox_a100_benchmark.ipynb` on Google Colab, NVIDIA A100-SXM4 40 GB, 17 November 2026.

| folder | engine | grids (3-D / 2-D) | references |
|---|---|---|---|
| `full/` | 1.1.2 | 64–256³ / 256–2048² | 512³ (single precision, 40 GB card), 2048² / 4096² |
| `quick/` | 1.1.1 | 16–128³ / 128–1024² | 128–256³, 1024² / 2048² |

The quick run used engine 1.1.1. Its 2-D turbulence **loss** led to the 1.1.2 fix (no stepping down while a cascade builds, and turbulent flows start on the finest grid); `full/` is the run after the fix.

Each folder holds:

- `SUMMARY.txt`: one verdict per case and precision, against the cheapest plain grid that is at least as accurate (WIN / TIE / LOSS / TRADE / NOISY).
- `verdict.csv`, `runs_table.csv`: the same as tables, with wall time, dissipation-history error, peak-dissipation error and final-energy error for every run.
- `accuracy_vs_wall.png`: error against wall time per case.
- `runs.jsonl.gz`: every run, with full energy / dissipation / tier histories and controller events.
- `validation.json`, `throughput.json`, `env.json`: GPU validation, ms per step by grid size, environment.

Accuracy is scored on statistics (dissipation history, peak dissipation, final energy), not pointwise fields. Where finer plain grids are not more accurate (chaotic late-time behaviour), the case is marked NOISY and ARR is compared directly with the finest grid.
