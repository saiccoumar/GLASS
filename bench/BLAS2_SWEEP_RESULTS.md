# blas2 sweep — warp/block picks for the ladder's blind-spot ops

Source: `bench/bench_blas2.cu`, driven by `bench/tune.py --legs blas2`. Ops the
mega-sweep ladder does not cover: `syrk`, `syr2k`, `ldlt`, `ldltsv`
(= `ldlt` + `ldlt_solve`, the LDLᵀ analogue of the ladder's `posv` row), `inv`
(Gauss-Jordan on the augmented `[A | I]` layout), `trmv`, and `ger`. Square N
over the ladder's N set {4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}, f32 + f64,
one problem per block (BLOCK, TB ∈ {32, 64, 128, 256}) vs one problem per warp
(WARP, WPB ∈ {1..32}).

Two contenders only:

- **No nvidia leg** — none of these ops has a `glass::nvidia::` counterpart.
- `inv`, `trmv`, `ger` are **block-only** (no `glass::warp::` variant exists);
  they are measured and recorded, not picked.

These measurements do **not** regenerate a shipped header table yet — the
`suggested_backend<>` defaults-table extension for these ops is a follow-up.
Until then this file is the record of the measured winner per (op, N, dtype);
the block between the markers below is auto-refreshed by `bench/tune.py`
through the shared `tune_pick` margin rule.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_No quiet-GPU sweep recorded yet. Run `python3 bench/tune.py --sm auto --legs
blas2,rect` on an idle GPU (after `--prebuild`); this block is then auto-filled
with the raw ns/problem numbers and the margin-aware warp-vs-block picks._

<!-- END tune.py -->

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild --legs blas2   # compile (GPU may be busy)
python3 bench/tune.py --sm auto --legs blas2              # timed run — QUIET GPU only
# or run the harness directly:
#   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 \
#        -I.. -I../src bench_blas2.cu -o bench_blas2
#   ./bench_blas2 [nprob=8192] [reps=500] [dtype=f32|f64]
```
