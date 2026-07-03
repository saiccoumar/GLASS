# rect sweep — warp/block picks for rectangular gemv/gemm

Source: `bench/bench_rect.cu`, driven by `bench/tune.py --legs rect`. The
mega-sweep ladder is square-only, but consumers' Jacobians are rectangular;
this leg measures:

- **gemv** `(M, N)`: tall {64×8, 128×16, 256×32} + wide {8×64, 16×128, 32×256}
  — `y(M) = A(M×N)·x(N)`.
- **gemm** `(M, K, N)`: {(32,8,32), (8,32,8), (64,16,16), (16,64,16), (6,6,64),
  (64,6,6)} — `C(M×N) = A(M×K)·B(K×N)` (glass template order is
  `gemm<T,M,N,K>`).

f32 + f64, one problem per block (BLOCK, TB ∈ {32, 64, 128, 256}) vs one
problem per warp (WARP, WPB ∈ {1..32}). **The nvidia leg is skipped** for
rectangular shapes: forcing cuBLASDx here would need new per-(M,N,K)/(M,N)
`DEFINE_NVIDIA_*` descriptor instantiations, and per-shape cuBLASDx-vs-SIMT
decisions already live in the `shapes` leg (`bench/autotune.py` →
`src/nvidia/tuning_table.cuh`) — rectangular vendor coverage belongs there.

These measurements do **not** regenerate a shipped header table yet (the
square-N `ideal_sm120` ladder stays authoritative for dispatch); the block
between the markers below is auto-refreshed by `bench/tune.py` through the
shared `tune_pick` margin rule.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_No quiet-GPU sweep recorded yet. Run `python3 bench/tune.py --sm auto --legs
blas2,rect` on an idle GPU (after `--prebuild`); this block is then auto-filled
with the raw ns/problem numbers and the margin-aware warp-vs-block picks._

<!-- END tune.py -->

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild --legs rect    # compile (GPU may be busy)
python3 bench/tune.py --sm auto --legs rect               # timed run — QUIET GPU only
# or run the harness directly:
#   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 \
#        -I.. -I../src bench_rect.cu -o bench_rect
#   ./bench_rect [nprob=8192] [reps=500] [dtype=f32|f64]
```
