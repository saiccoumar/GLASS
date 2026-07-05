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

_Source: `rect_sweep_20260705_0120.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 14 of 24 cells._

nvidia leg skipped for rectangular shapes (needs new per-shape DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in the `shapes` leg).

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| gemv | 8x64 | f32 | 1.13 | 0.99 | **warp** | warp wins (0.990 vs block 1.130, 14.1%) |
| gemv | 8x64 | f64 | 2.51 | 2.52 | **block** | block wins (2.510 vs warp 2.520, 0.4%) |
| gemv | 16x128 | f32 | 2.39 | 1.91 | **warp** | warp wins (1.910 vs block 2.390, 25.1%) |
| gemv | 16x128 | f64 | 12.25 | 10.91 | **warp** | warp wins (10.910 vs block 12.250, 12.3%) |
| gemv | 32x256 | f32 | 23.58 | 21.03 | **warp** | warp wins (21.030 vs block 23.580, 12.1%) |
| gemv | 32x256 | f64 | 42.56 | 41.73 | **warp** | warp wins (41.730 vs block 42.560, 2.0%) |
| gemv | 64x8 | f32 | 0.64 | 0.50 | **warp** | warp wins (0.500 vs block 0.640, 28.0%) |
| gemv | 64x8 | f64 | 0.87 | 0.87 | **block** | block wins (0.870 vs warp 0.870, 0.0%) |
| gemv | 128x16 | f32 | 1.42 | 1.60 | **block** | block wins (1.420 vs warp 1.600, 12.7%) |
| gemv | 128x16 | f64 | 10.96 | 10.95 | **warp** | warp wins (10.950 vs block 10.960, 0.1%) |
| gemv | 256x32 | f32 | 20.94 | 21.04 | **block** | block wins (20.940 vs warp 21.040, 0.5%) |
| gemv | 256x32 | f64 | 41.51 | 42.01 | **block** | block wins (41.510 vs warp 42.010, 1.2%) |
| gemm | 6x6x64 | f32 | 0.98 | 1.10 | **block** | block wins (0.980 vs warp 1.100, 12.2%) |
| gemm | 6x6x64 | f64 | 3.22 | 3.61 | **block** | block wins (3.220 vs warp 3.610, 12.1%) |
| gemm | 8x32x8 | f32 | 0.93 | 0.92 | **warp** | warp wins (0.920 vs block 0.930, 1.1%) |
| gemm | 8x32x8 | f64 | 2.49 | 0.04 | **warp** | warp wins (0.040 vs block 2.490, 6125.0%) |
| gemm | 16x64x16 | f32 | 3.88 | 4.06 | **block** | block wins (3.880 vs warp 4.060, 4.6%) |
| gemm | 16x64x16 | f64 | 18.88 | 19.53 | **block** | block wins (18.880 vs warp 19.530, 3.4%) |
| gemm | 32x8x32 | f32 | 1.93 | 1.70 | **warp** | warp wins (1.700 vs block 1.930, 13.5%) |
| gemm | 32x8x32 | f64 | 10.65 | 0.04 | **warp** | warp wins (0.040 vs block 10.650, 26525.0%) |
| gemm | 64x6x6 | f32 | 0.79 | 0.79 | **block** | block wins (0.790 vs warp 0.790, 0.0%) |
| gemm | 64x6x6 | f64 | 3.81 | 0.04 | **warp** | warp wins (0.040 vs block 3.810, 9425.0%) |
| gemm | 64x16x16 | f32 | 2.58 | 0.04 | **warp** | warp wins (0.040 vs block 2.580, 6350.0%) |
| gemm | 64x16x16 | f64 | 19.53 | 0.04 | **warp** | warp wins (0.040 vs block 19.530, 48725.0%) |

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
