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

_Source: `blas2_sweep_20260705_0120.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 27 of 154 cells._

inv/trmv/ger are BLOCK-ONLY (no `glass::warp::` variant); none of these ops has a `glass::nvidia::` counterpart.

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| syrk | N=4 | f32 | 0.58 | 0.22 | **warp** | warp wins (0.220 vs block 0.580, 163.6%) |
| syrk | N=4 | f64 | 0.61 | 0.35 | **warp** | warp wins (0.350 vs block 0.610, 74.3%) |
| syrk | N=6 | f32 | 0.61 | 0.32 | **warp** | warp wins (0.320 vs block 0.610, 90.6%) |
| syrk | N=6 | f64 | 0.75 | 0.71 | **warp** | warp wins (0.710 vs block 0.750, 5.6%) |
| syrk | N=8 | f32 | 0.61 | 0.40 | **warp** | warp wins (0.400 vs block 0.610, 52.5%) |
| syrk | N=8 | f64 | 0.89 | 0.89 | **block** | block wins (0.890 vs warp 0.890, 0.0%) |
| syrk | N=12 | f32 | 0.83 | 0.92 | **block** | block wins (0.830 vs warp 0.920, 10.8%) |
| syrk | N=12 | f64 | 2.70 | 2.84 | **block** | block wins (2.700 vs warp 2.840, 5.2%) |
| syrk | N=16 | f32 | 1.49 | 1.64 | **block** | block wins (1.490 vs warp 1.640, 10.1%) |
| syrk | N=16 | f64 | 5.31 | 5.69 | **block** | block wins (5.310 vs warp 5.690, 7.2%) |
| syrk | N=24 | f32 | 4.35 | 5.48 | **block** | block wins (4.350 vs warp 5.480, 26.0%) |
| syrk | N=24 | f64 | 16.60 | 18.05 | **block** | block wins (16.600 vs warp 18.050, 8.7%) |
| syrk | N=32 | f32 | 11.32 | 18.70 | **block** | block wins (11.320 vs warp 18.700, 65.2%) |
| syrk | N=32 | f64 | 38.34 | 44.12 | **block** | block wins (38.340 vs warp 44.120, 15.1%) |
| syrk | N=48 | f32 | 39.36 | 74.37 | **block** | block wins (39.360 vs warp 74.370, 88.9%) |
| syrk | N=48 | f64 | 111.44 | 147.37 | **block** | block wins (111.440 vs warp 147.370, 32.2%) |
| syrk | N=64 | f32 | 69.65 | 159.42 | **block** | block wins (69.650 vs warp 159.420, 128.9%) |
| syrk | N=64 | f64 | 222.52 | 553.17 | **block** | block wins (222.520 vs warp 553.170, 148.6%) |
| syrk | N=96 | f32 | 207.01 | 1311.29 | **block** | block wins (207.010 vs warp 1311.290, 533.4%) |
| syrk | N=96 | f64 | 665.27 | 2666.12 | **block** | block wins (665.270 vs warp 2666.120, 300.8%) |
| syrk | N=128 | f32 | 546.09 | 3459.94 | **block** | block wins (546.090 vs warp 3459.940, 533.6%) |
| syrk | N=128 | f64 | 1589.47 | 6571.78 | **block** | block wins (1589.470 vs warp 6571.780, 313.5%) |
| syr2k | N=4 | f32 | 0.58 | 0.24 | **warp** | warp wins (0.240 vs block 0.580, 141.7%) |
| syr2k | N=4 | f64 | 0.70 | 0.69 | **warp** | warp wins (0.690 vs block 0.700, 1.4%) |
| syr2k | N=6 | f32 | 0.62 | 0.40 | **warp** | warp wins (0.400 vs block 0.620, 55.0%) |
| syr2k | N=6 | f64 | 1.66 | 1.68 | **block** | block wins (1.660 vs warp 1.680, 1.2%) |
| syr2k | N=8 | f32 | 0.66 | 0.48 | **warp** | warp wins (0.480 vs block 0.660, 37.5%) |
| syr2k | N=8 | f64 | 2.17 | 2.19 | **block** | block wins (2.170 vs warp 2.190, 0.9%) |
| syr2k | N=12 | f32 | 1.37 | 1.47 | **block** | block wins (1.370 vs warp 1.470, 7.3%) |
| syr2k | N=12 | f64 | 7.48 | 7.72 | **block** | block wins (7.480 vs warp 7.720, 3.2%) |
| syr2k | N=16 | f32 | 2.63 | 2.97 | **block** | block wins (2.630 vs warp 2.970, 12.9%) |
| syr2k | N=16 | f64 | 15.47 | 16.00 | **block** | block wins (15.470 vs warp 16.000, 3.4%) |
| syr2k | N=24 | f32 | 8.09 | 14.10 | **block** | block wins (8.090 vs warp 14.100, 74.3%) |
| syr2k | N=24 | f64 | 51.55 | 53.13 | **block** | block wins (51.550 vs warp 53.130, 3.1%) |
| syr2k | N=32 | f32 | 20.77 | 36.51 | **block** | block wins (20.770 vs warp 36.510, 75.8%) |
| syr2k | N=32 | f64 | 120.24 | 129.61 | **block** | block wins (120.240 vs warp 129.610, 7.8%) |
| syr2k | N=48 | f32 | 66.01 | 141.22 | **block** | block wins (66.010 vs warp 141.220, 113.9%) |
| syr2k | N=48 | f64 | 358.98 | 571.82 | **block** | block wins (358.980 vs warp 571.820, 59.3%) |
| syr2k | N=64 | f32 | 125.93 | 689.38 | **block** | block wins (125.930 vs warp 689.380, 447.4%) |
| syr2k | N=64 | f64 | 714.46 | 1532.15 | **block** | block wins (714.460 vs warp 1532.150, 114.4%) |
| syr2k | N=96 | f32 | 379.24 | 3072.43 | **block** | block wins (379.240 vs warp 3072.430, 710.2%) |
| syr2k | N=96 | f64 | 2133.62 | 5425.87 | **block** | block wins (2133.620 vs warp 5425.870, 154.3%) |
| syr2k | N=128 | f32 | 1225.50 | 6935.39 | **block** | block wins (1225.500 vs warp 6935.390, 465.9%) |
| syr2k | N=128 | f64 | 5327.81 | 12683.27 | **block** | block wins (5327.810 vs warp 12683.270, 138.1%) |
| ldlt | N=4 | f32 | 0.64 | 0.39 | **warp** | warp wins (0.390 vs block 0.640, 64.1%) |
| ldlt | N=4 | f64 | 1.94 | 3.35 | **block** | block wins (1.940 vs warp 3.350, 72.7%) |
| ldlt | N=6 | f32 | 1.03 | 0.81 | **warp** | warp wins (0.810 vs block 1.030, 27.2%) |
| ldlt | N=6 | f64 | 4.50 | 6.44 | **block** | block wins (4.500 vs warp 6.440, 43.1%) |
| ldlt | N=8 | f32 | 1.58 | 1.27 | **warp** | warp wins (1.270 vs block 1.580, 24.4%) |
| ldlt | N=8 | f64 | 7.13 | 10.04 | **block** | block wins (7.130 vs warp 10.040, 40.8%) |
| ldlt | N=12 | f32 | 2.98 | 2.41 | **warp** | warp wins (2.410 vs block 2.980, 23.7%) |
| ldlt | N=12 | f64 | 14.24 | 19.03 | **block** | block wins (14.240 vs warp 19.030, 33.6%) |
| ldlt | N=16 | f32 | 4.63 | 3.84 | **warp** | warp wins (3.840 vs block 4.630, 20.6%) |
| ldlt | N=16 | f64 | 24.21 | 30.35 | **block** | block wins (24.210 vs warp 30.350, 25.4%) |
| ldlt | N=24 | f32 | 9.04 | 7.87 | **warp** | warp wins (7.870 vs block 9.040, 14.9%) |
| ldlt | N=24 | f64 | 52.18 | 60.64 | **block** | block wins (52.180 vs warp 60.640, 16.2%) |
| ldlt | N=32 | f32 | 16.19 | 14.12 | **warp** | warp wins (14.120 vs block 16.190, 14.7%) |
| ldlt | N=32 | f64 | 90.62 | 100.94 | **block** | block wins (90.620 vs warp 100.940, 11.4%) |
| ldlt | N=48 | f32 | 43.95 | 43.56 | **warp** | warp wins (43.560 vs block 43.950, 0.9%) |
| ldlt | N=48 | f64 | 215.75 | 235.44 | **block** | block wins (215.750 vs warp 235.440, 9.1%) |
| ldlt | N=64 | f32 | 93.82 | 93.49 | **warp** | warp wins (93.490 vs block 93.820, 0.4%) |
| ldlt | N=64 | f64 | 404.01 | 432.90 | **block** | block wins (404.010 vs warp 432.900, 7.2%) |
| ldlt | N=96 | f32 | 284.12 | 302.04 | **block** | block wins (284.120 vs warp 302.040, 6.3%) |
| ldlt | N=96 | f64 | 1021.57 | 1074.41 | **block** | block wins (1021.570 vs warp 1074.410, 5.2%) |
| ldlt | N=128 | f32 | 664.31 | 804.87 | **block** | block wins (664.310 vs warp 804.870, 21.2%) |
| ldlt | N=128 | f64 | 2038.61 | 2462.80 | **block** | block wins (2038.610 vs warp 2462.800, 20.8%) |
| ldltsv | N=4 | f32 | 0.74 | 0.58 | **warp** | warp wins (0.580 vs block 0.740, 27.6%) |
| ldltsv | N=4 | f64 | 2.98 | 4.39 | **block** | block wins (2.980 vs warp 4.390, 47.3%) |
| ldltsv | N=6 | f32 | 1.42 | 1.00 | **warp** | warp wins (1.000 vs block 1.420, 42.0%) |
| ldltsv | N=6 | f64 | 5.75 | 7.61 | **block** | block wins (5.750 vs warp 7.610, 32.3%) |
| ldltsv | N=8 | f32 | 2.12 | 1.52 | **warp** | warp wins (1.520 vs block 2.120, 39.5%) |
| ldltsv | N=8 | f64 | 8.56 | 11.35 | **block** | block wins (8.560 vs warp 11.350, 32.6%) |
| ldltsv | N=12 | f32 | 3.90 | 2.81 | **warp** | warp wins (2.810 vs block 3.900, 38.8%) |
| ldltsv | N=12 | f64 | 16.01 | 20.65 | **block** | block wins (16.010 vs warp 20.650, 29.0%) |
| ldltsv | N=16 | f32 | 5.75 | 4.41 | **warp** | warp wins (4.410 vs block 5.750, 30.4%) |
| ldltsv | N=16 | f64 | 26.50 | 32.43 | **block** | block wins (26.500 vs warp 32.430, 22.4%) |
| ldltsv | N=24 | f32 | 10.78 | 8.76 | **warp** | warp wins (8.760 vs block 10.780, 23.1%) |
| ldltsv | N=24 | f64 | 54.90 | 63.42 | **block** | block wins (54.900 vs warp 63.420, 15.5%) |
| ldltsv | N=32 | f32 | 19.66 | 15.90 | **warp** | warp wins (15.900 vs block 19.660, 23.6%) |
| ldltsv | N=32 | f64 | 94.17 | 104.39 | **block** | block wins (94.170 vs warp 104.390, 10.9%) |
| ldltsv | N=48 | f32 | 51.94 | 49.60 | **warp** | warp wins (49.600 vs block 51.940, 4.7%) |
| ldltsv | N=48 | f64 | 223.34 | 242.67 | **block** | block wins (223.340 vs warp 242.670, 8.7%) |
| ldltsv | N=64 | f32 | 105.42 | 103.84 | **warp** | warp wins (103.840 vs block 105.420, 1.5%) |
| ldltsv | N=64 | f64 | 416.68 | 444.50 | **block** | block wins (416.680 vs warp 444.500, 6.7%) |
| ldltsv | N=96 | f32 | 306.35 | 330.60 | **block** | block wins (306.350 vs warp 330.600, 7.9%) |
| ldltsv | N=96 | f64 | 1044.30 | 1092.17 | **block** | block wins (1044.300 vs warp 1092.170, 4.6%) |
| ldltsv | N=128 | f32 | 733.09 | 860.36 | **block** | block wins (733.090 vs warp 860.360, 17.4%) |
| ldltsv | N=128 | f64 | 2050.01 | 2534.16 | **block** | block wins (2050.010 vs warp 2534.160, 23.6%) |
| inv | N=4 | f32 | 0.70 | — | **block** | block only impl measured (0.700) |
| inv | N=4 | f64 | 1.44 | — | **block** | block only impl measured (1.440) |
| inv | N=6 | f32 | 1.12 | — | **block** | block only impl measured (1.120) |
| inv | N=6 | f64 | 2.93 | — | **block** | block only impl measured (2.930) |
| inv | N=8 | f32 | 1.82 | — | **block** | block only impl measured (1.820) |
| inv | N=8 | f64 | 4.99 | — | **block** | block only impl measured (4.990) |
| inv | N=12 | f32 | 4.53 | — | **block** | block only impl measured (4.530) |
| inv | N=12 | f64 | 10.55 | — | **block** | block only impl measured (10.550) |
| inv | N=16 | f32 | 9.31 | — | **block** | block only impl measured (9.310) |
| inv | N=16 | f64 | 22.28 | — | **block** | block only impl measured (22.280) |
| inv | N=24 | f32 | 26.42 | — | **block** | block only impl measured (26.420) |
| inv | N=24 | f64 | 63.79 | — | **block** | block only impl measured (63.790) |
| inv | N=32 | f32 | 52.08 | — | **block** | block only impl measured (52.080) |
| inv | N=32 | f64 | 139.67 | — | **block** | block only impl measured (139.670) |
| inv | N=48 | f32 | 171.79 | — | **block** | block only impl measured (171.790) |
| inv | N=48 | f64 | 453.49 | — | **block** | block only impl measured (453.490) |
| inv | N=64 | f32 | 357.68 | — | **block** | block only impl measured (357.680) |
| inv | N=64 | f64 | 1024.82 | — | **block** | block only impl measured (1024.820) |
| inv | N=96 | f32 | 1455.68 | — | **block** | block only impl measured (1455.680) |
| inv | N=96 | f64 | 3408.61 | — | **block** | block only impl measured (3408.610) |
| inv | N=128 | f32 | 3474.43 | — | **block** | block only impl measured (3474.430) |
| inv | N=128 | f64 | 18707.47 | — | **block** | block only impl measured (18707.470) |
| trmv | N=4 | f32 | 0.59 | — | **block** | block only impl measured (0.590) |
| trmv | N=4 | f64 | 0.63 | — | **block** | block only impl measured (0.630) |
| trmv | N=6 | f32 | 0.60 | — | **block** | block only impl measured (0.600) |
| trmv | N=6 | f64 | 0.68 | — | **block** | block only impl measured (0.680) |
| trmv | N=8 | f32 | 0.62 | — | **block** | block only impl measured (0.620) |
| trmv | N=8 | f64 | 0.83 | — | **block** | block only impl measured (0.830) |
| trmv | N=12 | f32 | 0.64 | — | **block** | block only impl measured (0.640) |
| trmv | N=12 | f64 | 0.92 | — | **block** | block only impl measured (0.920) |
| trmv | N=16 | f32 | 0.75 | — | **block** | block only impl measured (0.750) |
| trmv | N=16 | f64 | 1.55 | — | **block** | block only impl measured (1.550) |
| trmv | N=24 | f32 | 0.83 | — | **block** | block only impl measured (0.830) |
| trmv | N=24 | f64 | 1.60 | — | **block** | block only impl measured (1.600) |
| trmv | N=32 | f32 | 1.14 | — | **block** | block only impl measured (1.140) |
| trmv | N=32 | f64 | 2.24 | — | **block** | block only impl measured (2.240) |
| trmv | N=48 | f32 | 1.92 | — | **block** | block only impl measured (1.920) |
| trmv | N=48 | f64 | 5.39 | — | **block** | block only impl measured (5.390) |
| trmv | N=64 | f32 | 3.52 | — | **block** | block only impl measured (3.520) |
| trmv | N=64 | f64 | 12.36 | — | **block** | block only impl measured (12.360) |
| trmv | N=96 | f32 | 15.10 | — | **block** | block only impl measured (15.100) |
| trmv | N=96 | f64 | 26.27 | — | **block** | block only impl measured (26.270) |
| trmv | N=128 | f32 | 24.86 | — | **block** | block only impl measured (24.860) |
| trmv | N=128 | f64 | 44.30 | — | **block** | block only impl measured (44.300) |
| ger | N=4 | f32 | 0.59 | — | **block** | block only impl measured (0.590) |
| ger | N=4 | f64 | 0.62 | — | **block** | block only impl measured (0.620) |
| ger | N=6 | f32 | 0.63 | — | **block** | block only impl measured (0.630) |
| ger | N=6 | f64 | 0.67 | — | **block** | block only impl measured (0.670) |
| ger | N=8 | f32 | 0.83 | — | **block** | block only impl measured (0.830) |
| ger | N=8 | f64 | 0.88 | — | **block** | block only impl measured (0.880) |
| ger | N=12 | f32 | 1.07 | — | **block** | block only impl measured (1.070) |
| ger | N=12 | f64 | 1.14 | — | **block** | block only impl measured (1.140) |
| ger | N=16 | f32 | 1.48 | — | **block** | block only impl measured (1.480) |
| ger | N=16 | f64 | 1.60 | — | **block** | block only impl measured (1.600) |
| ger | N=24 | f32 | 1.96 | — | **block** | block only impl measured (1.960) |
| ger | N=24 | f64 | 2.30 | — | **block** | block only impl measured (2.300) |
| ger | N=32 | f32 | 2.75 | — | **block** | block only impl measured (2.750) |
| ger | N=32 | f64 | 3.47 | — | **block** | block only impl measured (3.470) |
| ger | N=48 | f32 | 4.58 | — | **block** | block only impl measured (4.580) |
| ger | N=48 | f64 | 26.76 | — | **block** | block only impl measured (26.760) |
| ger | N=64 | f32 | 23.66 | — | **block** | block only impl measured (23.660) |
| ger | N=64 | f64 | 45.59 | — | **block** | block only impl measured (45.590) |
| ger | N=96 | f32 | 54.93 | — | **block** | block only impl measured (54.930) |
| ger | N=96 | f64 | 101.44 | — | **block** | block only impl measured (101.440) |
| ger | N=128 | f32 | 91.06 | — | **block** | block only impl measured (91.060) |
| ger | N=128 | f64 | 177.47 | — | **block** | block only impl measured (177.470) |

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
