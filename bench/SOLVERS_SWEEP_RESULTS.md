# solvers sweep — bdsv-vs-pcg crossover, SPD solve paths, syev

Source: `bench/bench_solvers.cu`, driven by `bench/tune.py --legs solvers`.
Three sections, all **characterization only** — the measured numbers are
recorded here, but **no dispatch table is regenerated** (unlike the
ladder/shapes/reduced legs):

1. **`bdsv` vs `pcg`** — the direct block-Cholesky sweep vs the block-Jacobi
   preconditioned CG on the **identical** block-tridiagonal SPD input
   (`[L|D|R]` strips + padded vectors), (BlockSize, Knots) ∈
   {(2,8), (2,32), (6,8), (6,32), (6,64), (12,16)}. These *are* two impls of
   the same solve, but the right choice is **problem-dependent**: pcg's cost
   scales with its iteration count (i.e. with conditioning and the
   preconditioner's quality), while bdsv is exact in one sweep whose knot
   chain is inherently serial. The table records the measured crossover on the
   harness's test system — a diagonally-dominant one mirroring
   `test/test_pcg.py::make_spd_banded` (`D = M·Mᵀ + SS·I` blocks, ±0.1
   off-diagonals), on which block-Jacobi PCG converges in only a few
   iterations. Read the `pcg iters` column before generalizing: an
   ill-conditioned Riccati/KKT chain can need 10-100× more iterations, moving
   the crossover proportionally toward bdsv.
2. **`gesv` vs `posv` vs `inv`+`gemv`** on one SPD system (single RHS,
   N ∈ {4, 8, 16, 32, 64}): what the pivoted-LU robustness fallback costs
   where Cholesky suffices, and what the invert-then-multiply anti-pattern
   costs vs a factor+solve.
3. **`syev` + `eig_clamp`** (N ∈ {4, 8, 16, 32}) — timing only, no contender
   (the cyclic-Jacobi eigensolver and the decompose-clamp-reconstruct op it
   feeds).

## Methodology: restore-outside-timing

These ops **mutate their input** (bdsv factors the strips in place, gesv/posv
factor A, inv overwrites the augmented `[A|I]`), so the steady-state
rerun-in-place policy of the other harnesses would time garbage after rep 1.
Instead `bench_solvers.cu` keeps NPROB independent problem copies in global
memory and, per rep:

- restores the mutated buffers from pristine device copies (device-to-device
  memcpy; pcg's solution vector is re-zeroed for its warm zero start)
  **outside** the timed window, then
- times exactly one kernel launch spanning all NPROB problems (one block per
  problem) between `cudaEvent`s.

ns/problem = summed event time / (reps × NPROB), min of 3 trials, TB ∈
{32, 256} swept per contender. Reps are modest (default 50) since each rep
already spans NPROB problems. Both bdsv and pcg read per-problem copies of the
same strips from global memory (bdsv's restore source doubles as pcg's `S`),
so the memory-traffic footing is identical.

**Correctness guard (no silent caps):** before any timing, every section-A
shape solves problem 0 with *both* bdsv and pcg and compares against a host
double-precision dense Cholesky solve (max|Δ| < 1e-3; pcg must converge in
0 < iters < 200 — `rel_tol=1e-6`, `abs_tol=1e-12`). Section-B shapes check
gesv/posv/inv+gemv the same way. Any mismatch aborts the run.

The block between the markers below is auto-refreshed by `bench/tune.py`.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `solvers_sweep_20260708_0058.txt` · NPROB=8192 ns/problem (best swept TB, min of 3 trials, restore-outside-timing protocol) · characterization only — no dispatch table is regenerated._

### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input

bdsv is faster in 1 of 12 cells **on this well-conditioned test system** (see the iters column — pcg's cost scales with the iteration count, so the crossover moves with conditioning).

| BlockSize | Knots | dtype | bdsv ns | pcg ns | pcg iters | pcg/bdsv |
|-----------|-------|-------|---------|--------|-----------|----------|
| 2 | 8 | f32 | 6.06 | 2.25 | 3 | 0.37 |
| 2 | 8 | f64 | 26.01 | 8.24 | 3 | 0.32 |
| 2 | 32 | f32 | 24.00 | 3.49 | 3 | 0.15 |
| 2 | 32 | f64 | 106.59 | 11.49 | 3 | 0.11 |
| 6 | 8 | f32 | 18.64 | 6.43 | 3 | 0.34 |
| 6 | 8 | f64 | 93.92 | 30.98 | 3 | 0.33 |
| 6 | 32 | f32 | 85.23 | 30.45 | 3 | 0.36 |
| 6 | 32 | f64 | 388.88 | 130.13 | 3 | 0.33 |
| 6 | 64 | f32 | 177.78 | 82.46 | 3 | 0.46 |
| 6 | 64 | f64 | 785.74 | 254.54 | 3 | 0.32 |
| 12 | 16 | f32 | 108.43 | 195.73 | 2 | 1.81 |
| 12 | 16 | f64 | 459.02 | 230.36 | 2 | 0.50 |

### gesv vs posv vs inv+gemv — same SPD system, single RHS

posv (Cholesky) is the intended SPD path; gesv prices the pivoted-LU robustness fallback, inv+gemv the invert-then-multiply anti-pattern.

| N | dtype | gesv ns | posv ns | inv+gemv ns | gesv/posv | inv/posv |
|---|-------|---------|---------|-------------|-----------|----------|
| 4 | f32 | 1.25 | 1.23 | 1.00 | 1.02 | 0.81 |
| 4 | f64 | 3.74 | 5.50 | 1.99 | 0.68 | 0.36 |
| 8 | f32 | 2.50 | 2.48 | 2.25 | 1.01 | 0.91 |
| 8 | f64 | 8.99 | 12.36 | 5.70 | 0.73 | 0.46 |
| 16 | f32 | 6.50 | 5.52 | 10.57 | 1.18 | 1.91 |
| 16 | f64 | 25.39 | 29.42 | 23.13 | 0.86 | 0.79 |
| 32 | f32 | 27.59 | 15.28 | 57.37 | 1.81 | 3.75 |
| 32 | f64 | 85.46 | 78.25 | 150.82 | 1.09 | 1.93 |
| 64 | f32 | 160.33 | 76.74 | 357.33 | 2.09 | 4.66 |
| 64 | f64 | 417.37 | 264.98 | 1065.11 | 1.58 | 4.02 |

### syev + eig_clamp — timing only (no contender)

| N | dtype | syev ns | eig_clamp ns |
|---|-------|---------|--------------|
| 4 | f32 | 3.89 | 3.99 |
| 4 | f64 | 58.00 | 58.64 |
| 8 | f32 | 25.33 | 25.25 |
| 8 | f64 | 384.99 | 386.82 |
| 16 | f32 | 114.81 | 115.58 |
| 16 | f64 | 1769.52 | 1741.92 |
| 32 | f32 | 875.65 | 1012.52 |
| 32 | f64 | 8808.53 | 9263.20 |

<!-- END tune.py -->

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild --legs solvers  # compile (GPU may be busy)
python3 bench/tune.py --sm auto --legs solvers             # timed run — QUIET GPU only
# or run the harness directly:
#   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 \
#        -I.. -I../src bench_solvers.cu -o bench_solvers
#   ./bench_solvers [nprob=8192] [reps=50] [dtype=f32|f64]
# re-report from an existing capture (no GPU):
python3 bench/tune.py --legs solvers --from-solvers bench/solvers_sweep_<ts>.txt --dry-run
```
