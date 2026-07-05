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

_Source: `solvers_sweep_20260705_0122.txt` · NPROB=8192 ns/problem (best swept TB, min of 3 trials, restore-outside-timing protocol) · characterization only — no dispatch table is regenerated._

### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input

bdsv is faster in 1 of 12 cells **on this well-conditioned test system** (see the iters column — pcg's cost scales with the iteration count, so the crossover moves with conditioning).

| BlockSize | Knots | dtype | bdsv ns | pcg ns | pcg iters | pcg/bdsv |
|-----------|-------|-------|---------|--------|-----------|----------|
| 2 | 8 | f32 | 6.22 | 2.31 | 3 | 0.37 |
| 2 | 8 | f64 | 26.27 | 8.24 | 3 | 0.31 |
| 2 | 32 | f32 | 24.25 | 3.49 | 3 | 0.14 |
| 2 | 32 | f64 | 107.31 | 11.50 | 3 | 0.11 |
| 6 | 8 | f32 | 18.80 | 6.42 | 3 | 0.34 |
| 6 | 8 | f64 | 94.58 | 31.13 | 3 | 0.33 |
| 6 | 32 | f32 | 85.75 | 30.75 | 3 | 0.36 |
| 6 | 32 | f64 | 391.32 | 130.70 | 3 | 0.33 |
| 6 | 64 | f32 | 178.61 | 82.69 | 3 | 0.46 |
| 6 | 64 | f64 | 791.59 | 254.81 | 3 | 0.32 |
| 12 | 16 | f32 | 109.77 | 197.32 | 2 | 1.80 |
| 12 | 16 | f64 | 462.74 | 230.49 | 2 | 0.50 |

### gesv vs posv vs inv+gemv — same SPD system, single RHS

posv (Cholesky) is the intended SPD path; gesv prices the pivoted-LU robustness fallback, inv+gemv the invert-then-multiply anti-pattern.

| N | dtype | gesv ns | posv ns | inv+gemv ns | gesv/posv | inv/posv |
|---|-------|---------|---------|-------------|-----------|----------|
| 4 | f32 | 1.24 | 1.24 | 0.99 | 1.00 | 0.80 |
| 4 | f64 | 3.75 | 5.51 | 2.00 | 0.68 | 0.36 |
| 8 | f32 | 2.49 | 2.48 | 2.24 | 1.00 | 0.90 |
| 8 | f64 | 9.01 | 12.51 | 5.74 | 0.72 | 0.46 |
| 16 | f32 | 6.50 | 5.52 | 10.57 | 1.18 | 1.91 |
| 16 | f64 | 25.53 | 29.66 | 23.28 | 0.86 | 0.78 |
| 32 | f32 | 27.78 | 15.32 | 57.65 | 1.81 | 3.76 |
| 32 | f64 | 86.08 | 78.24 | 151.87 | 1.10 | 1.94 |
| 64 | f32 | 162.62 | 77.44 | 360.05 | 2.10 | 4.65 |
| 64 | f64 | 420.49 | 266.77 | 1072.76 | 1.58 | 4.02 |

### syev + eig_clamp — timing only (no contender)

| N | dtype | syev ns | eig_clamp ns |
|---|-------|---------|--------------|
| 4 | f32 | 3.93 | 4.00 |
| 4 | f64 | 58.36 | 59.09 |
| 8 | f32 | 25.44 | 25.50 |
| 8 | f64 | 387.69 | 390.82 |
| 16 | f32 | 115.72 | 116.41 |
| 16 | f64 | 1781.74 | 1755.48 |
| 32 | f32 | 881.85 | 1019.85 |
| 32 | f64 | 8849.26 | 9306.22 |

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
