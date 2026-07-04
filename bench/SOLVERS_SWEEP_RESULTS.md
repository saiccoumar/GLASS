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

_No quiet-GPU sweep recorded yet. Run `python3 bench/tune.py --sm auto --legs
solvers` on an idle GPU (after `--prebuild`); this block is then auto-filled
with the measured ns/problem tables._

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
