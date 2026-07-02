#pragma once
#include <cstdint>

/**
 * @file posv.cuh
 * @brief SPD linear solve via Cholesky + two triangular solves (pure SIMT).
 *
 * `posv` / `potrs` are thin single-block compositions of `potrf`
 * (`potrf.cuh`) and `trsv` (`trsv.cuh`). Both callees end with a trailing
 * `__syncthreads()`, so the factor and the two solves compose with NO inter-call
 * barrier. Pure-SIMT companion to `glass::nvidia::posv`. Column-major throughout.
 */

/**
 * @brief Add a diagonal regularization shift to A in place (single-block helper).
 *
 * `REG_DIAG=false` adds `rho·I` (Marquardt shift); `REG_DIAG=true` adds
 * `rho·diag(A)`, i.e. scales each diagonal by `(1+rho)` (Levenberg shift —
 * scale-invariant across rows of very different magnitude, e.g. mixed
 * prismatic/revolute Jacobians). Trailing `__syncthreads()` so the shifted A is
 * block-visible before factoring. Internal; used by the flagged `posv` overloads.
 */
template <typename T, bool REG_DIAG = false>
__device__ void _posv_regularize(uint32_t n, T *A, T rho)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size) {
        if constexpr (REG_DIAG) A[i*n + i] += rho * A[i*n + i];   // rho*diag(A)
        else                    A[i*n + i] += rho;                // rho*I
    }
    __syncthreads();
}

/**
 * @brief Solve the SPD system `A x = b` in one block (LAPACK posv).
 *
 * Factors `A = L Lᵀ` in place via Cholesky, then forward-solves `L y = b` and
 * back-solves `Lᵀ x = y`. On return `A` holds its lower Cholesky factor `L` and
 * `b` holds the solution `x`. `A` must be symmetric positive-definite; behaviour
 * on non-SPD input is undefined (the Cholesky step produces NaN, no info flag).
 * Thread-count invariant. NumPy equivalent: `x = np.linalg.solve(A, b)` (A SPD).
 *
 * @note Regularize / check / Levenberg flags live on the **multi-RHS** overload
 * (`b` is an `n×1` column-major `B`, so the single-RHS flagged solve is
 * `posv<T, N, 1, REGULARIZE, CHECK, REG_DIAG>(A, b, rho, s_fail)`). The single-RHS
 * overload deliberately carries no flag template params: a flagged single-RHS
 * form (`<T, N, bool…>`) would be ambiguous with `<T, N, NRHS, bool…>` at
 * `NRHS∈{0,1}` (identical resolved signature), so flags are routed through NRHS.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Dimension (`A` is `n×n`, `b` has length `n`).
 * @param A  In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T>
__device__ void posv(uint32_t n, T *A, T *b)
{
    potrf<T>(n, A);            // A -> L (lower); trailing __syncthreads
    trsv<T>(n, A, b);                                                        // forward: L y = b
    trsv<T, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(n, A, b);    // back:   Lᵀ x = y
}

/**
 * @brief Compile-time-size SPD solve `A x = b` (LAPACK posv).
 *
 * Same as the runtime `posv` with the dimension as a template parameter.
 * NumPy equivalent: `x = np.linalg.solve(A, b)` (A SPD). For the regularized /
 * checked / Levenberg path use the multi-RHS overload with NRHS=1, e.g.
 * `posv<T, N, 1, true, true, true>(A, b, rho, s_fail)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension (`A` is `N×N`, `b` has length `N`).
 * @param A  In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T, uint32_t N>
__device__ void posv(T *A, T *b) { posv<T>(N, A, b); }

/**
 * @brief Solve the SPD system `A x = b` from a precomputed Cholesky factor (LAPACK potrs).
 *
 * Given the lower factor `L` (e.g. from `potrf`), solves
 * `L Lᵀ x = b` by forward then back substitution — the reusable-factor /
 * multi-solve path (no re-factor). `L` is read-only; `b` is overwritten with `x`.
 * Thread-count invariant. SciPy equivalent: `x = scipy.linalg.cho_solve((L, True), b)`.
 *
 * @tparam T  Scalar type.
 * @param n  Dimension (`L` is `n×n`, `b` has length `n`).
 * @param L  Lower Cholesky factor (column-major, `n*n`; read-only).
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T>
__device__ void potrs(uint32_t n, const T *L, T *b)
{
    trsv<T>(n, L, b);                                                        // forward: L y = b
    trsv<T, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(n, L, b);    // back:   Lᵀ x = y
}

/**
 * @brief Compile-time-size SPD solve from a precomputed Cholesky factor (LAPACK potrs).
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension.
 * @param L  Lower Cholesky factor (column-major, `N*N`; read-only).
 * @param b  In/out right-hand side; on return holds the solution `x`.
 */
template <typename T, uint32_t N>
__device__ void potrs(const T *L, T *b) { potrs<T>(N, L, b); }

// ─── multi-RHS overloads (column-major B, factor once / solve per column) ─────

/**
 * @brief Solve the SPD system `A X = B` with multiple right-hand sides (LAPACK posv).
 *
 * Factors `A = L Lᵀ` in place **once** via Cholesky, then solves each of the
 * `nrhs` columns of `B` by a forward (`L y = b`) and back (`Lᵀ x = y`)
 * substitution. On return `A` holds its lower Cholesky factor `L` and `B` holds
 * the solution `X`. `A` must be symmetric positive-definite; behaviour on non-SPD
 * input is undefined (the Cholesky step produces NaN, no info flag).
 *
 * `B` (and `X`) is `n × nrhs` stored **column-major**: column `c` begins at
 * `B + c*n` and occupies `n` contiguous elements. The Cholesky factor completes
 * before the solve (its trailing `__syncthreads()`); all columns are then solved
 * together by the multi-RHS `trsm` (per-step barriers shared across right-hand
 * sides). Thread-count invariant. NumPy equivalent:
 * `X = np.linalg.solve(A, B)` (A SPD, B `n×nrhs`).
 *
 * @par Regularize + check (`REGULARIZE` / `CHECK` / `REG_DIAG`, all compile-out, default off)
 * `REGULARIZE` adds a shift to `A`'s diagonal before factoring — `rho·I`
 * (Marquardt) by default, or `rho·diag(A)` (Levenberg, scale-invariant) when
 * `REG_DIAG` is also set — used to push a borderline-indefinite Hessian (e.g. `Huu`)
 * back to SPD; `CHECK` forwards to the checked Cholesky and sets `*s_fail = 1` on
 * a non-PD pivot, so a caller can escalate `rho` and retry. All default false and
 * compile out (`if constexpr`), leaving the unflagged instantiation byte-identical
 * to the original. This is the fused "regularize → factor → solve" path: e.g.
 * `posv<T, N, NRHS, true, true>(A, B, rho, s_fail)` (add a trailing `true` for Levenberg).
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam REGULARIZE  If true, shift A before factoring (default false, compiles out).
 * @tparam CHECK  If true, report a non-PD pivot via `s_fail` (default false, compiles out).
 * @tparam REG_DIAG    With REGULARIZE: shift by `rho·diag(A)` instead of `rho·I` (default false).
 * @param n      Dimension (`A` is `n×n`, each column of `B` has length `n`).
 * @param nrhs   Number of right-hand sides (columns of `B`).
 * @param A      In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param B      In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 * @param rho    Diagonal shift added to A when REGULARIZE (ignored otherwise).
 * @param s_fail Optional non-PD flag when CHECK (set to 1 on a non-PD pivot, else 0).
 */
template <typename T, bool REGULARIZE = false, bool CHECK = false, bool REG_DIAG = false>
__device__ void posv(uint32_t n, uint32_t nrhs, T *A, T *B, T rho = T(0), int *s_fail = nullptr)
{
    if constexpr (REGULARIZE) _posv_regularize<T, REG_DIAG>(n, A, rho);  // rho*I or rho*diag(A)
    potrf<T, CHECK>(n, A, s_fail);   // A -> L (lower); trailing __syncthreads
    trsm<T>(n, nrhs, A, B);                                                        // forward: L Y = B
    trsm<T, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(n, nrhs, A, B);    // back:   Lᵀ X = Y
}

/**
 * @brief Compile-time-size multi-RHS SPD solve `A X = B` (LAPACK posv).
 *
 * Same as the runtime multi-RHS `posv` with the dimension and right-hand-side
 * count as template parameters. `B` is `N × NRHS` column-major (column `c` at
 * `B + c*N`). Factored once, solved per column. NumPy equivalent:
 * `X = np.linalg.solve(A, B)` (A SPD).
 *
 * The optional `REGULARIZE` / `CHECK` / `REG_DIAG` flags (default off, compile out)
 * add a diagonal shift before factoring and report a non-PD pivot via `s_fail` — the
 * fused regularize→factor→solve path `posv<T, N, NRHS, true, true>(A, B, rho, s_fail)`.
 * `REG_DIAG` (appended last so existing `<…, true, true>` callers are unaffected)
 * switches the shift from `rho·I` to `rho·diag(A)` (Levenberg). A flagged single-RHS
 * solve is just NRHS=1: `posv<T, N, 1, true, true, true>(A, b, rho, s_fail)`.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension (`A` is `N×N`, each column of `B` has length `N`).
 * @tparam NRHS  Number of right-hand sides (columns of `B`).
 * @tparam REGULARIZE  If true, shift A before factoring (default false, compiles out).
 * @tparam CHECK  If true, report a non-PD pivot via `s_fail` (default false, compiles out).
 * @tparam REG_DIAG    With REGULARIZE: shift by `rho·diag(A)` instead of `rho·I` (default false).
 * @param A  In/out SPD matrix (column-major); overwritten with its factor `L`.
 * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 * @param rho    Diagonal shift added to A when REGULARIZE (ignored otherwise).
 * @param s_fail Optional non-PD flag when CHECK (set to 1 on a non-PD pivot, else 0).
 */
template <typename T, uint32_t N, uint32_t NRHS, bool REGULARIZE = false, bool CHECK = false, bool REG_DIAG = false>
__device__ void posv(T *A, T *B, T rho = T(0), int *s_fail = nullptr)
{
    posv<T, REGULARIZE, CHECK, REG_DIAG>(N, NRHS, A, B, rho, s_fail);
}

/**
 * @brief Multi-RHS SPD solve `A X = B` from a precomputed Cholesky factor (LAPACK potrs).
 *
 * Given the lower factor `L` (e.g. from `potrf`), solves
 * `L Lᵀ X = B` for each of the `nrhs` columns by forward then back substitution
 * — the reusable-factor / multi-solve path (no re-factor). `L` is read-only; `B`
 * is overwritten with `X`.
 *
 * `B` (and `X`) is `n × nrhs` stored **column-major**: column `c` begins at
 * `B + c*n`. All columns are solved together by the multi-RHS `trsm` (per-step
 * barriers shared across right-hand sides). Thread-count invariant. SciPy
 * equivalent: `X = scipy.linalg.cho_solve((L, True), B)`.
 *
 * @tparam T     Scalar type.
 * @param n      Dimension (`L` is `n×n`, each column of `B` has length `n`).
 * @param nrhs   Number of right-hand sides (columns of `B`).
 * @param L      Lower Cholesky factor (column-major, `n*n`; read-only).
 * @param B      In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 */
template <typename T>
__device__ void potrs(uint32_t n, uint32_t nrhs, const T *L, T *B)
{
    trsm<T>(n, nrhs, L, B);                                                        // forward: L Y = B
    trsm<T, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(n, nrhs, L, B);    // back:   Lᵀ X = Y
}

/**
 * @brief Compile-time-size multi-RHS SPD solve from a precomputed Cholesky factor (LAPACK potrs).
 *
 * `B` is `N × NRHS` column-major (column `c` at `B + c*N`). Solved per column,
 * no re-factor. SciPy equivalent: `X = scipy.linalg.cho_solve((L, True), B)`.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension.
 * @tparam NRHS  Number of right-hand sides (columns of `B`).
 * @param L  Lower Cholesky factor (column-major, `N*N`; read-only).
 * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 */
template <typename T, uint32_t N, uint32_t NRHS>
__device__ void potrs(const T *L, T *B) { potrs<T>(N, NRHS, L, B); }
