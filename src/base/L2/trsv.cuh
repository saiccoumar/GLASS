#pragma once
#include <cstdint>
#include "../barrier.cuh" // ct_size (moved there so subset-vendoring consumers get it)
#include "../flags.cuh"   // FillMode / Diag

// ─────────────────────────────────────────────────────────────────────────────
// trsv — triangular solve op(A) x = b, in place (BLAS TRSV).
// trmv — triangular matrix-vector product y = op(A) x (BLAS TRMV).
//
// Storage is column-major: the (row, col) element of the n×n matrix A is
// A[row + col*n]. The diagonal element of column j is A[j + j*n] in every flag
// combination.
//
// Flag semantics (NORMATIVE):
//   FILL   — which stored triangle of A holds the data (FillMode::Lower: the
//            strictly-upper entries are ignored; FillMode::Upper: vice versa.
//            FillMode::Full is invalid here — a triangular op needs a triangle).
//   DIAG   — Diag::Unit means the diagonal is implicitly 1 (A's diagonal is not
//            read and the divide is skipped); Diag::NonUnit reads it.
//   TRANSPOSE  — when true the routine works with op(A) = Aᵀ against that SAME
//            stored triangle: trsv solves Aᵀx = b, trmv computes Aᵀx.
//
// op(A) is lower-triangular (forward sweep) when
//   (FILL==Lower) != TRANSPOSE
// and upper-triangular (backward sweep) otherwise.
// ─────────────────────────────────────────────────────────────────────────────

// core impl: explicit rank/size + flags. Solves op(A) x = b in place.
// SizeT is deduced: uint32_t from the runtime overload, ct_size<N> from the
// compile-time overload (constant-folds the trip counts / indexing).
template <typename T, FillMode FILL, Diag DIAG, bool TRANSPOSE, typename SizeT>
__device__ void trsv_impl(uint32_t rank, uint32_t size, SizeT n, const T* A, T* x)
{
    static_assert(FILL != FillMode::Full, "trsv: FILL must name a triangle (Lower or Upper)");
    constexpr bool LOWER = (FILL == FillMode::Lower);
    constexpr bool UNIT  = (DIAG == Diag::Unit);
    // op(A) lower-triangular ⇒ forward elimination over pivots k = 0..n-1.
    constexpr bool FORWARD = (LOWER != TRANSPOSE);
    for (uint32_t step = 0; step < n; step++) {
        uint32_t k = FORWARD ? step : (n - 1 - step);
        // resolve pivot x[k] = x[k] / op(A)[k][k]   (diag is A[k+k*n])
        if (!UNIT) {
            if (rank == 0) x[k] = x[k] / A[k + k * n];
            __syncthreads();           // all threads read the resolved pivot
        }
        T xk = x[k];
        // subtract x[k] * op(A)[i][k] from the trailing/leading unknowns:
        //   op(A)[i][k] = TRANSPOSE ? A[k + i*n] : A[i + k*n]
        if (FORWARD) {
            for (uint32_t i = rank + k + 1; i < n; i += size)
                x[i] -= (TRANSPOSE ? A[k + i * n] : A[i + k * n]) * xk;
        } else {
            for (uint32_t i = rank; i < k; i += size)
                x[i] -= (TRANSPOSE ? A[k + i * n] : A[i + k * n]) * xk;
        }
        __syncthreads();               // pivot column consumed before next step
    }
}

// ─── trsv: runtime size ───────────────────────────────────────────────────────

/**
 * @brief Triangular solve `op(A) x = b` in place (TRSV).
 *
 * Solves the triangular system for `x`, overwriting the right-hand side `x`
 * (`x` holds `b` on entry, the solution on return). `A` is an `n×n` triangular
 * matrix stored column-major; only the triangle selected by `FILL` is read.
 * Set `TRANSPOSE=true` to solve `Aᵀx = b` against that same stored triangle, and
 * `DIAG=Diag::Unit` for an implicit unit diagonal (the diagonal of `A` is not
 * read). Column-oriented elimination (forward when `op(A)` is lower-triangular,
 * i.e. `(FILL==Lower) != TRANSPOSE`, backward otherwise); ends with a trailing
 * `__syncthreads()` so it composes without a defensive barrier.
 * SciPy equivalent:
 * `x = scipy.linalg.solve_triangular(A, b, lower=(FILL==Lower), unit_diagonal=(DIAG==Unit), trans=(1 if TRANSPOSE else 0))`.
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true solve `Aᵀx = b` (default false).
 * @param n  Dimension (`A` is `n×n`, `x` has length `n`).
 * @param A  Triangular matrix (column-major, `n*n` elements; read-only).
 * @param x  In/out right-hand side; on return holds the solution.
 */
template <typename T, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trsv(uint32_t n, const T* A, T* x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    trsv_impl<T, FILL, DIAG, TRANSPOSE>(rank, size, n, A, x);
}

// ─── trsv: compile-time size ──────────────────────────────────────────────────

/**
 * @brief Triangular solve `op(A) x = b` in place (TRSV), compile-time size.
 *
 * Same as the runtime `trsv` but with the dimension as a template parameter.
 * SciPy equivalent:
 * `x = scipy.linalg.solve_triangular(A, b, lower=(FILL==Lower), unit_diagonal=(DIAG==Unit), trans=(1 if TRANSPOSE else 0))`.
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam N     Dimension (`A` is `N×N`, `x` has length `N`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true solve `Aᵀx = b` (default false).
 * @param A  Triangular matrix (column-major, `N*N` elements; read-only).
 * @param x  In/out right-hand side; on return holds the solution.
 */
template <typename T, uint32_t N, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trsv(const T* A, T* x)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    trsv_impl<T, FILL, DIAG, TRANSPOSE>(rank, size, ct_size<N>{}, A, x);
}

// ─── trmv: out-of-place core ──────────────────────────────────────────────────

// core impl: explicit rank/size + flags. Computes y = op(A) x out of place.
// Each thread owns disjoint outputs y[i] and only reads the intact x — no
// interior barrier required. SizeT deduced (uint32_t or ct_size<N>, see trsv_impl).
template <typename T, FillMode FILL, Diag DIAG, bool TRANSPOSE, typename SizeT>
__device__ void trmv_impl(uint32_t rank, uint32_t size, SizeT n,
                          const T* A, const T* x, T* y)
{
    static_assert(FILL != FillMode::Full, "trmv: FILL must name a triangle (Lower or Upper)");
    constexpr bool LOWER = (FILL == FillMode::Lower);
    constexpr bool UNIT  = (DIAG == Diag::Unit);
    for (uint32_t i = rank; i < n; i += size) {
        // y[i] = sum_k op(A)[i][k] * x[k], summed over the triangle.
        // op(A)[i][k] = TRANSPOSE ? A[k + i*n] : A[i + k*n]; lower-triangular op
        // (forward) ⇒ k <= i, upper-triangular op ⇒ k >= i.
        constexpr bool LOWER_OP = (LOWER != TRANSPOSE);
        T acc = UNIT ? x[i] : static_cast<T>(0);
        if (LOWER_OP) {
            uint32_t k_end = UNIT ? i : (i + 1);      // exclude diag when UNIT
            for (uint32_t k = 0; k < k_end; k++)
                acc += (TRANSPOSE ? A[k + i * n] : A[i + k * n]) * x[k];
        } else {
            uint32_t k_start = UNIT ? (i + 1) : i;    // exclude diag when UNIT
            for (uint32_t k = k_start; k < n; k++)
                acc += (TRANSPOSE ? A[k + i * n] : A[i + k * n]) * x[k];
        }
        y[i] = acc;
    }
}

/**
 * @brief Triangular matrix-vector product `y = op(A) x`, out of place (TRMV).
 *
 * Computes the triangular matvec into a separate output `y` (distinct from the
 * input `x`). `A` is an `n×n` triangular matrix stored column-major; only the
 * triangle selected by `FILL` is read. Set `TRANSPOSE=true` to compute `Aᵀx`
 * against that same stored triangle, and `DIAG=Diag::Unit` for an implicit unit
 * diagonal. No interior barrier: each thread owns disjoint outputs and reads
 * the intact `x`. NumPy equivalent (lower, non-unit): `y = np.tril(A) @ x`
 * (upper: `np.triu(A) @ x`; transposed: `op(A).T @ x`; unit: diagonal forced
 * to 1).
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true compute `Aᵀx` (default false).
 * @param n  Dimension (`A` is `n×n`, `x` and `y` have length `n`).
 * @param A  Triangular matrix (column-major, `n*n` elements; read-only).
 * @param x  Input vector (length `n`; read-only).
 * @param y  Output vector (length `n`); must not alias `x`.
 */
template <typename T, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trmv(uint32_t n, const T* A, const T* x, T* y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    trmv_impl<T, FILL, DIAG, TRANSPOSE>(rank, size, n, A, x, y);
}

/**
 * @brief Triangular matrix-vector product `y = op(A) x`, out of place (TRMV), compile-time size.
 *
 * Compile-time-`N` overload of the out-of-place TRMV. NumPy equivalent (lower,
 * non-unit): `y = np.tril(A) @ x`.
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam N     Dimension (`A` is `N×N`, `x` and `y` have length `N`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true compute `Aᵀx` (default false).
 * @param A  Triangular matrix (column-major, `N*N` elements; read-only).
 * @param x  Input vector (length `N`; read-only).
 * @param y  Output vector (length `N`); must not alias `x`.
 */
template <typename T, uint32_t N, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trmv(const T* A, const T* x, T* y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    trmv_impl<T, FILL, DIAG, TRANSPOSE>(rank, size, ct_size<N>{}, A, x, y);
}

// ─── trmv: in-place wrapper (needs scratch) ──────────────────────────────────

/**
 * @brief Scratch length (in elements of `T`) required by the in-place `trmv`.
 *
 * The in-place TRMV wrapper computes `op(A) x` into a temporary then copies it
 * back over `x`; that temporary is `n` elements long.
 *
 * @param n  Dimension passed to the in-place `trmv`.
 * @return   Bytes the `scratch` buffer must hold.
 */
template <typename T>
__host__ __device__ inline constexpr std::size_t trmv_scratch_bytes(uint32_t n) { return static_cast<std::size_t>(n) * sizeof(T); }

/**
 * @brief Triangular matrix-vector product `x = op(A) x`, in place (TRMV).
 *
 * In-place form: overwrites `x` with `op(A) x`. Because `trmv` reads the whole
 * `x` while writing each output, the wrapper computes into a caller-provided
 * `scratch` (length `n`, see `trmv_scratch_bytes`) and copies the result back
 * with a single barrier in between. Ends with a trailing `__syncthreads()`.
 * NumPy equivalent (lower, non-unit): `x = np.tril(A) @ x`.
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true compute `Aᵀx` (default false).
 * @param n        Dimension (`A` is `n×n`, `x` and `scratch` have length `n`).
 * @param A        Triangular matrix (column-major, `n*n` elements; read-only).
 * @param x        In/out vector (length `n`); on return holds `op(A) x`.
 * @param scratch  Workspace of length `n` (see `trmv_scratch_bytes`).
 */
template <typename T, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trmv(uint32_t n, const T* A, T* x, T* scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    trmv_impl<T, FILL, DIAG, TRANSPOSE>(rank, size, n, A, x, scratch);
    __syncthreads();                              // scratch fully written before read-back
    for (uint32_t i = rank; i < n; i += size) x[i] = scratch[i];
    __syncthreads();
}

/**
 * @brief Triangular matrix-vector product `x = op(A) x`, in place (TRMV), compile-time size.
 *
 * Compile-time-`N` overload of the in-place TRMV. NumPy equivalent (lower,
 * non-unit): `x = np.tril(A) @ x`.
 *
 * @tparam T     Scalar type (e.g. `float`, `double`).
 * @tparam N     Dimension (`A` is `N×N`, `x` and `scratch` have length `N`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true compute `Aᵀx` (default false).
 * @param A        Triangular matrix (column-major, `N*N` elements; read-only).
 * @param x        In/out vector (length `N`); on return holds `op(A) x`.
 * @param scratch  Workspace of length `N` (see `trmv_scratch_bytes`).
 */
template <typename T, uint32_t N, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trmv(const T* A, T* x, T* scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    trmv_impl<T, FILL, DIAG, TRANSPOSE>(rank, size, ct_size<N>{}, A, x, scratch);
    __syncthreads();
    for (uint32_t i = rank; i < N; i += size) x[i] = scratch[i];
    __syncthreads();
}
