#pragma once
#include <cstdint>

// ─── shared 4-row register-tile helpers ──────────────────────────────────────
// Identical guarded copy lives in L2/gemv.cuh, L3/gemm.cuh, and L3/syrk.cuh:
// these base headers are #include'd inside `namespace glass { }` (with #pragma
// once), so whichever the umbrella pulls in first defines the block and the
// guard skips the twins — no dependency on include order.
#ifndef GLASS_TILE4_HELPERS_DEFINED
#define GLASS_TILE4_HELPERS_DEFINED

// Scalar types with a 16-byte vector load (float4 / 2×double2).
template <typename T> struct tile4_has_vec { static constexpr bool value = false; };
template <> struct tile4_has_vec<float>  { static constexpr bool value = true; };
template <> struct tile4_has_vec<double> { static constexpr bool value = true; };

// Load 4 consecutive elements p[0..3] into registers. The vector overloads
// require `p` 16-byte aligned. Loads only change HOW values reach registers —
// never any output's accumulation order — so vector vs scalar is bit-identical.
__device__ __forceinline__ void tile4_load(const float *p, float &a0, float &a1, float &a2, float &a3)
{
    float4 v = *reinterpret_cast<const float4 *>(p);
    a0 = v.x; a1 = v.y; a2 = v.z; a3 = v.w;
}
__device__ __forceinline__ void tile4_load(const double *p, double &a0, double &a1, double &a2, double &a3)
{
    double2 v0 = *reinterpret_cast<const double2 *>(p);
    double2 v1 = *reinterpret_cast<const double2 *>(p + 2);
    a0 = v0.x; a1 = v0.y; a2 = v1.x; a3 = v1.y;
}
template <typename T>
__device__ __forceinline__ void tile4_load(const T *p, T &a0, T &a1, T &a2, T &a3)
{
    a0 = p[0]; a1 = p[1]; a2 = p[2]; a3 = p[3];
}

// Base-pointer half of the vector-load precondition: with a leading dimension
// that is a multiple of 4 (float; 2 suffices for double but we require 4
// uniformly) and 4-row tiles starting at row r ≡ 0 (mod 4), every tile start
// A + r + k*ld stays 16-byte aligned iff the base pointer is.
template <typename T>
__device__ __forceinline__ bool tile4_aligned(const T *p)
{
    return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
}
// Deterministic epilogue `a*x + b*y` (used by paths that are validated
// BIT-FOR-BIT against a differently-shaped/instantiated twin, e.g. warp CT
// syrk vs block runtime syrk): computed as fma(a, x, mul_rn(b, y)) via
// intrinsics that the compiler is documented never to contract/merge, so every
// instantiation (rolled or unrolled, CT or RT) emits the identical sequence.
__device__ __forceinline__ float tile4_axpby(float a, float x, float b, float y)
{
    return __fmaf_rn(a, x, __fmul_rn(b, y));
}
__device__ __forceinline__ double tile4_axpby(double a, double x, double b, double y)
{
    return __fma_rn(a, x, __dmul_rn(b, y));
}
template <typename T>
__device__ __forceinline__ T tile4_axpby(T a, T x, T b, T y)
{
    return a*x + b*y;
}
#endif  // GLASS_TILE4_HELPERS_DEFINED

// ─── 4-row register-tiled gemv cores (TRANSPOSE=false, ROW_MAJOR_A=false) ────
// Same skeleton as gemm's register tiling with x[col] as the reused "B value":
// each thread owns 4 CONSECUTIVE ROWS of y (ceil(m/4) row-tiles strided over
// the block); per column, x[col] is loaded once and reused by 4 FMAs against
// the contiguous A[r..r+3, col] (vector-loaded when m%4==0 and A is 16-byte
// aligned; scalar otherwise and for the m%4 tail rows). Each y element keeps
// the SAME serial ascending-col accumulation chain as the untiled loop and is
// written by exactly one thread ⇒ bit-identical thread-count invariance.
// The TRANSPOSE / ROW_MAJOR_A combos keep the untiled loop: their per-output
// row data is strided (TRANSPOSE walks a column contiguously per output, which
// the flat loop already does optimally; ROW_MAJOR_A rows are contiguous per
// output) — 4-row tiling would turn those contiguous walks into strided ones.

template <typename T, bool HAS_BETA, bool VEC>
__device__ __forceinline__ void gemv_tile4_loop(uint32_t rank, uint32_t size,
                                                uint32_t m, uint32_t n,
                                                T alpha, const T *__restrict__ A,
                                                const T *__restrict__ x,
                                                T beta, T *__restrict__ y)
{
    const uint32_t ntiles = (m + 3u) / 4u;
    for (uint32_t t = rank; t < ntiles; t += size) {
        const uint32_t r = t * 4u;
        if (r + 4u <= m) {                     // full 4-row tile
            T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0);
            T acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
            for (uint32_t col = 0; col < n; col++) {
                const T xv = x[col];
                T a0, a1, a2, a3;
                if constexpr (VEC) {
                    tile4_load(A + (r + col*m), a0, a1, a2, a3);
                } else {
                    a0 = A[r      + col*m]; a1 = A[r + 1u + col*m];
                    a2 = A[r + 2u + col*m]; a3 = A[r + 3u + col*m];
                }
                acc0 += a0 * xv; acc1 += a1 * xv; acc2 += a2 * xv; acc3 += a3 * xv;
            }
            T *__restrict__ yr = y + r;
            if constexpr (HAS_BETA) {
                yr[0] = alpha*acc0 + beta*yr[0];
                yr[1] = alpha*acc1 + beta*yr[1];
                yr[2] = alpha*acc2 + beta*yr[2];
                yr[3] = alpha*acc3 + beta*yr[3];
            } else {
                yr[0] = alpha*acc0; yr[1] = alpha*acc1;
                yr[2] = alpha*acc2; yr[3] = alpha*acc3;
            }
        } else {                               // m%4 tail rows: scalar per row
            for (uint32_t row = r; row < m; row++) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < n; col++)
                    res += A[row + col*m] * x[col];
                if constexpr (HAS_BETA) y[row] = alpha*res + beta*y[row];
                else                    y[row] = alpha*res;
            }
        }
    }
}

template <typename T, bool HAS_BETA>
__device__ void gemv_tile4(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, const T *__restrict__ A, const T *__restrict__ x,
                           T beta, T *__restrict__ y)
{
    if constexpr (tile4_has_vec<T>::value) {
        if ((m % 4u == 0u) && tile4_aligned(A)) {
            gemv_tile4_loop<T, HAS_BETA, true>(rank, size, m, n, alpha, A, x, beta, y);
            return;
        }
    }
    gemv_tile4_loop<T, HAS_BETA, false>(rank, size, m, n, alpha, A, x, beta, y);
}

// compile-time twin: M, N as template params (fully unrolled column loop; the
// M%4 tail is statically absent when M is a multiple of 4).
template <typename T, uint32_t M, uint32_t N, bool HAS_BETA, bool VEC>
__device__ __forceinline__ void gemv_tile4_loop_ct(uint32_t rank, uint32_t size,
                                                   T alpha, const T *__restrict__ A,
                                                   const T *__restrict__ x,
                                                   T beta, T *__restrict__ y)
{
    constexpr uint32_t NTILES = (M + 3u) / 4u;
    for (uint32_t t = rank; t < NTILES; t += size) {
        const uint32_t r = t * 4u;
        if (r + 4u <= M) {                     // full 4-row tile
            T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0);
            T acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++) {
                const T xv = x[col];
                T a0, a1, a2, a3;
                if constexpr (VEC) {
                    tile4_load(A + (r + col*M), a0, a1, a2, a3);
                } else {
                    a0 = A[r      + col*M]; a1 = A[r + 1u + col*M];
                    a2 = A[r + 2u + col*M]; a3 = A[r + 3u + col*M];
                }
                acc0 += a0 * xv; acc1 += a1 * xv; acc2 += a2 * xv; acc3 += a3 * xv;
            }
            T *__restrict__ yr = y + r;
            if constexpr (HAS_BETA) {
                yr[0] = alpha*acc0 + beta*yr[0];
                yr[1] = alpha*acc1 + beta*yr[1];
                yr[2] = alpha*acc2 + beta*yr[2];
                yr[3] = alpha*acc3 + beta*yr[3];
            } else {
                yr[0] = alpha*acc0; yr[1] = alpha*acc1;
                yr[2] = alpha*acc2; yr[3] = alpha*acc3;
            }
        } else {                               // M%4 tail rows: scalar per row
            for (uint32_t row = r; row < M; row++) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < N; col++)
                    res += A[row + col*M] * x[col];
                if constexpr (HAS_BETA) y[row] = alpha*res + beta*y[row];
                else                    y[row] = alpha*res;
            }
        }
    }
}

template <typename T, uint32_t M, uint32_t N, bool HAS_BETA>
__device__ void gemv_tile4_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, const T *__restrict__ x,
                              T beta, T *__restrict__ y)
{
    if constexpr (tile4_has_vec<T>::value && (M % 4u == 0u)) {
        if (tile4_aligned(A)) {
            gemv_tile4_loop_ct<T, M, N, HAS_BETA, true>(rank, size, alpha, A, x, beta, y);
            return;
        }
    }
    gemv_tile4_loop_ct<T, M, N, HAS_BETA, false>(rank, size, alpha, A, x, beta, y);
}

// core impl: explicit rank/size + layout flags
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, const T *__restrict__ A, const T *__restrict__ x, T beta, T *__restrict__ y)
{
    if constexpr (TRANSPOSE) {
        for (uint32_t row = rank; row < n; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < m; col++) {
                T a = ROW_MAJOR_A ? A[col*n + row] : A[col + row*m];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    } else if constexpr (!ROW_MAJOR_A) {
        gemv_tile4<T, true>(rank, size, m, n, alpha, A, x, beta, y);
    } else {
        for (uint32_t row = rank; row < m; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < n; col++) {
                T a = A[row*n + col];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    }
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, const T *__restrict__ A, const T *__restrict__ x, T *__restrict__ y)
{
    if constexpr (TRANSPOSE) {
        for (uint32_t row = rank; row < n; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < m; col++) {
                T a = ROW_MAJOR_A ? A[col*n + row] : A[col + row*m];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    } else if constexpr (!ROW_MAJOR_A) {
        gemv_tile4<T, false>(rank, size, m, n, alpha, A, x, static_cast<T>(0), y);
    } else {
        for (uint32_t row = rank; row < m; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < n; col++) {
                T a = A[row*n + col];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    }
}

// ─── runtime variants ─────────────────────────────────────────────────────────

/**
 * @brief Matrix-vector product: `y = alpha * A * x + beta * y` (GEMV).
 *
 * Threads are distributed over the output rows of the `m×n` matrix `A`. Set
 * `TRANSPOSE=true` to compute `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`
 * (`A` is column-major by default). NumPy equivalent: `y = alpha*A@x + beta*y`
 * (or `alpha*A.T@x + beta*y` when transposed).
 *
 * Unlike `gemm` — where a row-major operand is just a transpose, so the only
 * layout flag is `ROW_MAJOR_C` — GEMV keeps a per-matrix `ROW_MAJOR` flag:
 * `TRANSPOSE` already selects the mathematical operation (`A·x` vs `Aᵀ·x`), so it
 * cannot also stand in for the storage order. `TRANSPOSE` and `ROW_MAJOR` are
 * therefore independent. (This flag fully subsumes the former `gemv_ex`, which
 * was just `gemv` with the defaults removed and has been deleted.)
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param m      Number of rows of `A`.
 * @param n      Number of columns of `A`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `m*n` elements.
 * @param x      Input vector (length `n`, or `m` when transposed).
 * @param beta   Scalar multiplier on the prior `y`.
 * @param y      In/out vector (length `m`, or `n` when transposed).
 */
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, const T *__restrict__ A, const T *__restrict__ x, T beta, T *__restrict__ y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Matrix-vector product: `y = alpha * A * x` (GEMV), no-beta overload.
 *
 * Same as the full GEMV but overwrites `y` (no `beta * y` term). Set
 * `TRANSPOSE=true` for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. NumPy
 * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param m      Number of rows of `A`.
 * @param n      Number of columns of `A`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `m*n` elements.
 * @param x      Input vector (length `n`, or `m` when transposed).
 * @param y      Output vector (length `m`, or `n` when transposed).
 */
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, const T *__restrict__ A, const T *__restrict__ x, T *__restrict__ y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// compile-time impl: M, N as template params so inner col-loop is fully unrolled
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, const T *__restrict__ x, T beta, T *__restrict__ y)
{
    if constexpr (TRANSPOSE) {
        for (uint32_t row = rank; row < N; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < M; col++) {
                T a = ROW_MAJOR_A ? A[col*N + row] : A[col + row*M];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    } else if constexpr (!ROW_MAJOR_A) {
        gemv_tile4_ct<T, M, N, true>(rank, size, alpha, A, x, beta, y);
    } else {
        for (uint32_t row = rank; row < M; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++) {
                T a = A[row*N + col];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    }
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, const T *__restrict__ x, T *__restrict__ y)
{
    if constexpr (TRANSPOSE) {
        for (uint32_t row = rank; row < N; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < M; col++) {
                T a = ROW_MAJOR_A ? A[col*N + row] : A[col + row*M];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    } else if constexpr (!ROW_MAJOR_A) {
        gemv_tile4_ct<T, M, N, false>(rank, size, alpha, A, x, static_cast<T>(0), y);
    } else {
        for (uint32_t row = rank; row < M; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++) {
                T a = A[row*N + col];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    }
}

// ─── compile-time size variants ───────────────────────────────────────────────

/**
 * @brief Matrix-vector product: `y = alpha * A * x + beta * y` (GEMV), compile-time size.
 *
 * Compile-time-`M`,`N` overload; the inner column loop is fully unrolled. Set
 * `TRANSPOSE=true` for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. NumPy
 * equivalent: `y = alpha*A@x + beta*y` (or `alpha*A.T@x + beta*y`).
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam M          Number of rows of `A` (compile-time constant).
 * @tparam N          Number of columns of `A` (compile-time constant).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `M*N` elements.
 * @param x      Input vector (length `N`, or `M` when transposed).
 * @param beta   Scalar multiplier on the prior `y`.
 * @param y      In/out vector (length `M`, or `N` when transposed).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, const T *__restrict__ A, const T *__restrict__ x, T beta, T *__restrict__ y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, x, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Matrix-vector product: `y = alpha * A * x` (GEMV), compile-time size, no-beta overload.
 *
 * Compile-time-`M`,`N` overload that overwrites `y` (no `beta * y` term). NumPy
 * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam M          Number of rows of `A` (compile-time constant).
 * @tparam N          Number of columns of `A` (compile-time constant).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `M*N` elements.
 * @param x      Input vector (length `N`, or `M` when transposed).
 * @param y      Output vector (length `M`, or `N` when transposed).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, const T *__restrict__ A, const T *__restrict__ x, T *__restrict__ y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, x, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    // Single-warp GEMV: one 32-lane warp computes the matvec, lanes striding over
    // the output rows (lane i owns output rows i, i+32, …). Each lane's row is an
    // independent inner product — no cross-lane communication, no shared scratch,
    // no `__syncthreads`. Reuses the block impl `gemv_impl_ct(lane, 32u, …)` exactly
    // as `warp::gemm` reuses `gemm_impl_ct`. For warp-per-problem kernels packing
    // many small matvecs into one block via independent warps. Full 32 lanes
    // required.

    /**
     * @brief Matrix-vector product within one warp: `y = alpha * A * x + beta * y` (GEMV), single-warp, compile-time size.
     *
     * One 32-lane warp computes the matvec with lanes striding over the output rows
     * of the `M×N` matrix `A` (each row an independent inner product). Set
     * `TRANSPOSE=true` for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. No shared
     * scratch, no `__syncthreads`; independent warps may run distinct problems
     * concurrently. Full 32 lanes required. `C`/`y` is read (the `beta * y` term);
     * use the no-beta overload to write into uninitialized destinations. NumPy
     * equivalent: `y = alpha*A@x + beta*y` (or `alpha*A.T@x + beta*y` when transposed).
     *
     * @tparam T          Scalar type (e.g. `float`, `double`).
     * @tparam M          Number of rows of `A` (compile-time constant).
     * @tparam N          Number of columns of `A` (compile-time constant).
     * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
     * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A      Input matrix of `M*N` elements.
     * @param x      Input vector (length `N`, or `M` when transposed).
     * @param beta   Scalar multiplier on the prior `y`.
     * @param y      In/out vector (length `M`, or `N` when transposed).
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void gemv(T alpha, const T *__restrict__ A, const T *__restrict__ x, T beta, T *__restrict__ y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(lane, 32u, alpha, A, x, beta, y);
        __syncwarp();
    }

    /**
     * @brief Matrix-vector product within one warp: `y = alpha * A * x` (GEMV), single-warp, compile-time size, implicit beta = 0.
     *
     * Overwrites `y` (no `beta * y` term — `y` is never read, so it is safe to write
     * into cold/uninitialized scratch). Otherwise identical to the beta overload
     * above. No shared scratch, no `__syncthreads`. Full 32 lanes required. NumPy
     * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
     *
     * @tparam T          Scalar type (e.g. `float`, `double`).
     * @tparam M          Number of rows of `A` (compile-time constant).
     * @tparam N          Number of columns of `A` (compile-time constant).
     * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
     * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A      Input matrix of `M*N` elements.
     * @param x      Input vector (length `N`, or `M` when transposed).
     * @param y      Output vector (length `M`, or `N` when transposed; overwritten).
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void gemv(T alpha, const T *__restrict__ A, const T *__restrict__ x, T *__restrict__ y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(lane, 32u, alpha, A, x, y);
        __syncwarp();
    }
}
