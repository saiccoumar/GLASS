#pragma once
#include <cstdint>
#include "../flags.cuh"   // FillMode (shared with trsv/trmv/trsm)

// FillMode (see flags.cuh) selects which triangle of the symmetric result C is
// written:
//   Lower — only cells with row >= col
//   Upper — only cells with row <= col
//   Full  — both triangles (C is materialized as a full symmetric matrix)

// ─── helpers: is this (row,col) cell in the canonical (computed) triangle? ────
// Lower/Full compute the lower triangle (row>=col); Upper computes row<=col.
// For Full we ALSO materialize the mirror, but only the lower-owning thread
// writes it — see the write phase below.
__device__ __forceinline__ bool syrk_in_canonical(FillMode fill, uint32_t row, uint32_t col)
{
    return (fill == FillMode::Upper) ? (row <= col) : (row >= col);
}

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

// ─── 4-row register-tiled syrk CT core (TRANSPOSE=false, ROW_MAJOR=false) ────
// Same skeleton as gemm's register tiling: each thread owns 4 CONSECUTIVE ROWS
// of one C column (ceil(N/4)*N row-tiles strided over the block). The
// "B value" is A[col, i], loaded once per i and reused by 4 FMAs against the
// contiguous A[r..r+3, i] (vector-loaded when N%4==0 and A is 16-byte
// aligned). Triangle ownership: tiles fully outside the canonical triangle are
// skipped (no wasted FLOPs), tiles fully inside take the 4-row fast path, and
// straddling / N%4-tail tiles fall back to the per-row scalar body with the
// same syrk_in_canonical check as the flat loop. Every canonical element (and
// its Full-fill mirror) is written by exactly one thread with the same serial
// ascending-i accumulation chain as the untiled loop ⇒ bit-identical
// thread-count invariance, no interior barrier.
template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool HAS_BETA, bool VEC>
__device__ __forceinline__ void syrk_tile4_loop_ct(uint32_t rank, uint32_t size,
                                                   T alpha, const T *__restrict__ A,
                                                   T beta, T *__restrict__ C)
{
    constexpr uint32_t TPC    = (N + 3u) / 4u;   // 4-row tiles per column of C
    constexpr uint32_t NTILES = TPC * N;
    for (uint32_t t = rank; t < NTILES; t += size) {
        const uint32_t col = t / TPC;
        const uint32_t r   = (t % TPC) * 4u;
        const bool full_tile = (r + 4u <= N);
        bool none_canon, all_canon;
        if constexpr (FILL == FillMode::Upper) {   // canonical: row <= col
            none_canon = (r > col);
            all_canon  = full_tile && (r + 3u <= col);
        } else {                                    // Lower/Full canonical: row >= col
            none_canon = (r + 3u < col);            // max possible row < col
            all_canon  = full_tile && (r >= col);
        }
        if (none_canon) continue;                   // mirror (or nothing) handles it
        if (all_canon) {                            // full 4-row canonical tile
            T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0);
            T acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
            for (uint32_t i = 0; i < K; i++) {
                const T ac = A[col + i*N];
                T a0, a1, a2, a3;
                if constexpr (VEC) {
                    tile4_load(A + (r + i*N), a0, a1, a2, a3);
                } else {
                    a0 = A[r      + i*N]; a1 = A[r + 1u + i*N];
                    a2 = A[r + 2u + i*N]; a3 = A[r + 3u + i*N];
                }
                acc0 += a0 * ac; acc1 += a1 * ac; acc2 += a2 * ac; acc3 += a3 * ac;
            }
            // canonical cells C[r+j, col] (col-major, contiguous), then the
            // Full-fill mirrors C[col, r+j] — each mirror cell is touched by
            // ONLY this thread (the owner of its transposed canonical cell),
            // exactly like the flat loop, so no cross-thread hazard.
            T *__restrict__ c = C + (r + col*N);
            if constexpr (HAS_BETA) {
                c[0] = tile4_axpby(alpha, acc0, beta, c[0]);
                c[1] = tile4_axpby(alpha, acc1, beta, c[1]);
                c[2] = tile4_axpby(alpha, acc2, beta, c[2]);
                c[3] = tile4_axpby(alpha, acc3, beta, c[3]);
            } else {
                c[0] = alpha*acc0; c[1] = alpha*acc1;
                c[2] = alpha*acc2; c[3] = alpha*acc3;
            }
            if constexpr (FILL == FillMode::Full) {
                if constexpr (HAS_BETA) {
                    if (r      != col) C[col + (r     )*N] = tile4_axpby(alpha, acc0, beta, C[col + (r     )*N]);
                    if (r + 1u != col) C[col + (r + 1u)*N] = tile4_axpby(alpha, acc1, beta, C[col + (r + 1u)*N]);
                    if (r + 2u != col) C[col + (r + 2u)*N] = tile4_axpby(alpha, acc2, beta, C[col + (r + 2u)*N]);
                    if (r + 3u != col) C[col + (r + 3u)*N] = tile4_axpby(alpha, acc3, beta, C[col + (r + 3u)*N]);
                } else {
                    if (r      != col) C[col + (r     )*N] = alpha*acc0;
                    if (r + 1u != col) C[col + (r + 1u)*N] = alpha*acc1;
                    if (r + 2u != col) C[col + (r + 2u)*N] = alpha*acc2;
                    if (r + 3u != col) C[col + (r + 3u)*N] = alpha*acc3;
                }
            }
        } else {                                    // straddling or tail tile: scalar
            const uint32_t rmax = full_tile ? (r + 4u) : N;
            for (uint32_t row = r; row < rmax; row++) {
                if (!syrk_in_canonical(FILL, row, col)) continue;
                T res = static_cast<T>(0);
                for (uint32_t i = 0; i < K; i++)
                    res += A[row + i*N] * A[col + i*N];
                const uint32_t cidx = col*N + row;
                if constexpr (HAS_BETA) C[cidx] = tile4_axpby(alpha, res, beta, C[cidx]);
                else                    C[cidx] = alpha*res;
                if (FILL == FillMode::Full && row != col) {
                    const uint32_t midx = row*N + col;
                    if constexpr (HAS_BETA) C[midx] = tile4_axpby(alpha, res, beta, C[midx]);
                    else                    C[midx] = alpha*res;
                }
            }
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool HAS_BETA>
__device__ void syrk_tile4_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
{
    if constexpr (tile4_has_vec<T>::value && (N % 4u == 0u)) {
        if (tile4_aligned(A)) {
            syrk_tile4_loop_ct<T, N, K, FILL, HAS_BETA, true>(rank, size, alpha, A, beta, C);
            return;
        }
    }
    syrk_tile4_loop_ct<T, N, K, FILL, HAS_BETA, false>(rank, size, alpha, A, beta, C);
}

// runtime twin of syrk_tile4_loop_ct — the SAME loop skeleton with runtime
// n,k. Keeping the runtime and compile-time bodies identically shaped matters
// beyond perf: warp::syrk (CT) is validated bit-for-bit against the runtime
// block syrk, and matching body shapes keep the compiler's FMA-contraction
// choices identical across the two instantiations.
template <typename T, FillMode FILL, bool HAS_BETA, bool VEC>
__device__ __forceinline__ void syrk_tile4_loop(uint32_t rank, uint32_t size,
                                                uint32_t n, uint32_t k,
                                                T alpha, const T *__restrict__ A,
                                                T beta, T *__restrict__ C)
{
    const uint32_t tpc    = (n + 3u) / 4u;   // 4-row tiles per column of C
    const uint32_t ntiles = tpc * n;
    for (uint32_t t = rank; t < ntiles; t += size) {
        const uint32_t col = t / tpc;
        const uint32_t r   = (t % tpc) * 4u;
        const bool full_tile = (r + 4u <= n);
        bool none_canon, all_canon;
        if constexpr (FILL == FillMode::Upper) {   // canonical: row <= col
            none_canon = (r > col);
            all_canon  = full_tile && (r + 3u <= col);
        } else {                                    // Lower/Full canonical: row >= col
            none_canon = (r + 3u < col);            // max possible row < col
            all_canon  = full_tile && (r >= col);
        }
        if (none_canon) continue;                   // mirror (or nothing) handles it
        if (all_canon) {                            // full 4-row canonical tile
            T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0);
            T acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
            for (uint32_t i = 0; i < k; i++) {
                const T ac = A[col + i*n];
                T a0, a1, a2, a3;
                if constexpr (VEC) {
                    tile4_load(A + (r + i*n), a0, a1, a2, a3);
                } else {
                    a0 = A[r      + i*n]; a1 = A[r + 1u + i*n];
                    a2 = A[r + 2u + i*n]; a3 = A[r + 3u + i*n];
                }
                acc0 += a0 * ac; acc1 += a1 * ac; acc2 += a2 * ac; acc3 += a3 * ac;
            }
            T *__restrict__ c = C + (r + col*n);
            if constexpr (HAS_BETA) {
                c[0] = tile4_axpby(alpha, acc0, beta, c[0]);
                c[1] = tile4_axpby(alpha, acc1, beta, c[1]);
                c[2] = tile4_axpby(alpha, acc2, beta, c[2]);
                c[3] = tile4_axpby(alpha, acc3, beta, c[3]);
            } else {
                c[0] = alpha*acc0; c[1] = alpha*acc1;
                c[2] = alpha*acc2; c[3] = alpha*acc3;
            }
            if constexpr (FILL == FillMode::Full) {
                if constexpr (HAS_BETA) {
                    if (r      != col) C[col + (r     )*n] = tile4_axpby(alpha, acc0, beta, C[col + (r     )*n]);
                    if (r + 1u != col) C[col + (r + 1u)*n] = tile4_axpby(alpha, acc1, beta, C[col + (r + 1u)*n]);
                    if (r + 2u != col) C[col + (r + 2u)*n] = tile4_axpby(alpha, acc2, beta, C[col + (r + 2u)*n]);
                    if (r + 3u != col) C[col + (r + 3u)*n] = tile4_axpby(alpha, acc3, beta, C[col + (r + 3u)*n]);
                } else {
                    if (r      != col) C[col + (r     )*n] = alpha*acc0;
                    if (r + 1u != col) C[col + (r + 1u)*n] = alpha*acc1;
                    if (r + 2u != col) C[col + (r + 2u)*n] = alpha*acc2;
                    if (r + 3u != col) C[col + (r + 3u)*n] = alpha*acc3;
                }
            }
        } else {                                    // straddling or tail tile: scalar
            const uint32_t rmax = full_tile ? (r + 4u) : n;
            for (uint32_t row = r; row < rmax; row++) {
                if (!syrk_in_canonical(FILL, row, col)) continue;
                T res = static_cast<T>(0);
                for (uint32_t i = 0; i < k; i++)
                    res += A[row + i*n] * A[col + i*n];
                const uint32_t cidx = col*n + row;
                if constexpr (HAS_BETA) C[cidx] = tile4_axpby(alpha, res, beta, C[cidx]);
                else                    C[cidx] = alpha*res;
                if (FILL == FillMode::Full && row != col) {
                    const uint32_t midx = row*n + col;
                    if constexpr (HAS_BETA) C[midx] = tile4_axpby(alpha, res, beta, C[midx]);
                    else                    C[midx] = alpha*res;
                }
            }
        }
    }
}

template <typename T, FillMode FILL, bool HAS_BETA>
__device__ void syrk_tile4(uint32_t rank, uint32_t size,
                           uint32_t n, uint32_t k,
                           T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
{
    if constexpr (tile4_has_vec<T>::value) {
        if ((n % 4u == 0u) && tile4_aligned(A)) {
            syrk_tile4_loop<T, FILL, HAS_BETA, true>(rank, size, n, k, alpha, A, beta, C);
            return;
        }
    }
    syrk_tile4_loop<T, FILL, HAS_BETA, false>(rank, size, n, k, alpha, A, beta, C);
}

// ─── syrk core impl: explicit rank/size + layout flags ───────────────────────
// C = alpha * op(A) * op(A)^T + beta * C, C is n x n symmetric.
//   TRANSPOSE == false: op(A) = A   (A is n x k) → C = alpha*A*A^T + beta*C
//   TRANSPOSE == true : op(A) = A^T (A is k x n) → C = alpha*A^T*A + beta*C
// Flat-element parallelism over the n*n grid exactly like gemm_impl: each thread
// owns disjoint output cells, so NO interior barrier is needed (guide §1a
// counter-note). The k-loop runs only in the canonical triangle (the symmetry
// win); off-triangle cells are filled by the mirror write of their transpose.
template <typename T, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syrk_impl(uint32_t rank, uint32_t size,
                          uint32_t n, uint32_t k,
                          T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
{
    if constexpr (!TRANSPOSE && !ROW_MAJOR) {
        syrk_tile4<T, FILL, true>(rank, size, n, k, alpha, A, beta, C);
        return;
    }
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;  // mirror handles it
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            // TRANSPOSE=false: A is n x k → A[row,i], A[col,i].
            // TRANSPOSE=true : A is k x n → A[i,row], A[i,col].
            T ar, ac;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

// beta = 0 form: never reads C.
template <typename T, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syrk_impl(uint32_t rank, uint32_t size,
                          uint32_t n, uint32_t k,
                          T alpha, const T *__restrict__ A, T *__restrict__ C)
{
    if constexpr (!TRANSPOSE && !ROW_MAJOR) {
        syrk_tile4<T, FILL, false>(rank, size, n, k, alpha, A, static_cast<T>(0), C);
        return;
    }
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            T ar, ac;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res;
        }
    }
}

// compile-time impl: N, K as template params so el%N / el/N use magic-number
// multiply instead of MUFU.RCP (mirrors gemm_impl_ct). The no-transpose
// column-major combo routes to the 4-row register-tiled core; TRANSPOSE /
// ROW_MAJOR (strided A rows) keep the original flat loop, byte-identical.
template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syrk_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
{
    if constexpr (!TRANSPOSE && !ROW_MAJOR) {
        syrk_tile4_ct<T, N, K, FILL, true>(rank, size, alpha, A, beta, C);
        return;
    }
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syrk_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, const T *__restrict__ A, T *__restrict__ C)
{
    if constexpr (!TRANSPOSE && !ROW_MAJOR) {
        syrk_tile4_ct<T, N, K, FILL, false>(rank, size, alpha, A, static_cast<T>(0), C);
        return;
    }
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res;
        }
    }
}

// ─── syr2k core impl: explicit rank/size + layout flags ──────────────────────
// C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C, C n x n symmetric.
// Symmetric by construction: cell (row,col) is the symmetric dot
//   Σ_i ( a(row,i)*b(col,i) + b(row,i)*a(col,i) )  [TRANSPOSE=false reading semantics]
// which equals cell (col,row), so the same mirror-write trick applies.
template <typename T, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syr2k_impl(uint32_t rank, uint32_t size,
                           uint32_t n, uint32_t k,
                           T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
{
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            T ar, ac, br, bc;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
                br = ROW_MAJOR ? B[i*n + row] : B[row*k + i];
                bc = ROW_MAJOR ? B[i*n + col] : B[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
                br = ROW_MAJOR ? B[row*k + i] : B[i*n + row];
                bc = ROW_MAJOR ? B[col*k + i] : B[i*n + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

template <typename T, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syr2k_impl(uint32_t rank, uint32_t size,
                           uint32_t n, uint32_t k,
                           T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            T ar, ac, br, bc;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
                br = ROW_MAJOR ? B[i*n + row] : B[row*k + i];
                bc = ROW_MAJOR ? B[i*n + col] : B[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
                br = ROW_MAJOR ? B[row*k + i] : B[i*n + row];
                bc = ROW_MAJOR ? B[col*k + i] : B[i*n + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res;
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syr2k_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
{
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac, br, bc;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
                br = ROW_MAJOR ? B[i*N + row] : B[row*K + i];
                bc = ROW_MAJOR ? B[i*N + col] : B[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
                br = ROW_MAJOR ? B[row*K + i] : B[i*N + row];
                bc = ROW_MAJOR ? B[col*K + i] : B[i*N + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__device__ void syr2k_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac, br, bc;
            if (TRANSPOSE) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
                br = ROW_MAJOR ? B[i*N + row] : B[row*K + i];
                bc = ROW_MAJOR ? B[i*N + col] : B[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
                br = ROW_MAJOR ? B[row*K + i] : B[i*N + row];
                bc = ROW_MAJOR ? B[col*K + i] : B[i*N + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res;
        }
    }
}

// ─── syrk runtime variants ───────────────────────────────────────────────────

/**
 * @brief Symmetric rank-k update: `C = alpha * op(A) * op(A)^T + beta * C` (SYRK).
 *
 * Runtime-size, single-block, flat-element parallelism: each thread owns output
 * cells of the n x n symmetric `C` strided over the block. The length-k dot is
 * computed ONLY in the canonical triangle (the symmetry win, ~half the FLOPs of
 * a GEMM); for `Full` the lower-cell-owning thread also writes the mirror
 * `C[col,row]` (diagonal written once) so each cell is written exactly once and
 * NO interior barrier is needed. `Lower`/`Upper` write only the named triangle
 * and leave the other untouched.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op(A)=A (A is n x k); if true, op(A)=A^T (A is k x n).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (n x k if TRANSPOSE=false, else k x n).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out n x n symmetric result matrix.
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha * A @ A.T + beta * C`;
 * TRANSPOSE=true → `alpha * A.T @ A + beta * C`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syrk(uint32_t n, uint32_t k, T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl<T, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, n, k, alpha, A, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SYRK with implicit `beta = 0`: `C = alpha * op(A) * op(A)^T` (SYRK).
 *
 * Runtime-size overload that overwrites C (C is overwritten, not read), avoiding
 * the `beta * C` term — safe to write into uninitialized scratch. For `Full`,
 * the full symmetric matrix is written; for `Lower`/`Upper`, only the named
 * triangle is written and the other is left untouched. Single-block,
 * flat-element parallelism; no interior barrier.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op(A)=A (A is n x k); if true, op(A)=A^T (A is k x n).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (n x k if TRANSPOSE=false, else k x n).
 * @param C  Output n x n symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha * A @ A.T`; TRANSPOSE=true → `alpha * A.T @ A`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syrk(uint32_t n, uint32_t k, T alpha, const T *__restrict__ A, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl<T, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, n, k, alpha, A, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── syrk compile-time size variants ─────────────────────────────────────────

/**
 * @brief Compile-time-size SYRK: `C = alpha * op(A) * op(A)^T + beta * C` (SYRK).
 *
 * Dimensions are template parameters so the compiler unrolls the inner loop and
 * replaces the `el % N` / `el / N` index math with magic-number multiplies.
 * Single-block, flat-element parallelism, symmetry-exploiting (canonical
 * triangle + mirror write), no interior barrier. C is read; caller must
 * initialize it.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op(A)=A (A is N x K); if true, op(A)=A^T (A is K x N).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (N x K if TRANSPOSE=false, else K x N).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out N x N symmetric result matrix.
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha * A @ A.T + beta * C`;
 * TRANSPOSE=true → `alpha * A.T @ A + beta * C`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syrk(T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Compile-time-size SYRK with implicit `beta = 0`: `C = alpha * op(A) * op(A)^T`.
 *
 * Compile-time-size overload that overwrites C (C is overwritten, not read).
 * Single-block, flat-element parallelism, symmetry-exploiting; no interior
 * barrier. Safe to write into uninitialized scratch.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op(A)=A (A is N x K); if true, op(A)=A^T (A is K x N).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (N x K if TRANSPOSE=false, else K x N).
 * @param C  Output N x N symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha * A @ A.T`; TRANSPOSE=true → `alpha * A.T @ A`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syrk(T alpha, const T *__restrict__ A, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── syr2k runtime variants ──────────────────────────────────────────────────

/**
 * @brief Symmetric rank-2k update: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C` (SYR2K).
 *
 * Runtime-size, single-block, flat-element parallelism. The result is symmetric
 * by construction; the length-k dot is computed only in the canonical triangle
 * and (for `Full`) mirrored, so each cell is written once and NO interior
 * barrier is needed. `Lower`/`Upper` write only the named triangle.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op = identity (A,B are n x k); if true, op = transpose (A,B are k x n).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (n x k if TRANSPOSE=false, else k x n).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out n x n symmetric result matrix.
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha*(A@B.T + B@A.T) + beta*C`;
 * TRANSPOSE=true → `alpha*(A.T@B + B.T@A) + beta*C`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syr2k(uint32_t n, uint32_t k, T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl<T, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, n, k, alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief SYR2K with implicit `beta = 0`: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T)`.
 *
 * Runtime-size overload that overwrites C (C is overwritten, not read). Safe to
 * write into uninitialized scratch. Single-block, flat-element parallelism,
 * symmetry-exploiting; no interior barrier.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op = identity (A,B are n x k); if true, op = transpose (A,B are k x n).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (n x k if TRANSPOSE=false, else k x n).
 * @param C  Output n x n symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha*(A@B.T + B@A.T)`;
 * TRANSPOSE=true → `alpha*(A.T@B + B.T@A)`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syr2k(uint32_t n, uint32_t k, T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl<T, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, n, k, alpha, A, B, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── syr2k compile-time size variants ────────────────────────────────────────

/**
 * @brief Compile-time-size SYR2K: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C`.
 *
 * Dimensions are template parameters so the inner loop unrolls and `el % N` /
 * `el / N` become magic-number multiplies. Single-block, flat-element
 * parallelism, symmetry-exploiting; no interior barrier. C is read; caller must
 * initialize it.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op = identity (A,B are N x K); if true, op = transpose (A,B are K x N).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (N x K if TRANSPOSE=false, else K x N).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out N x N symmetric result matrix.
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha*(A@B.T + B@A.T) + beta*C`;
 * TRANSPOSE=true → `alpha*(A.T@B + B.T@A) + beta*C`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syr2k(T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Compile-time-size SYR2K with implicit `beta = 0`: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T)`.
 *
 * Compile-time-size overload that overwrites C (C is overwritten, not read).
 * Safe to write into uninitialized scratch. Single-block, flat-element
 * parallelism, symmetry-exploiting; no interior barrier.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANSPOSE  If false, op = identity (A,B are N x K); if true, op = transpose (A,B are K x N).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (N x K if TRANSPOSE=false, else K x N).
 * @param C  Output N x N symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANSPOSE=false → `alpha*(A@B.T + B@A.T)`;
 * TRANSPOSE=true → `alpha*(A.T@B + B.T@A)`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void syr2k(T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, B, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    // Single-warp SYRK / SYR2K: one 32-lane warp owns the symmetric output via the
    // SAME validated `syrk_impl_ct` / `syr2k_impl_ct` flat-element kernels, just
    // dispatched with (lane, 32) instead of (rank, blockDim). Each output element is
    // written once (no cross-lane reduction — the K contraction is a per-lane serial
    // loop), so this is bit-identical to the block form restricted to one warp. For
    // warp-per-problem normal-equation builds (e.g. HJCD's JᵀJ). Full 32 lanes
    // required; independent warps may run distinct problems. No `__syncwarp` needed
    // (no inter-lane dependency); compile-time size only, mirroring `warp::gemm`.

    /**
     * @brief Single-warp SYRK `C = alpha*op(A)*op(A)ᵀ + beta*C` (compile-time size).
     * @see ::syrk  (block form; identical math, `(lane,32)` element striping)
     */
    template <typename T, uint32_t N, uint32_t K,
              FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void syrk(T alpha, const T *__restrict__ A, T beta, T *__restrict__ C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        syrk_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(lane, 32, alpha, A, beta, C);
    }

    /**
     * @brief Single-warp SYRK with implicit `beta = 0`: `C = alpha*op(A)*op(A)ᵀ` (overwrite).
     * @see ::syrk
     */
    template <typename T, uint32_t N, uint32_t K,
              FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void syrk(T alpha, const T *__restrict__ A, T *__restrict__ C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        syrk_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(lane, 32, alpha, A, C);
    }

    /**
     * @brief Single-warp SYR2K `C = alpha*(op(A)op(B)ᵀ + op(B)op(A)ᵀ) + beta*C` (compile-time size).
     * @see ::syr2k
     */
    template <typename T, uint32_t N, uint32_t K,
              FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void syr2k(T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        syr2k_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(lane, 32, alpha, A, B, beta, C);
    }

    /**
     * @brief Single-warp SYR2K with implicit `beta = 0` (overwrite).
     * @see ::syr2k
     */
    template <typename T, uint32_t N, uint32_t K,
              FillMode FILL = FillMode::Full, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void syr2k(T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        syr2k_impl_ct<T, N, K, FILL, TRANSPOSE, ROW_MAJOR>(lane, 32, alpha, A, B, C);
    }
}
