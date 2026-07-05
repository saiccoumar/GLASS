#pragma once
#include <cstdint>

/**
 * @file barrier.cuh
 * @brief Barrier policy for the shared `*_impl` bodies (cgrps dedup + TRAILING_SYNC).
 *
 * Every public single-block op is written ONCE as a `*_impl(Bar bar, ...)` body
 * templated on a *barrier policy* that carries the thread `rank()`/`size()` and a
 * `sync()`. The plain `glass::` surface passes `BlockBarrier` (threadIdx/blockDim
 * + `__syncthreads()`); the `glass::cgrps::` surface passes a `GroupBarrier`
 * (cooperative-groups handle + `cooperative_groups::sync`, defined in
 * `src/cgrps/`). Routing BOTH the internal and the trailing barrier through
 * `bar.sync()` is what lets one body serve both surfaces — that barrier
 * primitive is the only thing that ever differed between them.
 *
 * `BlockBarrier` names no cooperative-groups type, so `glass.cuh` stays
 * dependency-free; the `GroupBarrier` twin is only compiled when a caller
 * includes `<cooperative_groups.h>` via `glass-cgrps.cuh`.
 *
 * Uniformity rule (project-wide): every public op takes `bool TRAILING_SYNC=true`
 * and ends on `if constexpr (TRAILING_SYNC) bar.sync();`, so the result is valid
 * for ALL threads by default. Callers that own the following barrier pass
 * `false` to elide it. For ops that already ended on a barrier this is
 * byte-identical; for the elementwise/reduce-tail ops it adds a strictly-safer
 * trailing barrier.
 */
struct BlockBarrier {
    __device__ __forceinline__ uint32_t rank() const {
        return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    }
    __device__ __forceinline__ uint32_t size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// ct_size — compile-time size carrier for the factor/solve `*_impl` bodies.
//
// The shared impls take their dimension through a deduced `SizeT` template
// parameter instead of a hard `uint32_t`. The runtime public overloads pass a
// plain `uint32_t` (unchanged behavior); the compile-time-size overloads pass
// `ct_size<N>{}`, whose constexpr conversion makes every use of the dimension a
// compile-time constant after inlining — so nvcc unrolls the loops and
// strength-reduces the `%`/`/` indexing with NO body duplication (the same
// effect `gemm_impl_ct` gets from its dedicated body). A self-made carrier
// rather than std::integral_constant so no system header lands inside
// `namespace glass` (this file is included inside the namespace by glass.cuh).
// Defined here (not in trsv.cuh, its original home) because barrier.cuh is the
// FIRST header glass.cuh pulls in and the one downstream vendoring already
// carries — every factor/solve user (trsv/inv/syev/ldlt/…) sees it regardless
// of which subset of headers a consumer embeds.
// ─────────────────────────────────────────────────────────────────────────────
template <uint32_t V>
struct ct_size {
    __host__ __device__ constexpr operator uint32_t() const { return V; }
};
