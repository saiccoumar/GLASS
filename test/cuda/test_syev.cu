// test_syev.cu — glass::syev (cyclic-Jacobi symmetric eigendecomposition) and
// glass::eig_clamp (eigenvalue clamp + reconstruct) drivers.
//
// Ops (argv[2] is the usual run_op version slot, ignored):
//   syev      simple <n> <threads> <A.bin>        → W ascending (line 1),
//                                                    V col-major (line 2)
//   syev_ct   simple <n> <threads> <A.bin>        → same, compile-time-N overload
//                                                    (n restricted to SYEV_SIZES)
//   eig_clamp simple <n> <threads> <eps> <A.bin>  → clamped A, n*n col-major (line 1)
//
// A.bin : n*n float32 (column-major, symmetric). Scratch is dynamic shared
// memory sized by glass::syev_scratch_bytes / glass::eig_clamp_scratch_bytes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

__global__ void k_syev(uint32_t n, const float* A, float* W, float* V) {
    extern __shared__ float s[];
    glass::syev<float>(n, A, W, V, s);
}

template <uint32_t N>
__global__ void k_syev_ct(const float* A, float* W, float* V) {
    extern __shared__ float s[];
    glass::syev<float, N>(A, W, V, s);
}

__global__ void k_eig_clamp(uint32_t n, float* A, float eps) {
    extern __shared__ float s[];
    glass::eig_clamp<float>(n, A, eps, s);
}

#define SYEV_SIZES(F) F(1) F(2) F(3) F(4) F(7) F(12) F(16) F(32)

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "usage: %s syev|syev_ct <version> <n> <threads> <A.bin>\n"
            "       %s eig_clamp    <version> <n> <threads> <eps> <A.bin>\n",
            argv[0], argv[0]);
        return 1;
    }
    const char* op = argv[1];               // argv[2] = version ("simple"), unused
    int n       = atoi(argv[3]);
    int threads = atoi(argv[4]);

    if (strcmp(op, "syev") == 0 || strcmp(op, "syev_ct") == 0) {
        float* dA = read_device_vec(argv[5], n * n);
        float* dW = alloc_device_vec(n);
        float* dV = alloc_device_vec(n * n);
        int sm = (int)glass::syev_scratch_bytes<float>((uint32_t)n);
        if (strcmp(op, "syev") == 0) {
            k_syev<<<1, threads, sm>>>((uint32_t)n, dA, dW, dV);
        } else {
            bool ok = false;
            #define DN(N_) if (!ok && n == N_) { k_syev_ct<N_><<<1, threads, sm>>>(dA, dW, dV); ok = true; }
            SYEV_SIZES(DN)
            #undef DN
            if (!ok) { fprintf(stderr, "syev_ct: unsupported n=%d\n", n); return 2; }
        }
        cudaDeviceSynchronize();
        print_device_vec(dW, n);
        print_device_vec(dV, n * n);
    } else if (strcmp(op, "eig_clamp") == 0) {
        if (argc < 7) { fprintf(stderr, "eig_clamp needs <eps> <A.bin>\n"); return 1; }
        float eps = (float)atof(argv[5]);
        float* dA = read_device_vec(argv[6], n * n);
        int sm = (int)glass::eig_clamp_scratch_bytes<float>((uint32_t)n);
        k_eig_clamp<<<1, threads, sm>>>((uint32_t)n, dA, eps);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);
    } else {
        fprintf(stderr, "unknown op %s\n", op);
        return 1;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 3; }
    return 0;
}
