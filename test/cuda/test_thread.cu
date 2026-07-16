// test_thread.cu — driver for the glass::thread:: surface (one problem per THREAD).
//
// Two launch models over the SAME inputs, selected by the <model> argument:
//
//   thread  <<<ceil(P/TPB), TPB>>>  — thread p owns problem p; operands are copied
//                                     into thread-local arrays, computed on, copied
//                                     back. This is the tier under test.
//   block1  <<<P, 1>>>              — block p owns problem p but runs ONE thread, so
//                                     the block-scoped glass:: op degenerates to the
//                                     same sequential algorithm. This is the ORACLE.
//
// Why block1 is the oracle: GLASS guarantees thread-count invariance ("identical
// output at 1 thread, 32, a partial warp, or many warps"), and every thread:: op
// delegates to the same *_impl body via ThreadBarrier{rank=0,size=1,no-op sync}.
// So thread == block1 must hold BIT-FOR-BIT, not just to a tolerance — the two
// run the identical instruction sequence over the identical operand order. A
// mismatch means one of the two is wrong. (`dot` is the deliberate exception: the
// block-scoped dot reduces with a halving TREE, thread::dot accumulates serially,
// so they agree only to float tolerance. See test_thread.py.)
//
// P is deliberately >32 in the pytest driver so problems span multiple warps and
// several blocks with a RAGGED tail — the configuration that catches a stray
// block-wide __syncthreads() inside a thread:: op (divergent participation once
// the tail block's out-of-range threads have returned ⇒ UB/hang).
//
// Usage: ./test_thread <op> <model> <N> <P> <files...>
//   ops:    dot gemv gemv_t gemm potrf trsv posv
//   models: thread block1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass.cuh"

#define TPB 64   // threads per block for the `thread` model (ragged tail when P%TPB)

// ─── THREAD model: thread p owns problem p, operands thread-local ─────────────

template <int N> __global__ void kt_dot(int P, float* x, float* y, float* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    out[p] = glass::thread::dot<float, N>(x + (size_t)p*N, y + (size_t)p*N);
}
template <int N> __global__ void kt_gemv(int P, float alpha, float* A, float* x, float* y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    float a[N*N], xv[N], yv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::gemv<float, N, N>(alpha, a, xv, yv);
    for (int i = 0; i < N;   i++) y[(size_t)p*N + i] = yv[i];
}
template <int N> __global__ void kt_gemv_t(int P, float alpha, float* A, float* x, float* y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    float a[N*N], xv[N], yv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::gemv<float, N, N, /*TRANSPOSE=*/true>(alpha, a, xv, yv);
    for (int i = 0; i < N;   i++) y[(size_t)p*N + i] = yv[i];
}
template <int N> __global__ void kt_gemm(int P, float alpha, float* A, float* B, float* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    float a[N*N], b[N*N], c[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    for (int i = 0; i < N*N; i++) b[i] = B[(size_t)p*N*N + i];
    glass::thread::gemm<float, N, N, N>(alpha, a, b, c);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N + i] = c[i];
}
template <int N> __global__ void kt_potrf(int P, float* A) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    float a[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    glass::thread::potrf<float, N>(a);
    for (int i = 0; i < N*N; i++) A[(size_t)p*N*N + i] = a[i];
}
template <int N> __global__ void kt_trsv(int P, float* A, float* x) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    float a[N*N], xv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::trsv<float, N>(a, xv);
    for (int i = 0; i < N;   i++) x[(size_t)p*N + i] = xv[i];
}
template <int N> __global__ void kt_posv(int P, float* A, float* b) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    float a[N*N], bv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[(size_t)p*N + i];
    glass::thread::posv<float, N>(a, bv);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = bv[i];
}

// ─── BLOCK-1 oracle: block p owns problem p, launched with ONE thread ─────────

template <int N> __global__ void kb1_dot(float* x, float* y, float* out) {
    int p = blockIdx.x;
    glass::dot<float, N>(x + (size_t)p*N, y + (size_t)p*N);   // destructive: result in y[0]
    out[p] = y[(size_t)p*N];
}
template <int N> __global__ void kb1_gemv(float alpha, float* A, float* x, float* y) {
    int p = blockIdx.x;
    glass::gemv<float, N, N>(alpha, A + (size_t)p*N*N, x + (size_t)p*N, y + (size_t)p*N);
}
template <int N> __global__ void kb1_gemv_t(float alpha, float* A, float* x, float* y) {
    int p = blockIdx.x;
    glass::gemv<float, N, N, /*TRANSPOSE=*/true>(alpha, A + (size_t)p*N*N, x + (size_t)p*N, y + (size_t)p*N);
}
template <int N> __global__ void kb1_gemm(float alpha, float* A, float* B, float* C) {
    int p = blockIdx.x;
    glass::gemm<float, N, N, N>(alpha, A + (size_t)p*N*N, B + (size_t)p*N*N, C + (size_t)p*N*N);
}
template <int N> __global__ void kb1_potrf(float* A) {
    int p = blockIdx.x;
    glass::potrf<float, N>(A + (size_t)p*N*N);
}
template <int N> __global__ void kb1_trsv(float* A, float* x) {
    int p = blockIdx.x;
    glass::trsv<float, N>(A + (size_t)p*N*N, x + (size_t)p*N);
}
template <int N> __global__ void kb1_posv(float* A, float* b) {
    int p = blockIdx.x;
    glass::posv<float, N>(A + (size_t)p*N*N, b + (size_t)p*N);
}

// ─── dispatch ────────────────────────────────────────────────────────────────

static int  g_P;
static bool g_thread;   // true => thread model, false => block1 oracle

// grid/block for the model under test
static inline dim3 grid()  { return g_thread ? dim3((g_P + TPB - 1) / TPB) : dim3(g_P); }
static inline dim3 block() { return g_thread ? dim3(TPB) : dim3(1); }

template <int N>
static void run(const char* op, float* A, float* B, float* x, float* y, float* out)
{
    const float alpha = 1.0f;
    if (!strcmp(op, "dot")) {
        if (g_thread) kt_dot<N><<<grid(), block()>>>(g_P, x, y, out);
        else          kb1_dot<N><<<grid(), block()>>>(x, y, out);
    } else if (!strcmp(op, "gemv")) {
        if (g_thread) kt_gemv<N><<<grid(), block()>>>(g_P, alpha, A, x, y);
        else          kb1_gemv<N><<<grid(), block()>>>(alpha, A, x, y);
    } else if (!strcmp(op, "gemv_t")) {
        if (g_thread) kt_gemv_t<N><<<grid(), block()>>>(g_P, alpha, A, x, y);
        else          kb1_gemv_t<N><<<grid(), block()>>>(alpha, A, x, y);
    } else if (!strcmp(op, "gemm")) {
        if (g_thread) kt_gemm<N><<<grid(), block()>>>(g_P, alpha, A, B, y);
        else          kb1_gemm<N><<<grid(), block()>>>(alpha, A, B, y);
    } else if (!strcmp(op, "potrf")) {
        if (g_thread) kt_potrf<N><<<grid(), block()>>>(g_P, A);
        else          kb1_potrf<N><<<grid(), block()>>>(A);
    } else if (!strcmp(op, "trsv")) {
        if (g_thread) kt_trsv<N><<<grid(), block()>>>(g_P, A, x);
        else          kb1_trsv<N><<<grid(), block()>>>(A, x);
    } else if (!strcmp(op, "posv")) {
        if (g_thread) kt_posv<N><<<grid(), block()>>>(g_P, A, x);
        else          kb1_posv<N><<<grid(), block()>>>(A, x);
    } else {
        fprintf(stderr, "unknown op %s\n", op); exit(1);
    }
}

int main(int argc, char** argv)
{
    if (argc < 5) { fprintf(stderr, "usage: %s <op> <model> <N> <P> <files...>\n", argv[0]); return 1; }
    const char* op    = argv[1];
    const char* model = argv[2];
    int N = atoi(argv[3]);
    g_P   = atoi(argv[4]);
    g_thread = !strcmp(model, "thread");
    if (!g_thread && strcmp(model, "block1")) { fprintf(stderr, "model must be thread|block1\n"); return 1; }

    const int mm = g_P * N * N, vv = g_P * N;
    float *A = nullptr, *B = nullptr, *x = nullptr, *y = nullptr, *out = nullptr;
    int f = 5;

    // Operand files, in the order each op consumes them.
    if (!strcmp(op, "dot")) {
        x = read_device_vec(argv[f++], vv);
        y = read_device_vec(argv[f++], vv);
        cudaMalloc(&out, g_P * sizeof(float));
    } else if (!strcmp(op, "gemv") || !strcmp(op, "gemv_t")) {
        A = read_device_vec(argv[f++], mm);
        x = read_device_vec(argv[f++], vv);
        cudaMalloc(&y, vv * sizeof(float));
    } else if (!strcmp(op, "gemm")) {
        A = read_device_vec(argv[f++], mm);
        B = read_device_vec(argv[f++], mm);
        cudaMalloc(&y, mm * sizeof(float));   // C reuses the `y` slot
    } else if (!strcmp(op, "potrf")) {
        A = read_device_vec(argv[f++], mm);
    } else if (!strcmp(op, "trsv") || !strcmp(op, "posv")) {
        A = read_device_vec(argv[f++], mm);
        x = read_device_vec(argv[f++], vv);
    }

    switch (N) {
        case 4: run<4>(op, A, B, x, y, out); break;
        case 5: run<5>(op, A, B, x, y, out); break;
        case 6: run<6>(op, A, B, x, y, out); break;
        case 7: run<7>(op, A, B, x, y, out); break;
        case 8: run<8>(op, A, B, x, y, out); break;
        default: fprintf(stderr, "N=%d not instantiated (want 4..8)\n", N); return 1;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    // Emit the op's result buffer.
    if      (!strcmp(op, "dot"))                              print_device_vec(out, g_P);
    else if (!strcmp(op, "gemv") || !strcmp(op, "gemv_t"))    print_device_vec(y, vv);
    else if (!strcmp(op, "gemm"))                             print_device_vec(y, mm);
    else if (!strcmp(op, "potrf"))                            print_device_vec(A, mm);
    else                                                      print_device_vec(x, vv);  // trsv, posv
    cudaDeviceSynchronize();
    return 0;
}
