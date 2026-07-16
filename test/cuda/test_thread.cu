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
// DTYPE: templated on the scalar type; <dtype> = f32|f64 picks the instantiation.
// Operand .bin files are always float32 (what the Python harness writes); the
// driver widens them to T on load, so the f64 path exercises the tier's "BOTH
// dtypes" register-residency claim (see CLAUDE.md) over the same inputs.
//
// FLAGS: trsv and gemv carry their compile-time flag surface so the sweep hits it
// rather than trusting the thread:: overloads to forward it correctly —
//   gemv <trans> <rowmajor>              (each 0/1)
//   trsv <lower> <unit>  <trans>         (each 0/1)
// both are instantiated for the block1 oracle too, so the bit-identical check
// covers every flag combination.
//
// Usage: ./test_thread <op> <model> <dtype> <N> <P> [flags...] <files...>
//   ops:    dot gemv gemm potrf trsv posv potrs
//   models: thread block1     dtype: f32 f64

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass.cuh"

#define TPB 64   // threads per block for the `thread` model (ragged tail when P%TPB)

using glass::FillMode;
using glass::Diag;

// ─── dtype-generic I/O (helpers.cuh is float32-only) ─────────────────────────
// Read a float32 .bin and widen to T; print with round-trip precision so that
// two bit-equal T values always render to identical text (keeps the thread vs
// block1 comparison exact even in f64).

template <typename T>
static T* read_dev(const char* path, int n) {
    float* h = read_host_vec(path, n);
    T* hT = (T*)malloc(n * sizeof(T));
    for (int i = 0; i < n; i++) hT[i] = (T)h[i];
    free(h);
    T* d; cudaMalloc(&d, n * sizeof(T));
    cudaMemcpy(d, hT, n * sizeof(T), cudaMemcpyHostToDevice);
    free(hT);
    return d;
}
template <typename T>
static T* alloc_dev(int n) { T* d; cudaMalloc(&d, n * sizeof(T)); cudaMemset(d, 0, n * sizeof(T)); return d; }

template <typename T> __global__ void print_kernelT(const T* d, int n) {
    for (int i = 0; i < n; i++) {
        if constexpr (sizeof(T) == 8) printf("%.17g", (double)d[i]);
        else                          printf("%.9g",  (double)d[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
}
template <typename T> static void print_dev(const T* d, int n) {
    print_kernelT<T><<<1,1>>>(d, n); cudaDeviceSynchronize();
}

// ─── THREAD model: thread p owns problem p, operands thread-local ─────────────

template <typename T, int N> __global__ void kt_dot(int P, T* x, T* y, T* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    out[p] = glass::thread::dot<T, N>(x + (size_t)p*N, y + (size_t)p*N);
}
template <typename T, int N, bool TR, bool RM> __global__ void kt_gemv(int P, T alpha, T* A, T* x, T* y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], xv[N], yv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::gemv<T, N, N, TR, RM>(alpha, a, xv, yv);
    for (int i = 0; i < N;   i++) y[(size_t)p*N + i] = yv[i];
}
template <typename T, int N> __global__ void kt_gemm(int P, T alpha, T* A, T* B, T* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], b[N*N], c[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    for (int i = 0; i < N*N; i++) b[i] = B[(size_t)p*N*N + i];
    glass::thread::gemm<T, N, N, N>(alpha, a, b, c);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N + i] = c[i];
}
template <typename T, int N> __global__ void kt_potrf(int P, T* A) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    glass::thread::potrf<T, N>(a);
    for (int i = 0; i < N*N; i++) A[(size_t)p*N*N + i] = a[i];
}
template <typename T, int N, FillMode F, Diag D, bool TR> __global__ void kt_trsv(int P, T* A, T* x) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], xv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::trsv<T, N, F, D, TR>(a, xv);
    for (int i = 0; i < N;   i++) x[(size_t)p*N + i] = xv[i];
}
template <typename T, int N> __global__ void kt_posv(int P, T* A, T* b) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], bv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[(size_t)p*N + i];
    glass::thread::posv<T, N>(a, bv);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = bv[i];
}
template <typename T, int N> __global__ void kt_potrs(int P, T* L, T* b) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T l[N*N], bv[N];
    for (int i = 0; i < N*N; i++) l[i]  = L[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[(size_t)p*N + i];
    glass::thread::potrs<T, N>(l, bv);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = bv[i];
}

// ─── BLOCK-1 oracle: block p owns problem p, launched with ONE thread ─────────

template <typename T, int N> __global__ void kb1_dot(T* x, T* y, T* out) {
    int p = blockIdx.x;
    glass::dot<T, N>(x + (size_t)p*N, y + (size_t)p*N);   // destructive: result in y[0]
    out[p] = y[(size_t)p*N];
}
template <typename T, int N, bool TR, bool RM> __global__ void kb1_gemv(T alpha, T* A, T* x, T* y) {
    int p = blockIdx.x;
    glass::gemv<T, N, N, TR, RM>(alpha, A + (size_t)p*N*N, x + (size_t)p*N, y + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_gemm(T alpha, T* A, T* B, T* C) {
    int p = blockIdx.x;
    glass::gemm<T, N, N, N>(alpha, A + (size_t)p*N*N, B + (size_t)p*N*N, C + (size_t)p*N*N);
}
template <typename T, int N> __global__ void kb1_potrf(T* A) {
    int p = blockIdx.x;
    glass::potrf<T, N>(A + (size_t)p*N*N);
}
template <typename T, int N, FillMode F, Diag D, bool TR> __global__ void kb1_trsv(T* A, T* x) {
    int p = blockIdx.x;
    glass::trsv<T, N, F, D, TR>(A + (size_t)p*N*N, x + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_posv(T* A, T* b) {
    int p = blockIdx.x;
    glass::posv<T, N>(A + (size_t)p*N*N, b + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_potrs(T* L, T* b) {
    int p = blockIdx.x;
    glass::potrs<T, N>(L + (size_t)p*N*N, b + (size_t)p*N);
}

// ─── dispatch ────────────────────────────────────────────────────────────────

static int  g_P;
static bool g_thread;          // true => thread model, false => block1 oracle
static int  g_f0, g_f1, g_f2;  // op-specific compile-time flags (parsed in main)

// grid/block for the model under test
static inline dim3 grid()  { return g_thread ? dim3((g_P + TPB - 1) / TPB) : dim3(g_P); }
static inline dim3 block() { return g_thread ? dim3(TPB) : dim3(1); }

// Runtime flag ints -> compile-time template args, instantiating both models.
template <typename T, int N, bool TR, bool RM>
static void launch_gemv(T alpha, T* A, T* x, T* y) {
    if (g_thread) kt_gemv<T,N,TR,RM><<<grid(),block()>>>(g_P, alpha, A, x, y);
    else          kb1_gemv<T,N,TR,RM><<<grid(),block()>>>(alpha, A, x, y);
}
template <typename T, int N>
static void dispatch_gemv(T alpha, T* A, T* x, T* y, bool tr, bool rm) {
    if (tr)  { if (rm) launch_gemv<T,N,true ,true >(alpha,A,x,y); else launch_gemv<T,N,true ,false>(alpha,A,x,y); }
    else     { if (rm) launch_gemv<T,N,false,true >(alpha,A,x,y); else launch_gemv<T,N,false,false>(alpha,A,x,y); }
}

template <typename T, int N, FillMode F, Diag D, bool TR>
static void launch_trsv(T* A, T* x) {
    if (g_thread) kt_trsv<T,N,F,D,TR><<<grid(),block()>>>(g_P, A, x);
    else          kb1_trsv<T,N,F,D,TR><<<grid(),block()>>>(A, x);
}
template <typename T, int N, FillMode F, Diag D>
static void dispatch_trsv_t(T* A, T* x, bool tr) {
    if (tr) launch_trsv<T,N,F,D,true>(A,x); else launch_trsv<T,N,F,D,false>(A,x);
}
template <typename T, int N>
static void dispatch_trsv(T* A, T* x, bool lower, bool unit, bool tr) {
    if (lower) {
        if (unit) dispatch_trsv_t<T,N,FillMode::Lower,Diag::Unit   >(A,x,tr);
        else      dispatch_trsv_t<T,N,FillMode::Lower,Diag::NonUnit>(A,x,tr);
    } else {
        if (unit) dispatch_trsv_t<T,N,FillMode::Upper,Diag::Unit   >(A,x,tr);
        else      dispatch_trsv_t<T,N,FillMode::Upper,Diag::NonUnit>(A,x,tr);
    }
}

template <typename T, int N>
static void dispatch(const char* op, T* A, T* B, T* x, T* y, T* out)
{
    const T alpha = (T)1;
    if (!strcmp(op, "dot")) {
        if (g_thread) kt_dot<T,N><<<grid(),block()>>>(g_P, x, y, out);
        else          kb1_dot<T,N><<<grid(),block()>>>(x, y, out);
    } else if (!strcmp(op, "gemv")) {
        dispatch_gemv<T,N>(alpha, A, x, y, g_f0, g_f1);
    } else if (!strcmp(op, "gemm")) {
        if (g_thread) kt_gemm<T,N><<<grid(),block()>>>(g_P, alpha, A, B, y);
        else          kb1_gemm<T,N><<<grid(),block()>>>(alpha, A, B, y);
    } else if (!strcmp(op, "potrf")) {
        if (g_thread) kt_potrf<T,N><<<grid(),block()>>>(g_P, A);
        else          kb1_potrf<T,N><<<grid(),block()>>>(A);
    } else if (!strcmp(op, "trsv")) {
        dispatch_trsv<T,N>(A, x, g_f0, g_f1, g_f2);
    } else if (!strcmp(op, "posv")) {
        if (g_thread) kt_posv<T,N><<<grid(),block()>>>(g_P, A, x);
        else          kb1_posv<T,N><<<grid(),block()>>>(A, x);
    } else if (!strcmp(op, "potrs")) {
        if (g_thread) kt_potrs<T,N><<<grid(),block()>>>(g_P, A, x);
        else          kb1_potrs<T,N><<<grid(),block()>>>(A, x);
    } else {
        fprintf(stderr, "unknown op %s\n", op); exit(1);
    }
}

template <typename T>
static int run_all(const char* op, char** argv, int f)
{
    const int N = atoi(argv[4]);
    const int mm = g_P * N * N, vv = g_P * N;
    T *A = nullptr, *B = nullptr, *x = nullptr, *y = nullptr, *out = nullptr;

    // Operand files, in the order each op consumes them.
    if (!strcmp(op, "dot")) {
        x = read_dev<T>(argv[f++], vv);
        y = read_dev<T>(argv[f++], vv);
        out = alloc_dev<T>(g_P);
    } else if (!strcmp(op, "gemv")) {
        A = read_dev<T>(argv[f++], mm);
        x = read_dev<T>(argv[f++], vv);
        y = alloc_dev<T>(vv);
    } else if (!strcmp(op, "gemm")) {
        A = read_dev<T>(argv[f++], mm);
        B = read_dev<T>(argv[f++], mm);
        y = alloc_dev<T>(mm);   // C reuses the `y` slot
    } else if (!strcmp(op, "potrf")) {
        A = read_dev<T>(argv[f++], mm);
    } else if (!strcmp(op, "trsv") || !strcmp(op, "posv") || !strcmp(op, "potrs")) {
        A = read_dev<T>(argv[f++], mm);   // trsv: triangular A; posv: SPD A; potrs: lower factor L
        x = read_dev<T>(argv[f++], vv);
    }

    switch (N) {
        case 4: dispatch<T,4>(op, A, B, x, y, out); break;
        case 5: dispatch<T,5>(op, A, B, x, y, out); break;
        case 6: dispatch<T,6>(op, A, B, x, y, out); break;
        case 7: dispatch<T,7>(op, A, B, x, y, out); break;
        case 8: dispatch<T,8>(op, A, B, x, y, out); break;
        default: fprintf(stderr, "N=%d not instantiated (want 4..8)\n", N); return 1;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    // Emit the op's result buffer.
    if      (!strcmp(op, "dot"))    print_dev<T>(out, g_P);
    else if (!strcmp(op, "gemv"))   print_dev<T>(y, vv);
    else if (!strcmp(op, "gemm"))   print_dev<T>(y, mm);
    else if (!strcmp(op, "potrf"))  print_dev<T>(A, mm);
    else                            print_dev<T>(x, vv);  // trsv, posv, potrs
    cudaDeviceSynchronize();
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 6) { fprintf(stderr, "usage: %s <op> <model> <dtype> <N> <P> [flags...] <files...>\n", argv[0]); return 1; }
    const char* op    = argv[1];
    const char* model = argv[2];
    const char* dtype = argv[3];
    // argv[4] = N (parsed in run_all), argv[5] = P
    g_P = atoi(argv[5]);
    g_thread = !strcmp(model, "thread");
    if (!g_thread && strcmp(model, "block1")) { fprintf(stderr, "model must be thread|block1\n"); return 1; }

    // Op-specific compile-time flags follow P; files follow the flags.
    int f = 6;
    g_f0 = g_f1 = g_f2 = 0;
    if (!strcmp(op, "gemv")) {                       // <trans> <rowmajor>
        g_f0 = atoi(argv[f++]); g_f1 = atoi(argv[f++]);
    } else if (!strcmp(op, "trsv")) {                // <lower> <unit> <trans>
        g_f0 = atoi(argv[f++]); g_f1 = atoi(argv[f++]); g_f2 = atoi(argv[f++]);
    }

    if      (!strcmp(dtype, "f32")) return run_all<float >(op, argv, f);
    else if (!strcmp(dtype, "f64")) return run_all<double>(op, argv, f);
    fprintf(stderr, "dtype must be f32|f64\n");
    return 1;
}
