"""test_syev.py — glass::syev (cyclic-Jacobi symmetric eigendecomposition) and
glass::eig_clamp (eigenvalue clamp + reconstruct) vs numpy.

Coverage: syev vs np.linalg.eigh across sizes (eigenvalues, orthogonality,
V diag(W) Vᵀ reconstruction), per-column eigenvector match up to SIGN on a
well-separated spectrum, the compile-time-N overload, repeated eigenvalues
(identity + rank-1), ill-conditioned SPD (cond=1e6), negative-definite and
indefinite inputs, eig_clamp SPD-ness + numpy-clamped reconstruction, and the
THREAD_SWEEP byte-identical invariance rule.
"""
import numpy as np
import pytest

from conftest import run_op, THREAD_SWEEP, make_spd

SIZES = [1, 2, 3, 4, 7, 12, 16, 32]


def make_sym(n, seed=0):
    """Random symmetric n x n (generally indefinite), float32."""
    G = np.random.default_rng(seed).standard_normal((n, n))
    return ((G + G.T) / 2).astype(np.float32)


def make_sym_spectrum(n, eigs, seed=0):
    """Symmetric matrix with a prescribed spectrum: Q diag(eigs) Qᵀ."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q @ np.diag(np.asarray(eigs, dtype=np.float64)) @ Q.T).astype(np.float32)


def run_syev(bins, n, threads, A, op="syev"):
    out = run_op(bins["syev"], op, "simple", args=[n, threads],
                 inputs=[np.asfortranarray(A).ravel(order="F")])
    W, V = out[0], out[1].reshape(n, n, order="F")
    return W, V


def run_eig_clamp(bins, n, threads, A, eps):
    out = run_op(bins["syev"], "eig_clamp", "simple", args=[n, threads, eps],
                 inputs=[np.asfortranarray(A).ravel(order="F")])
    return out.reshape(n, n, order="F")


def check_eig(A, W, V, rtol=1e-3):
    """Shared correctness net: ascending order, eigenvalues vs eigh,
    orthogonality of V, and the V diag(W) Vᵀ reconstruction."""
    n = A.shape[0]
    wref = np.linalg.eigvalsh(A.astype(np.float64))
    scale = max(np.abs(wref).max(), 1e-6)
    assert np.all(np.diff(W) >= 0), "eigenvalues not ascending"
    assert np.allclose(W, wref, rtol=rtol, atol=rtol * scale), \
        f"eigenvalues off: {W} vs {wref}"
    V64 = V.astype(np.float64)
    assert np.allclose(V64.T @ V64, np.eye(n), atol=5e-3), "V not orthogonal"
    R = (V64 * W.astype(np.float64)) @ V64.T
    assert np.allclose(R, A, atol=rtol * scale, rtol=rtol), "V W Vᵀ != A"


@pytest.mark.parametrize("n", SIZES)
def test_syev_vs_numpy(bins, n):
    A = make_sym(n, seed=11 * n + 1)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)


@pytest.mark.parametrize("n", SIZES)
def test_syev_eigenvectors_match_up_to_sign(bins, n):
    # Well-separated spectrum 1..n so the eigh comparison is stable: each GPU
    # column must line up with the reference column up to SIGN, i.e.
    # diag(|V_refᵀ V_gpu|) ≈ 1.
    A = make_sym_spectrum(n, np.arange(1, n + 1), seed=7 * n + 3)
    W, V = run_syev(bins, n, 128, A)
    wref, vref = np.linalg.eigh(A.astype(np.float64))
    assert np.allclose(W, wref, rtol=1e-3, atol=1e-3 * n)
    d = np.abs(np.diag(vref.T @ V.astype(np.float64)))
    assert np.allclose(d, 1.0, atol=1e-2), f"eigenvector columns misaligned: {d}"


@pytest.mark.parametrize("n", SIZES)
def test_syev_compile_time_overload(bins, n):
    A = make_sym(n, seed=5 * n + 2)
    W, V = run_syev(bins, n, 64, A, op="syev_ct")
    check_eig(A, W, V)


def test_syev_repeated_eigenvalues(bins):
    # identity + rank-1: eigenvalues {1 (x n-1), 1 + ||u||²} — a degenerate
    # cluster. Per-column vector match is ill-posed here; the reconstruction +
    # orthogonality net still fully pins correctness.
    n = 12
    u = np.random.default_rng(3).standard_normal(n)
    A = (np.eye(n) + np.outer(u, u)).astype(np.float32)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    wref = np.sort(np.concatenate([np.ones(n - 1), [1 + u @ u]]))
    assert np.allclose(W, wref, rtol=1e-3, atol=1e-3)


def test_syev_ill_conditioned(bins):
    n = 16
    A = make_spd(n, seed=9, cond=1e6)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    assert np.all(W > 0), "SPD input must give positive eigenvalues"


def test_syev_negative_definite(bins):
    n = 12
    A = (-make_spd(n, seed=21)).astype(np.float32)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    assert np.all(W < 0), "negative-definite input must give negative eigenvalues"


def test_syev_indefinite(bins):
    n = 16
    eigs = np.linspace(-5.0, 5.0, n)          # mixed signs, well separated
    A = make_sym_spectrum(n, eigs, seed=13)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    assert (W < 0).sum() == (eigs < 0).sum(), "inertia mismatch"


@pytest.mark.parametrize("n", [4, 12, 16])
def test_eig_clamp_indefinite_becomes_spd(bins, n):
    eps = 0.05
    A = make_sym(n, seed=17 * n + 5)          # generally indefinite
    out = run_eig_clamp(bins, n, 128, A, eps)
    # exact symmetry by construction (canonical FMA operand order)
    assert np.array_equal(out, out.T), "clamped output not bit-symmetric"
    w_out = np.linalg.eigvalsh(out.astype(np.float64))
    assert np.all(w_out >= eps - 1e-3), f"not SPD at floor eps: {w_out}"
    # equals the numpy-clamped reconstruction
    w, v = np.linalg.eigh(A.astype(np.float64))
    ref = (v * np.maximum(w, eps)) @ v.T
    scale = max(np.abs(w).max(), 1.0)
    assert np.allclose(out, ref, rtol=1e-2, atol=1e-2 * scale)


def test_eig_clamp_spd_passthrough(bins):
    # eigenvalues already above the floor: the clamp must return A (round-off only).
    n = 7
    A = make_spd(n, seed=8)                   # eigenvalues >= n by construction
    out = run_eig_clamp(bins, n, 64, A, 1e-3)
    assert np.allclose(out, A, rtol=1e-3, atol=1e-3 * np.abs(A).max())


@pytest.mark.parametrize("n", [7, 16])
def test_syev_thread_invariance(bins, n):
    A = make_sym(n, seed=100 + n)
    refW = refV = None
    for th in THREAD_SWEEP:
        W, V = run_syev(bins, n, th, A)
        if refW is None:
            refW, refV = W, V
        else:
            assert np.array_equal(W, refW), f"W non-invariant at threads={th}"
            assert np.array_equal(V, refV), f"V non-invariant at threads={th}"


def test_eig_clamp_thread_invariance(bins):
    n = 7
    A = make_sym(n, seed=42)
    ref = None
    for th in THREAD_SWEEP:
        out = run_eig_clamp(bins, n, th, A, 0.05)
        if ref is None:
            ref = out
        else:
            assert np.array_equal(out, ref), f"non-invariant at threads={th}"
