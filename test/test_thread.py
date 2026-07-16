"""glass::thread:: surface tests — one problem per THREAD.

The tier's oracle is the library's own thread-count-invariance guarantee: every
`thread::` op delegates to the same `*_impl` body as `glass::`, via
`ThreadBarrier{rank=0, size=1, no-op sync}`. So the `thread` model and a `block1`
model (`<<<P, 1>>>`, the block-scoped op degenerated to one thread) run the
identical instruction sequence over the identical operand order and must agree
**BIT-FOR-BIT** — not merely to a tolerance. We assert exact equality, and
separately check both against a numpy/scipy oracle so a shared bug in both
surfaces can't pass silently.

`dot` is the deliberate exception: block-scoped `dot` reduces with a halving
TREE while `thread::dot` accumulates serially, so the two differ in summation
ORDER by design and are compared to float tolerance only. That asymmetry is why
the tier ships no `_fast`/`_lowmem` twins (a single thread has no reduction
strategy to choose) — see the `thread::` block in src/base/L1/dot.cuh.

P is >32 and NOT a multiple of the driver's TPB, so problems span multiple warps
and several blocks with a ragged tail — the shape that catches a stray
block-wide `__syncthreads()` inside a thread:: op (once the tail block's
out-of-range threads return, such a barrier has divergent participation ⇒
UB/hang). That is a real bug this suite found in `trsv_impl`.

float32. Sizes bracket the measured N<=7 register-residency ceiling (N=8 still
computes correctly, it just spills `A` to local memory).
"""

import numpy as np
import pytest
import scipy.linalg
from conftest import run_op

RNG = np.random.default_rng(11)

RTOL = 1e-4
ATOL = 1e-4

# 4..7 register-resident; 8 is past the measured ceiling (correct, just spilled).
SIZES = [4, 5, 6, 7, 8]
# >32 (multi-warp) and not a multiple of TPB=64 (ragged tail block).
NPROB = 100


def _spd(n):
    """A well-conditioned SPD matrix (symmetric ⇒ layout-agnostic)."""
    M = RNG.standard_normal((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


def _lower(n):
    """A well-conditioned lower-triangular matrix with a strong diagonal."""
    L = np.tril(RNG.standard_normal((n, n)).astype(np.float32))
    L[np.diag_indices(n)] = np.abs(L[np.diag_indices(n)]) + n
    return L.astype(np.float32)


def _batch(fn, P, n):
    """P independent problems, flattened column-major and concatenated."""
    mats = [fn(n) for _ in range(P)]
    flat = np.concatenate([m.T.ravel() for m in mats])   # .T.ravel() == column-major
    return mats, flat.astype(np.float32)


def _both(bins, op, n, P, inputs):
    """Run the same inputs through both models; return (thread_out, block1_out)."""
    t = run_op(bins["thread"], op, "thread", [n, P], inputs)
    b = run_op(bins["thread"], op, "block1", [n, P], inputs)
    return np.asarray(t, dtype=np.float32), np.asarray(b, dtype=np.float32)


def _assert_bit_identical(t, b, op):
    """thread:: and a 1-thread block run the same instruction sequence."""
    assert t.shape == b.shape, f"{op}: shape mismatch {t.shape} vs {b.shape}"
    if not np.array_equal(t, b):
        bad = int(np.sum(t != b))
        i = int(np.argmax(t != b))
        raise AssertionError(
            f"{op}: thread:: is not bit-identical to a 1-thread block "
            f"({bad}/{t.size} elements differ; first at {i}: "
            f"{t[i]!r} vs {b[i]!r}). Either the thread tier diverged from the "
            f"shared *_impl body, or the block path is not thread-count invariant."
        )


# ─── dot (tolerance only: tree reduce vs serial accumulate — different by design) ──

@pytest.mark.parametrize("n", SIZES)
def test_dot(bins, n):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    y = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "dot", n, NPROB, [x, y])

    want = np.array([x[p*n:(p+1)*n] @ y[p*n:(p+1)*n] for p in range(NPROB)], dtype=np.float32)
    np.testing.assert_allclose(t, want, rtol=RTOL, atol=ATOL)
    # NOT bit-identical to block1 by design; both must still hit the same answer.
    np.testing.assert_allclose(t, b, rtol=RTOL, atol=ATOL)


# ─── gemv / gemv_t ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_gemv(bins, n):
    mats, A = _batch(lambda k: RNG.standard_normal((k, k)).astype(np.float32), NPROB, n)
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "gemv", n, NPROB, [A, x])

    want = np.concatenate([mats[p] @ x[p*n:(p+1)*n] for p in range(NPROB)]).astype(np.float32)
    np.testing.assert_allclose(t, want, rtol=RTOL, atol=ATOL)
    _assert_bit_identical(t, b, "gemv")


@pytest.mark.parametrize("n", SIZES)
def test_gemv_t(bins, n):
    mats, A = _batch(lambda k: RNG.standard_normal((k, k)).astype(np.float32), NPROB, n)
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "gemv_t", n, NPROB, [A, x])

    want = np.concatenate([mats[p].T @ x[p*n:(p+1)*n] for p in range(NPROB)]).astype(np.float32)
    np.testing.assert_allclose(t, want, rtol=RTOL, atol=ATOL)
    _assert_bit_identical(t, b, "gemv_t")


# ─── gemm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_gemm(bins, n):
    ma, A = _batch(lambda k: RNG.standard_normal((k, k)).astype(np.float32), NPROB, n)
    mb, B = _batch(lambda k: RNG.standard_normal((k, k)).astype(np.float32), NPROB, n)
    t, b = _both(bins, "gemm", n, NPROB, [A, B])

    want = np.concatenate([(ma[p] @ mb[p]).T.ravel() for p in range(NPROB)]).astype(np.float32)
    np.testing.assert_allclose(t, want, rtol=RTOL, atol=ATOL)
    _assert_bit_identical(t, b, "gemm")


# ─── potrf ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_potrf(bins, n):
    mats, A = _batch(_spd, NPROB, n)
    t, b = _both(bins, "potrf", n, NPROB, [A])

    # Column-major lower factor; only the lower triangle is written.
    for p in range(NPROB):
        got = t[p*n*n:(p+1)*n*n].reshape(n, n).T      # back to row-major view
        want = np.linalg.cholesky(mats[p].astype(np.float64))
        np.testing.assert_allclose(np.tril(got), np.tril(want), rtol=1e-3, atol=1e-3)
    _assert_bit_identical(t, b, "potrf")


# ─── trsv ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_trsv(bins, n):
    mats, A = _batch(_lower, NPROB, n)
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "trsv", n, NPROB, [A, x])

    want = np.concatenate([
        scipy.linalg.solve_triangular(mats[p], x[p*n:(p+1)*n], lower=True)
        for p in range(NPROB)
    ]).astype(np.float32)
    np.testing.assert_allclose(t, want, rtol=1e-3, atol=1e-3)
    _assert_bit_identical(t, b, "trsv")


# ─── posv (the pyroffi-relevant op: factor + both solves, layout-consistent) ───

@pytest.mark.parametrize("n", SIZES)
def test_posv(bins, n):
    mats, A = _batch(_spd, NPROB, n)
    bvec = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "posv", n, NPROB, [A, bvec])

    want = np.concatenate([
        np.linalg.solve(mats[p].astype(np.float64), bvec[p*n:(p+1)*n].astype(np.float64))
        for p in range(NPROB)
    ]).astype(np.float32)
    np.testing.assert_allclose(t, want, rtol=1e-3, atol=1e-3)
    _assert_bit_identical(t, b, "posv")
