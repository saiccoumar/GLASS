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

COVERAGE:
  * Both dtypes (f32/f64) — the tier claims a register-residency ceiling for
    BOTH (CLAUDE.md); each op runs under both instantiations.
  * Sizes bracket the measured N<=7 ceiling (N=8 still computes correctly, it
    just spills `A` to local memory).
  * trsv sweeps its full flag surface (Lower/Upper × Unit/NonUnit × trans) and
    gemv sweeps (trans × row-major) — the thread:: overloads only forward these
    to the shared `*_impl`, but we exercise the combinations rather than trust
    the forwarding.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest
import scipy.linalg

RNG = np.random.default_rng(11)

# 4..7 register-resident; 8 is past the measured ceiling (correct, just spilled).
SIZES = [4, 5, 6, 7, 8]
DTYPES = ["f32", "f64"]
# >32 (multi-warp) and not a multiple of TPB=64 (ragged tail block).
NPROB = 100

# Per-dtype tolerances: f32 accumulates visibly, f64 is near-exact.
_TOL = {"f32": dict(rtol=1e-3, atol=1e-3), "f64": dict(rtol=1e-10, atol=1e-10)}
_NPDT = {"f32": np.float32, "f64": np.float64}


# ─── local runner (dtype- and flag-aware; conftest.run_op is float32-only) ────

def _run(binary, op, model, dtype, n, P, inputs, flags=()):
    """Write float32 .bin inputs, invoke the driver, parse stdout at `dtype`.

    Inputs are always float32 on disk (the driver widens to T on load); the
    output is parsed at the native dtype so the bit-identical check stays exact
    even in f64.
    """
    tmp = []
    try:
        for arr in inputs:
            fh = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(fh)
            fh.close()
            tmp.append(fh.name)
        cmd = [str(binary), op, model, dtype, str(n), str(P)] + [str(x) for x in flags] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Binary failed:\n{r.stderr}")
        return np.fromstring(r.stdout.strip(), sep=" ").astype(_NPDT[dtype])
    finally:
        for t in tmp:
            os.unlink(t)


def _both(bins, op, dtype, n, P, inputs, flags=()):
    """Run the same inputs through both models; return (thread_out, block1_out)."""
    t = _run(bins["thread"], op, "thread", dtype, n, P, inputs, flags)
    b = _run(bins["thread"], op, "block1", dtype, n, P, inputs, flags)
    return t, b


def _colmajor_batch(mats):
    """Flatten a list of matrices column-major and concatenate (`.T.ravel()`)."""
    return np.concatenate([m.T.ravel() for m in mats]).astype(np.float32)


def _assert_bit_identical(t, b, tag):
    """thread:: and a 1-thread block run the same instruction sequence."""
    assert t.shape == b.shape, f"{tag}: shape mismatch {t.shape} vs {b.shape}"
    if not np.array_equal(t, b):
        bad = int(np.sum(t != b))
        i = int(np.argmax(t != b))
        raise AssertionError(
            f"{tag}: thread:: is not bit-identical to a 1-thread block "
            f"({bad}/{t.size} elements differ; first at {i}: "
            f"{t[i]!r} vs {b[i]!r}). Either the thread tier diverged from the "
            f"shared *_impl body, or the block path is not thread-count invariant."
        )


def _spd(n):
    """A well-conditioned SPD matrix (symmetric ⇒ layout-agnostic)."""
    M = RNG.standard_normal((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


def _tri(n, lower):
    """A well-conditioned triangular matrix with a strong diagonal."""
    A = RNG.standard_normal((n, n)).astype(np.float32)
    A = np.tril(A) if lower else np.triu(A)
    A[np.diag_indices(n)] = np.abs(A[np.diag_indices(n)]) + n
    return A.astype(np.float32)


# ─── dot (tolerance only: tree reduce vs serial accumulate — different by design) ──

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_dot(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    y = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "dot", dtype, n, NPROB, [x, y])

    # Oracle in float64: the device widens the float32 inputs to T, so the f64
    # path is more accurate than a float32-precision reference would be.
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    want = np.array([x64[p*n:(p+1)*n] @ y64[p*n:(p+1)*n] for p in range(NPROB)])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    # NOT bit-identical to block1 by design; both must still hit the same answer.
    np.testing.assert_allclose(t, b, **_TOL[dtype])


# ─── gemv (sweeps trans × row-major) ──────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("rowmajor", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_gemv(bins, n, rowmajor, trans, dtype):
    mats = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    # ROW_MAJOR is a STORAGE flag: same logical matrix, different byte order.
    ravel = (lambda m: m.ravel()) if rowmajor else (lambda m: m.T.ravel())
    A = np.concatenate([ravel(m) for m in mats]).astype(np.float32)
    flags = (int(trans), int(rowmajor))
    t, b = _both(bins, "gemv", dtype, n, NPROB, [A, x], flags)

    x64 = x.astype(np.float64)
    want = np.concatenate([
        (mats[p].T if trans else mats[p]).astype(np.float64) @ x64[p*n:(p+1)*n]
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_bit_identical(t, b, f"gemv(trans={trans},rowmajor={rowmajor})")


# ─── gemm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_gemm(bins, n, dtype):
    ma = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    mb = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    A, B = _colmajor_batch(ma), _colmajor_batch(mb)
    t, b = _both(bins, "gemm", dtype, n, NPROB, [A, B])

    want = np.concatenate([
        (ma[p].astype(np.float64) @ mb[p].astype(np.float64)).T.ravel()
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_bit_identical(t, b, "gemm")


# ─── potrf ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_potrf(bins, n, dtype):
    mats = [_spd(n) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    t, b = _both(bins, "potrf", dtype, n, NPROB, [A])

    for p in range(NPROB):
        got = t[p*n*n:(p+1)*n*n].reshape(n, n).T      # column-major -> row-major view
        want = np.linalg.cholesky(mats[p].astype(np.float64))
        np.testing.assert_allclose(np.tril(got), np.tril(want), **_TOL[dtype])
    _assert_bit_identical(t, b, "potrf")


# ─── trsv (sweeps Lower/Upper × Unit/NonUnit × trans) ─────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("unit", [False, True])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", SIZES)
def test_trsv(bins, n, lower, unit, trans, dtype):
    mats = [_tri(n, lower) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    flags = (int(lower), int(unit), int(trans))
    t, b = _both(bins, "trsv", dtype, n, NPROB, [A, x], flags)

    want = np.concatenate([
        scipy.linalg.solve_triangular(
            mats[p].astype(np.float64), x[p*n:(p+1)*n].astype(np.float64),
            lower=lower, trans=(1 if trans else 0), unit_diagonal=unit)
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_bit_identical(t, b, f"trsv(lower={lower},unit={unit},trans={trans})")


# ─── posv (the pyroffi-relevant op: factor + both solves, layout-consistent) ───

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_posv(bins, n, dtype):
    mats = [_spd(n) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    bvec = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "posv", dtype, n, NPROB, [A, bvec])

    want = np.concatenate([
        np.linalg.solve(mats[p].astype(np.float64), bvec[p*n:(p+1)*n].astype(np.float64))
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_bit_identical(t, b, "posv")


# ─── potrs (reusable-factor path: solve from a precomputed Cholesky factor) ────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_potrs(bins, n, dtype):
    mats = [_spd(n) for _ in range(NPROB)]
    facts = [np.linalg.cholesky(m.astype(np.float64)).astype(np.float32) for m in mats]
    L = _colmajor_batch(facts)                       # lower factor, column-major
    bvec = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "potrs", dtype, n, NPROB, [L, bvec])

    # Solve against the SAME (float32-rounded) factor the device sees, not the
    # exact A: potrs consumes L, so `L Lᵀ x = b` is the reference, not `A x = b`
    # (they differ at float32-factor precision, ~1e-7 — below the f64 tolerance).
    want = np.concatenate([
        scipy.linalg.cho_solve((facts[p].astype(np.float64), True),
                               bvec[p*n:(p+1)*n].astype(np.float64))
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_bit_identical(t, b, "potrs")
