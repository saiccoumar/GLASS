#!/usr/bin/env bash
# Run the GPU test suite and emit a SIGNED gpu-proof receipt (test/gpu-proof.json).
#
# This is the local half of GLASS's GPU CI: the suite runs on the lab GPU box
# (no cloud-GPU fees), signs a receipt binding {git SHA, source fingerprint,
# per-test outcomes, GPU info}, and the CPU-only GitHub Action
# (.github/workflows/verify-gpu-proof.yml) verifies the signature against the
# committer's github.com/<user>.keys on every push. Commit the receipt together
# with (or right after) the change it attests.
#
# Requires: the pytest-gpu-proof submodule (git submodule update --init),
# an SSH signing key (uses git config user.signingKey or ~/.ssh/id_*),
# and a GPU. Usage:  ./test/run_gpu_proof.sh [extra pytest args]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
$PY -m pip install -q -e test/pytest-gpu-proof

# No --gpu-proof-fail-on-skip: GLASS has 10 permanent, documented skips
# (8x test_l3 "cg kernel covers the default-flag path only", 2x test_getrf
# "zero leading pivot is singular at n=1"). They are recorded in the receipt;
# the CI verify step passes --allow-skipped and prints them for inspection.
$PY -m pytest test/ -q "$@" \
    --gpu-proof-enable \
    --gpu-proof-out test/gpu-proof.json \
    --gpu-proof-fingerprint-paths "glass.cuh,glass-cgrps.cuh,glass-nvidia.cuh,glass-defaults.cuh,src,test/cuda,test/conftest.py"

echo
echo "Signed receipt: test/gpu-proof.json — 'git add test/gpu-proof.json' to attest this run."
