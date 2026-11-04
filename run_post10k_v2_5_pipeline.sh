#!/usr/bin/env bash
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Phase B Master Post-Step-10,000 Pipeline (Symmetric Pre-SFT vs Post-SFT, VAULT-Only — No External Models):
#   1. Archive & write-protect `checkpoints/vault_v2_4_step10000_best.pt` (`chmod 444`).
#   2. Run FULL Benchmark Suite on Pre-SFT `VAULT v2.4 @ Step 10,000` (10-task `lm-eval` + Latency + 8-Domain + 2K/4K/128K + Mitosis).
#   3. Run Two-Stage Precision SFT (`train_vault_v2_5_sft.py`: 1,400 @ 2K + 400 @ 4K + Polyak EMA + Top-4 Soup).
#   4. Run IDENTICAL FULL Benchmark Suite on Post-SFT `VAULT v2.5` (`checkpoints/vault_v2_5_best.pt`).
#   5. Generate Head-to-Head Pre-SFT vs Post-SFT Comparison + Side-by-Side Verbatim Outputs (`compare_v2_4_step10k_vs_v2_5.py`).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/phil/.gemini/antigravity/scratch/venv/bin/python"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SRC_CKPT="${ROOT_DIR}/checkpoints/vault_v2_4_best.pt"
IMMUTABLE_10K="${ROOT_DIR}/checkpoints/vault_v2_4_step10000_best.pt"
V25_BEST="${ROOT_DIR}/checkpoints/vault_v2_5_best.pt"

PRE_SUITE_OUT="${ROOT_DIR}/benchmarks/results_vault_v2_4_step10k_suite.json"
PRE_8DOM_OUT="${ROOT_DIR}/benchmarks/results_vault_v2_4_step10k_8domain.json"
POST_SUITE_OUT="${ROOT_DIR}/benchmarks/results_vault_v2_5_post10k.json"
POST_8DOM_OUT="${ROOT_DIR}/benchmarks/results_vault_v2_5_8domain.json"
COMP_OUT="${ROOT_DIR}/benchmarks/comparison_v2_4_step10k_vs_v2_5_sft.json"

echo "============================================================================================================"
echo "[Phase B - Step 1/5] Archiving & Write-Protecting Step 10,000 VAULT v2.4 Checkpoint"
echo "============================================================================================================"
if [[ ! -f "${IMMUTABLE_10K}" ]]; then
    cp -f "${SRC_CKPT}" "${IMMUTABLE_10K}"
    chmod 444 "${IMMUTABLE_10K}"
    echo "[+] Saved immutable archive: ${IMMUTABLE_10K} (mode 444)"
else
    echo "[i] Immutable archive already present: ${IMMUTABLE_10K}"
fi

echo "============================================================================================================"
echo "[Phase B - Step 2/5] FULL PRE-SFT BENCHMARK SUITE on VAULT v2.4 @ Step 10,000 (VAULT Only, Zero External Models)"
echo "============================================================================================================"
if "${PY}" -c "import json; d=json.load(open('${ROOT_DIR}/benchmarks/results_vault_traditional.json')); assert 'VAULT_v2_4_Step10000_180M' in d.get('models', {})" 2>/dev/null; then
    echo "[i] Traditional 10-task zero-shot results for VAULT_v2_4_Step10000_180M already present — reusing!"
else
    "${PY}" -u "${ROOT_DIR}/benchmarks/bench_vault_traditional.py" \
        --only VAULT_v2_4_Step10000_180M
fi

if "${PY}" -c "import json; d=json.load(open('${PRE_SUITE_OUT}')); assert 'timestamp_completed' in d and 'extreme_niah_2k_to_128k' in d" 2>/dev/null; then
    echo "[i] Pre-SFT Full Long-Context/Mitosis/8-Domain Suite (${PRE_SUITE_OUT}) already completed — reusing!"
else
    "${PY}" -u "${ROOT_DIR}/benchmarks/bench_vault_v2_5_post10k_suite.py" \
        --ckpt "${IMMUTABLE_10K}" \
        --eval_native_v2_4 \
        --stage_label VAULT_v2_4_Step10000 \
        --domain8_out "${PRE_8DOM_OUT}" \
        --out "${PRE_SUITE_OUT}"
fi

echo "============================================================================================================"
echo "[Phase B - Step 3/5] Running VAULT v2.5 Two-Stage Precision SFT (1,400 @ 2K + 400 @ 4K + Polyak EMA + Soup)"
echo "============================================================================================================"
if [[ -f "${V25_BEST}" ]] && "${PY}" -c "import json; d=json.load(open('${ROOT_DIR}/benchmarks/vault_v2_5_sft_telemetry.json')); assert 'completed_at' in d" 2>/dev/null; then
    echo "[i] VAULT v2.5 SFT already completed (${V25_BEST}) — reusing!"
else
    "${PY}" -u "${ROOT_DIR}/train_vault_v2_5_sft.py" \
        --init_ckpt "${IMMUTABLE_10K}"
fi

echo "============================================================================================================"
echo "[Phase B - Step 4/5] IDENTICAL FULL POST-SFT BENCHMARK SUITE on VAULT v2.5 (VAULT Only, Zero External Models)"
echo "============================================================================================================"
if "${PY}" -c "import json; d=json.load(open('${ROOT_DIR}/benchmarks/results_vault_traditional.json')); assert 'VAULT_v2_5_180M' in d.get('models', {})" 2>/dev/null; then
    echo "[i] Traditional 10-task zero-shot results for VAULT_v2_5_180M already present — reusing!"
else
    "${PY}" -u "${ROOT_DIR}/benchmarks/bench_vault_traditional.py" \
        --only VAULT_v2_5_180M
fi

if "${PY}" -c "import json; d=json.load(open('${POST_SUITE_OUT}')); assert 'timestamp_completed' in d and 'extreme_niah_2k_to_128k' in d" 2>/dev/null; then
    echo "[i] Post-SFT Full Long-Context/Mitosis/8-Domain Suite (${POST_SUITE_OUT}) already completed — reusing!"
else
    "${PY}" -u "${ROOT_DIR}/benchmarks/bench_vault_v2_5_post10k_suite.py" \
        --ckpt "${V25_BEST}" \
        --skip_v2_4_ref \
        --stage_label VAULT_v2_5_SFT \
        --domain8_out "${POST_8DOM_OUT}" \
        --out "${POST_SUITE_OUT}"
fi

echo "============================================================================================================"
echo "[Phase B - Step 5/5] Head-to-Head Comparison: VAULT v2.4 @ Step 10k vs VAULT v2.5 SFT (+ Verbatim Outputs)"
echo "============================================================================================================"
"${PY}" -u "${ROOT_DIR}/benchmarks/compare_v2_4_step10k_vs_v2_5.py" \
    --pre_suite_json "${PRE_SUITE_OUT}" \
    --post_suite_json "${POST_SUITE_OUT}" \
    --pre_8dom_json "${PRE_8DOM_OUT}" \
    --post_8dom_json "${POST_8DOM_OUT}" \
    --out "${COMP_OUT}"

echo "============================================================================================================"
echo "[COMPLETE] Pre-SFT Full Suite -> v2.5 SFT -> Post-SFT Full Suite -> Head-to-Head Comparison Finished!"
echo "============================================================================================================"
