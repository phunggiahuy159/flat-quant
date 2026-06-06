#!/usr/bin/env bash
# =============================================================================
# Download the model + datasets FlatQuant needs into the layout the code expects.
#
# Usage:
#     bash scripts/download_assets.sh [HF_MODEL_REPO]
#
# Examples:
#     bash scripts/download_assets.sh                              # default Qwen2.5-7B-Instruct
#     bash scripts/download_assets.sh Qwen/Qwen2.5-7B-Instruct
#     bash scripts/download_assets.sh meta-llama/Meta-Llama-3-8B   # (gated: run `hf auth login` first)
#
# Downloads:
#   - model        -> ./modelzoo/<repo>
#   - WikiText-2   -> ./datasets/wikitext            (calibration + PPL)
#   - C4 (2 shards)-> ./datasets/allenai/c4/en       (calibration + PPL)
#
# The commonsense-QA datasets used by --lm_eval (piqa, hellaswag, arc, winogrande,
# lambada) are fetched automatically by lm-eval at evaluation time, so they are
# NOT downloaded here. Pass HF_DATASETS_TRUST_REMOTE_CODE=1 when running with
# --lm_eval (the provided run scripts already do).
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODEL_REPO="${1:-Qwen/Qwen2.5-7B-Instruct}"
HF="$(command -v hf || true)"
if [[ -z "$HF" ]]; then
    HF="$(python -m pip show huggingface_hub >/dev/null 2>&1 && echo "python -m huggingface_hub" || true)"
fi
[[ -z "$HF" ]] && { echo "ERROR: 'hf' CLI not found. Run: pip install -U huggingface_hub"; exit 1; }

echo ">>> [1/3] Model: $MODEL_REPO -> ./modelzoo/$MODEL_REPO"
hf download "$MODEL_REPO" --local-dir "./modelzoo/$MODEL_REPO"

echo ">>> [2/3] WikiText-2 -> ./datasets/wikitext"
hf download wikitext --repo-type dataset --local-dir ./datasets/wikitext

echo ">>> [3/3] C4 (validation + 1 train shard) -> ./datasets/allenai/c4"
hf download allenai/c4 --repo-type dataset \
    --include "en/c4-validation.00000-of-00008.json.gz" "en/c4-train.00000-of-01024.json.gz" \
    --local-dir ./datasets/allenai/c4

echo
echo ">>> Done. Assets are in ./modelzoo and ./datasets"
