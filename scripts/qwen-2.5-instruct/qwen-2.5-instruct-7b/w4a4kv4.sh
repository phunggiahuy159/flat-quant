#!/bin/bash
set -e
# Run from the repo root regardless of where this script is invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Needed so lm-eval can fetch the commonsense-QA datasets (piqa, winogrande, ...).
export HF_DATASETS_TRUST_REMOTE_CODE=1
# Help torch find the conda CUDA toolkit headers/libs if a rebuild is triggered.
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

python ./main.py \
    --model ./modelzoo/Qwen/Qwen2.5-7B-Instruct \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --save_matrix \
    --lm_eval --lm_eval_batch_size 16 \
    --deactive_amp --direct_inv
