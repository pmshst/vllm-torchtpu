#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Concurrency sweep against the PD proxy for serve_qwen35_397b_nvfp4_pd.sh.
# Same shape as scripts/vllm/benchmark/inferencex/qwen3.5/bench.sh (random
# dataset, range ratio 0.8, ignore_eos), using the installed `vllm bench serve`.
#
#   ISL=8192 OSL=1024 CONCS="1 16 64 128" bash bench_qwen35_397b_nvfp4_pd.sh
set -euo pipefail

REPO="${REPO:-/zhaoleo/vllm-torchtpu}"
MODEL="${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONCS="${CONCS:-1 16 64 128}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
PROMPTS_PER_CONC="${PROMPTS_PER_CONC:-4}"
RESULT_DIR="${RESULT_DIR:-/zhaoleo/logs/bench_$(cat /zhaoleo/logs/current_pd_run_id 2>/dev/null || date +%m%d_%H%M)}"

export HF_HOME="${HF_HOME:-/zhaoleo}"
# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"
mkdir -p "${RESULT_DIR}"

for conc in ${CONCS}; do
  num_prompts=$((conc * PROMPTS_PER_CONC))
  ((num_prompts < 8)) && num_prompts=8
  vllm bench serve \
    --backend vllm \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${conc}" \
    --request-rate inf \
    --ignore-eos \
    --num-warmups "${conc}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "isl${ISL}_osl${OSL}_conc${conc}.json" \
    2>&1 | tee "${RESULT_DIR}/isl${ISL}_osl${OSL}_conc${conc}.log"
done
echo "RESULT_DIR=${RESULT_DIR}"
