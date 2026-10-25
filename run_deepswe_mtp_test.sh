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

# MTP-acceptance-drop reproduction on a PD pair: 3 DeepSWE clients, each
# shuffling the same 113 tasks at concurrency 16, repeating rounds until
# DURATION_S, plus a /metrics monitor on the decode server. MODE=live (default)
# runs the real agent; MODE=replay replays recorded trajectories.
#
#   PREFILL_IP=10.128.0.181 DECODE_IP=10.128.0.180 DURATION_S=3600 \
#     bash run_deepswe_mtp_test.sh
set -euo pipefail

REPO="${REPO:-/zhaoleo/vllm-torchtpu}"
PREFILL_IP="${PREFILL_IP:?}"
DECODE_IP="${DECODE_IP:?}"
PROXY_PORT="${PROXY_PORT:-8001}"
DURATION_S="${DURATION_S:-3600}"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
CONCURRENCY="${CONCURRENCY:-16}"
NUM_TASKS="${NUM_TASKS:-113}"
# live: mini-swe-agent on the real tasks; replay: recorded trajectories.
MODE="${MODE:-live}"
AGENT_PY="${AGENT_PY:-/zhaoleo/deepswe/agent-venv/bin/python}"
OUT_DIR="${OUT_DIR:-/zhaoleo/logs/deepswe_mtp_$(date +%m%d_%H%M)}"
PY=("${REPO}/.venv/bin/python" -I "${REPO}/deepswe_mtp_replay.py")

mkdir -p "${OUT_DIR}"
echo "OUT_DIR=${OUT_DIR}"

if ! curl -s -m 5 -o /dev/null "http://127.0.0.1:${PROXY_PORT}/healthcheck"; then
  ROLE=proxy PREFILL_IP="${PREFILL_IP}" DECODE_IP="${DECODE_IP}" \
    PROXY_PORT="${PROXY_PORT}" setsid nohup bash "${REPO}/serve_qwen35_397b_nvfp4_pd.sh" \
    >"${OUT_DIR}/proxy.log" 2>&1 </dev/null &
  sleep 15
fi

"${PY[@]}" monitor --metrics-url "http://${DECODE_IP}:9400/metrics" \
  --duration-s $((DURATION_S + 300)) --out-dir "${OUT_DIR}" \
  >"${OUT_DIR}/monitor.log" 2>&1 &
monitor_pid=$!

client_pids=()
for ((i = 0; i < NUM_CLIENTS; i++)); do
  if [[ "${MODE}" == live ]]; then
    # Real agent loop on github.com/datacurve-ai/deep-swe tasks.
    "${AGENT_PY}" -I "${REPO}/deepswe_live_agent.py" --client-id "${i}" \
      --base-url "http://127.0.0.1:${PROXY_PORT}/v1" \
      --concurrency "${CONCURRENCY}" \
      --duration-s "${DURATION_S}" --out-dir "${OUT_DIR}" \
      >"${OUT_DIR}/client${i}.log" 2>&1 &
  else
    "${PY[@]}" client --client-id "${i}" \
      --base-url "http://127.0.0.1:${PROXY_PORT}" \
      --num-tasks "${NUM_TASKS}" --concurrency "${CONCURRENCY}" \
      --duration-s "${DURATION_S}" --out-dir "${OUT_DIR}" \
      >"${OUT_DIR}/client${i}.log" 2>&1 &
  fi
  client_pids+=("$!")
done

wait "${client_pids[@]}"
sleep 60
kill "${monitor_pid}" 2>/dev/null || true
echo "DONE OUT_DIR=${OUT_DIR}"
