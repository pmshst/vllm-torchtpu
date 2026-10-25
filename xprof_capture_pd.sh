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

# Capture xprof traces on a PD pair whose servers were started with
# PROFILER_DIR set (serve_qwen35_397b_nvfp4_pd.sh). Two captures:
#   1. <prefix>_deepswe_c<conc>: steady state under a live DeepSWE agent load
#   2. <prefix>_single_8k: one ~8K-token prompt through the proxy (PD request)
#
#   PREFILL_IP=10.128.0.182 DECODE_IP=10.128.0.147 PROXY_PORT=8000 bash xprof_capture_pd.sh
set -euo pipefail

REPO="${REPO:-/zhaoleo/vllm-torchtpu}"
PREFILL_IP="${PREFILL_IP:?}"
DECODE_IP="${DECODE_IP:?}"
PROXY_PORT="${PROXY_PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-16}"
WARMUP_S="${WARMUP_S:-240}"
CAPTURE_S="${CAPTURE_S:-20}"
PREFIX="${PREFIX:-pair1}"
OUT_DIR="${OUT_DIR:-/zhaoleo/logs/xprof_${PREFIX}_$(date +%m%d_%H%M)}"
MODEL="${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
P="http://${PREFILL_IP}:8400"
D="http://${DECODE_IP}:9400"
mkdir -p "${OUT_DIR}"

start_both() {
  local body="{\"profile_prefix\": \"$1\"}"
  curl -sS -m 600 -X POST "${P}/start_profile" -H 'Content-Type: application/json' -d "${body}" &
  curl -sS -m 600 -X POST "${D}/start_profile" -H 'Content-Type: application/json' -d "${body}" &
  wait
  echo "$(date -u +%T) profiling started: $1"
}

stop_both() {
  curl -sS -m 1800 -X POST "${P}/stop_profile" &
  curl -sS -m 1800 -X POST "${D}/stop_profile" &
  wait
  echo "$(date -u +%T) profiling stopped"
}

# 1. Steady state under agent load.
"${REPO}/../deepswe/agent-venv/bin/python" -I "${REPO}/deepswe_live_agent.py" \
  --client-id 0 --base-url "http://127.0.0.1:${PROXY_PORT}/v1" \
  --concurrency "${CONCURRENCY}" --duration-s $((WARMUP_S + CAPTURE_S + 60)) \
  --out-dir "${OUT_DIR}/load" >"${OUT_DIR}/load.log" 2>&1 &
load_pid=$!
echo "$(date -u +%T) load started (pid ${load_pid}); warming up ${WARMUP_S}s"
sleep "${WARMUP_S}"
start_both "${PREFIX}_deepswe_c${CONCURRENCY}"
sleep "${CAPTURE_S}"
stop_both
wait "${load_pid}" || true
echo "$(date -u +%T) load finished"

# 2. One isolated PD request with an ~8K-token prompt.
sleep 30
start_both "${PREFIX}_single_8k"
python3 - "${PROXY_PORT}" "${MODEL}" <<'EOF'
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
prompt = ("Explain in detail how a distributed key-value cache transfer works "
          "between a prefill server and a decode server. ") * 320
body = {"model": model, "prompt": prompt, "max_tokens": 256, "temperature": 0}
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(req, timeout=600))
print(f"single request: {time.time() - t0:.2f}s usage={out.get('usage')}")
EOF
sleep 2
stop_both
echo "OUT_DIR=${OUT_DIR}"
