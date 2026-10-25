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

# Bring up "pair 1": Qwen3.5-397B-A17B-NVFP4 PD disaggregation.
#   prefill (PCP8)  on drzl 10.128.0.182 :8400
#   decode (DP4xTP2) on gw7c 10.128.0.147 :9400   (started over ssh)
#   proxy            on drzl               :8000
# Run on drzl. Waits for both servers, starts the proxy and runs a smoke test.
#
#   bash start_pair1.sh                # normal mode
#   bash start_pair1_debug.sh          # same + xprof profiler + DEBUG logs
#
# Overrides: MAX_MODEL_LEN (default 65536; the production config uses 1048576),
# RUN_ID, PREFILL_IP, DECODE_IP, DECODE_HOST, STARTUP_TIMEOUT_S.
set -euo pipefail

REPO="${REPO:-/zhaoleo/vllm-torchtpu}"
PREFILL_IP="${PREFILL_IP:-10.128.0.182}"
DECODE_IP="${DECODE_IP:-10.128.0.147}"
DECODE_HOST="${DECODE_HOST:-instance-group-1-zhaoleo-tpuv7x-gw7c}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
DEBUG="${DEBUG:-0}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-7200}"
MODE_TAG=$([[ "${DEBUG}" == 1 ]] && echo dbg || echo run)
RUN_ID="${RUN_ID:-q397nvfp4_${MODE_TAG}_$(date +%m%d_%H%M)}"
LOG_DIR=/zhaoleo/logs
PREFILL_LOG="${LOG_DIR}/${RUN_ID}_prefill.log"
DECODE_LOG="${LOG_DIR}/${RUN_ID}_decode.log"
PROXY_LOG="${LOG_DIR}/${RUN_ID}_proxy.log"
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=10)

# Shared compile caches; ssh commands do not load ~/.bashrc, so pass them.
COMMON_ENV="export VLLM_CACHE_ROOT=/zhaoleo TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/zhaoleo/persistent_cache VLLM_ENGINE_READY_TIMEOUT_S=${STARTUP_TIMEOUT_S}"
PREFILL_ENV="${COMMON_ENV}"
DECODE_ENV="${COMMON_ENV}"
if [[ "${DEBUG}" == 1 ]]; then
  PROFILE_ROOT="/zhaoleo/xprof/${RUN_ID}"
  PREFILL_ENV+=" VLLM_LOGGING_LEVEL=DEBUG PROFILER_DIR=${PROFILE_ROOT}/prefill"
  DECODE_ENV+=" VLLM_LOGGING_LEVEL=DEBUG PROFILER_DIR=${PROFILE_ROOT}/decode"
fi

log() { echo "[$(date -u +%T)] $*"; }

# ------------------------------------------------------------------ preflight
log "RUN_ID=${RUN_ID} DEBUG=${DEBUG} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
if pgrep -f "[b]in/vllm serve" >/dev/null; then
  echo "ERROR: a vLLM server is already running on $(hostname). Stop it first:" >&2
  pgrep -af "[b]in/vllm serve" | cut -c1-120 >&2
  exit 1
fi
if "${SSH[@]}" "${DECODE_HOST}" 'pgrep -f "[b]in/vllm serve"' >/dev/null; then
  echo "ERROR: a vLLM server is already running on ${DECODE_HOST}. Stop it first." >&2
  exit 1
fi
if ss -ltn | grep -qE ":(8400|14579|27000|29200|19600) "; then
  echo "ERROR: pair-1 ports are busy on $(hostname):" >&2
  ss -ltnp 2>/dev/null | grep -E ":(8400|14579|27000|29200|19600) " >&2
  exit 1
fi
echo "${RUN_ID}" >"${LOG_DIR}/current_pd_run_id"
if [[ "${DEBUG}" == 1 ]]; then
  mkdir -p "${PROFILE_ROOT}/prefill" "${PROFILE_ROOT}/decode"
fi

# --------------------------------------------------------------- start servers
log "starting prefill on $(hostname) -> ${PREFILL_LOG}"
bash -c "${PREFILL_ENV}; cd ${REPO}; ROLE=prefill PREFILL_IP=${PREFILL_IP} RUN_ID=${RUN_ID} \
  MAX_MODEL_LEN=${MAX_MODEL_LEN} setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh \
  >${PREFILL_LOG} 2>&1 </dev/null & disown"

log "starting decode on ${DECODE_HOST} -> ${DECODE_LOG}"
timeout 30 "${SSH[@]}" "${DECODE_HOST}" "${DECODE_ENV}; cd ${REPO}; ROLE=decode \
  DECODE_IP=${DECODE_IP} RUN_ID=${RUN_ID} MAX_MODEL_LEN=${MAX_MODEL_LEN} \
  setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh >${DECODE_LOG} 2>&1 </dev/null & disown"

# --------------------------------------------------------------- wait for both
log "waiting for both servers (cold start ~45 min, warm ~15-25 min)"
deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while true; do
  p=$(grep -c "Application startup complete" "${PREFILL_LOG}" 2>/dev/null || true)
  d=$(grep -c "Application startup complete" "${DECODE_LOG}" 2>/dev/null || true)
  if [[ "${p}" -ge 1 && "${d}" -ge 1 ]]; then
    log "both servers ready"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" "${PREFILL_LOG}" "${DECODE_LOG}" 2>/dev/null; then
    echo "ERROR: startup failed; first error:" >&2
    grep -hE -A20 "Traceback|EngineCore failed" "${PREFILL_LOG}" "${DECODE_LOG}" | head -40 >&2
    exit 1
  fi
  if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
    echo "ERROR: prefill process exited; see ${PREFILL_LOG}" >&2
    exit 1
  fi
  if ((SECONDS > deadline)); then
    echo "ERROR: not ready after ${STARTUP_TIMEOUT_S}s" >&2
    exit 1
  fi
  stage=$(tr '\r' '\n' <"${DECODE_LOG}" 2>/dev/null \
    | grep -oE "Loading safetensors checkpoint shards: +[0-9]+%|Compiling a graph for compile range \([0-9]+|spec-decode warmup batch: R=[0-9]+|Application startup complete" \
    | tail -1 || true)
  log "  prefill_ready=${p} decode_ready=${d}  decode stage: ${stage:-starting}"
  sleep 60
done

# --------------------------------------------------------------- proxy + smoke
if curl -s -m 5 -o /dev/null localhost:8000/healthcheck; then
  log "proxy already listening on :8000"
else
  log "starting proxy on :8000 -> ${PROXY_LOG}"
  ROLE=proxy PREFILL_IP="${PREFILL_IP}" DECODE_IP="${DECODE_IP}" \
    setsid nohup bash "${REPO}/serve_qwen35_397b_nvfp4_pd.sh" >"${PROXY_LOG}" 2>&1 </dev/null &
  disown
  sleep 15
fi

log "smoke test"
curl -s -m 300 localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
echo

cat <<EOF

Pair 1 is up (RUN_ID=${RUN_ID}).
  endpoint : http://${PREFILL_IP}:8000/v1   (proxy)
  logs     : ${PREFILL_LOG}
             ${DECODE_LOG}
             ${PROXY_LOG}
  status   : bash ${REPO}/pd_hosts_status.sh
  stop     : pkill -TERM -f "[b]in/vllm serve nvidia"; ${SSH[*]} ${DECODE_HOST} 'pkill -TERM -f "[b]in/vllm serve nvidia"'
EOF
if [[ "${DEBUG}" == 1 ]]; then
  cat <<EOF

Debug mode:
  - VLLM_LOGGING_LEVEL=DEBUG on prefill and decode (logs are much larger).
  - xprof traces are written under ${PROFILE_ROOT}/{prefill,decode}/<prefix>/.
    Capture a window (start, send traffic, stop):
      curl -X POST ${PREFILL_IP}:8400/start_profile -H 'Content-Type: application/json' -d '{"profile_prefix":"mycase"}'
      curl -X POST ${DECODE_IP}:9400/start_profile  -H 'Content-Type: application/json' -d '{"profile_prefix":"mycase"}'
      ... send requests to :8000 ...
      curl -X POST ${PREFILL_IP}:8400/stop_profile; curl -X POST ${DECODE_IP}:9400/stop_profile
    Or run the automated capture:
      PREFILL_IP=${PREFILL_IP} DECODE_IP=${DECODE_IP} OUT_DIR=/zhaoleo/logs/xprof_${RUN_ID} bash ${REPO}/xprof_capture_pd.sh
    View: /zhaoleo/xprof-venv/bin/xprof --logdir ${PROFILE_ROOT} --port 6006
          (from your laptop: ssh -L 6006:localhost:6006 <drzl>)
EOF
fi
