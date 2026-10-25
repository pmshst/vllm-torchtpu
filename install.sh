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

# Retry `uv sync --group vllm` until it succeeds.

set -u  # error on undefined vars (no -e: we handle failures ourselves)

export VLLM_TARGET_DEVICE=empty
export UV_HTTP_TIMEOUT=320

MAX_ATTEMPTS=${MAX_ATTEMPTS:-0}   # 0 = retry forever
BASE_DELAY=${BASE_DELAY:-10}      # seconds before first retry
MAX_DELAY=${MAX_DELAY:-120}       # cap for backoff
LOG_FILE=${LOG_FILE:-uv_sync.log}

attempt=1
delay=$BASE_DELAY

# Allow Ctrl+C to stop the loop cleanly
trap 'echo "[$(date "+%F %T")] Interrupted. Exiting."; exit 130' INT TERM

while true; do
  echo "[$(date '+%F %T')] Attempt #${attempt}: uv sync --group vllm" | tee -a "$LOG_FILE"

  if uv sync --group vllm 2>&1 | tee -a "$LOG_FILE"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "[$(date '+%F %T')] ✅ uv sync completed successfully on attempt #${attempt}" | tee -a "$LOG_FILE"
    exit 0
  fi

  echo "[$(date '+%F %T')] ❌ Attempt #${attempt} failed." | tee -a "$LOG_FILE"

  if [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(date '+%F %T')] Reached MAX_ATTEMPTS=${MAX_ATTEMPTS}. Giving up." | tee -a "$LOG_FILE"
    exit 1
  fi

  echo "[$(date '+%F %T')] Retrying in ${delay}s..." | tee -a "$LOG_FILE"
  sleep "$delay"

  attempt=$((attempt + 1))
  delay=$(( delay * 2 ))
  [ "$delay" -gt "$MAX_DELAY" ] && delay=$MAX_DELAY
done

