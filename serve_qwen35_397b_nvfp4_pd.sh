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

# nvidia/Qwen3.5-397B-A17B-NVFP4, 1P1D Raiden PD disaggregation on standalone
# tpu7x-8 hosts. Translated from 0522-thinking-re-nvfp4-inputscale/
# (l20g.config + l20g_{prefill,decode}.config + products/no_v6d overlay).
# See qwen35_397b_nvfp4_pd_l20g_recipe.md for what was kept, changed, dropped.
#
# Usage (same RUN_ID on every host):
#   ROLE=prefill PREFILL_IP=<ip>                            bash serve_qwen35_397b_nvfp4_pd.sh
#   ROLE=decode  DECODE_IP=<ip>                             bash serve_qwen35_397b_nvfp4_pd.sh
#   ROLE=proxy   PREFILL_IP=<ip> DECODE_IP=<ip>             bash serve_qwen35_397b_nvfp4_pd.sh
set -euo pipefail

ROLE="${ROLE:?ROLE must be prefill, decode or proxy}"
REPO="${REPO:-/zhaoleo/vllm-torchtpu}"
VENV="${VENV:-${REPO}/.venv}"
MODEL="${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
RUN_ID="${RUN_ID:-q397nvfp4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
# products/no_v6d: Raiden host-DRAM offload on the prefill engine (64 GiB).
PREFILL_OFFLOAD="${PREFILL_OFFLOAD:-1}"
PREFILL_CPU_BYTES_TO_USE="${PREFILL_CPU_BYTES_TO_USE:-68719476736}"
# Set to a directory to enable xprof capture via /start_profile + /stop_profile,
# e.g. PROFILER_DIR=/zhaoleo/xprof/${RUN_ID}/${ROLE} (shared NFS, all hosts).
PROFILER_DIR="${PROFILER_DIR:-}"

export HF_HOME="${HF_HOME:-/zhaoleo}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ "${ROLE}" == proxy ]]; then
  exec python "${REPO}/examples/disagg/toy_proxy_server.py" \
    --host 0.0.0.0 --port "${PROXY_PORT}" \
    --prefiller-hosts "${PREFILL_IP:?}" --prefiller-ports "${PREFILL_PORT}" \
    --decoder-hosts "${DECODE_IP:?}" --decoder-ports "${DECODE_PORT}"
fi

case "${ROLE}" in
  prefill) HOST_IP="${PREFILL_IP:?}" ;;
  decode) HOST_IP="${DECODE_IP:?}" ;;
  *) echo "ERROR: unknown ROLE=${ROLE}" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- l20g.config
# Run each host as a standalone tpu7x-8. The VM's tpu-env describes the whole
# tpu7x-32 slice (HOST_BOUNDS 1,1,4), which would make libtpu wait for peers.
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1
export TPU_TOPOLOGY_WRAP=false,false,false
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export TPU_SKIP_MDS_QUERY=true
export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
export TPU_PROCESS_PORT=8471
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434,8435,8436,8437,8438
export TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL=2m

export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export VLLM_HOST_IP="${HOST_IP}"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export OMP_NUM_THREADS=1

export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export RAGGED_GATHER_REDUCE_VERSION=v3
export TPU_MOE_OWNER_OUTPUT_MODE=on
export TPU_MOE_SKIP_PADDED_TOKENS=0
export TPU_GDN_CONV_STATE_TILE_PAD=1
export TPU_PARALLEL_PRECOMPILE=1
export TPU_PREMAPPED_BUFFER_SIZE=4294967296

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_TRANSFER_PARALLELISM=8
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="${HOST_IP}"
export TPU_RAIDEN_RESHARD_PORT_BASE=27000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
export TPU_RAIDEN_POOL_STAGING_LEASES=1
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
# Decode is TP2: every decode TP worker is a destination shard. The l20g
# config sets this on prefill only; it equals decode TP, so set it on both.
export TPU_RAIDEN_DST_SHARDS=2
export TPU_KV_SHM_POOL_GB=8
export TPU_KV_TRANSFER_PORT=29200
export TPU_SIDE_CHANNEL_PORT=19600
export TPU_P2P_WAIT_PULL_TIMEOUT=10

export TPU_RAIDEN_JOB_NAME="${ROLE}"
export TPU_RAIDEN_ENGINE_ID="${RUN_ID}-${ROLE}-${HOST_IP}"
export TPU_KV_TRANSFER_NAMESPACE="${RUN_ID}-${ROLE}-${HOST_IP}"

CC_INDUCTOR='"inductor_compile_config":{"alignment_asserts":false,"enable_auto_functionalized_v2":false,"scalar_asserts":false,"size_asserts":false}'
CC_BACKEND='"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor"'
# FP8 online quantization of the bf16 (ModelOpt-excluded) attention, GDN and
# shared-expert projections; the NVFP4 experts are untouched.
ONLINE_FP8='"tpu_online_fp8":{"enabled":true,"exclude":[],"targets":["*.self_attn.*_proj","*.linear_attn.in_proj_*","*.linear_attn.out_proj","*.mlp.shared_expert.*_proj"],"weight_scheme":"per_channel"}'
RAIDEN_KV='"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_port":14579'

COMMON_ARGS=(
  --host 0.0.0.0
  --trust-remote-code
  --async-scheduling
  --attention-backend CUSTOM
  --disable-custom-all-reduce
  --no-disable-hybrid-kv-cache-manager
  --enable-expert-parallel
  --enable-prefix-caching
  --kv-cache-dtype fp8
  --language-model-only
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --mamba-cache-mode align
  --mamba-ssm-cache-dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN}"
  --hf-overrides '{"text_config":{"max_position_embeddings":1048576,"rope_parameters":{"partial_rotary_factor":0.25,"rope_theta":10000000,"type":"yarn","rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}}'
)

if [[ "${ROLE}" == prefill ]]; then
  # ------------------------------------------------------ l20g_prefill.config
  export MOE_FUSED_EP_KERNEL_IMPL=adaptive
  export MOE_REQUANTIZE_BLOCK_SIZE=256
  if [[ "${PREFILL_OFFLOAD}" == 1 ]]; then
    KV_CFG='{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{'"${RAIDEN_KV}"',"kv_role":"kv_producer","kv_load_failure_policy":"fail"},{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${PREFILL_CPU_BYTES_TO_USE}"',"raiden_controller_port":47800}}]}}'
  else
    KV_CFG='{'"${RAIDEN_KV}"',"kv_role":"kv_producer"}'
  fi
  ROLE_ARGS=(
    --port "${PREFILL_PORT}"
    --tensor-parallel-size 1
    --prefill-context-parallel-size 8
    --cp-kv-cache-interleave-size 256
    --block-size 768
    --gpu-memory-utilization 0.75
    --max-num-batched-tokens 32768
    --long-prefill-token-threshold 32768
    --max-num-seqs 128
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    --compilation-config '{'"${CC_BACKEND}"',"compile_sizes":[512,1024,2048,4096],'"${CC_INDUCTOR}"'}'
    --additional-config '{'"${ONLINE_FP8}"'}'
    --kv-transfer-config "${KV_CFG}"
  )
else
  # ------------------------------------------------------- l20g_decode.config
  ROLE_ARGS=(
    --port "${DECODE_PORT}"
    --tensor-parallel-size 2
    --data-parallel-size 4
    --data-parallel-size-local 4
    --prefill-context-parallel-size 1
    --block-size 2304
    --gpu-memory-utilization 0.9
    --max-num-batched-tokens 768
    --max-num-seqs 192
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --compilation-config '{'"${CC_BACKEND}"',"compile_sizes":[32,64,128,256,384,512,768],'"${CC_INDUCTOR}"'}'
    --additional-config '{'"${ONLINE_FP8}"'}'
    --kv-transfer-config '{'"${RAIDEN_KV}"',"kv_role":"kv_consumer"}'
  )
fi

# Optional xprof capture: enables POST /start_profile and /stop_profile.
if [[ -n "${PROFILER_DIR:-}" ]]; then
  mkdir -p "${PROFILER_DIR}"
  ROLE_ARGS+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"'"${PROFILER_DIR}"'","ignore_frontend":true}')
fi

set -x
exec vllm serve "${MODEL}" "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}"
