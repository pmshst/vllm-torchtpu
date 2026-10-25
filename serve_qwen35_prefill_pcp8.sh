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

# Qwen3.5-35B-A3B-FP8 Raiden prefill (kv_producer), PCP8, verified up on 2026-10-22.
# - interleave 128: the hybrid-Mamba KV page is forced to 384 tokens, and the
#   Raiden connector needs page_tokens % interleave == 0 (256 fails).
# - compile_sizes [4096]: the PCP GDN kernel needs per-rank bucket rows to be a
#   multiple of the interleave; the default ladder starts at 16.
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8400 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --mamba-cache-mode align \
  --no-disable-hybrid-kv-cache-manager \
  --max-model-len 8192 \
  --max-num-batched-tokens 32768 \
  --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  --kv-transfer-config '{
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    "kv_role": "kv_producer",
    "kv_port": 14579
  }'
