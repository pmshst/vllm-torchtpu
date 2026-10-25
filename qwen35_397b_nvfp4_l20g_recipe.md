# Qwen3.5-397B-A17B-NVFP4: PD-Disaggregated Serving from the `l20g` config

This recipe runs `nvidia/Qwen3.5-397B-A17B-NVFP4` as plain `vllm serve`
commands. It is a translation of the production config bundle in
`0522-thinking-re-nvfp4-inputscale/`.

## 0. How the config bundle maps to this recipe

The bundle is layered. Later files override earlier ones:

| Role | Files merged, in order |
|---|---|
| common | `l20b.config` → `l20d.config` → `l20g.config` |
| prefill | common + `l20b_prefill` → `l20d_prefill` → `l20g_prefill` → `products/no_v6d/l20g_prefill` |
| decode | common + `l20b_decode` → `l20d_decode` → `l20g_decode` (→ `products/no_v6d/l20g_decode`, which is empty) |

`l20b`/`l20d` (and their role files) only hold `DS_LLM_*`, Vineyard (`V6D_*`),
and `AQUILA_*` / `NVIDIA_*` settings. Those belong to the internal serving
wrapper, not to vLLM. The recipe uses the **`products/no_v6d`** variant: it
replaces the Vineyard/Tair connector with the in-repo
`TPURaidenOffloadingConnector` (host-DRAM KV offload), so you need no external
KV service.

Topology (one `tpu7x-8` host per role: 2x2x1 chips, 8 TPU devices):

| Role | Parallelism | Raiden admission topology |
|---|---|---|
| Prefill (`kv_producer`) | `TP=1, PCP=8`, EP, MTP x1 | `pcp8_prefill` |
| Decode (`kv_consumer`) | `TP=2, DP=4`, EP, MTP x3 | `dp4tp2_decode` |

### What was dropped or changed, and why

| Config item | Treatment | Reason |
|---|---|---|
| `"quantization": "fp8"` (`l20g`) | **dropped** | The checkpoint is ModelOpt NVFP4. vLLM's `ModelOptNvFp4Config.override_quantization_method` silently changes it to `modelopt_fp4` anyway (`vllm/config/model.py`). Passing `fp8` is misleading, so leave it unset (or pass `--quantization modelopt_fp4`). |
| `enable_think`, `think_mode`, `DS_LLM_MAX_THINK_TOKENS` | dropped | Wrapper-only options. Qwen3.5 thinks by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`. Add `--reasoning-parser qwen3` if you want `reasoning_content` split out. |
| `disaggregation_mode`, `enable_disaggregated_prefilling`, `disaggregated_prefilling_if_name` | dropped | Wrapper-only. In vLLM the role comes from `kv_role` in `--kv-transfer-config`. |
| `DS_*`, `V6D_*`, `TAIR_MEMPOOL_*`, `SRPC_*`, `MC_TENT_CONF`, `DATA_IFACE`, `DATA_IP`, `DSV_*` | dropped | Wrapper / Vineyard / Tair stack. The `no_v6d` product does not use them. |
| `TPUV6dObjectConnector` (`kvs_connector` plugin) | replaced | Replaced by `TPURaidenOffloadingConnector`, following `products/no_v6d`. `VLLM_PLUGINS` drops `kvs_connector`. |
| `"backend": "kvt"` in the multi-connector extra config, and `additional_config.raiden_pd_send_completion` | dropped | Nothing in this repo reads them. |
| `limit_mm_per_prompt`, `mm_encoder_tp_mode` | dropped | No-ops with `--language-model-only`. |
| `profiler_config` | optional | Add `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm_profile","ignore_frontend":true}'` if you need it. |
| `TPU_RAIDEN_CONTROLLER_ADDRESS` | kept but inert | Store mode (`TPU_RAIDEN_RESHARD_IMPL=store`) ignores it. No controller process is needed. |
| `TPU_RAIDEN_DST_SHARDS=2` (prefill only in config) | set on **both** | Only the producer reads it today. It must equal the decode TP (2) for `dp4tp2_decode`. Setting it on decode too does no harm and survives future changes. |
| `${POD_IP}` | → `${HOST_IP}` | The routable IP of the host the role runs on. |

These env vars are in the config, but no Python in this repo reads them. They
are passed through to `torch_tpu` / tpu-sync / libtpu. Keep them as-is:
`TPU_GDN_CONV_STATE_TILE_PAD`, `TPU_KV_RESHARD_DST_PAGE_TOKENS`,
`TPU_VLLM_KV_CACHE_ALIAS_FALLBACK`, `TPU_RAIDEN_DATA_NICS`,
`TPU_RAIDEN_CONTROL_PLANE_BACKEND`, `TPU_RAIDEN_DATA_CONN_REUSE`,
`TPU_RAIDEN_ENABLE_SOURCE_IP_BIND`, `TPU_RAIDEN_BIND_IP`,
`TPU_RAIDEN_TELEMETRY_BACKENDS`, `TPU_RAIDEN_TRANSFER_STREAMS`,
`USE_BATCHED_RPA_LONGCTX`. If your installed build does not know them, they
do nothing.

## 1. Shared environment (both hosts)

Both hosts need the same `vllm`, `vllm-torchtpu`, `torch-tpu`, and tpu-sync
builds and the same model snapshot. They also need bidirectional TCP
reachability on the ports in §5.

```bash
# --- common.env : source on BOTH hosts --------------------------------------
export MODEL=nvidia/Qwen3.5-397B-A17B-NVFP4
export HOST_IP=<THIS_HOST_ROUTABLE_IP>          # was ${POD_IP}

# vLLM / plugin
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export VLLM_HOST_IP=${HOST_IP}
export VLLM_KV_CACHE_LAYOUT=HND                  # seq-on-lane FA layout
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1           # 1M context via YaRN override
export VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=1

# Single-host tpu7x-8 slice (2x2x1 chips = 8 devices)
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_TOPOLOGY=2x2x1
export TPU_TOPOLOGY_WRAP=false,false,false
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export TPU_SKIP_MDS_QUERY=true
export TPU_PROCESS_PORT=8471
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434,8435,8436,8437,8438
export TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL=2m
export TPU_PARALLEL_PRECOMPILE=1
export TPU_PREMAPPED_BUFFER_SIZE=4294967296

# MoE kernels
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export RAGGED_GATHER_REDUCE_VERSION=v3
export TPU_MOE_OWNER_OUTPUT_MODE=on
export TPU_MOE_SKIP_PADDED_TOKENS=0

# Hybrid (GDN + full attention) KV pool + Raiden Stage-3 transport
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_VLLM_KV_CACHE_ALIAS_FALLBACK=0
export TPU_GDN_CONV_STATE_TILE_PAD=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_KV_RESHARD_DST_PAGE_TOKENS=8192
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_TRANSFER_PARALLELISM=8         # = prefill PCP*TP; same on P and D
export TPU_RAIDEN_DST_SHARDS=2                   # = decode TP; same on P and D
export TPU_RAIDEN_TRANSFER_STREAMS=1
export TPU_RAIDEN_POOL_STAGING_LEASES=1
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
export TPU_RAIDEN_DATA_NICS=eth0:0,eth1:1        # adjust to your NIC names
export TPU_RAIDEN_DATA_CONN_REUSE=pooled
export TPU_RAIDEN_ENABLE_SOURCE_IP_BIND=1
export TPU_RAIDEN_CONTROL_PLANE_BACKEND=grpc
export TPU_RAIDEN_TELEMETRY_BACKENDS=buffered
export TPU_RAIDEN_ADVERTISE_HOST=${HOST_IP}
export TPU_RAIDEN_BIND_IP=${HOST_IP}
export TPU_RAIDEN_CONTROLLER_ADDRESS=${HOST_IP}:28100   # inert in store mode
export TPU_RAIDEN_RESHARD_PORT_BASE=27000        # + dp_rank
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100 # + dp_rank
export TPU_KV_TRANSFER_PORT=29200
export TPU_SIDE_CHANNEL_PORT=19600
export TPU_KV_SHM_POOL_GB=8
export TPU_P2P_WAIT_PULL_TIMEOUT=10

COMPILE_PREFIX='"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","inductor_compile_config":{"alignment_asserts":false,"enable_auto_functionalized_v2":false,"scalar_asserts":false,"size_asserts":false}'
ONLINE_FP8='"tpu_online_fp8":{"enabled":true,"exclude":[],"weight_scheme":"per_channel","targets":["*.self_attn.*_proj","*.linear_attn.in_proj_*","*.linear_attn.out_proj","*.mlp.shared_expert.*_proj"]}'
HF_OVERRIDES='{"text_config":{"max_position_embeddings":1048576,"rope_parameters":{"partial_rotary_factor":0.25,"rope_theta":10000000,"type":"yarn","rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}}'

COMMON_ARGS=(
  --trust-remote-code
  --language-model-only
  --max-model-len 1048576
  --hf-overrides "${HF_OVERRIDES}"
  --enable-expert-parallel
  --disable-custom-all-reduce
  --kv-cache-dtype fp8
  --attention-backend CUSTOM
  --no-disable-hybrid-kv-cache-manager
  --mamba-cache-mode align
  --mamba-ssm-cache-dtype bfloat16
  --enable-prefix-caching
  --async-scheduling
)
```

`tpu_online_fp8` quantizes the bf16 parts that the NVFP4 checkpoint leaves
out (attention, GDN projections, shared experts) to FP8 per-channel at load
time. `attach_online_fp8` accepts it for `modelopt_fp4` configs.

## 2. Prefill instance (host A)

```bash
source common.env            # HOST_IP=<host A IP>

export TPU_RAIDEN_JOB_NAME=prefill
export TPU_RAIDEN_ENGINE_ID=l20g-prefill-${HOST_IP}
export TPU_KV_TRANSFER_NAMESPACE=l20g-prefill-${HOST_IP}
# NVFP4 MoE: W4A8 adaptive fused-EP kernel, weights requantized to block-256
export MOE_FUSED_EP_KERNEL_IMPL=adaptive
export MOE_REQUANTIZE_BLOCK_SIZE=256
PREFILL_CPU_BYTES_TO_USE=68719476736      # 64 GiB host DRAM for KV offload

P_KV='{
  "kv_connector": "TPUMultiConnector",
  "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector",
  "kv_role": "kv_producer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {"connectors": [
    {"kv_connector": "TPURaidenConnector",
     "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
     "kv_role": "kv_producer", "kv_port": 14579, "kv_load_failure_policy": "fail"},
    {"kv_connector": "TPURaidenOffloadingConnector",
     "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
     "kv_role": "kv_both",
     "kv_connector_extra_config": {"cpu_bytes_to_use": '"${PREFILL_CPU_BYTES_TO_USE}"',
                                   "raiden_controller_port": 47800}}
  ]}
}'

vllm serve ${MODEL} \
  --host 0.0.0.0 --port 8400 \
  "${COMMON_ARGS[@]}" \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 256 \
  --block-size 768 \
  --gpu-memory-utilization 0.75 \
  --max-num-batched-tokens 32768 \
  --long-prefill-token-threshold 32768 \
  --max-num-seqs 128 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --additional-config "{${ONLINE_FP8}}" \
  --compilation-config "{${COMPILE_PREFIX},\"compile_sizes\":[512,1024,2048,4096]}" \
  --kv-transfer-config "${P_KV}"
```

Notes:

- `--block-size 768` is a multiple of the interleave (256). Stage-3 PCP
  requires `page_tokens % cp_kv_cache_interleave_size == 0`.
- `gpu_memory_utilization 0.75` leaves HBM headroom for PCP8 activations at
  32K-token chunks.
- If you do not want host offload, use a plain `TPURaidenConnector` producer
  config instead of `TPUMultiConnector` (same shape as the decode one below,
  with `kv_role: kv_producer`).

## 3. Decode instance (host B)

```bash
source common.env            # HOST_IP=<host B IP>

export TPU_RAIDEN_JOB_NAME=decode
export TPU_RAIDEN_ENGINE_ID=l20g-decode-${HOST_IP}
export TPU_KV_TRANSFER_NAMESPACE=l20g-decode-${HOST_IP}
export USE_BATCHED_RPA_LONGCTX=1
# MoE stays on the default NVFP4 W4A16 GMM path (no MOE_FUSED_EP_KERNEL_IMPL)

vllm serve ${MODEL} \
  --host 0.0.0.0 --port 9400 \
  "${COMMON_ARGS[@]}" \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --prefill-context-parallel-size 1 \
  --block-size 2304 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --additional-config "{${ONLINE_FP8}}" \
  --compilation-config "{${COMPILE_PREFIX},\"compile_sizes\":[32,64,128,256,384,512,768]}" \
  --kv-transfer-config '{
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    "kv_role": "kv_consumer",
    "kv_port": 14579
  }'
```

`--max-num-seqs` is per DP rank: 4 x 192 = 768 concurrent decode streams.
With MTP x3, each step verifies up to 4 tokens per sequence. The 768-token
batch cap and the top compile bucket are sized for that.

## 4. Proxy and smoke test

```bash
python examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts <HOST_A_IP> --prefiller-ports 8400 \
  --decoder-hosts  <HOST_B_IP> --decoder-ports 9400

curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/Qwen3.5-397B-A17B-NVFP4",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 512, "temperature": 0
}'
```

Check the logs for:

- P: `NVFP4 fused EP W4A8 enabled: block-256`. D: `NVFP4 GMM W4A16 enabled: block-16`.
- `TPURaidenConnectorScheduler --> kv_ip=... | kv_port=14579` on both roles,
  then `TPURaidenConnectorScheduler Stage-3 send` on P and `Stage-3 load` on D
  for the first request. If the admission topology is wrong, startup fails
  with an explicit `Raiden Qwen3.5 admission topology ...` error. No error
  means P was admitted as `pcp8_prefill` and D as `dp4tp2_decode`.

## 5. Ports to open between hosts

| Port(s) | Use |
|---|---|
| 8400 / 9400 / 8000 | prefill / decode / proxy HTTP |
| 14579 | `kv_port` |
| 27000-27003, 27100-27103 (27000/27100 on P) | Raiden reshard store / dispatch (`base + dp_rank`) |
| 29200, 19600 | `TPU_KV_TRANSFER_PORT`, `TPU_SIDE_CHANNEL_PORT` |
| 47800 (host-local, P) | offload controller (`raiden_controller_port + dp_rank_local`) |

## 6. Common failures

| Symptom | Fix |
|---|---|
| `GDN state transfer cannot change Q/K replication` / dst-shard errors | `TPU_RAIDEN_DST_SHARDS` must equal decode TP (2). |
| `_validate_stage3_transfer_parallelism` mismatch | `TPU_RAIDEN_TRANSFER_PARALLELISM=8` on both roles. |
| `Stage-3 PCP page geometry requires page_tokens divisible by ...` | Keep prefill `--block-size` a multiple of 256. |
| `tpu_online_fp8 supports FP8 and ModelOpt NVFP4 ...` | The checkpoint was not detected as `modelopt_fp4`. Check `hf_quant_config.json` and do not force another `--quantization`. |
| `adaptive NVFP4 block-256 must ... divide hidden` | Unset `MOE_REQUANTIZE_BLOCK_SIZE` (the adaptive kernel then picks its smallest block), or drop `MOE_FUSED_EP_KERNEL_IMPL` to fall back to W4A16 GMM. |
| `cpu_bytes_to_use too small for even one ...` | Raise `PREFILL_CPU_BYTES_TO_USE`. |
| First transfer hangs | `TPU_RAIDEN_ADVERTISE_HOST` is not routable from the peer, or §5 ports are blocked. |
