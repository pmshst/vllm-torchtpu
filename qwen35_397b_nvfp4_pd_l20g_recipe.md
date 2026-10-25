# Qwen3.5-397B-A17B-NVFP4 PD Disaggregation: `l20g` Production Config Recipe

This recipe serves `nvidia/Qwen3.5-397B-A17B-NVFP4` with Raiden prefill/decode
disaggregation. It follows the TPU production config in
`0522-thinking-re-nvfp4-inputscale/`:

- base: `l20g.config`
- role overlays: `l20g_prefill.config` / `l20g_decode.config`
- prefill KV overlay without vineyard: `products/no_v6d/l20g_prefill.config`

`l20b*` / `l20d*` are GPU deployments (`NVIDIA_IMEX_CHANNELS`) and are not used.

Launcher: `serve_qwen35_397b_nvfp4_pd.sh` (`ROLE=prefill|decode|proxy`).

## 1. Topology

| Role | Host (this slice) | Parallelism | Raiden admission |
|---|---|---|---|
| Prefill (`kv_producer`) | drzl `10.128.0.182` | TP1, PCP8, EP, MTP-1 | `pcp8_prefill` |
| Decode (`kv_consumer`) | gw7c `10.128.0.147` | TP2 x DP4, EP, MTP-3 | `dp4tp2_decode` |
| Proxy | any host | `toy_proxy_server.py` | n/a |

Each host runs as a **standalone tpu7x-8**. Each VM's `tpu-env` metadata
describes the full tpu7x-32 slice (`HOST_BOUNDS 1,1,4`, `TOPOLOGY 2x2x4`).
The launcher overrides this with the config's `TPU_HOST_BOUNDS=1,1,1`,
`TPU_TOPOLOGY=2x2x1`, `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`,
`TPU_WORKER_HOSTNAMES=localhost`, `TPU_WORKER_ID=0` and
`TPU_SKIP_MDS_QUERY=true`. Without these, libtpu waits for the other 3 hosts.

## 2. Run

```bash
cd /zhaoleo/vllm-torchtpu
export RUN_ID=q397nvfp4_$(date +%Y%m%d_%H%M)        # same value on both hosts

# drzl (prefill)
ROLE=prefill PREFILL_IP=10.128.0.182 RUN_ID=$RUN_ID \
  bash serve_qwen35_397b_nvfp4_pd.sh 2>&1 | tee /zhaoleo/logs/${RUN_ID}_prefill.log

# gw7c (decode)
ROLE=decode DECODE_IP=10.128.0.147 RUN_ID=$RUN_ID \
  bash serve_qwen35_397b_nvfp4_pd.sh 2>&1 | tee /zhaoleo/logs/${RUN_ID}_decode.log

# proxy (after both report "Application startup complete")
ROLE=proxy PREFILL_IP=10.128.0.182 DECODE_IP=10.128.0.147 \
  bash serve_qwen35_397b_nvfp4_pd.sh

curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d \
 '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Knobs: `MAX_MODEL_LEN` (default 1048576), `PREFILL_OFFLOAD=0` (pure Raiden
producer, no host-DRAM offload), `PREFILL_CPU_BYTES_TO_USE`, and the ports
`PREFILL_PORT`, `DECODE_PORT`, `PROXY_PORT`.

## 3. Effective configuration

### Both roles (`l20g.config`)

- **Engine args:**
  - `--async-scheduling --attention-backend CUSTOM --disable-custom-all-reduce`
  - `--no-disable-hybrid-kv-cache-manager --enable-expert-parallel --enable-prefix-caching`
  - `--kv-cache-dtype fp8 --language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'`
  - `--mamba-cache-mode align --mamba-ssm-cache-dtype bfloat16`
  - `--max-model-len 1048576`, with `--hf-overrides` setting YaRN x4 rope
    (262144 -> 1048576 positions)
- **Online FP8 for the bf16 parts** (`--additional-config tpu_online_fp8`,
  per-channel): `self_attn.*_proj`, `linear_attn.in_proj_*`, `linear_attn.out_proj`,
  `mlp.shared_expert.*_proj`. These are exactly the layers ModelOpt left in
  bf16. The NVFP4 experts are untouched. `online_fp8.py` accepts this for
  `modelopt_fp4` checkpoints.
- **KV layout and Raiden:**
  - `VLLM_KV_CACHE_LAYOUT=HND` (seq-on-lane FA layout)
  - `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`, `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`,
    `TPU_RAIDEN_QWEN35_ADMISSION=1`, `TPU_KV_RESHARD_TRANSPORT=raiden`
  - `TPU_RAIDEN_TRANSFER_PARALLELISM=8`, `TPU_RAIDEN_RESHARD_IMPL=store`
    (ports 27000/27100), `TPU_RAIDEN_POOL_STAGING_LEASES=1`,
    `TPU_RAIDEN_PREFIX_AWARE_LOAD=1`
- **MoE / kernels:** `TPU_MOE_OWNER_OUTPUT_MODE=on`,
  `ONEHOT_MOE_PERMUTE_THRESHOLD=32768`, `RAGGED_GATHER_REDUCE_VERSION=v3`,
  `TPU_PARALLEL_PRECOMPILE=1`, `TPU_GDN_CONV_STATE_TILE_PAD=1`
- **Transfer and buffers:** `TPU_PREMAPPED_BUFFER_SIZE=4GiB`,
  `TPU_KV_TRANSFER_PORT=29200`, `TPU_SIDE_CHANNEL_PORT=19600`,
  `TPU_KV_SHM_POOL_GB=8`, `TPU_P2P_WAIT_PULL_TIMEOUT=10`

### Prefill (`l20g_prefill.config` + `products/no_v6d`)

- **MoE:** `MOE_FUSED_EP_KERNEL_IMPL=adaptive` + `MOE_REQUANTIZE_BLOCK_SIZE=256`.
  The NVFP4 experts run W4A8 on the adaptive fused-EP kernel, requantized to
  block 256.
- **Parallelism and scheduling:** `--prefill-context-parallel-size 8
  --cp-kv-cache-interleave-size 256 --block-size 768`,
  `--gpu-memory-utilization 0.75`, `--max-num-batched-tokens 32768
  --long-prefill-token-threshold 32768 --max-num-seqs 128`
- **Speculative decoding and compile:** MTP with 1 speculative token,
  `compile_sizes [512,1024,2048,4096]`
- **KV transfer:** `TPUMultiConnector` = `TPURaidenConnector` (`kv_producer`)
  + `TPURaidenOffloadingConnector` (`kv_both`, 64 GiB host DRAM, controller
  port 47800), with `kv_load_failure_policy=fail`

### Decode (`l20g_decode.config`)

- **Parallelism:** `--tensor-parallel-size 2 --data-parallel-size 4
  --data-parallel-size-local 4 --block-size 2304` (= 3 x 768)
- **Scheduling:** `--gpu-memory-utilization 0.9`, `--max-num-batched-tokens 768`,
  `--max-num-seqs 192` (per DP rank)
- **Speculative decoding and compile:** MTP with 3 speculative tokens,
  `compile_sizes [32,64,128,256,384,512,768]`
- **MoE:** default NVFP4 GMM path; the config sets no fused-EP knob on decode
- **KV transfer:** plain `TPURaidenConnector` (`kv_consumer`)

## 4. Differences from the source config

| Item in config | What this recipe does | Why |
|---|---|---|
| base `"quantization": "fp8"` | **dropped** | The checkpoint is ModelOpt NVFP4 (auto-detected as `modelopt_fp4`). Forcing `fp8` conflicts with the checkpoint's quantization config. |
| `TPU_RAIDEN_DST_SHARDS=2` (prefill only) | set on **both** roles | It must equal decode TP (2). The connector also reads it when registering blocks, and the default (1) would not match TP2 decode. |
| `${POD_IP}` | `PREFILL_IP` / `DECODE_IP` | GKE pod IP replaced by the VM's routable IP (`ens5`). |
| `TPU_RAIDEN_DATA_NICS=eth0:0,eth1:1`, `TPU_RAIDEN_CONTROL_PLANE_BACKEND`, `TPU_RAIDEN_DATA_CONN_REUSE`, `TPU_RAIDEN_TRANSFER_STREAMS`, `TPU_RAIDEN_BIND_IP`, `TPU_RAIDEN_ENABLE_SOURCE_IP_BIND`, `TPU_RAIDEN_TELEMETRY_BACKENDS`, `TPU_KV_RESHARD_DST_PAGE_TOKENS`, `TPU_VLLM_KV_CACHE_ALIAS_FALLBACK`, `USE_BATCHED_RPA_LONGCTX`, `MODEL_IMPL_TYPE` | dropped | Not read by this repo's code. These VMs have one NIC (`ens5`), not `eth0`/`eth1`. |
| `TPU_RAIDEN_CONTROLLER_ADDRESS` | dropped | Ignored when `TPU_RAIDEN_RESHARD_IMPL=store`. |
| `additional_config.raiden_pd_send_completion`, multi-connector `"backend":"kvt"` | dropped | Not read by this repo's code. |
| `DS_LLM_*`, `V6D_*`, `TAIR_MEMPOOL_*`, `SRPC_*`, `MC_TENT_CONF`, `DSV_*`, `AQUILA_*`, `VLLM_*DASHLOG*` | dropped | Belong to the internal serving wrapper, vineyard and Tair; not available here. |
| `enable_think`, `think_mode`, `disaggregation_mode`, `enable_disaggregated_prefilling`, `disaggregated_prefilling_if_name` | dropped | Wrapper arguments, not vLLM engine args. Thinking is controlled per request through the chat template. |
| `TPUV6dObjectConnector` (`kvs_connector` plugin) | replaced by the `no_v6d` overlay (`TPURaidenOffloadingConnector`) | The `kvs_connector` package is not installed. |
| `profiler_config`, `mm_encoder_tp_mode` | dropped | Profiling only; the model is served text-only. |

## 5. Things to watch on first start

- **First start is slow:** 1M `max-model-len`, large `compile_sizes` and
  `TPU_PARALLEL_PRECOMPILE` make the first start take a long time
  (`VLLM_ENGINE_READY_TIMEOUT_S=7200`). For a quick smoke test, run with
  `MAX_MODEL_LEN=65536`.
- **Expected log lines:**
  - `NVFP4 fused EP W4A8 enabled: block-256` on prefill
  - `NVFP4 GMM W4A16 enabled` on decode
  - the `tpu_online_fp8` audit passes
  - admission `pcp8_prefill` / `dp4tp2_decode`
- **Prefill HBM:** `--gpu-memory-utilization 0.75` leaves room for the Raiden
  staging buffers and offload. If prefill runs out of HBM at 1M context, lower
  `MAX_MODEL_LEN` before touching other knobs.
- **Free ports needed:**
  - API: 8400 (prefill), 9400 (decode), 8000 (proxy)
  - KV / Raiden: 14579, 27000-27007, 27100-27107, 29200+, 19600+, 47800+
  - TPU runtime: 8431-8438, 8471
