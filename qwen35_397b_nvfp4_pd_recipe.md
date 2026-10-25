# Qwen3.5-397B-A17B-NVFP4: PD-Disaggregated Serving on TPU7x (Raiden)

Recipe for serving `nvidia/Qwen3.5-397B-A17B-NVFP4` with prefill/decode
disaggregation on `tpu7x` hosts (8 devices / host, ~96 GB HBM per device).

> **Status: not run end to end with this checkpoint.** The topology and
> runtime profile are the golden Qwen3.5-397B **FP8** Raiden pair
> (`examples/disagg/run_qwen35_raiden_pcp8_dp8_live_pair.sh`: PCP8 prefill on
> one host, DP8 decode on another). Only the checkpoint-specific settings were
> changed. In this repo, NVFP4 is validated for `Qwen3-30B-A3B` and `GLM-5.2`,
> but not yet for Qwen3.5. Run the smoke test in Step 5 before benchmarking.

## 1. What is different about this checkpoint

From the checkpoint's `hf_quant_config.json` (`quant_algo: NVFP4`,
`group_size: 16`, `kv_cache_quant_algo: FP8`):

| Part | Precision in checkpoint | Notes |
|---|---|---|
| Routed experts (512 x 60 layers) | NVFP4 (E2M1 + E4M3 block-16 scale + FP32 global) | ~97% of the params |
| GDN `linear_attn`, full-attn `self_attn` | bf16 (excluded) | replicated on every DP rank |
| Shared experts, `shared_expert_gate`, `lm_head` | bf16 (excluded) | |
| MTP layer (`mtp.layers.0*`) | bf16 (excluded) | its 512 experts are sharded by EP |

Consequences:

- **Do not pass `--quantization fp8`** (the FP8 recipes do). vLLM detects the
  ModelOpt checkpoint as `modelopt_fp4`, which maps to `VllmNvfp4Config`
  (`src/vllm_torchtpu/layers/adapter/quantization/nvfp4.py`). Passing
  `--quantization modelopt_fp4` explicitly is equivalent.
- **Default MoE path is W4A16**: fp4 weights, bf16 activations, GMM kernel.
  At load time the block-16 scales are fused into **FP32**, so each expert
  weight costs 0.5 B + 0.25 B (scale) ≈ 0.75 B/param.
- **W4A8 path**: `MOE_FUSED_EP_KERNEL_IMPL=adaptive` (fused EP kernel, fp8
  activations). It requantizes the weights to block-64 (≈0.56 B/param) and
  uses the fp8 MXU. It is experimental. Treat it as the main A/B knob (§4).

Approximate per-device HBM for DP8/PCP8 + EP8 on one host:

| | experts / device | replicated non-expert | total weights |
|---|---|---|---|
| FP8 checkpoint (reference) | ~48 GB | ~10 GB | ~58 GB |
| NVFP4, W4A16 (default) | ~36 GB | ~21 GB (bf16) | ~57 GB |
| NVFP4, W4A8 adaptive (block 64) | ~27 GB | ~21 GB (bf16) | ~48 GB |

The 397B model fits on a single 8-device host for each role. The KV/state
memory gain from NVFP4 only shows up with the W4A8 path, or when
`MOE_REQUANTIZE_BLOCK_SIZE` is set on the GMM path (§4).

## 2. Topology

Recommended starting point is **1P1D** on two hosts of the `tpu7x-32` slice:

| Role | Host | Parallelism | Raiden admission topology |
|---|---|---|---|
| Prefill (`kv_producer`) | host A, 8 devices | `TP=1, PCP=8` + EP | `pcp8_prefill` |
| Decode (`kv_consumer`) | host B, 8 devices | `TP=1, DP=8` + EP | `dp8_decode` |

Why this layout:

- It is the golden validated pair for Qwen3.5-397B (FP8).
- Qwen3.5 Raiden admission (`tpu_connector.py`
  `_raiden_qwen35_admission_topology`) only accepts these producer layouts:
  TP=1 with PCP∈{4,8}, DP∈{4,8,16}, or PP∈{2,4,8}. Consumer layouts must be
  PCP=1, TP∈{1,2}, DP∈{2,4,8,16}. **Any TP>1 prefill is rejected.**
- PCP8 splits each long prompt across all 8 devices, which gives the best
  TTFT. DP8 + EP is the fastest decode layout for this model (it is also the
  single-host throughput golden layout in `docs/deployment/v7x_recipes.md`).
- Decode alternative for lower per-token latency at low concurrency:
  `TP=2, DP=4` (`dp4tp2_decode`). It needs `TPU_RAIDEN_DST_SHARDS=2` on
  **both** roles.

## 3. Recipe

Each role runs on its own host. Use the same `RUN_ID` on both. Both hosts need
the same installed `vllm` / `vllm-torchtpu` / `torch-tpu` / `tpu-sync-torch`,
the same model snapshot, and bidirectional TCP reachability, because Raiden
advertises dynamic worker endpoints.

No standalone Raiden controller is needed. The in-engine C++ reshard store
(`TPU_RAIDEN_RESHARD_IMPL=store`) is the only working control plane, and it
ignores `TPU_RAIDEN_CONTROLLER_ADDRESS`.

### 3.0 Shared environment (both hosts)

```bash
export MODEL=nvidia/Qwen3.5-397B-A17B-NVFP4
export RUN_ID=q397nvfp4_$(date +%Y%m%d_%H%M%S)   # pick once, reuse on both hosts

export JAX_PLATFORMS=tpu,cpu
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export VLLM_XLA_CHECK_RECOMPILATION=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200        # cold compile of 397B is slow

# Golden r20 runtime profile: leave the latency-hiding scheduler, sparse-core
# MoE and ragged kernels at their defaults; do not carry legacy overrides.
unset LIBTPU_INIT_ARGS USE_MOE_SPARSE_CORE ONEHOT_MOE_PERMUTE_THRESHOLD
unset TPU_RAGGED_GATHER_REDUCE_IMPL TPU_RAGGED_GATHER_IMPL
unset DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1

# Hybrid (GDN + full attention) KV layout and Raiden Stage-3 transport.
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_TRANSFER_PARALLELISM=8       # = prefill PCP * TP; same on P and D
export TPU_GDN_CONV_QK_PAIR_LAYOUT=1
export TPU_RAIDEN_RESHARD_IMPL=store

export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export TPU_P2P_WAIT_PULL_TIMEOUT=600
export TPU_KV_SHM_POOL_GB=8
export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
export TPU_CHIPS_PER_HOST_BOUNDS=1,8,1

KV_CFG_COMMON='"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_port":14579'
CC_PREFIX='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false},'

COMMON_ARGS=(
  --trust-remote-code
  --seed 42
  --max-model-len 65536
  --enable-expert-parallel
  --disable-custom-all-reduce
  --gpu-memory-utilization 0.9
  --kv-cache-dtype fp8
  --language-model-only
  --no-disable-hybrid-kv-cache-manager
  --attention-backend CUSTOM
  --mamba-cache-mode align
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --tensor-parallel-size 1
)
```

Set `--max-model-len` to your real workload (ISL + OSL + ~20). A shorter
length leaves more HBM for decode batch size. Prefix caching is left at the
vLLM default. Add `--no-enable-prefix-caching` for synthetic random-prompt
benchmarks.

### 3.1 Prefill instance (host A)

```bash
PREFILL_IP=<HOST_A_ROUTABLE_IP>

export VLLM_HOST_IP=${PREFILL_IP}
export TPU_RAIDEN_ADVERTISE_HOST=${PREFILL_IP}
export TPU_RAIDEN_RESHARD_PORT_BASE=27000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
export TPU_KV_TRANSFER_PORT=9100
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_KV_TRANSFER_NAMESPACE=${RUN_ID}_prefill
export TPU_RAIDEN_JOB_NAME=prefill
export TPU_RAIDEN_ENGINE_ID=${RUN_ID}-prefill-engine

vllm serve ${MODEL} \
  --host 0.0.0.0 --port 8400 \
  "${COMMON_ARGS[@]}" \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 256 \
  --max-num-batched-tokens 32768 \
  --long-prefill-token-threshold 32768 \
  --max-num-seqs 8 \
  --compilation-config "${CC_PREFIX}"'"compile_sizes":[4096]}' \
  --kv-transfer-config '{'"${KV_CFG_COMMON}"',"kv_role":"kv_producer"}'
```

Notes:

- `--cp-kv-cache-interleave-size 256` comes from the golden 397B run. The
  Raiden connector requires `page_tokens % interleave == 0`. If startup fails
  on that check (as it did for the 35B model, whose hybrid page is forced to
  384 tokens), use `128`.
- `compile_sizes [4096]`: the PCP GDN kernel needs per-rank bucket rows to be a
  multiple of the interleave. 32768 tokens / 8 ranks = 4096 rows.

### 3.2 Decode instance (host B)

```bash
DECODE_IP=<HOST_B_ROUTABLE_IP>

export VLLM_HOST_IP=${DECODE_IP}
export TPU_RAIDEN_ADVERTISE_HOST=${DECODE_IP}
export TPU_RAIDEN_RESHARD_PORT_BASE=28000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
export TPU_KV_TRANSFER_PORT=9100
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_KV_TRANSFER_NAMESPACE=${RUN_ID}_decode
export TPU_RAIDEN_JOB_NAME=decode
export TPU_RAIDEN_ENGINE_ID=${RUN_ID}-decode-engine

vllm serve ${MODEL} \
  --host 0.0.0.0 --port 9400 \
  "${COMMON_ARGS[@]}" \
  --prefill-context-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --max-num-batched-tokens 4384 \
  --max-num-seqs 32 \
  --async-scheduling \
  --compilation-config "${CC_PREFIX}"'"compile_sizes":[8,16,32,4352,4384]}' \
  --kv-transfer-config '{'"${KV_CFG_COMMON}"',"kv_role":"kv_consumer"}'
```

Notes:

- `--max-num-seqs` is per DP rank, so 32 x 8 = 256 concurrent decode requests.
  At shorter `--max-model-len`, raise it (for example 64) and add the matching
  decode bucket to `compile_sizes`.
- The golden correctness run used `--no-async-scheduling`. For throughput use
  `--async-scheduling`. Switch back only if you need bit-reproducible output.

### 3.3 Proxy (any host that reaches both)

```bash
python examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts ${PREFILL_IP} --prefiller-ports 8400 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9400
```

### 3.4 Smoke test

```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Check the logs for:

- `NVFP4 MoE weights prepared (FP4 (block-16))` and `NVFP4 GMM W4A16 enabled`
  (or `NVFP4 fused EP W4A8 enabled: block-64` with the W4A8 knob);
- `TPUConnector --> using Raiden backend`;
- admission topology `pcp8_prefill` on P and `dp8_decode` on D.

Then run a short accuracy check (for example GSM8K through the proxy) and
compare it against the FP8 checkpoint before measuring performance.

## 4. Performance tuning (A/B in this order)

1. **W4A8 fused EP MoE.** This is the biggest NVFP4-specific lever. W4A16 runs
   the experts at bf16 MXU rate, which is slower than the FP8 checkpoint's
   native fp8 path on compute-bound prefill. W4A8 restores fp8 MXU and cuts
   expert weights to block-64.

   ```bash
   export MOE_FUSED_EP_KERNEL_IMPL=adaptive   # on decode first, then prefill
   ```

   The adaptive kernel is experimental. It raises an error instead of falling
   back. It has unit coverage at 2048 and 16384 tokens per layer, but no
   full-model performance validation. If it fails, the production v2 kernel
   is an alternative: `USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_ENABLE_W4A8=1`.
2. **Smaller scale overhead on the GMM path.** If you stay on W4A16, set
   `MOE_REQUANTIZE_BLOCK_SIZE=64` (or `128`). This cuts the FP32 scale cost
   from 0.25 to 0.06 (or 0.03) B/param, about 9 GB per device, and turns it
   into KV / batch space. Check accuracy after changing it.
3. **MTP speculative decoding** (the MTP head is in the checkpoint, in bf16).
   It helps decode latency at low to mid concurrency:
   `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` on
   decode, `'{"method":"mtp","num_speculative_tokens":1}'` on prefill. These
   are the defaults in `launch_qwen35_disagg_multihost.sh`. At high
   concurrency it can cost throughput, so measure both.
4. **Decode batch size.** Sweep `--max-num-seqs` per rank against the TPOT SLO,
   using the memory freed by items 1 and 2.
5. **P:D ratio on the 4-host slice.** 1P1D uses 2 of the 4 hosts. Use
   **1P3D** when OSL dominates (decode-bound, e.g. 1K/8K) and **2P2D** for
   long-context, prefill-heavy traffic (e.g. 32K/1K). List every instance in
   the proxy's `--prefiller-hosts/--decoder-hosts`. Each instance needs a
   unique `TPU_RAIDEN_ENGINE_ID` and `TPU_KV_TRANSFER_NAMESPACE`. Multi-instance
   Raiden has not been validated for Qwen3.5. Validate 1P1D first.

## 5. Common failures

| Symptom | Cause / fix |
|---|---|
| `Raiden Qwen3.5 admission topology ... requires kv_producer with tensor_parallel_size=1` | Prefill must be TP=1 with PCP∈{4,8} (or DP∈{4,8,16}, PP∈{2,4,8}). |
| Unknown connector `TpuRaidenConnector` | Use `TPURaidenConnector` and module path `vllm_torchtpu.distributed.kv_transfer.tpu_connector`. |
| Hangs on first transfer | Wrong `TPU_RAIDEN_ADVERTISE_HOST` (must be routable from the peer host), or a firewall on ports 27000-27199 / 28000-28199 / 9100+ / 9600+ / 14579. |
| `page_tokens % interleave` error on prefill | Use `--cp-kv-cache-interleave-size 128`. |
| HBM OOM at load on decode | Lower `--max-model-len` / `--max-num-seqs`, or apply §4 item 1 or 2. |
| Weight-loading error on `linear_attn` / `self_attn` scales | The ModelOpt exclude list was not mapped to vLLM names. Report it. This path has not been exercised for Qwen3.5 on TPU. |
