# Multi-Host Disaggregated Serving with Raiden (`TPURaidenConnector`): Fixed Recipe

This doc reviews the recipe in section 3 of
`docs/user_guide/disaggregated_serving.md` ("Multi-Host Disaggregated Serving
with Raiden"). It lists the bugs and gives a corrected recipe. Every finding
was checked against the source code and the two shipped launchers,
`examples/disagg/launch_qwen35_disagg_multihost.sh` and
`examples/disagg/run_qwen35_raiden_pcp8_dp8_live_pair.sh`.

## Bugs in the original recipe

| # | Where | Bug | Evidence |
|---|-------|-----|----------|
| 1 | Steps 2 and 3, `kv_connector` | `"TpuRaidenConnector"` does not exist. The class is `TPURaidenConnector`. The platform asserts the connector name is in an allow-list, so startup fails. | `platforms/tpu_platform.py` (`_TPU_SUPPORTED_KV_CONNECTORS`), `distributed/kv_transfer/tpu_connector.py` (`class TPURaidenConnector`) |
| 2 | Steps 2 and 3, `kv_connector_module_path` | `vllm_torchtpu.distributed.kv_transfer.raiden.raiden_connector` does not exist (the `raiden/` package has no `raiden_connector.py`). The connector lives in `vllm_torchtpu.distributed.kv_transfer.tpu_connector`. | `tpu_platform.py:365` registers `("TPURaidenConnector", "vllm_torchtpu.distributed.kv_transfer.tpu_connector")` |
| 3 | Step 1 | `run_raiden_controller.py` takes only `--port`, and `--port` is required. `--host 0.0.0.0` makes argparse exit with `unrecognized arguments`. "Default 50055" is also wrong because there is no default. | `examples/disagg/run_raiden_controller.py` |
| 4 | Step 1, `TPU_RAIDEN_CONTROLLER_ADDRESS` | The external Python controller path no longer works: "latest tpu-sync removed the Python pool-reshard path (`RaidenController._start_pool_reshard_transfer`), so the in-engine C++ reshard store is the only control plane that still works." Use `TPU_RAIDEN_RESHARD_IMPL=store`. In store mode, `TPU_RAIDEN_CONTROLLER_ADDRESS` is ignored and no controller process is needed. Store mode requires `TPU_RAIDEN_ADVERTISE_HOST`. | `run_qwen35_raiden_pcp8_dp8_live_pair.sh` (comment in `configure_role_environment`), `tpu_connector.py` `_resolved_reshard_controller_address` and `_reshard_advertise_host` |
| 5 | Steps 2 and 3, env | `TPU_KV_RESHARD_TRANSPORT=raiden` is missing. Without it, the Raiden Stage-3 transport (the explicit-pool reshard path that `TPU_RAIDEN_QWEN35_ADMISSION` is for) is never turned on, because the default is `zmq`. | `tpu_connector.py` `_use_raiden_stage3_transport`; `envs.py` (`TPU_KV_RESHARD_TRANSPORT` default `"zmq"`) |
| 6 | Step 2, prefill topology | `--tensor-parallel-size 4 --prefill-context-parallel-size 2` is rejected by Qwen3.5 Raiden admission. A Qwen3.5 producer must have `tensor_parallel_size=1` and either `prefill_context_parallel_size ∈ {4, 8}`, or `data_parallel_size ∈ {4, 8, 16}`, or `pipeline_parallel_size ∈ {2, 4, 8}` (TP>1 is allowed only with the HND seq-on-lane layout and DP=1). Startup fails with `Raiden Qwen3.5 admission topology pcp8_prefill, pcp4_prefill, ... requires kv_producer with tensor_parallel_size=1 ...`. | `tpu_connector.py` `_raiden_qwen35_admission_topology` |
| 7 | Steps 2 and 3, env | `TPU_RAIDEN_TRANSFER_PARALLELISM` must equal the producer's `PCP × TP` rank count, and both roles must export the same value. It also selects the GDN conv QK-pair layout, which must match on P and D. The original sets nothing; it only works if the default of 8 happens to match. Set it explicitly. | `tpu_connector.py` `_validate_stage3_transfer_parallelism`; `env_override.py` (`TPU_GDN_CONV_QK_PAIR_LAYOUT`) |
| 8 | Step 3, decode `TP=2` | A head-sharded decode (`dp{N}tp2_decode`) needs `TPU_RAIDEN_DST_SHARDS=<decode TP>` on **both** roles. The producer uses it to route KV and GDN head shards. If unset, it defaults to 1, which does not match TP=2. | `envs.py` (`TPU_RAIDEN_DST_SHARDS` comment), `tpu_connector.py` `_raiden_dst_shards`, `_validate_gdn_transfer_geometry` |
| 9 | Steps 2 and 3, flags | Qwen3.5 is multimodal: both launchers pass `--language-model-only`. They also pass `--mamba-cache-mode align` and `--no-disable-hybrid-kv-cache-manager`. Prefix caching on a hybrid Mamba model raises `NotImplementedError` unless `mamba_cache_mode='align'`. | `tpu_platform.py` (hybrid prefix-caching check); both launchers' `common_args` |
| 10 | Steps 2 and 3, `--block-size 256` | This is not a hard error, but it is misleading. For hybrid models a user `--block-size` is only a *floor*: each role raises it to fit the pooled GDN state (thousands of tokens). Leave it out, or treat it as a floor shared by both roles. | `platforms/tpu_block_size_utils.py` `_derive_tpu_block_slot_config` (`user_block_size_floor`) |
| 11 | Step 3 | Multi-host DP decode should pin `--data-parallel-size-local` to the same value as `--data-parallel-size` (both launchers do), so all DP ranks run on the decode host. | both launchers |

Minor: `TPU_RAIDEN_ENGINE_ID="0"` for both roles is legal because the job
name disambiguates them. A per-run unique ID, as in the launchers, keeps
stale registrations from a previous run from colliding.

## Corrected recipe (Qwen3.5-35B-A3B-FP8, prefill host + decode host, 8 chips each)

This topology is the validated one used by both shipped launchers:
**Prefill PCP=8, TP=1 → Decode DP=8, TP=1**.

Store mode needs no controller process. Each engine's rank-0 worker hosts
its reshard store at `TPU_RAIDEN_RESHARD_PORT_BASE + dp_rank` (default
27000) and runs its dispatch controller at
`TPU_RAIDEN_STORE_DISPATCH_PORT_BASE + dp_rank` (default 27100). These
ports, plus the KV/side-channel ports, must be reachable between the two
hosts.

### Step 1: (Removed) Raiden controller

Skip this step: store mode replaces the external controller. If you still
want one for debugging, the only valid invocation is
`python examples/disagg/run_raiden_controller.py --port 50055`.

### Step 2: Launch the prefill instance (kv_producer) on the prefill host

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)          # use the same RUN_ID on both hosts
PREFILL_IP=<PREFILL_HOST_ROUTABLE_IP>

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_TRANSFER_PARALLELISM=8      # = prefill PCP * TP
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="${PREFILL_IP}"
export TPU_RAIDEN_RESHARD_PORT_BASE=27000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
export TPU_RAIDEN_JOB_NAME=prefill
export TPU_RAIDEN_ENGINE_ID="prefill-engine-${RUN_ID}"
export TPU_KV_TRANSFER_NAMESPACE="prefill_${RUN_ID}"
export VLLM_HOST_IP="${PREFILL_IP}"

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8400 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 256 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --mamba-cache-mode align \
  --no-disable-hybrid-kv-cache-manager \
  --max-model-len 8192 \
  --kv-transfer-config '{
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    "kv_role": "kv_producer",
    "kv_port": 14579
  }'
```

### Step 3: Launch the decode instance (kv_consumer) on the decode host

```bash
DECODE_IP=<DECODE_HOST_ROUTABLE_IP>

export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_TRANSFER_PARALLELISM=8      # same value as prefill
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="${DECODE_IP}"
export TPU_RAIDEN_RESHARD_PORT_BASE=27000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
export TPU_RAIDEN_JOB_NAME=decode
export TPU_RAIDEN_ENGINE_ID="decode-engine-${RUN_ID}"
export TPU_KV_TRANSFER_NAMESPACE="decode_${RUN_ID}"
export VLLM_HOST_IP="${DECODE_IP}"

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 9400 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --mamba-cache-mode align \
  --no-disable-hybrid-kv-cache-manager \
  --max-model-len 8192 \
  --kv-transfer-config '{
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    "kv_role": "kv_consumer",
    "kv_port": 14579
  }'
```

#### Variant: head-sharded decode (DP=4, TP=2), closer to the original intent

Keep prefill at PCP=8, TP=1. On the decode host, replace the parallelism
flags with `--tensor-parallel-size 2 --data-parallel-size 4
--data-parallel-size-local 4`. Then export the following on **both** hosts,
before launching either server:

```bash
export TPU_RAIDEN_DST_SHARDS=2   # = decode TP size; identical on P and D
```

This is admitted as `pcp8_prefill → dp4tp2_decode` (unit-tested in
`tests/distributed/test_tpu_connector.py::test_v1_admission_accepts_tp2_decode_topology`).
The end-to-end validated pair is still DP=8, TP=1.

### Step 4: Launch the disaggregated proxy server

This step was already correct:

```bash
python examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <PREFILL_HOST_IP> \
  --prefiller-ports 8400 \
  --decoder-hosts <DECODE_HOST_IP> \
  --decoder-ports 9400
```

### Step 5: Smoke test

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.5-35B-A3B-FP8", "prompt": "The capital of France is", "max_tokens": 16}'
```

The logs should show `TPUConnector --> using Raiden backend` and a Raiden
admission topology of `pcp8_prefill` on P and `dp8_decode` (or
`dp4tp2_decode`) on D.

## Simplest path

The repo launcher already encodes all of the above. Run it once per host
with the same `RUN_ID`:

```bash
# prefill host
ROLE=prefill RUN_ID=<id> PREFILL_ADVERTISE_HOST=<PREFILL_IP> \
  bash examples/disagg/launch_qwen35_disagg_multihost.sh
# decode host
ROLE=decode  RUN_ID=<id> DECODE_ADVERTISE_HOST=<DECODE_IP> \
  bash examples/disagg/launch_qwen35_disagg_multihost.sh
```
