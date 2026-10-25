
there are bugs in following step, please find and fix it, an give a right recipe, save in a doc

3. Multi-Host Disaggregated Serving with Raiden (TpuRaidenConnector)¶
For large models such as Qwen/Qwen3.5-397B-A17B, Qwen/Qwen3.5-35B-A3B-FP8, moonshotai/Kimi-K3, and deepseek-ai/DeepSeek-V4, use TpuRaidenConnector with TPU_USE_RAIDEN_KV_CACHE_MANAGER=1.

Step 1: Start the Raiden Controller¶
Run the Raiden controller on a reachable host/port (default 50055):


python examples/disagg/run_raiden_controller.py \
  --host 0.0.0.0 \
  --port 50055
Step 2: Launch Prefill Instance (kv_producer)¶

export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_CONTROLLER_ADDRESS="<CONTROLLER_IP>:50055"
export TPU_RAIDEN_JOB_NAME="prefill"
export TPU_RAIDEN_ENGINE_ID="0"
export TPU_RAIDEN_QWEN35_ADMISSION=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 8400 \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 2 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --attention-backend CUSTOM \
  --kv-transfer-config '{
    "kv_connector": "TpuRaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.raiden.raiden_connector",
    "kv_role": "kv_producer"
  }'
Step 3: Launch Decode Instance (kv_consumer)¶

export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_CONTROLLER_ADDRESS="<CONTROLLER_IP>:50055"
export TPU_RAIDEN_JOB_NAME="decode"
export TPU_RAIDEN_ENGINE_ID="0"
export TPU_RAIDEN_QWEN35_ADMISSION=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 9400 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --attention-backend CUSTOM \
  --kv-transfer-config '{
    "kv_connector": "TpuRaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.raiden.raiden_connector",
    "kv_role": "kv_consumer"
  }'
Step 4: Launch the Disaggregated Proxy Server¶

python examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <PREFILL_HOST_IP> \
  --prefiller-ports 8400 \
  --decoder-hosts <DECODE_HOST_IP> \
  --decoder-ports 9400