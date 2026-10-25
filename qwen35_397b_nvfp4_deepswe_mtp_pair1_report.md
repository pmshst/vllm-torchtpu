# DeepSWE MTP Acceptance Test: Pair 1 (64K context) Report

| | |
|---|---|
| Goal | Reproduce the reported "MTP acceptance rate drops to 0%" issue on Qwen3.5-397B with a pcp8 -> dp4tp2 PD configuration under DeepSWE agentic load |
| Test window | 2026-10-25 07:38:14 -> 08:38:14 UTC (1 h), last agents drained by 08:42 |
| Server run ID | `q397nvfp4_1025_0553` |
| Results | `/zhaoleo/logs/deepswe_mtp_live_q397nvfp4_1025_0553/` |
| **Result** | **Not reproduced.** 0 zero-acceptance windows; overall acceptance 85.6%, never below 68.4% in any 15 s window on any DP engine |

## 1. Environment

### Hardware

| Item | Value |
|---|---|
| Slice | `tpu7x-32` (4 hosts), GCE machine type `tpu7x-ultranet-4t` |
| Hosts used | drzl `10.128.0.182` (prefill + proxy + load clients), gw7c `10.128.0.147` (decode) |
| Per host | 8 TPU7x devices (~96 GB HBM each), 224 vCPU, 944 GB RAM |
| TPU mode | Each host runs as a standalone tpu7x-8 (`TPU_HOST_BOUNDS=1,1,1`, `TPU_TOPOLOGY=2x2x1`) |
| OS / kernel | Ubuntu 24.04.5 LTS, Linux `6.8.0-1070-gcp` (both hosts) |
| Storage | Model and compile caches on NFS `10.128.0.45:/zhaoleo` |

### Software: serving venv (`/zhaoleo/vllm-torchtpu/.venv`, identical on both hosts)

| Package | Version |
|---|---|
| Python | 3.12.3 |
| vllm | 0.32.0+empty |
| vllm-torchtpu (vllm-torchtpu-core, editable) | 0.1.dev2841+g5b4512bdd |
| vllm-torchtpu git commit | `5b4512bdd9d8470b1a1dcc04909af31190cbc287` (2026-10-22, "[Omni] Materialize the Wan timestep before the transformer on TPU (#1451)"). Working tree: only `examples/disagg/toy_proxy_server.py` modified (extra decode-error logging); no `src/` changes |
| torch | 2.13.0+cpu |
| torch-tpu | 0.2.0 |
| libtpu | 0.0.50 |
| jax / jaxlib | 0.11.2 / 0.11.2 |
| tpu-sync-torch (Raiden) | 0.0.1.dev20261012084241 |
| transformers | 5.19.0 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.33.0 |
| triton | 3.8.0 |
| numpy | 2.3.5 |

### Software: load-client venv (`/zhaoleo/deepswe/agent-venv`, drzl)

| Package | Version |
|---|---|
| Python | 3.12.3 |
| mini-swe-agent | 2.4.6 |
| litellm | 1.106.3 |
| openai | 2.54.0 |
| pydantic | 2.14.1 |
| Docker | 29.9.1 (build dfc520d) |

### Model and dataset

| Item | Value |
|---|---|
| Model | `nvidia/Qwen3.5-397B-A17B-NVFP4`, HF snapshot `12b28061efd8145b11226e4fab2528ee4f39ef09` |
| Dataset | DeepSWE, `github.com/datacurve-ai/deep-swe` @ `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` (2026-08-26), 113 tasks (Harbor format) |
| Task images | `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1` (113 prebuilt images) |

## 2. Server configuration

Launcher: `serve_qwen35_397b_nvfp4_pd.sh`. It is translated from
`0522-thinking-re-nvfp4-inputscale/` (`l20g` + `l20g_{prefill,decode}` +
`products/no_v6d`); details are in `qwen35_397b_nvfp4_pd_l20g_recipe.md`.

| | Prefill (drzl) | Decode (gw7c) |
|---|---|---|
| Parallelism | TP1 x **PCP8**, EP8 | **TP2 x DP4**, EP8 |
| Raiden admission | `pcp8_prefill` | `dp4tp2_decode` |
| Speculative decoding | MTP, `num_speculative_tokens=1` | **MTP, `num_speculative_tokens=3`** |
| MoE | NVFP4 -> adaptive fused-EP W4A8, requant block 256 | NVFP4 GMM W4A16 |
| Online FP8 | attn / GDN / shared-expert projections (285 layers) | same |
| Block size / KV | 768, fp8, 9.43M tokens | 2304, fp8, 3.72M tokens per DP rank |
| max_num_seqs / batched tokens | 128 / 32768 | 192 per rank / 768 |
| **max_model_len** | **65536** (bring-up shortcut; the production config uses 1,048,576) | 65536 |
| Other | prefix caching, async scheduling, `VLLM_KV_CACHE_LAYOUT=HND`, `TPU_RAIDEN_PREFIX_AWARE_LOAD=1`, `TPU_RAIDEN_DST_SHARDS=2` | same |
| KV connector | `TPUMultiConnector` (Raiden producer + 64 GiB host offload) | `TPURaidenConnector` consumer |

## 3. Test method

Harness: `run_deepswe_mtp_test.sh` (MODE=live) + `deepswe_live_agent.py`.

- **Agent:** mini-swe-agent v2 (the agent used on the DeepSWE leaderboard),
  using the text-based SWE-bench prompts (`swebench_backticks.yaml`).
  - The model's reply is parsed for one bash command, which is executed in the
    task's container (`/app`, `--network none`, 2 CPUs, 8 GB per `task.toml`).
    The output is fed back as the next turn.
  - Qwen thinking is on (the template default). The `<think>` block is
    stripped before parsing.
- **Sampling:** temperature 0.6, top_p 0.95, max_tokens 8192 per turn,
  step limit 250.
- **Load:** **3 clients**, each shuffling the 113 tasks with its own seed,
  **16 tasks in flight per client** (48 concurrent agents). Rounds repeat
  until 1 h; agents still running at the deadline stop via wall-time limit.
- **Endpoint:** PD proxy `toy_proxy_server.py` at drzl:8000 -> prefill
  :8400 -> Raiden -> decode :9400.
- **Monitor:** every 15 s, scrapes decode `/metrics`
  (`vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`, plus
  per position). It records the per-window delta acceptance per DP engine and
  overall, and flags windows with drafts > 0 and acceptance <= 1%.

## 4. Results

### 4.1 MTP acceptance (decode, MTP-3)

| Metric | Value |
|---|---|
| Windows (15 s) with traffic | 243 (07:38–08:39), plus 15 idle windows after the end of the test |
| Total drafts / draft tokens / accepted | 1,839,412 / 5,517,980 / 4,720,819 |
| **Overall acceptance rate** | **85.55%** |
| **Mean acceptance length** (tokens per step incl. bonus, max 4) | **3.57** |
| Per-window acceptance min / p5 / p50 / p95 / max | 73.8% / 80.9% / 85.5% / 88.8% / 92.8% |
| Window stdev | 2.7 pts |
| Per-position acceptance (draft-weighted) | pos1 **92.9%**, pos2 **85.6%**, pos3 **78.2%** |
| **Zero-acceptance windows (<= 1%), any engine** | **0** |
| Windows < 50%, any engine | **0** |

Per DP engine (whole test):

| Engine | Drafts | Acceptance | Mean len | Window min / max |
|---|---|---|---|---|
| DP0 | 457,931 | 85.5% | 3.57 | 69.8% / 99.2% |
| DP1 | 459,240 | 85.6% | 3.57 | 69.5% / 100% |
| DP2 | 458,815 | 85.9% | 3.58 | 68.4% / 96.4% |
| DP3 | 463,426 | 85.2% | 3.56 | 68.8% / 93.9% |

Timeline (5-minute buckets, all engines):

| Bucket (UTC) | Drafts | Drafts/s | Acceptance | Mean len |
|---|---|---|---|---|
| 07:35 (from 07:38) | 25,195 | n/a | 79.0% | 3.37 |
| 07:40 | 151,631 | 505 | 83.7% | 3.51 |
| 07:45 | 157,555 | 525 | 84.7% | 3.54 |
| 07:50 | 145,631 | 485 | 84.8% | 3.54 |
| 07:55 | 149,996 | 500 | 84.0% | 3.52 |
| 08:00 | 161,210 | 537 | 86.0% | 3.58 |
| 08:05 | 148,205 | 494 | 85.7% | 3.57 |
| 08:10 | 144,251 | 481 | 85.8% | 3.57 |
| 08:15 | 137,490 | 458 | 85.5% | 3.57 |
| 08:20 | 142,775 | 476 | 85.5% | 3.56 |
| 08:25 | 148,874 | 496 | 86.9% | 3.61 |
| 08:30 | 178,090 | 594 | 86.1% | 3.58 |
| 08:35 | 148,509 | 495 | 89.1% | 3.67 |

Acceptance does not degrade over the hour. It rises slightly (83.7% -> 89.1%)
as agent histories lengthen and contain more repeated code and tool output.

### 4.2 Load and agent outcomes

| Metric | Value |
|---|---|
| Agent steps (LLM calls) completed | 19,179 (~5.3/s sustained); prefill served 19,241 chat requests with HTTP 200 |
| Tasks finished | 195 (109 distinct tasks; each client completed 1 shuffled round before the deadline) |
| Steps per task | median 98, max 211 |
| Task wall time | median 924 s, max 2,026 s |
| Context per transferred request (Raiden events) | p50 ~30.8K, p90 ~58.6K, max 65.5K tokens |

| Exit status | Count | Median steps | Meaning |
|---|---|---|---|
| `InternalServerError` | 100 | 112 | Agent history exceeded **max_model_len 65536**. Prefill returns 400 (`maximum context length is 65536`), the proxy returns 500, and litellm retries, then gives up. 2,999 such 400s were seen. A test-config limit, not a server fault. |
| `RepeatedFormatError` | 44 | 103 | 3 consecutive replies without exactly one command block (agent/prompt-format issue, independent of MTP) |
| `TimeExceeded` | 38 | 47 | Still running at the 1 h deadline |
| `Submitted` | 13 | 84 | Agent finished and submitted a patch (not graded) |

Per client: client 0 = 1 / 35 / 14 / 13, client 1 = 5 / 31 / 17 / 12, client 2 =
7 / 34 / 13 / 13 (Submitted / ISE / FormatError / TimeExceeded).

### 4.3 Server health

| Check | Result |
|---|---|
| Prefill / decode `Traceback` or `EngineCore failed` | 0 / 0 |
| Proxy decode-side errors (`DECODE-ERROR`) | 0 |
| Process restarts | none (servers up since 06:20 / 06:43) |
| Raiden reshard | 8 source ranks -> 2 destination units per request; plan + controller ~6–7 ms |

## 5. Conclusions

1. **The 0% MTP acceptance drop did not reproduce** on pair 1. Conditions:
   pcp8 -> dp4tp2, NVFP4, MTP-3 on decode, 48 concurrent live DeepSWE agents
   for 1 h, contexts up to 64K. Every 15 s window on every DP engine stayed
   >= 68%, and overall acceptance was 85.6% (3.57 tokens/step).
2. **Long-context coverage is limited by max_model_len=65536.** About half
   of finished tasks (100/195) stopped when their history passed 64K tokens,
   so the >64K regime that production (1M) serves was **not** exercised here.
   If the drop depends on long contexts, it would not show in this run.
3. Pair 1 therefore serves as the **64K control**. The same test is running
   on **pair 2 (svps -> wx0l) at max_model_len = 1,048,576**, matching the
   production `l20g` config.

## 6. Differences from the reported production setup

- `max_model_len` 65536 here vs 1,048,576 in the `l20g` config (addressed by pair 2).
- The vineyard/v6d and Tair KV tiers in the production config are not
  available. Prefill used the `products/no_v6d` variant (Raiden host
  offload) instead.
- Client harness: mini-swe-agent text-based prompts on the open-source task
  set, which may differ from the internal DeepSWE client (prompt format,
  sampling, tool-call style).
- A single PD pair per test (1P1D). Production may run more instances
  behind its router.

## 7. Artifacts

| File | Content |
|---|---|
| `mtp_acceptance.csv` | Per-15 s window, per engine: drafts, draft tokens, accepted, rate, mean length, per-position rates, zero flag |
| `monitor.log` | Human-readable monitor output |
| `client{0,1,2}/tasks.csv` | Per task: start/end, round, task id, exit status, steps, duration, error |
| `client{0,1,2}/traj/*.traj.json` | Full agent trajectories (messages, actions, outputs) |
| `client{0,1,2}.log` | Client stdout |
| `/zhaoleo/logs/q397nvfp4_1025_0553_{prefill,decode,proxy}.log` | Server and proxy logs |

All of these are in `/zhaoleo/logs/deepswe_mtp_live_q397nvfp4_1025_0553/`
except the server and proxy logs.
