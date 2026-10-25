# DeepSWE MTP Acceptance Test: Final Report (Pair 1 @ 64K, Pair 2 @ 1M)

| | |
|---|---|
| Goal | Reproduce the reported "MTP acceptance rate drops to 0%" issue on Qwen3.5-397B, pcp8 -> dp4tp2 PD, under DeepSWE load (3 clients x 113 shuffled tasks, concurrency 16, repeated until a time limit) |
| Date | 2026-10-25 |
| **Result** | **Not reproduced on either pair.** 0 zero-acceptance windows in 486 active 15 s windows. Overall MTP-3 acceptance was **85.6%** (64K) and **86.4%** (1M). No DP engine ever went below 68.4% in any window. |
| Per-pair detail | `qwen35_397b_nvfp4_deepswe_mtp_pair1_report.md` (pair 1, full environment list) |

## 1. Environment

Both pairs use the same software, host type and model.

| Item | Value |
|---|---|
| Hardware | `tpu7x-32` slice, 4 VMs (`tpu7x-ultranet-4t`). Each host: 8 TPU7x devices (~96 GB HBM each), 224 vCPU, 944 GB RAM. Each host runs as a standalone tpu7x-8. |
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-1070-gcp |
| vllm-torchtpu | commit `5b4512bdd9d8470b1a1dcc04909af31190cbc287` (2026-10-22), vllm-torchtpu-core 0.1.dev2841+g5b4512bdd; no `src/` changes |
| vllm | 0.32.0+empty |
| torch / torch-tpu / libtpu | 2.13.0+cpu / 0.2.0 / 0.0.50 |
| jax / jaxlib | 0.11.2 / 0.11.2 |
| tpu-sync-torch (Raiden) | 0.0.1.dev20261012084241 |
| transformers / tokenizers / safetensors / huggingface-hub | 5.19.0 / 0.23.3 / 0.8.0 / 1.33.0 |
| triton / numpy / Python | 3.8.0 / 2.3.5 / 3.12.3 |
| Load client | mini-swe-agent 2.4.6, litellm 1.106.3, openai 2.54.0, pydantic 2.14.1, Docker 29.9.1 |
| Model | `nvidia/Qwen3.5-397B-A17B-NVFP4`, snapshot `12b28061efd8145b11226e4fab2528ee4f39ef09` |
| Dataset | `github.com/datacurve-ai/deep-swe` @ `0b9fabbb` (113 tasks, prebuilt images `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`) |

## 2. Configuration

Launcher: `serve_qwen35_397b_nvfp4_pd.sh` (from the `l20g` production config;
see `qwen35_397b_nvfp4_pd_l20g_recipe.md`).

| | Prefill | Decode |
|---|---|---|
| Parallelism / admission | TP1 x PCP8, EP8 / `pcp8_prefill` | TP2 x DP4, EP8 / `dp4tp2_decode` |
| MTP | 1 speculative token | **3 speculative tokens** |
| MoE | NVFP4 -> adaptive fused-EP W4A8, block 256 | NVFP4 GMM W4A16 |
| Online FP8 | attn / GDN / shared-expert (285 layers) | same |
| Block / KV dtype | 768 / fp8 | 2304 / fp8 |
| Other | prefix caching, async scheduling, `VLLM_KV_CACHE_LAYOUT=HND`, `TPU_RAIDEN_PREFIX_AWARE_LOAD=1`, `TPU_RAIDEN_DST_SHARDS=2`; prefill `TPUMultiConnector` (Raiden + 64 GiB host offload) | `TPURaidenConnector` consumer |

| | Pair 1 | Pair 2 |
|---|---|---|
| Hosts | drzl `10.128.0.182` -> gw7c `10.128.0.147` | svps `10.128.0.181` -> wx0l `10.128.0.180` |
| Server run ID | `q397nvfp4_1025_0553` | `q397nvfp4_mtp1m_1025_0758` |
| **max_model_len** | **65,536** (control) | **1,048,576** (production value) |
| Prefill KV capacity | 9.43M tokens | 16.76M tokens |
| Decode KV capacity / DP rank | 3.72M tokens | 5.56M tokens |
| Test window (UTC) | 07:38 -> 08:39 | 08:43 -> 09:45 |
| Results dir | `/zhaoleo/logs/deepswe_mtp_live_q397nvfp4_1025_0553/` | `/zhaoleo/logs/deepswe_mtp_live_q397nvfp4_mtp1m_1025_0758/` |

## 3. Test method (identical for both pairs)

Harness: `run_deepswe_mtp_test.sh` (MODE=live) + `deepswe_live_agent.py`.

- **Agent:** mini-swe-agent v2 (text-based SWE-bench prompts). The model's
  command runs in the task's container (`/app`, no network, 2 CPUs, 8 GB).
  Thinking is on; the `<think>` block is stripped before parsing.
- **Sampling and limits:** temperature 0.6, top_p 0.95, max 8192 tokens per
  turn, 250-step limit.
- **Load:** 3 clients x 16 concurrent agents (48 agents), each client
  shuffling the 113 tasks independently, repeated for 1 h. Agents still
  running at the deadline stop via wall-time limit.
- **Monitor:** decode `/metrics` scraped every 15 s. It computes the
  acceptance rate per window per DP engine, plus per position, and flags
  windows with drafts > 0 and acceptance <= 1%.

## 4. Results

### 4.1 MTP acceptance

| Metric | Pair 1 (64K) | Pair 2 (1M) |
|---|---|---|
| Active 15 s windows | 243 | 243 |
| Drafts / draft tokens / accepted | 1,839,412 / 5,517,980 / 4,720,819 | 1,856,165 / 5,568,495 / 4,811,045 |
| **Acceptance rate** | **85.55%** | **86.40%** |
| **Mean acceptance length** (max 4) | **3.57** | **3.59** |
| Window min / p5 / p50 / p95 / max | 73.8 / 80.9 / 85.5 / 88.8 / 92.8% | 75.0 / 80.9 / 86.6 / 89.6 / 100%* |
| Window stdev | 2.7 pts | 3.0 pts |
| Per position (pos1 / pos2 / pos3) | 92.9 / 85.6 / 78.2% | 93.3 / 86.4 / 79.6% |
| Per-engine acceptance (DP0 / DP1 / DP2 / DP3) | 85.5 / 85.6 / 85.9 / 85.2% | 86.3 / 86.7 / 86.4 / 86.2% |
| Lowest single window per engine | 69.8 / 69.5 / 68.4 / 68.8% | 68.6 / 68.6 / 69.1 / 73.0% |
| **Zero-acceptance windows (<= 1%), any engine** | **0** | **0** |
| Windows < 50%, any engine | 0 | 0 |

\* 100% windows occurred only on low-traffic per-engine windows (few drafts).

Timeline (5-minute buckets, overall acceptance):

| Minute of test | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | 40–45 | 45–50 | 50–55 | 55–60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pair 1 (64K) | 83.7 | 84.7 | 84.8 | 84.0 | 86.0 | 85.7 | 85.8 | 85.5 | 85.5 | 86.9 | 86.1 | 89.1 |
| Pair 2 (1M) | 84.6 | 87.1 | 86.1 | 86.6 | 84.8 | 87.6 | 85.0 | 86.1 | 87.0 | 86.3 | 88.0 | 88.1 |

Neither pair shows any downward trend. Acceptance drifts slightly **up** as
agent histories get longer.

### 4.2 Load and context coverage

| Metric | Pair 1 (64K) | Pair 2 (1M) |
|---|---|---|
| Agent steps (LLM calls) | 19,179 | 16,612 |
| Prefill HTTP 200 / 400 | 19,241 / 2,999 | 16,612 / **0** |
| Draft rate (decode) | ~500 drafts/s | ~520 drafts/s |
| Context per transfer (Raiden sender events) p50 / p90 / max | ~31K / ~59K / 65.5K (capped) | **48K / 99K / 203K** |
| Transfers > 64K tokens | 0 (rejected by the cap) | **43,296 events (~33%)** |
| Decode prefix-cache hit rate | ~82% | ~89% |
| Preemptions (all engines) | 0 | 0 |

**Pair 2 exercised the long-context regime that pair 1 could not.** A third
of its KV transfers were longer than 64K, up to 203K tokens. Acceptance
there was **slightly higher**, not lower.

### 4.3 Agent outcomes

| Exit status | Pair 1 (64K) | Pair 2 (1M) | Meaning |
|---|---|---|---|
| `Submitted` | 13 | **56** | Agent finished and submitted a patch (not graded) |
| `InternalServerError` | 100 | **0** | History exceeded 64K, so prefill returned 400 (context-length cap) |
| `RepeatedFormatError` | 44 | 0 | 3 replies in a row without exactly one command block |
| `LimitsExceeded` | 0 | 14 | Hit the 250-step limit (only reachable with long contexts) |
| `TimeExceeded` | 38 | 47 | Still running at the 1 h deadline |
| Tasks finished (distinct) | 195 (109) | 117 (85) | Each client completed 1 shuffled round |
| Steps per task, median / max | 98 / 211 | 135 / 250 | |

### 4.4 Server health

| Check | Pair 1 | Pair 2 |
|---|---|---|
| `Traceback` / `EngineCore failed` (prefill, decode) | 0, 0 | 0, 0 |
| Proxy decode-side errors | 0 | 0 |
| Restarts during test | none | none |
| Notable warnings | `VLLM_DP_Coordinator ... out-of-order step` (stats ordering, benign; 1,435x) | same (benign) |

## 5. Conclusions

1. **The 0% MTP acceptance drop did not reproduce** under the reported
   method: pcp8 -> dp4tp2, NVFP4, MTP-3, 3 clients x 113 shuffled DeepSWE
   tasks at concurrency 16, for 1 h on each of two independent PD pairs.
   This holds at both 64K and the production 1M context.
2. **Long contexts are not the trigger on this build.** Pair 2 transferred
   43K contexts longer than 64K (up to 203K) with 86.4% acceptance and a
   per-window floor of ~69%.
3. Acceptance is stable over time, uniform across the 4 DP engines, and
   degrades in the normal way by draft position (pos1 ~93% -> pos3 ~79%).

## 6. Why it may not reproduce here, and next steps

| Not exercised here | Suggested next test |
|---|---|
| **Older production build.** This build (2026-10-22) includes 8 MTP/PD fixes from 09-29..10-15, e.g. `84e3425af` "MTP: keep GDN state across scheduling gaps", `f37321669` "pad EP-DP lockstep draft steps", `ce4169c91` "greedy rows in a mixed batch", `e9b5870c3` / `87529ec41` "draft KV write ordering vs Raiden publication" | Redeploy one pair on the **exact production commit**; if it reproduces there, bisect across those commits |
| **KV pressure / scheduling gaps:** 0 preemptions on both pairs | Shrink decode KV (`--num-gpu-blocks-override` / lower `--gpu-memory-utilization`) or run 3 x 64 agents until preemptions occur |
| **Idle DP engines:** all 4 engines were busy at all times | Low or bursty load (total concurrency 1–4, with pauses) |
| **Mixed greedy and sampled requests** | One client at temperature 0, two at 0.6 |
| **Production KV tiers:** v6d/Tair with `kv_load_failure_policy: recompute` (we used `no_v6d` + `fail`) | Enable the production connector stack if available |
| **Env differences:** `VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP`, `TPU_RAIDEN_STAGE3_REGISTER_AHEAD` | Diff the production env against ours |
| **Measurement artifact:** a window with 0 drafts (idle engine) can be shown as "0%" by a naive dashboard | Confirm the drafted count was > 0 when 0% was observed |
| **Duration:** 1 h per pair | 6–12 h soak combined with KV pressure |

## 7. Artifacts

Each results dir contains:

- `mtp_acceptance.csv`: per 15 s window, per engine
- `monitor.log`
- `client{0,1,2}/tasks.csv`
- `client{0,1,2}/traj/*.traj.json`: full agent trajectories
- `client*.log`

Server logs are at `/zhaoleo/logs/<run_id>_{prefill,decode}.log`.
