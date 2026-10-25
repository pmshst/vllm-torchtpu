# Qwen3.5-397B-A17B-NVFP4 PD Disaggregation: Performance Report

- **Run ID:** `q397nvfp4_1025_0553`
- **Date:** 2026-10-25
- **Recipe:** `qwen35_397b_nvfp4_pd_l20g_recipe.md`
- **Launcher:** `serve_qwen35_397b_nvfp4_pd.sh`
- **Bench script:** `bench_qwen35_397b_nvfp4_pd.sh`
- **Raw results:** `/zhaoleo/logs/bench_q397nvfp4_1025_0553/*.json`

## Setup

| | Prefill | Decode |
|---|---|---|
| Host | drzl (`10.128.0.182`), tpu7x-8 standalone | gw7c (`10.128.0.147`), tpu7x-8 standalone |
| Parallelism | TP1 x PCP8, EP8 | TP2 x DP4, EP8 |
| MoE | NVFP4 -> adaptive fused-EP W4A8, block 256 | NVFP4 GMM W4A16 (block 16) |
| Online FP8 | attn / GDN / shared-expert projections (285 layers) | same |
| Spec decode | MTP, 1 token | MTP, 3 tokens |
| KV | fp8, block 768, 9.43M tokens | fp8, block 2304, 3.72M tokens per DP rank |
| Connector | `TPUMultiConnector` (Raiden producer + 64 GiB host offload) | `TPURaidenConnector` consumer |

- **Raiden admission:** `pcp8_prefill` -> `dp4tp2_decode`. Each request's KV
  is resharded from 8 source ranks to 2 destination units.
- **Proxy:** `examples/disagg/toy_proxy_server.py` on drzl.
- **Server limits:** `--max-model-len 65536`; prefix caching on.

**Workload.** `vllm bench serve`, random dataset, ISL 8192 / OSL 1024, range
ratio 0.8 (so ISL varies over 6554–9830), `--ignore-eos`, request rate inf.
Each level sends `4 x concurrency` prompts (minimum 8), after `concurrency`
warmup requests. This is the same shape as
`scripts/vllm/benchmark/inferencex/qwen3.5/bench.sh`.

## Results

| Concurrency | Requests | Duration (s) | Req/s | Output tok/s | Total tok/s | TTFT mean / p50 / p99 (ms) | TPOT mean / p50 / p99 (ms) | ITL mean / p99 (ms) | E2E mean (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 99.0 | 0.08 | 105 | 597 | 328 / 333 / 471 | 9.3 / 8.5 / 11.2 | 31.7 / 37.9 | 12.4 |
| 16 | 64 | 50.0 | 1.28 | 1,336 | 11,991 | 843 / 547 / 3,226 | 8.5 / 7.7 / 12.0 | 28.6 / 36.6 | 9.5 |
| 64 | 256 | 72.5 | 3.53 | 3,734 | 33,628 | 2,164 / 1,093 / 10,062 | 12.7 / 11.9 / 18.9 | 41.6 / 52.7 | 15.3 |
| 128 | 512 | 97.4 | 5.26 | 5,459 | 49,971 | 5,395 / 3,548 / 18,972 | 15.8 / 14.8 / 22.4 | 52.0 / 65.3 | 21.5 |

Per-host normalization (2 hosts, 16 devices): at concurrency 128 that is
**~2,730 output tok/s per host** and **~341 output tok/s per device**.

### MTP acceptance (decode, cumulative over the whole run)

Read from the decode server's Prometheus `vllm:spec_decode_*` counters,
summed over the 4 DP engines:

| Metric | Value |
|---|---|
| Draft tokens / accepted | 863,886 / 675,067 |
| Acceptance rate | **78.1%** |
| Mean acceptance length (tokens per verify step, including the bonus token) | **3.34** of a possible 4 |
| Per-position acceptance (engine 0) | pos1 89.6%, pos2 79.7%, pos3 69.0% |

This is consistent with ITL ≈ 3.4 x TPOT in the table.

## Observations

1. **Decode is fast and scales well.** TPOT is 7.7 ms (p50) at concurrency 16
   and 14.8 ms at 128, i.e. ~68 tok/s per user even at 128 concurrent
   streams. MTP-3 contributes ~3.3x tokens per step.
2. **Prefill is the bottleneck at high concurrency.** TTFT p50 rises from
   0.55 s (c16) to 3.5 s (c128), and p99 reaches 19 s. At c128 the single
   PCP8 prefill engine processes ~44.5k input tokens/s
   (4.33M tokens / 97 s). That is close to its ceiling for 8K prompts, while
   decode TPOT is still low. For this ISL/OSL ratio (8:1), a **2P1D** split
   (or 2P2D on the 4-host slice) should raise throughput and cut TTFT.
3. **Single-request latency:** at c1, TTFT is 328 ms for an ~6K–10K-token
   prompt (PCP8 prefill + Raiden transfer + first decode step), and TPOT is
   8.5 ms (~118 tok/s).
4. **Raiden transfer overhead is small.** The smoke test showed reshard plan
   + controller time of 6–7 ms per request, and proxy prefill-to-first-chunk
   time is dominated by compute.

## Caveats

- **Synthetic prompts.** Random token IDs with `--ignore-eos` tend to produce
  repetitive continuations, which MTP predicts unusually well. The 78%
  acceptance (3.34 tokens/step) is therefore **optimistic** compared with real
  traffic. The DeepSWE replay test (pair 2: svps/wx0l) measures acceptance on
  realistic agentic prompts.
- **Short runs.** Each level is short (4 x concurrency requests; 50–100 s),
  so the p99 numbers rest on few samples.
- **Shortened context.** `--max-model-len 65536` was used for this run,
  instead of the config's 1M. A 1M setting reduces KV capacity per request but
  should not change these 8K-context numbers much.
- **No prefix reuse.** Prefix caching was on but random prompts give no
  reuse, so the host offload connector stayed idle.
- **No accuracy check.** Accuracy was not evaluated beyond the smoke tests
  ("Paris.", 17x23 = 391).

## Startup cost (cold, first run)

| | Prefill | Decode |
|---|---|---|
| Weight load (NFS, cold page cache) | ~15 min | ~17 min |
| Engine init (profile + KV + warmup) | 577 s (compile 364 s) | 1,887 s (compile 1,622 s) |
| Ready at | 06:20 | 06:43 |

Decode's long init is mostly spec-decode warmup over R = 2..192
(`max_num_seqs=192`, MTP-3) plus compiling 7 sizes. The compile caches are
now on NFS (`/zhaoleo/torch_compile_cache`), so restarts are faster.
