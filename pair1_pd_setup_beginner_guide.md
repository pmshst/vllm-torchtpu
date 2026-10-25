# Beginner's Guide: Serving Qwen3.5-397B-A17B-NVFP4 with PD Disaggregation on Two TPU7x Hosts ("Pair 1")

This guide walks you through, from scratch, how to bring up the setup we call
**pair 1**:

- **drzl** (`10.128.0.182`) runs the **prefill** server (and the proxy).
- **gw7c** (`10.128.0.147`) runs the **decode** server.

It also explains how to check that it works, how to read the logs, and how to
debug the problems we actually hit. Every command here was used on this
cluster. Replace host names and IPs if yours differ.

---

## 0. Concepts in five minutes

**What is PD disaggregation?** An LLM request has two phases:

1. **Prefill.** The model reads the whole prompt at once. This is
   compute-heavy and produces the **KV cache** (the model's memory of the
   prompt).
2. **Decode.** The model generates output tokens one step at a time, reading
   that KV cache. This is memory-bandwidth-heavy.

PD disaggregation runs the two phases on **different machines**, each tuned
for its job. After prefill, the KV cache is **transferred** to the decode
machine. In this repo that transfer is done by **Raiden**
(`TPURaidenConnector`). The KV is "resharded" because the two machines split
the model differently.

```text
   client ──► proxy :8000 ──► PREFILL :8400 (drzl, 8 TPU devices)
                    │              │  KV cache (Raiden, TCP)
                    │              ▼
                    └────────► DECODE :9400 (gw7c, 8 TPU devices) ──► tokens back to client
```

**Pair 1 layout:**

| | Prefill (drzl) | Decode (gw7c) |
|---|---|---|
| Parallelism | `PCP=8`: one prompt is split across 8 devices (fast long prompts) | `DP=4 x TP=2`: 4 independent engines, each using 2 devices |
| Experts (MoE) | spread over all 8 devices (EP) | same |
| Speculative decoding | MTP, 1 draft token | **MTP, 3 draft tokens** |

**What is MTP?** The model has an extra "Multi-Token Prediction" head that
**guesses** the next few tokens cheaply. The main model then **verifies** the
guesses in one step. The **acceptance rate** is the fraction of guesses that
were correct. Ours is ~85%, which means ~3.5 tokens per decode step instead
of 1.

**Hardware facts:**

- Each host is a `tpu7x` VM with 8 TPU devices (~96 GB HBM each), 224 CPUs
  and 944 GB RAM.
- The 4 VMs form one `tpu7x-32` slice. We run **each host standalone**
  (a tpu7x-8) and let Raiden connect them over the network.

---

## 1. Prerequisites (one time)

### 1.1 Accounts and access

- You can log in to drzl as your user (`zhaoleo_google_com` here).
- The project uses **OS Login**, so normal `authorized_keys` metadata does
  not apply. To reach the other hosts the first time:

  ```bash
  gcloud compute ssh instance-group-1-zhaoleo-tpuv7x-gw7c --zone us-central1-c --internal-ip
  ```

### 1.2 Passwordless SSH between hosts

This lets scripts start servers on other hosts. On **each** host, as your
user:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub        # copy this line
```

Append **every** host's public key to `~/.ssh/authorized_keys` on **every**
host (`chmod 600`). Then add this to `~/.ssh/config`, so the first connection
doesn't stop at a yes/no host-key question:

```text
Host instance-group-1-zhaoleo-tpuv7x-* 10.128.0.*
  StrictHostKeyChecking accept-new
```

Test it. This must print the host name without asking for anything:

```bash
ssh -o BatchMode=yes instance-group-1-zhaoleo-tpuv7x-gw7c hostname
```

If a host has no home directory yet (the user never logged in), create one
first:

```bash
sudo mkdir -p /home/$U && sudo cp -rT /etc/skel /home/$U && sudo chown -R $U:$U /home/$U
```

### 1.3 Shared storage (NFS)

The model, the code and the compile caches live on NFS at `/zhaoleo`, so all
hosts see the same files. Check it on each host:

```bash
mountpoint /zhaoleo && df -h /zhaoleo
```

If it is not mounted (this is what `/zhaoleo/mount.sh` does):

```bash
sudo apt-get update && sudo apt-get install -y nfs-common
sudo mkdir -p /zhaoleo
sudo mount -t nfs -o rw,hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,async \
  10.128.0.45:/zhaoleo /zhaoleo
```

> This mount does **not** survive a reboot (there is no `/etc/fstab` entry).
> Re-run it after a reboot.

### 1.4 Code and Python environment

The repo and its virtualenv are already on NFS:

```bash
ls /zhaoleo/vllm-torchtpu/.venv/bin/python     # must exist
cd /zhaoleo/vllm-torchtpu && git log -1 --oneline
```

Versions used for pair 1 (record yours when reporting issues):

| Package | Version |
|---|---|
| vllm-torchtpu | commit `5b4512bdd` |
| vllm | 0.32.0 |
| torch | 2.13.0 |
| torch-tpu | 0.2.0 |
| libtpu | 0.0.50 |
| jax | 0.11.2 |
| tpu-sync-torch | 0.0.1.dev20261012084241 |
| Python | 3.12.3 |

Print your versions:

```bash
/zhaoleo/vllm-torchtpu/.venv/bin/python -I -m pip list 2>/dev/null \
  | grep -iE "^(vllm|torch|libtpu|jax|tpu-sync)"
```

### 1.5 Download the model (~251 GB, once)

```bash
export HF_HOME=/zhaoleo
/zhaoleo/vllm-torchtpu/.venv/bin/hf download nvidia/Qwen3.5-397B-A17B-NVFP4 \
  > /zhaoleo/logs/nvfp4_download.log 2>&1 &
```

- It takes a while, and **NFS becomes very slow for everyone during the
  download** (even `du` can time out).
- Check completeness afterwards. The snapshot should contain 11
  `model-0000x-of-00011.safetensors` files and no `*.incomplete` blobs that
  are still growing:

  ```bash
  ls -laL /zhaoleo/hub/models--nvidia--Qwen3.5-397B-A17B-NVFP4/snapshots/*/ | head -30
  ```

---

## 2. The files you will use

All of these are in `/zhaoleo/vllm-torchtpu/`:

| File | What it is |
|---|---|
| `serve_qwen35_397b_nvfp4_pd.sh` | **The launcher.** Run with `ROLE=prefill`, `decode` or `proxy`. It sets every env var and engine flag. |
| `qwen35_397b_nvfp4_pd_l20g_recipe.md` | Explains every setting and how it maps to the production `l20g` config |
| `bench_qwen35_397b_nvfp4_pd.sh` | Throughput/latency benchmark (random prompts) |
| `run_deepswe_mtp_test.sh`, `deepswe_live_agent.py`, `deepswe_mtp_replay.py` | DeepSWE agent load test + MTP acceptance monitor |
| `pd_hosts_status.sh` | One-shot status of all hosts and tests |

Open the launcher once and skim it. Each block is commented. The most
important variables are:

| Variable | Default | Meaning |
|---|---|---|
| `ROLE` | (required) | `prefill`, `decode` or `proxy` |
| `PREFILL_IP` / `DECODE_IP` | (required) | This host's **routable** IP (`hostname -I`, first entry) |
| `RUN_ID` | `q397nvfp4` | A name for this run; use the **same** value on both hosts |
| `MAX_MODEL_LEN` | `1048576` | Max context. Pair 1 used `65536` for faster bring-up; production is 1M. |
| `PREFILL_OFFLOAD` | `1` | Prefill also keeps KV in 64 GiB of host RAM |

---

## 3. Start pair 1 step by step

### Step 1: Pick a run ID and make sure the TPUs and ports are free

```bash
export RUN_ID=q397nvfp4_$(date +%m%d_%H%M)
echo $RUN_ID > /zhaoleo/logs/current_pd_run_id

# Nothing may already be serving on either host.
pgrep -af "[b]in/vllm serve"
ssh -n instance-group-1-zhaoleo-tpuv7x-gw7c 'pgrep -af "[b]in/vllm serve"'

# Ports must be free (8400/9400/8000 API, 14579 KV, 27000+, 29200+, 19600+, 47800+, 8431-8438, 8471).
ss -ltn | grep -E ":(8400|9400|8000|14579|2700[0-9]|2710[0-9]|29200|19600|47800|8471) "
```

> **Tip: the `[b]` in `pgrep -f "[b]in/vllm serve"` matters.** Without it,
> `pgrep` (especially through `ssh ... bash -c`) can match **its own command
> line** and report a process that doesn't exist.

### Step 2: Set the cache locations

Compiling the model takes ~6 min (prefill) to ~27 min (decode) the first
time. Pointing both hosts at the **shared** caches lets later starts reuse
that work:

```bash
export VLLM_CACHE_ROOT=/zhaoleo
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/zhaoleo/persistent_cache
export VLLM_ENGINE_READY_TIMEOUT_S=7200
```

> **Gotcha:** commands run with `ssh host 'cmd'` do **not** load your
> `~/.bashrc`. Pass these exports explicitly in remote commands (Step 4
> does). Otherwise the remote server silently uses `~/.cache/vllm` and
> recompiles from scratch.

### Step 3: Start the prefill server on drzl

```bash
cd /zhaoleo/vllm-torchtpu
ROLE=prefill PREFILL_IP=10.128.0.182 RUN_ID=$RUN_ID MAX_MODEL_LEN=65536 \
  setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh \
  > /zhaoleo/logs/${RUN_ID}_prefill.log 2>&1 < /dev/null &
```

- `setsid nohup ... &` keeps the server running after you log out.
- All output goes to `/zhaoleo/logs/${RUN_ID}_prefill.log`.

### Step 4: Start the decode server on gw7c (from drzl)

```bash
CE='export VLLM_CACHE_ROOT=/zhaoleo TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/zhaoleo/persistent_cache VLLM_ENGINE_READY_TIMEOUT_S=7200; cd /zhaoleo/vllm-torchtpu'
timeout 20 ssh -n -o BatchMode=yes instance-group-1-zhaoleo-tpuv7x-gw7c \
  "$CE; ROLE=decode DECODE_IP=10.128.0.147 RUN_ID=$RUN_ID MAX_MODEL_LEN=65536 \
   setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh > /zhaoleo/logs/${RUN_ID}_decode.log 2>&1 < /dev/null & disown; echo started"
```

> **Gotcha:** use `ssh -n` (no stdin), `< /dev/null`, `& disown` and an outer
> `timeout`. Without them, `ssh` can **hang forever** waiting on the
> background process. Any commands queued after it then never run.

Both hosts read the log on NFS, so you can watch decode from drzl too.

### Step 5: Wait for startup (cold start ~45–50 min, warm start ~15–25 min)

Watch both logs in two terminals:

```bash
tail -f /zhaoleo/logs/${RUN_ID}_prefill.log | grep -vE "shardy_xla_pass|UserWarning|get_attr_node"
tail -f /zhaoleo/logs/${RUN_ID}_decode.log  | grep -vE "shardy_xla_pass|UserWarning|get_attr_node"
```

Section 5 explains each phase. The server is ready when you see:

```text
INFO:     Application startup complete.
```

Prefill prints it once. Decode prints it once per API server, 4 times.

### Step 6: Start the proxy on drzl

The proxy is what clients talk to. It sends each request to prefill, waits
for the KV transfer, then streams the decode output back.

```bash
ROLE=proxy PREFILL_IP=10.128.0.182 DECODE_IP=10.128.0.147 \
  setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh \
  > /zhaoleo/logs/${RUN_ID}_proxy.log 2>&1 < /dev/null &
sleep 15; ss -ltn | grep ":8000 " && echo proxy-listening
```

### Step 7: Smoke test

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d \
 '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"The capital of France is","max_tokens":32,"temperature":0}'

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","messages":[{"role":"user","content":"What is 17*23? Answer with just the number."}],
   "max_tokens":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Expected answers: `" Paris."` and `"391"`.

Health endpoints:

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/healthcheck   # proxy
curl -s -o /dev/null -w "%{http_code}\n" 10.128.0.182:8400/health      # prefill
curl -s -o /dev/null -w "%{http_code}\n" 10.128.0.147:9400/health      # decode
```

---

## 4. Verify it is really doing PD (not just "it answered")

Run these after the smoke test.

**1. Both sides were admitted with the expected topology:**

```bash
grep -h "Raiden pool admission complete" /zhaoleo/logs/${RUN_ID}_prefill.log | head -1
grep -h "Raiden pool admission complete" /zhaoleo/logs/${RUN_ID}_decode.log  | head -1
```

Expected:

```text
... topology=pcp8_prefill role=kv_producer ... pools=106 ... fa=16 gdn.conv=45 gdn.ssm=45
... topology=dp4tp2_decode role=kv_consumer ...
```

**2. KV was actually transferred for your request.** Look for
`raiden_pool_reshard_senders_dispatched` (prefill) and `Raiden Stage 3
reshard` (decode):

```bash
grep -c "raiden_pool_reshard_senders_dispatched" /zhaoleo/logs/${RUN_ID}_prefill.log
grep -c "Raiden Stage 3 reshard" /zhaoleo/logs/${RUN_ID}_decode.log
```

**3. The proxy timing for each request:**

```bash
grep "PERF PROXY" /zhaoleo/logs/${RUN_ID}_proxy.log | tail -3
```

Example:

```text
PERF PROXY prefill_recv req_id=... dur_ms=112.42     <- prefill took 112 ms
PERF PROXY first_chunk ... ttft_ms=540.40            <- time to first token for the client
```

---

## 5. How to read the logs

### 5.1 Startup timeline: what you should see, in order

| # | Phase | Log line to look for | Typical time |
|---|---|---|---|
| 1 | Config parsed | `non-default args: {...}` (check your flags here) | seconds |
| 2 | Workers start | `rank N in world size 8 is assigned as DP rank ..., TP rank ...` | ~1 min |
| 3 | Weight loading | `Loading safetensors checkpoint shards: 18% Completed \| 2/11` | **15–17 min cold**, ~3 min warm (files in RAM cache) |
| 4 | NVFP4 experts prepared | `NVFP4 MoE weights prepared (FP4 (jax requant block-256))` + `NVFP4 fused EP W4A8 enabled: block-256` (prefill), `NVFP4 GMM W4A16 enabled` (decode) | end of loading |
| 5 | Online FP8 | `TPU online FP8 validated 285 runtime Linear layers` | end of loading |
| 6 | KV cache sized | `TPU KV cache size: 9,427,353 tokens, Maximum concurrency ...` | ~1 min |
| 7 | Raiden admission | `Raiden pool admission complete topology=...` | right after |
| 8 | Compilation | `Compiling a graph for compile range (512, 512) takes 160.18 s` | prefill ~6 min, decode up to ~27 min cold; minutes with a warm cache |
| 9 | Spec-decode warmup (decode only) | `spec-decode warmup batch: R=130 P=32 ...` with R counting 2 -> 192, then a `P=768` pass | ~15 min |
| 10 | Engine done | `init engine (profile, create kv cache, warmup model) took 577.18 s (compilation: 363.54 s)` | end |
| 11 | Connectors up | `TPUConnector --> using Raiden backend`, `KVCacheStore up` (prefill offload) | end |
| 12 | **Ready** | `Application startup complete.` | end |

To see progress during weight loading (it uses carriage returns):

```bash
tr '\r' '\n' < /zhaoleo/logs/${RUN_ID}_decode.log | grep "Loading safetensors" | tail -1
```

### 5.2 Messages that look scary but are harmless

| Message | Why it's fine |
|---|---|
| `shardy_xla_pass ... stale model export, try re-exporting` | XLA warning during compile; no effect |
| `No available shared memory broadcast block found in 60 seconds` | The engine is waiting while workers compile; normal during startup |
| `TPURaidenOffloadingConnector: ... offload controller at 127.0.0.1:47800 not reachable yet` | Workers start before the scheduler creates the controller. It clears by itself once startup reaches the scheduler. |
| HTML `Error 404 (Not Found)` from `metadata.google.internal` | Something asks GCE metadata for keys that don't exist |
| `Received stats for out-of-order step ... from engine N` (`VLLM_DP_Coordinator`) | DP load-stats ordering; appears thousands of times on healthy runs |
| `UserWarning: Attempted to insert a get_attr Node ...` | torch_tpu FX-pass warning |
| `Add 3 padding layers, may waste at most 6.67% KV cache memory` | Informational |

### 5.3 Messages that are real problems

Search for them:

```bash
grep -nE "Traceback|EngineCore failed|RuntimeError|ValueError: |OOM|out of memory" /zhaoleo/logs/${RUN_ID}_*.log | head
```

Everything after the first `Traceback` explains the failure. Read the
**last** lines of the traceback first.

---

## 6. Monitoring a running server

### 6.1 Quick status of everything

```bash
bash /zhaoleo/vllm-torchtpu/pd_hosts_status.sh
```

It shows server liveness per host, readiness and error counts per log, and
DeepSWE MTP statistics if a test is running.

### 6.2 Prometheus metrics (most useful)

Every server exposes `/metrics`. The decode server is the interesting one for
MTP:

```bash
curl -s 10.128.0.147:9400/metrics | grep -E "^vllm:spec_decode" | grep -v _created
```

| Metric | Meaning |
|---|---|
| `vllm:spec_decode_num_drafts_total{engine=N}` | Number of MTP guess rounds on DP engine N |
| `vllm:spec_decode_num_draft_tokens_total` | Guessed tokens (= drafts x 3) |
| `vllm:spec_decode_num_accepted_tokens_total` | Guesses that were correct |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total{position=k}` | Correct guesses at position k (1st, 2nd, 3rd) |

- **Acceptance rate** = accepted / draft tokens. Ours is ~85%.
- **Mean acceptance length** = 1 + accepted / drafts. Ours is ~3.57 tokens
  per step.

These counters are **cumulative since the server started**. To get "the rate
right now", take the difference between two readings a few seconds apart.
`deepswe_mtp_replay.py monitor` does exactly that every 15 s:

```bash
/zhaoleo/vllm-torchtpu/.venv/bin/python -I /zhaoleo/vllm-torchtpu/deepswe_mtp_replay.py monitor \
  --metrics-url http://10.128.0.147:9400/metrics --interval-s 15 --duration-s 600 --out-dir /tmp/mtp_watch
```

> **Important:** a window with **0 drafts** (an idle engine) is **not** a
> 0% acceptance rate. It means "no data". Always check the draft count
> before concluding acceptance dropped.

Other useful decode metrics:

| Metric | Healthy value / meaning |
|---|---|
| `vllm:num_requests_running` / `num_requests_waiting` | Load per engine; a growing `waiting` means overloaded |
| `vllm:kv_cache_usage_perc` | Fraction of KV cache in use |
| `vllm:num_preemptions_total` | Requests evicted for lack of KV space (0 in our runs) |
| `vllm:prefix_cache_hits_total / prefix_cache_queries_total` | Prefix reuse (~82% for multi-turn agents) |

### 6.3 Is a process alive?

```bash
pgrep -af "[b]in/vllm serve"                                    # local
ssh -n instance-group-1-zhaoleo-tpuv7x-gw7c 'pgrep -af "[b]in/vllm serve"'
```

---

## 7. Running the tests

### 7.1 Performance benchmark (random prompts, ISL 8192 / OSL 1024)

```bash
ISL=8192 OSL=1024 CONCS="1 16 64 128" bash bench_qwen35_397b_nvfp4_pd.sh
```

Results go to `/zhaoleo/logs/bench_<RUN_ID>/isl*_conc*.json`. Our numbers are
in `qwen35_397b_nvfp4_pd_perf_report.md`: TPOT p50 7.7 ms at concurrency 16,
5.5k output tok/s at 128.

### 7.2 DeepSWE agent load + MTP monitor

The data is the public `github.com/datacurve-ai/deep-swe` repo (113 tasks
plus prebuilt Docker images). One-time setup on drzl:

```bash
mkdir -p /zhaoleo/deepswe/src && cd /zhaoleo/deepswe/src
git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git
~/.local/bin/uv venv --python 3.12 /zhaoleo/deepswe/agent-venv
VIRTUAL_ENV=/zhaoleo/deepswe/agent-venv ~/.local/bin/uv pip install mini-swe-agent
# Optional: pre-pull the 113 task images (~0.84 GB compressed each)
grep -h '^docker_image' deep-swe/tasks/*/task.toml | sed 's/.*"\(.*\)"/\1/' | sort -u | xargs -P 8 -n 1 docker pull -q
```

Run 1 hour, 3 clients x 16 concurrent agents:

```bash
cd /zhaoleo/vllm-torchtpu
MODE=live PREFILL_IP=10.128.0.182 DECODE_IP=10.128.0.147 PROXY_PORT=8000 DURATION_S=3600 \
  OUT_DIR=/zhaoleo/logs/deepswe_mtp_live_$RUN_ID bash run_deepswe_mtp_test.sh
```

| Output | Content |
|---|---|
| `monitor.log` | One line per 15 s: overall acceptance, plus `ZERO-ACCEPTANCE` if any engine falls to <= 1% |
| `mtp_acceptance.csv` | The same per engine, with per-position rates |
| `client*/tasks.csv` | One line per finished task (`Submitted`, `RepeatedFormatError`, `InternalServerError`, `TimeExceeded`) |
| `client*/traj/*.json` | Full agent conversations |

> **Never import `torch` / `transformers` in a client script on drzl or
> gw7c.** Importing them initializes `torch_tpu` and touches the TPU that the
> server owns. The test clients deliberately use only `aiohttp` and
> mini-swe-agent (separate venv).

---

## 8. Debugging guide (problems we actually hit)

### 8.1 Startup problems

| Symptom | Cause | Fix |
|---|---|---|
| Server hangs at startup, waiting for other hosts | The VM's `tpu-env` describes the whole 4-host slice (`HOST_BOUNDS 1,1,4`) | The launcher sets `TPU_HOST_BOUNDS=1,1,1`, `TPU_TOPOLOGY=2x2x1`, `TPU_WORKER_HOSTNAMES=localhost`. Don't remove them. |
| `Raiden Qwen3.5 admission topology ... requires kv_producer with tensor_parallel_size=1` | Prefill launched with TP > 1 | Prefill must be TP=1 with PCP 4/8 (or DP 4/8/16, PP 2/4/8) |
| `Unknown connector TpuRaidenConnector` | Wrong class name or module | `TPURaidenConnector` in module `vllm_torchtpu.distributed.kv_transfer.tpu_connector` |
| `page_tokens % interleave` error | Interleave doesn't divide the page size | Use `--cp-kv-cache-interleave-size 128` (pair 1 uses 256 with block 768) |
| `Port N is busy` / bind error | An old server is still running | Stop it (section 9), check `ss -ltn` |
| Weight loading takes > 30 min | NFS is busy (a download, or several hosts loading at once) | Wait. A second start is much faster (RAM page cache). Don't start a download while servers load. |
| Decode recompiles everything (~27 min) every time | `VLLM_CACHE_ROOT` not set in the remote shell | Pass the exports in the ssh command (Step 4) |
| HBM OOM at load | Context or batch too large for memory | Lower `MAX_MODEL_LEN` or `--max-num-seqs`, or `--gpu-memory-utilization` |

### 8.2 Request problems

| Symptom | Cause | Fix |
|---|---|---|
| Proxy returns **500** for long conversations; prefill log shows `400 Bad Request` | The prompt is longer than `max_model_len`. Read the real reason by sending the same request **directly to prefill :8400**: `This model's maximum context length is 65536 tokens...` | Start with a larger `MAX_MODEL_LEN` (production: 1048576) |
| First request hangs, no transfer | Wrong `TPU_RAIDEN_ADVERTISE_HOST` / IP, or a firewall | Use each host's routable IP; ports 27000-27199, 29200+, 19600+, 14579 must be reachable between hosts |
| Proxy shows `[DECODE-ERROR] ... status=...` | Decode rejected the request | The body after `body=` is decode's error message |

How to see the real error behind a proxy 500:

```bash
grep -E "DECODE-ERROR|HTTPStatusError|Bad Request" /zhaoleo/logs/${RUN_ID}_proxy.log | tail
curl -s 10.128.0.182:8400/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
```

### 8.3 MTP acceptance looks wrong

1. **Check the draft count first.** If drafts = 0 in that window, the engine
   was idle; that is not a failure.
2. **Check per engine.** One DP engine can misbehave while the overall rate
   looks fine:

   ```bash
   awk -F, '$9==1' /zhaoleo/logs/deepswe_mtp_live_$RUN_ID/mtp_acceptance.csv
   ```

3. **Check per position.** The 3rd guess is always the worst (~78% for us).
   If **position 1** collapses, the draft head or its KV is broken.
4. **Look for pressure:** `num_preemptions_total > 0` or a growing
   `num_requests_waiting` around the time of the drop.
5. **Record versions and commit** (section 1.4). Several MTP/PD fixes landed
   between 2026-09-29 and 2026-10-15 (e.g. `84e3425af` "MTP: keep GDN state
   across scheduling gaps", `f37321669` "pad EP-DP lockstep draft steps"). An
   older build may behave differently.

### 8.4 Process and shell pitfalls

| Pitfall | What happens | Do instead |
|---|---|---|
| `pgrep -f "vllm serve"` through ssh | Matches its own command line, so a dead server looks alive | `pgrep -f "[b]in/vllm serve"` |
| `ssh host 'nohup ... &'` | ssh never returns | `ssh -n ... '... < /dev/null & disown'`, wrapped in `timeout 20` |
| Killing a test client with SIGKILL | Its Docker containers keep running (`sleep 4h`) | Stop clients with SIGTERM, then remove leftover `minisweagent-*` containers that show no activity |
| Reading a big growing dir on NFS during a download | `du` / `grep -r` time out | Copy what you need to local disk first, or wait |

---

## 9. Stop everything

```bash
# Proxy and prefill on drzl
pkill -TERM -f "[t]oy_proxy_server"
pkill -TERM -f "[b]in/vllm serve nvidia"

# Decode on gw7c
ssh -n instance-group-1-zhaoleo-tpuv7x-gw7c 'pkill -TERM -f "[b]in/vllm serve nvidia"'

# Wait ~45 s, then confirm nothing is left (TPU released)
pgrep -af "[b]in/vllm serve|EngineCore|Worker_"
ssh -n instance-group-1-zhaoleo-tpuv7x-gw7c 'pgrep -af "[b]in/vllm serve|EngineCore|Worker_"'

# Leftover agent containers from tests
docker ps --filter name=minisweagent -q | xargs -r docker rm -f
```

---

## 10. Cheat sheet

```bash
# start
RUN_ID=q397nvfp4_$(date +%m%d_%H%M); echo $RUN_ID > /zhaoleo/logs/current_pd_run_id
ROLE=prefill PREFILL_IP=10.128.0.182 RUN_ID=$RUN_ID MAX_MODEL_LEN=65536 setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh > /zhaoleo/logs/${RUN_ID}_prefill.log 2>&1 < /dev/null &
timeout 20 ssh -n instance-group-1-zhaoleo-tpuv7x-gw7c "export VLLM_CACHE_ROOT=/zhaoleo TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/zhaoleo/persistent_cache; cd /zhaoleo/vllm-torchtpu; ROLE=decode DECODE_IP=10.128.0.147 RUN_ID=$RUN_ID MAX_MODEL_LEN=65536 setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh > /zhaoleo/logs/${RUN_ID}_decode.log 2>&1 < /dev/null & disown"
# ready?
grep -c "Application startup complete" /zhaoleo/logs/${RUN_ID}_{prefill,decode}.log
# proxy + test
ROLE=proxy PREFILL_IP=10.128.0.182 DECODE_IP=10.128.0.147 setsid nohup bash serve_qwen35_397b_nvfp4_pd.sh > /zhaoleo/logs/${RUN_ID}_proxy.log 2>&1 < /dev/null &
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"The capital of France is","max_tokens":8}'
# status / errors / MTP
bash pd_hosts_status.sh
grep -nE "Traceback|EngineCore failed" /zhaoleo/logs/${RUN_ID}_*.log
curl -s 10.128.0.147:9400/metrics | grep -E "^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total"
```
