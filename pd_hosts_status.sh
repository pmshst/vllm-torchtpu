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

# One-shot status of the 4-host PD setup and the DeepSWE MTP tests.
R=$(cat /zhaoleo/logs/current_pd_run_id)
R2=$(cat /zhaoleo/logs/current_pd_run_id_pair2)
P=instance-group-1-zhaoleo-tpuv7x-
echo "time: $(date -u +%H:%M:%S) UTC  pair1=${R}  pair2=${R2}"
for h in drzl gw7c svps wx0l; do
  pid=$(timeout 15 ssh -n -o BatchMode=yes "${P}${h}" 'pgrep -f "[b]in/vllm serve nvidia" | head -1')
  echo "host ${h}: vllm_pid=${pid:-DOWN}"
done
for run in "${R}" "${R2}"; do
  for role in prefill decode; do
    log=/zhaoleo/logs/${run}_${role}.log
    ready=$(grep -c "Application startup complete" "${log}" 2>/dev/null)
    errs=$(grep -cE "Traceback|EngineCore failed" "${log}" 2>/dev/null)
    last=$(tr '\r' '\n' <"${log}" | grep -vE 'shardy|UserWarning|get_attr|online_fp8.py:213|not reachable|<|^\s*$|device_assignment|Prefetching|tpu_connector.py:6727|POST /v1' | tail -1 | cut -c1-140)
    echo "${run} ${role}: ready=${ready} errors=${errs} last: ${last}"
  done
done
for run in "${R}" "${R2}"; do
  O=/zhaoleo/logs/deepswe_mtp_live_${run}
  echo "== DeepSWE ${run}"
  if [[ ! -f ${O}/mtp_acceptance.csv ]]; then echo "  not started"; continue; fi
  python3 - "${O}" <<'EOF'
import csv, sys, time, glob
o = sys.argv[1]
rows = list(csv.DictReader(open(o + "/mtp_acceptance.csv")))
al = [r for r in rows if r["engine"] == "all" and r["accept_rate"]]
if not al:
    print("  no windows yet"); sys.exit()
rates = [float(r["accept_rate"]) for r in al]
t = lambda r: time.strftime("%H:%M:%S", time.gmtime(float(r["ts"])))
print(f"  windows={len(al)} {t(al[0])}->{t(al[-1])} accept min/mean/max="
      f"{min(rates):.1%}/{sum(rates)/len(rates):.1%}/{max(rates):.1%} "
      f"zero_windows={sum(r['zero_flag']=='1' for r in rows)}")
print("  last4=" + ", ".join(f"{float(r['accept_rate']):.1%}" for r in al[-4:]))
eng = {}
for r in rows:
    if r["engine"] != "all" and r["accept_rate"]:
        eng.setdefault(r["engine"], []).append(float(r["accept_rate"]))
print("  per-engine min: " + ", ".join(f"e{k}={min(v):.1%}" for k, v in sorted(eng.items())))
counts = {}
for f in glob.glob(o + "/client*/tasks.csv"):
    for r in csv.DictReader(open(f)):
        counts[r["exit_status"]] = counts.get(r["exit_status"], 0) + 1
print("  tasks: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none finished"))
EOF
  grep -h "round .* finished\|done after" "${O}"/client*.log 2>/dev/null | tail -3 | sed 's/^/  /'
  grep -q "DONE OUT_DIR" "${O}"/../*.none 2>/dev/null
done
echo "containers on drzl: $(docker ps -q | wc -l)"
