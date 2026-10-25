#!/usr/bin/env bash
# Run the xprof CLI analyses over a capture made by xprof_capture_pd.sh.
#   PROFILE_ROOT=/zhaoleo/xprof/<run_id> OUT=/zhaoleo/logs/xprof_<run_id>/analysis \
#     HOST_RANK=0 bash xprof_analyze.sh
set -uo pipefail

X="${X:-/zhaoleo/xprof-venv/bin/xprof}"
PROFILE_ROOT="${PROFILE_ROOT:?}"
OUT="${OUT:?}"
HOST_RANK="${HOST_RANK:-0}"
mkdir -p "${OUT}"
cd /tmp

for role_dir in "${PROFILE_ROOT}"/*/; do
  role=$(basename "${role_dir}")
  for sess_dir in "${role_dir}"*/; do
    name=$(basename "${sess_dir}")
    session=$(ls "${sess_dir}plugins/profile" | head -1)
    host=$(ls "${sess_dir}plugins/profile/${session}" | grep "^rank${HOST_RANK}_" | sed 's/\.xplane\.pb$//')
    tag="${role}_${name}_rank${HOST_RANK}"
    common=(--session_id="${session}" --logdir="${sess_dir%/}" --host="${host}")
    echo "== ${tag} (session ${session}, host ${host})"
    run() {
      local what="$1"; shift
      local t0=${SECONDS}
      "${X}" "$@" "${common[@]}" >"${OUT}/${tag}_${what}.json" 2>"${OUT}/${tag}_${what}.err"
      echo "   ${what}: rc=$? $((SECONDS - t0))s"
    }
    run overview get_overview
    run top_hlo_ops get_top_hlo_ops --limit=25
    run host_boundness check_host_boundness
    run time_attribution get_device_time_attribution --include_summary=True
    run kv_annotations aggregate_xplane_events --event_regex='KV_Cache_.*' --group_by=name
  done
done
echo "ANALYSIS_DONE ${OUT}"
