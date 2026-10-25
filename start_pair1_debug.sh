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

# Debug bring-up of pair 1: same as start_pair1.sh, plus
#   - xprof profiler enabled (POST /start_profile, /stop_profile on :8400 and :9400),
#     traces written to /zhaoleo/xprof/<RUN_ID>/{prefill,decode}/
#   - VLLM_LOGGING_LEVEL=DEBUG on both servers
#
#   bash start_pair1_debug.sh
set -euo pipefail
DEBUG=1 exec bash "$(dirname "$(readlink -f "$0")")/start_pair1.sh" "$@"
