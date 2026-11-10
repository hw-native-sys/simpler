#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (($# != 6 && $# != 7)) || [[ $# == 7 && $7 != --materialize ]]; then
    echo 'usage: production/run.sh BUNDLE RUNTIME_LIB OUTPUT LEVEL WARMUP ROUNDS' >&2
    exit 2
fi
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
bundle=$(realpath "$1")
runtime=$(realpath "$2")
ranks=$("${DSV41_PYTHON:-python3}" "$root/production/topology.py" "$bundle" "${7:-}")
mkdir -p "$3"
output=$(realpath "$3")
[[ ! -e "$output/rank0" && ! -e "$output/rank1" ]] || { echo 'Refusing to overwrite a run' >&2; exit 2; }
if [[ -z "${TASK_DEVICE:-}" ]]; then
    if [[ -n "${DSV41_DEVICES:-}" && "$DSV41_DEVICES" != auto ]]; then
        IFS=',' read -r -a requested <<< "$DSV41_DEVICES"
        [[ ${#requested[@]} == "$ranks" ]] || { echo "Requested devices do not match case topology" >&2; exit 2; }
    fi
    bash "$root/../../../.claude/skills/onboard-arch-precheck/check.sh" a5
    printf -v command 'TASK_DEVICE="$TASK_DEVICE" bash %q' "$root/production/run.sh"
    for argument in "$@"; do printf -v command '%s %q' "$command" "$argument"; done
    command -v task-submit >/dev/null || { echo 'task-submit is required to reserve devices' >&2; exit 2; }
    exec task-submit --device "${DSV41_DEVICES:-auto}" --device-num "$ranks" --max-time "${DSV41_MAX_TIME:-300}" --run "$command"
fi
IFS=',' read -r -a devices <<< "$TASK_DEVICE"
[[ ${#devices[@]} == "$ranks" ]] || { echo "Case requires $ranks devices; reservation contains ${#devices[@]}" >&2; exit 2; }
for ((rank=0; rank<ranks; ++rank)); do
    [[ ${devices[$rank]} =~ ^[0-9]+$ ]] || { echo 'Invalid device ID' >&2; exit 2; }
    for ((peer=0; peer<rank; ++peer)); do
        [[ ${devices[$rank]} != "${devices[$peer]}" ]] || { echo 'Device IDs must be distinct' >&2; exit 2; }
    done
done
runner=${DSV41_RUNNER:-"${DSV41_BUILD_DIR:-$root/build/portable}/host/native-runner"}
[[ -x "$runner" && -f "$bundle/callable.bin" ]] || { echo 'Build the native runner and bundle first' >&2; exit 2; }
pids=()
for ((rank=0; rank<ranks; ++rank)); do
    mkdir -p "$output/rank$rank/ascend"
    ASCEND_PROCESS_LOG_PATH="$output/rank$rank/ascend" DSV41_RECORD_PATH="$output/rank$rank/submissions.bin" DSV41_ORIGINAL_ORCH="$bundle/orchestration.so" \
    "$runner" "$bundle" "$runtime" "${devices[$rank]}" "$rank" "$output/rootinfo" \
        "$output/rank$rank" "$4" "$5" "$6" "${@:7}" > "$output/rank$rank/console.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for ((rank=0; rank<ranks; ++rank)); do tail -n 12 "$output/rank$rank/console.log"; done
exit "$status"
