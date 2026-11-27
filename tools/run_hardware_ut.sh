#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# task-submit supplies the allocated device IDs. Keep this one allocation while
# collecting all selected suites' results, even when one suite fails.
set -eu
arch="${1:?expected architecture}"
run_cpp="${2:?expected C++ suite selection}"
IFS=, read -ra devices <<< "${TASK_DEVICE:?expected task-submit device IDs}"

status=0
python -m pytest tests/ut -m 'requires_hardware and device_count' \
    --platform "$arch" --device "$TASK_DEVICE" -q || status=$?
python -m pytest tests/ut -m 'requires_hardware and not device_count' \
    --platform "$arch" --device "$TASK_DEVICE" -n "${#devices[@]}" -q || status=$?

if [[ "$run_cpp" == true ]]; then
    python - <<'PY'
import json
import os
from pathlib import Path

npus = [{"id": device, "slots": 1} for device in os.environ["TASK_DEVICE"].split(",")]
resources = {"version": {"major": 1, "minor": 0}, "local": [{"npus": npus}]}
Path("build/ut-hardware/resources.json").write_text(json.dumps(resources))
PY
    ctest --test-dir build/ut-hardware -L "^requires_hardware_${arch}$" \
        --resource-spec-file "$PWD/build/ut-hardware/resources.json" \
        -j "$(nproc)" --output-on-failure --timeout 300 --no-tests=error || status=$?
fi
exit "$status"
