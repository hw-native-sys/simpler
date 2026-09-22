#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Smoke-run this tool on one device. tools/build_cann_examples.sh --run calls it
# after building, with $1 = device id; it is also the shortest by-hand run.
#
# Kept next to the tool because the env var names it reads, the directory it must
# run from, and the argv that makes a short run are all specific to it -- so a
# shared runner cannot know them, and a by-name list in CI would have to.
set -euo pipefail
DEV="${1:?device id required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"

# Path is per-arch, so the caller supplies it: build_cann_examples.sh --run
# derives it from the ccec target, and a by-hand run points it at the arch being
# exercised. Defaulting to one arch here would send an a5 run at a2a3's build.
: "${SIMPLER_DISPATCHER_SO:?set SIMPLER_DISPATCHER_SO to build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so}"
if [ ! -f "$SIMPLER_DISPATCHER_SO" ]; then
    echo "dispatcher SO not found at $SIMPLER_DISPATCHER_SO — run pip install first" >&2
    exit 1
fi

export AICPU_DEVICE_QUERY_SO="$HERE/device/build/libquery_device.so"
exec "$HERE/host/build/query_device_hal" "$DEV"
