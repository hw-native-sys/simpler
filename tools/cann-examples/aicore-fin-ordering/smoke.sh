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

# a2a3-only at runtime. The COND register offset this tool polls is per-arch --
# REG_SPR_COND_OFFSET is 0x4C8 on a2a3 and 0x5108 on a5
# (src/{a2a3,a5}/platform/include/common/platform_config.h) -- and the handshake
# header hardcodes the a2a3 value. The AICore halves compile for either arch, so
# the build is covered on both; only the run is gated. Exit 77 marks a skip the
# runner counts and names, rather than a pass.
if [ "${SIMPLER_SMOKE_ARCH:-a2a3}" != "a2a3" ]; then
    echo "skipping: hardcodes a2a3's COND offset (0x4C8); a5 uses 0x5108" >&2
    exit 77
fi

# Path is per-arch, so the caller supplies it: build_cann_examples.sh --run
# derives it from the ccec target, and a by-hand run points it at the arch being
# exercised. Defaulting to one arch here would send an a5 run at a2a3's build.
: "${SIMPLER_DISPATCHER_SO:?set SIMPLER_DISPATCHER_SO to build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so}"
if [ ! -f "$SIMPLER_DISPATCHER_SO" ]; then
    echo "dispatcher SO not found at $SIMPLER_DISPATCHER_SO — run pip install first" >&2
    exit 1
fi

export FIN_ORDER_CONSUMER_SO="$HERE/device-aicpu/build/libfin_order_consumer.so"
export FIN_ORDER_PRODUCER_O="$HERE/device-aicore/build/fin_order_producer.o"
# stage 0 with the smallest useful round count: the bisect stages (1-3) stop
# early and exit non-zero with an empty table, which reads as a failure.
# 2 rounds per cell keeps the whole matrix under a smoke budget.
exec "$HERE/host/build/launch_fin_order" "$DEV" 0 2
