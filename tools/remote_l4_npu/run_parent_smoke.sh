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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SIMPLER_REMOTE_L4_NPU_MACHINE_A:?set machine A daemon as HOST:PORT}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_B:?set machine B daemon as HOST:PORT}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICES:=0,1}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICES:=0,1}"
: "${SIMPLER_REMOTE_L4_NPU_PLATFORM:=a2a3}"
: "${SIMPLER_REMOTE_L4_NPU_RUNTIME:=tensormap_and_ringbuffer}"
: "${SIMPLER_REMOTE_L4_NPU_SESSION_LISTEN_HOST:=0.0.0.0}"
: "${SIMPLER_REMOTE_L4_NPU_SESSION_TIMEOUT:=120}"

cd "${ROOT_DIR}"
source .venv/bin/activate

exec python -m tools.remote_l4_npu.remote_l4_npu_smoke \
  --machine-a "${SIMPLER_REMOTE_L4_NPU_MACHINE_A}" \
  --machine-b "${SIMPLER_REMOTE_L4_NPU_MACHINE_B}" \
  --machine-a-devices "${SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICES}" \
  --machine-b-devices "${SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICES}" \
  --platform "${SIMPLER_REMOTE_L4_NPU_PLATFORM}" \
  --runtime "${SIMPLER_REMOTE_L4_NPU_RUNTIME}" \
  --session-listen-host "${SIMPLER_REMOTE_L4_NPU_SESSION_LISTEN_HOST}" \
  --session-timeout "${SIMPLER_REMOTE_L4_NPU_SESSION_TIMEOUT}"
