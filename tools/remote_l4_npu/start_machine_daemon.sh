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

: "${SIMPLER_REMOTE_L4_NPU_ROLE:=machine}"
: "${SIMPLER_REMOTE_L4_NPU_HOST:=0.0.0.0}"
: "${SIMPLER_REMOTE_L4_NPU_PORT:=19073}"

cd "${ROOT_DIR}"
source .venv/bin/activate

echo "[remote-l4-npu] role=${SIMPLER_REMOTE_L4_NPU_ROLE}"
echo "[remote-l4-npu] daemon=${SIMPLER_REMOTE_L4_NPU_HOST}:${SIMPLER_REMOTE_L4_NPU_PORT}"

exec python -m simpler.remote_l3_worker \
  --host "${SIMPLER_REMOTE_L4_NPU_HOST}" \
  --port "${SIMPLER_REMOTE_L4_NPU_PORT}"
