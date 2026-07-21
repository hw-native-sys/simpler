#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark the three DeepSeek-V4 FLASH decode attentions (swa/csa/hca) with
# PYPTO_BENCH. Reads effective_us over 100 rounds. Pins ptoas 0.48 (csa cannot
# compile on 0.50) and start_pos=8192 (canonical long-context workload).
#   usage: bench_attn.sh <SIMPLER_WORKTREE_ROOT> <DEVICE_ID> [START_POS]
set -uo pipefail
ROOT=${1:?simpler worktree root}
DEV=${2:?device id (single die)}
START_POS=${3:-8192}
PTOAS_VER=${PTOAS_VER:-0.48}
PTO_ISA_COMMIT=${SIMPLER_PTO_ISA_COMMIT:-83d01313}

source "$ROOT/.venv/bin/activate"
eval "$(pypto-setup --export 2>/dev/null)"
# Pin ptoas AFTER pypto-setup --export, whose PATH otherwise wins with the
# slow default ptoas-bin that times out compiling csa.
export PTOAS_ROOT="/usr/local/ptoas/$PTOAS_VER"
export PATH="/usr/local/ptoas/$PTOAS_VER/bin:$PATH"
export PTO_ISA_ROOT="$ROOT/build/pto-isa" SIMPLER_PTO_ISA_COMMIT="$PTO_ISA_COMMIT"
export PYPTO_BENCH=1

cd "$ROOT/build/pypto-lib" || exit 1
echo "ptoas=$(command -v ptoas)  start_pos=$START_POS  device=$DEV"
for V in swa csa hca; do
  echo "===== decode_attention_$V (start_pos=$START_POS) ====="
  python models/deepseek/v4/decode_attention_$V.py -p a2a3 -d "$DEV" \
    --start-pos "$START_POS" 2>&1 \
    | grep -E "effective_us|PASS|FAIL|timed out" | awk 'NR<=4'
done
echo DONE
