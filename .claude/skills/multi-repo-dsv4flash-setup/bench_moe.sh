#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark the DeepSeek-V4 FLASH distributed MoE with PYPTO_BENCH. Reads
# fast_effective_us (the clean per-round fastest-rank metric) over 100 rounds.
# Requires moe.py:36 edited to the desired experts-per-card (// 16 => 16/card)
# and newest simpler carrying the _start_hierarchical idempotency guard, else
# it dies with "Worker: add_worker after init". See SKILL.md §3.4 / §3.5.
#   usage: bench_moe.sh <SIMPLER_WORKTREE_ROOT> <DEVICE_LIST> [EP]
#   e.g.:  bench_moe.sh "$PWD" 2,4 2
set -uo pipefail
ROOT=${1:?simpler worktree root}
DEVS=${2:?device list, e.g. 2,4}
EP=${3:-2}
PTOAS_VER=${PTOAS_VER:-0.48}
PTO_ISA_COMMIT=${SIMPLER_PTO_ISA_COMMIT:-83d01313}

source "$ROOT/.venv/bin/activate"
eval "$(pypto-setup --export 2>/dev/null)"
export PTOAS_ROOT="/usr/local/ptoas/$PTOAS_VER"
export PATH="/usr/local/ptoas/$PTOAS_VER/bin:$PATH"
export PTO_ISA_ROOT="$ROOT/build/pto-isa" SIMPLER_PTO_ISA_COMMIT="$PTO_ISA_COMMIT"
export PYPTO_BENCH=1

cd "$ROOT/build/pypto-lib" || exit 1
echo "=== MoE EP$EP, devs=$DEVS (per-card experts from moe.py:36 divisor) ==="
python models/deepseek/v4/moe.py -p a2a3 -d "$DEVS" --ep "$EP" 2>&1 \
  | grep -iE "fast_effective_us|effective_us|N_LOCAL|expert|PASS|FAIL|timed out|add_worker|Error" \
  | awk 'NR<=15'
echo DONE
