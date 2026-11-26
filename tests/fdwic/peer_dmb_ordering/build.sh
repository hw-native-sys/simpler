#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -eo pipefail
source "${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}"
set -u
src=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$src/../../.." && pwd)
mkdir -p "${1:?pass a build directory}" "$1/scratch"
out=$(cd "$1" && pwd)
export TMPDIR="$out/scratch"
"$ASCEND_HOME_PATH/bin/ccec" -O2 -g -c -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-c310-cube \
    -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false -I"$ASCEND_HOME_PATH/include" \
    "$src/probe.cpp" -o "$out/probe.o"
"$ASCEND_HOME_PATH/bin/ld.lld" -m aicorelinux -Ttext=0 -static -o "$out/probe.bin" "$out/probe.o"
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" -d "$out/probe.bin" > "$out/probe.asm"
g++ -O2 -std=c++17 -Wall -Wextra "$src/launch.cpp" "$root/src/common/log/"{host_log,unified_log_host}.cpp \
    -I"$root/src/common/log/include" -I"$root/src/common/platform/include" -I"$root/src/a5/platform/include" \
    -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/pkg_inc" -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -L"$ASCEND_HOME_PATH/lib64" -L"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64/driver" \
    -Wl,--no-as-needed -lascend_hal -lascendcl -lruntime -ldl -pthread -Wl,--allow-shlib-undefined -o "$out/launch"
