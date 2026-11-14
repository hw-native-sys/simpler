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
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$root/../../.." && pwd)
build=${DSV41_BUILD_DIR:-"$root/build/portable"}
jobs=${DSV41_BUILD_JOBS:-4}
host_only=0
[[ ${1:-} != a5 ]] || shift
if [[ ${1:-} == --host-only ]]; then host_only=1; shift; fi
if [[ ${1:-} == --help ]]; then
    echo 'usage: build.sh [a5] [--host-only] [--flavor pypto_lib|pypto_lib_dsl|pypto_lib_latest|both|all] [--pto-isa PATH]'
    echo 'DSV41_BUILD_DIR, DSV41_BUILD_JOBS and DSV41_PYTHON select the build directory, concurrency and Python.'
    exit 0
fi
cmake -S "$root/production" -B "$build/host" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSIMPLER_ROOT="$repo"
cmake --build "$build/host" --parallel "$jobs"
if ((host_only == 0)); then
    export PYTHONPATH="$repo:$repo/python${PYTHONPATH:+:$PYTHONPATH}"
    exec "${DSV41_PYTHON:-python3}" "$root/production/build_sources.py" --build-dir "$build" --jobs "$jobs" "$@"
fi
