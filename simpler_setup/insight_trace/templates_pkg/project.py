# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from ..models import TraceConfig
from .common import _arg_to_json, _require_kernel


def render_cmake(config: TraceConfig) -> str:
    arch = config.platform_arch
    runtime_includes = "\n".join(f"    ${{REPO_ROOT}}/{r}" for r in arch.runtime_include_roots)
    platform_includes = "\n".join(f"    ${{REPO_ROOT}}/{p}" for p in arch.platform_include_roots)
    return f"""cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project(insight_trace_replay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set (source CANN set_env.sh first)")
endif()
set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
set(SOC_VERSION {arch.soc_version} CACHE STRING "Simulator SoC version")
set(PTO_ISA_ROOT $ENV{{PTO_ISA_ROOT}} CACHE PATH "PTO ISA root")
set(REPO_ROOT $ENV{{REPO_ROOT}} CACHE PATH "simpler repo root")

add_compile_options(
    -D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong -fPIC
)
add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce -fenable-matrix --cce-aicore-enable-tl -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)
set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

set(COMMON_INCLUDES
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/include/pto
{runtime_includes}
    ${{REPO_ROOT}}/src/common/task_interface
{platform_includes}
    ${{REPO_ROOT}}/simpler_setup/incore
    ${{ASCEND_HOME_PATH}}/pkg_inc
    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling
    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
    ${{ASCEND_HOME_PATH}}/include
)

add_library(replay_kernel SHARED replay_kernel.cpp replay_launch.cpp)
target_compile_options(replay_kernel PRIVATE
    ${{CMAKE_CCE_COMPILE_OPTIONS}}
    --cce-aicore-arch={arch.cce_aicore_arch}
    -DREGISTER_BASE -std=c++17)
target_include_directories(replay_kernel PRIVATE ${{COMMON_INCLUDES}})
target_link_options(replay_kernel PRIVATE --cce-fatobj-link)

add_executable(replay_host replay_host.cpp)
target_compile_options(replay_host PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
target_include_directories(replay_host PRIVATE ${{COMMON_INCLUDES}})
target_link_directories(replay_host PUBLIC
    ${{ASCEND_HOME_PATH}}/lib64
    ${{ASCEND_HOME_PATH}}/aarch64-linux/simulator/${{SOC_VERSION}}/lib
)
target_link_libraries(replay_host PRIVATE
    replay_kernel
    runtime_camodel
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)
"""


def render_run_collect(config: TraceConfig) -> str:
    cann_default = str(config.cann_home) if config.cann_home else ""
    pto_default = str(config.pto_isa_root) if config.pto_isa_root else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

CANN_HOME="${{CANN_HOME:-{cann_default}}}"
PTO_ISA_ROOT="${{PTO_ISA_ROOT:-{pto_default}}}"
REPO_ROOT="${{REPO_ROOT:-{config.repo_root}}}"
: "${{CANN_HOME:?CANN_HOME must be set}}"
: "${{PTO_ISA_ROOT:?PTO_ISA_ROOT must be set}}"
: "${{REPO_ROOT:?REPO_ROOT must be set}}"

WS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SOC_VERSION="${{SOC_VERSION:-{config.platform_arch.soc_version}}}"
DEVICE_ID="${{TARGET_DEVICE_ID:-${{NPU_LOCKED_DEVICE:-{config.device}}}}}"
BUILD_DIR="$WS/build"
COLLECT_DIR="$WS/msprof_collect"
EXPORT_ROOT="$WS/insight_export"

source "$CANN_HOME/../cann/set_env.sh" 2>/dev/null \
  || source "$CANN_HOME/set_env.sh"
export ASCEND_HOME_PATH="$CANN_HOME"
SIM_LIB_DIR="$CANN_HOME/aarch64-linux/simulator/$SOC_VERSION/lib"
export LD_LIBRARY_PATH="$BUILD_DIR:$SIM_LIB_DIR:$CANN_HOME/lib64:\
$CANN_HOME/aarch64-linux/devlib:$CANN_HOME/devlib:${{LD_LIBRARY_PATH:-}}"
export ACL_DEVICE_ID="$DEVICE_ID"
mkdir -p "$BUILD_DIR" "$COLLECT_DIR" "$EXPORT_ROOT"
chmod 700 "$COLLECT_DIR" "$EXPORT_ROOT"

cmake -G Ninja -S "$WS" -B "$BUILD_DIR" \
    -DSOC_VERSION="$SOC_VERSION" \
    -DPTO_ISA_ROOT="$PTO_ISA_ROOT" \
    -DREPO_ROOT="$REPO_ROOT"
cmake --build "$BUILD_DIR" --target replay_host

msprof op simulator \
  --application="$BUILD_DIR/replay_host" \
  --kernel-name="replay_entry" \
  --launch-count={config.launch_count} \
  --soc-version="$SOC_VERSION" \
  --timeout={config.timeout} \
  --output="$COLLECT_DIR/out" \
  2>&1 | tee "$COLLECT_DIR/msprof_collect.log"

OPPROF_DIR="$(find "$COLLECT_DIR/out" -maxdepth 1 -mindepth 1 -type d -name 'OPPROF_*' | sort | tail -n 1)"
test -n "$OPPROF_DIR"
if [[ -d "$OPPROF_DIR/device0/tmp_dump" ]]; then
  EXPORT_SRC="$OPPROF_DIR/device0/tmp_dump"
else
  EXPORT_SRC="$OPPROF_DIR/dump"
fi

msprof op simulator --export="$EXPORT_SRC" --output="$EXPORT_ROOT" \
  2>&1 | tee "$EXPORT_ROOT/msprof_export.log"
"""


def render_config(config: TraceConfig) -> dict:
    kernel = _require_kernel(config)
    result = {
        "backend": config.backend.value,
        "test_module": str(config.test_module) if config.test_module else None,
        "case": config.case_name,
        "kernel": {
            "name": kernel.name,
            "func_id": kernel.func_id,
            "core_type": kernel.core_type,
            "source": str(kernel.source_path),
            "shape": kernel.shape.value,
        },
        "replay": {
            "hw_block_num": config.hw_block_num,
            "soc_version": config.platform_arch.soc_version,
            "platform": config.platform_arch.family.value,
            "timeout": config.timeout,
            "launch_count": config.launch_count,
        },
        "args": [_arg_to_json(arg) for arg in config.args],
    }
    if config.spmd_meta is not None:
        result["spmd"] = {
            "hw_block_dim": config.spmd_meta.hw_block_dim,
            "aiv_lanes_per_core": config.spmd_meta.aiv_lanes_per_core,
            "fifo_sizes": list(config.spmd_meta.fifo_sizes),
            "dispatches": [
                {
                    "logical_block_num": dispatch.logical_block_num,
                    "scalar_overrides": [[index, value] for index, value in dispatch.scalar_overrides],
                }
                for dispatch in config.spmd_meta.dispatches
            ],
        }
    return result
