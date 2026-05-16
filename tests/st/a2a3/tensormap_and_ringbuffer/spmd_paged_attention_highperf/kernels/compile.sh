#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bisheng \
    -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 \
    -I"${SCRIPT_DIR}" \
    -I"${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw" \
    -I"${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl" \
    -I"${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface" \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime" \
    -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling" \
    "${SCRIPT_DIR}/paged_attention_wrapper.cpp" \
    -o "${SCRIPT_DIR}/pa_lib.so"
