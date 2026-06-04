# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Shared compiler-sanitizer helper.
#
# `SIMPLER_SANITIZERS` is a comma-separated `-fsanitize` token list (e.g.
# "address,undefined" or "thread"), passed straight through to the compiler.
# Empty (the default) makes every call below a no-op, so ordinary builds are
# byte-for-byte unchanged.
#
# Apply ONLY to host-compiled targets — the sim runtime/kernels/orchestration
# and the onboard *host* runtime. NEVER call it for device toolchains (ccec for
# AICore, aarch64 cross for the AICPU): they run on the NPU and cannot carry a
# host sanitizer runtime.
#
# `-O1` (the last `-O` wins, overriding an earlier `-O3`) plus frame pointers
# keep sanitizer stack traces from being inlined away — the standard
# good-report settings.

function(simpler_apply_sanitizers tgt)
  if(NOT SIMPLER_SANITIZERS)
    return()
  endif()
  target_compile_options(${tgt} PRIVATE
    -fsanitize=${SIMPLER_SANITIZERS}
    -fno-omit-frame-pointer
    -O1)
  target_link_options(${tgt} PRIVATE -fsanitize=${SIMPLER_SANITIZERS})
  # TSAN can't model standalone std::atomic_thread_fence and warns about it;
  # the AICPU target compiles with -Werror, which would make that warning
  # fatal. Keep the warning visible but non-fatal — the limitation is known
  # and acceptable (fence-ordered accesses just aren't TSAN-tracked there).
  if(SIMPLER_SANITIZERS MATCHES "thread")
    target_compile_options(${tgt} PRIVATE -Wno-error=tsan)
  endif()
endfunction()
