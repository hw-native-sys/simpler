/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <stdint.h>

#define SIMPLER_HOST_SPAN_ABI_VERSION 1U

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SimplerHostSpan {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t invocation_id;
    uint64_t callable_hash;
    int32_t depth;
    int32_t reserved;
    int64_t timestamp_ns;
    int64_t duration_ns;
    const char *name;
    const char *attributes;
} SimplerHostSpan;

typedef void (*SimplerHostSpanSink)(const SimplerHostSpan *span);

void simpler_log_emit_host_span(const SimplerHostSpan *span);

#ifdef __cplusplus
}
#endif
