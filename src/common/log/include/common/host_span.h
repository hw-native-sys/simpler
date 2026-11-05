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

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Widths of the two text fields a span record carries, published because a
 * producer that formats one of them has to size its own buffer to match: the
 * logger copies with `snprintf`, which truncates in silence, and a producer
 * whose buffer is wider than the field truncates twice — once unmarked in the
 * record it hands over, once marked by the logger.
 *
 * POSIX guarantees atomic pipe writes up to _POSIX_PIPE_BUF (512 bytes), and a
 * conservative bound for the logger prefix, the fixed-width STRACE fields, and
 * the newline is 256 bytes, which leaves these two the other half.
 */
#define SIMPLER_HOST_SPAN_NAME_CAPACITY 64
#define SIMPLER_HOST_SPAN_ATTRIBUTES_CAPACITY 192

/*
 * One span record on its way to the logger. A stack temporary, handed to the
 * link-time unified_log_host_span in the same DSO, so it carries no version or
 * size word: the two producers are inline in this repository's headers and the
 * validator is compiled from the same header in the same build, which leaves
 * nothing for such a word to detect.
 */
typedef struct SimplerHostSpan {
    uint64_t invocation_id;
    uint64_t callable_hash;
    int32_t depth;
    int32_t reserved;
    int64_t timestamp_ns;
    int64_t duration_ns;
    const char *name;
    const char *attributes;
} SimplerHostSpan;

/* Link-time adapters used by native Host consumers. The enabled query is the
 * cheap pre-format gate for TIMING-level spans; emission rechecks the level. */
int unified_log_host_span_enabled(void);
void unified_log_host_span(const SimplerHostSpan *span);

/* Thread-owned scalars with no pointers into a producer DSO. */
typedef struct SimplerHostTraceState {
    uint64_t inv;
    int32_t depth;
    uint64_t hid;
} SimplerHostTraceState;

/* Null when the process owner has not initialized tracing or allocation fails. */
SimplerHostTraceState *unified_log_host_trace_state(void);
/* IDs start at 1; zero means no invocation was allocated because tracing is unavailable. */
uint64_t unified_log_host_trace_allocate_inv(void);

#ifdef __cplusplus
}
#endif
