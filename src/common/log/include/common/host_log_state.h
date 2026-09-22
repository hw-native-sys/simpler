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

/* Holds either an explicitly selected directory or the process-session spool. */
#define SIMPLER_HOST_LOG_DIR_CAPACITY 1024

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Process-owned state shared by the private HostLogger copy compiled into
 * every host-side DSO. Fixed-width fields plus a C callback keep the boundary
 * independent of the C++ library ABI; host_log.cpp performs all mutable scalar
 * accesses with compiler atomic builtins.
 *
 * log_directory is where the process writer appends records, one file per
 * process. It is empty until a caller supplies either an explicit destination
 * or the process tree's stable session spool, and the writer uses stderr while
 * it is. The destination is a property of the logger, so it applies to every
 * record from every caller — there is no per-record or per-call-site routing.
 *
 * The first non-empty path wins: log_directory_bound is release-stored after the
 * path is filled and acquire-loaded before it is read, so a reader sees either no
 * directory or the whole one, and a reader that has already opened the file never
 * has the path change under it.
 *
 * sink_owner_pid is negative while the process owner creates the bounded sink.
 * sink_process_pid distinguishes a parent restarting after a quiescent fork
 * boundary from a child that inherited owner=0; the child
 * starts fresh counters. The high bit of sink_producer_state closes admission;
 * its low bits count callers that may still hold sink_context. Bound private
 * logger copies submit complete records through sink_enqueue without exporting
 * or interposing another ELF symbol.
 *
 * dropped_record_count is the total; dropped_by_reason attributes it. A total
 * without the breakdown cannot answer the question a reader actually has —
 * whether the queue was too small, the claim budget too tight, or the
 * destination broken — and those three want opposite fixes. reported_drop_count
 * is the total the last written loss summary named, so the quiescent boundary
 * reports a growth rather than repeating a cumulative figure at every fork.
 */
struct SimplerHostLogState;

/* Why a record never reached the destination. Each drop increments the total and
 * exactly one of these. */
enum SimplerHostLogDropReason {
    /* The consumer had not freed the slot this record's position maps to. */
    SIMPLER_HOST_LOG_DROP_QUEUE_FULL = 0,
    /* kProducerClaimAttempts atomic attempts did not win a position. Distinct
     * from a full queue: the queue had room and contention took the budget. */
    SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED = 1,
    /* write(2) to the file or to stderr failed. */
    SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED = 2,
    /* Admission was closed, or no sink was published to submit to. */
    SIMPLER_HOST_LOG_DROP_NOT_ADMITTED = 3,
    SIMPLER_HOST_LOG_DROP_REASON_COUNT = 4
};

/* Return nonzero only after the complete record has been accepted by the
 * process sink. A zero return has already been attributed to a reason by the
 * callee, so the caller must not count it again. */
typedef int (*SimplerHostLogEnqueueFn)(
    void *context, struct SimplerHostLogState *state, const char *record, uint32_t size
);

typedef struct SimplerHostLogState {
    int32_t threshold;
    int32_t log_directory_bound;
    char log_directory[SIMPLER_HOST_LOG_DIR_CAPACITY];
    int32_t sink_owner_pid;
    int32_t sink_process_pid;
    void *sink_context;
    SimplerHostLogEnqueueFn sink_enqueue;
    uint64_t dropped_record_count;
    uint64_t pending_record_count;
    uint64_t sink_producer_state;
    uint64_t dropped_by_reason[SIMPLER_HOST_LOG_DROP_REASON_COUNT];
    uint64_t reported_drop_count;
} SimplerHostLogState;

typedef int (*SimplerHostLogBindStateFn)(SimplerHostLogState *state);

/* The only cross-DSO logger symbol. Each consumer exports its own copy and its
 * loader resolves it from that consumer's handle before the module is used. */
int simpler_host_log_bind_state(SimplerHostLogState *state);

#ifdef __cplusplus
}
#endif
