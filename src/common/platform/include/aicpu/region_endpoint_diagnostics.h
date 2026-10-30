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

#include <stdio.h>

#include "aicpu/region_instance_view.h"
#include "common/endpoint_failure_identity.h"
#include "common/region_task_args.h"
#include "orchestration_api.h"

// A detailed line carries fields past the Host short record, so the native log
// is not itself a Host marker.

inline const char *region_view_op_token(RegionViewOp op) {
    switch (op) {
    case RegionViewOp::CONSTRUCT:
        return "construct";
    case RegionViewOp::PAYLOAD_READ:
        return "payload_read";
    case RegionViewOp::PAYLOAD_WRITE:
        return "payload_write";
    case RegionViewOp::NOTIFY:
        return "notify";
    case RegionViewOp::TEST:
        return "test";
    case RegionViewOp::WAIT:
        return "wait";
    case RegionViewOp::COUNTER_ADDR:
        return "counter_addr";
    }
    return "unknown";
}

inline void region_copy_log_text(char *dst, size_t dst_size, const char *src) {
    if (dst == nullptr || dst_size == 0) return;
    size_t written = 0;
    if (src != nullptr) {
        for (; src[written] != '\0' && written + 1 < dst_size; ++written) {
            const char ch = src[written];
            dst[written] = (ch == '\n' || ch == '\r') ? ' ' : ch;
        }
    }
    dst[written] = '\0';
}

inline const char *region_task_arg_error_text(RegionTaskArgError error) {
    switch (error) {
    case RegionTaskArgError::OK:
        return "ok";
    case RegionTaskArgError::NULL_POINTER:
        return "null pointer";
    case RegionTaskArgError::BAD_SCALAR_COUNT:
        return "bad scalar count";
    case RegionTaskArgError::BAD_MAGIC_VERSION:
        return "bad magic version";
    case RegionTaskArgError::BAD_REGION_ID:
        return "bad region id";
    case RegionTaskArgError::BAD_TRANSACTION_ID:
        return "bad transaction id";
    case RegionTaskArgError::BAD_PAYLOAD_RANGE:
        return "bad payload range";
    case RegionTaskArgError::BAD_COUNTER_RANGE:
        return "bad counter range";
    }
    return "invalid task args";
}

inline void region_report_endpoint_error(
    uint64_t session_bits, uint64_t transaction_id, uint64_t region_id, const RegionViewError &error,
    uint64_t counter_addr, int32_t counter_operand
) {
    char detail[256];
    region_copy_log_text(detail, sizeof(detail), error.message);
    char message[640];
    const bool trusted = transaction_id != 0;
    if (trusted) {
        snprintf(
            message, sizeof(message),
            "Region endpoint error session=0x%016llx transaction=%llu region=%llu op=%s kind=%u "
            "counter_addr=%llu counter_operand=%d observed_counter=%d msg=%s",
            static_cast<unsigned long long>(session_bits), static_cast<unsigned long long>(transaction_id),
            static_cast<unsigned long long>(region_id), region_view_op_token(error.op),
            static_cast<unsigned>(error.kind), static_cast<unsigned long long>(counter_addr), counter_operand,
            error.observed, detail
        );
    } else {
        snprintf(
            message, sizeof(message), "Region endpoint error identity=untrusted op=%s kind=%u msg=%s",
            region_view_op_token(error.op), static_cast<unsigned>(error.kind), detail
        );
    }
    rt_report_endpoint_failure(
        kEndpointFailureKindRegion, trusted ? session_bits : 0, trusted ? transaction_id : 0, message
    );
}

inline void region_report_invalid_task_args(RegionTaskArgError error) {
    char message[256];
    snprintf(
        message, sizeof(message), "Region endpoint error identity=untrusted op=decode kind=%u msg=%s",
        static_cast<unsigned>(error), region_task_arg_error_text(error)
    );
    rt_report_endpoint_failure(kEndpointFailureKindRegion, 0, 0, message);
}
