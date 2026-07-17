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

/**
 * Runtime Failure Line (host)
 *
 * The whole host-side report of a latched runtime failure: the machine-readable line, then
 * what the code means and what to do about it. One macro, because the two halves are only
 * correct together -- a caller cannot emit the raw codes and forget the explanation.
 *
 * A macro rather than a function on purpose. LOG_ERROR bakes in __FUNCTION__, __FILENAME__
 * and __LINE__ at its expansion point, so wrapping it in a function would stamp every line
 * with this header's name instead of the site that actually detected the failure. Expanding
 * at the call site keeps the log pointing at validate_runtime_impl in the caller's
 * runtime_maker.cpp, which is the thing a reader needs.
 *
 * The first line is parsed by the STs (which match on "orch_error_code=N") and mirrored in
 * tests/ut/py/test_resource_failure_summary.py, so its text is kept byte-for-byte stable.
 * The meaning goes on the lines *after* it, never inside it.
 *
 * Kept separate from error_names.h so that header stays free of any logging dependency
 * and can be unit-tested on its own. Include the runtime's pto_runtime_status.h first.
 */

#ifndef SRC_COMMON_RUNTIME_STATUS_ERROR_LOG_H_
#define SRC_COMMON_RUNTIME_STATUS_ERROR_LOG_H_

#include <stdint.h>

#include "error_names.h"
#include "common/unified_log.h"

// Reports a non-zero runtime_status: the codes, then their meaning and triage hint. A code
// this build does not know about is left un-annotated rather than mislabeled -- a bare number
// beats a neighbouring code's text.
#define LOG_RUNTIME_FAILURE(orch_error_code, sched_error_code, runtime_status)                                      \
    do {                                                                                                            \
        const int32_t rtf_orch_ = (orch_error_code);                                                                \
        const int32_t rtf_sched_ = (sched_error_code);                                                              \
        LOG_ERROR(                                                                                                  \
            "PTO2 runtime failed: orch_error_code=%d sched_error_code=%d runtime_status=%d", rtf_orch_, rtf_sched_, \
            (runtime_status)                                                                                        \
        );                                                                                                          \
        const int32_t rtf_code_ = latched_error_code(rtf_orch_, rtf_sched_);                                        \
        if (rtf_code_ != PTO2_ERROR_NONE && error_desc(rtf_code_)[0] != '\0') {                                     \
            LOG_ERROR(                                                                                              \
                "error detail: %s=%d %s - %s", latched_error_field(rtf_orch_, rtf_sched_), rtf_code_,               \
                error_name(rtf_code_), error_desc(rtf_code_)                                                        \
            );                                                                                                      \
            if (error_hint(rtf_code_)[0] != '\0') {                                                                 \
                LOG_ERROR("error hint: %s. See docs/troubleshooting/device-error-codes.md", error_hint(rtf_code_)); \
            }                                                                                                       \
        }                                                                                                           \
    } while (0)

#endif  // SRC_COMMON_RUNTIME_STATUS_ERROR_LOG_H_
