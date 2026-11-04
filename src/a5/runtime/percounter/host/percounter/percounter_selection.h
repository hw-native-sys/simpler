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
 * Whether a run asked for the percounter scheduler.
 *
 * Host-only: it reads the environment. Header-only because the UT targets that
 * compile runtime_maker.cpp list their sources by name, so a separate
 * translation unit here would have to be added to each of them.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "common/unified_log.h"

namespace simpler::hbg::percounter {

// Runtime enum selector, named per docs/dfx/profiling-config-naming.md.
inline constexpr const char *SCHEDULER_TYPE_ENV = "SIMPLER_HBG_SCHEDULER_TYPE";

enum class SchedulerRequest : uint8_t {
    DEFAULT = 0,
    PERCOUNTER,
};

/**
 * Parse a SIMPLER_HBG_SCHEDULER_TYPE value.
 *
 * nullptr and "" both mean unset and yield DEFAULT. Any other spelling than
 * "percounter" returns false: a misspelled knob has to fail the bind, because
 * silently running the default scheduler only surfaces much later as a
 * performance mystery.
 */
inline bool parse_scheduler_request(const char *text, SchedulerRequest *out) {
    if (text == nullptr || text[0] == '\0') {
        *out = SchedulerRequest::DEFAULT;
        return true;
    }
    if (std::strcmp(text, "percounter") == 0) {
        *out = SchedulerRequest::PERCOUNTER;
        return true;
    }
    return false;
}

/**
 * Read SIMPLER_HBG_SCHEDULER_TYPE. Returns false, after logging the value, when
 * it is not a scheduler type; the caller fails the bind.
 */
inline bool resolve_scheduler_request(SchedulerRequest *out) {
    const char *text = std::getenv(SCHEDULER_TYPE_ENV);
    if (!parse_scheduler_request(text, out)) {
        LOG_ERROR("A5 HBG: %s=%s is not a scheduler type; leave it unset or set percounter", SCHEDULER_TYPE_ENV, text);
        return false;
    }
    return true;
}

}  // namespace simpler::hbg::percounter
