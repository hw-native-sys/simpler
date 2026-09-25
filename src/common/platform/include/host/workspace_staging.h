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

#include <cstdint>

#include "runtime_c_api.h"

/**
 * What a device context was asked to do about its workspace, before it is in a
 * position to do it.
 *
 * Management belongs only to a program context, and the execution mode is not
 * latched until `simpler_init` runs — while the two public requests
 * (`simpler_set_workspace_budget_ctx` and
 * `simpler_enable_workspace_management_ctx`) are made before it. So the
 * request is recorded here and consumed once, after the latch and before the
 * eager prewarm takes the first device memory.
 *
 * Holds no resource: a context that records a request and never installs it is
 * exactly the context it was before.
 *
 * Its own type rather than two fields on the runner because the "once"
 * semantics used to come from `WorkspaceManager::configure` refusing a second
 * call, and deferring the install moved that guarantee somewhere it has to be
 * stated and tested on its own.
 */
class WorkspaceStagingRequest {
public:
    /** What an install should do with this request. */
    enum class Install : uint32_t {
        /** Nothing was asked for; leave the context unmanaged. */
        Nothing = 0,
        /** Manage the four regions, enforcing no byte limit. */
        ManageOnly,
        /** Manage them and enforce `limit_bytes`. */
        ManageWithLimit,
    };

    /**
     * Record one request. `limit_bytes == 0` asks for management alone.
     *
     * The first accepted request stands: a second is refused rather than
     * replacing it, so what a context ends up enforcing does not depend on
     * call order.
     *
     * @param installed  whether management is already live on this context
     * @return 0 on success; PTO_RUNTIME_ERR_INVALID_STATE once installed;
     *         PTO_RUNTIME_ERR_INVALID_ARGUMENT for a second request
     */
    int record(uint64_t limit_bytes, bool installed) {
        if (installed) return PTO_RUNTIME_ERR_INVALID_STATE;
        if (requested_) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        requested_ = true;
        limit_bytes_ = limit_bytes;
        return 0;
    }

    /** Forget a request that never became an installation. */
    void clear() noexcept {
        requested_ = false;
        limit_bytes_ = 0;
    }

    bool requested() const noexcept { return requested_; }
    uint64_t limit_bytes() const noexcept { return limit_bytes_; }

    /**
     * What the install step owes this request.
     *
     * A limit recorded with no management request is unreachable through the
     * two public entries — both go through `record` — so it is reported as a
     * caller error rather than silently enforcing nothing.
     */
    Install plan() const noexcept {
        if (!requested_) return Install::Nothing;
        return limit_bytes_ == 0 ? Install::ManageOnly : Install::ManageWithLimit;
    }

private:
    bool requested_{false};
    uint64_t limit_bytes_{0};
};
