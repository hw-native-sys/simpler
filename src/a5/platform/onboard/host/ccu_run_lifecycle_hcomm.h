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
#include <functional>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_res.h"
#include "platform_comm/ccu/ccu_run_lifecycle.h"

namespace simpler::ccu {

struct HcommMissionStream {
    uint64_t handle{};
    ThreadHandle thread{};
    aclrtStream stream{};
};

class HcommRunLifecycleSdk final : public RunLifecycleSdk {
public:
    using ReadyEventPoller = std::function<int(uint64_t, uint64_t, bool *)>;

    HcommRunLifecycleSdk(std::vector<HcommMissionStream> streams, ReadyEventPoller ready_event_poller);

    int launch(
        uint64_t program_handle, const uint64_t *arguments, uint32_t argument_count, uint64_t stream_handle
    ) override;
    int poll_ready_event(uint64_t address, uint64_t mask, bool *ready) override;
    int query_stream(uint64_t stream_handle, StreamState *state) override;
    int abort_stream(uint64_t stream_handle) override;

private:
    const HcommMissionStream *find_stream(uint64_t handle) const;

    std::vector<HcommMissionStream> streams_;
    ReadyEventPoller ready_event_poller_;
};

}  // namespace simpler::ccu
