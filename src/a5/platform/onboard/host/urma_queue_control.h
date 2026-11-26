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

#include <optional>
#include "acl/acl.h"
#include "platform_comm/urma_run_lease.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "../urma_queue_command.h"
#include "aiv_control_binary.h"

namespace simpler::urma {

// A stopped Run owns this domain until the AIV has stopped and both queue tails have been published.
class UrmaQueueControl {
public:
    bool has_command() const { return active_.has_value(); }

    UrmaQueueState run(QueueCommand command, UrmaWait mode, const UrmaLaunchPermission &permission) {
        if (!active_) {
            const auto gate = permission.lock();
            if (!gate || !gate->open()) return UrmaQueueState::PENDING;
            if (!open()) return UrmaQueueState::FAILED;
            // An errored kernel can leave a stream idle without writing its result.
            const uint32_t unwritten = UINT32_MAX;
            if (!ok(aclrtMemcpy(result_, sizeof(unwritten), &unwritten, sizeof(unwritten), ACL_MEMCPY_HOST_TO_DEVICE),
                    "initialize result"))
                return UrmaQueueState::FAILED;
            command.result_address = reinterpret_cast<uint64_t>(result_);
            aclrtLaunchKernelAttr engine{};
            engine.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
            engine.value.engineType = ACL_RT_ENGINE_TYPE_AIV;
            aclrtLaunchKernelCfg config{&engine, 1};
            if (!gate->submit([&] {
                    active_ = command;
                    stopped_ = false;
                    accepted_ =
                        ok(aclrtLaunchKernelWithHostArgs(
                               function_, 1, stream_, &config, &command, sizeof(command), nullptr, 0
                           ),
                           "launch");
                }))
                return UrmaQueueState::PENDING;
            if (!accepted_) return UrmaQueueState::FAILED;
        }
        if (command.action != active_->action || command.address != active_->address ||
            command.depth != active_->depth || command.completed != active_->completed ||
            command.target != active_->target)
            return UrmaQueueState::FAILED;

        if (!stopped_) {
            if (mode == UrmaWait::Poll) {
                aclrtStreamStatus status{};
                if (!ok(aclrtStreamQuery(stream_, &status), "query")) return UrmaQueueState::FAILED;
                if (status == ACL_STREAM_STATUS_NOT_READY) return UrmaQueueState::PENDING;
                if (status != ACL_STREAM_STATUS_COMPLETE) return UrmaQueueState::FAILED;
            } else if (!ok(aclrtSynchronizeStreamWithTimeout(stream_, PLATFORM_STREAM_SYNC_TIMEOUT_MS), "wait")) {
                return UrmaQueueState::FAILED;
            }
            stopped_ = true;
        }
        uint32_t result = UINT32_MAX;
        if (accepted_ &&
            !ok(aclrtMemcpy(&result, sizeof(result), result_, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST), "result"))
            return UrmaQueueState::FAILED;
        active_.reset();
        return result <= static_cast<uint32_t>(UrmaQueueState::FAILED) ? static_cast<UrmaQueueState>(result) :
                                                                         UrmaQueueState::FAILED;
    }

    bool close() {
        if (has_command()) return false;
        if (stream_ && !ok(aclrtDestroyStream(stream_), "destroy stream")) return false;
        stream_ = nullptr;
        if (result_ && !ok(aclrtFree(result_), "free result")) return false;
        result_ = nullptr;
        if (binary_ && !ok(aclrtBinaryUnLoad(binary_), "unload binary")) return false;
        binary_ = nullptr;
        function_ = nullptr;
        return true;
    }

private:
    static bool ok(aclError rc, const char *operation) {
        if (rc == ACL_SUCCESS) return true;
        LOG_ERROR("URMA queue control: %s failed: %d", operation, static_cast<int>(rc));
        return false;
    }

    bool open() {
        if (!binary_) {
            const auto path = a5::aiv_control_binary(reinterpret_cast<const void *>(&ok));
            if (path.empty() || !ok(aclrtBinaryLoadFromFile(path.c_str(), nullptr, &binary_), "load binary"))
                return false;
        }
        if (!function_ &&
            !ok(aclrtBinaryGetFunction(binary_, "simpler_urma_queue_control", &function_), "get function"))
            return false;
        if (!stream_ && !ok(aclrtCreateStream(&stream_), "create stream")) return false;
        return result_ || ok(aclrtMalloc(&result_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST), "allocate result");
    }

    aclrtBinHandle binary_ = nullptr;
    aclrtFuncHandle function_ = nullptr;
    aclrtStream stream_ = nullptr;
    void *result_ = nullptr;
    std::optional<QueueCommand> active_;
    bool accepted_ = false;
    bool stopped_ = false;
};

}  // namespace simpler::urma
