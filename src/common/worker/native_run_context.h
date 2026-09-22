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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include "call_config.h"
#include "common/host_api.h"
#include "native_run_execution.h"
#include "runtime_c_api.h"
#include "runtime.h"

/** Internal phase of the caller-owned opaque native-run storage. */
enum class NativeRunPhase : uint8_t {
    Prepared,
    Running,
    Complete,
};

/**
 * The single placement-owned carrier for one progressable native lifecycle.
 * It captures immutable selection, identity, acceptance, configuration, and
 * HostApi binding during prepare; later phases never recover them from a
 * runner field or thread-local. The object is address-stable through finalize.
 */
template <typename Runner>
struct NativeRunContext {
    static constexpr uint64_t kMagic = UINT64_C(0x534d504c52554e31);  // "SMPLRUN1"

    NativeRunContext(
        Runner *runner_in, const CallConfig &config_in, uint64_t trace_hid_in, const NativeRunDescriptor &descriptor_in,
        const HostApiOps *host_api_ops
    ) :
        runner(runner_in),
        config(config_in),
        descriptor(descriptor_in),
        host_api(
            runner_in, descriptor_in.pipeline_slot, descriptor_in.arena_bank, descriptor_in.run_epoch, host_api_ops
        ),
        trace_hid(trace_hid_in) {
        // Publish the storage tag only after every potentially-throwing member
        // has been constructed. A failed placement construction must leave the
        // caller-owned slot reusable rather than looking like a prepared run.
        magic = kMagic;
    }

    NativeRunContext(const NativeRunContext &) = delete;
    NativeRunContext &operator=(const NativeRunContext &) = delete;
    NativeRunContext(NativeRunContext &&) = delete;
    NativeRunContext &operator=(NativeRunContext &&) = delete;

    /** Publish acceptance only from this run's completed launch receipt. */
    bool publish_acceptance(const LaunchReceipt &receipt) const noexcept {
        if (!receipt.matches(identity())) return false;
        if (descriptor.accepted_state != nullptr) {
            __atomic_store_n(descriptor.accepted_state, descriptor.accepted_value, __ATOMIC_RELEASE);
        }
        return true;
    }

    uint64_t magic{0};
    Runner *runner{nullptr};
    CallConfig config{};
    NativeRunDescriptor descriptor{};
    HostApi host_api;
    Runtime runtime{};
    uint64_t trace_hid{0};
    uint64_t trace_inv{0};
    long long trace_start_ns{0};
    long long runner_trace_start_ns{0};
    uint64_t clock_log_offset{0};
    int completion_rc{-1};
    std::atomic<NativeRunPhase> phase{NativeRunPhase::Prepared};
    std::unique_ptr<typename Runner::PreparedExecution> prepared_execution{};
    std::unique_ptr<typename Runner::ActiveExecution> active_execution{};
    LaunchPermit launch_permit{};
    char trace_attrs[192]{};
    bool runner_resources_owned{false};
    bool runner_reserved{false};
    bool runner_claimed{false};

    NativeRunIdentity identity() const {
        return NativeRunIdentity{
            descriptor.run_epoch, descriptor.generation, descriptor.dispatch_id, descriptor.pipeline_slot
        };
    }
};

/** End object lifetime, then mark the caller-owned storage reusable. */
template <typename Runner>
void destroy_native_run_context(NativeRunContext<Runner> *context) {
    void *storage = context;
    context->~NativeRunContext<Runner>();
    constexpr uint64_t kEmpty = 0;
    std::memcpy(storage, &kEmpty, sizeof(kEmpty));
}
