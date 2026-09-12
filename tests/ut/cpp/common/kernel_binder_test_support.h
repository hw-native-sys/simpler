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

#include <gtest/gtest.h>
#include <array>
#include <future>
#include <mutex>
#include <vector>

#include "host/kernel_launch_binder.h"

namespace kernel_binder_test {
using namespace simpler::kernel_launch;
using Step = KernelLaunchStep;
constexpr Step Cancel = static_cast<Step>(100);
inline void *ptr(uintptr_t n) { return reinterpret_cast<void *>(n); }

struct Fake {
    std::mutex mutex;
    bool ready{true}, frozen{true}, poisoned{false}, complete{true}, pending_prepare{true};
    uintptr_t previous_caller{0};
    void *candidate_caller{nullptr};
    int calls{0}, fail_at{0}, second_fail_at{0}, query_error{0}, validation_error{0};
    int queries{0}, acquisitions{0}, finishes{0}, runtime_error{0}, cleanup_error{0};
    std::vector<Step> trace;
    std::promise<void> *entered{nullptr};
    std::shared_future<void> release;
    KernelLaunchHandles handles{nullptr, ptr(1), ptr(2), ptr(3), ptr(4), ptr(5), ptr(6), ptr(7), true};
    Fake() { trace.reserve(128); }
    int append(Step step) noexcept {
        trace.push_back(step);
        ++calls;
        return calls == fail_at || calls == second_fail_at ? -1700 - calls : 0;
    }
    KernelLaunchGateOps gate() {
        return {
            this,
            [](void *p, const KernelInvocationBinding &, void *caller, KernelLaunchAdmission *out) noexcept -> int {
                auto &f = *static_cast<Fake *>(p);
                if (!f.mutex.try_lock()) return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
                if (!f.ready || !f.frozen || f.poisoned || f.validation_error) {
                    f.mutex.unlock();
                    return f.validation_error ? f.validation_error : PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
                }
                ++f.acquisitions;
                f.candidate_caller = caller;
                *out = {f.handles, f.previous_caller};
                out->handles.consume_prepare_tail = f.pending_prepare;
                if (f.entered) {
                    f.entered->set_value();
                    f.release.wait();
                }
                return 0;
            },
            [](void *p, const KernelLaunchResult &result) noexcept {
                auto &f = *static_cast<Fake *>(p);
                ++f.finishes;
                if (result.status == 0) {
                    f.previous_caller = reinterpret_cast<uintptr_t>(f.candidate_caller);
                    f.pending_prepare = false;
                } else if (result.enqueue_started) {
                    f.poisoned = true;
                    f.runtime_error = result.status;
                    f.cleanup_error = result.cleanup_status;
                }
                f.mutex.unlock();
            },
            [](void *p, void *, bool *complete) noexcept {
                auto &f = *static_cast<Fake *>(p);
                ++f.queries;
                *complete = f.complete;
                return f.query_error;
            }
        };
    }
    KernelLaunchOps ops() {
        return {
            this,
            [](void *p, void *stream, void *event) noexcept {
                auto &f = *static_cast<Fake *>(p);
                const auto step = event == ptr(3) ? Step::PrepareWait :
                                  event == ptr(4) ? (stream == ptr(2) ? Step::AicoreWait : Step::AicpuWait) :
                                  event == ptr(6) ? Step::JoinAicpu :
                                                    Step::JoinAicore;
                return f.append(step);
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Step::Clear);
            },
            [](void *p, void *event, void *) noexcept {
                return static_cast<Fake *>(p)->append(
                    event == ptr(4) ? Step::Start :
                    event == ptr(5) ? Step::AicoreDone :
                    event == ptr(6) ? Step::AicpuDone :
                                      Step::SerialTail
                );
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Step::AicpuLaunch);
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Step::AicoreLaunch);
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Cancel);
            }
        };
    }
    void clear_trace() {
        trace.clear();
        calls = 0;
    }
};

const std::vector<Step> success{Step::PrepareWait,  Step::Clear,      Step::Start,      Step::AicoreWait,
                                Step::AicoreLaunch, Step::AicoreDone, Step::AicpuWait,  Step::AicpuLaunch,
                                Step::AicpuDone,    Step::JoinAicpu,  Step::JoinAicore, Step::SerialTail};
struct Fixture {
    Fake fake;
    std::array<uint64_t, 10> packet{};
    KernelInvocationBinding binding;
    void initialize() { binding = {reinterpret_cast<const uint8_t *>(packet.data()), sizeof(packet)}; }
    KernelLaunchResult launch(void *caller = ptr(100)) {
        return launch_bound_kernel(binding, caller, fake.gate(), fake.ops());
    }
};
}  // namespace kernel_binder_test
