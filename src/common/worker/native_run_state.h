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

#ifndef SRC_COMMON_WORKER_NATIVE_RUN_STATE_H_
#define SRC_COMMON_WORKER_NATIVE_RUN_STATE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

#include "call_config.h"
#include "native_run_launch_signal.h"
#include "runtime.h"

/** Internal phase of the caller-owned opaque native-run storage. */
enum class NativeRunPhase : uint8_t {
    Prepared,
    Launching,
    Running,
    Complete,
};

/**
 * Caller-owned state for one progressable native lifecycle. Runtime,
 * CallConfig, and resource selection are per-run. Runner-owned execution,
 * diagnostics, and timing remain exclusive from launch through finalize.
 */
template <typename Runner>
struct NativeRunState {
    static constexpr uint64_t kMagic = UINT64_C(0x534d504c52554e31);  // "SMPLRUN1"

    NativeRunState(Runner *runner_in, const CallConfig &config_in, uint64_t trace_hid_in) :
        runner(runner_in),
        config(config_in),
        trace_hid(trace_hid_in) {}

    ~NativeRunState() {
        if (executor.joinable()) executor.join();
        if (host_thread_state != nullptr) {
            runner->destroy_native_run_thread_state(host_thread_state);
        }
    }

    /** Move prepare-thread state into the executor before runner->run(). */
    void adopt_host_thread_state() noexcept {
        void *snapshot = host_thread_state;
        host_thread_state = nullptr;
        if (snapshot != nullptr) runner->adopt_native_run_thread_state(snapshot);
    }

    void publish_execution_complete(int rc) noexcept {
        execution_rc.store(rc, std::memory_order_relaxed);
        execution_done.store(true, std::memory_order_release);
        execution_cv.notify_all();
    }

    void wait_for_execution_complete() {
        std::unique_lock<std::mutex> lock(execution_mu);
        execution_cv.wait(lock, [this]() {
            return execution_done.load(std::memory_order_acquire);
        });
    }

    uint64_t magic{kMagic};
    Runner *runner{nullptr};
    CallConfig config{};
    Runtime runtime{};
    uint64_t trace_hid{0};
    unsigned trace_inv{0};
    long long trace_start_ns{0};
    std::thread executor{};
    std::atomic<int> execution_rc{-1};
    std::atomic<bool> execution_done{false};
    std::mutex execution_mu{};
    std::condition_variable execution_cv{};
    std::atomic<NativeRunPhase> phase{NativeRunPhase::Prepared};
    NativeRunLaunchSignal launch_signal{};
    void *host_thread_state{nullptr};
    uint64_t run_id{0};
    uint64_t generation{0};
    uint64_t dispatch_id{0};
    uint64_t run_epoch{0};
    uint32_t pipeline_slot{0};
    uint32_t arena_bank{0};
    char trace_attrs[192]{};
    bool runner_resources_owned{false};
    bool runner_reserved{false};
    bool runner_claimed{false};
};

/** End object lifetime, then mark the caller-owned storage reusable. */
template <typename Runner>
void destroy_native_run_state(NativeRunState<Runner> *state) {
    void *storage = state;
    state->~NativeRunState<Runner>();
    constexpr uint64_t kEmpty = 0;
    std::memcpy(storage, &kEmpty, sizeof(kEmpty));
}

#endif  // SRC_COMMON_WORKER_NATIVE_RUN_STATE_H_
