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
#include "host_build_graph/host/graph_recorder_pool.h"

#include <functional>
#include <utility>

// Included for the two ops declarations below. The pool itself takes the context
// only as an opaque owner token, so nothing here dereferences it.
#include "host_build_graph/runtime_core.h"

namespace {
struct RecorderLifetime {
    std::mutex mutex;
    std::condition_variable cv;
    std::unique_ptr<GraphAsyncRecordingState> pool;
    size_t owners{0};
    bool closing{false};
};
RecorderLifetime &recorder_lifetime() {
    static RecorderLifetime lifetime;
    return lifetime;
}
}  // namespace

GraphAsyncRecordingState &graph_recorder_pool() {
    auto &lifetime = recorder_lifetime();
    std::scoped_lock lock(lifetime.mutex);
    if (!lifetime.pool) lifetime.pool = std::make_unique<GraphAsyncRecordingState>();
    return *lifetime.pool;
}

// RuntimeModule owns the lease, so callable and native-run owners sharing a
// module keep its workers alive until the last safe module release.
extern "C" int simpler_runtime_acquire_host_services() noexcept {
    try {
        auto &lifetime = recorder_lifetime();
        std::unique_lock<std::mutex> lock(lifetime.mutex);
        lifetime.cv.wait(lock, [&] {
            return !lifetime.closing;
        });
        ++lifetime.owners;
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void simpler_runtime_release_host_services() noexcept {
    auto &lifetime = recorder_lifetime();
    GraphAsyncRecordingState *pool;
    {
        std::scoped_lock lock(lifetime.mutex);
        if (lifetime.owners == 0) return;
        --lifetime.owners;
        if (lifetime.owners != 0) return;
        lifetime.closing = true;
        pool = lifetime.pool.get();
    }
    // Jobs can still use graph_record_wait_impl while draining, so the lifetime
    // mutex must not be held while waiting for their closures to be destroyed.
    if (pool) pool->shutdown();
    {
        std::scoped_lock lock(lifetime.mutex);
        lifetime.pool.reset();
        lifetime.closing = false;
    }
    lifetime.cv.notify_all();
}

bool graph_recorder_prewarm() { return graph_recorder_pool().prewarm(); }

// The only definitions of the two graph_record_* ops. The table that names them is
// host-only, so nothing on the device resolves either.
//
// `job` points to the caller's std::function. The pool moves the closure out of it
// whether or not it queues it -- start() takes the callable before it checks capacity --
// so the caller must treat it as spent on return. No ownership crosses the .so boundary
// either way: the caller's std::function destructs on its own side.
//
// `args` is the in-flight entry's own boundary, which outlives every job that reads it
// (the bind or explicit commit drains that owner's jobs first), so the pool forwards it by
// reference rather than copying it.
bool graph_record_start_impl(OrchContext *ctx, const GraphTaskArgs &args, void *job) {
    if (job == nullptr) return false;
    auto *record = static_cast<std::function<void(const GraphTaskArgs &)> *>(job);
    return graph_recorder_pool().start(args, std::move(*record), ctx);
}

void graph_record_wait_impl(OrchContext *ctx) { graph_recorder_pool().wait(ctx); }
