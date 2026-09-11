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

#include <vector>
#include <utility>

#include "host_build_graph/kernel_graph_wire.h"

class KernelExecutionState;
struct RuntimeContext;
struct RuntimeArenaLayout;

namespace hbg {

struct GraphBuild;

struct GraphInvocationIdentity {
    int32_t callable_id{-1};
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    uint64_t callable_generation{0};
    uint64_t callable_hash{0};
    uint64_t argument_hash{0};
    uint64_t function_hash{0};
};

// Window-based scheduler reservations, independent of the program-mode maximum.
// TensorMap is Host-only in this runtime and is not part of the device image.
int make_kernel_graph_layout(uint64_t task_window, RuntimeArenaLayout &out);

// Owns the entire canonical packet. No pointer into GraphBuild, Host staging or
// a previous packet survives construction. Concurrent readers may share it.
class GraphLaunchTemplate {
public:
    GraphLaunchTemplate() = default;
    GraphLaunchTemplate(const GraphLaunchTemplate &) = default;
    GraphLaunchTemplate &operator=(const GraphLaunchTemplate &) = default;
    GraphLaunchTemplate(GraphLaunchTemplate &&other) noexcept :
        storage_(std::move(other.storage_)),
        size_(std::exchange(other.size_, 0)) {}
    GraphLaunchTemplate &operator=(GraphLaunchTemplate &&other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    const void *data() const { return storage_.data(); }
    size_t size() const { return size_; }

private:
    friend int make_graph_launch_template(
        const GraphBuild &, const RuntimeContext &, const KernelExecutionState &, int, uint64_t, uint64_t,
        const GraphInvocationIdentity &, GraphLaunchTemplate &
    );
    std::vector<uint64_t> storage_;
    size_t size_{0};
};

// Source build/context must share a live exclusive workspace lease. Resources
// were prepared using make_kernel_graph_layout(build.task_capacity) and frozen.
// All destination accesses here are address arithmetic, never device memory IO.
int make_graph_launch_template(
    const GraphBuild &build, const RuntimeContext &runtime, const KernelExecutionState &context, int device_id,
    uint64_t slot_generation, uint64_t runtime_binary_id, const GraphInvocationIdentity &identity,
    GraphLaunchTemplate &out
);

struct GraphHostArgs {
    std::vector<uint64_t> storage;
    size_t bytes{0};
    uint32_t address_offset{0};
    uint32_t data_offset{0};
};

// Fresh writable launch copy: CANN may patch it without changing the template.
// Offsets include the common invocation envelope, not just the HBG sub-packet.
int make_graph_host_args(const GraphLaunchTemplate &source, GraphHostArgs &out);

struct GraphHostLaunchOps {
    void *context{nullptr};
    // Consumes host arguments synchronously into task-owned storage, enqueues
    // the task, then returns. It must not retain a pointer to these Host bytes.
    int (*launch)(void *, GraphHostArgs &){nullptr};
};

int submit_graph_template(const GraphLaunchTemplate &source, const GraphHostLaunchOps &ops);

}  // namespace hbg
