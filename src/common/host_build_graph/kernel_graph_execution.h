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

#include "host_build_graph/kernel_graph_execution_validation.h"

struct RuntimeContext;

namespace hbg {

enum class GraphExecutionStatus : uint32_t {
    Ok,
    Rejected,
    InvalidImage,
    Busy,
    Exhausted,
    CopyFailed,
    NotReady,
    Quarantined,
    Poisoned
};

enum class GraphExecutionRetirement : uint32_t { Completed, ControlledFailure, FatalFailure };

// Independent error channels from the enclosing execution and cleanup owner.
struct GraphExecutionCompletion {
    GraphExecutionRetirement outcome;
    int runtime_status{0};
    int unexpected_teardown_status{0};
};

struct GraphExecutionPreparation {
    RuntimeContext *runtime{nullptr};
    uint64_t generation{0};
    uint64_t sm_bytes{0};
    uint32_t total_tasks{0};
};

// Optional synchronous memory backend. A failed operation may have written a
// prefix, but no dispatch is published. Null operations select device defaults.
struct GraphExecutionPreparationOps {
    void *context{nullptr};
    bool (*copy)(void *, void *, const void *, size_t){nullptr};
    bool (*zero)(void *, void *, size_t){nullptr};
    bool (*flush)(void *, const void *, size_t){nullptr};
};

// Prepare a fresh graph execution from an immutable packet in a registered device
// slot. Called by the AICPU leader after Start and before opening the dispatch window.
// Per-execution state is reinitialized; no previous execution is resumed.
// The owner excludes prior execution/readers, serializes this entire operation
// with close/registration, and retains immutable packet/callable storage.
// The caller supplies a binding validated under that same lease; this function
// checks graph image semantics before writing any destination. It does not parse
// packet framing or repeat callable/resource identity validation.
// Ready remains occupied until retirement; Failed remains quarantined until
// controlled cleanup or terminal poisoning. Every admitted retry reinitializes all regions.
// out changes only on success. This function neither launches nor synchronizes.
GraphExecutionStatus prepare_graph_execution(
    const ValidatedGraphExecutionBinding &binding, GraphExecutionPreparation &out,
    const GraphExecutionPreparationOps &ops = {}
) noexcept;

// Called after all participants have skipped dispatch on failure, shut down,
// passed completion gates and deinitialized, and all AICore/readers are quiescent.
// Only a known controlled fault with no runtime/teardown error permits retry.
// Native errors require FatalFailure and Host context poisoning; no reset/free.
// The owner serializes this with preparation, registration, poisoning and close.
GraphExecutionStatus retire_graph_execution(
    GraphSlotRegistry *registry, uint64_t attempt, const GraphExecutionCompletion &completion
) noexcept;

// The kernel entry distributes the successful leader's generation through its
// per-invocation barrier. Peers must never sample an unversioned old Ready flag.
// The execution lease prevents another preparation until these readers have exited.
GraphExecutionStatus acquire_graph_execution(
    const GraphSlotRegistry *registry, uint64_t generation, GraphExecutionPreparation &out
) noexcept;

}  // namespace hbg
