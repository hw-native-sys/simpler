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
#include "fixture.h"
#include "host_build_graph/host/orch_ops.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/dep_gen_host_graph.h"
#include "host_build_graph/host/graph_submit.h"

namespace native_admission_test {
namespace {
OrchContext *current_orchestrator = nullptr;
OrchContext *other_orchestrator = nullptr;
void record_entry(const ChipTaskArgs &args, OrchContext *&binding) {
    ++callbacks;
    auto &orch = *binding->orchestrator;
    uint32_t shape[] = {16};
    GraphTaskArgs boundary;
    boundary.add_input(args.tensor(0).ref());
    auto graph = orch.graph_begin(0x521, boundary);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary));
    CoreTaskArgs task;
    task.add_input(graph.params->tensor(0).ref());
    TensorCreateInfo output(shape, 1, DataType::UINT8);
    task.add_output(output);
    ASSERT_TRUE(orch.submit_dummy_task(task).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();
    binding = nullptr;
    if (auto hook = std::exchange(callback_hook, {})) hook();
    if (refuse_after_callback) control().ceiling = control().bytes;
}
void entry(const ChipTaskArgs &args) { record_entry(args, current_orchestrator); }
void bind(OrchContext *ctx) { current_orchestrator = ctx; }
void other_entry(const ChipTaskArgs &args) { record_entry(args, other_orchestrator); }
void other_bind(OrchContext *ctx) { other_orchestrator = ctx; }
void record_kernel_entry(const ChipTaskArgs &args, bool in_graph, uint32_t declaration = 0) {
    ++callbacks;
    auto &orch = *current_orchestrator->orchestrator;
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    CoreTaskArgs task;
    const auto declare = [&](TaskId id) {
        ASSERT_TRUE(id.is_valid());
        if (declaration == 0) return;
        const bool payload = declaration >= 3;
        const auto cache =
            declaration == 2 ? AsyncUseCacheContract::UNQUALIFIED : AsyncUseCacheContract::PUBLISHED_CLEAN_LINES;
        ASSERT_TRUE(declare_async_use(
            &orch, id,
            {payload ? AsyncUseDeclarationKind::COUNTER_PAYLOAD : AsyncUseDeclarationKind::COUNTER_ONLY, 0,
             payload ? 0u : UINT32_MAX, UINT32_MAX, UINT32_MAX, 4, cache}
        ));
        if (declaration == 4)
            ASSERT_TRUE(declare_async_use(
                &orch, id,
                {AsyncUseDeclarationKind::CONSUMER_CLEAN_READ, UINT32_MAX, 0, UINT32_MAX, UINT32_MAX, 0, cache}
            ));
    };
    if (in_graph) {
        GraphTaskArgs boundary;
        boundary.add_input(args.tensor(0).ref());
        const auto graph = orch.graph_begin(0x522, boundary);
        ASSERT_TRUE(graph.recording);
        ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary));
        task.add_input(graph.params->tensor(0).ref());
        declare(orch.submit_task(kernels, task).task_id());
        ASSERT_TRUE(orch.graph_end());
        orch.graph_commit();
    } else {
        task.add_input(args.tensor(0).ref());
        declare(orch.submit_task(kernels, task).task_id());
    }
    current_orchestrator = nullptr;
}
void ordinary_kernel_entry(const ChipTaskArgs &args) { record_kernel_entry(args, false); }
void graph_kernel_entry(const ChipTaskArgs &args) { record_kernel_entry(args, true); }
template <bool Graph, uint32_t Declaration>
void counter_entry(const ChipTaskArgs &args) {
    record_kernel_entry(args, Graph, Declaration);
}
void empty_entry(const ChipTaskArgs &) {
    ++callbacks;
    current_orchestrator = nullptr;
}
struct EntryPoints {
    void (*entry)(const ChipTaskArgs &);
    void (*bind)(OrchContext *);
    void (*finish)();
} entry_points{entry, bind, nullptr}, other_entry_points{other_entry, other_bind, nullptr},
    ordinary_kernel_entry_points{ordinary_kernel_entry, bind, nullptr},
    graph_kernel_entry_points{graph_kernel_entry, bind, nullptr}, empty_points{empty_entry, bind, nullptr};
}  // namespace

void *primary_entry_points() { return &entry_points; }
void *alternate_entry_points() { return &other_entry_points; }
void *empty_entry_points() { return &empty_points; }
void *kernel_entry_points(bool in_graph) {
    return in_graph ? &graph_kernel_entry_points : &ordinary_kernel_entry_points;
}
void *counter_entry_points(bool in_graph, uint32_t declaration) {
    static EntryPoints entries[2][4]{
        {{counter_entry<false, 1>, bind, nullptr},
         {counter_entry<false, 2>, bind, nullptr},
         {counter_entry<false, 3>, bind, nullptr},
         {counter_entry<false, 4>, bind, nullptr}},
        {{counter_entry<true, 1>, bind, nullptr},
         {counter_entry<true, 2>, bind, nullptr},
         {counter_entry<true, 3>, bind, nullptr},
         {counter_entry<true, 4>, bind, nullptr}}
    };
    return declaration >= 1 && declaration <= 4 ? &entries[in_graph ? 1 : 0][declaration - 1] : nullptr;
}
void set_dep_gen_capture(bool enabled) { dep_gen_host_graph_set_enabled(enabled); }
uint32_t arena_bank(uint32_t slot) { return slot; }

}  // namespace native_admission_test
