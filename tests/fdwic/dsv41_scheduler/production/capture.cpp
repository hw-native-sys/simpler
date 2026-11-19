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
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>

#include "orchestration_api.h"
#include "record.h"

namespace {
OrchContext *context;
const OrchOps *original;
OrchOps observed;
std::ofstream output;

void capture(const CoreTaskArgs &args, const TaskOutputTensors &result, dsv41::Header header) {
    if (args.predicate().op != PredicateOp::NONE) throw std::runtime_error("Capture does not support predicates");
    header.task_id = TaskId::to_uint64(result.task_id());
    header.tensor_count = args.tensor_count();
    header.scalar_count = args.scalar_count();
    header.dep_count = args.explicit_dep_count();
    header.block_num = args.launch_spec.block_num();
    header.sync_start = args.launch_spec.require_sync_start();
    header.early_resolve = args.allow_early_resolve();
    header.timing_slot = args.task_timing_slot();
    dsv41::write(output, header);
    uint32_t created = 0;
    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        dsv41::Operand operand{};
        operand.tag = static_cast<uint32_t>(args.tag(i));
        const Tensor &tensor = args.tag(i) == TensorArgType::OUTPUT ? result.get_ref(created++) : args.tensor(i).ref();
        operand.tensor.init_from(tensor);
        dsv41::write(output, operand);
    }
    std::vector<uint64_t> scalars(header.scalar_count);
    args.pack_scalars(scalars.data());
    for (auto scalar : scalars)
        dsv41::write(output, scalar);
    for (uint32_t i = 0; i < header.dep_count; ++i)
        dsv41::write(output, TaskId::to_uint64(args.explicit_dep(i)));
}

TaskOutputTensors submit(OrchContext *ctx, const MixedKernels &kernels, const CoreTaskArgs &args) {
    auto result = original->submit_task(ctx, kernels, args);
    dsv41::Header header{};
    header.operation = dsv41::Operation::SUBMIT;
    header.kernels[0] = kernels.aic_kernel_id;
    header.kernels[1] = kernels.aiv0_kernel_id;
    header.kernels[2] = kernels.aiv1_kernel_id;
    capture(args, result, header);
    return result;
}

TaskOutputTensors allocate(OrchContext *ctx, const CoreTaskArgs &args) {
    auto result = original->alloc_tensors(ctx, args);
    dsv41::Header header{};
    header.operation = dsv41::Operation::ALLOCATE;
    capture(args, result, header);
    return result;
}

TaskOutputTensors dummy(OrchContext *ctx, const CoreTaskArgs &args) {
    auto result = original->submit_dummy_task(ctx, args);
    dsv41::Header header{};
    header.operation = dsv41::Operation::DUMMY;
    capture(args, result, header);
    return result;
}

void begin(OrchContext *ctx, ScopeMode mode) {
    dsv41::Header header{};
    header.operation = dsv41::Operation::SCOPE_BEGIN;
    header.scope_mode = static_cast<int32_t>(mode);
    dsv41::write(output, header);
    original->scope_begin(ctx, mode);
}

void end(OrchContext *ctx) {
    original->scope_end(ctx);
    dsv41::Header header{};
    header.operation = dsv41::Operation::SCOPE_END;
    dsv41::write(output, header);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) void framework_bind_orch_context(OrchContext *ctx) {
    context = ctx;
    if (ctx == nullptr) return;
    original = ctx->ops;
    observed = *original;
    observed.submit_task = submit;
    observed.alloc_tensors = allocate;
    observed.submit_dummy_task = dummy;
    observed.scope_begin = begin;
    observed.scope_end = end;
    ctx->ops = &observed;
}

extern "C" __attribute__((visibility("hidden"))) OrchContext *framework_current_orch_context() { return context; }

void dsv41_capture_begin(const ChipTaskArgs &args) {
    const char *path = std::getenv("DSV41_RECORD_PATH");
    if (path == nullptr) throw std::runtime_error("DSV41_RECORD_PATH is required for original Host capture");
    output.open(path, std::ios::binary | std::ios::trunc);
    dsv41::write(output, dsv41::record_magic);
    dsv41::write(output, uint64_t{sizeof(dsv41::Operand)});
    dsv41::Header header{};
    header.operation = dsv41::Operation::BOUNDARY;
    header.tensor_count = args.tensor_count();
    header.scalar_count = args.scalar_count();
    dsv41::write(output, header);
    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        dsv41::Operand operand{};
        operand.tensor.init_from(args.tensor(i).ref());
        dsv41::write(output, operand);
    }
    std::vector<uint64_t> scalars(header.scalar_count);
    args.pack_scalars(scalars.data());
    for (auto scalar : scalars)
        dsv41::write(output, scalar);
}

void dsv41_capture_end() {
    output.close();
    context->ops = original;
}

namespace {
void *original_library() {
    static void *handle = [] {
        const char *path = std::getenv("DSV41_ORIGINAL_ORCH");
        if (path == nullptr) throw std::runtime_error("DSV41_ORIGINAL_ORCH is required");
        void *result = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (result == nullptr) throw std::runtime_error(dlerror());
        return result;
    }();
    return handle;
}

template <class Function>
Function original_symbol(const char *name) {
    void *address = dlsym(original_library(), name);
    if (address == nullptr) throw std::runtime_error(dlerror());
    return reinterpret_cast<Function>(address);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &args) {
    return original_symbol<OrchestrationConfig (*)(const ChipTaskArgs &)>("aicpu_orchestration_config")(args);
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    dsv41_capture_begin(args);
    original_symbol<void (*)(OrchContext *)>("framework_bind_orch_context")(context);
    original_symbol<void (*)(const ChipTaskArgs &)>("aicpu_orchestration_entry")(args);
    dsv41_capture_end();
}
