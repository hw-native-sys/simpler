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
/**
 * Runtime Class - Implementation
 *
 * Device execution and handshake control.
 * Task graph construction is handled by RuntimeContext.
 */

#include "runtime.h"

#include "runtime_types.h"
#include "shared_memory.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // NOTE: host_api is initialized in InitRuntime() (host-only code)
    // because the CApi functions don't exist when compiled for device.

    // Initialize the device-copied descriptor (`dev`).
    memset(dev.workers, 0, sizeof(dev.workers));
    dev.worker_count = 0;
    dev.aicpu_thread_num = 1;
    memset(dev.aicpu_allowed_cpus, 0, sizeof(dev.aicpu_allowed_cpus));
    dev.aicpu_allowed_cpu_count = 0;
    dev.aicpu_launch_count = 0;
    dev.serial_orch_sched = false;
    dev.gm_sm_ptr_ = nullptr;
    dev.orch_args_storage_.clear();
    dev.prebuilt_arena_base_ = nullptr;
    dev.prebuilt_runtime_offset_ = 0;
    dev.active_callable_id_ = -1;
    dev.entry_tensor_count_ = 0;
    dev.entry_scalar_count_ = 0;
    dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::Descriptor);
    dev.callable_table_addr_ = 0;
    dev.callable_table_len_ = 0;
}

// =============================================================================
// Device orchestration
// =============================================================================

void *Runtime::get_gm_sm_ptr() const { return dev.gm_sm_ptr_; }
const simpler::tmr::EntryArgsStorage &Runtime::get_orch_args() const { return dev.orch_args_storage_; }
void Runtime::set_gm_sm_ptr(void *p) { dev.gm_sm_ptr_ = p; }
// The one place a boundary ChipTensor becomes this runtime's Tensor. Called from
// the host, before any orchestration runs, so nothing inside the runtime — on the
// host or on the AICPU — ever holds the boundary form.
void Runtime::set_orch_args(const ChipStorageTaskArgs &args) {
    dev.orch_args_storage_.clear();
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        dev.orch_args_storage_.add_tensor(simpler::tmr::Tensor::from_boundary(args.tensor(i)));
    }
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        dev.orch_args_storage_.add_scalar(args.scalar(i));
    }
}

EntryArgsSource Runtime::get_entry_args_source() const { return static_cast<EntryArgsSource>(dev.entry_args_source_); }
uint32_t Runtime::get_entry_tensor_count() const { return dev.entry_tensor_count_; }
uint32_t Runtime::get_entry_scalar_count() const { return dev.entry_scalar_count_; }

// The counts are checked against the descriptor's own before the payload is
// read, so a launch package and a descriptor that disagree are rejected rather
// than reconciled in favour of whichever arrived. Capacity is the storage's own
// gate, inside load_from_wire.
bool Runtime::adopt_entry_args_from_launch(const void *payload, uint32_t tensor_count, uint32_t scalar_count) {
    if (tensor_count != dev.entry_tensor_count_ || scalar_count != dev.entry_scalar_count_) return false;
    return dev.orch_args_storage_.load_from_wire(payload, tensor_count, scalar_count);
}

void Runtime::set_prebuilt_arena(void *arena_base, size_t runtime_off) {
    dev.prebuilt_arena_base_ = arena_base;
    dev.prebuilt_runtime_offset_ = runtime_off;
}
void *Runtime::get_prebuilt_arena_base() const { return dev.prebuilt_arena_base_; }
size_t Runtime::get_prebuilt_runtime_offset() const { return dev.prebuilt_runtime_offset_; }

void Runtime::set_active_callable_id(int32_t callable_id) { dev.active_callable_id_ = callable_id; }

int32_t Runtime::get_active_callable_id() const { return dev.active_callable_id_; }

uint64_t Runtime::get_function_bin_addr(int func_id) const {
    if (callable_table_host_ == nullptr || func_id < 0 || static_cast<uint32_t>(func_id) >= dev.callable_table_len_) {
        return 0;
    }
    return callable_table_host_[func_id];
}

void Runtime::set_callable_tables(
    const uint64_t *host_view, uint64_t object_table_addr, uint64_t entry_table_addr, uint32_t len
) {
    if (host_view == nullptr || object_table_addr == 0 || len == 0) {
        clear_callable_tables();
        return;
    }
    callable_table_host_ = host_view;
    callable_entry_table_addr_ = entry_table_addr;
    dev.callable_table_addr_ = object_table_addr;
    dev.callable_table_len_ = len;
}

void Runtime::clear_callable_tables() {
    callable_table_host_ = nullptr;
    callable_entry_table_addr_ = 0;
    dev.callable_table_addr_ = 0;
    dev.callable_table_len_ = 0;
}

uint64_t Runtime::callable_entry_table_addr() const { return callable_entry_table_addr_; }

uint32_t Runtime::callable_table_len() const { return dev.callable_table_len_; }

// trb's device image is just the `dev` descriptor (the rest of Runtime is
// host-only). A steady-state run re-publishes it up to the handshake region:
// the AICore writes its report there and the AICPU the task pointer it answers
// with, so no host value is consumed. A5 has no post-close gate array, so the
// initialized prefix and the device extent coincide; all three entry points
// exist so the shared host paths need no per-runtime branch.
//
// The length stops inside the entry args, after the tensor slots this run
// filled, which is why it is the one length that depends on the run rather than
// only on the runtime: `orch_args_storage_` is the last uploaded member and its
// tensor array is the last thing in it. At capacity this is exactly
// offsetof(workers); with no tensors it is 1,280 bytes on both arches. Slots the
// length leaves out keep whatever an earlier run of the same allocation wrote,
// so every consumer reads through the counts — see
// ChipTaskArgs::create_from_entry_storage.
size_t runtime_device_copy_size(const Runtime &rt) {
    return offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + rt.get_orch_args().used_prefix_bytes();
}

// The routing facts, read while `rt` is still this run's. Deliberately a
// snapshot of values rather than a pointer into the descriptor: the launch side
// consumes it after a successor's prepare may already have refilled the source.
LaunchEntryArgsVerdict classify_launch_entry_args(
    const Runtime &rt, uint32_t source, uint32_t offset, uint32_t tensor_count, uint32_t scalar_count
) {
    if (source != static_cast<uint32_t>(EntryArgsSource::Descriptor) &&
        source != static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)) {
        return LaunchEntryArgsVerdict::UndefinedSource;
    }
    if (source != static_cast<uint32_t>(rt.get_entry_args_source())) return LaunchEntryArgsVerdict::SourceMismatch;
    if (source == static_cast<uint32_t>(EntryArgsSource::Descriptor)) return LaunchEntryArgsVerdict::Descriptor;
    if (offset != LAUNCH_ENVELOPE_HEADER_BYTES) return LaunchEntryArgsVerdict::UnexpectedOffset;
    if (tensor_count > CHIP_MAX_TENSOR_ARGS || scalar_count > CHIP_MAX_SCALAR_ARGS) {
        return LaunchEntryArgsVerdict::CountsPastCapacity;
    }
    if (tensor_count != rt.get_entry_tensor_count() || scalar_count != rt.get_entry_scalar_count()) {
        return LaunchEntryArgsVerdict::CountsMismatch;
    }
    return LaunchEntryArgsVerdict::Adopt;
}

LaunchEntryArgsPlan runtime_launch_entry_args_plan(const Runtime &rt) {
    const simpler::tmr::EntryArgsStorage &entry = rt.get_orch_args();
    const int32_t tensors = entry.tensor_count();
    const int32_t scalars = entry.scalar_count();
    LaunchEntryArgsPlan plan;
    plan.supported = true;
    // A count neither builder can produce — ChipStorageTaskArgs and
    // EntryArgsStorage both throw at capacity — so reaching this means the
    // descriptor was written past those. Reported as invalid rather than as an
    // absent route, because the two call for opposite handling: one publishes
    // through the descriptor, the other must not publish at all.
    if (tensors < 0 || static_cast<size_t>(tensors) > CHIP_MAX_TENSOR_ARGS || scalars < 0 ||
        static_cast<size_t>(scalars) > CHIP_MAX_SCALAR_ARGS) {
        plan.counts_valid = false;
        return plan;
    }
    plan.tensor_count = static_cast<uint32_t>(tensors);
    plan.scalar_count = static_cast<uint32_t>(scalars);
    plan.control_offset = offsetof(DeviceRuntimeLaunchDesc, entry_tensor_count_);
    plan.tensor_offset =
        offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + offsetof(simpler::tmr::EntryArgsStorage, tensors_);
    plan.scalar_offset =
        offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + offsetof(simpler::tmr::EntryArgsStorage, scalars_);
    plan.payload_bytes = simpler::tmr::EntryArgsStorage::wire_bytes(plan.tensor_count, plan.scalar_count);
    plan.descriptor_bytes_when_launched = offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_);
    return plan;
}

// The first publication onto an allocation adds the handshake region, so it
// starts from the ctor-zeroed host copy rather than from whatever rtMalloc
// left. This runtime has no host-uninitialized tail, so that reaches the end.
size_t runtime_device_initialized_prefix_size(const Runtime &) { return sizeof(DeviceRuntimeLaunchDesc); }

size_t runtime_device_extent_size(const Runtime &) { return sizeof(DeviceRuntimeLaunchDesc); }
