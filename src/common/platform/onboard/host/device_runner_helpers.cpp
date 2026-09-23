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
 * Onboard host common helpers — `KernelArgsHelper` implementation.
 *
 * Linked into both a2a3 and a5 `libhost_runtime.so`. The arch-specific
 * `KernelArgs` layout is brought in via `common/kernel_args.h` on the
 * include path (each arch CMake adds the right one).
 */

#include "device_runner_helpers.h"

#include <runtime/rt.h>

#include <cstring>

#include "acl/error_codes/rt_error_codes.h"
#include "common/unified_log.h"
#include "host/acl_error_log.h"

namespace {

int query_stream_nonblocking(rtStream_t stream, const char *name) {
    if (stream == nullptr) {
        LOG_ERROR("rtStreamQuery (%s) received a null stream", name);
        return SIMPLER_NATIVE_RUN_POLL_ERROR;
    }

    const rtError_t rc = rtStreamQuery(stream);
    if (rc == RT_ERROR_NONE) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    if (rc == ACL_ERROR_RT_STREAM_NOT_COMPLETE) return SIMPLER_NATIVE_RUN_POLL_NOT_READY;

    LOG_ERROR("rtStreamQuery (%s) failed: %d", name, static_cast<int>(rc));
    ACL_LOG_ERROR_DETAIL(rc);
    return SIMPLER_NATIVE_RUN_POLL_ERROR;
}

int query_stream_error(rtStream_t stream, const char *name) {
    if (stream == nullptr) {
        LOG_ERROR("rtStreamQuery (%s) received a null stream", name);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    const rtError_t rc = rtStreamQuery(stream);
    if (rc == RT_ERROR_NONE || rc == ACL_ERROR_RT_STREAM_NOT_COMPLETE) return 0;

    LOG_ERROR("rtStreamQuery (%s) reports a device error: %d", name, static_cast<int>(rc));
    ACL_LOG_ERROR_DETAIL(rc);
    return static_cast<int>(rc);
}

}  // namespace

int query_stream_pair_nonblocking(rtStream_t aicpu_stream, rtStream_t aicore_stream) {
    // Query both even when the first is pending. Besides making completion a
    // true pair fence, this preserves an error from either device queue.
    const int aicpu_rc = query_stream_nonblocking(aicpu_stream, "AICPU");
    const int aicore_rc = query_stream_nonblocking(aicore_stream, "AICore");
    if (aicpu_rc == SIMPLER_NATIVE_RUN_POLL_ERROR || aicore_rc == SIMPLER_NATIVE_RUN_POLL_ERROR) {
        return SIMPLER_NATIVE_RUN_POLL_ERROR;
    }
    if (aicpu_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE && aicore_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }
    return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int query_stream_pair_error(rtStream_t aicpu_stream, rtStream_t aicore_stream) {
    const int aicpu_rc = query_stream_error(aicpu_stream, "AICPU");
    if (aicpu_rc != 0) return aicpu_rc;
    return query_stream_error(aicore_stream, "AICore");
}

int KernelArgsHelper::prepare_runtime_args(
    const Runtime &host_runtime, MemoryAllocator &allocator, SlotPersistentArgs &slot
) {
    if (runtime_args_state_ == RuntimeArgsState::Prepared) return PTO_RUNTIME_ERR_INVALID_STATE;
    // Ahead of the allocation and of the snapshot, so a descriptor carrying
    // counts no consumer may believe fails the run here: nothing is allocated,
    // nothing is copied, and the caller's prepare returns the error rather than
    // publishing lengths the device would trust.
    const LaunchEntryArgsPlan plan = runtime_launch_entry_args_plan(host_runtime);
    if (!plan.counts_valid) {
        LOG_ERROR("prepare_runtime_args: this run's entry-argument counts are outside capacity");
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }
    release_run_view();
    allocator_ = &allocator;

    // Both runtime variants publish the descriptor at offset zero. Host-only
    // orchestration state and tensor leases remain outside this snapshot, and so
    // does any tail no host copy reaches: the block is sized to the whole
    // descriptor because the device addresses that range inside it, while only a
    // prefix is snapshotted and copied.
    const uint64_t runtime_extent = runtime_device_extent_size(host_runtime);
    // The length is a property of the runtime variant, which is fixed for a
    // runner, so a committed block always fits. A mismatch would mean the
    // block belongs to a different variant than the run being prepared.
    if (slot.runtime_args != nullptr && slot.runtime_bytes != runtime_extent) {
        LOG_ERROR(
            "runtime_args block is %llu bytes but this run needs %llu",
            static_cast<unsigned long long>(slot.runtime_bytes), static_cast<unsigned long long>(runtime_extent)
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (slot.runtime_args == nullptr) {
        void *runtime_dev = allocator_->alloc(runtime_extent);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        slot.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);
        slot.runtime_bytes = runtime_extent;
        slot.workers_initialized = false;
    }
    // A block whose handshake region has never been published still holds
    // whatever the allocator returned, so the first publication onto it carries
    // that region from the ctor-zeroed host copy. Every later publication stops
    // before it: the device owns those words from then on, and re-sending them
    // would overwrite a report or a reply.
    const bool initializing = !slot.workers_initialized;
    // The snapshot is always the descriptor route's length, which is the longer
    // of the two: the launch route publishes a prefix of it. Capturing the
    // longer one is what lets the route be chosen after this point without
    // reading the source again.
    const size_t publish_bytes =
        initializing ? runtime_device_initialized_prefix_size(host_runtime) : runtime_device_copy_size(host_runtime);
    runtime_image_.prepare(host_runtime, publish_bytes);
    initializing_slot_ = initializing ? &slot : nullptr;
    slot_ = &slot;
    plan_ = plan;
    if (plan_.supported) {
        // Grow-only: a slot that has served a wider run keeps the capacity.
        const size_t package_bytes = LAUNCH_ENVELOPE_HEADER_BYTES + plan_.payload_bytes;
        if (slot.launch_package.size() < package_bytes) slot.launch_package.resize(package_bytes);
    }
    args.runtime_args = slot.runtime_args;
    runtime_args_state_ = RuntimeArgsState::Prepared;
    return 0;
}

bool KernelArgsHelper::build_launch_package() {
    std::byte *package = slot_->launch_package.data();
    const size_t package_bytes = LAUNCH_ENVELOPE_HEADER_BYTES + plan_.payload_bytes;
    if (slot_->launch_package.size() < package_bytes) return false;

    std::byte *payload = package + LAUNCH_ENVELOPE_HEADER_BYTES;
    const size_t tensor_bytes = plan_.payload_bytes - static_cast<size_t>(plan_.scalar_count) * sizeof(uint64_t);
    // Out of the snapshot, not out of the caller's Runtime: the values a launch
    // carries and the ones its descriptor names are the same capture.
    if (!runtime_image_.read(plan_.tensor_offset, payload, tensor_bytes)) return false;
    if (!runtime_image_.read(
            plan_.scalar_offset, payload + tensor_bytes, static_cast<size_t>(plan_.scalar_count) * sizeof(uint64_t)
        )) {
        return false;
    }
    // Name the region this package now carries. The header bytes themselves are
    // copied in at the submission, because fields of this run are still being
    // armed between here and there.
    args.entry_args_offset = static_cast<uint32_t>(LAUNCH_ENVELOPE_HEADER_BYTES);
    args.entry_tensor_count = plan_.tensor_count;
    args.entry_scalar_count = plan_.scalar_count;
    args.entry_args_source = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    // Padding between the header and the region: the device reads none of it,
    // and zeroing keeps a predecessor's bytes out of this run's package.
    std::memset(package + sizeof(KernelArgs), 0, LAUNCH_ENVELOPE_HEADER_BYTES - sizeof(KernelArgs));
    launch_payload_ = package;
    launch_payload_bytes_ = package_bytes;
    return true;
}

void *KernelArgsHelper::launch_payload() {
    if (launch_payload_ == nullptr) return nullptr;
    if (launch_payload_ != static_cast<void *>(&args)) {
        // The header, as of this submission. Arming runs between publication
        // and here and writes into `args`, so copying it earlier would send the
        // device a run's stale bases. The entry region behind it is not
        // rewritten: it holds the prepare snapshot's values.
        std::memcpy(launch_payload_, &args, sizeof(KernelArgs));
    }
    return launch_payload_;
}

int KernelArgsHelper::publish_runtime_args(bool launch_route_permitted) {
    if (runtime_args_state_ != RuntimeArgsState::Prepared) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (args.runtime_args == nullptr || slot_ == nullptr) return PTO_RUNTIME_ERR_INTERNAL;

    // The first publication onto a block sends the whole initialized prefix,
    // entry storage included, so its values are in the descriptor whether or
    // not the launch could also have carried them. Routing it through the
    // launch as well would send them twice.
    const bool launch_route = launch_route_permitted && plan_.supported && initializing_slot_ == nullptr;
    const EntryArgsSource source = launch_route ? EntryArgsSource::LaunchEnvelope : EntryArgsSource::Descriptor;

    // Patch the route into the snapshot before it is published, so the byte the
    // device reads and the length this copy sends cannot disagree. The three
    // words sit ahead of the entry storage, inside every publication length.
    const uint32_t control[3] = {plan_.tensor_count, plan_.scalar_count, static_cast<uint32_t>(source)};
    if (plan_.supported && !runtime_image_.patch(plan_.control_offset, control, sizeof(control))) {
        LOG_ERROR("runtime metadata publication: the captured snapshot does not contain its control words");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // The launch package is built before the copy, because publication consumes
    // the snapshot both read from.
    if (launch_route && !build_launch_package()) {
        LOG_ERROR("runtime metadata publication: the captured snapshot does not contain its entry values");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (!launch_route) {
        args.entry_args_offset = 0;
        args.entry_tensor_count = 0;
        args.entry_scalar_count = 0;
        args.entry_args_source = static_cast<uint32_t>(EntryArgsSource::Descriptor);
        launch_payload_ = &args;
        launch_payload_bytes_ = sizeof(KernelArgs);
    }

    const size_t publish_bytes = launch_route ? plan_.descriptor_bytes_when_launched : runtime_image_.size();
    // The consumed snapshot is neither pending nor published during the copy.
    // Reentrant publish is rejected; copy failure leaves fresh prepare admissible.
    runtime_args_state_ = RuntimeArgsState::Empty;
    const int rc = runtime_image_.publish(
        [this](const void *source, size_t bytes) {
            return rtMemcpy(args.runtime_args, bytes, source, bytes, RT_MEMCPY_HOST_TO_DEVICE);
        },
        publish_bytes
    );
    if (rc != 0) {
        LOG_ERROR("runtime metadata publication failed: %d", rc);
        args.runtime_args = nullptr;
        // The block keeps whatever its handshake region held, so the next
        // prepare on it must send the initializing prefix again. The package is
        // withdrawn with it: a run whose descriptor never landed submits no
        // kernel, so nothing may read a payload naming that descriptor.
        initializing_slot_ = nullptr;
        launch_payload_ = nullptr;
        launch_payload_bytes_ = 0;
        return rc;
    }
    runtime_args_state_ = RuntimeArgsState::Published;
    // Committed here and nowhere earlier: this copy is what put a defined value
    // in that block's handshake region, so the fact is recorded by the call
    // that earned it rather than by a caller that has to remember the order.
    if (initializing_slot_ != nullptr) {
        initializing_slot_->workers_initialized = true;
        initializing_slot_ = nullptr;
    }
    return rc;
}

int release_slot_persistent_args(SlotPersistentArgs &slot, MemoryAllocator &allocator) {
    int first_error = 0;
    if (slot.runtime_args != nullptr) {
        const int rc = allocator.free(slot.runtime_args);
        if (rc != 0) {
            if (first_error == 0) first_error = rc;
        } else {
            // Whole-struct reset, so every fact recorded about the released
            // block goes with it and a successor allocation starts uninitialized.
            slot = SlotPersistentArgs{};
        }
    }
    return first_error;
}

void abandon_slot_persistent_args(SlotPersistentArgs &slot) { slot = SlotPersistentArgs{}; }
