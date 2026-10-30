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
#include <new>

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "common/unified_log.h"
#include "common/native_execution_control.h"
#include "host/acl_error_log.h"

extern "C" __attribute__((weak)) size_t native_capture_result_bytes_impl() { return 0; }

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

bool launch_route_permitted_by_capture(int query_rc, int capture_status) {
    // Anything but a successful "capturing nothing" routes through the
    // descriptor. An answer the query could not give is treated as capturing,
    // which keeps the route this run would have taken before the launch route
    // existed — a preserved behaviour, not a demonstration that a capture
    // would have been handled. It is not free either: that route publishes the
    // longer prefix.
    if (query_rc != ACL_SUCCESS) return false;
    return capture_status == static_cast<int>(ACL_MODEL_RI_CAPTURE_STATUS_NONE);
}

bool launch_entry_args_permitted(rtStream_t aicpu_stream) {
    if (aicpu_stream == nullptr) return false;
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model_ri = nullptr;
    const aclError rc = aclmdlRICaptureGetInfo(aicpu_stream, &status, &model_ri);
    if (rc != ACL_SUCCESS) {
        // Not a run failure, and not retried. Logged because a persistent one
        // silently costs every run the longer descriptor prefix.
        LOG_INFO("aclmdlRICaptureGetInfo unavailable (%d); entry args take the descriptor route", static_cast<int>(rc));
    }
    return launch_route_permitted_by_capture(static_cast<int>(rc), static_cast<int>(status));
}

int publish_for_launch(KernelArgsHelper &kernel_args, rtStream_t aicpu_stream, LaunchProgressSink &sink) {
    if (kernel_args.runtime_args_published()) return 0;
    const bool permitted = launch_entry_args_permitted(aicpu_stream);
    const int rc = kernel_args.publish_runtime_args(permitted, aicpu_stream, sink);
    if (rc != 0) {
        LOG_ERROR("publish_for_launch: launch metadata publication failed: %d", rc);
    }
    return rc;
}

std::shared_ptr<PinnedUploadStorage> PinnedUploadStorage::retain() {
    if (retained_source_ || data() == nullptr) return {};
    if (captured_) return captured_;
    if (data_ == nullptr || cleanup_debt_ != nullptr) return {};
    auto retained = std::make_shared<PinnedUploadStorage>();
    retained->retained_source_ = true;
    retained->data_ = std::exchange(data_, nullptr);
    retained->capacity_ = std::exchange(capacity_, 0);
    captured_ = std::move(retained);
    return captured_;
}

int PinnedUploadStorage::reserve(uint64_t bytes, bool allow_growth) {
    if (!allow_growth && (cleanup_debt_ != nullptr || bytes > capacity() || (captured_ && captured_.use_count() > 1)))
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    if (retained_source_) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (captured_) {
        if (captured_.use_count() == 1) {
            data_ = std::exchange(captured_->data_, nullptr);
            capacity_ = std::exchange(captured_->capacity_, 0);
            captured_.reset();
        } else {
            // The old source stays reachable if allocation fails. A new source
            // is required even when the captured allocation has enough capacity.
            void *replacement = nullptr;
            if (bytes != 0) {
                const int rc = rtMallocHost(&replacement, bytes, 0);
                if (rc != 0) return rc;
                if (replacement == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
            }
            data_ = replacement;
            capacity_ = bytes;
            captured_.reset();
            return 0;
        }
    }
    // Settle a rejected replacement before attempting another allocation.
    if (cleanup_debt_ != nullptr) {
        const int rc = rtFreeHost(cleanup_debt_);
        if (rc != 0) return rc;
        cleanup_debt_ = nullptr;
    }
    if (bytes <= capacity_) return 0;
    void *replacement = nullptr;
    const int alloc_rc = rtMallocHost(&replacement, bytes, 0);
    if (alloc_rc != 0) return alloc_rc;
    if (replacement == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (data_ != nullptr) {
        const int free_rc = rtFreeHost(data_);
        if (free_rc != 0) {
            if (rtFreeHost(replacement) != 0) cleanup_debt_ = replacement;
            return free_rc;
        }
    }
    data_ = replacement;
    capacity_ = bytes;
    return 0;
}

int PinnedUploadStorage::release() {
    if (captured_) {
        if (captured_.use_count() == 1) {
            const int rc = captured_->release();
            if (rc != 0) return rc;
        }
        captured_.reset();
    }
    int first_error = 0;
    if (cleanup_debt_ != nullptr) {
        first_error = rtFreeHost(cleanup_debt_);
        if (first_error == 0) cleanup_debt_ = nullptr;
    }
    if (data_ != nullptr) {
        const int rc = rtFreeHost(data_);
        if (rc != 0) {
            if (first_error == 0) first_error = rc;
        } else {
            data_ = nullptr;
            capacity_ = 0;
        }
    }
    return first_error;
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
    const bool captured_destination = slot.retained_runtime_args && slot.retained_runtime_args.use_count() > 1;
    const size_t capture_bytes = native_capture_result_bytes_impl();
    const size_t execution_offset =
        (runtime_extent + capture_bytes + alignof(NativeExecutionControl) - 1) & ~(alignof(NativeExecutionControl) - 1);
    const size_t control_bytes = execution_offset - runtime_extent + sizeof(NativeExecutionControl);
    if (slot.capture_preparing && (slot.runtime_args == nullptr || captured_destination || capture_bytes == 0))
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    if (slot.runtime_args == nullptr || captured_destination) {
        void *runtime_dev = allocator_->alloc(runtime_extent + control_bytes);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (control_bytes != 0) {
            const int rc =
                aclrtMemset(static_cast<char *>(runtime_dev) + runtime_extent, control_bytes, 0, control_bytes);
            if (rc != 0) {
                (void)allocator_->free(runtime_dev);
                return rc;
            }
        }
        if (captured_destination) {
            slot.retained_runtime_args->slot_owned = false;
            slot.retained_runtime_args.reset();
        }
        slot.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);
        slot.runtime_bytes = runtime_extent;
        slot.workers_initialized = false;
        // A fresh block inherits no claim about its contents. The two teardown
        // paths reset the whole struct, but a slot reaching this branch with a
        // freed pointer has not been through either, so the transition is made
        // here rather than inherited.
        slot.published_prefix_bytes = 0;
    }
    const uint64_t metadata_bytes = slot.staged_launch_metadata.size();
    if (metadata_bytes > UINT64_MAX - runtime_extent) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const uint64_t upload_bytes = runtime_extent + metadata_bytes;
    const int host_rc = slot.runtime_upload.reserve(upload_bytes, !slot.capture_preparing);
    if (host_rc != 0) return host_rc;
    if (!slot.capture_preparing) slot.eager_upload_bytes = upload_bytes;
    // A block whose handshake region has never been published still holds
    // whatever the allocator returned, so the first publication onto it carries
    // that region from the ctor-zeroed host copy. Every later publication stops
    // before it: the device owns those words from then on, and re-sending them
    // would overwrite a report or a reply.
    const bool initializing = slot.capture_preparing || !slot.workers_initialized;
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
    args.execution_control = reinterpret_cast<uint64_t>(slot.runtime_args) + execution_offset;
    // Captured with the snapshot: the length this run's descriptor names for its
    // Definition section, which the publication compares against what the bind
    // staged on the slot.
    graph_section_bytes_ = runtime_graph_section_bytes(host_runtime);
    launch_metadata_ = std::move(slot.staged_launch_metadata);
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

int KernelArgsHelper::prepare_kernel_invocation(const simpler::tmr::KernelCallableIdentity &identity) {
    if (runtime_args_state_ != RuntimeArgsState::Prepared || !plan_.supported || slot_ == nullptr)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (!simpler::tmr::valid_kernel_identity(identity) || identity.tensor_count != plan_.tensor_count ||
        identity.scalar_count != plan_.scalar_count)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    using EntryStorage = std::decay_t<decltype(std::declval<const Runtime &>().get_orch_args())>;
    EntryStorage storage{};
    if (!runtime_image_.read(plan_.tensor_offset, storage.tensors_, plan_.tensor_count * sizeof(storage.tensors_[0])) ||
        !runtime_image_.read(plan_.scalar_offset, storage.scalars_, plan_.scalar_count * sizeof(uint64_t)))
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    storage.tensor_count_ = static_cast<int32_t>(plan_.tensor_count);
    storage.scalar_count_ = static_cast<int32_t>(plan_.scalar_count);
    const size_t bytes = simpler::tmr::kernel_invocation_bytes(identity.tensor_count, identity.scalar_count);
    std::vector<std::byte> packet(bytes);
    if (!simpler::tmr::encode_kernel_invocation(
            storage, identity, {reinterpret_cast<uint64_t>(args.runtime_args), args.run_result_epoch}, packet.data(),
            packet.size()
        ))
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const size_t package_bytes = LAUNCH_ENVELOPE_HEADER_BYTES + bytes;
    if (slot_->launch_package.size() < package_bytes) slot_->launch_package.resize(package_bytes);
    kernel_invocation_ = std::move(packet);
    return 0;
}

bool KernelArgsHelper::build_kernel_launch_package() {
    const size_t bytes = LAUNCH_ENVELOPE_HEADER_BYTES + kernel_invocation_.size();
    if (slot_->launch_package.size() < bytes) return false;
    auto *package = slot_->launch_package.data();
    std::memset(package + sizeof(KernelArgs), 0, LAUNCH_ENVELOPE_HEADER_BYTES - sizeof(KernelArgs));
    std::memcpy(package + LAUNCH_ENVELOPE_HEADER_BYTES, kernel_invocation_.data(), kernel_invocation_.size());
    args.entry_args_offset = static_cast<uint32_t>(LAUNCH_ENVELOPE_HEADER_BYTES);
    args.entry_tensor_count = plan_.tensor_count;
    args.entry_scalar_count = plan_.scalar_count;
    args.entry_args_source = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    launch_payload_ = package;
    launch_payload_bytes_ = bytes;
    return true;
}

int stage_graph_section(SlotPersistentArgs &slot, const void *bytes, size_t length) {
    if (bytes == nullptr || length == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    // `argsSize` is a uint32 at the launch boundary, so the whole package —
    // header included — has to be representable there. Checked before the
    // addition that would wrap, and no smaller limit is imposed: a graph is not
    // refused for being large, only for being unrepresentable.
    if (length > UINT32_MAX - LAUNCH_ENVELOPE_HEADER_BYTES) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const size_t package_bytes = LAUNCH_ENVELOPE_HEADER_BYTES + length;
    // Grow-only, like the entry route's use of the same buffer: a slot that has
    // served a wider run keeps the capacity.
    if (slot.launch_package.size() < package_bytes) slot.launch_package.resize(package_bytes);
    std::memcpy(slot.launch_package.data() + LAUNCH_ENVELOPE_HEADER_BYTES, bytes, length);
    slot.graph_section_bytes = static_cast<uint32_t>(length);
    return 0;
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

int KernelArgsHelper::publish_runtime_args(bool launch_route_permitted, rtStream_t stream, LaunchProgressSink &sink) {
    if (runtime_args_state_ != RuntimeArgsState::Prepared) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (args.runtime_args == nullptr || slot_ == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (stream == nullptr || slot_->runtime_upload.data() == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;

    // The first publication onto a block sends the whole initialized prefix,
    // entry storage included, so its values are in the descriptor whether or
    // not the launch could also have carried them. Program calls keep those
    // initial values on the descriptor; kernel calls also carry their validated
    // packet because the registration and run binding belong to that format.
    const bool kernel_route = !kernel_invocation_.empty();
    if (kernel_route && !launch_route_permitted) return PTO_RUNTIME_ERR_UNSUPPORTED;
    const bool launch_route = launch_route_permitted && plan_.supported && initializing_slot_ == nullptr;
    const EntryArgsSource source = kernel_route ? EntryArgsSource::KernelInvocation :
                                   launch_route ? EntryArgsSource::LaunchEnvelope :
                                                  EntryArgsSource::Descriptor;

    // The Graph Definition section, decided before anything is written, so a
    // refusal here leaves no half-formed payload behind. Its length is consumed
    // from the slot whatever this publication then does, because a length
    // staged for this run must not be read by the next one on the slot — not
    // even if this run fails below. It rides the same package as the entry
    // region and starts at the same offset; the two are alternatives, because
    // the runtime that carries values carries no graph and the one that carries
    // a graph carries no values.
    const uint32_t staged_graph_bytes = slot_->graph_section_bytes;
    slot_->graph_section_bytes = 0;
    // Both sources have to agree. A run whose descriptor names no section gets
    // none, even if a failed prepare left one staged on this slot; a mismatch is
    // this run's own bind disagreeing with itself and fails the publication.
    const uint32_t graph_bytes = staged_graph_bytes == graph_section_bytes_ ? staged_graph_bytes : 0;
    if (staged_graph_bytes != graph_section_bytes_ && graph_section_bytes_ != 0) {
        LOG_ERROR(
            "runtime metadata publication: staged Graph section is %u bytes but the descriptor names %u",
            staged_graph_bytes, graph_section_bytes_
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (graph_bytes != 0 && (launch_route || kernel_route)) {
        LOG_ERROR("runtime metadata publication: a run carries both entry values and a Graph section");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const size_t graph_package_bytes = LAUNCH_ENVELOPE_HEADER_BYTES + graph_bytes;
    if (graph_bytes != 0 && slot_->launch_package.size() < graph_package_bytes) {
        LOG_ERROR("runtime metadata publication: the staged Graph section is no longer in the slot's package");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

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
    if ((kernel_route && !build_kernel_launch_package()) ||
        (!kernel_route && launch_route && !build_launch_package())) {
        LOG_ERROR("runtime metadata publication: the captured snapshot does not contain its entry values");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (!launch_route && !kernel_route) {
        args.entry_args_offset = 0;
        args.entry_tensor_count = 0;
        args.entry_scalar_count = 0;
        args.entry_args_source = static_cast<uint32_t>(EntryArgsSource::Descriptor);
        launch_payload_ = &args;
        launch_payload_bytes_ = sizeof(KernelArgs);
    }

    args.graph_section_offset = 0;
    args.graph_section_bytes = 0;
    args.graph_section_source = static_cast<uint32_t>(GraphSectionSource::None);
    if (graph_bytes != 0) {
        args.graph_section_offset = static_cast<uint32_t>(LAUNCH_ENVELOPE_HEADER_BYTES);
        args.graph_section_bytes = graph_bytes;
        args.graph_section_source = static_cast<uint32_t>(GraphSectionSource::LaunchEnvelope);
        // Padding between the header and the section: the device reads none of
        // it, and zeroing keeps a predecessor's bytes out of this package.
        std::memset(
            slot_->launch_package.data() + sizeof(KernelArgs), 0, LAUNCH_ENVELOPE_HEADER_BYTES - sizeof(KernelArgs)
        );
        launch_payload_ = slot_->launch_package.data();
        launch_payload_bytes_ = graph_package_bytes;
    }

    const size_t publish_bytes = launch_route ? plan_.descriptor_bytes_when_launched : runtime_image_.size();
    // A publication this block may already be holding: the launch route's own
    // prefix, onto an allocation a previous launch-route publication recorded.
    // Every other publication — the first onto a block, the descriptor
    // fallback, a runtime with no launch route — writes a range nothing
    // recorded, so it copies and records nothing.
    const bool prefix_cacheable = launch_route && plan_.supported && initializing_slot_ == nullptr &&
                                  publish_bytes == plan_.descriptor_bytes_when_launched &&
                                  publish_bytes <= LAUNCH_ROUTE_PREFIX_CACHE_BYTES;
    // The consumed snapshot is neither pending nor published during the copy.
    // Reentrant publish is rejected; copy failure leaves fresh prepare admissible.
    runtime_args_state_ = RuntimeArgsState::Empty;
    const int rc = runtime_image_.publish(
        [this, prefix_cacheable, stream, &sink](const void *source, size_t bytes) -> int {
            // Metadata and descriptor occupy disjoint ranges of the slot's
            // pinned source. Neither range changes until this run retires.
            const int metadata_rc = launch_metadata_.publish(
                static_cast<std::byte *>(slot_->runtime_upload.data()) + slot_->runtime_bytes,
                slot_->runtime_upload.capacity() - slot_->runtime_bytes,
                [stream, &sink](void *target, const void *metadata, size_t length) {
                    sink.mark_submitted();
                    return rtMemcpyAsync(target, length, metadata, length, RT_MEMCPY_HOST_TO_DEVICE, stream);
                }
            );
            if (metadata_rc != 0) return metadata_rc;
            // Compared and recorded here, inside the callback, because `source`
            // is the snapshot the publication consumes. The DMA reads the slot's
            // pinned copy, which survives this callback.
            if (prefix_cacheable && slot_->published_prefix_bytes == bytes &&
                std::memcmp(source, slot_->published_prefix.data(), bytes) == 0) {
                // The block holds these bytes already, and a skipped copy
                // leaves the run as published as a copy would have.
                return 0;
            }
            // Invalidated before the copy, not after a failure: a copy that
            // modifies part of the block and then fails leaves contents no
            // record may describe, and the next publication must not be able to
            // match its way out of re-sending them. The length changing is not
            // the mechanism — a warm failure leaves `workers_initialized` set,
            // so the next prepare sends this same length again.
            slot_->published_prefix_bytes = 0;
            if (bytes > slot_->runtime_upload.capacity()) return PTO_RUNTIME_ERR_INTERNAL;
            std::memcpy(slot_->runtime_upload.data(), source, bytes);
            // A failed enqueue does not prove the stream holds no DMA reader.
            sink.mark_submitted();
            const int copy_rc = rtMemcpyAsync(
                args.runtime_args, bytes, slot_->runtime_upload.data(), bytes, RT_MEMCPY_HOST_TO_DEVICE, stream
            );
            if (copy_rc != 0) return copy_rc;
            if (prefix_cacheable) {
                // After the copy that earned it, into fixed storage: no
                // allocation and nothing that can throw between a device write
                // and the state that records it.
                std::memcpy(slot_->published_prefix.data(), source, bytes);
                slot_->published_prefix_bytes = static_cast<uint32_t>(bytes);
            }
            return copy_rc;
        },
        publish_bytes
    );
    if (rc != 0) {
        LOG_ERROR("runtime metadata publication failed: %d", rc);
        args.runtime_args = nullptr;
        // A failed first publication leaves `workers_initialized` false, so the
        // next prepare on this block sends the initializing prefix again. A
        // failed warm one leaves it set and sends the ordinary length; what
        // covers that block is the invalidated prefix record, not the length.
        // The package is withdrawn either way: a run whose descriptor never
        // landed submits no kernel, so nothing may read a payload naming that
        // descriptor.
        initializing_slot_ = nullptr;
        launch_payload_ = nullptr;
        launch_payload_bytes_ = 0;
        return rc;
    }
    runtime_args_state_ = RuntimeArgsState::Published;
    // An accepted initializing copy precedes the entry fork on the AICPU
    // stream; both kernels wait behind it. Slot retirement protects the next
    // publication from reusing this block or its source while that copy is live.
    if (initializing_slot_ != nullptr) {
        initializing_slot_->workers_initialized = true;
        initializing_slot_ = nullptr;
    }
    return rc;
}

int release_slot_persistent_args(SlotPersistentArgs &slot, MemoryAllocator &allocator) {
    int first_error = 0;
    if (slot.retained_runtime_args && slot.retained_runtime_args.use_count() > 1) {
        slot.retained_runtime_args->slot_owned = false;
        slot.retained_runtime_args.reset();
        slot.runtime_args = nullptr;
    } else if (slot.runtime_args != nullptr) {
        const int rc = allocator.free(slot.runtime_args);
        if (rc != 0) first_error = rc;
        else {
            slot.runtime_args = nullptr;
            if (slot.retained_runtime_args) slot.retained_runtime_args->address = nullptr;
            slot.retained_runtime_args.reset();
        }
    }
    const int host_rc = slot.runtime_upload.release();
    if (first_error == 0) first_error = host_rc;
    if (first_error == 0) {
        slot.~SlotPersistentArgs();
        new (&slot) SlotPersistentArgs{};
    }
    return first_error;
}

// No release calls: the generation has ended or completion is unproven, so a
// queued DMA may still read the pinned source as well as name the device block.
// Freeing that source before a reader stops is a host-memory use-after-free;
// abandonment leaves it allocated for the rest of this process.
void abandon_slot_persistent_args(SlotPersistentArgs &slot) {
    slot.~SlotPersistentArgs();
    new (&slot) SlotPersistentArgs{};
}
