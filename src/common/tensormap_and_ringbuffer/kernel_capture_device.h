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

#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#include "aicpu/cache_maintenance.h"
#include "common/kernel_args.h"
#include "common/device_run_result.h"
#include "kernel_capture_gate.h"
#include "kernel_invocation.h"
#include "runtime.h"
#include "worker/runtime_c_api.h"

namespace simpler::tmr {

// Registration happens on the cold control path. Model replay only looks up a
// retained record; it cannot create one from the pointer carried by a packet.
class KernelCaptureRegistry {
public:
    template <typename Lookup, typename Freshen>
    int update(const KernelCaptureRegistrationArgs &args, Lookup lookup, Freshen freshen) {
        if (args.version != KERNEL_CAPTURE_REGISTRATION_VERSION || args.reserved != 0 || args.diagnostics_reserved != 0)
            return -1;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto action = static_cast<CaptureRegistrationAction>(args.action);
        if (action == CaptureRegistrationAction::RevokeCallable) {
            for (auto &[id, record] : records_) {
                (void)id;
                if (record.args.callable_id == args.callable_id &&
                    record.args.callable_generation == args.callable_generation)
                    record.revoked = true;
            }
            return 0;
        }
        auto found = records_.find(args.reference.capture_generation);
        if (action == CaptureRegistrationAction::Revoke) {
            if (found == records_.end()) return 0;
            if (!same_reference(found->second.args.reference, args.reference)) return -1;
            records_.erase(found);
            return 0;
        }
        const auto identity = lookup(args.callable_id);
        if ((action != CaptureRegistrationAction::Register && action != CaptureRegistrationAction::RegisterOnce) ||
            found != records_.end() || !valid_kernel_identity(identity) ||
            identity.generation != args.callable_generation || identity.tensor_count != args.tensor_count ||
            identity.scalar_count != args.scalar_count || args.reference.control_address == 0 ||
            args.reference.control_address % alignof(KernelCaptureControl) != 0 ||
            args.reference.capture_generation == 0 || args.reference.context_generation == 0 ||
            args.runtime_address == 0 || args.run_result_address == 0 || args.binding_epoch == 0 ||
            args.launched_threads == 0 || args.worker_count == 0 || args.worker_count > RUNTIME_MAX_WORKER ||
            args.packet_bytes !=
                LAUNCH_ENVELOPE_HEADER_BYTES + kernel_invocation_bytes(args.tensor_count, args.scalar_count) ||
            args.header_bytes != sizeof(KernelArgs))
            return -1;
        KernelArgs trusted_header{};
        std::memcpy(&trusted_header, args.launch_header, sizeof(trusted_header));
        if (!valid_header(args, trusted_header)) return -1;
        for (size_t i = args.header_bytes; i < sizeof(args.launch_header); ++i)
            if (args.launch_header[i] != 0) return -1;
        // The platform closes host-DMA visibility after identity validation and
        // before the first control load. Revocation and replay never do this.
        freshen(reinterpret_cast<const void *>(args.reference.control_address), sizeof(KernelCaptureControl));
        const auto &control = *reinterpret_cast<const KernelCaptureControl *>(args.reference.control_address);
        if (control.reserved != 0) return -1;
        for (uint32_t reserved : control.reserved_tail)
            if (reserved != 0) return -1;
        const uint64_t initial_epoch = args.binding_epoch - 1;
        if (!matches_control(args, control) || control.replay_epoch != initial_epoch ||
            control.completed_epoch != initial_epoch || control.finished_threads != 0 || control.arrived_threads != 0 ||
            control.first_error != 0 || control.decision != static_cast<uint32_t>(CaptureDecision::Idle))
            return -1;
        records_.emplace(args.reference.capture_generation, Record{args, false});
        return 0;
    }

    template <typename Lookup>
    int admit(void *packet, Lookup lookup) {
        if (packet == nullptr) return -1;
        const auto &header = *static_cast<const KernelArgs *>(packet);
        KernelCaptureReference reference{};
        if (!read_reference(packet, header, &reference)) return -1;
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = records_.find(reference.capture_generation);
        if (found == records_.end() || !same_reference(found->second.args.reference, reference)) return -1;
        const Record &record = found->second;
        auto &control = *reinterpret_cast<KernelCaptureControl *>(reference.control_address);
        if (!matches_control(record.args, control)) return -1;
        const auto identity = lookup(record.args.callable_id);
        EntryArgsStorage checked{};
        const bool valid = !record.revoked && same_header(record.args, header) &&
                           consume_kernel_invocation(
                               static_cast<unsigned char *>(packet) + LAUNCH_ENVELOPE_HEADER_BYTES,
                               record.args.packet_bytes - LAUNCH_ENVELOPE_HEADER_BYTES, identity,
                               {record.args.runtime_address, record.args.binding_epoch, record.args.reference}, &checked
                           ) &&
                           identity.callable_id == record.args.callable_id &&
                           identity.generation == record.args.callable_generation;
        const int refusal = record.revoked || identity.generation != record.args.callable_generation ?
                                PTO_RUNTIME_ERR_INVALID_STATE :
                                PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (!capture_admit(control, valid, refusal)) {
            // No new fork may overwrite reports still owned by this group.
            // Its admitted late participants must retain their entry decision.
            capture_latch_error(control, PTO_RUNTIME_ERR_INVALID_STATE);
            cache_flush_range(&control, sizeof(control));
            return -1;
        }
        auto *result = reinterpret_cast<DeviceRunResultRegion *>(record.args.run_result_address);
        __atomic_store_n(&result->published, 0, __ATOMIC_RELEASE);
        cache_flush_range(&result->published, sizeof(result->published));
        auto *runtime = reinterpret_cast<Runtime *>(record.args.runtime_address);
        // The predecessor's whole-operator join precedes admission. Publish
        // zero markers before either stream can start this execution's cores.
        for (uint32_t i = 0; i < record.args.worker_count; ++i)
            runtime->dev.workers[i].aicore_done = 0;
        cache_flush_range(runtime->dev.workers, record.args.worker_count * sizeof(Handshake));
        cache_flush_range(&control, sizeof(control));
        return 0;
    }

    int enter(void *packet, KernelArgs *decoded, uint64_t *replay_sequence) {
        KernelCaptureReference reference{};
        if (decoded == nullptr || replay_sequence == nullptr || !read_reference(packet, *decoded, &reference))
            return -1;
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = records_.find(reference.capture_generation);
        if (found == records_.end() || !same_reference(found->second.args.reference, reference)) return -1;
        const auto &control = *reinterpret_cast<const KernelCaptureControl *>(reference.control_address);
        if (!matches_control(found->second.args, control) || control.replay_epoch == 0) return -1;
        const auto decision = static_cast<CaptureDecision>(__atomic_load_n(&control.decision, __ATOMIC_ACQUIRE));
        if (decision != CaptureDecision::Admitted && decision != CaptureDecision::Rejected) return -1;
        if (decision == CaptureDecision::Admitted && !same_header(found->second.args, *decoded)) return -1;
        // A recognized refusal cannot use the rejected frame's addresses or
        // diagnostic fields while performing the ordinary handshake/retirement.
        std::memcpy(decoded, found->second.args.launch_header, sizeof(*decoded));
        *replay_sequence = __atomic_load_n(&control.replay_epoch, __ATOMIC_ACQUIRE);
        if (decision == CaptureDecision::Admitted) return 1;
        return 0;
    }

    void finish(void *packet, uint64_t epoch, int status) {
        KernelCaptureReference reference{};
        const auto &header = *static_cast<const KernelArgs *>(packet);
        if (!read_reference(packet, header, &reference)) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = records_.find(reference.capture_generation);
        if (found == records_.end() || !same_reference(found->second.args.reference, reference)) return;
        auto &control = *reinterpret_cast<KernelCaptureControl *>(reference.control_address);
        const auto finalize = [&record = found->second](KernelCaptureControl &state, uint64_t) {
            if (state.decision != static_cast<uint32_t>(CaptureDecision::Admitted)) return;
            const auto *region = reinterpret_cast<const DeviceRunResultRegion *>(record.args.run_result_address);
            cache_invalidate_range(region, sizeof(*region));
            const auto terminal = device_run_result_terminal(*region, record.args.binding_epoch);
            // The mainline cleanup owner has returned before the last native
            // participant reaches here. Its validated terminal record decides
            // the runtime status; arrival order must not select a different
            // thread's provisional return or hide missing terminal publication.
            if (terminal.state == DeviceRunTerminalState::Failed) {
                __atomic_store_n(&state.first_error, terminal.code, __ATOMIC_RELEASE);
            } else if (terminal.state != DeviceRunTerminalState::Succeeded || state.first_error != 0) {
                __atomic_store_n(&state.first_error, PTO_RUNTIME_ERR_INTERNAL, __ATOMIC_RELEASE);
            }
        };
        if (!capture_finish(control, epoch, status, finalize))
            capture_latch_error(control, PTO_RUNTIME_ERR_INVALID_STATE);
        cache_flush_range(&control, sizeof(control));
        // The last callback no longer needs this single-use registration.
        // Host still retains the control until its own whole-operator boundary
        // completes, including the Core return. Graph references keep the record.
        if (found->second.args.action == static_cast<uint32_t>(CaptureRegistrationAction::RegisterOnce) &&
            control.completed_epoch == epoch && control.finished_threads == control.launched_threads)
            records_.erase(found);
    }

private:
    struct Record {
        KernelCaptureRegistrationArgs args;
        bool revoked;
    };
    static bool same_reference(const KernelCaptureReference &a, const KernelCaptureReference &b) {
        return a.control_address == b.control_address && a.capture_generation == b.capture_generation &&
               a.context_generation == b.context_generation;
    }
    static bool read_reference(const void *packet, const KernelArgs &header, KernelCaptureReference *out) {
        static_assert(sizeof(KernelArgs) <= LAUNCH_ENVELOPE_HEADER_BYTES);
        if (packet == nullptr || header.entry_args_source != static_cast<uint32_t>(EntryArgsSource::KernelInvocation) ||
            header.entry_args_offset != LAUNCH_ENVELOPE_HEADER_BYTES)
            return false;
        constexpr size_t reference_offset = LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(SimplerKernelInvocationHeader) +
                                            offsetof(EagerKernelBinding, reference);
        std::memcpy(out, static_cast<const unsigned char *>(packet) + reference_offset, sizeof(*out));
        return true;
    }
    static bool matches_control(const KernelCaptureRegistrationArgs &args, const KernelCaptureControl &control) {
        return control.context_generation == args.reference.context_generation &&
               control.capture_generation == args.reference.capture_generation &&
               control.runtime_address == args.runtime_address && control.binding_epoch == args.binding_epoch &&
               control.callable_generation == args.callable_generation && control.callable_id == args.callable_id &&
               control.tensor_count == args.tensor_count && control.scalar_count == args.scalar_count &&
               control.launched_threads == args.launched_threads && control.worker_count == args.worker_count;
    }
    static bool valid_header(const KernelCaptureRegistrationArgs &args, const KernelArgs &header) {
        return header.entry_args_source == static_cast<uint32_t>(EntryArgsSource::KernelInvocation) &&
               header.entry_args_offset == LAUNCH_ENVELOPE_HEADER_BYTES &&
               reinterpret_cast<uint64_t>(header.runtime_args) == args.runtime_address &&
               header.run_result_data_base == args.run_result_address &&
               header.run_result_epoch == args.binding_epoch && header.entry_tensor_count == args.tensor_count &&
               header.entry_scalar_count == args.scalar_count &&
               header.enable_profiling_flag == args.enable_profiling_flag &&
               header.device_wall_data_base == args.device_wall_data_base && header.graph_section_offset == 0 &&
               header.graph_section_bytes == 0 &&
               header.graph_section_source == static_cast<uint32_t>(GraphSectionSource::None);
    }
    static auto common_header_fields(const KernelArgs &header) {
        return std::tie(
            header.runtime_args, header.regs, header.dump_data_base, header.chip_swimlane_data_base,
            header.chip_swimlane_run_terminal_bank, header.pmu_data_base, header.dep_gen_data_base,
            header.chip_swimlane_aicore_rotation_table, header.scope_stats_data_base, header.device_wall_data_base,
            header.run_result_data_base, header.run_result_epoch, header.args_dump_run_terminal_bank,
            header.pmu_run_terminal_bank, header.dep_gen_run_terminal_bank, header.scope_stats_run_terminal_bank,
            header.enable_profiling_flag, header.entry_args_offset, header.entry_tensor_count,
            header.entry_scalar_count, header.entry_args_source, header.graph_section_offset,
            header.graph_section_bytes, header.graph_section_source
        );
    }
    template <typename Header>
    static auto platform_header_fields(const Header &header)
        -> decltype(std::tie(header.ffts_base_addr, header.pmu_reg_addrs)) {
        return std::tie(header.ffts_base_addr, header.pmu_reg_addrs);
    }
    template <typename Header>
    static auto platform_header_fields(const Header &header)
        -> decltype(std::tie(header.force_simt_anchor, header.aicore_pmu_ring_addrs)) {
        return std::tie(header.force_simt_anchor, header.aicore_pmu_ring_addrs);
    }
    static bool same_header(const KernelCaptureRegistrationArgs &args, const KernelArgs &header) {
        KernelArgs trusted{};
        std::memcpy(&trusted, args.launch_header, sizeof(trusted));
        // Value comparison excludes native padding from launch identity.
        return common_header_fields(trusted) == common_header_fields(header) &&
               platform_header_fields(trusted) == platform_header_fields(header);
    }
    std::mutex mutex_;
    std::map<uint64_t, Record> records_;
};

}  // namespace simpler::tmr
