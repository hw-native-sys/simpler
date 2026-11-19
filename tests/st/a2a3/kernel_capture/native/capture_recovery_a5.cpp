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

#include "capture_recovery.h"

#include <acl/acl.h>
#include <cstring>
#include "host_build_graph/runtime.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

namespace simpler::capture_test {
namespace {

// The original run owns this scheduler publication through the final replay.
struct CapturedResidentState {
    const void *run_control = nullptr;
    char *metadata_source = nullptr;
    size_t worker_contexts_offset = 0;
    int32_t worker_count = 0;
    uint64_t original_task_count = 0;
};

thread_local CapturedResidentState resident_state;

bool contains(const CapturedPublication &publication, uint64_t address, size_t bytes) {
    const auto base = reinterpret_cast<uintptr_t>(publication.destination);
    return publication.source != nullptr && address >= base && address - base <= publication.bytes &&
           bytes <= publication.bytes - (address - base);
}

CapturedResidentState
capture_resident_state(const CapturedLaunch &launch, const std::vector<CapturedPublication> &publications) {
    if (launch.publication_count > publications.size()) return {};
    const int32_t worker_count = launch.worker_count;
    if (launch.runtime_mode != SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING || launch.worker_context_base == 0 ||
        worker_count <= 0 || worker_count > PLATFORM_MAX_CORES) {
        return {};
    }

    const size_t context_bytes = static_cast<size_t>(worker_count) * sizeof(SchedulerWorkerContext);
    const CapturedPublication *metadata = nullptr;
    for (size_t index = 0; index < launch.publication_count; ++index) {
        const auto &publication = publications[index];
        if (contains(publication, launch.worker_context_base, context_bytes)) {
            if (metadata != nullptr) return {};
            metadata = &publication;
        }
    }
    if (metadata == nullptr) return {};
    const auto metadata_base = reinterpret_cast<uintptr_t>(metadata->destination);
    const size_t contexts_offset = launch.worker_context_base - metadata_base;
    auto *metadata_source = static_cast<char *>(const_cast<void *>(metadata->source));
    uint64_t run_control_offset = 0;
    uint64_t original_task_count = 0;
    for (int32_t worker = 0; worker < worker_count; ++worker) {
        SchedulerWorkerContext context{};
        std::memcpy(&context, metadata_source + contexts_offset + worker * sizeof(context), sizeof(context));
        if (worker == 0) {
            run_control_offset = context.run_control_offset;
            original_task_count = context.original_task_count;
        }
        if (context.worker_index != static_cast<uint64_t>(worker) ||
            context.runtime_worker_count != static_cast<uint64_t>(worker_count) ||
            context.scheduler_state_base_address != metadata_base ||
            context.worker_contexts_offset != contexts_offset || context.scheduler_state_bytes < metadata->bytes ||
            context.run_control_offset != run_control_offset || context.original_task_count != original_task_count) {
            return {};
        }
    }
    if (run_control_offset > contexts_offset || sizeof(SchedulerRunControl) > contexts_offset - run_control_offset) {
        return {};
    }
    SchedulerRunControl control{};
    std::memcpy(&control, metadata_source + run_control_offset, sizeof(control));
    if (control.scheduler_error != 0 || control.error_site != static_cast<uint64_t>(SchedulerErrorSite::UNKNOWN)) {
        return {};
    }
    return {
        reinterpret_cast<const void *>(metadata_base + run_control_offset),
        metadata_source,
        contexts_offset,
        worker_count,
        original_task_count,
    };
}

void set_captured_original_task_count(const CapturedResidentState &state, uint64_t count) {
    for (int32_t worker = 0; worker < state.worker_count; ++worker) {
        auto *field = state.metadata_source + state.worker_contexts_offset + worker * sizeof(SchedulerWorkerContext) +
                      offsetof(SchedulerWorkerContext, original_task_count);
        std::memcpy(field, &count, sizeof(count));
    }
}
static SchedulerRunControl read_resident_control(const CapturedResidentState &state) {
    SchedulerRunControl control{};
    check(
        aclrtMemcpy(&control, sizeof(control), state.run_control, sizeof(control), ACL_MEMCPY_DEVICE_TO_HOST),
        "read-resident-run-control"
    );
    std::printf(
        "resident-error=%llu site=%llu bootstrap-complete=%llu\n",
        static_cast<unsigned long long>(control.scheduler_error), static_cast<unsigned long long>(control.error_site),
        static_cast<unsigned long long>(control.bootstrap_complete)
    );
    return control;
}

static bool is_resident_bootstrap_failure(const SchedulerRunControl &control) {
    return control.bootstrap_complete == 0 &&
           control.scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS) &&
           control.error_site == static_cast<uint64_t>(SchedulerErrorSite::BOOTSTRAP_FAILED);
}

static bool is_resident_progress_timeout(const SchedulerRunControl &control) {
    const auto site = static_cast<SchedulerErrorSite>(control.error_site);
    // The supervisor and a peer share the same progress watchdog deadline.
    return control.bootstrap_complete != 0 &&
           control.scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT) &&
           (site == SchedulerErrorSite::EXECUTION_PROGRESS_TIMEOUT ||
            site == SchedulerErrorSite::AICPU_SHUTDOWN_BARRIER_TIMEOUT);
}
}  // namespace

size_t prepare_snapshot(DeviceContextHandle context, int scenario) {
    return needs_publications(scenario) ? get_scheduler_mode_publish_count(context) : 0;
}

void check_prepare(DeviceContextHandle context, int scenario, size_t before) {
    if (!needs_publications(scenario)) return;
    require(
        get_scheduler_mode_publish_count(context) == before + 1 &&
            get_last_scheduler_runtime_mode(context) == SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING,
        "prepare-published-resident-pending"
    );
}

bool needs_publications(int scenario) { return scenario == 12 || scenario == 13; }

void check_publications(int scenario, const CapturedSubmission &submission) {
    if (!needs_publications(scenario)) return;
    resident_state = {};
    require(submission.launches.size() == 1, "observed-single-core-launch");
    resident_state = capture_resident_state(submission.launches[0], submission.publications);
    require(resident_state.run_control != nullptr, "observed-resident-publication");
}

void release_boot_fault_source() {}
bool record_boot_fault_prefix(const unsigned char *, void *) { return false; }

void inject_boot_failure() { set_captured_original_task_count(resident_state, 0); }

void restore_boot_failure() {
    set_captured_original_task_count(resident_state, resident_state.original_task_count);
    check(0, "restore-injected-graph-view");
}

void check_boot_failure() {
    require(is_resident_bootstrap_failure(read_resident_control(resident_state)), "resident-bootstrap-failed");
}

void check_progress_timeout() {
    require(is_resident_progress_timeout(read_resident_control(resident_state)), "resident-progress-timeout");
}

}  // namespace simpler::capture_test
