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
#include "host_build_graph/runtime_status.h"
#include "scheduler/wire_types.h"
#include "task_image_layout.h"

namespace simpler::capture_test {
namespace {

using percounter::PercounterControl;

struct CapturedCalendarState {
    std::byte *sync_width_source = nullptr;
    const void *cluster_count = nullptr;
    const void *error_code = nullptr;
    uint64_t original_sync_width = 0;
};

thread_local CapturedCalendarState calendar_state;

CapturedCalendarState capture_calendar_state(const CapturedSubmission &submission) {
    if (submission.launches.size() != 1 || submission.cpu_launches.size() != 1) return {};
    const auto &launch = submission.launches.front();
    const auto &cpu = submission.cpu_launches.front();
    if (launch.runtime_mode != SCHEDULER_RUNTIME_MODE_CALENDAR || launch.worker_context_base == 0 ||
        launch.publication_count > submission.publications.size() || !cpu.inline_runtime || !cpu.working_address ||
        cpu.image_bytes < TASK_IMAGE_ERROR_CODE_OFFSET + sizeof(int32_t)) {
        return {};
    }

    const auto address = launch.worker_context_base;
    const CapturedPublication *metadata = nullptr;
    for (size_t index = 0; index < launch.publication_count; ++index) {
        const auto &publication = submission.publications[index];
        const auto base = reinterpret_cast<uintptr_t>(publication.destination);
        if (publication.source == nullptr || address < base || address - base > publication.bytes ||
            publication.bytes - (address - base) < sizeof(PercounterControl))
            continue;
        if (metadata != nullptr) return {};
        metadata = &publication;
    }
    if (metadata == nullptr) return {};
    const auto offset = address - reinterpret_cast<uintptr_t>(metadata->destination);
    auto *source = static_cast<std::byte *>(const_cast<void *>(metadata->source));
    auto *sync_width_source = source + offset + offsetof(PercounterControl, max_sync_blocks);
    uint64_t original_sync_width = 0;
    std::memcpy(&original_sync_width, sync_width_source, sizeof(original_sync_width));
    return {
        sync_width_source,
        reinterpret_cast<const void *>(address + offsetof(PercounterControl, cluster_count)),
        reinterpret_cast<const void *>(cpu.working_address + TASK_IMAGE_ERROR_CODE_OFFSET),
        original_sync_width,
    };
}

template <typename T>
T read_device_field(const void *address, const char *stage) {
    T value{};
    check(aclrtMemcpy(&value, sizeof(value), address, sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST), stage);
    return value;
}

void require_calendar_error(int32_t expected_error, bool bootstrap_complete, const char *stage) {
    const auto error = read_device_field<int32_t>(calendar_state.error_code, "read-calendar-error");
    const auto clusters = read_device_field<uint64_t>(calendar_state.cluster_count, "read-calendar-clusters");
    std::printf("calendar-error=%d clusters=%llu\n", error, static_cast<unsigned long long>(clusters));
    require(error == expected_error && (clusters != 0) == bootstrap_complete, stage);
}

}  // namespace

size_t prepare_snapshot(DeviceContextHandle context, int scenario) {
    return needs_publications(scenario) ? get_scheduler_mode_publish_count(context) : 0;
}

void check_prepare(DeviceContextHandle context, int scenario, size_t before) {
    if (!needs_publications(scenario)) return;
    require(
        get_scheduler_mode_publish_count(context) == before + 1 &&
            get_last_scheduler_runtime_mode(context) == SCHEDULER_RUNTIME_MODE_CALENDAR,
        "prepare-published-calendar"
    );
}

bool needs_publications(int scenario) { return scenario == 12 || scenario == 13; }

void check_publications(int scenario, const CapturedSubmission &submission) {
    if (!needs_publications(scenario)) return;
    calendar_state = capture_calendar_state(submission);
    require(calendar_state.sync_width_source != nullptr, "observed-calendar-publication");
}

void release_boot_fault_source() {}
bool record_boot_fault_prefix(const unsigned char *, void *) { return false; }

void inject_boot_failure() {
    const uint64_t invalid_width = PLATFORM_MAX_CORES + 1;
    std::memcpy(calendar_state.sync_width_source, &invalid_width, sizeof(invalid_width));
}

void restore_boot_failure() {
    const auto width = calendar_state.original_sync_width;
    std::memcpy(calendar_state.sync_width_source, &width, sizeof(width));
    check(0, "restore-injected-sync-width");
}

void check_boot_failure() { require_calendar_error(SIMPLER_ERROR_INVALID_ARGS, false, "calendar-bootstrap-failed"); }

void check_progress_timeout() {
    require_calendar_error(SIMPLER_ERROR_SCHEDULER_TIMEOUT, true, "calendar-progress-timeout");
}

}  // namespace simpler::capture_test
