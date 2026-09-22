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

/**
 * One fixed, fully populated run, used to pin the serialized artifact.
 *
 * Every value here is a literal, so the same fixture serializes to the same
 * bytes on every machine and every build. It reaches SCHED_PHASES and populates
 * each stream the writer can emit — AICore tasks, AICPU tasks, sched phases with
 * a non-zero device drop count, orchestrator phases, host phases and one JSON
 * extension — so that no emitted field and neither shared-memory
 * header read is left uncovered.
 *
 * Only the public collector API and direct shared-memory writes are used, so
 * the same fixture compiles against any revision of the collector.
 */

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"

namespace chip_swimlane_fixture {

inline constexpr uint64_t kEpoch = 7700;
inline constexpr int kNumAicore = 2;
inline constexpr int kAicpuThreads = 2;
inline constexpr uint32_t kSchedDropped = 5;

inline void *fixture_alloc(size_t size) { return std::calloc(1, size); }

inline int fixture_free(void *ptr) {
    std::free(ptr);
    return 0;
}

// Heap buffers standing in for phase pool buffers. `on_buffer_collected` reads
// only `host_buffer_ptr`, so the pool itself is not involved.
struct PhaseBuffers {
    ChipSwimlaneAicpuSchedPhaseBuffer *sched{nullptr};
    ChipSwimlaneAicpuOrchPhaseBuffer *orch{nullptr};
    ChipSwimlaneAicpuTaskBuffer *perf{nullptr};

    PhaseBuffers() {
        sched = static_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(std::calloc(1, sizeof(*sched)));
        orch = static_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(std::calloc(1, sizeof(*orch)));
        perf = static_cast<ChipSwimlaneAicpuTaskBuffer *>(std::calloc(1, sizeof(*perf)));
    }
    ~PhaseBuffers() {
        std::free(sched);
        std::free(orch);
        std::free(perf);
    }
    PhaseBuffers(const PhaseBuffers &) = delete;
    PhaseBuffers &operator=(const PhaseBuffers &) = delete;
};

inline void deliver(ChipSwimlaneCollector &collector, ProfBufferType type, uint32_t index, void *buffer) {
    ReadyBufferInfo info{};
    info.type = type;
    info.index = index;
    info.dev_buffer_ptr = buffer;
    info.host_buffer_ptr = buffer;
    info.buffer_seq = 0;
    collector.on_buffer_collected(info, /*collector_shard=*/0);
}

/**
 * Drive the fixture into `collector`, leaving it at the point production
 * reaches just before the export call.
 *
 * `buffers` must outlive the call; its storage is read during delivery only.
 */
inline void populate(ChipSwimlaneCollector &collector, PhaseBuffers &buffers, const std::string &output_prefix) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    collector.begin_run(output_prefix, ChipSwimlaneLevel::SCHED_PHASES);

    set_platform_run_result(/*region_base=*/0, kEpoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(
        reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(/*bank_index=*/0, kEpoch))
    );
    chip_swimlane_aicpu_init(kNumAicore);

    // AICore records through the real dispatch/flush path.
    for (int core = 0; core < kNumAicore; core++) {
        auto *ac_state = get_aicore_buffer_state(shm, core);
        auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(ac_state->head.current_buf_ptr);
        for (int i = 0; i < 2; i++) {
            chip_swimlane_aicpu_on_aicore_dispatch(core, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
            buf->records[i].start_time = 1000 + static_cast<uint64_t>(core * 10 + i);
            buf->records[i].end_time = 1500 + static_cast<uint64_t>(core * 10 + i);
            buf->records[i].reg_task_id = static_cast<uint32_t>(i + 1);
            buf->records[i].task_token_raw = static_cast<uint64_t>(core * 100 + i);
            buf->records[i].receive_to_start_cycles = static_cast<uint32_t>(7 + i);
        }
    }
    const int cores[] = {0, 1};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/kNumAicore);
    auto *header = get_chip_swimlane_header(shm);
    for (uint32_t i = 0; i < header->queue_tails[0]; i++) {
        const ReadyQueueEntry &entry = header->queues[0][i];
        if (entry.kind != ChipSwimlaneBufferKind::AicoreTask) continue;
        deliver(collector, ProfBufferType::AICORE_TASK, entry.core_index, reinterpret_cast<void *>(entry.buffer_ptr));
    }

    // AICPU task records.
    buffers.perf->run_epoch = kEpoch;
    buffers.perf->local_seq = 0;
    buffers.perf->count = 2;
    for (uint32_t i = 0; i < 2; i++) {
        buffers.perf->records[i].dispatch_time = 900 + i;
        buffers.perf->records[i].finish_time = 1600 + i;
        buffers.perf->records[i].reg_task_id = i + 1;
    }
    deliver(collector, ProfBufferType::AICPU_TASK, /*index=*/0, buffers.perf);

    // Sched-phase records, plus the device drop count the writer reads from the
    // header for this thread.
    buffers.sched->run_epoch = kEpoch;
    buffers.sched->local_seq = 0;
    buffers.sched->count = 2;
    for (uint32_t i = 0; i < 2; i++) {
        auto &record = buffers.sched->records[i];
        record.start_time = 2000 + i;
        record.end_time = 2100 + i;
        record.loop_iter = i;
        record.kind = i == 0 ? ChipSwimlaneSchedPhaseKind::Dispatch : ChipSwimlaneSchedPhaseKind::DummyTask;
        record.tasks_processed = i + 1;
        if (i == 0) {
            record.phase_data.dispatch.pop_hit = 3;
            record.phase_data.dispatch.pop_miss = 4;
        } else {
            record.phase_data.task_id.raw = 0x5150;
        }
        for (int q = 0; q < CHIP_SWIMLANE_NUM_QUEUE_SHAPES; q++) {
            record.shared_depth_at_start[q] = static_cast<int16_t>(q + 1);
            record.shared_depth_at_end[q] = static_cast<int16_t>(q + 2);
        }
    }
    deliver(collector, ProfBufferType::AICPU_SCHED_PHASE, /*index=*/0, buffers.sched);
    get_sched_phase_buffer_state(shm, 0)->head.dropped_record_count = kSchedDropped;

    // Orchestrator-phase records, plus the lane count the writer reads from the
    // header. One lane, so the value is distinguishable from the vector size.
    buffers.orch->run_epoch = kEpoch;
    buffers.orch->local_seq = 0;
    buffers.orch->count = 2;
    for (uint32_t i = 0; i < 2; i++) {
        buffers.orch->records[i].start_time = 3000 + i;
        buffers.orch->records[i].end_time = 3100 + i;
        buffers.orch->records[i].task_id = 0x6100 + i;
        buffers.orch->records[i].submit_idx = i;
    }
    deliver(collector, ProfBufferType::AICPU_ORCH_PHASE, /*index=*/0, buffers.orch);
    header->num_orch_phase_threads = 1;

    // Per-core mappings: core_to_thread comes out of the header at drain.
    header->num_phase_cores = kNumAicore;
    header->core_to_thread[0] = 0;
    header->core_to_thread[1] = 1;
    header->num_sched_phase_threads = 1;

    const CoreType core_types[kNumAicore] = {CoreType::AIC, CoreType::AIV};
    collector.set_core_types(core_types, kNumAicore);

    collector.read_phase_header_metadata();
    collector.reconcile_counters();
    collector.report_run_terminal_snapshot(/*bank_index=*/0, kEpoch);

    collector.set_json_extension(ChipSwimlaneExtensionSection::AicpuLifecycleRecords, "[[11, 12]]");

    std::vector<HostPhaseRecord> submits(1);
    submits[0].index = 1;
    submits[0].payload = 4242;
    submits[0].start_ns = 600;
    submits[0].end_ns = 700;
    std::vector<HostPhaseRecord> uploads(1);
    uploads[0].kind = 0;
    uploads[0].payload = 128;
    uploads[0].start_ns = 550;
    uploads[0].end_ns = 590;
    collector.set_host_phase_records(
        submits, uploads, /*submitted_tasks=*/1, /*total_records=*/3,
        /*dropped_records=*/0
    );
}

/** The key the writer emits only when the host can name its clock domain. */
inline constexpr const char *kHostClockDomainIdKey = "\"host_clock_domain_id\"";

/**
 * Whether this host can name its clock domain, by the writer's own rule.
 *
 * Mirrors `linux_boot_clock_domain_id()` in the collector, which is file-local
 * and cannot be called from here: the boot id must be readable, non-empty and
 * alphanumeric-or-dash. On Linux it normally succeeds; on macOS `/proc` does
 * not exist and it fails, which is why the writer legitimately omits the field
 * there.
 *
 * The test uses this to assert **presence or absence**, never to paper over
 * either — a writer that stopped emitting the field on a host that can name its
 * domain, or started emitting it on one that cannot, fails.
 */
inline bool host_clock_domain_id_available() {
    std::ifstream boot_id_file("/proc/sys/kernel/random/boot_id");
    std::string boot_id;
    if (!(boot_id_file >> boot_id) || boot_id.empty()) return false;
    for (unsigned char ch : boot_id) {
        if (!std::isalnum(ch) && ch != '-') return false;
    }
    return true;
}

/**
 * Drop the `host_clock_domain_id` line, including its newline and indent, so a
 * golden captured where the host could name its domain can be compared on a
 * host that cannot.
 *
 * Only the one line is removed, and only when the provider is genuinely absent;
 * every other column is still compared exactly.
 */
inline std::string without_host_clock_domain_id(std::string text) {
    const size_t at = text.find(kHostClockDomainIdKey);
    if (at == std::string::npos) return text;
    const size_t line_start = text.rfind('\n', at);
    const size_t line_end = text.find('\n', at);
    if (line_start == std::string::npos || line_end == std::string::npos) return text;
    text.erase(line_start, line_end - line_start);
    return text;
}

/**
 * Replace the two values that cannot be fixed by the fixture: the host's boot
 * clock identity (machine-specific) and the runtime name (build-specific).
 *
 * Only the identity's *value* is substituted. Whether the field is emitted at
 * all is a real property of the host and is asserted separately.
 */
inline std::string normalize(std::string text, const std::string &runtime_name) {
    const std::string domain_key = "\"host_clock_domain_id\": \"";
    size_t at = text.find(domain_key);
    if (at != std::string::npos) {
        const size_t value = at + domain_key.size();
        const size_t end = text.find('"', value);
        if (end != std::string::npos) text.replace(value, end - value, "<HOST_CLOCK_DOMAIN_ID>");
    }
    const std::string runtime_key = "\"runtime\": \"" + runtime_name + "\"";
    at = text.find(runtime_key);
    if (at != std::string::npos) text.replace(at, runtime_key.size(), "\"runtime\": \"<RUNTIME>\"");
    return text;
}

}  // namespace chip_swimlane_fixture
