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

#include "host/pmu_collector.h"

#include <cstring>
#include <fstream>

PmuCollector::~PmuCollector() {
    if (setup_header_dev_ != nullptr) {
        // Not called finalize() — best-effort free to avoid leaking device memory
        // on abnormal exits. Errors swallowed because we are already unwinding.
        (void)finalize();
    }
}

int PmuCollector::initialize(
    int num_cores, PmuEventType event_type, uint64_t *kernel_args_pmu_data_base, PmuAllocCallback alloc_cb,
    PmuFreeCallback free_cb, PmuCopyToDeviceCallback copy_to_dev_cb, PmuCopyFromDeviceCallback copy_from_dev_cb
) {
    if (num_cores <= 0 || num_cores > PLATFORM_MAX_CORES || kernel_args_pmu_data_base == nullptr ||
        alloc_cb == nullptr || free_cb == nullptr || copy_to_dev_cb == nullptr || copy_from_dev_cb == nullptr) {
        LOG_ERROR("PmuCollector::initialize invalid arguments (num_cores=%d)", num_cores);
        return -1;
    }

    num_cores_ = num_cores;
    event_type_ = event_type;
    alloc_cb_ = alloc_cb;
    free_cb_ = free_cb;
    copy_to_dev_cb_ = copy_to_dev_cb;
    copy_from_dev_cb_ = copy_from_dev_cb;
    pmu_buffer_bytes_ = sizeof(PmuBuffer);
    setup_region_bytes_ = calc_pmu_setup_size(num_cores_);

    // Allocate the [PmuSetupHeader][PmuBufferState[num_cores]] shared region
    // on device in a single allocation.
    setup_header_dev_ = alloc_cb_(setup_region_bytes_);
    if (setup_header_dev_ == nullptr) {
        LOG_ERROR("PmuCollector::initialize: failed to alloc PMU setup region (%zu bytes)", setup_region_bytes_);
        return -1;
    }

    // Allocate one PmuBuffer per core and remember the device pointers.
    core_buffers_dev_.assign(num_cores_, nullptr);
    std::vector<uint8_t> host_setup(setup_region_bytes_, 0);
    PmuSetupHeader *host_header = get_pmu_setup_header(host_setup.data());
    host_header->num_cores = static_cast<uint32_t>(num_cores_);
    host_header->event_type = static_cast<uint32_t>(event_type_);
    for (int i = 0; i < num_cores_; ++i) {
        void *buf_dev = alloc_cb_(pmu_buffer_bytes_);
        if (buf_dev == nullptr) {
            LOG_ERROR("PmuCollector::initialize: failed to alloc PmuBuffer for core %d", i);
            finalize();
            return -1;
        }
        core_buffers_dev_[i] = buf_dev;
        host_header->buffer_ptrs[i] = reinterpret_cast<uint64_t>(buf_dev);

        // Zero the buffer header on device so count starts at 0. Per-core
        // dropped/total counters live in PmuBufferState (zero-init below).
        PmuBuffer zero_hdr{};
        int rc = copy_to_dev_cb_(buf_dev, &zero_hdr, PMU_BUFFER_HEADER_BYTES);
        if (rc != 0) {
            LOG_ERROR("PmuCollector::initialize: failed to zero PmuBuffer header for core %d (rc=%d)", i, rc);
            finalize();
            return rc;
        }
    }

    // Publish the setup region (header + zero-initialized PmuBufferState[]) to device.
    int rc = copy_to_dev_cb_(setup_header_dev_, host_setup.data(), setup_region_bytes_);
    if (rc != 0) {
        LOG_ERROR("PmuCollector::initialize: failed to publish PMU setup region (rc=%d)", rc);
        finalize();
        return rc;
    }

    *kernel_args_pmu_data_base = reinterpret_cast<uint64_t>(setup_header_dev_);
    LOG_INFO_V0(
        "PMU collector initialized: %d cores, event_type=%u, setup_header=0x%lx", num_cores_,
        static_cast<uint32_t>(event_type_), static_cast<unsigned long>(*kernel_args_pmu_data_base)
    );
    return 0;
}

int PmuCollector::collect_all() {
    if (setup_header_dev_ == nullptr) {
        LOG_ERROR("PmuCollector::collect_all: not initialized");
        return -1;
    }
    collected_records_.assign(num_cores_, {});
    dropped_counts_.assign(num_cores_, 0);
    total_counts_.assign(num_cores_, 0);
    owning_thread_ids_.assign(num_cores_, 0);

    // Pull the full setup region back to read all PmuBufferState[] in one shot.
    std::vector<uint8_t> host_setup(setup_region_bytes_, 0);
    int rc = copy_from_dev_cb_(host_setup.data(), setup_header_dev_, setup_region_bytes_);
    if (rc != 0) {
        LOG_ERROR("PmuCollector::collect_all: setup region copy failed (rc=%d)", rc);
        return rc;
    }
    for (int i = 0; i < num_cores_; ++i) {
        PmuBufferState *state = get_pmu_buffer_state(host_setup.data(), i);
        dropped_counts_[i] = state->dropped_record_count;
        total_counts_[i] = state->total_record_count;
        owning_thread_ids_[i] = state->owning_thread_id;
    }

    PmuBuffer header_buf{};
    for (int i = 0; i < num_cores_; ++i) {
        void *buf_dev = core_buffers_dev_[i];
        if (buf_dev == nullptr) {
            continue;
        }
        // Copy the 64-byte header to learn record count.
        rc = copy_from_dev_cb_(&header_buf, buf_dev, PMU_BUFFER_HEADER_BYTES);
        if (rc != 0) {
            LOG_ERROR("PmuCollector::collect_all: header copy failed for core %d (rc=%d)", i, rc);
            return rc;
        }
        uint32_t count = header_buf.count;
        if (count > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
            LOG_WARN(
                "PmuCollector::collect_all: core %d count=%u clamped to capacity %d", i, count,
                PLATFORM_PMU_RECORDS_PER_BUFFER
            );
            count = PLATFORM_PMU_RECORDS_PER_BUFFER;
        }
        if (count == 0) {
            continue;
        }

        // Copy just the used portion of the records array.
        collected_records_[i].resize(count);
        const uint8_t *records_dev = reinterpret_cast<const uint8_t *>(buf_dev) + offsetof(PmuBuffer, records);
        rc = copy_from_dev_cb_(
            collected_records_[i].data(), const_cast<uint8_t *>(records_dev),
            static_cast<size_t>(count) * sizeof(PmuRecord)
        );
        if (rc != 0) {
            LOG_ERROR("PmuCollector::collect_all: records copy failed for core %d (rc=%d)", i, rc);
            return rc;
        }
    }
    return 0;
}

int PmuCollector::export_csv(const std::string &output_dir) {
    if (setup_header_dev_ == nullptr) {
        return -1;
    }

    const PmuEventConfig *cfg = pmu_resolve_event_config_a5(event_type_);
    if (cfg == nullptr) {
        cfg = &PMU_EVENTS_A5_PIPE_UTIL;
    }

    std::string csv_path = make_pmu_csv_path(output_dir);
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        LOG_ERROR("PmuCollector::export_csv: failed to open %s", csv_path.c_str());
        return -1;
    }

    // Count named columns. Matches pypto's tilefwk_pmu_to_csv.py
    // table_pmu_header_3510 — each event group lists only the columns that
    // correspond to a named counter. pypto then trims each row to
    // len(fixed_header) + len(named) via _trim_task_pmu_list; we do the
    // equivalent positional trim here (first `named_count` counter values,
    // pmu_counters[0..named_count-1]).
    int named_count = 0;
    for (int i = 0; i < PMU_COUNTER_COUNT_A5; ++i) {
        if (cfg->counter_names[i] != nullptr && cfg->counter_names[i][0] != '\0') {
            ++named_count;
        }
    }

    // Header: fixed columns + named counter names + event_type column.
    // Column order matches a2a3 host pmu_collector for downstream tooling parity.
    csv << "thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles";
    int emitted = 0;
    for (int i = 0; i < PMU_COUNTER_COUNT_A5 && emitted < named_count; ++i) {
        if (cfg->counter_names[i] != nullptr && cfg->counter_names[i][0] != '\0') {
            csv << "," << cfg->counter_names[i];
            ++emitted;
        }
    }
    csv << ",event_type\n";

    uint64_t total_rows = 0;
    uint64_t total_dropped = 0;
    uint64_t device_total = 0;
    for (int core_id = 0; core_id < num_cores_; ++core_id) {
        const auto &records = collected_records_[core_id];
        uint32_t thread_id = owning_thread_ids_[core_id];
        for (const auto &rec : records) {
            csv << thread_id << "," << core_id << ",0x" << std::hex << rec.task_id << std::dec << "," << rec.func_id
                << "," << static_cast<int>(rec.core_type) << "," << rec.pmu_total_cycles;
            for (int i = 0; i < named_count; ++i) {
                csv << "," << rec.pmu_counters[i];
            }
            csv << "," << static_cast<uint32_t>(event_type_) << "\n";
            ++total_rows;
        }
        total_dropped += dropped_counts_[core_id];
        device_total += total_counts_[core_id];
    }
    csv.flush();
    LOG_INFO_V0("PMU CSV written to %s", csv_path.c_str());

    // Cross-check device-side totals against what we wrote to CSV. Invariant:
    //   device_total == collected + dropped (buffer-full) + slot-mismatch
    // collected + dropped should account for every commit attempt; any
    // remainder is silent slot-mismatch loss (AICore not yet published).
    if (total_dropped > 0) {
        LOG_WARN(
            "PMU collector: %lu records dropped on device side (PmuBuffer full). "
            "Increase PLATFORM_PMU_RECORDS_PER_BUFFER if this is frequent.",
            static_cast<unsigned long>(total_dropped)
        );
    }
    if (total_rows + total_dropped != device_total) {
        LOG_WARN(
            "PMU collector: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu, "
            "diff=%ld silent slot-mismatch losses)",
            static_cast<unsigned long>(total_rows), static_cast<unsigned long>(total_dropped),
            static_cast<unsigned long>(device_total),
            static_cast<long>(device_total) - static_cast<long>(total_rows + total_dropped)
        );
    } else {
        LOG_INFO_V0(
            "PMU collector: record counts match (collected=%lu, dropped=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_rows), static_cast<unsigned long>(total_dropped),
            static_cast<unsigned long>(device_total)
        );
    }
    return 0;
}

int PmuCollector::finalize() {
    int rc = 0;
    for (void *&buf : core_buffers_dev_) {
        if (buf != nullptr) {
            int r = free_cb_ ? free_cb_(buf) : 0;
            if (r != 0 && rc == 0) {
                rc = r;
            }
            buf = nullptr;
        }
    }
    core_buffers_dev_.clear();

    if (setup_header_dev_ != nullptr && free_cb_ != nullptr) {
        int r = free_cb_(setup_header_dev_);
        if (r != 0 && rc == 0) {
            rc = r;
        }
    }
    setup_header_dev_ = nullptr;

    num_cores_ = 0;
    event_type_ = PmuEventType::PIPE_UTILIZATION;
    pmu_buffer_bytes_ = 0;
    setup_region_bytes_ = 0;
    collected_records_.clear();
    dropped_counts_.clear();
    total_counts_.clear();
    owning_thread_ids_.clear();
    alloc_cb_ = nullptr;
    free_cb_ = nullptr;
    copy_to_dev_cb_ = nullptr;
    copy_from_dev_cb_ = nullptr;
    return rc;
}
