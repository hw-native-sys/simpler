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

#include "host/host_phase_records.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <fstream>

#include "common/unified_log.h"

namespace simpler::dfx {
namespace {

// A pass identity must be unique across every pool in the process, not merely
// within one. A producer caches its lane against the identity of the pass it
// claimed it for, and a store is per DeviceRunner: a freshly constructed store can
// occupy a destroyed one's address with a per-pool counter that has reached the
// same value, at which point a stale cached lane looks valid for a pass it never
// claimed and appends at an offset into a buffer this pass has cleared.
uint32_t next_pass_generation() {
    static std::atomic<uint32_t> generation{0};
    return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

}  // namespace

HostPhaseRecordPool *HostPhaseRecordStore::arm(bool collect_records) {
    // Everything a producer reads about this pass is written before the generation
    // is published, and the generation is published last with a release. That store
    // is the pass's only publication point: a producer acquires the generation and
    // is then guaranteed to see the buffer pointer and count that go with it.
    // Publishing the generation first would let a producer pair a new generation
    // with the null pointer below and dereference it.
    pool_.next_buffer.store(0, std::memory_order_relaxed);
    pool_.dropped.store(0, std::memory_order_relaxed);
    pool_.buffers = nullptr;
    pool_.buffer_count = 0;
    armed_ = false;
    finished_ = false;
    records_written_ = false;
    submitted_tasks_ = 0;
    // A pass's identity must not survive into the next one: write_records_jsonl
    // asks only whether the store is armed, so a pass written before finish()
    // would otherwise carry the previous invocation id and join to the wrong span.
    invocation_id_ = 0;
    if (!collect_records) {
        buffers_.clear();
        buffers_.shrink_to_fit();
        // Still a new pass, and publishing it is what retires the producer lanes a
        // previous pass handed out — a producer holding one must not append into
        // storage this call just released.
        pool_.generation.store(next_pass_generation(), std::memory_order_release);
        return nullptr;
    }

    // Allocated once and reused for every later pass: a pass's records are only
    // valid until the next arm(), so the storage can be too.
    buffers_.resize(static_cast<size_t>(PLATFORM_HOST_PHASE_BUFFERS));
    for (auto &buffer : buffers_) {
        buffer.count = 0;
    }
    pool_.buffers = buffers_.data();
    pool_.buffer_count = static_cast<uint32_t>(buffers_.size());
    armed_ = true;
    // Last, and with a release: see the top of this function.
    pool_.generation.store(next_pass_generation(), std::memory_order_release);
    return &pool_;
}

HostPhaseRecordPool *HostPhaseRecordStore::resume() {
    if (!armed_ || !finished_ || records_written_) return nullptr;
    finished_ = false;
    // Existing buffers, their dropped ledger and the allocation cursor stay
    // owned by this run. A retry takes unused buffers instead of overwriting
    // a previous producer's partially filled one.
    pool_.generation.store(next_pass_generation(), std::memory_order_release);
    return &pool_;
}

uint64_t HostPhaseRecordStore::stored_records() const {
    uint64_t total = 0;
    for (const auto &buffer : buffers_) {
        total += buffer.count < PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER ?
                     buffer.count :
                     static_cast<uint32_t>(PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER);
    }
    return total;
}

void HostPhaseRecordStore::finish(uint64_t submitted_tasks, uint64_t invocation_id) {
    if (!armed_) return;
    submitted_tasks_ = submitted_tasks;
    invocation_id_ = invocation_id;
    finished_ = true;
    if (dropped_records() > 0) {
        LOG_WARN(
            "Host phase pool dropped %llu of %llu records (%d buffers of %d); trace data is incomplete",
            static_cast<unsigned long long>(dropped_records()), static_cast<unsigned long long>(total_records()),
            PLATFORM_HOST_PHASE_BUFFERS, PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER
        );
    }
}

int HostPhaseRecordStore::write_records_jsonl(const std::string &path) {
    if (!armed_ || records_written_) return 0;
    const std::vector<HostPhaseRecord> all = records();
    if (all.empty() && dropped_records() == 0) return 0;
    records_written_ = true;

    // Appended one object per pass: a --rounds run emits a pass per round, and
    // each object carries the identity needed to join it to that round's spans.
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        LOG_ERROR("Host phase records: cannot open '%s' for append", path.c_str());
        return -1;
    }
    // One line per pass (JSON Lines), matching scope_stats.jsonl: a reader can
    // take a pass at a time without buffering the file or tracking brace depth.
    out << "{\"pid\": " << static_cast<int>(getpid()) << ", \"inv\": " << invocation_id_
        << ", \"clock\": \"host_monotonic_ns\", \"dropped\": " << dropped_records() << ", \"records\": [";
    bool first = true;
    for (const auto &record : all) {
        if (!first) out << ", ";
        out << "{\"phase\": \"" << host_phase_kind_name(static_cast<HostPhaseKind>(record.kind))
            << "\", \"start_ns\": " << record.start_ns << ", \"end_ns\": " << record.end_ns
            << ", \"detail\": " << record.payload << ", \"tid\": " << record.thread_id << "}";
        first = false;
    }
    out << "]}\n";
    out.flush();
    if (!out.good()) {
        LOG_ERROR("Host phase records: write to '%s' failed", path.c_str());
        return -1;
    }
    LOG_DEBUG("Host phase records: appended %zu records to %s", all.size(), path.c_str());
    return 0;
}

std::vector<HostPhaseRecord> HostPhaseRecordStore::records() const {
    std::vector<HostPhaseRecord> out;
    if (!armed_) return out;
    out.reserve(static_cast<size_t>(stored_records()));
    for (const auto &buffer : buffers_) {
        const uint32_t count = buffer.count < PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER ?
                                   buffer.count :
                                   static_cast<uint32_t>(PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER);
        for (uint32_t i = 0; i < count; ++i) {
            out.push_back(buffer.records[i]);
        }
    }
    // A buffer holds one producer's records, so concatenating the buffers groups
    // by producer rather than ordering by time. Sorting here is off any measured
    // path and gives every reader one well-defined order.
    std::stable_sort(out.begin(), out.end(), [](const HostPhaseRecord &a, const HostPhaseRecord &b) {
        return a.start_ns < b.start_ns;
    });
    return out;
}

std::vector<HostPhaseRecord> HostPhaseRecordStore::submit_records() const {
    std::vector<HostPhaseRecord> out;
    for (const auto &record : records()) {
        if (host_phase_kind_submits_task(static_cast<HostPhaseKind>(record.kind))) {
            out.push_back(record);
        }
    }
    return out;
}

std::vector<HostPhaseRecord> HostPhaseRecordStore::device_upload_records() const {
    std::vector<HostPhaseRecord> out;
    for (const auto &record : records()) {
        if (host_phase_kind_is_device_upload(static_cast<HostPhaseKind>(record.kind))) {
            out.push_back(record);
        }
    }
    return out;
}

}  // namespace simpler::dfx
