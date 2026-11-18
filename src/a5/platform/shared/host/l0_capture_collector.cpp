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

#include "host/l0_capture_collector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>

#include "common/unified_log.h"
#include "host/profiling_copy.h"

namespace {

// Device address from a device base pointer plus a byte offset, without
// dereferencing device memory on the host.
inline const volatile void *dev_offset(const void *base, size_t off) {
    return reinterpret_cast<const volatile void *>(reinterpret_cast<uintptr_t>(base) + off);
}

}  // namespace

L0CaptureCollector::~L0CaptureCollector() { stop(); }

void L0CaptureCollector::stop_for_device_failure() { stop(); }

void L0CaptureCollector::abandon_after_device_reset() {
    stop();
    outputs_.clear();
    output_dir_.clear();
    physical_core_ids_.clear();
    ring_addrs_.clear();
    shm_host_.clear();
    shm_dev_ = nullptr;
    initialized_ = false;
    device_io_failed_ = false;
}

int L0CaptureCollector::init(
    const std::vector<uint32_t> &physical_core_ids, int num_threads, const std::vector<void *> &ring_addrs,
    Alloc alloc_cb, Free free_cb, int device_id
) {
    if (initialized_ || device_io_failed_) return -1;
    if (physical_core_ids.empty() || physical_core_ids.size() > PLATFORM_MAX_CORES || num_threads <= 0 ||
        num_threads > PLATFORM_MAX_AICPU_THREADS || ring_addrs.size() != physical_core_ids.size() || !alloc_cb ||
        !free_cb) {
        return -1;
    }
    try {
        physical_core_ids_ = physical_core_ids;
        ring_addrs_ = ring_addrs;
        shm_size_ = l0_dfx_data_size(static_cast<uint32_t>(physical_core_ids.size()));
        shm_host_.resize(shm_size_);
    } catch (const std::bad_alloc &) {
        return -1;
    }
    shm_dev_ = alloc_cb(shm_size_);
    if (!shm_dev_) return -1;
    std::memset(shm_host_.data(), 0, shm_size_);
    auto *header = reinterpret_cast<L0DataHeader *>(shm_host_.data());
    header->num_cores = static_cast<uint32_t>(physical_core_ids.size());
    header->num_threads = static_cast<uint32_t>(num_threads);
    num_threads_ = static_cast<uint32_t>(num_threads);
    for (uint32_t core = 0; core < header->num_cores; ++core) {
        l0_dfx_state(shm_host_.data(), core)->physical_core_id = physical_core_ids[core];
    }
    if (profiling_copy_to_device(shm_dev_, shm_host_.data(), shm_size_) != 0) {
        // A failed device copy may indicate a poisoned context. Leave the
        // allocation for the runner's confirmed device reset.
        device_io_failed_ = true;
        shm_dev_ = nullptr;
        return -1;
    }
    alloc_ = std::move(alloc_cb);
    free_ = std::move(free_cb);
    device_id_ = device_id;
    initialized_ = true;
    return 0;
}

int L0CaptureCollector::begin_run(
    uint64_t run_epoch, const std::filesystem::path &output_dir, uint32_t ring_len, uint32_t ring_capacity,
    uint64_t aic_frequency_hz, uint64_t aiv_frequency_hz
) {
    if (!initialized_ || !outputs_.empty() || ring_len == 0 || ring_capacity < ring_len) return -1;
    try {
        if (copy_buffer_.size() < ring_len) copy_buffer_.resize(ring_len);
    } catch (const std::bad_alloc &) {
        return -1;
    }
    const auto dir = output_dir / "l0_raw" / ("run_" + std::to_string(run_epoch));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return -1;
    outputs_.resize(physical_core_ids_.size());
    for (size_t core = 0; core < outputs_.size(); ++core) {
        const auto name = "core_" + std::to_string(physical_core_ids_[core]);
        auto &out = outputs_[core];
        out.raw.open(dir / (name + ".bin"), std::ios::binary | std::ios::trunc);
        out.index.open(dir / (name + ".csv"), std::ios::trunc);
        if (!out.raw || !out.index) {
            output_failed_.store(true);
            return -1;
        }
        out.index << "offset,bytes,chunk_sequence,ring_offset,start_wptr,end_wptr,start_samp_wrt,end_samp_wrt,tail_end,"
                     "flags\n";
    }
    run_epoch_ = run_epoch;
    ring_len_ = ring_len;
    ring_capacity_ = ring_capacity;
    aic_frequency_hz_ = aic_frequency_hz;
    aiv_frequency_hz_ = aiv_frequency_hz;
    live_batches_ = 0;
    output_dir_ = dir;
    output_failed_.store(false);
    return 0;
}

void L0CaptureCollector::start(const profiling_common::ThreadFactory &create_thread) {
    if (!initialized_) return;
    stop_.store(false, std::memory_order_release);
    drain_thread_ = create_thread([this] {
        drain_loop();
    });
}

void L0CaptureCollector::stop() {
    stop_.store(true, std::memory_order_release);
    if (drain_thread_.joinable()) {
        drain_thread_.join();
        drain_thread_ = std::thread();
    }
}

bool L0CaptureCollector::drain_available() {
    if (shm_dev_ == nullptr) return false;
    static_assert(
        offsetof(L0DataHeader, queue_tails) == offsetof(L0DataHeader, queue_heads) + sizeof(L0DataHeader::queue_heads)
    );
    uint32_t cursors[2 * PLATFORM_MAX_AICPU_THREADS]{};
    if (profiling_copy_from_device(
            cursors, dev_offset(shm_dev_, offsetof(L0DataHeader, queue_heads)), sizeof(cursors)
        ) != 0)
        return false;
    bool made_progress = false;
    for (uint32_t t = 0; t < num_threads_; ++t) {
        uint32_t head = cursors[t];
        const uint32_t tail = cursors[PLATFORM_MAX_AICPU_THREADS + t];
        while (head != tail) {
            L0RangeEntry entry;
            const size_t entry_off = offsetof(L0DataHeader, queues) +
                                     (static_cast<size_t>(t) * L0_DFX_READY_QUEUE_SIZE + head) * sizeof(L0RangeEntry);
            if (profiling_copy_from_device(&entry, dev_offset(shm_dev_, entry_off), sizeof(entry)) != 0) break;
            collect_range(entry);
            head = (head + 1) % L0_DFX_READY_QUEUE_SIZE;
            made_progress = true;
            if (profiling_copy_to_device(
                    const_cast<volatile void *>(
                        dev_offset(shm_dev_, offsetof(L0DataHeader, queue_heads) + t * sizeof(uint32_t))
                    ),
                    &head, sizeof(head)
                ) != 0) {
                output_failed_.store(true);
                return made_progress;
            }
        }
    }
    return made_progress;
}

void L0CaptureCollector::drain_loop() {
    while (!stop_.load(std::memory_order_acquire)) {
        if (!drain_available()) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
}

void L0CaptureCollector::drain_remaining() { drain_available(); }

bool L0CaptureCollector::dump_gm_rings() {
    if (!initialized_ || outputs_.size() != ring_addrs_.size() || copy_buffer_.empty()) return false;
    bool complete = true;
    for (size_t core = 0; core < outputs_.size(); ++core) {
        const auto path = output_dir_ / ("core_" + std::to_string(physical_core_ids_[core]) + ".gm.bin");
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        bool copied = out.good() && ring_addrs_[core] != nullptr;
        for (uint32_t offset = 0; copied && offset < ring_capacity_;) {
            const size_t bytes = std::min<size_t>(copy_buffer_.size(), ring_capacity_ - offset);
            copied = profiling_copy_from_device(copy_buffer_.data(), dev_offset(ring_addrs_[core], offset), bytes) == 0;
            if (copied) {
                out.write(reinterpret_cast<const char *>(copy_buffer_.data()), static_cast<std::streamsize>(bytes));
                copied = out.good();
            }
            offset += static_cast<uint32_t>(bytes);
        }
        out.close();
        copied = copied && out.good();
        std::error_code ec;
        if (copied) {
            std::filesystem::rename(temp, path, ec);
            copied = !ec;
        }
        if (copied) {
            outputs_[core].gm_snapshot_bytes = ring_capacity_;
        } else {
            std::filesystem::remove(temp, ec);
            complete = false;
        }
    }
    return complete;
}

void L0CaptureCollector::collect_range(const L0RangeEntry &entry) {
    if (entry.core_index >= outputs_.size() || entry.core_index >= ring_addrs_.size()) {
        output_failed_.store(true);
        return;
    }
    const auto range = simpler::dfx::l0::plan_ring_range(
        entry.start_wptr, entry.start_samp_wrt, entry.end_samp_wrt, entry.end_wptr, ring_len_, ring_capacity_
    );
    if (range.invalid_pointer || range.pointer_mismatch || range.possible_overrun || range.lost_bytes != 0 ||
        range.bytes > copy_buffer_.size()) {
        output_failed_.store(true);
        return;
    }
    const auto *ring = static_cast<const uint8_t *>(ring_addrs_[entry.core_index]);
    if (range.first_bytes != 0 &&
        profiling_copy_from_device(copy_buffer_.data(), dev_offset(ring, range.start), range.first_bytes) != 0) {
        output_failed_.store(true);
        return;
    }
    if (range.second_bytes != 0 &&
        profiling_copy_from_device(copy_buffer_.data() + range.first_bytes, dev_offset(ring, 0), range.second_bytes) !=
            0) {
        output_failed_.store(true);
        return;
    }
    if (!verify_live_copy(entry)) {
        output_failed_.store(true);
        return;
    }
    std::lock_guard<std::mutex> lock(output_mutex_);
    auto &out = outputs_[entry.core_index];
    if (entry.sequence != out.next_sequence) {
        output_failed_.store(true);
        return;
    }
    out.raw.write(reinterpret_cast<const char *>(copy_buffer_.data()), static_cast<std::streamsize>(range.bytes));
    out.index << out.offset << ',' << range.bytes << ',' << entry.sequence << ',' << range.start << ','
              << entry.start_wptr << ',' << entry.end_wptr << ',' << entry.start_samp_wrt << ',' << entry.end_samp_wrt
              << ',' << range.start + range.first_bytes << ',' << entry.flags << '\n';
    out.offset += range.bytes;
    ++out.next_sequence;
    if (!out.raw || !out.index) output_failed_.store(true);
    if (!advance_consumed(entry.core_index, entry.end_wptr, entry.end_samp_wrt)) output_failed_.store(true);
}

bool L0CaptureCollector::verify_live_copy(const L0RangeEntry &entry) {
    if ((entry.flags & L0_DFX_FLAG_LIVE) == 0) return true;
    static_assert(
        offsetof(L0BufferState, verify_response) == offsetof(L0BufferState, verified_samp_wrt) + sizeof(uint32_t)
    );
    static_assert(
        offsetof(L0BufferState, producer_stopped) == offsetof(L0BufferState, verify_response) + sizeof(uint32_t)
    );
    const size_t state_off = sizeof(L0DataHeader) + static_cast<size_t>(entry.core_index) * sizeof(L0BufferState);
    const uint32_t request = entry.sequence + 1;
    if (request == 0 ||
        profiling_copy_to_device(
            const_cast<volatile void *>(dev_offset(shm_dev_, state_off + offsetof(L0BufferState, verify_request))),
            &request, sizeof(request)
        ) != 0)
        return false;
    const uint32_t guard = std::min<uint32_t>(64u << 10, ring_len_ / 8);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    while (true) {
        // The AICPU writes the cursor before either flag (verify_response, then
        // producer_stopped) with a barrier in between, so a single multi-field
        // copy can pair a fresh flag with a stale cursor. Read each flag, then the
        // cursor, then each flag again, and accept the cursor only when the flag
        // that fired is stable across both reads.
        uint32_t response = 0;
        uint32_t stopped = 0;
        uint32_t cursor = 0;
        uint32_t response_after = 0;
        uint32_t stopped_after = 0;
        if (profiling_copy_from_device(
                &response, dev_offset(shm_dev_, state_off + offsetof(L0BufferState, verify_response)), sizeof(response)
            ) != 0 ||
            profiling_copy_from_device(
                &stopped, dev_offset(shm_dev_, state_off + offsetof(L0BufferState, producer_stopped)), sizeof(stopped)
            ) != 0 ||
            profiling_copy_from_device(
                &cursor, dev_offset(shm_dev_, state_off + offsetof(L0BufferState, verified_samp_wrt)), sizeof(cursor)
            ) != 0 ||
            profiling_copy_from_device(
                &response_after, dev_offset(shm_dev_, state_off + offsetof(L0BufferState, verify_response)),
                sizeof(response_after)
            ) != 0 ||
            profiling_copy_from_device(
                &stopped_after, dev_offset(shm_dev_, state_off + offsetof(L0BufferState, producer_stopped)),
                sizeof(stopped_after)
            ) != 0)
            return false;
        const bool verified = response == request && response_after == request;
        const bool halted = stopped != 0 && stopped_after != 0;
        if (halted || verified) {
            const uint32_t produced_since_start = cursor - entry.start_samp_wrt;
            const uint32_t published_span = entry.end_samp_wrt - entry.start_samp_wrt;
            return produced_since_start >= published_span && produced_since_start < ring_len_ - guard;
        }
        if (stop_.load(std::memory_order_acquire) || std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

bool L0CaptureCollector::advance_consumed(uint32_t core_index, uint32_t end_wptr, uint32_t end_samp_wrt) {
    static_assert(
        offsetof(L0BufferState, consumed_samp_wrt) == offsetof(L0BufferState, consumed_wptr) + sizeof(uint32_t)
    );
    const size_t state_off = sizeof(L0DataHeader) + static_cast<size_t>(core_index) * sizeof(L0BufferState);
    const uint32_t consumed[2] = {end_wptr, end_samp_wrt};
    return profiling_copy_to_device(
               const_cast<volatile void *>(dev_offset(shm_dev_, state_off + offsetof(L0BufferState, consumed_wptr))),
               consumed, sizeof(consumed)
           ) == 0;
}

bool L0CaptureCollector::reconcile(bool terminal_status_ok) {
    if (!initialized_ || shm_dev_ == nullptr || outputs_.size() != physical_core_ids_.size()) {
        output_failed_.store(true);
        return false;
    }
    // The manifest may claim success only after every raw and index file has
    // been closed successfully. No drain thread is running at this point.
    for (auto &out : outputs_) {
        out.raw.close();
        out.index.close();
        if (!out.raw || !out.index) output_failed_.store(true);
    }
    if (profiling_copy_from_device(shm_host_.data(), shm_dev_, shm_size_) != 0) {
        output_failed_.store(true);
        return false;
    }
    bool matched = !output_failed_.load();
    bool producers_stopped = true;
    live_batches_ = 0;
    uint64_t total_raw_bytes = 0;
    uint64_t total_gm_snapshot_bytes = 0;
    bool gm_snapshot_complete = true;
    std::ofstream manifest(output_dir_ / "manifest.json", std::ios::trunc);
    if (!manifest) matched = false;
    if (manifest) {
        manifest << "{\n  \"run_epoch\": " << run_epoch_
                 << ",\n  \"system_counter_frequency_hz\": " << PLATFORM_PROF_SYS_CNT_FREQ
                 << ",\n  \"aic_frequency_hz\": " << aic_frequency_hz_
                 << ",\n  \"aiv_frequency_hz\": " << aiv_frequency_hz_ << ",\n  \"cores\": [\n";
    }
    for (size_t core = 0; core < outputs_.size(); ++core) {
        const auto *state = l0_dfx_state(shm_host_.data(), static_cast<uint32_t>(core));
        live_batches_ += state->live_batch_count;
        total_raw_bytes += outputs_[core].offset;
        total_gm_snapshot_bytes += outputs_[core].gm_snapshot_bytes;
        if (outputs_[core].gm_snapshot_bytes != ring_capacity_) gm_snapshot_complete = false;
        if (manifest) {
            if (core != 0) manifest << ",\n";
            manifest << "    {\"core_index\": " << core << ", \"physical_core_id\": " << physical_core_ids_[core]
                     << ", \"raw_bytes\": " << outputs_[core].offset
                     << ", \"gm_snapshot_bytes\": " << outputs_[core].gm_snapshot_bytes
                     << ", \"overrun\": " << (state->capture_error == L0_DFX_CAPTURE_OVERRUN ? "true" : "false")
                     << ", \"chunk_count\": " << outputs_[core].next_sequence
                     << ", \"live_batch_count\": " << state->live_batch_count
                     << ", \"sample_count\": " << state->sample_count << ", \"advance_count\": " << state->advance_count
                     << ", \"derive_fallback_count\": " << state->derive_fallback_count
                     << ", \"last_observed_wptr\": " << state->last_observed_wptr
                     << ", \"last_observed_samp_wrt\": " << state->last_observed_samp_wrt
                     << ", \"core_type\": " << state->core_type << ", \"producer_stopped\": " << state->producer_stopped
                     << ", \"capture_error\": " << state->capture_error << "}";
        }
        if (state->producer_stopped != 1) producers_stopped = false;
        if (state->capture_error != L0_DFX_CAPTURE_OK || state->published_bytes != outputs_[core].offset) {
            matched = false;
        }
    }
    if (manifest) {
        manifest << "\n  ],\n  \"live_batch_count\": " << live_batches_
                 << ",\n  \"raw_data_present\": " << (total_raw_bytes != 0 ? "true" : "false")
                 << ",\n  \"gm_snapshot_bytes\": " << total_gm_snapshot_bytes
                 << ",\n  \"gm_snapshot_complete\": " << (gm_snapshot_complete ? "true" : "false")
                 << ",\n  \"capture_complete\": "
                 << (matched && terminal_status_ok && producers_stopped && total_raw_bytes != 0 ? "true" : "false")
                 << ",\n  \"raw_transport_complete\": " << (matched ? "true" : "false") << "\n}\n";
        manifest.flush();
        if (!manifest) matched = false;
    }
    if (!matched) output_failed_.store(true);
    return matched && producers_stopped;
}

std::vector<uint32_t> L0CaptureCollector::physical_core_types() const {
    std::vector<uint32_t> types(PLATFORM_MAX_CORES, L0_DFX_CORE_TYPE_UNKNOWN);
    if (shm_host_.size() < l0_dfx_data_size(static_cast<uint32_t>(physical_core_ids_.size()))) return types;
    const auto *states = reinterpret_cast<const L0BufferState *>(shm_host_.data() + sizeof(L0DataHeader));
    for (size_t core = 0; core < physical_core_ids_.size(); ++core) {
        const uint32_t physical_id = physical_core_ids_[core];
        if (physical_id < types.size()) types[physical_id] = states[core].core_type;
    }
    return types;
}

void L0CaptureCollector::finalize(Free free_cb) {
    stop();
    for (auto &out : outputs_) {
        if (out.raw.is_open()) out.raw.close();
        if (out.index.is_open()) out.index.close();
        if (!out.raw || !out.index) output_failed_.store(true);
    }
    outputs_.clear();
    output_dir_.clear();
    if (shm_dev_ != nullptr && free_cb) free_cb(shm_dev_);
    shm_dev_ = nullptr;
    shm_host_.clear();
    num_threads_ = 0;
    physical_core_ids_.clear();
    ring_addrs_.clear();
    initialized_ = false;
}
