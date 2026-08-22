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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "aicpu/scope_stats_collector_aicpu.h"
#include "host/scope_stats_collector.h"

// HBG scope boundaries execute before device collectors start.
class ScopeStatsHostCapture {
public:
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    void begin_capture(int32_t task_window_cap, uint64_t heap_cap, int32_t tensormap_cap) {
        captured_ = enabled_;
        records_.clear();
        scopes_.clear();
        pending_site_file_.clear();
        pending_site_line_ = 0;
        heap_wraps_.assign(1, {});
        header_ = {};
        header_.num_instances = 1;
        header_.task_window_cap[0] = task_window_cap;
        header_.heap_cap[0] = heap_cap;
        // HBG polling readiness has no dependency-list pool.
        header_.dep_pool_cap[0] = 0;
        header_.tensormap_cap = tensormap_cap;
    }

    void set_pending_site(const char *file, int line) {
        if (!enabled_) return;
        pending_site_file_ = file == nullptr ? "" : file;
        pending_site_line_ = line;
    }

    void begin(
        int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end,
        int32_t dep_pool_start, int32_t dep_pool_end, int32_t tensormap_used
    ) {
        if (!enabled_ || scopes_.size() >= static_cast<size_t>(std::numeric_limits<int16_t>::max())) return;
        scopes_.push_back({std::move(pending_site_file_), pending_site_line_, ring_id});
        pending_site_file_.clear();
        pending_site_line_ = 0;
        append_record(
            ring_id, SCOPE_STATS_PHASE_BEGIN, task_start, task_end, heap_start, heap_end, dep_pool_start, dep_pool_end,
            tensormap_used
        );
    }

    void
    end(int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end,
        int32_t dep_pool_start, int32_t dep_pool_end, int32_t tensormap_used) {
        if (!enabled_ || scopes_.empty()) return;
        append_record(
            ring_id, SCOPE_STATS_PHASE_END, task_start, task_end, heap_start, heap_end, dep_pool_start, dep_pool_end,
            tensormap_used
        );
        scopes_.pop_back();
    }

    void note_heap_wrap(int side) {
        if (!enabled_ || scopes_.empty()) return;
        if (side != SCOPE_STATS_HEAP_SIDE_ALLOC && side != SCOPE_STATS_HEAP_SIDE_RECLAIM) return;
        const int ring_id = scopes_.back().ring_id;
        if (ring_id < 0 || static_cast<size_t>(ring_id) >= heap_wraps_.size()) return;
        ++heap_wraps_[ring_id][side];
    }

    void on_fatal() {
        if (enabled_) header_.fatal_latched = 1;
    }

    int write_jsonl(const char *output_dir) const {
        if (output_dir == nullptr) return -1;
        if (!captured_) return -3;
        return write_scope_stats_jsonl(
            output_dir, header_, /*dropped_record_count=*/0, static_cast<uint32_t>(records_.size()), records_
        );
    }

    const std::vector<ScopeStatsRecord> &records() const { return records_; }
    const ScopeStatsDataHeader &header() const { return header_; }

private:
    static std::string_view basename_of(std::string_view path) {
        if (path.empty()) return "(unknown)";
        const size_t separator = path.find_last_of("/\\");
        return separator == std::string_view::npos ? path : path.substr(separator + 1);
    }

    static void copy_basename(char (&dst)[32], std::string_view path) {
        const std::string_view basename = basename_of(path);
        const size_t length = std::min(basename.size(), sizeof(dst) - 1);
        std::copy_n(basename.begin(), length, dst);
        dst[length] = '\0';
    }

    uint64_t unroll_heap_offset(uint64_t offset, int ring_id, int side) const {
        if (ring_id < 0 || static_cast<size_t>(ring_id) >= heap_wraps_.size()) return offset;
        return offset + heap_wraps_[ring_id][side] * header_.heap_cap[ring_id];
    }

    void append_record(
        int ring_id, int16_t phase, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end,
        int32_t dep_pool_start, int32_t dep_pool_end, int32_t tensormap_used
    ) {
        ScopeStatsRecord record{};
        const ScopeFrame &scope = scopes_.back();
        copy_basename(record.site_file_basename, scope.site_file);
        record.site_line = scope.site_line;
        record.depth = static_cast<int16_t>(scopes_.size() - 1);
        record.ring_id = static_cast<int16_t>(ring_id);
        record.phase = phase;
        record.task_start = task_start;
        record.task_end = task_end;
        record.dep_pool_start = dep_pool_start;
        record.dep_pool_end = dep_pool_end;
        record.tensormap_used = tensormap_used;
        record.heap_start = unroll_heap_offset(heap_start, ring_id, SCOPE_STATS_HEAP_SIDE_RECLAIM);
        record.heap_end = unroll_heap_offset(heap_end, ring_id, SCOPE_STATS_HEAP_SIDE_ALLOC);
        records_.push_back(record);
    }

    struct ScopeFrame {
        std::string site_file;
        int32_t site_line;
        int32_t ring_id;
    };

    bool enabled_{false};
    bool captured_{false};
    std::string pending_site_file_;
    int32_t pending_site_line_{0};
    std::vector<ScopeFrame> scopes_;
    std::vector<std::array<uint64_t, 2>> heap_wraps_;
    ScopeStatsDataHeader header_{};
    std::vector<ScopeStatsRecord> records_;
};
