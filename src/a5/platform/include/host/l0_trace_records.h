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

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace simpler::dfx::l0 {

enum class TraceKind : uint8_t { StartStamp, State, MarkStamp, Unknown, EndFlag };

struct TraceEvent {
    TraceKind kind;
    uint64_t anchor_syscnt;
    uint64_t elapsed_cycles;
    uint32_t raw_word;
    uint16_t block_id;
    uint16_t payload;
    uint8_t ctrl_type;
};

struct TraceStats {
    uint64_t start_stamps{0};
    uint64_t states{0};
    uint64_t mark_stamps{0};
    uint64_t unknown{0};
    uint64_t fillers{0};
    uint64_t unexpected_fillers{0};
    uint64_t end_flags{0};
    uint64_t malformed_start_groups{0};
    uint64_t words_without_start{0};
    uint64_t trailing_bytes{0};
    uint64_t incomplete_start_groups{0};

    bool structurally_complete() const {
        return start_stamps > 0 && malformed_start_groups == 0 && words_without_start == 0 && trailing_bytes == 0 &&
               incomplete_start_groups == 0 && unexpected_fillers == 0 && unknown == 0;
    }
};

// Decodes a single physical sub-core's GM byte stream. The caller supplies
// ranges in write order, including a wrapped range split at the ring boundary.
// It does not infer task ownership or silently discard STATE records.
class TraceDecoder {
public:
    template <typename Emit>
    void feed(const uint8_t *bytes, size_t size, Emit &&emit) {
        if (bytes == nullptr) return;
        for (size_t i = 0; i < size; ++i) {
            word_bytes_[word_bytes_used_++] = bytes[i];
            if (word_bytes_used_ == word_bytes_.size()) {
                uint32_t word = static_cast<uint32_t>(word_bytes_[0]) | (static_cast<uint32_t>(word_bytes_[1]) << 8) |
                                (static_cast<uint32_t>(word_bytes_[2]) << 16) |
                                (static_cast<uint32_t>(word_bytes_[3]) << 24);
                word_bytes_used_ = 0;
                accept_word(word, emit);
            }
        }
    }

    const TraceStats &finish() {
        stats_.trailing_bytes = word_bytes_used_;
        stats_.incomplete_start_groups = start_words_used_ != 0 ? 1 : 0;
        return stats_;
    }

    const TraceStats &stats() const { return stats_; }

private:
    template <typename Emit>
    void accept_word(uint32_t word, Emit &emit) {
        const uint8_t ctrl = static_cast<uint8_t>(word >> 28);
        if (word == 0xEFFFFFFFu || word == 0xDDDDDDDDu) {
            if (start_words_used_ != 0) {
                ++stats_.malformed_start_groups;
                start_words_used_ = 0;
                have_start_ = false;
            }
            if (word == 0xEFFFFFFFu) {
                ++stats_.end_flags;
                after_end_ = true;
                have_start_ = false;
                emit(TraceEvent{TraceKind::EndFlag, anchor_syscnt_, elapsed_cycles_, word, block_id_, 0, 14});
            } else {
                ++stats_.fillers;
                if (!after_end_) ++stats_.unexpected_fillers;
            }
            return;
        }
        if (start_words_used_ != 0) {
            if (ctrl == 14) {
                start_words_[start_words_used_++] = word;
                if (start_words_used_ == start_words_.size()) {
                    anchor_syscnt_ = 0;
                    for (int i = 3; i >= 0; --i)
                        anchor_syscnt_ = (anchor_syscnt_ << 16) | (start_words_[i] & 0xffffu);
                    block_id_ = static_cast<uint16_t>(
                        ((start_words_[1] >> 16) & 0xffu) << 8 | ((start_words_[0] >> 16) & 0xffu)
                    );
                    elapsed_cycles_ = 0;
                    have_start_ = true;
                    start_words_used_ = 0;
                    ++stats_.start_stamps;
                    emit(TraceEvent{TraceKind::StartStamp, anchor_syscnt_, 0, word, block_id_, 0, 14});
                }
                return;
            }
            ++stats_.malformed_start_groups;
            start_words_used_ = 0;
            have_start_ = false;
        }
        after_end_ = false;
        if (ctrl == 14) {
            start_words_[0] = word;
            start_words_used_ = 1;
            return;
        }
        if (!have_start_) {
            ++stats_.words_without_start;
            return;
        }
        elapsed_cycles_ += word & 0xffffu;
        const auto kind = ctrl <= 6 ? TraceKind::MarkStamp : ctrl == 15 ? TraceKind::State : TraceKind::Unknown;
        if (kind == TraceKind::MarkStamp) ++stats_.mark_stamps;
        else if (kind == TraceKind::State) ++stats_.states;
        else ++stats_.unknown;
        emit(
            TraceEvent{
                kind, anchor_syscnt_, elapsed_cycles_, word, block_id_, static_cast<uint16_t>((word >> 16) & 0xfffu),
                ctrl
            }
        );
    }

    std::array<uint8_t, 4> word_bytes_{};
    size_t word_bytes_used_{0};
    std::array<uint32_t, 4> start_words_{};
    size_t start_words_used_{0};
    uint64_t anchor_syscnt_{0};
    uint64_t elapsed_cycles_{0};
    uint16_t block_id_{0};
    bool have_start_{false};
    bool after_end_{false};
    TraceStats stats_{};
};

struct BusyInterval {
    uint64_t anchor_syscnt;
    uint64_t start_cycles;
    uint64_t end_cycles;
    uint16_t block_id;
    uint8_t lane;
    uint8_t clip_flags;  // bit 0: start clipped; bit 1: end clipped
};

struct BusyIntervalStats {
    uint64_t emitted{0};
    uint64_t clipped_start{0};
    uint64_t clipped_end{0};
};

// A STATE word is a seven-lane snapshot. Open intervals are clipped to the
// first observed busy sample or the last timestamp in this captured segment.
class BusyIntervalBuilder {
public:
    template <typename Emit>
    void accept(const TraceEvent &event, Emit &&emit) {
        if (event.kind == TraceKind::StartStamp) {
            close_segment(emit);
            anchor_syscnt_ = event.anchor_syscnt;
            block_id_ = event.block_id;
            last_cycles_ = 0;
            active_ = true;
            return;
        }
        if (event.kind == TraceKind::EndFlag) {
            if (active_) last_cycles_ = event.elapsed_cycles;
            close_segment(emit);
            return;
        }
        if (!active_) return;
        last_cycles_ = event.elapsed_cycles;
        if (event.kind != TraceKind::State) return;
        const uint16_t mask = event.payload & 0x7fu;
        for (uint8_t lane = 0; lane < lanes_.size(); ++lane) {
            const bool busy = (mask & (1u << lane)) != 0;
            auto &state = lanes_[lane];
            if (state == LaneState::Unknown) {
                state = busy ? LaneState::Busy : LaneState::Idle;
                if (busy) {
                    starts_[lane] = event.elapsed_cycles;
                    start_clipped_[lane] = true;
                    ++stats_.clipped_start;
                }
            } else if (state == LaneState::Idle && busy) {
                state = LaneState::Busy;
                starts_[lane] = event.elapsed_cycles;
                start_clipped_[lane] = false;
            } else if (state == LaneState::Busy && !busy) {
                emit(
                    BusyInterval{
                        anchor_syscnt_, starts_[lane], event.elapsed_cycles, block_id_, lane,
                        static_cast<uint8_t>(start_clipped_[lane] ? 1u : 0u)
                    }
                );
                ++stats_.emitted;
                state = LaneState::Idle;
            }
        }
    }

    template <typename Emit>
    const BusyIntervalStats &finish(Emit &&emit) {
        close_segment(emit);
        return stats_;
    }

private:
    enum class LaneState : uint8_t { Unknown, Idle, Busy };

    template <typename Emit>
    void close_segment(Emit &&emit) {
        if (!active_) return;
        for (uint8_t lane = 0; lane < lanes_.size(); ++lane) {
            if (lanes_[lane] != LaneState::Busy) continue;
            emit(
                BusyInterval{
                    anchor_syscnt_, starts_[lane], last_cycles_, block_id_, lane,
                    static_cast<uint8_t>((start_clipped_[lane] ? 1u : 0u) | 2u)
                }
            );
            ++stats_.emitted;
            ++stats_.clipped_end;
        }
        lanes_.fill(LaneState::Unknown);
        active_ = false;
    }

    std::array<LaneState, 7> lanes_{};
    std::array<uint64_t, 7> starts_{};
    std::array<bool, 7> start_clipped_{};
    uint64_t anchor_syscnt_{0};
    uint64_t last_cycles_{0};
    uint16_t block_id_{0};
    bool active_{false};
    BusyIntervalStats stats_{};
};

}  // namespace simpler::dfx::l0

// Convert closed raw L0 files into structured per-core records. Capture and
// transport have already completed; a conversion failure does not change them.
bool export_l0_event_records(
    const std::filesystem::path &run_dir, uint64_t run_epoch, uint64_t aic_frequency_hz, uint64_t aiv_frequency_hz,
    const std::vector<uint32_t> &physical_core_types
);
