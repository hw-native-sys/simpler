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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_MESSAGE_QUEUE_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_MESSAGE_QUEUE_H_

#include <stddef.h>
#include <stdint.h>
#include <new>
#include <string.h>

#include "aicpu/l3_l2_orch_endpoint.h"

static constexpr uint32_t L3L2_QUEUE_MAGIC = 0x4C335132u;  // "L3Q2"
static constexpr uint16_t L3L2_QUEUE_ABI_MAJOR = 1;
static constexpr uint16_t L3L2_QUEUE_ABI_MINOR = 1;
static constexpr uint64_t L3L2_QUEUE_DESC_SLOT_BYTES = 32;
static constexpr uint64_t L3L2_QUEUE_DESC_RING_ALIGNMENT = 8;
static constexpr uint64_t L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT = 64;
static constexpr uint64_t L3L2_QUEUE_COUNTER_STRIDE = 64;
static constexpr uint64_t L3L2_QUEUE_INPUT_DESC_TAIL_OFFSET = 0;
static constexpr uint64_t L3L2_QUEUE_INPUT_DESC_HEAD_OFFSET = 64;
static constexpr uint64_t L3L2_QUEUE_OUTPUT_DESC_TAIL_OFFSET = 128;
static constexpr uint64_t L3L2_QUEUE_OUTPUT_DESC_HEAD_OFFSET = 192;
static constexpr uint64_t L3L2_QUEUE_L3_ABORT_FLAG_OFFSET = 256;
static constexpr uint64_t L3L2_QUEUE_L2_ABORT_FLAG_OFFSET = 320;
static constexpr uint64_t L3L2_QUEUE_COUNTER_BYTES = 384;
static constexpr uint64_t L3L2_QUEUE_MAX_DEPTH = 1ull << 30;
static constexpr uint64_t L3L2_QUEUE_MAGIC_VERSION =
    l3_l2_orch_comm_pack_magic_version(L3L2_QUEUE_MAGIC, L3L2_QUEUE_ABI_MAJOR, L3L2_QUEUE_ABI_MINOR);

struct L3L2QueueDescSlot {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};

static_assert(sizeof(L3L2QueueDescSlot) == L3L2_QUEUE_DESC_SLOT_BYTES, "L3L2QueueDescSlot ABI size changed");
static_assert(offsetof(L3L2QueueDescSlot, seq) == 0, "L3L2QueueDescSlot::seq offset changed");
static_assert(offsetof(L3L2QueueDescSlot, opcode) == 8, "L3L2QueueDescSlot::opcode offset changed");
static_assert(offsetof(L3L2QueueDescSlot, payload_offset) == 16, "L3L2QueueDescSlot::payload_offset changed");
static_assert(offsetof(L3L2QueueDescSlot, payload_nbytes) == 24, "L3L2QueueDescSlot::payload_nbytes changed");

enum class L3L2QueueOpcode : uint64_t {
    INVALID = 0,
    DATA = 1,
    STOP = 2,
    ERROR = 3,
};

enum class L3L2QueueErrorKind : uint32_t {
    NONE = 0,
    BAD_ARGUMENT = 1,
    BAD_DESCRIPTOR = 2,
    INVALID_DESCRIPTOR = 3,
    OUT_OF_SPACE = 4,
    OWNERSHIP = 5,
    REMOTE_ABORTED = 6,
    ENDPOINT_ERROR = 7,
};

enum class L3L2QueueTimeoutStatus : uint32_t {
    ORDINARY_TIMEOUT = 0,
    REMOTE_ABORTED = 1,
};

enum class L3L2QueueOp : uint32_t {
    INIT = 1,
    TIMEOUT = 2,
    INPUT_TRY_PEEK = 3,
    INPUT_RELEASE = 4,
    OUTPUT_TRY_RESERVE = 5,
    OUTPUT_PUBLISH = 6,
};

inline const char *l3_l2_queue_op_to_string(L3L2QueueOp op) {
    switch (op) {
    case L3L2QueueOp::INIT:
        return "init";
    case L3L2QueueOp::TIMEOUT:
        return "timeout";
    case L3L2QueueOp::INPUT_TRY_PEEK:
        return "input.try_peek";
    case L3L2QueueOp::INPUT_RELEASE:
        return "input.release";
    case L3L2QueueOp::OUTPUT_TRY_RESERVE:
        return "output.try_reserve";
    case L3L2QueueOp::OUTPUT_PUBLISH:
        return "output.publish";
    default:
        return "unknown";
    }
}

struct L3L2QueueError {
    L3L2QueueErrorKind kind;
    L3L2QueueOp op;
    uint64_t region_id;
    char message[256];
};

struct L3L2QueueLayout {
    uint64_t depth;
    uint64_t input_desc_offset;
    uint64_t output_desc_offset;
    uint64_t input_arena_offset;
    uint64_t output_arena_offset;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t payload_bytes;
    uint64_t input_desc_tail_offset;
    uint64_t input_desc_head_offset;
    uint64_t output_desc_tail_offset;
    uint64_t output_desc_head_offset;
    uint64_t l3_abort_flag_offset;
    uint64_t l2_abort_flag_offset;
    uint64_t counter_bytes;
};

struct L3L2QueueArgs {
    uint64_t magic_version;
    uint64_t depth;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t payload_bytes;
    uint64_t counter_bytes;
};

struct L3L2QueueInputHandle {
    uint64_t seq;
    L3L2QueueOpcode opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
    L3L2OrchPayloadView payload;
};

struct L3L2QueueOutputReservation {
    uint64_t seq;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
    L3L2OrchPayloadView payload;
    bool valid;
};

static inline uint64_t l3_l2_queue_magic_version() { return L3L2_QUEUE_MAGIC_VERSION; }

static inline bool l3_l2_queue_is_power_of_two(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

static inline uint64_t l3_l2_queue_align_up(uint64_t value, uint64_t align) {
    if (align == 0) {
        return value;
    }
    uint64_t remainder = value % align;
    return remainder == 0 ? value : value + (align - remainder);
}

static inline bool l3_l2_queue_align_up_checked(uint64_t value, uint64_t align, uint64_t *out) {
    if (out == nullptr || align == 0) {
        return false;
    }
    uint64_t remainder = value % align;
    uint64_t bump = remainder == 0 ? 0 : align - remainder;
    if (l3_l2_orch_comm_add_overflows(value, bump)) {
        return false;
    }
    *out = value + bump;
    return true;
}

static inline bool l3_l2_queue_valid_opcode(L3L2QueueOpcode opcode) {
    return opcode == L3L2QueueOpcode::DATA || opcode == L3L2QueueOpcode::STOP || opcode == L3L2QueueOpcode::ERROR;
}

static inline bool
l3_l2_queue_make_layout(uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes, L3L2QueueLayout &out) {
    if (!l3_l2_queue_is_power_of_two(depth) || depth > L3L2_QUEUE_MAX_DEPTH || input_arena_bytes == 0 ||
        output_arena_bytes == 0 || input_arena_bytes % L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT != 0 ||
        output_arena_bytes % L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT != 0) {
        return false;
    }

    uint64_t desc_ring_bytes = depth * L3L2_QUEUE_DESC_SLOT_BYTES;
    uint64_t input_desc_offset = 0;
    if (l3_l2_orch_comm_add_overflows(input_desc_offset, desc_ring_bytes)) {
        return false;
    }
    uint64_t output_desc_offset = input_desc_offset + desc_ring_bytes;
    if (l3_l2_orch_comm_add_overflows(output_desc_offset, desc_ring_bytes)) {
        return false;
    }
    uint64_t desc_end = output_desc_offset + desc_ring_bytes;
    uint64_t input_arena_offset = 0;
    if (!l3_l2_queue_align_up_checked(desc_end, L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT, &input_arena_offset)) {
        return false;
    }
    if (l3_l2_orch_comm_add_overflows(input_arena_offset, input_arena_bytes)) {
        return false;
    }
    uint64_t input_arena_end = input_arena_offset + input_arena_bytes;
    uint64_t output_arena_offset = 0;
    if (!l3_l2_queue_align_up_checked(input_arena_end, L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT, &output_arena_offset)) {
        return false;
    }
    if (l3_l2_orch_comm_add_overflows(output_arena_offset, output_arena_bytes)) {
        return false;
    }
    uint64_t payload_bytes = output_arena_offset + output_arena_bytes;

    out = L3L2QueueLayout{
        depth,
        input_desc_offset,
        output_desc_offset,
        input_arena_offset,
        output_arena_offset,
        input_arena_bytes,
        output_arena_bytes,
        payload_bytes,
        L3L2_QUEUE_INPUT_DESC_TAIL_OFFSET,
        L3L2_QUEUE_INPUT_DESC_HEAD_OFFSET,
        L3L2_QUEUE_OUTPUT_DESC_TAIL_OFFSET,
        L3L2_QUEUE_OUTPUT_DESC_HEAD_OFFSET,
        L3L2_QUEUE_L3_ABORT_FLAG_OFFSET,
        L3L2_QUEUE_L2_ABORT_FLAG_OFFSET,
        L3L2_QUEUE_COUNTER_BYTES,
    };
    return output_desc_offset % L3L2_QUEUE_DESC_RING_ALIGNMENT == 0 &&
           input_arena_offset % L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT == 0 &&
           output_arena_offset % L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT == 0;
}

static inline bool
l3_l2_queue_validate_region(const L3L2OrchRegionDesc &desc, const L3L2QueueArgs &args, L3L2QueueLayout *out_layout) {
    L3L2QueueLayout layout{};
    if (args.magic_version != l3_l2_queue_magic_version() ||
        l3_l2_orch_comm_validate_desc(desc) != L3L2OrchCommValidationError::OK ||
        !l3_l2_queue_make_layout(args.depth, args.input_arena_bytes, args.output_arena_bytes, layout)) {
        return false;
    }
    if (args.payload_bytes != layout.payload_bytes || args.counter_bytes != layout.counter_bytes ||
        desc.payload_bytes != layout.payload_bytes || desc.counter_bytes != layout.counter_bytes) {
        return false;
    }
    if (out_layout != nullptr) {
        *out_layout = layout;
    }
    return true;
}

static inline void l3_l2_queue_encode_desc(
    L3L2QueueDescSlot *slot, uint64_t seq, L3L2QueueOpcode opcode, uint64_t payload_offset, uint64_t payload_nbytes
) {
    if (slot == nullptr) {
        return;
    }
    slot->seq = seq;
    slot->opcode = static_cast<uint64_t>(opcode);
    slot->payload_offset = payload_offset;
    slot->payload_nbytes = payload_nbytes;
}

static inline bool l3_l2_queue_reconstruct_counter(int32_t observed_low32, uint64_t depth, uint64_t &local_value) {
    if (depth > L3L2_QUEUE_MAX_DEPTH) {
        return false;
    }
    uint32_t local_low32 = static_cast<uint32_t>(local_value);
    int32_t delta = static_cast<int32_t>(static_cast<uint32_t>(observed_low32) - local_low32);
    if (delta < 0 || static_cast<uint64_t>(delta) > depth) {
        return false;
    }
    local_value += static_cast<uint64_t>(delta);
    return true;
}

namespace l3_l2_message_queue {

static inline uint64_t magic_version() { return ::l3_l2_queue_magic_version(); }

static inline bool is_power_of_two(uint64_t value) { return ::l3_l2_queue_is_power_of_two(value); }

static inline uint64_t align_up(uint64_t value, uint64_t align) { return ::l3_l2_queue_align_up(value, align); }

static inline bool align_up_checked(uint64_t value, uint64_t align, uint64_t *out) {
    return ::l3_l2_queue_align_up_checked(value, align, out);
}

static inline bool valid_opcode(L3L2QueueOpcode opcode) { return ::l3_l2_queue_valid_opcode(opcode); }

static inline bool
make_layout(uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes, L3L2QueueLayout &out) {
    return ::l3_l2_queue_make_layout(depth, input_arena_bytes, output_arena_bytes, out);
}

static inline bool
validate_region(const L3L2OrchRegionDesc &desc, const L3L2QueueArgs &args, L3L2QueueLayout *out_layout) {
    return ::l3_l2_queue_validate_region(desc, args, out_layout);
}

static inline void encode_desc(
    L3L2QueueDescSlot *slot, uint64_t seq, L3L2QueueOpcode opcode, uint64_t payload_offset, uint64_t payload_nbytes
) {
    ::l3_l2_queue_encode_desc(slot, seq, opcode, payload_offset, payload_nbytes);
}

static inline bool reconstruct_counter(int32_t observed_low32, uint64_t depth, uint64_t &local_value) {
    return ::l3_l2_queue_reconstruct_counter(observed_low32, depth, local_value);
}

}  // namespace l3_l2_message_queue

template <uint64_t MaxInflight = 1>
class L3L2QueueEndpoint {
    static_assert(MaxInflight > 0, "MaxInflight must be positive");

public:
    static constexpr uint64_t kStopEntrySlots = 1;
    static constexpr uint64_t kEntryCapacity = MaxInflight + kStopEntrySlots;

    class InputQueue {
        struct ActiveInputEntry {
            uint64_t seq;
            L3L2QueueOpcode opcode;
            uint64_t payload_offset;
            uint64_t payload_nbytes;
            L3L2OrchPayloadView payload;
            bool completed;
        };

    public:
        explicit InputQueue(L3L2QueueEndpoint *parent) :
            parent_(parent) {}

        InputQueue(const InputQueue &) = delete;
        InputQueue &operator=(const InputQueue &) = delete;
        InputQueue(InputQueue &&) = delete;
        InputQueue &operator=(InputQueue &&) = delete;

        friend class L3L2QueueEndpoint<MaxInflight>;

    private:
        bool initialize() {
            for (uint64_t i = 0; i < kEntryCapacity; ++i) {
                active_entries_[i] = ActiveInputEntry{};
            }
            active_head_ = 0;
            active_count_ = 0;
            active_non_stop_count_ = 0;
            input_head_ = 0;
            input_tail_ = 0;
            input_payload_head_ = 0;
            input_payload_acquire_head_ = 0;
            input_acquire_ = input_head_;
            stop_observed_ = false;
            drained_ = false;
            return true;
        }

    public:
        bool peek(uint64_t timeout_ns, L3L2QueueInputHandle &out) {
            uint64_t start = device_time_now_ticks();
            uint64_t frequency_hz = device_time_frequency_hz();
            uint64_t spins = 0;
            while (true) {
                if (try_peek(out)) {
                    return true;
                }
                if (parent_->error_.kind != L3L2QueueErrorKind::NONE) {
                    return false;
                }
                spins += 1;
                if (timeout_ns == 0 || (spins & 1023ull) == 0) {
                    uint64_t now = device_time_now_ticks();
                    if (timeout_ns == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout_ns) {
                        parent_->disambiguate_timeout();
                        return false;
                    }
                }
            }
        }

        bool try_peek(L3L2QueueInputHandle &out) {
            out = L3L2QueueInputHandle{0, L3L2QueueOpcode::INVALID, 0, 0, L3L2OrchPayloadView{0, 0}};
            if (!parent_->ensure_live()) {
                return false;
            }
            const L3L2QueueLayout &layout = parent_->layout_;
            L3L2OrchEndpoint &endpoint = parent_->endpoint_;
            if constexpr (MaxInflight == 1) {
                if (active_count_ != 0) {
                    parent_->poison(
                        L3L2QueueErrorKind::OWNERSHIP, L3L2QueueOp::INPUT_TRY_PEEK, "input handle already active"
                    );
                    return false;
                }
            }
            if (!parent_->refresh_counter(
                    layout.input_desc_tail_offset, input_tail_, layout.depth, L3L2QueueOp::INPUT_TRY_PEEK
                )) {
                return false;
            }
            if (stop_observed_) {
                if (input_tail_ != input_acquire_) {
                    parent_->poison(
                        L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK,
                        "input descriptor published after STOP"
                    );
                }
                return false;
            }
            if (input_tail_ == input_acquire_) {
                return false;
            }
            if (input_tail_ - input_head_ > layout.depth || input_acquire_ < input_head_ ||
                input_acquire_ > input_tail_) {
                parent_->poison(
                    L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK,
                    "input descriptor state invalid"
                );
                return false;
            }

            L3L2QueueDescSlot slot{};
            uint64_t slot_index = input_acquire_ & (layout.depth - 1);
            uint64_t slot_offset = layout.input_desc_offset + slot_index * sizeof(L3L2QueueDescSlot);
            if (!parent_->read_desc_slot(slot_offset, &slot, L3L2QueueOp::INPUT_TRY_PEEK)) {
                return false;
            }
            uint64_t expected_seq = input_acquire_ + 1;
            if (slot.seq != expected_seq) {
                parent_->poison(
                    L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK, "input descriptor seq mismatch"
                );
                return false;
            }
            L3L2QueueOpcode opcode = static_cast<L3L2QueueOpcode>(slot.opcode);
            if (!l3_l2_message_queue::valid_opcode(opcode)) {
                parent_->poison(
                    L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK, "invalid input opcode"
                );
                return false;
            }
            if (opcode == L3L2QueueOpcode::STOP && (slot.payload_offset != 0 || slot.payload_nbytes != 0)) {
                parent_->poison(
                    L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK,
                    "STOP descriptor must be zero-byte"
                );
                return false;
            }
            bool counts_against_window = opcode == L3L2QueueOpcode::DATA || opcode == L3L2QueueOpcode::ERROR;
            if (counts_against_window && active_non_stop_count_ >= MaxInflight) {
                return false;
            }
            if (active_count_ >= kEntryCapacity) {
                parent_->poison(L3L2QueueErrorKind::OWNERSHIP, L3L2QueueOp::INPUT_TRY_PEEK, "input window state full");
                return false;
            }

            L3L2OrchPayloadView view{0, 0};
            if (slot.payload_nbytes == 0) {
                if (slot.payload_offset != 0) {
                    parent_->poison(
                        L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK,
                        "zero-byte descriptor uses nonzero payload offset"
                    );
                    return false;
                }
            } else if (!parent_->payload_in_arena(
                           slot.payload_offset, slot.payload_nbytes, layout.input_arena_offset, layout.input_arena_bytes
                       )) {
                parent_->poison(
                    L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK, "input payload out of arena"
                );
                return false;
            } else if (!parent_->payload_matches_head(
                           input_payload_acquire_head_, slot.payload_offset, slot.payload_nbytes,
                           layout.input_arena_offset, layout.input_arena_bytes, L3L2QueueOp::INPUT_TRY_PEEK
                       )) {
                return false;
            } else if (!endpoint.payload_read(slot.payload_offset, slot.payload_nbytes, view)) {
                parent_->poison(
                    L3L2QueueErrorKind::ENDPOINT_ERROR, L3L2QueueOp::INPUT_TRY_PEEK, endpoint.error().message
                );
                return false;
            } else {
                parent_->advance_payload_head(
                    input_payload_acquire_head_, slot.payload_offset, slot.payload_nbytes, layout.input_arena_offset,
                    layout.input_arena_bytes, L3L2QueueOp::INPUT_TRY_PEEK
                );
                if (parent_->error_.kind != L3L2QueueErrorKind::NONE) {
                    return false;
                }
            }

            out = L3L2QueueInputHandle{slot.seq, opcode, slot.payload_offset, slot.payload_nbytes, view};
            uint64_t insert_index = (active_head_ + active_count_) % kEntryCapacity;
            active_entries_[insert_index] =
                ActiveInputEntry{slot.seq, opcode, slot.payload_offset, slot.payload_nbytes, view, false};
            active_count_ += 1;
            if (counts_against_window) {
                active_non_stop_count_ += 1;
            }
            input_acquire_ += 1;
            if (opcode == L3L2QueueOpcode::STOP) {
                stop_observed_ = true;
                if (input_tail_ != input_acquire_) {
                    parent_->poison(
                        L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::INPUT_TRY_PEEK,
                        "input descriptor published after STOP"
                    );
                    return false;
                }
            }
            return true;
        }

        bool release(const L3L2QueueInputHandle &handle) {
            if (!parent_->ensure_live()) {
                return false;
            }
            ActiveInputEntry *entry = entry_for_seq(handle.seq);
            if (entry == nullptr || handle.opcode != entry->opcode || handle.payload_offset != entry->payload_offset ||
                handle.payload_nbytes != entry->payload_nbytes || handle.payload.gm_addr != entry->payload.gm_addr ||
                handle.payload.nbytes != entry->payload.nbytes) {
                parent_->poison(
                    L3L2QueueErrorKind::OWNERSHIP, L3L2QueueOp::INPUT_RELEASE, "input handle is not active"
                );
                return false;
            }
            if (entry->completed) {
                parent_->poison(
                    L3L2QueueErrorKind::OWNERSHIP, L3L2QueueOp::INPUT_RELEASE, "input handle already released"
                );
                return false;
            }
            entry->completed = true;
            return release_completed_prefix();
        }

        bool drained() const { return drained_; }

    private:
        ActiveInputEntry *entry_for_seq(uint64_t seq) {
            uint64_t first_seq = input_head_ + 1;
            if (seq < first_seq) {
                return nullptr;
            }
            uint64_t ordinal = seq - first_seq;
            if (ordinal >= active_count_) {
                return nullptr;
            }
            uint64_t index = (active_head_ + ordinal) % kEntryCapacity;
            return active_entries_[index].seq == seq ? &active_entries_[index] : nullptr;
        }

        bool release_completed_prefix() {
            while (active_count_ != 0 && active_entries_[active_head_].completed) {
                ActiveInputEntry entry = active_entries_[active_head_];
                if (entry.payload_nbytes != 0) {
                    parent_->advance_payload_head(
                        input_payload_head_, entry.payload_offset, entry.payload_nbytes,
                        parent_->layout_.input_arena_offset, parent_->layout_.input_arena_bytes,
                        L3L2QueueOp::INPUT_RELEASE
                    );
                    if (parent_->error_.kind != L3L2QueueErrorKind::NONE) {
                        return false;
                    }
                }
                input_head_ += 1;
                if (entry.opcode == L3L2QueueOpcode::DATA || entry.opcode == L3L2QueueOpcode::ERROR) {
                    active_non_stop_count_ -= 1;
                }
                if (entry.opcode == L3L2QueueOpcode::STOP) {
                    drained_ = true;
                }
                active_entries_[active_head_] = ActiveInputEntry{};
                active_head_ = (active_head_ + 1) % kEntryCapacity;
                active_count_ -= 1;
                if (!parent_->notify_counter(
                        parent_->layout_.input_desc_head_offset, static_cast<int32_t>(input_head_),
                        L3L2QueueOp::INPUT_RELEASE
                    )) {
                    return false;
                }
            }
            return true;
        }

        L3L2QueueEndpoint *parent_;
        ActiveInputEntry active_entries_[kEntryCapacity]{};
        uint64_t active_head_{0};
        uint64_t active_count_{0};
        uint64_t active_non_stop_count_{0};
        uint64_t input_head_{0};
        uint64_t input_tail_{0};
        uint64_t input_payload_head_{0};
        uint64_t input_payload_acquire_head_{0};
        uint64_t input_acquire_{0};
        bool stop_observed_{false};
        bool drained_{false};
    };

    class OutputQueue {
    public:
        explicit OutputQueue(L3L2QueueEndpoint *parent) :
            parent_(parent) {}

        bool initialize() {
            output_head_ = 0;
            output_tail_ = 0;
            output_payload_head_ = 0;
            output_payload_tail_ = 0;
            reservation_active_ = false;
            reservation_seq_ = 0;
            reservation_offset_ = 0;
            reservation_nbytes_ = 0;
            return true;
        }

        bool reserve(uint64_t nbytes, uint64_t timeout_ns, L3L2QueueOutputReservation &out) {
            uint64_t start = device_time_now_ticks();
            uint64_t frequency_hz = device_time_frequency_hz();
            uint64_t spins = 0;
            while (true) {
                if (try_reserve(nbytes, out)) {
                    return true;
                }
                if (parent_->error_.kind != L3L2QueueErrorKind::NONE) {
                    return false;
                }
                spins += 1;
                if (timeout_ns == 0 || (spins & 1023ull) == 0) {
                    uint64_t now = device_time_now_ticks();
                    if (timeout_ns == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout_ns) {
                        parent_->disambiguate_timeout();
                        return false;
                    }
                }
            }
        }

        bool try_reserve(uint64_t nbytes, L3L2QueueOutputReservation &out) {
            out = L3L2QueueOutputReservation{0, 0, 0, L3L2OrchPayloadView{0, 0}, false};
            if (!parent_->ensure_live()) {
                return false;
            }
            const L3L2QueueLayout &layout = parent_->layout_;
            if (reservation_active_) {
                parent_->poison(
                    L3L2QueueErrorKind::OWNERSHIP, L3L2QueueOp::OUTPUT_TRY_RESERVE, "output reservation already active"
                );
                return false;
            }
            if (nbytes > layout.output_arena_bytes) {
                return false;
            }
            uint64_t old_head = output_head_;
            if (!parent_->refresh_counter(
                    layout.output_desc_head_offset, output_head_, layout.depth, L3L2QueueOp::OUTPUT_TRY_RESERVE
                )) {
                return false;
            }
            if (output_head_ != old_head &&
                !replay_output_releases(old_head, output_head_, L3L2QueueOp::OUTPUT_TRY_RESERVE)) {
                return false;
            }
            if (output_tail_ - output_head_ >= layout.depth) {
                return false;
            }

            uint64_t payload_offset = 0;
            L3L2OrchPayloadView view{0, 0};
            if (nbytes != 0) {
                uint64_t arena_base = layout.output_arena_offset;
                uint64_t arena_bytes = layout.output_arena_bytes;
                uint64_t arena_pos = output_payload_tail_ % arena_bytes;
                if (arena_pos + nbytes > arena_bytes) {
                    // Payloads are never split across arena wrap. The skipped tail bytes are retired in the
                    // monotonic virtual cursor even if this reservation later finds the arena full.
                    output_payload_tail_ += arena_bytes - arena_pos;
                    arena_pos = 0;
                }
                if (output_payload_tail_ + nbytes - output_payload_head_ > arena_bytes) {
                    return false;
                }
                payload_offset = arena_base + arena_pos;
                view = L3L2OrchPayloadView{parent_->endpoint_.descriptor().payload_base + payload_offset, nbytes};
                output_payload_tail_ += nbytes;
            }

            reservation_active_ = true;
            reservation_seq_ = output_tail_ + 1;
            reservation_offset_ = payload_offset;
            reservation_nbytes_ = nbytes;
            out = L3L2QueueOutputReservation{reservation_seq_, payload_offset, nbytes, view, true};
            return true;
        }

        bool publish(const L3L2QueueOutputReservation &reservation, L3L2QueueOpcode opcode) {
            if (!parent_->ensure_live()) {
                return false;
            }
            if (!reservation_active_ || !reservation.valid || reservation.seq != reservation_seq_ ||
                reservation.payload_offset != reservation_offset_ ||
                reservation.payload_nbytes != reservation_nbytes_) {
                parent_->poison(
                    L3L2QueueErrorKind::OWNERSHIP, L3L2QueueOp::OUTPUT_PUBLISH, "unknown output reservation"
                );
                return false;
            }
            if (opcode == L3L2QueueOpcode::STOP || !l3_l2_message_queue::valid_opcode(opcode)) {
                parent_->poison(
                    L3L2QueueErrorKind::INVALID_DESCRIPTOR, L3L2QueueOp::OUTPUT_PUBLISH, "invalid output opcode"
                );
                return false;
            }
            L3L2QueueDescSlot slot{};
            l3_l2_message_queue::encode_desc(&slot, 0, opcode, reservation.payload_offset, reservation.payload_nbytes);
            uint64_t slot_index = output_tail_ & (parent_->layout_.depth - 1);
            uint64_t slot_offset = parent_->layout_.output_desc_offset + slot_index * sizeof(L3L2QueueDescSlot);
            if (!parent_->write_desc_slot(slot_offset, slot, reservation.seq, L3L2QueueOp::OUTPUT_PUBLISH)) {
                return false;
            }
            output_tail_ += 1;
            reservation_active_ = false;
            reservation_seq_ = 0;
            reservation_offset_ = 0;
            reservation_nbytes_ = 0;
            return parent_->notify_counter(
                parent_->layout_.output_desc_tail_offset, static_cast<int32_t>(output_tail_),
                L3L2QueueOp::OUTPUT_PUBLISH
            );
        }

    private:
        bool replay_output_releases(uint64_t old_head, uint64_t new_head, L3L2QueueOp op) {
            uint64_t cursor = old_head;
            while (cursor < new_head) {
                L3L2QueueDescSlot slot{};
                uint64_t slot_index = cursor & (parent_->layout_.depth - 1);
                uint64_t slot_offset = parent_->layout_.output_desc_offset + slot_index * sizeof(L3L2QueueDescSlot);
                if (!parent_->read_desc_slot(slot_offset, &slot, op)) {
                    return false;
                }
                if (slot.seq != cursor + 1) {
                    parent_->poison(L3L2QueueErrorKind::INVALID_DESCRIPTOR, op, "output release replay seq mismatch");
                    return false;
                }
                if (slot.payload_nbytes != 0) {
                    parent_->advance_payload_head(
                        output_payload_head_, slot.payload_offset, slot.payload_nbytes,
                        parent_->layout_.output_arena_offset, parent_->layout_.output_arena_bytes, op
                    );
                    if (parent_->error_.kind != L3L2QueueErrorKind::NONE) {
                        return false;
                    }
                }
                cursor += 1;
            }
            return true;
        }

        L3L2QueueEndpoint *parent_;
        uint64_t output_head_{0};
        uint64_t output_tail_{0};
        uint64_t output_payload_head_{0};
        uint64_t output_payload_tail_{0};
        bool reservation_active_{false};
        uint64_t reservation_seq_{0};
        uint64_t reservation_offset_{0};
        uint64_t reservation_nbytes_{0};
    };

    L3L2QueueEndpoint(const L3L2OrchRegionDesc &desc, const L3L2QueueArgs &args) :
        endpoint_(desc),
        input_queue_(this),
        output_queue_(this) {
        if (endpoint_.error().kind != L3L2EndpointErrorKind::NONE ||
            !l3_l2_message_queue::validate_region(desc, args, &layout_)) {
            set_error(
                L3L2QueueErrorKind::BAD_DESCRIPTOR, L3L2QueueOp::INIT, desc.region_id, "invalid queue descriptor"
            );
            return;
        }
        if (MaxInflight > layout_.depth) {
            set_error(L3L2QueueErrorKind::BAD_ARGUMENT, L3L2QueueOp::INIT, desc.region_id, "invalid input window");
            return;
        }
        input_queue_.initialize();
        output_queue_.initialize();
    }

    L3L2QueueEndpoint(const L3L2QueueEndpoint &) = delete;
    L3L2QueueEndpoint &operator=(const L3L2QueueEndpoint &) = delete;
    L3L2QueueEndpoint(L3L2QueueEndpoint &&) = delete;
    L3L2QueueEndpoint &operator=(L3L2QueueEndpoint &&) = delete;

    const L3L2QueueError &error() const { return error_; }
    const L3L2QueueLayout &layout() const { return layout_; }
    InputQueue &input() { return input_queue_; }
    OutputQueue &output() { return output_queue_; }

    L3L2QueueTimeoutStatus disambiguate_timeout() {
        if (error_.kind != L3L2QueueErrorKind::NONE) {
            return error_.kind == L3L2QueueErrorKind::REMOTE_ABORTED ? L3L2QueueTimeoutStatus::REMOTE_ABORTED :
                                                                       L3L2QueueTimeoutStatus::ORDINARY_TIMEOUT;
        }
        L3L2OrchSignalTestResult result{};
        uint64_t addr = 0;
        if (!endpoint_.counter_addr(layout_.l3_abort_flag_offset, addr) ||
            !endpoint_.signal_test(addr, 1, L3L2OrchWaitCmp::GE, result)) {
            poison(L3L2QueueErrorKind::ENDPOINT_ERROR, L3L2QueueOp::TIMEOUT, endpoint_.error().message);
            return L3L2QueueTimeoutStatus::ORDINARY_TIMEOUT;
        }
        if (result.matched) {
            set_error(
                L3L2QueueErrorKind::REMOTE_ABORTED, L3L2QueueOp::TIMEOUT, endpoint_.descriptor().region_id,
                "remote abort"
            );
            return L3L2QueueTimeoutStatus::REMOTE_ABORTED;
        }
        return L3L2QueueTimeoutStatus::ORDINARY_TIMEOUT;
    }

private:
    bool ensure_live() {
        if (error_.kind == L3L2QueueErrorKind::NONE) {
            return true;
        }
        return false;
    }

    void set_error(L3L2QueueErrorKind kind, L3L2QueueOp op, uint64_t region_id, const char *message) {
        if (error_.kind != L3L2QueueErrorKind::NONE) {
            return;
        }
        error_ = L3L2QueueError{kind, op, region_id, ""};
        l3_l2_orch_comm::copy_error_message(error_.message, sizeof(error_.message), message);
    }

    void poison(L3L2QueueErrorKind kind, L3L2QueueOp op, const char *message) {
        set_error(kind, op, endpoint_.descriptor().region_id, message);
        if (kind != L3L2QueueErrorKind::REMOTE_ABORTED) {
            uint64_t addr = 0;
            if (endpoint_.counter_addr(layout_.l2_abort_flag_offset, addr)) {
                endpoint_.signal_notify(addr, 1, L3L2OrchNotifyOp::Set);
            }
        }
    }

    bool notify_counter(uint64_t offset, int32_t value, L3L2QueueOp op) {
        uint64_t addr = 0;
        if (!endpoint_.counter_addr(offset, addr) || !endpoint_.signal_notify(addr, value, L3L2OrchNotifyOp::Set)) {
            poison(L3L2QueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        return true;
    }

    bool refresh_counter(uint64_t offset, uint64_t &local, uint64_t depth, L3L2QueueOp op) {
        uint64_t addr = 0;
        L3L2OrchSignalTestResult result{};
        if (!endpoint_.counter_addr(offset, addr) ||
            !endpoint_.signal_test(addr, static_cast<int32_t>(local), L3L2OrchWaitCmp::NE, result)) {
            poison(L3L2QueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        if (!result.matched) {
            return true;
        }
        if (!l3_l2_message_queue::reconstruct_counter(result.observed, depth, local)) {
            poison(L3L2QueueErrorKind::INVALID_DESCRIPTOR, op, "counter reconstruction failed");
            return false;
        }
        return true;
    }

    bool read_desc_slot(uint64_t slot_offset, L3L2QueueDescSlot *slot, L3L2QueueOp op) {
        L3L2OrchPayloadView view{};
        if (!endpoint_.payload_read(slot_offset, sizeof(L3L2QueueDescSlot), view)) {
            poison(L3L2QueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        memcpy(slot, reinterpret_cast<const void *>(static_cast<uintptr_t>(view.gm_addr)), sizeof(L3L2QueueDescSlot));
        return true;
    }

    bool write_desc_slot(uint64_t slot_offset, const L3L2QueueDescSlot &slot, uint64_t seq, L3L2QueueOp op) {
        L3L2QueueDescSlot fields = slot;
        fields.seq = 0;
        if (!endpoint_.payload_write(slot_offset + offsetof(L3L2QueueDescSlot, opcode), &fields.opcode, 24)) {
            poison(L3L2QueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        if (!endpoint_.payload_write(slot_offset + offsetof(L3L2QueueDescSlot, seq), &seq, sizeof(seq))) {
            poison(L3L2QueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        return true;
    }

    static bool payload_in_arena(uint64_t offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes) {
        if (nbytes == 0 || l3_l2_orch_comm::add_overflows(offset, nbytes)) {
            return false;
        }
        return offset >= arena_offset && offset + nbytes <= arena_offset + arena_bytes;
    }

    static uint64_t
    payload_expected_offset(uint64_t cursor, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes) {
        uint64_t arena_pos = cursor % arena_bytes;
        return arena_pos + nbytes > arena_bytes ? arena_offset : arena_offset + arena_pos;
    }

    bool payload_matches_head(
        uint64_t cursor, uint64_t payload_offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes,
        L3L2QueueOp op
    ) {
        if (nbytes == 0) {
            return true;
        }
        uint64_t expected_offset = payload_expected_offset(cursor, nbytes, arena_offset, arena_bytes);
        if (payload_offset != expected_offset) {
            poison(L3L2QueueErrorKind::INVALID_DESCRIPTOR, op, "payload replay offset mismatch");
            return false;
        }
        return true;
    }

    void advance_payload_head(
        uint64_t &cursor, uint64_t payload_offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes,
        L3L2QueueOp op
    ) {
        uint64_t arena_pos = cursor % arena_bytes;
        uint64_t expected_offset = payload_expected_offset(cursor, nbytes, arena_offset, arena_bytes);
        if (expected_offset != payload_offset) {
            poison(L3L2QueueErrorKind::INVALID_DESCRIPTOR, op, "payload replay offset mismatch");
            return;
        }
        if (arena_pos + nbytes > arena_bytes) {
            cursor += arena_bytes - (cursor % arena_bytes);
        }
        cursor += nbytes;
    }

    L3L2OrchEndpoint endpoint_;
    L3L2QueueLayout layout_{};
    L3L2QueueError error_{L3L2QueueErrorKind::NONE, L3L2QueueOp::INIT, 0, ""};
    InputQueue input_queue_;
    OutputQueue output_queue_;
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_L3_L2_MESSAGE_QUEUE_H_
