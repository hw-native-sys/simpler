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

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <type_traits>
#include <utility>

#include "common/region_instance_semantics.h"

namespace spsc_queue {

inline constexpr uint32_t kSpscQueueMagic = 0x53505351u;
inline constexpr uint16_t kSpscQueueAbiMajor = 1;
inline constexpr uint16_t kSpscQueueAbiMinor = 0;
inline constexpr uint64_t kSpscQueueMagicVersion = (static_cast<uint64_t>(kSpscQueueMagic) << 32) |
                                                   (static_cast<uint64_t>(kSpscQueueAbiMajor) << 16) |
                                                   kSpscQueueAbiMinor;
inline constexpr size_t kSpscQueueEndpointBindingScalarCount = 10;
inline constexpr uint64_t kDescriptorBytes = 32;
inline constexpr uint64_t kArenaAlignment = 64;
inline constexpr uint64_t kDescriptorRingAlignment = 8;
inline constexpr uint64_t kCounterStride = 64;
inline constexpr uint64_t kCounterBytes = 384;
inline constexpr uint64_t kMaxDepth = 1ull << 30;
inline constexpr uint64_t kInputDescTailOffset = 0;
inline constexpr uint64_t kInputDescHeadOffset = 64;
inline constexpr uint64_t kOutputDescTailOffset = 128;
inline constexpr uint64_t kOutputDescHeadOffset = 192;
inline constexpr uint64_t kInitiatorAbortOffset = 256;
inline constexpr uint64_t kPeerAbortOffset = 320;

enum class SpscQueueOpcode : uint64_t {
    INVALID = 0,
    DATA = 1,
    STOP = 2,
    ERROR = 3,
};

struct SpscQueueDescriptor {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};

struct SpscQueueLayout {
    uint64_t depth;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t input_desc_offset;
    uint64_t output_desc_offset;
    uint64_t input_arena_offset;
    uint64_t output_arena_offset;
    uint64_t payload_bytes;
    uint64_t input_desc_tail_offset;
    uint64_t input_desc_head_offset;
    uint64_t output_desc_tail_offset;
    uint64_t output_desc_head_offset;
    uint64_t initiator_abort_offset;
    uint64_t peer_abort_offset;
    uint64_t counter_bytes;

    static bool create(uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes, SpscQueueLayout *out);
};

struct SpscQueueEndpointBinding {
    uint64_t magic_version;
    uint64_t session_instance_id_bits;
    uint64_t transaction_id;
    uint64_t payload_base;
    uint64_t payload_bytes;
    uint64_t counter_base;
    uint64_t counter_bytes;
    uint64_t depth;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
};

struct SpscQueuePayloadView {
    uint64_t local_addr;
    uint64_t nbytes;
};

struct SpscQueueCounterSample {
    bool matched;
    int32_t observed;
};

struct SpscQueueInputHandle {
    uint64_t seq;
    SpscQueueOpcode opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
    SpscQueuePayloadView payload;
    uint64_t owner_cookie;
};

struct SpscQueueOutputReservation {
    uint64_t seq;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
    SpscQueuePayloadView payload;
    uint64_t owner_cookie;
    bool valid;
};

enum class SpscQueueErrorKind : uint32_t {
    NONE = 0,
    BAD_ARGUMENT = 1,
    BAD_BINDING = 2,
    INVALID_DESCRIPTOR = 3,
    OWNERSHIP = 4,
    REMOTE_ABORTED = 5,
    ENDPOINT_ERROR = 6,
};

enum class SpscQueueOp : uint32_t {
    INIT = 1,
    TIMEOUT = 2,
    INPUT_TRY_PEEK = 3,
    INPUT_RELEASE = 4,
    OUTPUT_TRY_RESERVE = 5,
    OUTPUT_PUBLISH = 6,
};

struct SpscQueueError {
    SpscQueueErrorKind kind;
    SpscQueueOp op;
    uint64_t session_instance_id_bits;
    uint64_t transaction_id;
    char message[256];
};

struct MonotonicClock {
    uint64_t (*now_ns)();
};

inline bool is_power_of_two(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

inline bool mul_overflows(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_mul_overflow(a, b, &result);
#else
    return a != 0 && b > UINT64_MAX / a;
#endif
}

inline bool add_overflows(uint64_t a, uint64_t b) { return region_add_overflows(a, b); }

inline bool align_up(uint64_t value, uint64_t align, uint64_t *out) {
    if (out == nullptr || align == 0) {
        return false;
    }
    uint64_t remainder = value % align;
    uint64_t bump = remainder == 0 ? 0 : align - remainder;
    if (add_overflows(value, bump)) {
        return false;
    }
    *out = value + bump;
    return true;
}

inline void store_u64_le(uint8_t *dst, uint64_t value) {
    for (size_t index = 0; index < 8; ++index) {
        dst[index] = static_cast<uint8_t>((value >> (8 * index)) & 0xff);
    }
}

inline uint64_t load_u64_le(const uint8_t *src) {
    uint64_t value = 0;
    for (size_t index = 0; index < 8; ++index) {
        value |= static_cast<uint64_t>(src[index]) << (8 * index);
    }
    return value;
}

inline bool session_instance_id_to_bits(const uint8_t *bytes, size_t nbytes, uint64_t *out) {
    if (bytes == nullptr || out == nullptr || nbytes != 8) {
        return false;
    }
    *out = load_u64_le(bytes);
    return true;
}

inline bool session_instance_id_from_bits(uint64_t bits, uint8_t *out, size_t nbytes) {
    if (out == nullptr || nbytes != 8) {
        return false;
    }
    store_u64_le(out, bits);
    return true;
}

inline bool encode_descriptor(const SpscQueueDescriptor &descriptor, uint8_t *out, size_t nbytes) {
    if (out == nullptr || nbytes != kDescriptorBytes) {
        return false;
    }
    store_u64_le(out + 0, descriptor.seq);
    store_u64_le(out + 8, descriptor.opcode);
    store_u64_le(out + 16, descriptor.payload_offset);
    store_u64_le(out + 24, descriptor.payload_nbytes);
    return true;
}

inline bool decode_descriptor(const uint8_t *src, size_t nbytes, SpscQueueDescriptor *out) {
    if (src == nullptr || out == nullptr || nbytes != kDescriptorBytes) {
        return false;
    }
    SpscQueueDescriptor decoded{
        load_u64_le(src + 0),
        load_u64_le(src + 8),
        load_u64_le(src + 16),
        load_u64_le(src + 24),
    };
    *out = decoded;
    return true;
}

inline bool encode_endpoint_binding(const SpscQueueEndpointBinding &binding, uint64_t *out, size_t count) {
    if (out == nullptr || count != kSpscQueueEndpointBindingScalarCount ||
        binding.magic_version != kSpscQueueMagicVersion || binding.transaction_id == 0) {
        return false;
    }
    out[0] = binding.magic_version;
    out[1] = binding.session_instance_id_bits;
    out[2] = binding.transaction_id;
    out[3] = binding.payload_base;
    out[4] = binding.payload_bytes;
    out[5] = binding.counter_base;
    out[6] = binding.counter_bytes;
    out[7] = binding.depth;
    out[8] = binding.input_arena_bytes;
    out[9] = binding.output_arena_bytes;
    return true;
}

inline bool decode_endpoint_binding(const uint64_t *scalars, size_t count, SpscQueueEndpointBinding *out) {
    if (scalars == nullptr || out == nullptr || count != kSpscQueueEndpointBindingScalarCount ||
        scalars[0] != kSpscQueueMagicVersion || scalars[2] == 0) {
        return false;
    }
    SpscQueueEndpointBinding decoded{
        scalars[0], scalars[1], scalars[2], scalars[3], scalars[4],
        scalars[5], scalars[6], scalars[7], scalars[8], scalars[9],
    };
    *out = decoded;
    return true;
}

inline const char *spsc_queue_op_name(SpscQueueOp op) {
    switch (op) {
    case SpscQueueOp::INIT:
        return "init";
    case SpscQueueOp::TIMEOUT:
        return "timeout";
    case SpscQueueOp::INPUT_TRY_PEEK:
        return "input.try_peek";
    case SpscQueueOp::INPUT_RELEASE:
        return "input.release";
    case SpscQueueOp::OUTPUT_TRY_RESERVE:
        return "output.try_reserve";
    case SpscQueueOp::OUTPUT_PUBLISH:
        return "output.publish";
    }
    return "unknown";
}

inline bool valid_input_opcode(SpscQueueOpcode opcode) {
    return opcode == SpscQueueOpcode::DATA || opcode == SpscQueueOpcode::STOP;
}

inline bool valid_output_opcode(SpscQueueOpcode opcode) {
    return opcode == SpscQueueOpcode::DATA || opcode == SpscQueueOpcode::ERROR;
}

inline int32_t counter_low32(uint64_t value) { return static_cast<int32_t>(static_cast<uint32_t>(value)); }

inline bool reconstruct_counter(int32_t observed_low32, uint64_t depth, uint64_t *local_value) {
    if (local_value == nullptr || depth == 0 || depth > kMaxDepth) {
        return false;
    }
    uint32_t local_low32 = static_cast<uint32_t>(*local_value);
    int32_t delta = static_cast<int32_t>(static_cast<uint32_t>(observed_low32) - local_low32);
    if (delta < 0 || static_cast<uint64_t>(delta) > depth) {
        return false;
    }
    *local_value += static_cast<uint64_t>(delta);
    return true;
}

inline void copy_error_text(char *dst, size_t dst_size, const char *src) {
    if (dst == nullptr || dst_size == 0) {
        return;
    }
    const char *in = src == nullptr ? "" : src;
    size_t n = strnlen(in, dst_size - 1);
    memcpy(dst, in, n);
    dst[n] = '\0';
}

inline void append_error_note(char *dst, size_t dst_size, const char *note) {
    if (dst == nullptr || dst_size == 0 || note == nullptr || note[0] == '\0') {
        return;
    }
    size_t len = strnlen(dst, dst_size);
    if (len + 4 >= dst_size) {
        return;
    }
    int written = snprintf(dst + len, dst_size - len, " (%s)", note);
    (void)written;
}

inline bool payload_in_arena(uint64_t offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes) {
    if (nbytes == 0 || add_overflows(offset, nbytes) || add_overflows(arena_offset, arena_bytes)) {
        return false;
    }
    return offset >= arena_offset && offset + nbytes <= arena_offset + arena_bytes;
}

inline uint64_t payload_expected_offset(uint64_t cursor, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes) {
    uint64_t arena_pos = cursor % arena_bytes;
    return arena_pos + nbytes > arena_bytes ? arena_offset : arena_offset + arena_pos;
}

inline bool
SpscQueueLayout::create(uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes, SpscQueueLayout *out) {
    if (out == nullptr || !is_power_of_two(depth) || depth > kMaxDepth || input_arena_bytes == 0 ||
        output_arena_bytes == 0 || input_arena_bytes % kArenaAlignment != 0 ||
        output_arena_bytes % kArenaAlignment != 0) {
        return false;
    }
    if (mul_overflows(depth, kDescriptorBytes)) {
        return false;
    }
    uint64_t desc_ring_bytes = depth * kDescriptorBytes;
    uint64_t input_desc_offset = 0;
    if (add_overflows(input_desc_offset, desc_ring_bytes)) {
        return false;
    }
    uint64_t output_desc_offset = input_desc_offset + desc_ring_bytes;
    if (add_overflows(output_desc_offset, desc_ring_bytes)) {
        return false;
    }
    uint64_t desc_end = output_desc_offset + desc_ring_bytes;
    uint64_t input_arena_offset = 0;
    if (!align_up(desc_end, kArenaAlignment, &input_arena_offset)) {
        return false;
    }
    if (add_overflows(input_arena_offset, input_arena_bytes)) {
        return false;
    }
    uint64_t input_arena_end = input_arena_offset + input_arena_bytes;
    uint64_t output_arena_offset = 0;
    if (!align_up(input_arena_end, kArenaAlignment, &output_arena_offset)) {
        return false;
    }
    if (add_overflows(output_arena_offset, output_arena_bytes)) {
        return false;
    }
    uint64_t payload_bytes = output_arena_offset + output_arena_bytes;
    if (output_desc_offset % kDescriptorRingAlignment != 0 || input_arena_offset % kArenaAlignment != 0 ||
        output_arena_offset % kArenaAlignment != 0) {
        return false;
    }
    *out = SpscQueueLayout{
        depth,
        input_arena_bytes,
        output_arena_bytes,
        input_desc_offset,
        output_desc_offset,
        input_arena_offset,
        output_arena_offset,
        payload_bytes,
        kInputDescTailOffset,
        kInputDescHeadOffset,
        kOutputDescTailOffset,
        kOutputDescHeadOffset,
        kInitiatorAbortOffset,
        kPeerAbortOffset,
        kCounterBytes,
    };
    return true;
}

template <typename RegionView, uint64_t MaxInflight = 1>
class SpscQueueEndpoint {
    static_assert(MaxInflight >= 1, "MaxInflight must be at least 1");
    static_assert(MaxInflight <= kMaxDepth, "MaxInflight exceeds kMaxDepth");

public:
    static constexpr uint64_t kEntryCapacity = MaxInflight + 1;

    class InputQueue {
        struct ActiveInputEntry {
            uint64_t seq;
            SpscQueueOpcode opcode;
            uint64_t payload_offset;
            uint64_t payload_nbytes;
            SpscQueuePayloadView payload;
            bool completed;
        };

    public:
        explicit InputQueue(SpscQueueEndpoint *parent) :
            parent_(parent) {}

        InputQueue(const InputQueue &) = delete;
        InputQueue &operator=(const InputQueue &) = delete;
        InputQueue(InputQueue &&) = delete;
        InputQueue &operator=(InputQueue &&) = delete;

        bool peek(uint64_t timeout_ns, SpscQueueInputHandle &out) {
            out = SpscQueueInputHandle{};
            if (!parent_->ensure_live()) {
                return false;
            }
            if (timeout_ns == 0) {
                return false;
            }
            uint64_t now = parent_->clock_.now_ns();
            uint64_t deadline = add_overflows(now, timeout_ns) ? UINT64_MAX : now + timeout_ns;
            while (true) {
                if (try_peek(out)) {
                    return true;
                }
                if (parent_->error_.kind != SpscQueueErrorKind::NONE) {
                    return false;
                }
                if (!parent_->wait_progress(
                        parent_->layout_.input_desc_tail_offset, input_tail_, deadline, SpscQueueOp::INPUT_TRY_PEEK
                    )) {
                    return false;
                }
            }
        }

        bool try_peek(SpscQueueInputHandle &out) {
            out = SpscQueueInputHandle{};
            if (!parent_->ensure_live()) {
                return false;
            }
            const SpscQueueLayout &layout = parent_->layout_;
            if (!parent_->refresh_counter(layout.input_desc_tail_offset, input_tail_, SpscQueueOp::INPUT_TRY_PEEK)) {
                return false;
            }
            if (stop_observed_) {
                if (input_tail_ != input_acquire_) {
                    parent_->poison(
                        SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK,
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
                    SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK,
                    "input descriptor state invalid"
                );
                return false;
            }

            SpscQueueDescriptor slot{};
            uint64_t slot_index = input_acquire_ & (layout.depth - 1);
            uint64_t slot_offset = layout.input_desc_offset + slot_index * kDescriptorBytes;
            if (!parent_->read_descriptor(slot_offset, &slot, SpscQueueOp::INPUT_TRY_PEEK)) {
                return false;
            }
            uint64_t expected_seq = input_acquire_ + 1;
            if (slot.seq != expected_seq) {
                parent_->poison(
                    SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK, "input descriptor seq mismatch"
                );
                return false;
            }
            SpscQueueOpcode opcode = static_cast<SpscQueueOpcode>(slot.opcode);
            if (!valid_input_opcode(opcode)) {
                parent_->poison(
                    SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK, "invalid input opcode"
                );
                return false;
            }
            if (opcode == SpscQueueOpcode::STOP && (slot.payload_offset != 0 || slot.payload_nbytes != 0)) {
                parent_->poison(
                    SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK,
                    "STOP descriptor must be zero-byte"
                );
                return false;
            }
            bool counts_against_window = opcode == SpscQueueOpcode::DATA;
            if (counts_against_window && active_non_stop_count_ >= MaxInflight) {
                return false;
            }
            if (active_count_ >= kEntryCapacity) {
                parent_->poison(SpscQueueErrorKind::OWNERSHIP, SpscQueueOp::INPUT_TRY_PEEK, "input window state full");
                return false;
            }

            SpscQueuePayloadView view{0, 0};
            if (slot.payload_nbytes == 0) {
                if (slot.payload_offset != 0) {
                    parent_->poison(
                        SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK,
                        "zero-byte descriptor uses nonzero payload offset"
                    );
                    return false;
                }
            } else if (!payload_in_arena(
                           slot.payload_offset, slot.payload_nbytes, layout.input_arena_offset, layout.input_arena_bytes
                       )) {
                parent_->poison(
                    SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK, "input payload out of arena"
                );
                return false;
            } else if (!parent_->payload_matches_head(
                           input_payload_acquire_head_, slot.payload_offset, slot.payload_nbytes,
                           layout.input_arena_offset, layout.input_arena_bytes, SpscQueueOp::INPUT_TRY_PEEK
                       )) {
                return false;
            } else if (!parent_->payload_read(
                           slot.payload_offset, slot.payload_nbytes, view, SpscQueueOp::INPUT_TRY_PEEK
                       )) {
                return false;
            } else {
                parent_->advance_payload_head(
                    input_payload_acquire_head_, slot.payload_offset, slot.payload_nbytes, layout.input_arena_offset,
                    layout.input_arena_bytes, SpscQueueOp::INPUT_TRY_PEEK
                );
                if (parent_->error_.kind != SpscQueueErrorKind::NONE) {
                    return false;
                }
            }

            out = SpscQueueInputHandle{
                slot.seq, opcode, slot.payload_offset, slot.payload_nbytes, view, parent_->owner_cookie()
            };
            uint64_t insert_index = (active_head_ + active_count_) % kEntryCapacity;
            active_entries_[insert_index] =
                ActiveInputEntry{slot.seq, opcode, slot.payload_offset, slot.payload_nbytes, view, false};
            active_count_ += 1;
            if (counts_against_window) {
                active_non_stop_count_ += 1;
            }
            input_acquire_ += 1;
            if (opcode == SpscQueueOpcode::STOP) {
                stop_observed_ = true;
                if (input_tail_ != input_acquire_) {
                    parent_->poison(
                        SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::INPUT_TRY_PEEK,
                        "input descriptor published after STOP"
                    );
                    return false;
                }
            }
            return true;
        }

        bool release(const SpscQueueInputHandle &handle) {
            if (!parent_->ensure_live()) {
                return false;
            }
            if (handle.owner_cookie != parent_->owner_cookie()) {
                parent_->poison(
                    SpscQueueErrorKind::OWNERSHIP, SpscQueueOp::INPUT_RELEASE, "input handle is not active"
                );
                return false;
            }
            ActiveInputEntry *entry = entry_for_seq(handle.seq);
            if (entry == nullptr || handle.opcode != entry->opcode || handle.payload_offset != entry->payload_offset ||
                handle.payload_nbytes != entry->payload_nbytes ||
                handle.payload.local_addr != entry->payload.local_addr ||
                handle.payload.nbytes != entry->payload.nbytes) {
                parent_->poison(
                    SpscQueueErrorKind::OWNERSHIP, SpscQueueOp::INPUT_RELEASE, "input handle is not active"
                );
                return false;
            }
            if (entry->completed) {
                parent_->poison(
                    SpscQueueErrorKind::OWNERSHIP, SpscQueueOp::INPUT_RELEASE, "input handle already released"
                );
                return false;
            }
            entry->completed = true;
            return release_completed_prefix();
        }

        bool drained() const { return drained_; }

    private:
        friend class SpscQueueEndpoint;

        void initialize() {
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
            input_acquire_ = 0;
            stop_observed_ = false;
            drained_ = false;
        }

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
                        SpscQueueOp::INPUT_RELEASE
                    );
                    if (parent_->error_.kind != SpscQueueErrorKind::NONE) {
                        return false;
                    }
                }
                input_head_ += 1;
                if (entry.opcode == SpscQueueOpcode::DATA) {
                    active_non_stop_count_ -= 1;
                }
                if (entry.opcode == SpscQueueOpcode::STOP) {
                    drained_ = true;
                }
                active_entries_[active_head_] = ActiveInputEntry{};
                active_head_ = (active_head_ + 1) % kEntryCapacity;
                active_count_ -= 1;
                if (!parent_->notify_counter(
                        parent_->layout_.input_desc_head_offset, input_head_, SpscQueueOp::INPUT_RELEASE
                    )) {
                    return false;
                }
            }
            return true;
        }

        SpscQueueEndpoint *parent_;
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
        explicit OutputQueue(SpscQueueEndpoint *parent) :
            parent_(parent) {}

        OutputQueue(const OutputQueue &) = delete;
        OutputQueue &operator=(const OutputQueue &) = delete;
        OutputQueue(OutputQueue &&) = delete;
        OutputQueue &operator=(OutputQueue &&) = delete;

        bool reserve(uint64_t nbytes, uint64_t timeout_ns, SpscQueueOutputReservation &out) {
            out = SpscQueueOutputReservation{};
            if (!parent_->ensure_live()) {
                return false;
            }
            if (timeout_ns == 0) {
                return false;
            }
            if (nbytes > parent_->layout_.output_arena_bytes) {
                return false;
            }
            uint64_t now = parent_->clock_.now_ns();
            uint64_t deadline = add_overflows(now, timeout_ns) ? UINT64_MAX : now + timeout_ns;
            while (true) {
                if (try_reserve(nbytes, out)) {
                    return true;
                }
                if (parent_->error_.kind != SpscQueueErrorKind::NONE) {
                    return false;
                }
                if (!parent_->wait_progress(
                        parent_->layout_.output_desc_head_offset, output_head_, deadline,
                        SpscQueueOp::OUTPUT_TRY_RESERVE
                    )) {
                    return false;
                }
            }
        }

        bool try_reserve(uint64_t nbytes, SpscQueueOutputReservation &out) {
            out = SpscQueueOutputReservation{};
            if (!parent_->ensure_live()) {
                return false;
            }
            const SpscQueueLayout &layout = parent_->layout_;
            if (nbytes > layout.output_arena_bytes) {
                return false;
            }
            if (reservation_active_) {
                parent_->poison(
                    SpscQueueErrorKind::OWNERSHIP, SpscQueueOp::OUTPUT_TRY_RESERVE, "output reservation already active"
                );
                return false;
            }
            uint64_t old_head = output_head_;
            if (!parent_->refresh_counter(
                    layout.output_desc_head_offset, output_head_, SpscQueueOp::OUTPUT_TRY_RESERVE
                )) {
                return false;
            }
            if (output_head_ != old_head &&
                !replay_output_releases(old_head, output_head_, SpscQueueOp::OUTPUT_TRY_RESERVE)) {
                return false;
            }
            if (output_tail_ - output_head_ >= layout.depth) {
                return false;
            }

            uint64_t payload_offset = 0;
            SpscQueuePayloadView view{0, 0};
            uint64_t next_payload_tail = output_payload_tail_;
            if (nbytes != 0) {
                uint64_t arena_base = layout.output_arena_offset;
                uint64_t arena_bytes = layout.output_arena_bytes;
                uint64_t arena_pos = next_payload_tail % arena_bytes;
                if (arena_pos + nbytes > arena_bytes) {
                    next_payload_tail += arena_bytes - arena_pos;
                    arena_pos = 0;
                }
                if (next_payload_tail + nbytes - output_payload_head_ > arena_bytes) {
                    return false;
                }
                payload_offset = arena_base + arena_pos;
                view = SpscQueuePayloadView{parent_->view_.payload().span().base + payload_offset, nbytes};
                next_payload_tail += nbytes;
            }

            reservation_active_ = true;
            reservation_seq_ = output_tail_ + 1;
            reservation_offset_ = payload_offset;
            reservation_nbytes_ = nbytes;
            output_payload_tail_ = next_payload_tail;
            out = SpscQueueOutputReservation{
                reservation_seq_, payload_offset, nbytes, view, parent_->owner_cookie(), true
            };
            return true;
        }

        bool publish(const SpscQueueOutputReservation &reservation, SpscQueueOpcode opcode) {
            if (!parent_->ensure_live()) {
                return false;
            }
            if (!reservation_active_ || !reservation.valid || reservation.owner_cookie != parent_->owner_cookie() ||
                reservation.seq != reservation_seq_ || reservation.payload_offset != reservation_offset_ ||
                reservation.payload_nbytes != reservation_nbytes_) {
                parent_->poison(
                    SpscQueueErrorKind::OWNERSHIP, SpscQueueOp::OUTPUT_PUBLISH, "unknown output reservation"
                );
                return false;
            }
            if (!valid_output_opcode(opcode)) {
                parent_->poison(
                    SpscQueueErrorKind::INVALID_DESCRIPTOR, SpscQueueOp::OUTPUT_PUBLISH, "invalid output opcode"
                );
                return false;
            }
            uint64_t slot_index = output_tail_ & (parent_->layout_.depth - 1);
            uint64_t slot_offset = parent_->layout_.output_desc_offset + slot_index * kDescriptorBytes;
            if (!parent_->write_descriptor(
                    slot_offset, reservation.seq, opcode, reservation.payload_offset, reservation.payload_nbytes,
                    SpscQueueOp::OUTPUT_PUBLISH
                )) {
                return false;
            }
            output_tail_ += 1;
            reservation_active_ = false;
            reservation_seq_ = 0;
            reservation_offset_ = 0;
            reservation_nbytes_ = 0;
            return parent_->notify_counter(
                parent_->layout_.output_desc_tail_offset, output_tail_, SpscQueueOp::OUTPUT_PUBLISH
            );
        }

    private:
        friend class SpscQueueEndpoint;

        void initialize() {
            output_head_ = 0;
            output_tail_ = 0;
            output_payload_head_ = 0;
            output_payload_tail_ = 0;
            reservation_active_ = false;
            reservation_seq_ = 0;
            reservation_offset_ = 0;
            reservation_nbytes_ = 0;
        }

        bool replay_output_releases(uint64_t old_head, uint64_t new_head, SpscQueueOp op) {
            uint64_t cursor = old_head;
            while (cursor < new_head) {
                SpscQueueDescriptor slot{};
                uint64_t slot_index = cursor & (parent_->layout_.depth - 1);
                uint64_t slot_offset = parent_->layout_.output_desc_offset + slot_index * kDescriptorBytes;
                if (!parent_->read_descriptor(slot_offset, &slot, op)) {
                    return false;
                }
                if (slot.seq != cursor + 1) {
                    parent_->poison(SpscQueueErrorKind::INVALID_DESCRIPTOR, op, "output release replay seq mismatch");
                    return false;
                }
                if (slot.payload_nbytes != 0) {
                    parent_->advance_payload_head(
                        output_payload_head_, slot.payload_offset, slot.payload_nbytes,
                        parent_->layout_.output_arena_offset, parent_->layout_.output_arena_bytes, op
                    );
                    if (parent_->error_.kind != SpscQueueErrorKind::NONE) {
                        return false;
                    }
                }
                cursor += 1;
            }
            return true;
        }

        SpscQueueEndpoint *parent_;
        uint64_t output_head_{0};
        uint64_t output_tail_{0};
        uint64_t output_payload_head_{0};
        uint64_t output_payload_tail_{0};
        bool reservation_active_{false};
        uint64_t reservation_seq_{0};
        uint64_t reservation_offset_{0};
        uint64_t reservation_nbytes_{0};
    };

    SpscQueueEndpoint(const SpscQueueEndpointBinding &binding, RegionView view, MonotonicClock clock) :
        view_(std::move(view)),
        clock_(clock),
        input_queue_(this),
        output_queue_(this) {
        construct(binding);
    }

    SpscQueueEndpoint(const SpscQueueEndpoint &) = delete;
    SpscQueueEndpoint &operator=(const SpscQueueEndpoint &) = delete;
    SpscQueueEndpoint(SpscQueueEndpoint &&) = delete;
    SpscQueueEndpoint &operator=(SpscQueueEndpoint &&) = delete;

    const SpscQueueError &error() const { return error_; }
    const SpscQueueLayout &layout() const { return layout_; }
    bool live() const { return live_; }
    InputQueue &input() { return input_queue_; }
    OutputQueue &output() { return output_queue_; }

private:
    uint64_t owner_cookie() const { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this)); }

    void construct(const SpscQueueEndpointBinding &binding) {
        if (binding.magic_version != kSpscQueueMagicVersion || binding.transaction_id == 0) {
            identity_trusted_ = false;
            set_error(SpscQueueErrorKind::BAD_BINDING, SpscQueueOp::INIT, "invalid queue binding");
            return;
        }
        identity_trusted_ = true;
        error_.session_instance_id_bits = binding.session_instance_id_bits;
        error_.transaction_id = binding.transaction_id;

        if (clock_.now_ns == nullptr) {
            set_error(SpscQueueErrorKind::BAD_ARGUMENT, SpscQueueOp::INIT, "clock is null");
            return;
        }
        if (view_.failed()) {
            set_error(SpscQueueErrorKind::ENDPOINT_ERROR, SpscQueueOp::INIT, view_.error().message);
            return;
        }

        RegionPartLocalSpan payload_span = view_.payload().span();
        RegionPartLocalSpan counter_span = view_.counter().span();
        if (binding.payload_base == 0 || binding.payload_bytes == 0 || binding.counter_base == 0 ||
            binding.counter_bytes == 0 || payload_span.base != binding.payload_base ||
            payload_span.logical_bytes != binding.payload_bytes || counter_span.base != binding.counter_base ||
            counter_span.logical_bytes != binding.counter_bytes) {
            set_error(SpscQueueErrorKind::BAD_BINDING, SpscQueueOp::INIT, "view spans do not match binding");
            return;
        }
        if (!SpscQueueLayout::create(binding.depth, binding.input_arena_bytes, binding.output_arena_bytes, &layout_)) {
            set_error(SpscQueueErrorKind::BAD_BINDING, SpscQueueOp::INIT, "invalid queue layout");
            return;
        }
        if (layout_.payload_bytes != binding.payload_bytes || layout_.counter_bytes != binding.counter_bytes) {
            set_error(
                SpscQueueErrorKind::BAD_BINDING, SpscQueueOp::INIT, "reconstructed layout does not match binding"
            );
            return;
        }
        if (MaxInflight > layout_.depth) {
            set_error(SpscQueueErrorKind::BAD_ARGUMENT, SpscQueueOp::INIT, "invalid input window");
            return;
        }

        input_queue_.initialize();
        output_queue_.initialize();
        live_ = true;
    }

    bool ensure_live() const { return live_ && error_.kind == SpscQueueErrorKind::NONE; }

    void format_error_message(const char *detail) {
        const char *text = detail == nullptr ? "" : detail;
        if (identity_trusted_) {
            snprintf(
                error_.message, sizeof(error_.message),
                "SPSC queue endpoint error op=%s kind=%" PRIu32 " session=0x%016" PRIx64 " transaction=%" PRIu64
                " msg=%s",
                spsc_queue_op_name(error_.op), static_cast<uint32_t>(error_.kind), error_.session_instance_id_bits,
                error_.transaction_id, text
            );
        } else {
            snprintf(
                error_.message, sizeof(error_.message), "SPSC queue endpoint error op=%s kind=%" PRIu32 " msg=%s",
                spsc_queue_op_name(error_.op), static_cast<uint32_t>(error_.kind), text
            );
        }
    }

    void set_error(SpscQueueErrorKind kind, SpscQueueOp op, const char *message) {
        if (error_.kind != SpscQueueErrorKind::NONE) {
            return;
        }
        live_ = false;
        error_.kind = kind;
        error_.op = op;
        format_error_message(message);
    }

    void poison(SpscQueueErrorKind kind, SpscQueueOp op, const char *message) {
        bool first = error_.kind == SpscQueueErrorKind::NONE;
        set_error(kind, op, message);
        if (!first || kind == SpscQueueErrorKind::REMOTE_ABORTED) {
            return;
        }
        if (!view_.counter().notify(layout_.peer_abort_offset, 1, RegionNotifyOp::Set)) {
            append_error_note(error_.message, sizeof(error_.message), "abort notify failed");
        }
    }

    bool sample_peer_abort(SpscQueueOp op) {
        SpscQueueCounterSample sample{};
        if (!view_.counter().test(layout_.initiator_abort_offset, 1, RegionWaitCmp::GE, sample)) {
            if (view_.failed()) {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, view_.error().message);
            } else {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, "peer abort sample failed");
            }
            return false;
        }
        if (sample.matched) {
            poison(SpscQueueErrorKind::REMOTE_ABORTED, SpscQueueOp::TIMEOUT, "remote abort");
            return false;
        }
        return true;
    }

    bool wait_progress(uint64_t offset, uint64_t local_value, uint64_t deadline, SpscQueueOp op) {
        if (!sample_peer_abort(op)) {
            return false;
        }
        uint64_t now = clock_.now_ns();
        if (now >= deadline) {
            if (!sample_peer_abort(SpscQueueOp::TIMEOUT)) {
                return false;
            }
            return false;
        }
        uint64_t remaining = deadline - now;
        int32_t observed = 0;
        bool woke = view_.counter().wait(offset, counter_low32(local_value), RegionWaitCmp::NE, remaining, observed);
        if (view_.failed()) {
            poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, view_.error().message);
            return false;
        }
        if (woke) {
            return sample_peer_abort(op);
        }
        if (!sample_peer_abort(SpscQueueOp::TIMEOUT)) {
            return false;
        }
        return false;
    }

    bool refresh_counter(uint64_t offset, uint64_t &local, SpscQueueOp op) {
        SpscQueueCounterSample sample{};
        if (!view_.counter().test(offset, counter_low32(local), RegionWaitCmp::NE, sample)) {
            if (view_.failed()) {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, view_.error().message);
            } else {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, "counter test failed");
            }
            return false;
        }
        if (!sample.matched) {
            return true;
        }
        if (!reconstruct_counter(sample.observed, layout_.depth, &local)) {
            poison(SpscQueueErrorKind::INVALID_DESCRIPTOR, op, "counter reconstruction failed");
            return false;
        }
        return true;
    }

    bool notify_counter(uint64_t offset, uint64_t value, SpscQueueOp op) {
        if (!view_.counter().notify(offset, counter_low32(value), RegionNotifyOp::Set)) {
            if (view_.failed()) {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, view_.error().message);
            } else {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, "counter notify failed");
            }
            return false;
        }
        return true;
    }

    bool payload_read(uint64_t offset, uint64_t nbytes, SpscQueuePayloadView &out, SpscQueueOp op) {
        out = SpscQueuePayloadView{0, 0};
        if (!view_.payload().read(offset, nbytes, out)) {
            if (view_.failed()) {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, view_.error().message);
            } else {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, "payload read failed");
            }
            return false;
        }
        return true;
    }

    bool payload_write(uint64_t offset, const void *src, uint64_t nbytes, SpscQueueOp op) {
        if (!view_.payload().write(offset, src, nbytes)) {
            if (view_.failed()) {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, view_.error().message);
            } else {
                poison(SpscQueueErrorKind::ENDPOINT_ERROR, op, "payload write failed");
            }
            return false;
        }
        return true;
    }

    bool read_descriptor(uint64_t slot_offset, SpscQueueDescriptor *slot, SpscQueueOp op) {
        SpscQueuePayloadView view{};
        if (!payload_read(slot_offset, kDescriptorBytes, view, op)) {
            return false;
        }
        uint8_t bytes[kDescriptorBytes];
        memcpy(bytes, reinterpret_cast<const void *>(static_cast<uintptr_t>(view.local_addr)), kDescriptorBytes);
        return decode_descriptor(bytes, kDescriptorBytes, slot);
    }

    bool write_descriptor(
        uint64_t slot_offset, uint64_t seq, SpscQueueOpcode opcode, uint64_t payload_offset, uint64_t payload_nbytes,
        SpscQueueOp op
    ) {
        uint8_t encoded[kDescriptorBytes];
        SpscQueueDescriptor descriptor{0, static_cast<uint64_t>(opcode), payload_offset, payload_nbytes};
        if (!encode_descriptor(descriptor, encoded, kDescriptorBytes)) {
            poison(SpscQueueErrorKind::INVALID_DESCRIPTOR, op, "descriptor encode failed");
            return false;
        }
        if (!payload_write(slot_offset + 8, encoded + 8, 24, op)) {
            return false;
        }
        store_u64_le(encoded, seq);
        return payload_write(slot_offset, encoded, 8, op);
    }

    bool payload_matches_head(
        uint64_t cursor, uint64_t payload_offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes,
        SpscQueueOp op
    ) {
        if (nbytes == 0) {
            return true;
        }
        uint64_t expected_offset = payload_expected_offset(cursor, nbytes, arena_offset, arena_bytes);
        if (payload_offset != expected_offset) {
            poison(SpscQueueErrorKind::INVALID_DESCRIPTOR, op, "payload replay offset mismatch");
            return false;
        }
        return true;
    }

    void advance_payload_head(
        uint64_t &cursor, uint64_t payload_offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes,
        SpscQueueOp op
    ) {
        if (nbytes == 0) {
            return;
        }
        uint64_t expected_offset = payload_expected_offset(cursor, nbytes, arena_offset, arena_bytes);
        if (expected_offset != payload_offset) {
            poison(SpscQueueErrorKind::INVALID_DESCRIPTOR, op, "payload replay offset mismatch");
            return;
        }
        uint64_t arena_pos = cursor % arena_bytes;
        if (arena_pos + nbytes > arena_bytes) {
            cursor += arena_bytes - arena_pos;
        }
        cursor += nbytes;
    }

    RegionView view_;
    MonotonicClock clock_{};
    SpscQueueLayout layout_{};
    SpscQueueError error_{SpscQueueErrorKind::NONE, SpscQueueOp::INIT, 0, 0, ""};
    bool identity_trusted_{false};
    bool live_{false};
    InputQueue input_queue_;
    OutputQueue output_queue_;
};

static_assert(kSpscQueueMagic == 0x53505351u, "SPSQ magic changed");
static_assert(kSpscQueueAbiMajor == 1, "SPSQ ABI major changed");
static_assert(kSpscQueueAbiMinor == 0, "SPSQ ABI minor changed");
static_assert(kSpscQueueMagicVersion == 0x5350535100010000ull, "SPSQ packed magic_version changed");
static_assert(kSpscQueueEndpointBindingScalarCount == 10, "SPSQ binding scalar count changed");
static_assert(kDescriptorBytes == 32, "SPSQ descriptor size changed");
static_assert(kArenaAlignment == 64, "SPSQ arena alignment changed");
static_assert(kCounterStride == 64, "SPSQ counter stride changed");
static_assert(kCounterBytes == 384, "SPSQ counter bytes changed");
static_assert(kMaxDepth == (1ull << 30), "SPSQ max depth changed");
static_assert(kInputDescTailOffset == 0, "SPSQ input tail offset changed");
static_assert(kInputDescHeadOffset == 64, "SPSQ input head offset changed");
static_assert(kOutputDescTailOffset == 128, "SPSQ output tail offset changed");
static_assert(kOutputDescHeadOffset == 192, "SPSQ output head offset changed");
static_assert(kInitiatorAbortOffset == 256, "SPSQ initiator abort offset changed");
static_assert(kPeerAbortOffset == 320, "SPSQ peer abort offset changed");
static_assert(sizeof(SpscQueueDescriptor) == 32, "SpscQueueDescriptor ABI size changed");
static_assert(offsetof(SpscQueueDescriptor, seq) == 0, "SpscQueueDescriptor::seq offset changed");
static_assert(offsetof(SpscQueueDescriptor, opcode) == 8, "SpscQueueDescriptor::opcode offset changed");
static_assert(offsetof(SpscQueueDescriptor, payload_offset) == 16, "SpscQueueDescriptor::payload_offset changed");
static_assert(offsetof(SpscQueueDescriptor, payload_nbytes) == 24, "SpscQueueDescriptor::payload_nbytes changed");
static_assert(sizeof(SpscQueueEndpointBinding) == 80, "SpscQueueEndpointBinding ABI size changed");
static_assert(kSpscQueueEndpointBindingScalarCount * sizeof(uint64_t) == 80, "SPSQ binding scalar bytes changed");
static_assert(offsetof(SpscQueueEndpointBinding, magic_version) == 0, "binding magic_version offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, session_instance_id_bits) == 8, "binding session bits offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, transaction_id) == 16, "binding transaction_id offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, payload_base) == 24, "binding payload_base offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, payload_bytes) == 32, "binding payload_bytes offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, counter_base) == 40, "binding counter_base offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, counter_bytes) == 48, "binding counter_bytes offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, depth) == 56, "binding depth offset changed");
static_assert(offsetof(SpscQueueEndpointBinding, input_arena_bytes) == 64, "binding input_arena_bytes offset changed");
static_assert(
    offsetof(SpscQueueEndpointBinding, output_arena_bytes) == 72, "binding output_arena_bytes offset changed"
);
static_assert(std::is_standard_layout_v<SpscQueueDescriptor>, "SpscQueueDescriptor must be standard layout");
static_assert(std::is_trivially_copyable_v<SpscQueueDescriptor>, "SpscQueueDescriptor must be trivially copyable");
static_assert(std::is_standard_layout_v<SpscQueueEndpointBinding>, "SpscQueueEndpointBinding must be standard layout");
static_assert(
    std::is_trivially_copyable_v<SpscQueueEndpointBinding>, "SpscQueueEndpointBinding must be trivially copyable"
);
static_assert(std::is_standard_layout_v<SpscQueueLayout>, "SpscQueueLayout must be standard layout");
static_assert(std::is_standard_layout_v<SpscQueuePayloadView>, "SpscQueuePayloadView must be standard layout");
static_assert(std::is_trivially_copyable_v<SpscQueuePayloadView>, "SpscQueuePayloadView must be trivially copyable");
static_assert(sizeof(SpscQueuePayloadView) == 16, "SpscQueuePayloadView size changed");
static_assert(std::is_standard_layout_v<SpscQueueError>, "SpscQueueError must be standard layout");
static_assert(offsetof(SpscQueueError, session_instance_id_bits) == 8, "SpscQueueError session offset changed");

}  // namespace spsc_queue
