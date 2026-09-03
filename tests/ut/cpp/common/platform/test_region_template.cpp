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

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "common/region_template.h"

namespace {

struct LayoutCase {
    uint64_t depth;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t output_desc_offset;
    uint64_t input_arena_offset;
    uint64_t output_arena_offset;
    uint64_t payload_bytes;
};

// Golden vectors shared with tests/ut/py/test_worker/test_comm_region_template.py.
constexpr std::array<LayoutCase, 4> kLayoutGolden{{
    {1, 64, 64, 32, 64, 128, 192},
    {4, 128, 192, 128, 256, 384, 576},
    {8, 192, 64, 256, 512, 704, 768},
    {2, 64, 128, 64, 128, 192, 320},
}};

constexpr uint8_t kDescriptorGoldenBytes[32] = {
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

constexpr uint8_t kSessionGoldenBytes[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
constexpr uint64_t kSessionGoldenBits = 0x0807060504030201ull;
constexpr std::array<uint64_t, 10> kBindingGolden{
    0x5350535100010000ull, kSessionGoldenBits, 42, 0x1000, 192, 0x2000, 384, 1, 64, 64,
};

}  // namespace

TEST(RegionTemplateTest, PackedMagicVersionIsSpsqAbi10) {
    EXPECT_EQ(spsc_queue::kSpscQueueMagic, 0x53505351u);
    EXPECT_EQ(spsc_queue::kSpscQueueAbiMajor, 1);
    EXPECT_EQ(spsc_queue::kSpscQueueAbiMinor, 0);
    EXPECT_EQ(spsc_queue::kSpscQueueMagicVersion, 0x5350535100010000ull);
    EXPECT_EQ(spsc_queue::kSpscQueueEndpointBindingScalarCount, 10u);
    EXPECT_EQ(spsc_queue::kDescriptorBytes, 32u);
    EXPECT_EQ(spsc_queue::kArenaAlignment, 64u);
    EXPECT_EQ(spsc_queue::kCounterStride, 64u);
    EXPECT_EQ(spsc_queue::kCounterBytes, 384u);
    EXPECT_EQ(spsc_queue::kMaxDepth, 1ull << 30);
}

TEST(RegionTemplateTest, LayoutGoldenVectors) {
    for (const auto &test_case : kLayoutGolden) {
        spsc_queue::SpscQueueLayout layout{};
        ASSERT_TRUE(spsc_queue::SpscQueueLayout::create(
            test_case.depth, test_case.input_arena_bytes, test_case.output_arena_bytes, &layout
        ));
        EXPECT_EQ(layout.input_desc_offset, 0u);
        EXPECT_EQ(layout.output_desc_offset, test_case.output_desc_offset);
        EXPECT_EQ(layout.input_arena_offset, test_case.input_arena_offset);
        EXPECT_EQ(layout.output_arena_offset, test_case.output_arena_offset);
        EXPECT_EQ(layout.payload_bytes, test_case.payload_bytes);
        EXPECT_EQ(layout.input_arena_offset % 64u, 0u);
        EXPECT_EQ(layout.output_arena_offset % 64u, 0u);
        EXPECT_EQ(layout.input_desc_tail_offset, 0u);
        EXPECT_EQ(layout.input_desc_head_offset, 64u);
        EXPECT_EQ(layout.output_desc_tail_offset, 128u);
        EXPECT_EQ(layout.output_desc_head_offset, 192u);
        EXPECT_EQ(layout.initiator_abort_offset, 256u);
        EXPECT_EQ(layout.peer_abort_offset, 320u);
        EXPECT_EQ(layout.counter_bytes, 384u);
    }
}

TEST(RegionTemplateTest, LayoutMaxDepthGoldenVector) {
    spsc_queue::SpscQueueLayout layout{};
    ASSERT_TRUE(spsc_queue::SpscQueueLayout::create(spsc_queue::kMaxDepth, 64, 64, &layout));
    EXPECT_EQ(layout.output_desc_offset, spsc_queue::kMaxDepth * 32u);
    EXPECT_EQ(layout.input_arena_offset, spsc_queue::kMaxDepth * 64u);
    EXPECT_EQ(layout.output_arena_offset, spsc_queue::kMaxDepth * 64u + 64u);
    EXPECT_EQ(layout.payload_bytes, spsc_queue::kMaxDepth * 64u + 128u);
    EXPECT_EQ(layout.input_arena_offset % 64u, 0u);
    EXPECT_EQ(layout.output_arena_offset % 64u, 0u);
}

TEST(RegionTemplateTest, LayoutRejectsInvalidDepthAndArenaValues) {
    spsc_queue::SpscQueueLayout layout{};
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(3, 64, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(0, 64, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create((1ull << 30) + 1, 64, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, 0, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, 65, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, 64, 0, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, 64, 63, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, 64, 64, nullptr));
}

TEST(RegionTemplateTest, LayoutOverflowFailsClosedWithoutModifyingOutput) {
    spsc_queue::SpscQueueLayout layout{
        7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    };
    const spsc_queue::SpscQueueLayout original = layout;

    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, std::numeric_limits<uint64_t>::max() - 63, 64, &layout));
    EXPECT_EQ(layout.depth, original.depth);
    EXPECT_EQ(layout.payload_bytes, original.payload_bytes);

    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(1, 64, std::numeric_limits<uint64_t>::max() - 63, &layout));
    EXPECT_EQ(layout.depth, original.depth);
}

TEST(RegionTemplateTest, CheckedMulAddAlignOverflowBoundaries) {
    EXPECT_TRUE(spsc_queue::mul_overflows(1ull << 63, 2));
    EXPECT_TRUE(spsc_queue::add_overflows(std::numeric_limits<uint64_t>::max(), 1));
    uint64_t aligned = 0;
    EXPECT_FALSE(spsc_queue::align_up(std::numeric_limits<uint64_t>::max() - 31, 64, &aligned));
    EXPECT_FALSE(spsc_queue::mul_overflows(spsc_queue::kMaxDepth, 32));
    EXPECT_TRUE(spsc_queue::align_up(64, 64, &aligned));
    EXPECT_EQ(aligned, 64u);
    EXPECT_TRUE(spsc_queue::align_up(65, 64, &aligned));
    EXPECT_EQ(aligned, 128u);
}

TEST(RegionTemplateTest, OpcodeAndDescriptorBytesGoldenVector) {
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::INVALID), 0u);
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::DATA), 1u);
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::STOP), 2u);
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::ERROR), 3u);

    spsc_queue::SpscQueueDescriptor descriptor{
        7,
        static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::ERROR),
        128,
        16,
    };
    uint8_t encoded[32] = {};
    ASSERT_TRUE(spsc_queue::encode_descriptor(descriptor, encoded, sizeof(encoded)));
    EXPECT_EQ(std::memcmp(encoded, kDescriptorGoldenBytes, sizeof(kDescriptorGoldenBytes)), 0);

    spsc_queue::SpscQueueDescriptor decoded{};
    ASSERT_TRUE(spsc_queue::decode_descriptor(kDescriptorGoldenBytes, sizeof(kDescriptorGoldenBytes), &decoded));
    EXPECT_EQ(decoded.seq, 7u);
    EXPECT_EQ(decoded.opcode, 3u);
    EXPECT_EQ(decoded.payload_offset, 128u);
    EXPECT_EQ(decoded.payload_nbytes, 16u);
    EXPECT_FALSE(spsc_queue::decode_descriptor(kDescriptorGoldenBytes, 31, &decoded));
}

TEST(RegionTemplateTest, SessionIdentityLittleEndianRoundTrip) {
    uint64_t bits = 0;
    ASSERT_TRUE(spsc_queue::session_instance_id_to_bits(kSessionGoldenBytes, sizeof(kSessionGoldenBytes), &bits));
    EXPECT_EQ(bits, kSessionGoldenBits);

    uint8_t restored[8] = {};
    ASSERT_TRUE(spsc_queue::session_instance_id_from_bits(kSessionGoldenBits, restored, sizeof(restored)));
    EXPECT_EQ(std::memcmp(restored, kSessionGoldenBytes, sizeof(kSessionGoldenBytes)), 0);
}

TEST(RegionTemplateTest, BindingExactTenScalarGoldenVector) {
    spsc_queue::SpscQueueEndpointBinding binding{};
    ASSERT_TRUE(spsc_queue::decode_endpoint_binding(kBindingGolden.data(), kBindingGolden.size(), &binding));
    EXPECT_EQ(binding.magic_version, spsc_queue::kSpscQueueMagicVersion);
    EXPECT_EQ(binding.session_instance_id_bits, kSessionGoldenBits);
    EXPECT_EQ(binding.transaction_id, 42u);
    EXPECT_EQ(binding.payload_base, 0x1000u);
    EXPECT_EQ(binding.payload_bytes, 192u);
    EXPECT_EQ(binding.counter_base, 0x2000u);
    EXPECT_EQ(binding.counter_bytes, 384u);
    EXPECT_EQ(binding.depth, 1u);
    EXPECT_EQ(binding.input_arena_bytes, 64u);
    EXPECT_EQ(binding.output_arena_bytes, 64u);

    std::array<uint64_t, 10> encoded{};
    ASSERT_TRUE(spsc_queue::encode_endpoint_binding(binding, encoded.data(), encoded.size()));
    EXPECT_EQ(encoded, kBindingGolden);
}

TEST(RegionTemplateTest, BindingRejectsCountAndVersionMismatchWithoutWritingOutput) {
    spsc_queue::SpscQueueEndpointBinding original{
        7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
    };
    spsc_queue::SpscQueueEndpointBinding decoded = original;
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(kBindingGolden.data(), 9, &decoded));
    EXPECT_EQ(decoded.magic_version, original.magic_version);
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(kBindingGolden.data(), 11, &decoded));
    EXPECT_EQ(decoded.transaction_id, original.transaction_id);

    std::array<uint64_t, 10> wrong_magic = kBindingGolden;
    wrong_magic[0] = 0x4C33513200010001ull;
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(wrong_magic.data(), wrong_magic.size(), &decoded));
    EXPECT_EQ(decoded.payload_base, original.payload_base);

    std::array<uint64_t, 10> wrong_major = kBindingGolden;
    wrong_major[0] = (static_cast<uint64_t>(spsc_queue::kSpscQueueMagic) << 32) | (2ull << 16);
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(wrong_major.data(), wrong_major.size(), &decoded));

    std::array<uint64_t, 10> wrong_minor = kBindingGolden;
    wrong_minor[0] = (static_cast<uint64_t>(spsc_queue::kSpscQueueMagic) << 32) | (1ull << 16) | 1ull;
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(wrong_minor.data(), wrong_minor.size(), &decoded));
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(nullptr, 10, &decoded));
    EXPECT_FALSE(spsc_queue::encode_endpoint_binding(original, wrong_magic.data(), 10));
}

TEST(RegionTemplateTest, BindingAndDescriptorStaticAbi) {
    static_assert(sizeof(spsc_queue::SpscQueueEndpointBinding) == 80, "binding size");
    static_assert(sizeof(spsc_queue::SpscQueueDescriptor) == 32, "descriptor size");
    static_assert(std::is_standard_layout_v<spsc_queue::SpscQueueEndpointBinding>, "binding layout");
    static_assert(std::is_trivially_copyable_v<spsc_queue::SpscQueueEndpointBinding>, "binding copy");
    EXPECT_EQ(offsetof(spsc_queue::SpscQueueEndpointBinding, output_arena_bytes), 72u);
    EXPECT_EQ(offsetof(spsc_queue::SpscQueueDescriptor, payload_nbytes), 24u);
}

namespace {

enum class FakeAccessKind {
    PayloadRead = 0,
    PayloadWrite = 1,
    CounterTest = 2,
    CounterWait = 3,
    CounterNotify = 4,
};

struct FakeAccess {
    FakeAccessKind kind;
    uint64_t offset;
    uint64_t nbytes;
    int32_t value;
    RegionWaitCmp cmp;
    uint64_t timeout_ns;
};

struct FakeViewError {
    uint32_t kind;
    char message[256];
};

struct FakeState {
    std::vector<uint8_t> payload_mem;
    std::vector<uint8_t> counter_mem;
    std::vector<FakeAccess> log;
    bool sticky_failed = false;
    FakeViewError last_error{};
    bool wait_sticky = false;
    bool fail_peer_abort_notify = false;
    std::function<void()> on_wait;
};

constexpr uint32_t kFakeTimeoutKind = 4;
constexpr uint64_t kSessionBits = 0x0123456789abcdefull;
constexpr uint64_t kTransactionId = 99;
uint64_t g_now_ns = 1000;
bool g_auto_advance = false;

uint64_t test_now_ns() {
    uint64_t now = g_now_ns;
    if (g_auto_advance) {
        g_now_ns += 100;
    }
    return now;
}

void reset_clock() {
    g_now_ns = 1000;
    g_auto_advance = false;
}

void set_error_message(FakeViewError *error, const char *text) {
    const char *src = text == nullptr ? "" : text;
    size_t n = strnlen(src, sizeof(error->message) - 1);
    memcpy(error->message, src, n);
    error->message[n] = '\0';
}

struct FakeRegionView {
    std::shared_ptr<FakeState> state;

    explicit FakeRegionView(uint64_t payload_bytes) :
        state(std::make_shared<FakeState>()) {
        state->payload_mem.assign(static_cast<size_t>(payload_bytes), 0);
        state->counter_mem.assign(static_cast<size_t>(spsc_queue::kCounterBytes), 0);
    }

    static FakeRegionView prefailed() {
        FakeRegionView view(64);
        view.state->sticky_failed = true;
        view.state->last_error.kind = 1;
        set_error_message(&view.state->last_error, "pre-failed view");
        return view;
    }

    bool failed() const { return state->sticky_failed; }
    const FakeViewError &error() const { return state->last_error; }

    uint64_t payload_base() const {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state->payload_mem.data()));
    }
    uint64_t counter_base() const {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state->counter_mem.data()));
    }

    class PayloadPart {
    public:
        explicit PayloadPart(FakeRegionView *view) :
            view_(view) {}

        RegionPartLocalSpan span() const {
            return RegionPartLocalSpan{view_->payload_base(), view_->state->payload_mem.size()};
        }

        bool read(uint64_t offset, uint64_t nbytes, spsc_queue::SpscQueuePayloadView &out) {
            out = spsc_queue::SpscQueuePayloadView{0, 0};
            view_->state->log.push_back(FakeAccess{FakeAccessKind::PayloadRead, offset, nbytes, 0, RegionWaitCmp::EQ, 0}
            );
            if (view_->state->sticky_failed) {
                return false;
            }
            if (nbytes == 0 || offset + nbytes > view_->state->payload_mem.size()) {
                view_->state->last_error.kind = 2;
                set_error_message(&view_->state->last_error, "payload range is out of bounds");
                return false;
            }
            out = spsc_queue::SpscQueuePayloadView{view_->payload_base() + offset, nbytes};
            return true;
        }

        bool write(uint64_t offset, const void *src, uint64_t nbytes) {
            view_->state->log.push_back(
                FakeAccess{FakeAccessKind::PayloadWrite, offset, nbytes, 0, RegionWaitCmp::EQ, 0}
            );
            if (view_->state->sticky_failed) {
                return false;
            }
            if (src == nullptr || nbytes == 0 || offset + nbytes > view_->state->payload_mem.size()) {
                view_->state->last_error.kind = 2;
                set_error_message(&view_->state->last_error, "payload write out of bounds");
                return false;
            }
            memcpy(view_->state->payload_mem.data() + offset, src, static_cast<size_t>(nbytes));
            return true;
        }

    private:
        FakeRegionView *view_;
    };

    class CounterPart {
    public:
        explicit CounterPart(FakeRegionView *view) :
            view_(view) {}

        RegionPartLocalSpan span() const {
            return RegionPartLocalSpan{view_->counter_base(), view_->state->counter_mem.size()};
        }

        bool test(uint64_t offset, int32_t cmp_value, RegionWaitCmp cmp, spsc_queue::SpscQueueCounterSample &out) {
            out = spsc_queue::SpscQueueCounterSample{false, 0};
            view_->state->log.push_back(FakeAccess{FakeAccessKind::CounterTest, offset, 4, cmp_value, cmp, 0});
            int32_t observed = 0;
            if (!load(offset, observed)) {
                return false;
            }
            out = spsc_queue::SpscQueueCounterSample{region_compare_counter(observed, cmp_value, cmp), observed};
            return true;
        }

        bool wait(uint64_t offset, int32_t cmp_value, RegionWaitCmp cmp, uint64_t timeout_ns, int32_t &observed) {
            observed = 0;
            view_->state->log.push_back(FakeAccess{FakeAccessKind::CounterWait, offset, 4, cmp_value, cmp, timeout_ns});
            if (view_->state->on_wait) {
                view_->state->on_wait();
            }
            if (view_->state->wait_sticky) {
                view_->state->sticky_failed = true;
                view_->state->last_error.kind = 5;
                set_error_message(&view_->state->last_error, "injected wait failure");
                return false;
            }
            if (!load(offset, observed)) {
                return false;
            }
            if (region_compare_counter(observed, cmp_value, cmp)) {
                return true;
            }
            view_->state->last_error.kind = kFakeTimeoutKind;
            set_error_message(&view_->state->last_error, "wait timed out");
            return false;
        }

        bool notify(uint64_t offset, int32_t value, RegionNotifyOp op) {
            view_->state->log.push_back(
                FakeAccess{FakeAccessKind::CounterNotify, offset, 4, value, RegionWaitCmp::EQ, 0}
            );
            (void)op;
            if (view_->state->fail_peer_abort_notify && offset == spsc_queue::kPeerAbortOffset) {
                return false;
            }
            if (view_->state->sticky_failed) {
                return false;
            }
            return store(offset, value);
        }

    private:
        bool load(uint64_t offset, int32_t &out) {
            if (view_->state->sticky_failed) {
                return false;
            }
            if (offset + 4 > view_->state->counter_mem.size() || (offset % 4) != 0) {
                view_->state->last_error.kind = 2;
                set_error_message(&view_->state->last_error, "invalid counter address");
                return false;
            }
            memcpy(&out, view_->state->counter_mem.data() + offset, sizeof(out));
            return true;
        }

        bool store(uint64_t offset, int32_t value) {
            if (offset + 4 > view_->state->counter_mem.size() || (offset % 4) != 0) {
                view_->state->last_error.kind = 2;
                set_error_message(&view_->state->last_error, "invalid counter address");
                return false;
            }
            memcpy(view_->state->counter_mem.data() + offset, &value, sizeof(value));
            return true;
        }

        FakeRegionView *view_;
    };

    PayloadPart payload() { return PayloadPart(this); }
    CounterPart counter() { return CounterPart(this); }
};

spsc_queue::SpscQueueLayout make_layout(uint64_t depth, uint64_t input_arena, uint64_t output_arena) {
    spsc_queue::SpscQueueLayout layout{};
    EXPECT_TRUE(spsc_queue::SpscQueueLayout::create(depth, input_arena, output_arena, &layout));
    return layout;
}

spsc_queue::SpscQueueEndpointBinding
make_binding(const FakeRegionView &view, const spsc_queue::SpscQueueLayout &layout) {
    spsc_queue::SpscQueueEndpointBinding binding{};
    binding.magic_version = spsc_queue::kSpscQueueMagicVersion;
    binding.session_instance_id_bits = kSessionBits;
    binding.transaction_id = kTransactionId;
    binding.payload_base = view.payload_base();
    binding.payload_bytes = layout.payload_bytes;
    binding.counter_base = view.counter_base();
    binding.counter_bytes = layout.counter_bytes;
    binding.depth = layout.depth;
    binding.input_arena_bytes = layout.input_arena_bytes;
    binding.output_arena_bytes = layout.output_arena_bytes;
    return binding;
}

void store_counter(FakeState *state, uint64_t offset, int32_t value) {
    memcpy(state->counter_mem.data() + offset, &value, sizeof(value));
}

int32_t load_counter(const FakeState *state, uint64_t offset) {
    int32_t value = 0;
    memcpy(&value, state->counter_mem.data() + offset, sizeof(value));
    return value;
}

void plant_descriptor(
    FakeState *state, uint64_t slot_offset, uint64_t seq, spsc_queue::SpscQueueOpcode opcode, uint64_t payload_offset,
    uint64_t nbytes
) {
    spsc_queue::SpscQueueDescriptor descriptor{seq, static_cast<uint64_t>(opcode), payload_offset, nbytes};
    uint8_t encoded[32];
    ASSERT_TRUE(spsc_queue::encode_descriptor(descriptor, encoded, sizeof(encoded)));
    memcpy(state->payload_mem.data() + slot_offset, encoded, sizeof(encoded));
}

void plant_input(
    FakeState *state, const spsc_queue::SpscQueueLayout &layout, uint64_t seq, spsc_queue::SpscQueueOpcode opcode,
    uint64_t nbytes, const void *bytes
) {
    uint64_t slot_index = (seq - 1) & (layout.depth - 1);
    uint64_t slot_offset = layout.input_desc_offset + slot_index * spsc_queue::kDescriptorBytes;
    uint64_t payload_offset = nbytes == 0 ? 0 : layout.input_arena_offset;
    plant_descriptor(state, slot_offset, seq, opcode, payload_offset, nbytes);
    if (nbytes != 0 && bytes != nullptr) {
        memcpy(state->payload_mem.data() + payload_offset, bytes, static_cast<size_t>(nbytes));
    }
    store_counter(state, layout.input_desc_tail_offset, static_cast<int32_t>(seq));
}

bool has_session_marker(const char *message) { return std::string(message).find("session=0x") != std::string::npos; }

bool has_sixteen_hex_session(const char *message) {
    const char *p = std::strstr(message, "session=0x");
    if (p == nullptr) {
        return false;
    }
    p += 10;
    for (int i = 0; i < 16; ++i) {
        char c = p[i];
        bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!hex) {
            return false;
        }
    }
    return true;
}

spsc_queue::MonotonicClock test_clock() { return spsc_queue::MonotonicClock{&test_now_ns}; }

using Queue1 = spsc_queue::SpscQueueEndpoint<FakeRegionView, 1>;
using Queue2 = spsc_queue::SpscQueueEndpoint<FakeRegionView, 2>;

static_assert(!std::is_copy_constructible_v<Queue1>, "endpoint must not copy");
static_assert(!std::is_move_constructible_v<Queue1>, "endpoint must not move");
static_assert(!std::is_copy_assignable_v<Queue1>, "endpoint must not copy-assign");
static_assert(!std::is_move_assignable_v<Queue1>, "endpoint must not move-assign");

size_t count_kind(const std::vector<FakeAccess> &log, FakeAccessKind kind) {
    size_t n = 0;
    for (const auto &entry : log) {
        if (entry.kind == kind) {
            n += 1;
        }
    }
    return n;
}

bool has_notify(const std::vector<FakeAccess> &log, uint64_t offset) {
    for (const auto &entry : log) {
        if (entry.kind == FakeAccessKind::CounterNotify && entry.offset == offset) {
            return true;
        }
    }
    return false;
}

size_t count_notify(const std::vector<FakeAccess> &log, uint64_t offset) {
    size_t n = 0;
    for (const auto &entry : log) {
        if (entry.kind == FakeAccessKind::CounterNotify && entry.offset == offset) {
            n += 1;
        }
    }
    return n;
}

}  // namespace

TEST(RegionTemplateTest, ConstructionRejectsNullClockWithoutSharedAccess) {
    reset_clock();
    auto layout = make_layout(2, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), spsc_queue::MonotonicClock{nullptr});
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::BAD_ARGUMENT);
    EXPECT_TRUE(has_session_marker(queue.error().message));
    EXPECT_TRUE(has_sixteen_hex_session(queue.error().message));
    EXPECT_EQ(queue.error().session_instance_id_bits, kSessionBits);
    EXPECT_EQ(queue.error().transaction_id, kTransactionId);
    EXPECT_TRUE(state->log.empty());
}

TEST(RegionTemplateTest, ConstructionRejectsInvalidBindingWithoutSharedAccessOrIdentity) {
    reset_clock();
    auto layout = make_layout(2, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    binding.magic_version = 0x4C33513200010001ull;
    Queue1 queue(binding, std::move(view), test_clock());
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::BAD_BINDING);
    EXPECT_FALSE(has_session_marker(queue.error().message));
    EXPECT_EQ(std::string(queue.error().message).find("transaction="), std::string::npos);
    EXPECT_TRUE(state->log.empty());
}

TEST(RegionTemplateTest, ConstructionRejectsFailedViewWithoutSharedAccess) {
    reset_clock();
    auto layout = make_layout(2, 64, 64);
    FakeRegionView view = FakeRegionView::prefailed();
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::ENDPOINT_ERROR);
    EXPECT_NE(std::string(queue.error().message).find("pre-failed view"), std::string::npos);
    EXPECT_TRUE(has_session_marker(queue.error().message));
    EXPECT_TRUE(state->log.empty());
}

TEST(RegionTemplateTest, ConstructionRejectsMaxInflightGreaterThanDepthWithoutSharedAccess) {
    reset_clock();
    auto layout = make_layout(2, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    spsc_queue::SpscQueueEndpoint<FakeRegionView, 4> queue(binding, std::move(view), test_clock());
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::BAD_ARGUMENT);
    EXPECT_TRUE(state->log.empty());
}

TEST(RegionTemplateTest, EndpointMaxInflightOneDuplexZeroByteStopAndPublishOrder) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    ASSERT_TRUE(queue.live());
    EXPECT_TRUE(state->log.empty());

    const char payload[] = "abcdefgh";
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::DATA, 8, payload);
    spsc_queue::SpscQueueInputHandle handle{};
    ASSERT_TRUE(queue.input().try_peek(handle));
    EXPECT_EQ(handle.seq, 1u);
    EXPECT_EQ(handle.opcode, spsc_queue::SpscQueueOpcode::DATA);
    EXPECT_EQ(handle.payload_nbytes, 8u);
    EXPECT_EQ(
        std::memcmp(reinterpret_cast<const void *>(static_cast<uintptr_t>(handle.payload.local_addr)), payload, 8), 0
    );
    EXPECT_FALSE(queue.input().drained());
    ASSERT_TRUE(queue.input().release(handle));
    EXPECT_EQ(load_counter(state.get(), layout.input_desc_head_offset), 1);
    EXPECT_FALSE(queue.input().try_peek(handle));

    plant_input(state.get(), layout, 2, spsc_queue::SpscQueueOpcode::STOP, 0, nullptr);
    ASSERT_TRUE(queue.input().try_peek(handle));
    EXPECT_EQ(handle.opcode, spsc_queue::SpscQueueOpcode::STOP);
    EXPECT_EQ(handle.payload.local_addr, 0u);
    EXPECT_EQ(handle.payload.nbytes, 0u);
    ASSERT_TRUE(queue.input().release(handle));
    EXPECT_TRUE(queue.input().drained());
    EXPECT_FALSE(queue.input().try_peek(handle));

    spsc_queue::SpscQueueOutputReservation zero{};
    ASSERT_TRUE(queue.output().try_reserve(0, zero));
    EXPECT_EQ(zero.payload.local_addr, 0u);
    EXPECT_EQ(zero.payload.nbytes, 0u);
    ASSERT_TRUE(queue.output().publish(zero, spsc_queue::SpscQueueOpcode::DATA));

    spsc_queue::SpscQueueOutputReservation reserved{};
    ASSERT_TRUE(queue.output().try_reserve(8, reserved));
    EXPECT_EQ(reserved.payload.nbytes, 8u);
    EXPECT_NE(reserved.payload.local_addr, 0u);
    memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(reserved.payload.local_addr)), payload, 8);
    size_t writes_before = count_kind(state->log, FakeAccessKind::PayloadWrite);
    ASSERT_TRUE(queue.output().publish(reserved, spsc_queue::SpscQueueOpcode::ERROR));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);

    std::vector<FakeAccess> writes;
    for (const auto &entry : state->log) {
        if (entry.kind == FakeAccessKind::PayloadWrite) {
            writes.push_back(entry);
        }
    }
    ASSERT_GE(writes.size(), writes_before + 2);
    const FakeAccess &fields = writes[writes.size() - 2];
    const FakeAccess &seq = writes[writes.size() - 1];
    uint64_t slot1 = layout.output_desc_offset + spsc_queue::kDescriptorBytes;
    EXPECT_EQ(fields.offset, slot1 + 8);
    EXPECT_EQ(fields.nbytes, 24u);
    EXPECT_EQ(seq.offset, slot1);
    EXPECT_EQ(seq.nbytes, 8u);
    ASSERT_FALSE(state->log.empty());
    EXPECT_EQ(state->log.back().kind, FakeAccessKind::CounterNotify);
    EXPECT_EQ(state->log.back().offset, layout.output_desc_tail_offset);
    for (const auto &entry : writes) {
        EXPECT_FALSE(entry.offset >= layout.output_arena_offset && entry.offset < layout.payload_bytes);
    }
}

TEST(RegionTemplateTest, StopExtraSlotAndCompletedPrefix) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    ASSERT_TRUE(queue.live());

    const char first[] = "12345678";
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::DATA, 8, first);
    spsc_queue::SpscQueueInputHandle data{};
    ASSERT_TRUE(queue.input().try_peek(data));
    spsc_queue::SpscQueueInputHandle blocked{};
    EXPECT_FALSE(queue.input().try_peek(blocked));

    plant_descriptor(
        state.get(), layout.input_desc_offset + spsc_queue::kDescriptorBytes, 2, spsc_queue::SpscQueueOpcode::STOP, 0, 0
    );
    store_counter(state.get(), layout.input_desc_tail_offset, 2);
    spsc_queue::SpscQueueInputHandle stop{};
    ASSERT_TRUE(queue.input().try_peek(stop));
    EXPECT_EQ(stop.opcode, spsc_queue::SpscQueueOpcode::STOP);
    EXPECT_FALSE(queue.input().drained());
    EXPECT_EQ(load_counter(state.get(), layout.input_desc_head_offset), 0);

    ASSERT_TRUE(queue.input().release(stop));
    EXPECT_FALSE(queue.input().drained());
    EXPECT_EQ(load_counter(state.get(), layout.input_desc_head_offset), 0);
    ASSERT_TRUE(queue.input().release(data));
    EXPECT_TRUE(queue.input().drained());
    EXPECT_EQ(load_counter(state.get(), layout.input_desc_head_offset), 2);
}

TEST(RegionTemplateTest, MaxInflightTwoWindowAndStopExtraSlot) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue2 queue(binding, std::move(view), test_clock());
    ASSERT_TRUE(queue.live());

    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::DATA, 0, nullptr);
    spsc_queue::SpscQueueInputHandle a{};
    spsc_queue::SpscQueueInputHandle b{};
    spsc_queue::SpscQueueInputHandle extra{};
    ASSERT_TRUE(queue.input().try_peek(a));
    plant_descriptor(
        state.get(), layout.input_desc_offset + spsc_queue::kDescriptorBytes, 2, spsc_queue::SpscQueueOpcode::DATA, 0, 0
    );
    store_counter(state.get(), layout.input_desc_tail_offset, 2);
    ASSERT_TRUE(queue.input().try_peek(b));
    plant_descriptor(
        state.get(), layout.input_desc_offset + 2 * spsc_queue::kDescriptorBytes, 3, spsc_queue::SpscQueueOpcode::DATA,
        0, 0
    );
    store_counter(state.get(), layout.input_desc_tail_offset, 3);
    EXPECT_FALSE(queue.input().try_peek(extra));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    plant_descriptor(
        state.get(), layout.input_desc_offset + 2 * spsc_queue::kDescriptorBytes, 3, spsc_queue::SpscQueueOpcode::STOP,
        0, 0
    );
    ASSERT_TRUE(queue.input().try_peek(extra));
    EXPECT_EQ(extra.opcode, spsc_queue::SpscQueueOpcode::STOP);
    ASSERT_TRUE(queue.input().release(a));
    ASSERT_TRUE(queue.input().release(b));
    ASSERT_TRUE(queue.input().release(extra));
    EXPECT_TRUE(queue.input().drained());
}

TEST(RegionTemplateTest, OutputReservationOwnershipLifetimeAndOversize) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    ASSERT_TRUE(queue.live());

    spsc_queue::SpscQueueOutputReservation oversize{};
    EXPECT_FALSE(queue.output().try_reserve(layout.output_arena_bytes + 1, oversize));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_FALSE(oversize.valid);
    EXPECT_FALSE(queue.output().reserve(layout.output_arena_bytes + 1, 50, oversize));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_TRUE(state->log.empty());

    spsc_queue::SpscQueueOutputReservation reserved{};
    ASSERT_TRUE(queue.output().try_reserve(8, reserved));
    EXPECT_TRUE(reserved.valid);
    spsc_queue::SpscQueueOutputReservation second{};
    EXPECT_FALSE(queue.output().try_reserve(8, second));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::OWNERSHIP);

    FakeRegionView view2(layout.payload_bytes);
    auto state2 = view2.state;
    auto binding2 = make_binding(view2, layout);
    Queue1 queue2(binding2, std::move(view2), test_clock());
    spsc_queue::SpscQueueOutputReservation ok{};
    ASSERT_TRUE(queue2.output().try_reserve(8, ok));
    spsc_queue::SpscQueueOutputReservation stale = ok;
    ASSERT_TRUE(queue2.output().publish(ok, spsc_queue::SpscQueueOpcode::DATA));
    EXPECT_FALSE(queue2.output().publish(stale, spsc_queue::SpscQueueOpcode::DATA));
    EXPECT_EQ(queue2.error().kind, spsc_queue::SpscQueueErrorKind::OWNERSHIP);
    EXPECT_TRUE(has_session_marker(queue2.error().message));
}

TEST(RegionTemplateTest, WaitTimeoutKeepsErrorNoneAndIgnoresStaleViewKind) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    spsc_queue::SpscQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().peek(50, handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_EQ(state->last_error.kind, kFakeTimeoutKind);
    EXPECT_GE(count_kind(state->log, FakeAccessKind::CounterWait), 1u);

    const char payload[] = "abcdefgh";
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::DATA, 8, payload);
    ASSERT_TRUE(queue.input().try_peek(handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_EQ(handle.seq, 1u);
}

TEST(RegionTemplateTest, WaitStickyFailureIsEndpointError) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    state->wait_sticky = true;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    spsc_queue::SpscQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().peek(50, handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::ENDPOINT_ERROR);
    EXPECT_NE(std::string(queue.error().message).find("injected wait failure"), std::string::npos);
    EXPECT_TRUE(has_session_marker(queue.error().message));
    EXPECT_TRUE(has_notify(state->log, layout.peer_abort_offset));
}

TEST(RegionTemplateTest, PeerAbortSampledAtDeadlineWithoutLocalAbortNotify) {
    reset_clock();
    g_auto_advance = true;
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    store_counter(state.get(), layout.initiator_abort_offset, 1);
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    spsc_queue::SpscQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().peek(10, handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::REMOTE_ABORTED);
    EXPECT_TRUE(has_session_marker(queue.error().message));
    EXPECT_FALSE(has_notify(state->log, layout.peer_abort_offset));
}

TEST(RegionTemplateTest, FirstErrorWinsWhenAbortNotifyFails) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    state->fail_peer_abort_notify = true;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::ERROR, 0, nullptr);
    spsc_queue::SpscQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().try_peek(handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_NE(std::string(queue.error().message).find("abort notify failed"), std::string::npos);
    EXPECT_NE(std::string(queue.error().message).find("invalid input opcode"), std::string::npos);
}

TEST(RegionTemplateTest, CounterReconstructionInvalidDeltaPoisons) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    store_counter(state.get(), layout.input_desc_tail_offset, 5);
    spsc_queue::SpscQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().try_peek(handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_NE(std::string(queue.error().message).find("counter reconstruction failed"), std::string::npos);
}

TEST(RegionTemplateTest, OutputWrapReplayDoesNotFlushBorrowedPayload) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    spsc_queue::SpscQueueOutputReservation first{};
    ASSERT_TRUE(queue.output().try_reserve(40, first));
    EXPECT_EQ(first.payload_offset, layout.output_arena_offset);
    ASSERT_TRUE(queue.output().publish(first, spsc_queue::SpscQueueOpcode::DATA));
    store_counter(state.get(), layout.output_desc_head_offset, 1);
    spsc_queue::SpscQueueOutputReservation wrapped{};
    ASSERT_TRUE(queue.output().try_reserve(40, wrapped));
    EXPECT_EQ(wrapped.payload_offset, layout.output_arena_offset);
    for (const auto &entry : state->log) {
        if (entry.kind == FakeAccessKind::PayloadWrite) {
            EXPECT_LT(entry.offset, layout.output_arena_offset);
        }
    }
}

TEST(RegionTemplateTest, EncodeRejectsZeroTransactionWithoutWritingOutput) {
    spsc_queue::SpscQueueEndpointBinding binding{};
    ASSERT_TRUE(spsc_queue::decode_endpoint_binding(kBindingGolden.data(), kBindingGolden.size(), &binding));
    binding.transaction_id = 0;
    std::array<uint64_t, 10> encoded{};
    encoded.fill(0x1111111111111111ull);
    auto before = encoded;
    EXPECT_FALSE(spsc_queue::encode_endpoint_binding(binding, encoded.data(), encoded.size()));
    EXPECT_EQ(encoded, before);
}

TEST(RegionTemplateTest, DecodeRejectsZeroTransactionWithoutWritingTarget) {
    spsc_queue::SpscQueueEndpointBinding original{
        7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
    };
    spsc_queue::SpscQueueEndpointBinding decoded = original;
    std::array<uint64_t, 10> scalars = kBindingGolden;
    scalars[2] = 0;
    EXPECT_FALSE(spsc_queue::decode_endpoint_binding(scalars.data(), scalars.size(), &decoded));
    EXPECT_EQ(decoded.magic_version, original.magic_version);
    EXPECT_EQ(decoded.session_instance_id_bits, original.session_instance_id_bits);
    EXPECT_EQ(decoded.transaction_id, original.transaction_id);
    EXPECT_EQ(decoded.payload_base, original.payload_base);
    EXPECT_EQ(decoded.depth, original.depth);
}

TEST(RegionTemplateTest, DecodeAcceptsAllZeroSessionBits) {
    std::array<uint64_t, 10> scalars = kBindingGolden;
    scalars[1] = 0;
    spsc_queue::SpscQueueEndpointBinding binding{};
    ASSERT_TRUE(spsc_queue::decode_endpoint_binding(scalars.data(), scalars.size(), &binding));
    EXPECT_EQ(binding.session_instance_id_bits, 0u);
    EXPECT_EQ(binding.transaction_id, 42u);
}

TEST(RegionTemplateTest, ConstructRejectsZeroTransactionWithoutViewClockOrIdentity) {
    reset_clock();
    auto layout = make_layout(2, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    binding.transaction_id = 0;
    uint64_t now_before = g_now_ns;
    Queue1 queue(binding, std::move(view), test_clock());
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::BAD_BINDING);
    EXPECT_FALSE(has_session_marker(queue.error().message));
    EXPECT_EQ(std::string(queue.error().message).find("transaction="), std::string::npos);
    EXPECT_EQ(queue.error().transaction_id, 0u);
    EXPECT_TRUE(state->log.empty());
    EXPECT_EQ(g_now_ns, now_before);
}

TEST(RegionTemplateTest, CounterLow32PreservesSignedBitPattern) {
    EXPECT_EQ(spsc_queue::counter_low32(0x7fffffffull), 2147483647);
    EXPECT_EQ(spsc_queue::counter_low32(0x80000000ull), static_cast<int32_t>(0x80000000u));
    EXPECT_EQ(spsc_queue::counter_low32(0xffffffffull), -1);
    EXPECT_EQ(spsc_queue::counter_low32(0x100000000ull), 0);
    uint64_t local = 0x7fffffffull;
    ASSERT_TRUE(spsc_queue::reconstruct_counter(spsc_queue::counter_low32(0x80000000ull), 4, &local));
    EXPECT_EQ(local, 0x80000000ull);
    local = 0xffffffffull;
    ASSERT_TRUE(spsc_queue::reconstruct_counter(0, 4, &local));
    EXPECT_EQ(local, 0x100000000ull);
}

TEST(RegionTemplateTest, ZeroTimeoutPeekAndReserveAreNoAttempt) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    ASSERT_TRUE(queue.live());
    const char payload[] = "abcdefgh";
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::DATA, 8, payload);
    size_t log_before = state->log.size();
    uint64_t now_before = g_now_ns;
    spsc_queue::SpscQueueInputHandle peek_out{};
    peek_out.seq = 7;
    EXPECT_FALSE(queue.input().peek(0, peek_out));
    EXPECT_EQ(peek_out.seq, 0u);
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_TRUE(queue.live());
    EXPECT_EQ(state->log.size(), log_before);
    EXPECT_EQ(g_now_ns, now_before);

    spsc_queue::SpscQueueOutputReservation reserve_out{};
    reserve_out.seq = 9;
    EXPECT_FALSE(queue.output().reserve(8, 0, reserve_out));
    EXPECT_FALSE(reserve_out.valid);
    EXPECT_EQ(reserve_out.seq, 0u);
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_TRUE(queue.live());
    EXPECT_EQ(state->log.size(), log_before);
    EXPECT_EQ(g_now_ns, now_before);

    spsc_queue::SpscQueueInputHandle ready{};
    ASSERT_TRUE(queue.input().try_peek(ready));
    EXPECT_EQ(ready.seq, 1u);
    ASSERT_TRUE(queue.input().release(ready));
}

TEST(RegionTemplateTest, ZeroTimeoutReserveLeavesActiveReservationAndSkipsOversize) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    ASSERT_TRUE(queue.live());
    spsc_queue::SpscQueueOutputReservation first{};
    ASSERT_TRUE(queue.output().try_reserve(8, first));
    size_t log_before = state->log.size();
    spsc_queue::SpscQueueOutputReservation ignored{};
    EXPECT_FALSE(queue.output().reserve(8, 0, ignored));
    EXPECT_FALSE(ignored.valid);
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_TRUE(queue.live());
    EXPECT_EQ(state->log.size(), log_before);
    EXPECT_TRUE(first.valid);
    ASSERT_TRUE(queue.output().publish(first, spsc_queue::SpscQueueOpcode::DATA));

    FakeRegionView view2(layout.payload_bytes);
    auto state2 = view2.state;
    auto binding2 = make_binding(view2, layout);
    Queue1 queue2(binding2, std::move(view2), test_clock());
    spsc_queue::SpscQueueOutputReservation oversize{};
    EXPECT_FALSE(queue2.output().reserve(layout.output_arena_bytes + 1, 0, oversize));
    EXPECT_FALSE(oversize.valid);
    EXPECT_EQ(queue2.error().kind, spsc_queue::SpscQueueErrorKind::NONE);
    EXPECT_TRUE(queue2.live());
    EXPECT_TRUE(state2->log.empty());
    spsc_queue::SpscQueueOutputReservation ok{};
    ASSERT_TRUE(queue2.output().try_reserve(8, ok));
}

TEST(RegionTemplateTest, ZeroTimeoutPreservesPreexistingTerminalError) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::ERROR, 0, nullptr);
    spsc_queue::SpscQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().try_peek(handle));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    size_t log_before = state->log.size();
    EXPECT_FALSE(queue.input().peek(0, handle));
    spsc_queue::SpscQueueOutputReservation reserved{};
    EXPECT_FALSE(queue.output().reserve(8, 0, reserved));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_NE(std::string(queue.error().message).find("invalid input opcode"), std::string::npos);
    EXPECT_EQ(state->log.size(), log_before);
}

TEST(RegionTemplateTest, PublishStopPoisonsWithoutTailAdvance) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    spsc_queue::SpscQueueOutputReservation reserved{};
    ASSERT_TRUE(queue.output().try_reserve(8, reserved));
    EXPECT_FALSE(queue.output().publish(reserved, spsc_queue::SpscQueueOpcode::STOP));
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_NE(std::string(queue.error().message).find("invalid output opcode"), std::string::npos);
    EXPECT_EQ(count_notify(state->log, spsc_queue::kPeerAbortOffset), 1u);
    EXPECT_FALSE(has_notify(state->log, layout.output_desc_tail_offset));
    EXPECT_EQ(load_counter(state.get(), layout.output_desc_tail_offset), 0);
    EXPECT_FALSE(queue.output().publish(reserved, spsc_queue::SpscQueueOpcode::DATA));
    EXPECT_EQ(count_notify(state->log, spsc_queue::kPeerAbortOffset), 1u);
    EXPECT_NE(std::string(queue.error().message).find("invalid output opcode"), std::string::npos);
}

TEST(RegionTemplateTest, InputPayloadOutsideArenaPoisonsWithoutPayloadRead) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    auto binding = make_binding(view, layout);
    Queue1 queue(binding, std::move(view), test_clock());
    uint64_t outside = layout.input_arena_offset + layout.input_arena_bytes;
    plant_descriptor(state.get(), layout.input_desc_offset, 1, spsc_queue::SpscQueueOpcode::DATA, outside, 8);
    store_counter(state.get(), layout.input_desc_tail_offset, 1);
    spsc_queue::SpscQueueInputHandle handle{};
    handle.seq = 11;
    EXPECT_FALSE(queue.input().try_peek(handle));
    EXPECT_EQ(handle.seq, 0u);
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_NE(std::string(queue.error().message).find("input payload out of arena"), std::string::npos);
    for (const auto &entry : state->log) {
        if (entry.kind == FakeAccessKind::PayloadRead) {
            EXPECT_NE(entry.offset, outside);
        }
    }
    EXPECT_EQ(count_notify(state->log, spsc_queue::kPeerAbortOffset), 1u);
    EXPECT_FALSE(has_notify(state->log, layout.input_desc_head_offset));
    EXPECT_FALSE(queue.input().try_peek(handle));
    EXPECT_EQ(count_notify(state->log, spsc_queue::kPeerAbortOffset), 1u);
}

TEST(RegionTemplateTest, SecondNoncanonicalInputReplayPoisonsAndAbortNotifyFailureIsSecondary) {
    reset_clock();
    auto layout = make_layout(4, 64, 64);
    FakeRegionView view(layout.payload_bytes);
    auto state = view.state;
    state->fail_peer_abort_notify = true;
    auto binding = make_binding(view, layout);
    Queue2 queue(binding, std::move(view), test_clock());
    const char first[] = "12345678";
    plant_input(state.get(), layout, 1, spsc_queue::SpscQueueOpcode::DATA, 8, first);
    spsc_queue::SpscQueueInputHandle acquired{};
    ASSERT_TRUE(queue.input().try_peek(acquired));
    EXPECT_EQ(acquired.payload_offset, layout.input_arena_offset);
    plant_descriptor(
        state.get(), layout.input_desc_offset + spsc_queue::kDescriptorBytes, 2, spsc_queue::SpscQueueOpcode::DATA,
        layout.input_arena_offset, 8
    );
    store_counter(state.get(), layout.input_desc_tail_offset, 2);
    size_t arena_reads_before = 0;
    for (const auto &entry : state->log) {
        if (entry.kind == FakeAccessKind::PayloadRead && entry.offset == layout.input_arena_offset) {
            arena_reads_before += 1;
        }
    }
    spsc_queue::SpscQueueInputHandle second{};
    EXPECT_FALSE(queue.input().try_peek(second));
    EXPECT_EQ(second.seq, 0u);
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_NE(std::string(queue.error().message).find("payload replay offset mismatch"), std::string::npos);
    EXPECT_NE(std::string(queue.error().message).find("abort notify failed"), std::string::npos);
    size_t arena_reads_after = 0;
    for (const auto &entry : state->log) {
        if (entry.kind == FakeAccessKind::PayloadRead && entry.offset == layout.input_arena_offset) {
            arena_reads_after += 1;
        }
    }
    EXPECT_EQ(arena_reads_after, arena_reads_before);
    EXPECT_EQ(count_notify(state->log, spsc_queue::kPeerAbortOffset), 1u);
    EXPECT_FALSE(has_notify(state->log, layout.input_desc_head_offset));
    EXPECT_FALSE(queue.input().try_peek(second));
    EXPECT_EQ(count_notify(state->log, spsc_queue::kPeerAbortOffset), 1u);
    EXPECT_NE(std::string(queue.error().message).find("payload replay offset mismatch"), std::string::npos);
}
