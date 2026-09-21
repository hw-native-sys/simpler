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
// Wire-ABI tests for Buffer / Tensor (buffer.h); the contract they pin is described
// in docs/buffer-abi.md. Byte layout is pinned by static_assert in the header; these tests
// pin the sizes, enum values, and the shared receive-side gate from the outside.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "buffer.h"
#include "task_args.h"

namespace {

CanonicalIdentity make_identity() {
    CanonicalIdentity id{};
    for (uint32_t i = 0; i < OWNER_INSTANCE_ID_BYTES; ++i)
        id.owner_instance_id[i] = static_cast<uint8_t>(0xA0 + i);
    id.buffer_id = 0x0102030405060708ULL;
    id.generation = 7;
    return id;
}

// A POSIX_SHM body is the shm name the consumer opens; a valid descriptor always carries one.
constexpr const char *kShmName = "psm_deadbeef";

Tensor make_tensor() {
    Tensor r{};
    r.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    r.buffer.identity = make_identity();
    r.buffer.nbytes = 8192;  // must cover byte_offset + the strided extent below
    r.buffer.body_len = static_cast<uint16_t>(std::strlen(kShmName));
    std::memcpy(r.buffer.body, kShmName, r.buffer.body_len);
    r.byte_offset = 4096;
    r.ndims = 3;
    r.shapes[0] = 2;
    r.shapes[1] = 4;
    r.shapes[2] = 8;
    r.strides[0] = 32;
    r.strides[1] = 8;
    r.strides[2] = 1;
    r.dtype = DataType::FLOAT16;
    return r;
}

// An address-bearing backend body is exactly one u64 LE base.
void set_address_body(Tensor &r, uint64_t base) {
    std::memset(r.buffer.body, 0, DESC_MAX_BYTES);
    r.buffer.body_len = static_cast<uint16_t>(BACKEND_ADDRESS_BODY_BYTES);
    std::memcpy(r.buffer.body, &base, sizeof(base));
}

Tensor make_device_tensor() {
    Tensor r = make_tensor();
    r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::DEVICE_MALLOC);
    r.buffer.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    set_address_body(r, 0xDEAD0000ULL);
    return r;
}

// --- Layout / value contracts (frozen ABI) -----------------------------------------------------

TEST(BufferAbi, StructSizesAreFrozen) {
    EXPECT_EQ(sizeof(CanonicalIdentity), 32u);
    EXPECT_EQ(sizeof(Tensor), 144u);
    EXPECT_EQ(sizeof(BufferDescriptor), 88u);
}

TEST(BufferAbi, ConstantsAreFrozen) {
    EXPECT_EQ(BUFFER_DESCRIPTOR_MAGIC, 0x5342);
    EXPECT_EQ(OWNER_INSTANCE_ID_BYTES, 8u);
    EXPECT_EQ(DESC_MAX_BYTES, 32u);
}

TEST(BufferAbi, EnumValuesAreFrozen) {
    EXPECT_EQ(static_cast<uint8_t>(AddressSpace::HOST), 0);
    EXPECT_EQ(static_cast<uint8_t>(AddressSpace::DEVICE), 1);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::READ), 0);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::WRITE), 1);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::READWRITE), 2);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::FORK_SHM), 0);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::POSIX_SHM), 1);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::VMM_WINDOW), 2);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR), 3);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::DEVICE_MALLOC), 4);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::FORK_COW), 5);
}

// --- memcpy round trip -------------------------------------------------------------------------

TEST(BufferAbi, TensorSurvivesByteRoundTrip) {
    Tensor src = make_tensor();
    uint8_t bytes[sizeof(Tensor)];
    std::memcpy(bytes, &src, sizeof(Tensor));
    Tensor dst{};
    std::memcpy(&dst, bytes, sizeof(Tensor));
    EXPECT_EQ(std::memcmp(&src, &dst, sizeof(Tensor)), 0);
    EXPECT_EQ(dst.byte_offset, 4096u);
    EXPECT_EQ(dst.dtype, DataType::FLOAT16);
    EXPECT_EQ(dst.strides[0], 32u);
    EXPECT_EQ(dst.buffer.identity, src.buffer.identity);
}

TEST(BufferAbi, DescriptorSurvivesByteRoundTrip) {
    BufferDescriptor src{};
    src.magic = BUFFER_DESCRIPTOR_MAGIC;
    src.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    src.access = static_cast<uint8_t>(AccessMode::READWRITE);
    src.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    src.owner_worker_path_id = 3;
    src.identity = make_identity();
    src.nbytes = 1 << 20;
    const char *body = "psm_deadbeef";
    src.body_len = static_cast<uint16_t>(std::strlen(body));
    std::memcpy(src.body, body, src.body_len);

    uint8_t bytes[sizeof(BufferDescriptor)];
    std::memcpy(bytes, &src, sizeof(BufferDescriptor));
    BufferDescriptor dst{};
    std::memcpy(&dst, bytes, sizeof(BufferDescriptor));
    EXPECT_EQ(std::memcmp(&src, &dst, sizeof(BufferDescriptor)), 0);
    EXPECT_EQ(dst.magic, BUFFER_DESCRIPTOR_MAGIC);
    EXPECT_EQ(dst.owner_worker_path_id, 3u);
    EXPECT_EQ(dst.identity, src.identity);
    EXPECT_EQ(std::string(dst.body, dst.body_len), "psm_deadbeef");
}

// --- canonical identity: the import-registry key -----------------------------------------------

TEST(BufferAbi, IdentityDistinguishesGenerationAndIncarnation) {
    CanonicalIdentity a = make_identity();
    CanonicalIdentity b = make_identity();
    EXPECT_EQ(a, b);

    b.generation = a.generation + 1;  // buffer_id reuse across generations (ABA)
    EXPECT_NE(a, b);

    CanonicalIdentity c = make_identity();
    c.owner_instance_id[0] ^= 0xFF;  // different owner incarnation nonce
    EXPECT_NE(a, c);

    CanonicalIdentity d = make_identity();
    d.buffer_id ^= 0xFFULL;
    EXPECT_NE(a, d);
}

// Padding is wire-visible but semantically absent: two decodes of one backing must key identically,
// or the same buffer lands in two import-registry buckets and its dependencies split.
TEST(BufferAbi, IdentityPaddingIsExcludedFromKeyAndHash) {
    CanonicalIdentityHash h;
    CanonicalIdentity clean = make_identity();
    CanonicalIdentity dirty = make_identity();
    std::memset(dirty._pad, 0xA5, sizeof(dirty._pad));
    EXPECT_EQ(clean, dirty);
    EXPECT_EQ(h(clean), h(dirty));
}

TEST(BufferAbi, IdentityHashMatchesEquality) {
    CanonicalIdentityHash h;
    CanonicalIdentity a = make_identity();
    CanonicalIdentity b = make_identity();
    EXPECT_EQ(h(a), h(b));

    b.generation = a.generation + 1;
    EXPECT_NE(h(a), h(b));  // good hash separates the ABA case (not a strict requirement)
}

// --- validate_tensor: the shared receive-side gate ------------------------------------------

TEST(BufferAbi, ForkCowGrantsReadOnly) {
    Tensor r = make_tensor();
    r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::FORK_COW);
    r.buffer.access = static_cast<uint8_t>(AccessMode::READ);
    set_address_body(r, 0x7F0000000000ULL);  // a fork-inherited host VA, not a shm name
    EXPECT_NO_THROW(validate_tensor(r));

    // A write grant over copy-on-write would land in a private copy the owner never sees, so it is
    // refused rather than silently dropped.
    for (auto bad : {AccessMode::WRITE, AccessMode::READWRITE}) {
        r.buffer.access = static_cast<uint8_t>(bad);
        EXPECT_THROW(validate_tensor(r), std::invalid_argument);
    }

    // The same grants stay legal over MAP_SHARED, where a child's write does reach the owner.
    r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::FORK_SHM);
    r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    EXPECT_NO_THROW(validate_tensor(r));
}

// --- Malformed wire bytes: what only a decoder can produce ------------------------------------
//
// A receiver memcpy's an element out of shared memory a peer wrote and then validates it, so every
// field arrives as whatever bytes were there. These cases are unreachable through construction (the
// constructor validates first) and unreachable from Python (no binding turns bytes into a Tensor),
// which is why they live here: this is the only place the malformed state can be built at all.

// Overwrite `len` bytes at `offset` of a packed Tensor, then decode it the way a receiver does.
Tensor decode_with_corruption(size_t offset, const uint8_t *value, size_t len) {
    uint8_t bytes[sizeof(Tensor)];
    Tensor src = make_tensor();
    std::memcpy(bytes, &src, sizeof(Tensor));
    if (len > 0) std::memcpy(bytes + offset, value, len);
    Tensor out;
    std::memcpy(&out, bytes, sizeof(Tensor));
    return out;
}

TEST(BufferAbi, DecodeRejectsEveryCorruptedField) {
    // The untouched element is accepted, so each rejection below is attributable to the one field.
    EXPECT_NO_THROW(validate_tensor(decode_with_corruption(0, nullptr, 0)));  // no-op corruption

    struct Case {
        const char *what;
        size_t offset;
        std::vector<uint8_t> value;
    };
    // Offsets are the frozen ones the static_asserts above pin.
    const Case cases[] = {
        {"descriptor magic", offsetof(Tensor, buffer) + offsetof(BufferDescriptor, magic), {0x00, 0x00}},
        {"address_space out of range", offsetof(BufferDescriptor, address_space), {0x07}},
        {"access out of range", offsetof(BufferDescriptor, access), {0x09}},
        {"backend_kind out of range", offsetof(BufferDescriptor, backend_kind), {0x63}},
        {"body_len past the array", offsetof(BufferDescriptor, body_len), {0xFF, 0xFF}},
        {"generation 0 is reserved",
         offsetof(BufferDescriptor, identity) + offsetof(CanonicalIdentity, generation),
         {0, 0, 0, 0}},
        {"ndims out of range", offsetof(Tensor, ndims), {0x63, 0, 0, 0}},
        {"unknown dtype", offsetof(Tensor, dtype), {0x7F}},
        {"stride must be > 0", offsetof(Tensor, strides), {0, 0, 0, 0}},
        {"byte_offset is not a multiple", offsetof(Tensor, byte_offset), {0x03, 0, 0, 0, 0, 0, 0, 0}},
        {"reserved bytes past body_len", offsetof(BufferDescriptor, body) + DESC_MAX_BYTES - 1, {0x01}},
    };
    for (const auto &c : cases) {
        SCOPED_TRACE(c.what);
        EXPECT_THROW(
            validate_tensor(decode_with_corruption(c.offset, c.value.data(), c.value.size())), std::invalid_argument
        );
    }
}

// `backend_kind` says how the consumer reads `body`, so a body that does not fit that reading is not
// a smaller backing — it is a different value. The address case is the sharp one: a short body reads
// as a TRUNCATED pointer with nothing about it to distinguish it from a real one.
TEST(BufferAbi, DecodeRejectsABodyThatDoesNotMatchItsBackend) {
    Tensor dev = make_device_tensor();
    EXPECT_NO_THROW(validate_tensor(dev));

    for (uint16_t bad_len : {uint16_t{0}, uint16_t{3}, uint16_t{7}, uint16_t{9}, uint16_t{DESC_MAX_BYTES}}) {
        SCOPED_TRACE(bad_len);
        Tensor r = make_device_tensor();
        r.buffer.body_len = bad_len;
        EXPECT_THROW(validate_tensor(r), std::invalid_argument);
    }

    // Nothing is mapped or allocated at 0, so a zero base is an unfilled body rather than a location.
    Tensor null_base = make_device_tensor();
    set_address_body(null_base, 0);
    EXPECT_THROW(validate_tensor(null_base), std::invalid_argument);

    // POSIX_SHM carries a name the consumer opens, and the Python side decodes it as UTF-8 first.
    // Restricting it to printable ASCII makes that decode total: bytes that pass here always decode.
    Tensor no_name = make_tensor();
    no_name.buffer.body_len = 0;
    EXPECT_THROW(validate_tensor(no_name), std::invalid_argument);

    for (auto bad : {'\0', '\x01', ' ', '/', '\x7F', '\xFF'}) {
        SCOPED_TRACE(static_cast<int>(static_cast<unsigned char>(bad)));
        Tensor bad_name = make_tensor();
        bad_name.buffer.body[2] = bad;
        EXPECT_THROW(validate_tensor(bad_name), std::invalid_argument);
    }

    // A REMOTE_SIDECAR arg's authoritative descriptor rides in the per-task sidecar, so a body here
    // is a second, unread source of truth.
    Tensor sidecar = make_tensor();
    sidecar.buffer.backend_kind = static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR);
    EXPECT_THROW(validate_tensor(sidecar), std::invalid_argument);
    sidecar.buffer.body_len = 0;
    std::memset(sidecar.buffer.body, 0, DESC_MAX_BYTES);
    EXPECT_NO_THROW(validate_tensor(sidecar));
}

// The unused tail of `body` crosses a process boundary with the descriptor, so whatever the owner's
// memory held there crosses with it. Distinct from the struct's `_pad`, which equality and hashing
// ignore on purpose so two decodes of one backing key alike.
TEST(BufferAbi, DecodeRejectsNonZeroReservedTail) {
    Tensor r = make_tensor();
    EXPECT_NO_THROW(validate_tensor(r));

    for (uint32_t i : {r.buffer.body_len + 0u, DESC_MAX_BYTES / 2, DESC_MAX_BYTES - 1}) {
        SCOPED_TRACE(i);
        Tensor dirty = make_tensor();
        dirty.buffer.body[i] = static_cast<char>(0xA5);
        EXPECT_THROW(validate_tensor(dirty), std::invalid_argument);
    }
}

// A Tensor's shape/stride slots past `ndims` are the same case as the descriptor's body tail: the
// blob memcpy carries all 144 bytes whatever a producer chose to fill. Without this, two encodings
// of one view differ over the inactive slots while every active field agrees.
TEST(BufferAbi, DecodeRejectsNonZeroSlotsPastNdims) {
    Tensor r = make_tensor();  // ndims == 3, so slots 3 and 4 are inactive
    EXPECT_NO_THROW(validate_tensor(r));

    for (uint32_t i = r.ndims; i < static_cast<uint32_t>(MAX_TENSOR_DIMS); ++i) {
        SCOPED_TRACE(i);
        Tensor dirty_shape = make_tensor();
        dirty_shape.shapes[i] = 7;
        EXPECT_THROW(validate_tensor(dirty_shape), std::invalid_argument);

        Tensor dirty_stride = make_tensor();
        dirty_stride.strides[i] = 7;
        EXPECT_THROW(validate_tensor(dirty_stride), std::invalid_argument);
    }

    // The extent is unchanged by an inactive slot, so nothing else in the gate would have caught it.
    Tensor dirty = make_tensor();
    dirty.shapes[4] = 7;
    dirty.strides[4] = 7;
    EXPECT_EQ(tensor_extent_bytes(dirty), tensor_extent_bytes(r));
}

TEST(BufferAbi, DecodeRejectsAViewPastItsBacking) {
    Tensor r = make_tensor();
    r.buffer.nbytes = r.byte_offset + tensor_extent_bytes(r);
    EXPECT_NO_THROW(validate_tensor(r));  // exact fit

    r.buffer.nbytes -= 1;
    EXPECT_THROW(validate_tensor(r), std::invalid_argument);

    r = make_tensor();
    r.buffer.nbytes = r.byte_offset + tensor_extent_bytes(r);
    r.byte_offset += get_element_size(r.dtype);  // shifting the origin pushes the tail out
    EXPECT_THROW(validate_tensor(r), std::invalid_argument);
}

TEST(BufferAbi, DecodeRejectsAnExtentThatOverflows64Bits) {
    // shapes[] and strides[] are each u32, so one (shape-1)*stride product alone reaches ~2^64. An
    // extent summed without saturation wraps to a SMALL value, which then passes the bound check
    // below while the view really spans exabytes — accepted here, the consumer would address far
    // outside its backing. One dimension is enough to trigger it.
    Tensor r = make_tensor();
    r.dtype = DataType::FLOAT32;  // 4 B
    r.byte_offset = 0;
    r.ndims = 1;
    r.shapes[0] = 2147483649u;   // 2^31 + 1
    r.strides[0] = 2147483648u;  // 2^31   => (2^31)*(2^31)*4 == 2^64, wraps to 4 unsaturated
    r.buffer.nbytes = 4;
    EXPECT_EQ(tensor_extent_bytes(r), TENSOR_EXTENT_UNREPRESENTABLE);
    EXPECT_THROW(validate_tensor(r), std::invalid_argument);

    // Multi-dimensional variant: two terms that sum past 2^64. Rejected on the extent itself, so a
    // descriptor claiming an absurd nbytes cannot buy the view back.
    r = make_tensor();
    r.dtype = DataType::INT8;
    r.byte_offset = 0;
    r.ndims = 2;
    r.shapes[0] = 4294967295u;
    r.strides[0] = 4294967295u;
    r.shapes[1] = 7u;
    r.strides[1] = 2147483648u;
    r.buffer.nbytes = UINT64_MAX;
    EXPECT_EQ(tensor_extent_bytes(r), TENSOR_EXTENT_UNREPRESENTABLE);
    EXPECT_THROW(validate_tensor(r), std::invalid_argument);
}

// Two views of one backing that fully overlap must never be reported as disjoint: a saturating
// extent keeps the end offsets from wrapping a range into a false "no overlap", which at the
// dependency layer is a missed edge rather than a rejected argument.
TEST(BufferAbi, OverlapSurvivesASaturatedExtent) {
    Tensor a = make_tensor();
    Tensor b = a;
    EXPECT_TRUE(tensors_overlap(a, b));

    a.ndims = 1;
    a.shapes[0] = 4294967295u;
    a.strides[0] = 4294967295u;
    a.byte_offset = 1;  // nonzero, so the end offset is what overflows: 1 + saturated extent
    b = a;
    EXPECT_EQ(tensor_extent_bytes(a), TENSOR_EXTENT_UNREPRESENTABLE);
    EXPECT_TRUE(tensors_overlap(a, b));
}

TEST(BufferAbi, ValidateNeverWalksOffArbitraryBytes) {
    // A peer can write anything of the right length. None of it may be accepted without passing
    // every field check, and none of it may read past the struct — the fixed-length identity and
    // the bounded body_len/ndims are what make that structural rather than a matter of luck.
    std::mt19937 rng(0x5342);  // fixed seed: a failure here must reproduce
    for (int i = 0; i < 4096; ++i) {
        uint8_t bytes[sizeof(Tensor)];
        for (auto &b : bytes)
            b = static_cast<uint8_t>(rng());
        Tensor r;
        std::memcpy(&r, bytes, sizeof(Tensor));
        try {
            validate_tensor(r);
        } catch (const std::invalid_argument &) {
            continue;  // the expected outcome
        }
        // Surviving is legal only if every invariant genuinely holds.
        EXPECT_EQ(r.buffer.magic, BUFFER_DESCRIPTOR_MAGIC);
        EXPECT_NE(r.buffer.identity.generation, 0u);
        EXPECT_LE(r.buffer.body_len, DESC_MAX_BYTES);
        EXPECT_GT(r.ndims, 0u);
        EXPECT_LE(r.ndims, static_cast<uint32_t>(MAX_TENSOR_DIMS));
        // The footprint invariant belongs here too: without it, an extent that wrapped to a small
        // value survives as "in bounds" and the pass reports nothing.
        const uint64_t extent = tensor_extent_bytes(r);
        EXPECT_NE(extent, TENSOR_EXTENT_UNREPRESENTABLE);
        ASSERT_LE(r.byte_offset, r.buffer.nbytes);
        EXPECT_LE(extent, r.buffer.nbytes - r.byte_offset);
    }

    for (uint8_t fill : {uint8_t{0x00}, uint8_t{0xFF}}) {
        uint8_t bytes[sizeof(Tensor)];
        std::memset(bytes, fill, sizeof(bytes));
        Tensor r;
        std::memcpy(&r, bytes, sizeof(Tensor));
        EXPECT_THROW(validate_tensor(r), std::invalid_argument);
    }
}

TEST(BufferAbi, DecodedIdentityIgnoresWirePadding) {
    // Two decodes of one backing must key identically however the padding arrived, or an import
    // registry splits one buffer across two entries.
    uint8_t clean[sizeof(CanonicalIdentity)];
    CanonicalIdentity src = make_identity();
    std::memcpy(clean, &src, sizeof(CanonicalIdentity));

    uint8_t dirty[sizeof(CanonicalIdentity)];
    std::memcpy(dirty, clean, sizeof(CanonicalIdentity));
    std::memset(dirty + offsetof(CanonicalIdentity, _pad), 0xA5, sizeof(CanonicalIdentity::_pad));

    CanonicalIdentity a;
    CanonicalIdentity b;
    std::memcpy(&a, clean, sizeof(a));
    std::memcpy(&b, dirty, sizeof(b));
    EXPECT_TRUE(a == b);
    EXPECT_EQ(CanonicalIdentityHash{}(a), CanonicalIdentityHash{}(b));
}

}  // namespace
