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
// Wire-ABI tests for BufferHandle / BufferRef (buffer_handle.h). The byte layout is pinned by
// static_assert in the header; these tests pin the values that outlive a compile (sizes, enum
// values, version) and exercise the trivially-copyable memcpy round trip and the canonical
// identity key used by the import registry.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "buffer_handle.h"

namespace {

BufferRef make_ref() {
    BufferRef r{};
    r.identity.owner_instance_id = 0xA1A2A3A4A5A6A7A8ULL;
    r.identity.buffer_id = 0x0102030405060708ULL;
    r.identity.owner_worker_path.depth = 2;
    r.identity.owner_worker_path.hop[0] = 3;
    r.identity.owner_worker_path.hop[1] = 5;
    r.identity.generation = 7;
    r.byte_offset = 4096;
    r.ndims = 3;
    r.shapes[0] = 2;
    r.shapes[1] = 4;
    r.shapes[2] = 8;
    r.strides[0] = 32;
    r.strides[1] = 8;
    r.strides[2] = 1;
    r.dtype = DataType::FLOAT16;
    r.flags = BUFFER_REF_MANUAL_DEP | BUFFER_REF_CONTIGUOUS;
    return r;
}

// --- Layout / value contracts (survive a compile; document the frozen ABI) -------------------

TEST(BufferHandleAbi, StructSizesAreFrozen) {
    EXPECT_EQ(sizeof(OwnerWorkerPath), 20u);
    EXPECT_EQ(sizeof(CanonicalIdentity), 40u);
    EXPECT_EQ(sizeof(BufferRef), 96u);
    EXPECT_EQ(sizeof(BufferHandleDescriptor), 128u);
}

TEST(BufferHandleAbi, EnvelopeConstantsAreFrozen) {
    EXPECT_EQ(BUFFER_ABI_VERSION, 1u);
    EXPECT_EQ(MAX_WORKER_PATH_DEPTH, 4u);
    EXPECT_EQ(BACKEND_TOKEN_BYTES, 64u);
}

TEST(BufferHandleAbi, EnumValuesAreFrozen) {
    EXPECT_EQ(static_cast<uint8_t>(AddressSpace::HOST), 0);
    EXPECT_EQ(static_cast<uint8_t>(AddressSpace::DEVICE), 1);
    EXPECT_EQ(static_cast<uint8_t>(Visibility::PRIVATE), 0);
    EXPECT_EQ(static_cast<uint8_t>(Visibility::SHARED), 1);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::READ), 0);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::WRITE), 1);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::READWRITE), 2);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::INVALID), 0);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::FORK_SHM), 1);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::POSIX_SHM), 2);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::VMM_WINDOW), 3);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::DEVICE_MALLOC), 4);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR), 5);
}

// --- memcpy round trip (blob transport is memcpy of trivially-copyable structs) --------------

TEST(BufferHandleAbi, BufferRefSurvivesByteRoundTrip) {
    BufferRef src = make_ref();
    uint8_t bytes[sizeof(BufferRef)];
    std::memcpy(bytes, &src, sizeof(BufferRef));
    BufferRef dst{};
    std::memcpy(&dst, bytes, sizeof(BufferRef));
    EXPECT_EQ(std::memcmp(&src, &dst, sizeof(BufferRef)), 0);
    EXPECT_EQ(dst.byte_offset, 4096u);
    EXPECT_EQ(dst.dtype, DataType::FLOAT16);
    EXPECT_EQ(dst.strides[0], 32u);
    EXPECT_EQ(dst.flags & BUFFER_REF_MANUAL_DEP, BUFFER_REF_MANUAL_DEP);
}

TEST(BufferHandleAbi, HandleDescriptorSurvivesByteRoundTrip) {
    BufferHandleDescriptor src{};
    src.abi_version = BUFFER_ABI_VERSION;
    src.identity.owner_instance_id = 0xDEADBEEFULL;
    src.identity.buffer_id = 42;
    src.identity.generation = 1;
    src.nbytes = 1 << 20;
    src.address_space = AddressSpace::DEVICE;
    src.visibility = Visibility::SHARED;
    src.access = AccessMode::READWRITE;
    src.backend_kind = BackendKind::VMM_WINDOW;
    src.backend_handle = 0x7f00abcdULL;
    std::strncpy(src.token, "psm_deadbeef", BACKEND_TOKEN_BYTES - 1);

    uint8_t bytes[sizeof(BufferHandleDescriptor)];
    std::memcpy(bytes, &src, sizeof(BufferHandleDescriptor));
    BufferHandleDescriptor dst{};
    std::memcpy(&dst, bytes, sizeof(BufferHandleDescriptor));
    EXPECT_EQ(std::memcmp(&src, &dst, sizeof(BufferHandleDescriptor)), 0);
    EXPECT_EQ(dst.abi_version, BUFFER_ABI_VERSION);
    EXPECT_EQ(dst.address_space, AddressSpace::DEVICE);
    EXPECT_STREQ(dst.token, "psm_deadbeef");
}

// --- canonical identity: the import-registry key -----------------------------------------------

TEST(BufferHandleAbi, RefAndDescriptorAgreeOnIdentity) {
    BufferRef r = make_ref();
    BufferHandleDescriptor d{};
    d.identity = r.identity;
    EXPECT_EQ(r.identity, d.identity);
}

TEST(BufferHandleAbi, IdentityDistinguishesGenerationAndOwner) {
    BufferRef a = make_ref();
    BufferRef b = make_ref();
    EXPECT_EQ(a.identity, b.identity);

    b.identity.generation = a.identity.generation + 1;  // buffer_id reuse across generations (ABA)
    EXPECT_NE(a.identity, b.identity);

    BufferRef c = make_ref();
    c.identity.owner_instance_id = a.identity.owner_instance_id + 1;  // same buffer_id, different owner incarnation
    EXPECT_NE(a.identity, c.identity);

    BufferRef e = make_ref();
    e.identity.owner_worker_path.hop[0] = a.identity.owner_worker_path.hop[0] + 1;  // different owner path
    EXPECT_NE(a.identity, e.identity);
}

TEST(BufferHandleAbi, IdentityHashMatchesEquality) {
    CanonicalIdentityHash h;
    BufferRef a = make_ref();
    BufferRef b = make_ref();
    EXPECT_EQ(h(a.identity), h(b.identity));

    b.identity.generation = a.identity.generation + 1;
    // Not a strict requirement, but a good hash separates the ABA case.
    EXPECT_NE(h(a.identity), h(b.identity));
}

}  // namespace
