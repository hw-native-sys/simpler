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
// Wire-ABI tests for BufferHandle / BufferRef (buffer_handle.h), implementing the frozen logical
// schema in .docs/L3-new/worker-memory-model/bufferhandle-abi.md. Byte layout is pinned by
// static_assert in the header; these tests pin the sizes, enum values, and the blob codec.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "buffer_handle.h"
#include "task_args.h"

namespace {

CanonicalIdentity make_identity() {
    CanonicalIdentity id{};
    for (uint32_t i = 0; i < OWNER_INSTANCE_ID_BYTES; ++i)
        id.owner_instance_id[i] = static_cast<uint8_t>(0xA0 + i);
    id.buffer_id = 0x0102030405060708ULL;
    id.generation = 7;
    const char *path = "L4/L3[2]";
    id.path_len = static_cast<uint16_t>(std::strlen(path));
    std::memcpy(id.owner_worker_path, path, id.path_len);
    return id;
}

BufferRef make_ref() {
    BufferRef r{};
    r.handle.abi_version = BUFFER_ABI_VERSION;
    r.handle.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    r.handle.descriptor_version = BUFFER_DESCRIPTOR_VERSION;
    r.handle.identity = make_identity();
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

// --- Layout / value contracts (frozen ABI) -----------------------------------------------------

TEST(BufferHandleAbi, StructSizesAreFrozen) {
    EXPECT_EQ(sizeof(CanonicalIdentity), 96u);
    EXPECT_EQ(sizeof(BufferRef), 272u);
    EXPECT_EQ(sizeof(BufferHandleDescriptor), 216u);
}

TEST(BufferHandleAbi, ConstantsAreFrozen) {
    EXPECT_EQ(BUFFER_ABI_VERSION, 1);
    EXPECT_EQ(BUFFER_DESCRIPTOR_VERSION, 1);
    EXPECT_EQ(OWNER_INSTANCE_ID_BYTES, 16u);
    EXPECT_EQ(PATH_MAX_BYTES, 64u);
    EXPECT_EQ(DESC_MAX_BYTES, 96u);
}

TEST(BufferHandleAbi, EnumValuesAreFrozen) {
    EXPECT_EQ(static_cast<uint8_t>(AddressSpace::HOST), 0);
    EXPECT_EQ(static_cast<uint8_t>(AddressSpace::DEVICE), 1);
    EXPECT_EQ(static_cast<uint8_t>(Visibility::PRIVATE), 0);
    EXPECT_EQ(static_cast<uint8_t>(Visibility::SHARED), 1);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::READ), 0);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::WRITE), 1);
    EXPECT_EQ(static_cast<uint8_t>(AccessMode::READWRITE), 2);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::FORK_SHM), 0);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::POSIX_SHM), 1);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::VMM_WINDOW), 2);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR), 3);
    EXPECT_EQ(static_cast<uint8_t>(BackendKind::DEVICE_MALLOC), 4);
}

// --- memcpy round trip -------------------------------------------------------------------------

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
    EXPECT_EQ(dst.handle.identity, src.handle.identity);
}

TEST(BufferHandleAbi, HandleDescriptorSurvivesByteRoundTrip) {
    BufferHandleDescriptor src{};
    src.abi_version = BUFFER_ABI_VERSION;
    src.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    src.visibility = static_cast<uint8_t>(Visibility::SHARED);
    src.access = static_cast<uint8_t>(AccessMode::READWRITE);
    src.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    src.descriptor_version = BUFFER_DESCRIPTOR_VERSION;
    src.identity = make_identity();
    src.nbytes = 1 << 20;
    const char *body = "psm_deadbeef";
    src.body_len = static_cast<uint16_t>(std::strlen(body));
    std::memcpy(src.body, body, src.body_len);

    uint8_t bytes[sizeof(BufferHandleDescriptor)];
    std::memcpy(bytes, &src, sizeof(BufferHandleDescriptor));
    BufferHandleDescriptor dst{};
    std::memcpy(&dst, bytes, sizeof(BufferHandleDescriptor));
    EXPECT_EQ(std::memcmp(&src, &dst, sizeof(BufferHandleDescriptor)), 0);
    EXPECT_EQ(dst.abi_version, BUFFER_ABI_VERSION);
    EXPECT_EQ(dst.identity, src.identity);
    EXPECT_EQ(std::string(dst.body, dst.body_len), "psm_deadbeef");
}

// --- canonical identity: the import-registry key -----------------------------------------------

TEST(BufferHandleAbi, IdentityDistinguishesGenerationOwnerPath) {
    CanonicalIdentity a = make_identity();
    CanonicalIdentity b = make_identity();
    EXPECT_EQ(a, b);

    b.generation = a.generation + 1;  // buffer_id reuse across generations (ABA)
    EXPECT_NE(a, b);

    CanonicalIdentity c = make_identity();
    c.owner_instance_id[0] ^= 0xFF;  // different owner incarnation nonce
    EXPECT_NE(a, c);

    CanonicalIdentity e = make_identity();
    const char *p2 = "L4/L3[3]";  // different owner path (same length)
    std::memcpy(e.owner_worker_path, p2, e.path_len);
    EXPECT_NE(a, e);
}

TEST(BufferHandleAbi, IdentityHashMatchesEquality) {
    CanonicalIdentityHash h;
    CanonicalIdentity a = make_identity();
    CanonicalIdentity b = make_identity();
    EXPECT_EQ(h(a), h(b));

    b.generation = a.generation + 1;
    EXPECT_NE(h(a), h(b));  // good hash separates the ABA case (not a strict requirement)
}

// --- BufferRef wire blob: versioned length-prefixed round trip + rejection ----------------------

BufferRef make_ref_b() {
    BufferRef r = make_ref();
    r.handle.identity.buffer_id = 99;
    r.byte_offset = 0;
    r.ndims = 1;
    r.shapes[0] = 5;
    r.strides[0] = 1;
    r.dtype = DataType::INT32;
    return r;
}

TEST(BufferRefBlob, RoundTrip) {
    BufferRef refs[2] = {make_ref(), make_ref_b()};
    uint64_t scalars[2] = {42, 0xC0FFEE};
    size_t sz = bufferref_blob_size(2, 2);
    EXPECT_EQ(sz, BUFFERREF_BLOB_HEADER_SIZE + 2 * sizeof(BufferRef) + 2 * sizeof(uint64_t));

    std::vector<uint8_t> buf(sz);
    write_bufferref_blob(buf.data(), refs, 2, scalars, 2);

    BufferRefBlobView v = read_bufferref_blob(buf.data(), sz);
    ASSERT_EQ(v.ref_count, 2);
    ASSERT_EQ(v.scalar_count, 2);
    BufferRef r0 = v.ref(0);
    BufferRef r1 = v.ref(1);
    EXPECT_EQ(std::memcmp(&r0, &refs[0], sizeof(BufferRef)), 0);
    EXPECT_EQ(std::memcmp(&r1, &refs[1], sizeof(BufferRef)), 0);
    EXPECT_EQ(v.scalars[0], 42u);
    EXPECT_EQ(v.scalars[1], 0xC0FFEEu);
}

TEST(BufferRefBlob, EmptyBlob) {
    size_t sz = bufferref_blob_size(0, 0);
    EXPECT_EQ(sz, BUFFERREF_BLOB_HEADER_SIZE);
    std::vector<uint8_t> buf(sz);
    write_bufferref_blob(buf.data(), nullptr, 0, nullptr, 0);
    BufferRefBlobView v = read_bufferref_blob(buf.data(), sz);
    EXPECT_EQ(v.ref_count, 0);
    EXPECT_EQ(v.scalar_count, 0);
}

TEST(BufferRefBlob, RejectsUnknownVersion) {
    std::vector<uint8_t> buf(bufferref_blob_size(0, 0));
    write_bufferref_blob(buf.data(), nullptr, 0, nullptr, 0);
    uint32_t bad = BUFFER_ABI_VERSION + 1;
    std::memcpy(buf.data(), &bad, sizeof(bad));
    EXPECT_THROW(read_bufferref_blob(buf.data(), buf.size()), std::runtime_error);
}

TEST(BufferRefBlob, RejectsTruncatedCapacity) {
    BufferRef refs[1] = {make_ref()};
    std::vector<uint8_t> buf(bufferref_blob_size(1, 0));
    write_bufferref_blob(buf.data(), refs, 1, nullptr, 0);
    EXPECT_THROW(read_bufferref_blob(buf.data(), BUFFERREF_BLOB_HEADER_SIZE), std::runtime_error);
    EXPECT_THROW(read_bufferref_blob(buf.data(), 4), std::runtime_error);
}

TEST(BufferRefBlob, RejectsNegativeCount) {
    std::vector<uint8_t> buf(64, 0);
    uint32_t ver = BUFFER_ABI_VERSION;
    std::memcpy(buf.data() + 0, &ver, sizeof(ver));
    int32_t neg = -1;
    std::memcpy(buf.data() + 4, &neg, sizeof(neg));  // ref_count = -1
    EXPECT_THROW(read_bufferref_blob(buf.data(), buf.size()), std::runtime_error);
}

TEST(BufferRefBlob, RejectsNonZeroReserved) {
    std::vector<uint8_t> buf(bufferref_blob_size(0, 0));
    write_bufferref_blob(buf.data(), nullptr, 0, nullptr, 0);
    uint32_t dirty = 1;
    std::memcpy(buf.data() + 12, &dirty, sizeof(dirty));  // reserved header word
    EXPECT_THROW(read_bufferref_blob(buf.data(), buf.size()), std::runtime_error);
}

}  // namespace
