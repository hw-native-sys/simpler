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

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "worker_manager.h"

namespace {

// A POSIX_SHM descriptor of `nbytes`; the backend body is a name, so nothing here depends on a
// mapping existing. Only the length field participates in the span bound under test.
BufferDescriptor host_descriptor(uint64_t nbytes, uint64_t buffer_id) {
    BufferDescriptor desc{};
    desc.magic = BUFFER_DESCRIPTOR_MAGIC;
    desc.identity.buffer_id = buffer_id;
    desc.identity.generation = 1;
    desc.address_space = static_cast<uint8_t>(AddressSpace::HOST);
    desc.access = static_cast<uint8_t>(AccessMode::READWRITE);
    desc.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    desc.nbytes = nbytes;
    const char name[] = "simpler-test-backing";
    desc.body_len = sizeof(name) - 1;
    std::memcpy(desc.body, name, desc.body_len);
    return desc;
}

ControlCopyRequest request_with(const CopySpan &span) {
    return ControlCopyRequest{host_descriptor(64, 1), host_descriptor(64, 2), span};
}

}  // namespace

TEST(ControlCopyRequest, WholeBackingAtZeroOffsetIsAccepted) {
    EXPECT_NO_THROW(validate_control_copy_request(request_with(CopySpan{64, 0, 0})));
}

TEST(ControlCopyRequest, SpanEndingExactlyAtTheBackingEndIsAccepted) {
    EXPECT_NO_THROW(validate_control_copy_request(request_with(CopySpan{16, 48, 48})));
}

TEST(ControlCopyRequest, EmptySpanAtTheBackingEndIsAccepted) {
    // offset == nbytes is one past the last byte but names no byte, so it is in range.
    EXPECT_NO_THROW(validate_control_copy_request(request_with(CopySpan{0, 64, 64})));
}

TEST(ControlCopyRequest, LengthPastTheDestinationIsRejected) {
    EXPECT_THROW(validate_control_copy_request(request_with(CopySpan{65, 0, 0})), std::invalid_argument);
}

TEST(ControlCopyRequest, OffsetPlusLengthPastTheDestinationIsRejected) {
    // Each half fits on its own; only the sum overruns. This is what a bound on the length alone
    // would let through.
    EXPECT_THROW(validate_control_copy_request(request_with(CopySpan{17, 48, 0})), std::invalid_argument);
}

TEST(ControlCopyRequest, OffsetPastTheBackingIsRejected) {
    EXPECT_THROW(validate_control_copy_request(request_with(CopySpan{0, 65, 0})), std::invalid_argument);
}

TEST(ControlCopyRequest, WrappingOffsetPlusLengthIsRejected) {
    // `offset + nbytes` overflows back to 0 and would compare as in-range; the gate subtracts
    // instead, so both halves are caught on their own.
    constexpr uint64_t kMax = ~uint64_t{0};
    EXPECT_THROW(validate_control_copy_request(request_with(CopySpan{kMax, 1, 0})), std::invalid_argument);
    EXPECT_THROW(validate_control_copy_request(request_with(CopySpan{1, kMax, 0})), std::invalid_argument);
}

TEST(ControlCopyRequest, SourceEndIsBoundedIndependentlyOfTheDestination) {
    ControlCopyRequest request{host_descriptor(64, 1), host_descriptor(8, 2), CopySpan{16, 0, 0}};
    EXPECT_THROW(validate_control_copy_request(request), std::invalid_argument);
}

TEST(ControlCopyRequest, RoundTripsThroughTheControlFrame) {
    std::vector<char> frame(MAILBOX_FRAME_SIZE, 0);
    const BufferDescriptor dst = host_descriptor(64, 1);
    const BufferDescriptor src = host_descriptor(64, 2);
    write_control_copy_request(frame.data(), CTRL_COPY_TO, dst, src, CopySpan{16, 32, 8});

    uint64_t sub_cmd = 0;
    std::memcpy(&sub_cmd, frame.data() + MAILBOX_OFF_CALLABLE, sizeof(sub_cmd));
    EXPECT_EQ(sub_cmd, CTRL_COPY_TO);

    const ControlCopyRequest decoded = read_control_copy_request(frame.data());
    EXPECT_EQ(decoded.span.nbytes, 16u);
    EXPECT_EQ(decoded.span.dst_offset, 32u);
    EXPECT_EQ(decoded.span.src_offset, 8u);
    EXPECT_EQ(decoded.dst.identity.buffer_id, dst.identity.buffer_id);
    EXPECT_EQ(decoded.src.identity.buffer_id, src.identity.buffer_id);
}

TEST(ControlCopyRequest, StillFitsTheControlFrameArgsRegion) {
    // The static_assert in the header is the real gate; this states the headroom a reader would
    // otherwise have to compute to know how much room a further field has.
    EXPECT_LE(MAILBOX_OFF_ARGS + static_cast<ptrdiff_t>(sizeof(ControlCopyRequest)), MAILBOX_OFF_SHUTDOWN);
}
