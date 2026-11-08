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

#include "platform_comm/ccu/ccu_publication.h"

namespace {
using namespace simpler::ccu;

PublicationDomain domain() { return {9, 5, 2, 0, 2}; }

std::shared_ptr<PublicationImage> instance(int32_t callable_id = 7) {
    return PublicationImage::instance(
        domain(), callable_id, {{0x1000, 512, 0}, {0x2000, 512, 12}}, {{0x5555, 0x8000, 23}, {0x6666, 0x9000, 24}},
        1000000, std::make_shared<int>(1)
    );
}

template <typename T>
T read(const PublicationImage &image, size_t offset = 0) {
    T value{};
    std::memcpy(&value, static_cast<const unsigned char *>(image.data()) + offset, sizeof(value));
    return value;
}

TEST(CcuPublication, SequentialBindingsShareOneAlignedRequestAndKeepExplicitDoorbells) {
    const auto image = instance();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(image->data()) % 128, 0u);
    auto header = read<device::Workspace>(*image);
    EXPECT_EQ(static_cast<device::Admission>(header.admission), device::Admission::CLOSED);
    EXPECT_EQ(header.image_bytes, image->size());
    EXPECT_EQ(header.binding_count, 2u);
    auto first = read<device::ProgramBinding>(*image, header.bindings_offset);
    auto second = read<device::ProgramBinding>(*image, header.bindings_offset + sizeof(first));
    EXPECT_EQ(first.callable_id, 7);
    EXPECT_EQ(first.ccu_id, 0u);
    EXPECT_EQ(second.ccu_id, 1u);
    EXPECT_EQ(first.request_offset, second.request_offset);
    EXPECT_EQ(first.request_offset % 128, 0u);
    EXPECT_EQ(first.request_offset + sizeof(device::Request), image->size());
    EXPECT_EQ(first.platform_program, 0x5555u);
    EXPECT_EQ(first.doorbell_addr, 0x8000u);
    EXPECT_EQ(first.doorbell_value, 23u);
    EXPECT_EQ(second.doorbell_addr, 0x9000u);
    auto request = read<device::Request>(*image, first.request_offset);
    EXPECT_EQ(static_cast<device::Slot>(request.slot), device::Slot::FREE);
    EXPECT_EQ(request.sequence, 0u);
    EXPECT_EQ(request.completion.sequence, 0u);
    EXPECT_EQ(static_cast<device::Outcome>(request.completion.outcome), device::Outcome::PENDING);
    EXPECT_EQ(read<device::MemoryRange>(*image, header.ranges_offset).credential, 0u);
    image->open_for_publication();
    EXPECT_EQ(static_cast<device::Admission>(read<device::Workspace>(*image).admission), device::Admission::OPEN);
}

TEST(CcuPublication, ZeroAndMaximumDomainIdsPreserveIdentityAcrossPublication) {
    for (const uint64_t id : {uint64_t{0}, UINT64_MAX}) {
        auto selected = domain();
        selected.id = id;
        auto owner = std::make_shared<int>(1);
        auto image = PublicationImage::instance(selected, 0, {{0x1000, 64, 0}}, {{1, 0x8000, 1}}, 1000, owner);
        image->open_for_publication();
        const auto directory = PublicationImage::directory(selected, {{0, 0x10000, image, owner}});
        const auto workspace = read<device::Workspace>(*image);
        const auto header = read<device::DomainDirectory>(*directory);
        EXPECT_EQ(workspace.domain_id, id);
        EXPECT_EQ(header.domain_id, id);
        EXPECT_EQ(workspace.domain_owner, selected.owner);
        EXPECT_EQ(header.generation, selected.generation);
        auto different = selected;
        different.id = 1;
        EXPECT_THROW(PublicationImage::directory(different, {{0, 0x10000, image, owner}}), std::invalid_argument);
    }
}

TEST(CcuPublication, InstanceRejectsUnpreparedBindingsAndInvalidDomainMetadata) {
    auto make = [](PublicationDomain d, std::vector<PreparedProgramBinding> programs) {
        return PublicationImage::instance(d, 7, {{0x1000, 64, 0}}, programs, 1000, std::make_shared<int>(1));
    };
    for (auto bad : {PreparedProgramBinding{0, 0x8000, 1}, {1, 0, 1}, {1, 0x8001, 1}}) {
        EXPECT_THROW(make(domain(), {bad}), std::invalid_argument);
    }
    EXPECT_THROW(make(domain(), {}), std::invalid_argument);
    auto invalid = domain();
    invalid.rank = invalid.rank_count;
    EXPECT_THROW(make(invalid, {{1, 0x8000, 1}}), std::invalid_argument);
    invalid = domain();
    ++invalid.version;
    EXPECT_THROW(make(invalid, {{1, 0x8000, 1}}), std::invalid_argument);
    invalid = domain();
    invalid.generation = 0;
    EXPECT_THROW(make(invalid, {{1, 0x8000, 1}}), std::invalid_argument);
    invalid = domain();
    invalid.owner = 0;
    EXPECT_THROW(make(invalid, {{1, 0x8000, 1}}), std::invalid_argument);
    EXPECT_THROW(
        PublicationImage::instance(
            domain(), 7, {{UINT64_MAX - 7, 16, 0}}, {{1, 0x8000, 1}}, 1, std::make_shared<int>(1)
        ),
        std::invalid_argument
    );
}

TEST(CcuPublication, PlatformSessionContextIsPublishedWithoutOwningASecondRequestProtocol) {
    auto image = PublicationImage::instance(
        domain(), 7, {{0x1000, 64, 0}}, {{0x9000, 0x8000, 1}}, 1000, std::make_shared<int>(1), 0x9000, 4096
    );
    const auto header = read<device::Workspace>(*image);
    EXPECT_EQ(header.version, 2u);
    EXPECT_EQ(header.platform_context, 0x9000u);
    EXPECT_EQ(header.platform_context_bytes, 4096u);
}

TEST(CcuPublication, DirectoryMapsDistinctNativeCallableIdsWithoutRelocatingInstanceImages) {
    auto first = instance(7);
    auto second = instance(8);
    first->open_for_publication();
    second->open_for_publication();
    auto owner = std::make_shared<int>(1);
    const auto directory =
        PublicationImage::directory(domain(), {{7, 0x10000, first, owner}, {8, 0x20000, second, owner}});
    const auto header = read<device::DomainDirectory>(*directory);
    EXPECT_EQ(static_cast<device::Admission>(header.admission), device::Admission::CLOSED);
    EXPECT_EQ(header.entry_count, 2u);
    const auto one = read<device::InstanceEntry>(*directory, header.entries_offset);
    const auto two = read<device::InstanceEntry>(*directory, header.entries_offset + sizeof(one));
    EXPECT_EQ(one.callable_id, 7);
    EXPECT_EQ(two.callable_id, 8);
    EXPECT_EQ(one.image_addr, 0x10000u);
    EXPECT_EQ(two.image_addr, 0x20000u);
    EXPECT_EQ(one.image_bytes, first->size());
    directory->open_for_publication();
    EXPECT_EQ(
        static_cast<device::Admission>(read<device::DomainDirectory>(*directory).admission), device::Admission::OPEN
    );
}

TEST(CcuPublication, DirectoryRejectsClosedDuplicateMismatchedOrOverlappingEntries) {
    auto first = instance(7);
    auto owner = std::make_shared<int>(1);
    EXPECT_THROW(PublicationImage::directory(domain(), {{7, 0x10000, first, owner}}), std::invalid_argument);
    first->open_for_publication();
    auto second = instance(8);
    second->open_for_publication();
    EXPECT_THROW(PublicationImage::directory(domain(), {{7, 0, first, owner}}), std::invalid_argument);
    EXPECT_THROW(PublicationImage::directory(domain(), {{7, 0x10001, first, owner}}), std::invalid_argument);
    EXPECT_THROW(PublicationImage::directory(domain(), {{8, 0x10000, first, owner}}), std::invalid_argument);
    EXPECT_THROW(PublicationImage::directory(domain(), {{7, 0x10000, first, nullptr}}), std::invalid_argument);
    EXPECT_THROW(
        PublicationImage::directory(domain(), {{7, 0x10000, first, owner}, {7, 0x20000, first, owner}}),
        std::invalid_argument
    );
    EXPECT_THROW(
        PublicationImage::directory(domain(), {{7, 0x10000, first, owner}, {8, 0x10080, second, owner}}),
        std::invalid_argument
    );
    auto changed = domain();
    ++changed.generation;
    EXPECT_THROW(PublicationImage::directory(changed, {{7, 0x10000, first, owner}}), std::invalid_argument);
}

TEST(CcuPublication, DirectoryRetainsInstanceAndDeviceOwnersUntilItIsReleased) {
    auto image = instance();
    image->open_for_publication();
    auto device_owner = std::make_shared<int>(1);
    std::weak_ptr<int> owner_ref = device_owner;
    std::weak_ptr<PublicationImage> image_ref = image;
    auto directory = PublicationImage::directory(domain(), {{7, 0x10000, image, device_owner}});
    image.reset();
    device_owner.reset();
    EXPECT_FALSE(image_ref.expired());
    EXPECT_FALSE(owner_ref.expired());
    directory.reset();
    EXPECT_TRUE(image_ref.expired());
    EXPECT_TRUE(owner_ref.expired());
}
}  // namespace
