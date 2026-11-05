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

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "backend/sdma/sdma_metadata_handoff.h"

namespace {
constexpr uint32_t kLine = 64;
constexpr uint64_t kWorkspaceIndex = 5;
constexpr SdmaMetadataLayout kModelLayout{kLine, kLine, sizeof(uint32_t)};

struct Fixture {
    alignas(64) std::array<uint8_t, 49 * kLine> memory{};
    SchedulerSubmissionWorkspace workspace{};
    SchedulerSubmissionGrant grant{7, kWorkspaceIndex, 3, 0};
    SchedulerSubmissionExecutionIdentity identity{19, 3, 7, 11, 2, 1};
    SdmaMetadataLayout layout{kModelLayout};
    SdmaMetadataSession session{};
    struct alignas(128) Image {
        uint8_t prefix[128]{};
        uint8_t table[128]{};
        AsyncSubmissionContext context{};
    } image;
    DispatchPayload payload{};

    Fixture() {
        memory.fill(0x5a);
        workspace.address = reinterpret_cast<uint64_t>(memory.data());
        workspace.bytes = memory.size();
        workspace.qualification_epoch = identity.run_epoch;
        workspace.queue_num = 2;
        workspace.physical_channels = 48;
        workspace.qualification = SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN;
        session.valid_sdma = true;
        session.workspace_address = workspace.address;
        session.queue_num = workspace.queue_num;
        session.group_index = static_cast<uint32_t>(grant.group_index);
        auto &h = ctx().header;
        h.run_epoch = identity.run_epoch;
        h.region_address = reinterpret_cast<uint64_t>(&image);
        h.region_bytes = sizeof(image);
        h.submission_layout_offset = offsetof(Image, table);
        h.task_token = TaskId::to_uint64(TaskId::make_global(3));
        h.execution_index = identity.execution_index;
        h.dispatch_generation = identity.dispatch_generation;
        h.frame_address = reinterpret_cast<uint64_t>(&payload);
        h.block_idx = identity.block_idx;
        h.subtask_slot = identity.subtask_slot;
        h.completion_mode = CoreAsyncMode::DEFERRED;
        auto &e = ctx().evidence;
        e.run_epoch = identity.run_epoch;
        e.task_ref = identity.task_ref;
        e.execution_index = identity.execution_index;
        e.dispatch_generation = identity.dispatch_generation;
        e.block_idx = identity.block_idx;
        e.subtask_slot = identity.subtask_slot;
    }

    AsyncSubmissionContext &ctx() { return image.context; }
    bool preflight(SdmaMetadataHandoffRange *range) {
        return sdma_metadata_handoff_preflight(layout, kWorkspaceIndex, workspace, grant, identity, session, range);
    }
};

struct CacheEvent {
    uint64_t address;
    uint32_t bytes;
};

// This models the independently dirty scalar copy and bypass-written GM copy.
// It demonstrates the ordering counterexample, not silicon cache behaviour.
struct CacheModel {
    std::map<uint64_t, std::array<uint8_t, kLine>> dirty;
    std::vector<CacheEvent> events;

    void dirty_tail(uint64_t line, uint32_t tail) {
        auto &copy = dirty[line];
        std::memcpy(copy.data(), reinterpret_cast<const void *>(line), kLine);
        std::memcpy(copy.data() + kModelLayout.tail_offset, &tail, sizeof(tail));
    }
    void flush_range(volatile void *address, uint32_t bytes) {
        const auto begin = reinterpret_cast<uint64_t>(address);
        events.push_back({begin, bytes});
        for (uint32_t offset = 0; offset < bytes; offset += kLine) {
            const auto found = dirty.find(begin + offset);
            if (found == dirty.end()) continue;
            std::memcpy(reinterpret_cast<void *>(found->first), found->second.data(), kLine);
            dirty.erase(found);
        }
    }
    void finish() { events.push_back({0, 0}); }
    void evict_old_core() {
        for (const auto &[address, copy] : dirty)
            std::memcpy(reinterpret_cast<void *>(address), copy.data(), kLine);
        dirty.clear();
    }
};

void store_tail(uint64_t line, uint32_t value) {
    std::memcpy(reinterpret_cast<uint8_t *>(line) + kModelLayout.tail_offset, &value, sizeof(value));
}

uint32_t load_tail(uint64_t line) {
    uint32_t result = 0;
    std::memcpy(&result, reinterpret_cast<const uint8_t *>(line) + kModelLayout.tail_offset, sizeof(result));
    return result;
}

TEST(HbgSdmaMetadataHandoff, MetadataAdapterRequiresItsAuditedPtoPin) {
    auto directory = std::filesystem::path(__FILE__).parent_path();
    while (directory != directory.parent_path() && !std::filesystem::exists(directory / "pto_isa.pin"))
        directory = directory.parent_path();
    std::ifstream file(directory / "pto_isa.pin");
    ASSERT_TRUE(file.is_open());
    std::string pin;
    file >> pin;
    EXPECT_EQ(pin, SDMA_METADATA_PTO_PIN);
}

TEST(HbgSdmaMetadataHandoff, FirstAndLastGroupsCoverEveryConfiguredQueueIncludingPartialWidths) {
    for (uint32_t queues : {1u, 2u, 3u, 7u, 16u, 48u}) {
        for (uint32_t group : {0u, 48u / queues - 1}) {
            Fixture f;
            f.workspace.queue_num = f.session.queue_num = queues;
            f.grant.group_index = f.session.group_index = group;
            SdmaMetadataHandoffRange range{};
            ASSERT_TRUE(f.preflight(&range));
            EXPECT_EQ(
                range.address, f.workspace.address + f.layout.header_bytes + group * queues * f.layout.channel_bytes
            );
            EXPECT_EQ(range.bytes, queues * kLine);
            EXPECT_EQ(range.lines, queues);
            EXPECT_EQ(range.address % kLine, 0u);
            EXPECT_LE(range.address + range.bytes, f.workspace.address + f.workspace.bytes);
        }
    }
}

TEST(HbgSdmaMetadataHandoff, InvalidOwnershipSessionAndExtentRejectWithoutCacheAccess) {
    for (uint32_t invalid = 0; invalid < 25; ++invalid) {
        SCOPED_TRACE(invalid);
        Fixture f;
        switch (invalid) {
        case 0:
            f.workspace.address = 0;
            break;
        case 1:
            ++f.workspace.address;
            break;
        case 2:
            f.workspace.bytes = 0;
            break;
        case 3:
            --f.workspace.bytes;
            break;
        case 4:
            f.workspace.address = UINT64_MAX - 63;
            f.session.workspace_address = f.workspace.address;
            break;
        case 5:
            f.workspace.queue_num = 0;
            break;
        case 6:
            f.workspace.queue_num = 49;
            break;
        case 7:
            f.workspace.physical_channels = 0;
            break;
        case 8:
            f.workspace.physical_channels = 49;
            break;
        case 9:
            f.workspace.qualification = SchedulerSubmissionPoolQualification::UNKNOWN;
            break;
        case 10:
            ++f.workspace.qualification_epoch;
            break;
        case 11:
            f.workspace.reserved[2] = 1;
            break;
        case 12:
            f.grant.group_index = UINT64_MAX;
            break;
        case 13:
            f.grant.group_index = f.session.group_index = 24;
            break;
        case 14:
            ++f.grant.workspace_index;
            break;
        case 15:
            ++f.grant.execution_index;
            break;
        case 16:
            f.identity.dispatch_generation = 0;
            break;
        case 17:
            f.identity.subtask_slot = 0;
            break;
        case 18:
            f.session.valid_sdma = false;
            break;
        case 19:
            f.session.workspace_address += kLine;
            break;
        case 20:
            ++f.session.queue_num;
            break;
        case 21:
            ++f.session.group_index;
            break;
        case 22:
            f.layout.header_bytes = 32;
            break;
        case 23:
            f.layout.channel_bytes = 128;
            break;
        case 24:
            f.layout.tail_offset = 64;
            break;
        }
        SdmaMetadataHandoffRange range{1, 2, 3};
        EXPECT_FALSE(f.preflight(&range));
        EXPECT_EQ(range.address, 0u);
        EXPECT_EQ(range.bytes, 0u);
        EXPECT_EQ(range.lines, 0u);
        CacheModel cache;
        SdmaMetadataHandoffPublisher publisher(
            f.layout, kWorkspaceIndex, f.workspace, f.grant, f.identity, f.session, cache
        );
        EXPECT_FALSE(publisher.publish_backend_metadata());
        EXPECT_TRUE(cache.events.empty());
    }
    Fixture f;
    EXPECT_FALSE(f.preflight(nullptr));
}

TEST(HbgSdmaMetadataHandoff, OnlyOwnedChannelLinesAreFlushedAndThenOrdered) {
    Fixture f;
    SdmaMetadataHandoffRange range{};
    ASSERT_TRUE(f.preflight(&range));
    const auto untouched = f.memory;
    CacheModel cache;
    cache.dirty_tail(f.workspace.address + kLine, 97);
    SdmaMetadataHandoffPublisher publisher(
        f.layout, kWorkspaceIndex, f.workspace, f.grant, f.identity, f.session, cache
    );
    ASSERT_TRUE(publisher.preflight());
    ASSERT_TRUE(publisher.publish_backend_metadata());
    ASSERT_EQ(cache.events.size(), 2u);
    EXPECT_EQ(cache.events[0].address, range.address);
    EXPECT_EQ(cache.events[0].bytes, f.workspace.queue_num * kLine);
    EXPECT_EQ(cache.events[1].bytes, 0u);
    EXPECT_EQ(cache.dirty.count(f.workspace.address + kLine), 1u);
    EXPECT_EQ(f.memory, untouched);
}

TEST(HbgSdmaMetadataHandoff, OldCoreEvictionCannotRestoreItsTailAfterOrderedHandoff) {
    for (bool clean_before_handoff : {false, true}) {
        Fixture f;
        SdmaMetadataHandoffRange range{};
        ASSERT_TRUE(f.preflight(&range));
        CacheModel old_core;
        for (uint32_t q = 0; q < range.lines; ++q) {
            const uint64_t line = range.address + q * kLine;
            store_tail(line, 11);
            old_core.dirty_tail(line, 11);
        }
        if (clean_before_handoff) {
            SdmaMetadataHandoffPublisher publisher(
                f.layout, kWorkspaceIndex, f.workspace, f.grant, f.identity, f.session, old_core
            );
            ASSERT_TRUE(publisher.publish_backend_metadata());
        }
        for (uint32_t q = 0; q < range.lines; ++q)
            store_tail(range.address + q * kLine, 29);
        old_core.evict_old_core();
        for (uint32_t q = 0; q < range.lines; ++q)
            EXPECT_EQ(load_tail(range.address + q * kLine), clean_before_handoff ? 29u : 11u);
        EXPECT_EQ(f.memory.front(), 0x5a);
        EXPECT_EQ(f.memory.back(), 0x5a);
    }
}

TEST(HbgSdmaMetadataHandoff, CommitPublishesEvidenceOnlyAfterTheMetadataBarrier) {
    Fixture f;
    CacheModel cache;
    SdmaMetadataHandoffPublisher publisher(
        f.layout, kWorkspaceIndex, f.workspace, f.grant, f.identity, f.session, cache
    );
    ASSERT_TRUE(publisher.preflight());
    ASSERT_TRUE(async_submission_begin(f.ctx()));
    ASSERT_TRUE(async_submission_commit(f.ctx(), publisher));
    ASSERT_EQ(cache.events.size(), 4u);
    EXPECT_NE(cache.events[0].address, 0u);
    EXPECT_EQ(cache.events[0].bytes, 2 * kLine);
    EXPECT_EQ(cache.events[1].address, 0u);
    EXPECT_EQ(cache.events[2].address, reinterpret_cast<uint64_t>(&f.ctx().evidence));
    EXPECT_EQ(cache.events[2].bytes, sizeof(AsyncSubmissionEvidence));
    EXPECT_EQ(cache.events[3].address, 0u);
    EXPECT_EQ(f.ctx().evidence.attempted, 1u);
    EXPECT_EQ(f.ctx().evidence.committed, 1u);
    EXPECT_EQ(f.ctx().evidence.flags, ASYNC_SUBMISSION_METADATA_PUBLISHED);
}

TEST(HbgSdmaMetadataHandoff, ChangedGrantAfterBeginCannotPublishCommittedEvidence) {
    Fixture f;
    CacheModel cache;
    SdmaMetadataHandoffPublisher publisher(
        f.layout, kWorkspaceIndex, f.workspace, f.grant, f.identity, f.session, cache
    );
    ASSERT_TRUE(publisher.preflight());
    ASSERT_TRUE(async_submission_begin(f.ctx()));
    ++f.grant.group_index;
    EXPECT_FALSE(async_submission_commit(f.ctx(), publisher));
    EXPECT_TRUE(cache.events.empty());
    EXPECT_EQ(f.ctx().evidence.attempted, 1u);
    EXPECT_EQ(f.ctx().evidence.committed, 0u);
    EXPECT_EQ(f.ctx().evidence.flags, 0u);
    EXPECT_EQ(f.ctx().evidence.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}
}  // namespace
