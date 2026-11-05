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

#include <cerrno>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {
int unlink_calls = 0;
int unlink_failures = 0;
int unmap_calls = 0;
int unmap_failures = 0;

int fault_shm_unlink(const char *name) {
    ++unlink_calls;
    if (unlink_failures > 0) {
        --unlink_failures;
        errno = EACCES;
        return -1;
    }
    return shm_unlink(name);
}

int fault_munmap(void *address, size_t size) {
    ++unmap_calls;
    if (unmap_failures > 0) {
        --unmap_failures;
        errno = EINVAL;
        return -1;
    }
    return munmap(address, size);
}
}  // namespace

// Only cleanup syscalls are intercepted; allocation and backing are real POSIX shm.
#define shm_unlink fault_shm_unlink
#define munmap fault_munmap
#include "platform_comm/comm_sim.cpp"
#undef munmap
#undef shm_unlink

namespace {
class CommSimDomainCleanupTest : public ::testing::Test {
protected:
    void SetUp() override {
        unlink_calls = unlink_failures = unmap_calls = unmap_failures = 0;
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string identity = std::to_string(getpid()) + ":" + info->name();
        handle_ = comm_init(0, 1, nullptr, identity.c_str());
        ASSERT_NE(handle_, nullptr);
        const uint32_t ranks[] = {0};
        uint64_t ctx = 0;
        uint64_t base = 0;
        size_t size = 0;
        ASSERT_EQ(comm_alloc_domain_windows(handle_, 1, ranks, 1, 0, 4096, 0, &ctx, &base, &size), 0);
        name_ = handle_->domain_allocations.at(1)->shm_name;
        ASSERT_TRUE(name_present());
    }

    void TearDown() override {
        // The harness reclaims retained test resources outside the failing production path.
        if (handle_ != nullptr) {
            for (const auto &entry : handle_->domain_allocations) {
                const auto &alloc = *entry.second;
                if (alloc.mmap_base != nullptr) munmap(alloc.mmap_base, alloc.mmap_size);
                shm_unlink(alloc.shm_name.c_str());
            }
            delete handle_;
        }
        if (!name_.empty()) shm_unlink(name_.c_str());
    }

    bool name_present() const {
        const int fd = shm_open(name_.c_str(), O_RDWR, 0600);
        if (fd < 0) return false;
        close(fd);
        return true;
    }

    void destroy_successfully() {
        const int rc = comm_destroy(handle_);
        if (rc == 0) handle_ = nullptr;
        ASSERT_EQ(rc, 0);
    }

    CommHandle handle_ = nullptr;
    std::string name_;
};

TEST_F(CommSimDomainCleanupTest, DestroyUnlinksAfterAbortAlreadyUnmapped) {
    unlink_failures = 1;
    ASSERT_NE(comm_abort_domain_windows(handle_, 1, 1, 0), 0);
    ASSERT_EQ(handle_->domain_allocations.at(1)->mmap_base, nullptr);
    ASSERT_TRUE(name_present());
    destroy_successfully();
    EXPECT_EQ(unmap_calls, 1);
    EXPECT_EQ(unlink_calls, 2);
    EXPECT_FALSE(name_present());
}

TEST_F(CommSimDomainCleanupTest, DestroyRetainsOwnerWhenUnlinkFailsAgain) {
    unlink_failures = 2;
    ASSERT_NE(comm_abort_domain_windows(handle_, 1, 1, 0), 0);
    const int rc = comm_destroy(handle_);
    if (rc != COMM_RECLAMATION_UNCONFIRMED) handle_ = nullptr;
    ASSERT_EQ(rc, COMM_RECLAMATION_UNCONFIRMED);
    ASSERT_EQ(handle_->domain_allocations.count(1), 1u);
    EXPECT_EQ(handle_->domain_allocations.at(1)->mmap_base, nullptr);
    EXPECT_EQ(unmap_calls, 1);
    EXPECT_EQ(unlink_calls, 2);
    EXPECT_TRUE(name_present());
}

TEST_F(CommSimDomainCleanupTest, DestroyCompletesAbortAfterUnmapFailure) {
    unmap_failures = 1;
    ASSERT_NE(comm_abort_domain_windows(handle_, 1, 1, 0), 0);
    ASSERT_NE(handle_->domain_allocations.at(1)->mmap_base, nullptr);
    ASSERT_EQ(unlink_calls, 0);
    destroy_successfully();
    EXPECT_EQ(unmap_calls, 2);
    EXPECT_EQ(unlink_calls, 1);
    EXPECT_FALSE(name_present());
}

TEST_F(CommSimDomainCleanupTest, DestroyAcceptsNameAlreadyUnlinkedByPeer) {
    unlink_failures = 1;
    ASSERT_NE(comm_abort_domain_windows(handle_, 1, 1, 0), 0);
    ASSERT_EQ(shm_unlink(name_.c_str()), 0);
    destroy_successfully();
    EXPECT_EQ(unmap_calls, 1);
    EXPECT_EQ(unlink_calls, 2);
    EXPECT_FALSE(name_present());
}

TEST_F(CommSimDomainCleanupTest, SnapshotNamesOnlyTheActualPublishedDomainOwner) {
    CommDomainSnapshot snapshot;
    ASSERT_EQ(comm_query_domain_snapshot(handle_, 1, &snapshot), 0);
    EXPECT_EQ(snapshot.use.kind, CommDomainKind::LOCAL);
    EXPECT_EQ(snapshot.use.id, 1u);
    EXPECT_EQ(snapshot.use.reserved, 0u);
    EXPECT_EQ(
        snapshot.use.device_context, reinterpret_cast<uint64_t>(handle_->domain_allocations.at(1)->host_ctx.get())
    );
    EXPECT_EQ(snapshot.context.rankId, 0u);
    EXPECT_EQ(snapshot.context.rankNum, 1u);
    EXPECT_EQ(snapshot.context.winSize, 4096u);
    EXPECT_EQ(snapshot.context.windowsIn[0], handle_->domain_allocations.at(1)->host_ctx->windowsIn[0]);
    EXPECT_EQ(snapshot.context.windowsOut[0], snapshot.context.windowsIn[0]);
    EXPECT_EQ(snapshot.context.workspaces[0].addr, 0u);  // No simulated SDMA qualification.
    const CommDomainSnapshot saved = snapshot;
    EXPECT_NE(comm_query_domain_snapshot(handle_, 2, &snapshot), 0);
    EXPECT_EQ(snapshot.use.device_context, 0u);
    EXPECT_EQ(snapshot.context.winSize, 0u);
    EXPECT_NE(comm_query_domain_snapshot(nullptr, 1, &snapshot), 0);
    EXPECT_NE(comm_query_domain_snapshot(handle_, 0, &snapshot), 0);
    EXPECT_NE(comm_query_domain_snapshot(handle_, 1, nullptr), 0);
    ASSERT_EQ(comm_release_domain_windows(handle_, 1, 1, 0), 0);
    EXPECT_NE(comm_query_domain_snapshot(handle_, 1, &snapshot), 0);
    EXPECT_EQ(snapshot.use.id, 0u);
    EXPECT_EQ(saved.context.winSize, 4096u);  // The answer is a value, not another allocation owner.
}

TEST_F(CommSimDomainCleanupTest, SnapshotRefusesPartiallyReleasedDomainWithRetainedMetadata) {
    unlink_failures = 1;
    ASSERT_NE(comm_abort_domain_windows(handle_, 1, 1, 0), 0);
    ASSERT_NE(handle_->domain_allocations.at(1)->host_ctx, nullptr);
    ASSERT_EQ(handle_->domain_allocations.at(1)->mmap_base, nullptr);
    CommDomainSnapshot snapshot;
    snapshot.use.id = 99;
    EXPECT_NE(comm_query_domain_snapshot(handle_, 1, &snapshot), 0);
    EXPECT_EQ(snapshot.use.id, 0u);
    EXPECT_EQ(snapshot.context.windowsIn[0], 0u);
    destroy_successfully();
}

TEST_F(CommSimDomainCleanupTest, LocalZeroSnapshotRequiresItsActualLiveAllocation) {
    CommDomainSnapshot snapshot{};
    EXPECT_NE(comm_query_domain_snapshot(handle_, 0, &snapshot), 0);
    const uint32_t ranks[] = {0};
    uint64_t context = 0;
    uint64_t base = 0;
    size_t size = 0;
    ASSERT_EQ(comm_alloc_domain_windows(handle_, 0, ranks, 1, 0, 4096, 0, &context, &base, &size), 0);
    ASSERT_EQ(comm_query_domain_snapshot(handle_, 0, &snapshot), 0);
    EXPECT_TRUE(comm_domain_snapshot_valid(snapshot));
    EXPECT_EQ(snapshot.use.kind, CommDomainKind::LOCAL);
    EXPECT_EQ(snapshot.use.id, 0u);
    EXPECT_EQ(snapshot.use.device_context, context);
    EXPECT_EQ(snapshot.context.windowsIn[0], base);
    EXPECT_EQ(snapshot.context.winSize, 4096u);
    EXPECT_EQ(snapshot.context.workspaces[0].addr, 0u);
    ASSERT_EQ(comm_release_domain_windows(handle_, 0, 1, 0), 0);
    EXPECT_NE(comm_query_domain_snapshot(handle_, 0, &snapshot), 0);
    EXPECT_FALSE(comm_domain_snapshot_valid(snapshot));
    EXPECT_EQ(snapshot.use.device_context, 0u);
    EXPECT_EQ(snapshot.context.windowsIn[0], 0u);
}

TEST_F(CommSimDomainCleanupTest, GlobalSnapshotRequiresCompletedImportAndRevokesAfterRelease) {
    const uint64_t id = (static_cast<uint64_t>(getpid()) << 32) | 73;
    CommGlobalDomainDescriptor descriptor{};
    uint64_t base = 0, context = 0;
    ASSERT_EQ(comm_global_domain_prepare(id, 0, 1, 4096, COMM_GLOBAL_DOMAIN_PROFILE_SIM_SHM, &descriptor, &base), 0);
    CommDomainSnapshot snapshot;
    EXPECT_NE(comm_query_global_domain_snapshot(id, &snapshot), 0);
    EXPECT_EQ(snapshot.use.id, 0u);
    const int imported = comm_global_domain_import(id, &descriptor, 1, &context);
    EXPECT_EQ(imported, 0);
    if (imported == 0) {
        ASSERT_EQ(comm_query_global_domain_snapshot(id, &snapshot), 0);
        EXPECT_EQ(snapshot.use.kind, CommDomainKind::GLOBAL);
        EXPECT_EQ(snapshot.use.id, id);
        EXPECT_EQ(snapshot.use.device_context, context);
        EXPECT_EQ(snapshot.context.windowsIn[0], base);
        EXPECT_EQ(snapshot.context.winSize, 4096u);
    }
    EXPECT_NE(comm_query_global_domain_snapshot(id + 1, &snapshot), 0);
    EXPECT_EQ(snapshot.use.device_context, 0u);
    EXPECT_NE(comm_query_global_domain_snapshot(0, &snapshot), 0);
    EXPECT_NE(comm_query_global_domain_snapshot(id, nullptr), 0);
    EXPECT_EQ(comm_global_domain_release(id), 0);
    EXPECT_NE(comm_query_global_domain_snapshot(id, &snapshot), 0);
    EXPECT_EQ(snapshot.context.winSize, 0u);
}
}  // namespace
