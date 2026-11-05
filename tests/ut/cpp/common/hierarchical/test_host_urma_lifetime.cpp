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
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#define private public
#define protected public
#include "chip_run_lane.h"
#include "chip_worker.h"
#include "runtime_module.h"
#undef private
#undef protected

namespace {
struct Faults {
    bool supported = true;
    int prepare = 0;
    int publish = 0;
    int revoke = 0;
    int release = 0;
    std::vector<std::string> events;
} faults;
constexpr uint64_t kHost = 0x10000;
constexpr uint64_t kMapped = 0x20000;
constexpr uint64_t kComm = 0x30000;
constexpr uint64_t kWorkspace = 0x40000;
}  // namespace

// RuntimeModule resolves the executable's symbols, exercising production lookup and ownership.
extern "C" int host_urma_supported_ctx(void *) { return faults.supported; }
extern "C" int host_urma_publish_ctx(void *, uint64_t workspace) {
    faults.events.push_back(workspace ? "publish" : "revoke");
    return workspace ? faults.publish : faults.revoke;
}
extern "C" int comm_prepare_host_urma(void *, uint64_t, uint64_t, uint64_t, uint64_t *workspace) {
    faults.events.push_back("prepare");
    *workspace = kWorkspace;
    return faults.prepare;
}
extern "C" int comm_release_host_urma(void *, uint64_t) {
    faults.events.push_back("release");
    return faults.release;
}
extern "C" int host_ddr_unmap_ctx(void *, int, uint64_t) {
    faults.events.push_back("unmap");
    return 0;
}

namespace {
class HostUrmaLifetime : public testing::Test {
protected:
    ChipWorker worker;
    void SetUp() override {
        faults = {};
        worker.runtime_module_ = std::make_shared<RuntimeModule>(nullptr);
        worker.device_ctx_ = reinterpret_cast<void *>(0x50000);
        worker.initialized_ = true;
        worker.host_ddr_mappings_.emplace(kHost, ChipWorker::HostDdrMapping{4096, kMapped, 1, false});
        worker.create_comm_session(reinterpret_cast<void *>(kComm), nullptr, true);
    }
    void TearDown() override {
        // No hardware exists in this fixture. Disarm artificial debt after checking retention.
        worker.host_urma_comm_ = 0;
        worker.host_ddr_mappings_.clear();
        worker.comm_reclamation_unconfirmed_ = false;
        worker.device_ctx_ = nullptr;
        worker.initialized_ = false;
    }
    void expect_retained() {
        EXPECT_NE(worker.host_urma_comm_, 0U);
        EXPECT_THROW(worker.unmap_host_ddr(kHost), std::runtime_error);
        EXPECT_THROW(worker.comm_destroy(kComm), std::runtime_error);
        EXPECT_THROW(worker.comm_destroy_all(), std::runtime_error);
        EXPECT_THROW(worker.finalize(), std::runtime_error);
    }
};

TEST_F(HostUrmaLifetime, SuccessfulReuseAndRevokeBeforeReleaseBeforeUnmap) {
    EXPECT_EQ(worker.prepare_host_urma(kComm, 1, kHost, 4096), kWorkspace);
    EXPECT_EQ(worker.prepare_host_urma(kComm, 1, kHost, 4096), kWorkspace);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "publish"}));
    EXPECT_THROW(worker.prepare_host_urma(kComm, 2, kHost, 4096), std::runtime_error);
    expect_retained();
    worker.release_host_urma();
    worker.release_host_urma();
    worker.unmap_host_ddr(kHost);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "publish", "revoke", "release", "unmap"}));
}

TEST_F(HostUrmaLifetime, PartialPrepareRetainsBackingUntilExplicitCollectiveRelease) {
    faults.prepare = 42;
    EXPECT_THROW(worker.prepare_host_urma(kComm, 1, kHost, 4096), std::runtime_error);
    expect_retained();
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare"}));
    worker.release_host_urma();
    worker.unmap_host_ddr(kHost);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "revoke", "release", "unmap"}));
}

TEST_F(HostUrmaLifetime, PublicationUncertaintyNeverCallsReleaseOrUnmap) {
    faults.publish = 42;
    EXPECT_THROW(worker.prepare_host_urma(kComm, 1, kHost, 4096), std::runtime_error);
    expect_retained();
    EXPECT_THROW(worker.release_host_urma(), std::runtime_error);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "publish"}));
    EXPECT_TRUE(worker.comm_reclamation_unconfirmed_);
}

TEST_F(HostUrmaLifetime, RevocationUncertaintyNeverCallsReleaseOrUnmap) {
    worker.prepare_host_urma(kComm, 1, kHost, 4096);
    faults.revoke = 42;
    EXPECT_THROW(worker.release_host_urma(), std::runtime_error);
    expect_retained();
    EXPECT_THROW(worker.release_host_urma(), std::runtime_error);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "publish", "revoke"}));
}

TEST_F(HostUrmaLifetime, FailedReleaseRetainsOwnerAndDoesNotRepublish) {
    worker.prepare_host_urma(kComm, 1, kHost, 4096);
    faults.release = 42;
    EXPECT_THROW(worker.release_host_urma(), std::runtime_error);
    expect_retained();
    EXPECT_THROW(worker.prepare_host_urma(kComm, 1, kHost, 4096), std::runtime_error);
    EXPECT_THROW(worker.release_host_urma(), std::runtime_error);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "publish", "revoke", "release", "release"}));
}

TEST_F(HostUrmaLifetime, UnsupportedAndInvalidMappingsHaveNoSideEffects) {
    faults.supported = false;
    EXPECT_FALSE(worker.host_urma_supported());
    EXPECT_THROW(worker.prepare_host_urma(kComm, 1, kHost, 4096), std::runtime_error);
    faults.supported = true;
    EXPECT_THROW(worker.prepare_host_urma(kComm, 1, kHost, 2048), std::invalid_argument);
    EXPECT_THROW(worker.prepare_host_urma(kComm + 1, 1, kHost, 4096), std::invalid_argument);
    EXPECT_THROW(worker.prepare_host_urma(kComm, 0, kHost, 4096), std::invalid_argument);
    EXPECT_TRUE(faults.events.empty());
    EXPECT_EQ(worker.host_urma_comm_, 0U);
}

TEST_F(HostUrmaLifetime, NativeRunOwnershipBlocksPreparationAndRelease) {
    worker.native_runs_.emplace(1, std::make_unique<ChipWorker::NativeRunEntry>());
    EXPECT_THROW(worker.prepare_host_urma(kComm, 1, kHost, 4096), std::runtime_error);
    EXPECT_TRUE(faults.events.empty());
    worker.native_runs_.clear();
    worker.prepare_host_urma(kComm, 1, kHost, 4096);
    worker.native_runs_.emplace(2, std::make_unique<ChipWorker::NativeRunEntry>());
    EXPECT_THROW(worker.release_host_urma(), std::runtime_error);
    EXPECT_EQ(faults.events, (std::vector<std::string>{"prepare", "publish"}));
    worker.native_runs_.clear();
    worker.release_host_urma();
}
}  // namespace

TEST_F(HostUrmaLifetime, FullBackingExtentIsNotTruncatedToPeerMetadataLength) {
    constexpr uint64_t extent = (uint64_t{1} << 32) + 65536;
    worker.host_ddr_mappings_.at(kHost).bytes = extent;
    EXPECT_EQ(worker.prepare_host_urma(kComm, 1, kHost, extent), kWorkspace);
    worker.release_host_urma();
    worker.unmap_host_ddr(kHost);
}
