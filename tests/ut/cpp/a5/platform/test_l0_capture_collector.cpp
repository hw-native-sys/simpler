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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "common/l0_profiling.h"
#include "host/l0_capture_collector.h"
#include "common/platform/support/profiling_copy_fault.h"

extern "C" int l0_test_hal_open_channels();

extern "C" int rtMemset(void *destination, size_t capacity, int32_t value, size_t bytes) {
    if (bytes > capacity) return -1;
    std::memset(destination, value, bytes);
    return 0;
}

class L0CaptureSessionIo : public testing::TestWithParam<size_t> {
protected:
    L0CaptureSession session;
    std::vector<void *> allocations;

    void SetUp() override {
        copy_fault::reset();
        ASSERT_EQ(
            session.arm(
                0, 1, 1, 42,
                [this](size_t bytes) {
                    void *pointer = std::malloc(bytes);
                    allocations.push_back(pointer);
                    return pointer;
                },
                [this](void *pointer) {
                    allocations.erase(std::find(allocations.begin(), allocations.end(), pointer));
                    std::free(pointer);
                    return 0;
                }
            ),
            0
        );
    }

    void TearDown() override {
        session.reset(true);
        for (void *pointer : allocations)
            std::free(pointer);
        copy_fault::reset();
    }
};

TEST_P(L0CaptureSessionIo, SnapshotReadFailureRequiresDeviceReset) {
    copy_fault::arm({GetParam(), -7});

    EXPECT_EQ(session.read_terminal_snapshot(), -7);
    EXPECT_EQ(copy_fault::counts().from_device_failures, 1);
    EXPECT_TRUE(session.device_io_failed());
    EXPECT_NE(session.read_terminal_snapshot(), 0);
    EXPECT_EQ(copy_fault::counts().from_device_failures, 1);

    session.reset(true);
    EXPECT_TRUE(session.device_io_failed());
}

INSTANTIATE_TEST_SUITE_P(
    SnapshotStages, L0CaptureSessionIo, testing::Values(sizeof(L0CoreRegSnapshot), sizeof(L0CoreEndSnapshot))
);

class L0CollectorLifecycle : public testing::Test {
protected:
    L0CaptureCollector collector;
    NativeRunIdentity identity{42, 7, 19, 0};
    std::vector<void *> allocations;
    size_t frees{0};
    void *failing_free{nullptr};

    int prepare_terminal(const std::filesystem::path &path, bool stopped, uint32_t produced) {
        int rc = initialize(identity);
        if (rc != 0) return rc;
        rc = collector.begin_run(path);
        if (rc != 0) return rc;
        return collector.configure_run(identity, [=](L0ConfigArgs &args) {
            const auto *buffers = reinterpret_cast<const uint64_t *>(args.gm_addr_table);
            auto *start = reinterpret_cast<L0CoreRegSnapshot *>(args.snapshots);
            auto *end = reinterpret_cast<L0CoreEndSnapshot *>(args.end_snapshots);
            start[0].requested_base = buffers[0];
            start[0].observed_base = buffers[0];
            start[0].observed_len = L0CaptureSession::kRingLength;
            start[0].observed_global_en = 1;
            start[0].initial_wptr = 0;
            start[0].initial_samp_wrt = 0;
            end[0].wptr = produced;
            end[0].samp_wrt = produced;
            end[0].reserved = L0_DFX_STATUS_STAGED;
            auto *state = l0_dfx_state(reinterpret_cast<void *>(args.dfx_shm), 0);
            state->producer_stopped = stopped;
            state->core_type = L0_DFX_CORE_TYPE_AIC;
            const uint32_t words[]{0xE0000001u, 0xE0000000u, 0xE0000000u, 0xE0000000u,
                                   0xF0000001u, 0xF0010001u, 0xF0000001u, 0xEFFFFFFFu};
            std::memcpy(reinterpret_cast<void *>(buffers[0]), words, sizeof(words));
            return 0;
        });
    }

    int initialize(const NativeRunIdentity &run, int core_count = 1) {
        return collector.initialize(
            run, 0, core_count, 1,
            [this](size_t bytes) {
                void *pointer = std::malloc(bytes);
                if (pointer != nullptr) allocations.push_back(pointer);
                return pointer;
            },
            [this](void *pointer) {
                ++frees;
                if (pointer == failing_free) return -23;
                allocations.erase(std::find(allocations.begin(), allocations.end(), pointer));
                std::free(pointer);
                return 0;
            }
        );
    }

    void TearDown() override {
        collector.finalize_capture(true);
        for (void *pointer : allocations)
            std::free(pointer);
        copy_fault::reset();
    }
};

TEST(L0CaptureSession, RemappedDeviceOwnsClockAndChannels) {
    L0CaptureSession session;
    const char *previous = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    const std::string saved = previous ? previous : "";
    const bool had_previous = previous != nullptr;
    ASSERT_EQ(setenv("ASCEND_RT_VISIBLE_DEVICES", "5", 1), 0);
    const int rc = session.arm(0, 1, 1, 42, std::malloc, [](void *pointer) {
        std::free(pointer);
        return 0;
    });
    if (had_previous) {
        ASSERT_EQ(setenv("ASCEND_RT_VISIBLE_DEVICES", saved.c_str(), 1), 0);
    } else {
        ASSERT_EQ(unsetenv("ASCEND_RT_VISIBLE_DEVICES"), 0);
    }
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(l0_test_hal_open_channels(), 0);
    ASSERT_EQ(session.start_capture_channels(), 0);
    EXPECT_GT(l0_test_hal_open_channels(), 0);
    EXPECT_EQ(session.aic_frequency_hz(), 1300000000u);
    EXPECT_EQ(session.aiv_frequency_hz(), 1400000000u);
    EXPECT_EQ(session.stop_producer(), 0);
    EXPECT_EQ(l0_test_hal_open_channels(), 0);
}

TEST_F(L0CollectorLifecycle, RingFreeFailureReachesCloseAndBlocksReuse) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-ring-free-fault";
    ASSERT_EQ(prepare_terminal(path, true, 32), 0);
    failing_free = allocations.front();
    EXPECT_EQ(
        collector.close_run(
            identity, true,
            [] {
                return 0;
            }
        ),
        -23
    );
    std::filesystem::remove_all(path);
    EXPECT_EQ(allocations.size(), 2u);  // Failed ring and remaining metadata stay owned until reset.
    EXPECT_TRUE(collector.device_io_failed());
    EXPECT_FALSE(collector.capture_active());
    EXPECT_NE(initialize(identity), 0);
}

TEST_F(L0CollectorLifecycle, MetadataFreeFailureReachesCloseAndBlocksReuse) {
    ASSERT_EQ(initialize(identity), 0);
    failing_free = allocations.back();
    EXPECT_EQ(
        collector.close_run(
            identity, false,
            [] {
                return 0;
            }
        ),
        -23
    );
    EXPECT_TRUE(collector.device_io_failed());
    EXPECT_EQ(allocations.size(), 1u);
    EXPECT_NE(initialize(identity), 0);
}

TEST_F(L0CollectorLifecycle, IdleFinalizationDoesNotCreateDeviceFailure) {
    collector.finalize_capture(true);
    EXPECT_FALSE(collector.device_io_failed());
    EXPECT_FALSE(collector.capture_active());
    EXPECT_TRUE(allocations.empty());
}

TEST_F(L0CollectorLifecycle, SuccessorPreparationCannotReleasePredecessor) {
    ASSERT_EQ(initialize(identity), 0);
    auto successor = identity;
    ++successor.run_epoch;
    EXPECT_NE(initialize(successor), 0);
    bool synchronized = false;
    EXPECT_EQ(
        collector.close_run(
            successor, false,
            [&] {
                synchronized = true;
                return 0;
            }
        ),
        0
    );
    EXPECT_FALSE(synchronized);
    EXPECT_EQ(frees, 0u);
    EXPECT_TRUE(collector.capture_active());
    EXPECT_EQ(
        collector.close_run(
            identity, false,
            [] {
                return 0;
            }
        ),
        0
    );
    EXPECT_TRUE(allocations.empty());
    EXPECT_FALSE(collector.capture_active());
}

TEST_F(L0CollectorLifecycle, CleanupChecksWholeRunIdentityAndWaitsBeforeFreeing) {
    ASSERT_EQ(initialize(identity), 0);
    auto foreign = identity;
    ++foreign.generation;
    EXPECT_EQ(
        collector.close_run(
            foreign, false,
            [] {
                ADD_FAILURE();
                return -1;
            }
        ),
        0
    );
    EXPECT_NE(
        collector.configure_run(
            foreign,
            [](const L0ConfigArgs &) {
                ADD_FAILURE();
                return 0;
            }
        ),
        0
    );
    bool synchronized = false;
    EXPECT_EQ(
        collector.close_run(
            identity, false,
            [&] {
                EXPECT_EQ(frees, 0u);
                synchronized = true;
                return 0;
            }
        ),
        0
    );
    EXPECT_TRUE(synchronized);
    EXPECT_TRUE(allocations.empty());
    const size_t freed = frees;
    EXPECT_EQ(
        collector.close_run(
            identity, false,
            [] {
                ADD_FAILURE();
                return -1;
            }
        ),
        0
    );
    EXPECT_EQ(frees, freed);
}

TEST_F(L0CollectorLifecycle, FailedProducerStopRetainsDeviceAllocations) {
    ASSERT_EQ(initialize(identity), 0);
    EXPECT_NE(
        collector.close_run(
            identity, false,
            [] {
                return -7;
            }
        ),
        0
    );
    EXPECT_EQ(frees, 0u);
    EXPECT_TRUE(collector.device_io_failed());
    EXPECT_FALSE(collector.capture_active());
    EXPECT_NE(initialize(identity), 0);
}

TEST_F(L0CollectorLifecycle, OutputFailureDisablesOnlyCaptureAndReleasesResources) {
    ASSERT_EQ(initialize(identity), 0);
    const auto path = std::filesystem::temp_directory_path() /
                      ("simpler-l0-output-file-" + std::to_string(reinterpret_cast<uintptr_t>(this)));
    {
        std::ofstream file(path);
        file << "occupied";
    }
    EXPECT_NE(collector.start_run(identity, path.string()), 0);
    std::filesystem::remove(path);
    EXPECT_FALSE(collector.device_io_failed());
    EXPECT_FALSE(collector.capture_active());
    EXPECT_TRUE(allocations.empty());
    EXPECT_EQ(initialize(identity), 0);
    collector.finalize_capture(false);
    EXPECT_TRUE(allocations.empty());
}

TEST_F(L0CollectorLifecycle, TerminalCopyPreservesIndependentOfflineArtifacts) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-terminal-copy";
    std::filesystem::remove_all(path);
    const auto dir = path / "l0_raw/run_42";
    std::filesystem::create_directories(dir);
    for (const char *suffix : {".events.json", ".csv", ".gm.bin"})
        std::ofstream(dir / (std::string("core_0") + suffix)) << "stale";
    ASSERT_EQ(prepare_terminal(path, true, 32), 0);
    ASSERT_EQ(
        collector.close_run(
            identity, true,
            [] {
                return -1;
            }
        ),
        0
    );
    EXPECT_EQ(std::filesystem::file_size(dir / "core_0.bin"), 32u);
    EXPECT_TRUE(std::filesystem::exists(dir / "core_0.events.json"));
    EXPECT_TRUE(std::filesystem::exists(dir / "core_0.csv"));
    EXPECT_TRUE(std::filesystem::exists(dir / "core_0.gm.bin"));
    std::ifstream file(dir / "manifest.json");
    const std::string manifest(std::istreambuf_iterator<char>{file}, {});
    EXPECT_NE(manifest.find("\"capture_complete\": true"), std::string::npos);
    EXPECT_TRUE(allocations.empty());
    std::filesystem::remove_all(path);
}

TEST_F(L0CollectorLifecycle, TerminalCopyPreservesOrderAcrossRingWrap) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-terminal-wrap";
    std::filesystem::remove_all(path);
    ASSERT_EQ(prepare_terminal(path, true, 32), 0);
    const std::string expected = "0123456789abcdefABCDEFGHIJKLMNOP";
    ASSERT_EQ(
        collector.configure_run(
            identity,
            [&](L0ConfigArgs &args) {
                const auto *buffers = reinterpret_cast<const uint64_t *>(args.gm_addr_table);
                auto *start = reinterpret_cast<L0CoreRegSnapshot *>(args.snapshots);
                auto *end = reinterpret_cast<L0CoreEndSnapshot *>(args.end_snapshots);
                start[0].initial_wptr = L0CaptureSession::kRingLength - 16;
                end[0].wptr = 16;
                auto *buffer = reinterpret_cast<char *>(buffers[0]);
                std::memcpy(buffer + start[0].initial_wptr, expected.data(), 16);
                std::memcpy(buffer, expected.data() + 16, 16);
                return 0;
            }
        ),
        0
    );
    ASSERT_EQ(
        collector.close_run(
            identity, true,
            [] {
                return -1;
            }
        ),
        0
    );
    std::ifstream file(path / "l0_raw/run_42/core_0.bin", std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>{file}, {}), expected);
    EXPECT_TRUE(allocations.empty());
    std::filesystem::remove_all(path);
}

TEST_F(L0CollectorLifecycle, TerminalCopyFailureRetainsDeviceAllocationsUntilReset) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-terminal-copy-failure";
    std::filesystem::remove_all(path);
    ASSERT_EQ(prepare_terminal(path, true, 16), 0);
    copy_fault::arm({16, -7});
    EXPECT_NE(
        collector.close_run(
            identity, true,
            [] {
                return -1;
            }
        ),
        0
    );
    EXPECT_EQ(copy_fault::counts().from_device_failures, 1);
    EXPECT_TRUE(collector.device_io_failed());
    EXPECT_EQ(frees, 0u);
    EXPECT_FALSE(allocations.empty());
    EXPECT_FALSE(collector.capture_active());
    std::filesystem::remove_all(path);
}

TEST_F(L0CollectorLifecycle, MissingProducerStopMarksTerminalCopyIncomplete) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-terminal-stop";
    std::filesystem::remove_all(path);
    ASSERT_EQ(prepare_terminal(path, false, 32), 0);
    ASSERT_EQ(
        collector.close_run(
            identity, true,
            [] {
                return -1;
            }
        ),
        0
    );
    const auto dir = path / "l0_raw/run_42";
    EXPECT_EQ(std::filesystem::file_size(dir / "core_0.bin"), 0u);
    std::ifstream file(dir / "manifest.json");
    const std::string manifest(std::istreambuf_iterator<char>{file}, {});
    EXPECT_NE(manifest.find("\"capture_complete\": false"), std::string::npos);
    std::filesystem::remove_all(path);
}

TEST_F(L0CollectorLifecycle, OverwrittenTerminalRangeCannotClaimCompleteCapture) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-terminal-overrun";
    std::filesystem::remove_all(path);
    ASSERT_EQ(prepare_terminal(path, true, L0CaptureSession::kRingLength + 1), 0);
    ASSERT_EQ(
        collector.close_run(
            identity, true,
            [] {
                return -1;
            }
        ),
        0
    );
    const auto dir = path / "l0_raw/run_42";
    EXPECT_EQ(std::filesystem::file_size(dir / "core_0.bin"), 0u);
    std::ifstream file(dir / "manifest.json");
    const std::string manifest(std::istreambuf_iterator<char>{file}, {});
    EXPECT_NE(manifest.find("\"capture_complete\": false"), std::string::npos);
    std::filesystem::remove_all(path);
}

TEST_F(L0CollectorLifecycle, OneInvalidCoreKeepsOtherCoresRawData) {
    const auto path = std::filesystem::temp_directory_path() / "simpler-l0-partial-cores";
    std::filesystem::remove_all(path);
    ASSERT_EQ(initialize(identity, 2), 0);
    ASSERT_EQ(collector.begin_run(path), 0);
    const std::string expected = "0123456789abcdefABCDEFGHIJKLMNOP";
    ASSERT_EQ(
        collector.configure_run(
            identity,
            [&](L0ConfigArgs &args) {
                const auto *buffers = reinterpret_cast<const uint64_t *>(args.gm_addr_table);
                auto *starts = reinterpret_cast<L0CoreRegSnapshot *>(args.snapshots);
                auto *ends = reinterpret_cast<L0CoreEndSnapshot *>(args.end_snapshots);
                for (uint32_t core = 0; core < 2; ++core) {
                    starts[core].requested_base = starts[core].observed_base = buffers[core];
                    starts[core].observed_len = L0CaptureSession::kRingLength;
                    starts[core].observed_global_en = 1;
                    ends[core].wptr = ends[core].samp_wrt = static_cast<uint32_t>(expected.size());
                    ends[core].reserved = core == 0 ? L0_DFX_STATUS_UNTRUSTWORTHY : L0_DFX_STATUS_STAGED;
                    auto *state = l0_dfx_state(reinterpret_cast<void *>(args.dfx_shm), core);
                    state->producer_stopped = 1;
                    state->capture_error = core == 0 ? L0_DFX_CAPTURE_RANGE_ERROR : L0_DFX_CAPTURE_OK;
                    state->core_type = L0_DFX_CORE_TYPE_AIC;
                    std::memcpy(reinterpret_cast<void *>(buffers[core]), expected.data(), expected.size());
                }
                return 0;
            }
        ),
        0
    );
    ASSERT_EQ(
        collector.close_run(
            identity, true,
            [] {
                return 0;
            }
        ),
        0
    );
    const auto run = path / "l0_raw/run_42";
    EXPECT_EQ(std::filesystem::file_size(run / "core_0.bin"), 0u);
    std::ifstream raw(run / "core_1.bin", std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>{raw}, {}), expected);
    std::ifstream manifest_file(run / "manifest.json");
    const std::string manifest(std::istreambuf_iterator<char>{manifest_file}, {});
    EXPECT_NE(manifest.find("\n  \"capture_complete\": false"), std::string::npos);
    EXPECT_NE(manifest.find("\n  \"raw_transport_complete\": false"), std::string::npos);
    EXPECT_FALSE(collector.device_io_failed());
    EXPECT_TRUE(allocations.empty());
    std::filesystem::remove_all(path);
}
