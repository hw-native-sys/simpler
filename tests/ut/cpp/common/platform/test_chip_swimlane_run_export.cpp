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
/**
 * The sealed run export owns a completed run's diagnostic data.
 *
 * The artifact is pinned against a golden captured from the **pre-change**
 * writer (`fe8b33899`), not against this writer's own output: comparing a
 * writer to itself would accept a dropped column, a renamed key or a reordered
 * section. `kGoldenArtifact` below is that capture, produced by driving
 * `chip_swimlane_run_export_fixture.h` through the old collector.
 *
 * The fixture is a fixed input — every timestamp, count and id is a literal —
 * so it is not a comparison between two real runs, whose timestamps differ by
 * construction. Two things cannot be fixed by a fixture:
 *
 *   - `metadata.runtime` is the runtime this binary was built for, so its
 *     *value* is normalized;
 *   - `host_clock_domain_id` is the host's boot identity, so its *value* is
 *     normalized — and the writer emits the field at all only on a host that
 *     can name its domain. macOS has no `/proc` and correctly omits it. That is
 *     a presence difference, not a value one, so it is **asserted** against the
 *     provider rather than normalized away, and the two dedicated presence
 *     cases below cover the emitted and omitted branches.
 */

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "chip_swimlane_run_export_fixture.h"

namespace {

namespace fixture = chip_swimlane_fixture;

// Captured from the writer at fe8b33899 by `.docs/arms/golden/gen_golden.sh`,
// driving the same fixture. Regenerate it there if the schema ever changes on
// purpose.
constexpr const char *kGoldenArtifact = R"GOLDEN({
  "chip_swimlane_level": 3,
  "metadata": {
    "runtime": "<RUNTIME>",
    "clock_freq_hz": 50000000,
    "num_cores": 2,
    "core_types": ["aic", "aiv"],
    "orchestrator_source": "host",
    "orchestrator_clock_domain": "host_monotonic_ns",
    "device_clock_domain": "device_syscnt_cycles",
    "host_timestamp_resolution_ns": 1,
    "host_timestamp_quantization_ns": 0,
    "host_orchestration_origin_ns": 550,
    "timeline_relation": "host_orchestration_precedes_device",
    "host_capture": {"status": "complete", "expected_records": 1, "recorded_records": 1, "pool_records": 3, "dropped_records": 0, "error": null},
    "host_clock_domain_id": "<HOST_CLOCK_DOMAIN_ID>",
    "core_to_thread": [0, 1]
  },
  "aicore_tasks": [
    [0, 0, 1, 1000, 1500, 7, 7700],
    [0, 1, 2, 1001, 1501, 8, 7700],
    [1, 100, 1, 1010, 1510, 7, 7700],
    [1, 101, 2, 1011, 1511, 8, 7700]
  ],
  "scheduler_tasks": {
    "producer": "aicpu",
    "records": [
    [0, 1, 900, 1600, 7700],
    [0, 2, 901, 1601, 7700]
    ]
  },
  "scheduler_records": {
    "streams": [
      {"platform": "a2a3", "producer": "aicpu", "scheduler_id": 0, "worker_id": 0, "core_type": "aicpu", "physical_core_id": null, "capture": {"committed": 2, "dropped": 5, "truncated": true}, "records": [
        {"start_cycles": 2000, "end_cycles": 2100, "run_epoch": 7700, "loop_iter": 0, "kind": "dispatch", "tasks_processed": 1, "task_id": null},
        {"start_cycles": 2001, "end_cycles": 2101, "run_epoch": 7700, "loop_iter": 1, "kind": "dummy_task", "tasks_processed": 2, "task_id": 20816}
      ], "metrics": [
        {"record_index": 0, "pop_hit": 3, "pop_miss": 4, "shared_at_start": [1,2,3], "shared_at_end": [2,3,4]},
        {"record_index": 1, "shared_at_start": [1,2,3], "shared_at_end": [2,3,4]}
      ]}
    ]
  },
  "aicpu_orchestrator_phases": [
    [
      {"submit_idx": 0, "task_id": 24832, "start_cycles": 3000, "end_cycles": 3100, "run_epoch": 7700},
      {"submit_idx": 1, "task_id": 24833, "start_cycles": 3001, "end_cycles": 3101, "run_epoch": 7700}
    ]
  ],
  "host_orchestrator_phases": [[
      {"submit_idx": 1, "task_id": 4242, "start_host_ns": 600, "end_host_ns": 700}
    ]],
  "host_device_uploads": [
      {"phase": "args", "start_host_ns": 550, "end_host_ns": 590, "detail": 128}
    ],
  "aicpu_lifecycle_records": [[11, 12]]
}
)GOLDEN";

// ctest runs the per-runtime variants of this binary concurrently, so the pid
// keeps their artifact directories apart.
std::string scratch_dir(const char *name) {
    return std::string(testing::TempDir()) + "/chip_swimlane_run_export." + name + "." + std::to_string(getpid());
}

struct ScratchDir {
    explicit ScratchDir(const char *name) :
        path(scratch_dir(name)) {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    ~ScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::string path;
};

int init_collector(ChipSwimlaneCollector &collector) {
    return collector.initialize(
        fixture::kNumAicore, fixture::kAicpuThreads, /*device_id=*/0, ChipSwimlaneLevel::SCHED_PHASES,
        fixture::fixture_alloc, nullptr, fixture::fixture_free
    );
}

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string artifact_path(const std::string &prefix) { return prefix + "/chip_swimlane_records.json"; }

std::string written_artifact(const std::string &prefix) {
    return fixture::normalize(read_file(artifact_path(prefix)), SIMPLER_RUNTIME_NAME);
}

/**
 * The golden as this host should produce it.
 *
 * The capture was taken where the host could name its clock domain, so it
 * carries `host_clock_domain_id`. A host that cannot name its domain — macOS,
 * which has no `/proc` — correctly omits that one line, and the golden is
 * adapted for it. This is not a blanket normalization: the field's presence is
 * asserted against the provider in `HostClockDomainIdPresenceFollowsTheProvider`
 * and by `expect_matches_golden` below, so a writer that emits the field on the
 * wrong host, or drops it on the right one, still fails. Every other column is
 * compared exactly on both.
 */
std::string expected_golden() {
    std::string golden(kGoldenArtifact);
    return fixture::host_clock_domain_id_available() ? golden : fixture::without_host_clock_domain_id(golden);
}

void expect_matches_golden(const std::string &prefix) {
    const std::string written = written_artifact(prefix);
    const bool emitted = written.find(fixture::kHostClockDomainIdKey) != std::string::npos;
    EXPECT_EQ(emitted, fixture::host_clock_domain_id_available())
        << "the host clock domain id must be emitted exactly when this host can name its domain";
    EXPECT_EQ(written, expected_golden());
}

}  // namespace

// The artifact is unchanged, field for field, against the writer this change
// replaced. A dropped column, a renamed key or a reordered section fails here.
TEST(ChipSwimlaneRunExportTest, SealedArtifactMatchesThePreChangeWriter) {
    ScratchDir scratch("golden");

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    fixture::PhaseBuffers buffers;
    fixture::populate(collector, buffers, scratch.path);

    ASSERT_EQ(collector.export_swimlane_json(), 0);
    expect_matches_golden(scratch.path);

    collector.finalize(nullptr, fixture::fixture_free);
}

// The point of the change: a sealed scope is this run's, and the collector
// moving on to another run — or being torn down entirely — cannot reach into
// it. The artifact is still the pre-change writer's, produced after both.
TEST(ChipSwimlaneRunExportTest, SealedScopeSurvivesBeginRunAndFinalize) {
    ScratchDir sealed("survives_sealed");
    ScratchDir successor("survives_successor");

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    fixture::PhaseBuffers buffers;
    fixture::populate(collector, buffers, sealed.path);

    ChipSwimlaneCollector::RunExport data = collector.seal_run_export();

    // Nothing of this run is left behind — neither the merged vectors nor the
    // per-shard copies the merge made.
    EXPECT_TRUE(collector.get_records().empty()) << "merged records stay with the sealed run";
    for (const auto &core : collector.collected_aicore_records_for_test()) {
        EXPECT_TRUE(core.empty()) << "per-shard AICore copies must be released at seal";
    }
    for (const auto &core : collector.collected_perf_records_for_test()) {
        EXPECT_TRUE(core.empty()) << "per-shard AICPU copies must be released at seal";
    }

    // A successor run resets every per-run field the writer used to read, and
    // finalize tears the region down underneath it.
    collector.begin_run(successor.path, ChipSwimlaneLevel::TASK_TIMING);
    collector.finalize(nullptr, fixture::fixture_free);

    EXPECT_EQ(data.output_prefix, sealed.path);
    EXPECT_EQ(data.level, ChipSwimlaneLevel::SCHED_PHASES);
    ASSERT_EQ(data.aicore_records.size(), static_cast<size_t>(fixture::kNumAicore));
    EXPECT_EQ(data.aicore_records[0].size(), 2u);
    ASSERT_FALSE(data.sched_phase_dropped_records.empty());
    EXPECT_EQ(data.sched_phase_dropped_records[0], fixture::kSchedDropped);
    EXPECT_EQ(data.num_orch_phase_threads, 1u);
    EXPECT_TRUE(data.host_phase_records_present);
    EXPECT_EQ(data.armed_run_epoch, fixture::kEpoch);
    EXPECT_TRUE(data.terminal_reported);

    ASSERT_EQ(ChipSwimlaneCollector::write_swimlane_json(data), 0);
    SCOPED_TRACE("the sealed run serializes the same after its collector moved on and shut down");
    expect_matches_golden(sealed.path);
    EXPECT_FALSE(std::filesystem::exists(artifact_path(successor.path)));
}

// Every populated field group travels with the run, including the two the
// writer used to read from the shared-memory region mid-serialization.
TEST(ChipSwimlaneRunExportTest, SealCapturesEveryPopulatedFieldGroup) {
    ScratchDir scratch("captures");

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    fixture::PhaseBuffers buffers;
    fixture::populate(collector, buffers, scratch.path);

    ChipSwimlaneCollector::RunExport data = collector.seal_run_export();

    EXPECT_EQ(data.num_aicore, fixture::kNumAicore);
    ASSERT_EQ(data.core_types.size(), 2u);
    EXPECT_EQ(data.core_types[0], CoreType::AIC);
    ASSERT_EQ(data.core_to_thread.size(), 2u);
    EXPECT_EQ(data.core_to_thread[1], 1);

    ASSERT_EQ(data.host_submit_records.size(), 1u);
    EXPECT_TRUE(data.host_phase_records_present);
    EXPECT_EQ(data.host_submit_records[0].payload, 4242u);
    ASSERT_EQ(data.host_upload_records.size(), 1u);
    EXPECT_EQ(data.host_phase_submitted_tasks, 1u);
    EXPECT_EQ(data.host_phase_total_records, 3u);

    EXPECT_EQ(
        data.json_extensions[static_cast<size_t>(ChipSwimlaneExtensionSection::AicpuLifecycleRecords)], "[[11, 12]]"
    );

    EXPECT_EQ(data.total_aicore_collected, 4u);
    EXPECT_EQ(data.total_perf_collected, 2u);
    EXPECT_TRUE(data.aicore_accounting.known);
    EXPECT_EQ(data.aicore_accounting.host_collected, 4u);
    EXPECT_TRUE(data.has_phase_data);
    EXPECT_TRUE(data.terminal_reported) << "the terminal read this run performed must travel with it";
    EXPECT_EQ(data.terminal_snapshot.run_epoch, fixture::kEpoch);

    // The two reads that moved out of the writer.
    ASSERT_GT(data.sched_phase_dropped_records.size(), 0u);
    EXPECT_EQ(data.sched_phase_dropped_records[0], fixture::kSchedDropped);
    EXPECT_EQ(data.num_orch_phase_threads, 1u);

    collector.finalize(nullptr, fixture::fixture_free);
}

// A run that produced nothing still refuses, still writes no file, and still
// returns the same rc through the production entry point.
TEST(ChipSwimlaneRunExportTest, SealedRunWithNoDataWritesNoArtifact) {
    ScratchDir scratch("no_data");

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    collector.begin_run(scratch.path, ChipSwimlaneLevel::SCHED_PHASES);

    ChipSwimlaneCollector::RunExport data = collector.seal_run_export();
    EXPECT_EQ(ChipSwimlaneCollector::write_swimlane_json(data), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(std::filesystem::exists(artifact_path(scratch.path)));

    EXPECT_EQ(collector.export_swimlane_json(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(std::filesystem::exists(artifact_path(scratch.path)));

    collector.finalize(nullptr, fixture::fixture_free);
}

// The region is gone, so the production entry refuses before sealing anything.
TEST(ChipSwimlaneRunExportTest, ExportWithoutARegionStillFails) {
    ChipSwimlaneCollector collector;
    EXPECT_EQ(collector.export_swimlane_json(), PTO_RUNTIME_ERR_INTERNAL);
}

// Emitted branch on a host that can name its clock domain, omitted branch on
// one that cannot — the field follows the provider, not the platform name and
// not the test's convenience.
TEST(ChipSwimlaneRunExportTest, HostClockDomainIdPresenceFollowsTheProvider) {
    ScratchDir scratch("clock_domain_presence");

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    fixture::PhaseBuffers buffers;
    fixture::populate(collector, buffers, scratch.path);
    ASSERT_EQ(collector.export_swimlane_json(), 0);

    // The raw artifact, so the identity's real value is visible.
    const std::string raw = read_file(artifact_path(scratch.path));
    const size_t at = raw.find(fixture::kHostClockDomainIdKey);

    if (fixture::host_clock_domain_id_available()) {
        ASSERT_NE(at, std::string::npos) << "a host that can name its clock domain must publish it";
        EXPECT_NE(raw.find("\"host_clock_domain_id\": \"linux-boot-id:", at), std::string::npos)
            << "the published identity must be the boot id the provider produced";
    } else {
        EXPECT_EQ(at, std::string::npos) << "a host that cannot name its clock domain must publish no identity";
    }

    collector.finalize(nullptr, fixture::fixture_free);
}

// The omitted branch, reachable on every host: with neither host phases nor a
// clock session there is no clock domain to name, whatever the provider could
// have supplied.
TEST(ChipSwimlaneRunExportTest, HostClockDomainIdOmittedWithoutHostPhaseOrClock) {
    ScratchDir scratch("clock_domain_gated_off");

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    fixture::PhaseBuffers buffers;
    fixture::populate(collector, buffers, scratch.path);

    ChipSwimlaneCollector::RunExport data = collector.seal_run_export();
    data.host_phase_records_present = false;
    ASSERT_EQ(ChipSwimlaneCollector::write_swimlane_json(data), 0);
    const std::string raw = read_file(artifact_path(scratch.path));
    EXPECT_EQ(raw.find(fixture::kHostClockDomainIdKey), std::string::npos);
    // The rest of the artifact is still produced, so this is the field's own
    // gate rather than an empty export.
    EXPECT_NE(raw.find("\"aicore_tasks\""), std::string::npos);

    collector.finalize(nullptr, fixture::fixture_free);
}
