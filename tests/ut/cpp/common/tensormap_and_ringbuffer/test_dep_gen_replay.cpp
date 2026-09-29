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

#include "dep_gen_replay.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <unistd.h>

#include "common/dep_gen.h"
#include "tensormap_and_ringbuffer/task_id.h"

namespace {

std::filesystem::path output_path() {
    return std::filesystem::temp_directory_path() /
           ("simpler_dep_gen_invalid_flags_" + std::to_string(::getpid()) + ".json");
}

std::filesystem::path empty_graph_output_path() {
    return std::filesystem::temp_directory_path() /
           ("simpler_dep_gen_empty_graph_" + std::to_string(::getpid()) + ".json");
}

std::filesystem::path chain_output_path(const char *name) {
    return std::filesystem::temp_directory_path() /
           ("simpler_dep_gen_chain_" + std::string(name) + "_" + std::to_string(::getpid()) + ".json");
}

/** A base record that declares a continuation, with no inline deps of its own. */
DepGenRecord chain_base(uint32_t local_id) {
    DepGenRecord base{};
    base.task_id = TaskId::make(0, local_id);
    base.flags = DEP_GEN_FLAG_HAS_OVERFLOW;
    base.explicit_dep_count = 0;
    base.tensor_count = 0;
    return base;
}

/** One continuation slot for `owner`, carrying `deps` dependencies. */
DepGenRecord chain_link(const DepGenRecord &owner, bool last, uint16_t deps, uint32_t first_dep_local) {
    DepGenRecord slot{};
    auto *over = reinterpret_cast<DepGenOverflowRecord *>(&slot);
    over->task_id = owner.task_id;
    over->flags = DEP_GEN_FLAG_OVERFLOW | (last ? DEP_GEN_FLAG_LAST_OVERFLOW : 0u);
    over->dep_count = deps;
    for (uint16_t i = 0; i < deps; i++) {
        over->deps[i] = TaskId::make(0, first_dep_local + i);
        over->kinds[i] = 0x3;  // wait | retain, the kinds the producer stamps
    }
    return slot;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream in(path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

}  // namespace

TEST(DepGenReplayTest, RejectsUnknownExplicitDepFlagBits) {
    const std::filesystem::path path = output_path();
    std::filesystem::remove(path);

    DepGenRecord record{};
    record.task_id = TaskId::make(0, 2);
    record.explicit_dep_count = 1;
    record.explicit_deps[0] = TaskId::make(0, 1);
    record.explicit_dep_kinds[0] = 0xff;

    EXPECT_EQ(dep_gen_replay_emit_deps_json(&record, 1, path.c_str()), -7);
    EXPECT_FALSE(std::filesystem::exists(path));
}

// Zero records is a valid graph, not a reason to write nothing: a run that
// submitted no task has an empty graph, and suppressing the file would make that
// indistinguishable from a collection failure. The host emit path relies on this
// — it hands over an empty span rather than declining when the window collected
// nothing — so the contract is pinned here.
TEST(DepGenReplayTest, ZeroRecordsEmitsAnEmptyGraph) {
    const std::filesystem::path path = empty_graph_output_path();
    std::filesystem::remove(path);

    // Null records with a zero count is the shape an empty std::vector yields.
    ASSERT_EQ(dep_gen_replay_emit_deps_json(nullptr, 0, path.c_str()), 0);
    ASSERT_TRUE(std::filesystem::exists(path)) << "a zero-record run produced no deps.json at all";

    const std::string json = read_file(path);
    EXPECT_NE(json.find("\"tasks\""), std::string::npos) << "empty graph is missing the tasks section: " << json;
    EXPECT_NE(json.find("\"edges\""), std::string::npos) << "empty graph is missing the edges section: " << json;

    std::filesystem::remove(path);
}

// A chain whose structure is broken must produce no graph.
//
// Every record below passes the per-record layout check and every count is in
// range, so nothing one-slot-at-a-time can see is wrong. The dual-pass
// self-check cannot catch it either: both passes are fed the same truncated
// dependency list and therefore agree. The replay used to log the missing
// terminator and emit a graph from that prefix — a graph missing edges, which
// `deps.json` has nowhere to describe.
TEST(DepGenReplayTest, RejectsAnUnterminatedOverflowChain) {
    const std::filesystem::path path = chain_output_path("unterminated");
    std::filesystem::remove(path);

    // The follow-up's exact input: a valid base declaring a continuation, with
    // no tensor args and no inline deps, and one matching continuation slot
    // that never says it is the last.
    DepGenRecord records[2];
    records[0] = chain_base(2);
    records[1] = chain_link(records[0], /*last=*/false, /*deps=*/0, /*first_dep_local=*/0);

    EXPECT_EQ(dep_gen_replay_emit_deps_json(records, 2, path.c_str()), -5);
    EXPECT_FALSE(std::filesystem::exists(path)) << "a structurally incomplete chain produced a graph";
}

// A continuation slot no base record claims is a rejection, not something to
// skip past. The outer scan used to ignore every overflow slot, so an orphan
// left no trace at all.
TEST(DepGenReplayTest, RejectsAnOrphanOverflowSlot) {
    const std::filesystem::path path = chain_output_path("orphan");
    std::filesystem::remove(path);

    const DepGenRecord owner = chain_base(5);
    DepGenRecord orphan = chain_link(owner, /*last=*/true, /*deps=*/1, /*first_dep_local=*/1);

    EXPECT_EQ(dep_gen_replay_emit_deps_json(&orphan, 1, path.c_str()), -5);
    EXPECT_FALSE(std::filesystem::exists(path)) << "an orphan continuation slot produced a graph";
}

// A continuation carrying someone else's task id breaks the chain it appears
// to continue, and leaves the real chain unterminated.
TEST(DepGenReplayTest, RejectsAChainWhoseContinuationNamesAnotherTask) {
    const std::filesystem::path path = chain_output_path("mismatched");
    std::filesystem::remove(path);

    const DepGenRecord other = chain_base(9);
    DepGenRecord records[2];
    records[0] = chain_base(2);
    records[1] = chain_link(other, /*last=*/true, /*deps=*/1, /*first_dep_local=*/1);

    EXPECT_EQ(dep_gen_replay_emit_deps_json(records, 2, path.c_str()), -5);
    EXPECT_FALSE(std::filesystem::exists(path));
}

// The rejection must be structural, not "any chain is suspicious": a chain of
// several legitimate continuation slots is exactly what a big-fanin submit
// produces, and it still emits its graph.
TEST(DepGenReplayTest, AcceptsAMultiSegmentChain) {
    const std::filesystem::path path = chain_output_path("multi_segment");
    std::filesystem::remove(path);

    // Two full continuation slots and a short terminator: the shape a submit
    // with more than 64 + DEP_GEN_OVERFLOW_DEPS_PER_RECORD deps takes.
    DepGenRecord records[4];
    records[0] = chain_base(4000);
    records[1] = chain_link(records[0], /*last=*/false, DEP_GEN_OVERFLOW_DEPS_PER_RECORD, /*first_dep_local=*/1);
    records[2] = chain_link(
        records[0], /*last=*/false, DEP_GEN_OVERFLOW_DEPS_PER_RECORD,
        /*first_dep_local=*/1 + DEP_GEN_OVERFLOW_DEPS_PER_RECORD
    );
    records[3] =
        chain_link(records[0], /*last=*/true, /*deps=*/3, /*first_dep_local=*/1 + 2 * DEP_GEN_OVERFLOW_DEPS_PER_RECORD);

    ASSERT_EQ(dep_gen_replay_emit_deps_json(records, 4, path.c_str()), 0);
    ASSERT_TRUE(std::filesystem::exists(path)) << "a well-formed multi-segment chain produced no graph";
    const std::string json = read_file(path);
    EXPECT_NE(json.find("\"edges\""), std::string::npos) << json;
    std::filesystem::remove(path);
}
