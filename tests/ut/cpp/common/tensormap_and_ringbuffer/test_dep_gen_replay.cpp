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
    record.task_id = TaskId::make(0, 2).raw;
    record.explicit_dep_count = 1;
    record.explicit_deps[0] = TaskId::make(0, 1).raw;
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
