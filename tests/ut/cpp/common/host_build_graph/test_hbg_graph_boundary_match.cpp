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
#include <cstdint>

#include "graph_boundary_match.h"
#include "host_build_graph/task_id.h"

namespace {

// A boundary tensor carrying exactly the geometry the predicates read. Every field a
// comparison looks at is a parameter here, so a case varies one of them and nothing else.
simpler::hbg::Tensor make_boundary_tensor(
    uint64_t address, uint64_t size = 64, uint64_t start_offset = 0, DataType dtype = DataType::FLOAT32,
    uint32_t extent = 16
) {
    simpler::hbg::TensorData data{};
    data.buffer.addr = address;
    data.buffer.size = size;
    data.owner_task_id = TaskId::invalid();
    data.start_offset = start_offset;
    data.extent_elem_cache = extent;
    data.shapes[0] = extent;
    data.strides[0] = 1;
    data.ndims = 1;
    data.dtype = dtype;
    data.is_contiguous = true;
    simpler::hbg::Tensor tensor{};
    tensor.init_from(data);
    return tensor;
}

// The recorded side as the runtime builds it: a capture over the boundary's own arguments,
// then the alias representative written onto each entry. Both halves are needed because
// the capture does not set alias_rep and the arrangement check reads nothing else.
struct RecordedBoundaryTensors {
    explicit RecordedBoundaryTensors(const GraphTaskArgs &args) {
        std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> rep{};
        accepted = graph_alias_partition(args, rep.data());
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            match[i] = graph_boundary_tensor_match_of(args.tensor(i).ref(), args.tag(i));
            match[i].alias_rep = rep[i];
        }
    }

    bool accepted{false};
    std::array<GraphBoundaryTensorMatch, GRAPH_MAX_TENSOR_ARGS> match{};
};

// The recorded side as the runtime builds it: the boundary's own parameter list, then the
// capture over it. Going through both is the point -- a capture that disagreed with the
// comparison would still pass a test that hand-built the recorded array.
struct RecordedBoundaryScalars {
    explicit RecordedBoundaryScalars(const GraphTaskArgs &args) {
        params.gen_scalar_params_from_args(args);
        graph_boundary_capture_scalars(match.data(), params);
    }

    GraphTaskArgs params;
    std::array<GraphBoundaryScalarMatch, GRAPH_MAX_SCALAR_ARGS> match{};
};

}  // namespace

// ---------------------------------------------------------------------------------------
// graph_alias_partition: the hash implementation against the sort it concedes to
// ---------------------------------------------------------------------------------------

namespace {

// One contract, two implementations. They are compared rather than each checked against a
// hand-written expectation because what the pair must hold is agreement: a boundary the two
// answer differently is accepted or refused according to how much probing it happened to
// cost, which is a property of the address bits rather than of the boundary.
void expect_implementations_agree(const GraphTaskArgs &args) {
    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> hashed{};
    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> sorted{};
    hashed.fill(0xFFFF);
    sorted.fill(0xFFFF);

    const bool hashed_ok = graph_alias_partition(args, hashed.data());
    const bool sorted_ok = graph_alias_partition_sorted(args, sorted.data());

    ASSERT_EQ(hashed_ok, sorted_ok);
    if (!hashed_ok) return;
    // Only an accepted boundary has a settled partition: a refusal returns as soon as it
    // has its answer, so the entries past that point were never written.
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        EXPECT_EQ(hashed[i], sorted[i]) << "parameter " << i;
    }
}

// Multiplicative inverse of `a` modulo 2^64, for odd `a`. Newton's iteration doubles the
// number of correct low bits per step, and three correct bits seed it, so five steps reach
// all 64.
constexpr uint64_t mod_inverse_64(uint64_t a) {
    uint64_t x = a;
    for (int i = 0; i < 5; ++i) {
        x *= 2 - a * x;
    }
    return x;
}

// An address that addr_to_slot sends to `slot`, distinct for each `nonce`. The slot is the
// top bits of addr * GOLDEN, so inverting that constant turns a chosen product into the
// address producing it.
uint64_t address_hashing_to_slot(uint32_t slot, uint64_t nonce) {
    constexpr uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
    constexpr uint64_t GOLDEN_INVERSE = mod_inverse_64(GOLDEN);
    const uint64_t product = (static_cast<uint64_t>(slot) << 56) | nonce;
    return product * GOLDEN_INVERSE;
}

uint64_t next_random(uint64_t &state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 17;
}

}  // namespace

// The sweep the two implementations have to survive: parameter counts from one to the cap,
// drawn from a buffer pool narrow enough that aliasing is the common case rather than the
// exception, and wide enough that a run of distinct buffers also occurs.
TEST(GraphAliasPartitionAgreement, TheHashAndTheSortSettleTheSamePartition) {
    uint64_t state = 0x9E3779B9ULL;
    for (int32_t trial = 0; trial < 400; ++trial) {
        const int32_t count = 1 + static_cast<int32_t>(next_random(state) % GRAPH_MAX_TENSOR_ARGS);
        const uint64_t pool = 1 + next_random(state) % 6;

        std::array<simpler::hbg::Tensor, GRAPH_MAX_TENSOR_ARGS> storage{};
        GraphTaskArgs args;
        for (int32_t i = 0; i < count; ++i) {
            const uint64_t slot = next_random(state) % pool;
            storage[i] = make_boundary_tensor(0x10000 + slot * 0x1000, 64);
            args.add_input(storage[i]);
        }
        ASSERT_EQ(args.tensor_count(), count);
        expect_implementations_agree(args);
    }
}

// A boundary whose addresses all land on one slot of the hash table. Past its probe budget
// the hash implementation concedes to the sort, and the concession rewrites every entry of
// rep_out rather than mixing with the ones it had already settled -- so the answer still
// has to be the sort's.
TEST(GraphAliasPartitionAgreement, ASlotCollisionRunConcedesToTheSortAndKeepsItsAnswer) {
    constexpr int32_t COUNT = GRAPH_MAX_TENSOR_ARGS;
    constexpr uint32_t SLOT_BITS = 8;
    constexpr uint32_t TARGET_SLOT = 0x5A;
    std::array<simpler::hbg::Tensor, COUNT> storage{};
    GraphTaskArgs args;
    for (int32_t i = 0; i < COUNT; ++i) {
        // Distinct addresses, one slot: the probe budget is what they exhaust, not the
        // table's capacity.
        const uint64_t address = address_hashing_to_slot(TARGET_SLOT, static_cast<uint64_t>(i));
        // The concession is what this case exists to reach, and it is reached only if the
        // crafted addresses really collide -- so the collision is asserted rather than
        // assumed. Probing walks the run already placed, so the parameters spend about
        // COUNT^2/2 probes against a budget of COUNT*4.
        ASSERT_EQ(addr_to_slot(address, SLOT_BITS), TARGET_SLOT) << "parameter " << i;
        storage[i] = make_boundary_tensor(address, 64);
        args.add_input(storage[i]);
    }

    expect_implementations_agree(args);

    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> rep{};
    ASSERT_TRUE(graph_alias_partition(args, rep.data()));
    // Distinct buffers, so every parameter represents itself.
    for (int32_t i = 0; i < COUNT; ++i) {
        EXPECT_EQ(rep[i], static_cast<uint16_t>(i));
    }
}

// An empty buffer holds no tensor, and its zero-width window would share an address with
// the parameter after it. Both implementations answer that from one address's own group,
// so neither may let it through.
TEST(GraphAliasPartitionAgreement, BothRefuseAnEmptyBuffer) {
    simpler::hbg::Tensor occupied = make_boundary_tensor(0x20000, 64);
    simpler::hbg::Tensor empty = make_boundary_tensor(0x30000, 0);
    GraphTaskArgs args;
    args.add_input(occupied);
    args.add_input(empty);

    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> rep{};
    EXPECT_FALSE(graph_alias_partition(args, rep.data()));
    EXPECT_FALSE(graph_alias_partition_sorted(args, rep.data()));
}

// One address carrying two sizes contradicts what a parameter's recording-space window is:
// relocation reserves one window per address and sizes it from the representative, so the
// group has to agree on how wide that is.
TEST(GraphAliasPartitionAgreement, BothRefuseOneAddressCarryingTwoSizes) {
    simpler::hbg::Tensor wide = make_boundary_tensor(0x40000, 128);
    simpler::hbg::Tensor narrow = make_boundary_tensor(0x40000, 64);
    GraphTaskArgs args;
    args.add_input(wide);
    args.add_input(narrow);

    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> rep{};
    EXPECT_FALSE(graph_alias_partition(args, rep.data()));
    EXPECT_FALSE(graph_alias_partition_sorted(args, rep.data()));
}

// A representative is the lowest-numbered member of its group, which argument order gives
// for free on the hash side and the sort has to reproduce through an address ordering that
// says nothing about argument index.
TEST(GraphAliasPartitionAgreement, ARepresentativeIsItsGroupsLowestNumberedMember) {
    // Descending addresses, so the sort's scan order is the reverse of argument order.
    simpler::hbg::Tensor high = make_boundary_tensor(0x90000, 64);
    simpler::hbg::Tensor low = make_boundary_tensor(0x50000, 64);
    simpler::hbg::Tensor high_again = make_boundary_tensor(0x90000, 64);
    GraphTaskArgs args;
    args.add_input(high);
    args.add_input(low);
    args.add_input(high_again);

    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> rep{};
    ASSERT_TRUE(graph_alias_partition(args, rep.data()));
    EXPECT_EQ(rep[0], 0u);
    EXPECT_EQ(rep[1], 1u);
    EXPECT_EQ(rep[2], 0u) << "parameter 2 shares parameter 0's buffer";
    expect_implementations_agree(args);
}

// ---------------------------------------------------------------------------------------
// graph_boundary_tensor_matches: one parameter's geometry
// ---------------------------------------------------------------------------------------

TEST(GraphBoundaryTensorCondition, TheSameGeometryMatches) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    // A different address, which is deliberately not compared: a captured parameter carries
    // the recording's own address rather than the caller's.
    simpler::hbg::Tensor actual = make_boundary_tensor(0x70000);
    EXPECT_TRUE(graph_boundary_tensor_matches(recorded.match[0], actual, TensorArgType::INPUT));
}

// start_offset is captured but not compared here. An argument may slide between
// invocations; what may not move is its offset within its alias partition, which
// graph_boundary_arrangement_matches checks for the boundary as a whole.
TEST(GraphBoundaryTensorCondition, AnOriginThatSlidStillMatches) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000, 64, 0);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    simpler::hbg::Tensor slid = make_boundary_tensor(0x60000, 64, 4);
    EXPECT_TRUE(graph_boundary_tensor_matches(recorded.match[0], slid, TensorArgType::INPUT));
}

TEST(GraphBoundaryTensorCondition, ABufferSizeThatMovedIsRefused) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000, 64);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    simpler::hbg::Tensor resized = make_boundary_tensor(0x60000, 128);
    EXPECT_FALSE(graph_boundary_tensor_matches(recorded.match[0], resized, TensorArgType::INPUT));
}

TEST(GraphBoundaryTensorCondition, ADtypeThatMovedIsRefused) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000, 64, 0, DataType::FLOAT32);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    simpler::hbg::Tensor retyped = make_boundary_tensor(0x60000, 64, 0, DataType::INT32);
    EXPECT_FALSE(graph_boundary_tensor_matches(recorded.match[0], retyped, TensorArgType::INPUT));
}

TEST(GraphBoundaryTensorCondition, AShapeThatMovedIsRefused) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000, 64, 0, DataType::FLOAT32, 16);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    simpler::hbg::Tensor reshaped = make_boundary_tensor(0x60000, 64, 0, DataType::FLOAT32, 8);
    EXPECT_FALSE(graph_boundary_tensor_matches(recorded.match[0], reshaped, TensorArgType::INPUT));
}

// The tag is the parameter's direction, which the Definition's dataflow is recorded
// against: the same buffer presented as an output is not the same parameter.
TEST(GraphBoundaryTensorCondition, ATagThatMovedIsRefused) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    simpler::hbg::Tensor actual = make_boundary_tensor(0x60000);
    EXPECT_FALSE(graph_boundary_tensor_matches(recorded.match[0], actual, TensorArgType::INOUT));
}

// Dependency tracking is inferred from the recorded flag, so a parameter that stopped
// declaring creator-only tracking replays a DAG the body never had.
TEST(GraphBoundaryTensorCondition, AManualDepDeclarationThatMovedIsRefused) {
    simpler::hbg::Tensor recorded_tensor = make_boundary_tensor(0x60000);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(recorded_tensor);
    const RecordedBoundaryTensors recorded(recorded_args);

    simpler::hbg::Tensor actual = make_boundary_tensor(0x60000);
    actual.manual_dep = true;
    EXPECT_FALSE(graph_boundary_tensor_matches(recorded.match[0], actual, TensorArgType::INPUT));
}

// ---------------------------------------------------------------------------------------
// graph_boundary_arrangement_matches: the partition, and each origin within it
// ---------------------------------------------------------------------------------------

// Every recorded tensor is rebuilt against the origin of the parameter it came from, so a
// slide that moves a whole partition together keeps every distance inside it.
TEST(GraphBoundaryArrangementCondition, AUniformSlideOfOnePartitionMatches) {
    simpler::hbg::Tensor first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor second = make_boundary_tensor(0x80000, 64, 8);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(first);
    recorded_args.add_input(second);
    const RecordedBoundaryTensors recorded(recorded_args);
    ASSERT_TRUE(recorded.accepted);

    simpler::hbg::Tensor slid_first = make_boundary_tensor(0x80000, 64, 4);
    simpler::hbg::Tensor slid_second = make_boundary_tensor(0x80000, 64, 12);
    GraphTaskArgs args;
    args.add_input(slid_first);
    args.add_input(slid_second);

    EXPECT_TRUE(graph_boundary_arrangement_matches(recorded.match.data(), args));
}

// A differential slide does not survive: the body's WAR/WAW edges were inferred from where
// the two views sat in the shared buffer, and those edges are fixed in the Definition.
TEST(GraphBoundaryArrangementCondition, ADifferentialSlideIsRefused) {
    simpler::hbg::Tensor first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor second = make_boundary_tensor(0x80000, 64, 8);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(first);
    recorded_args.add_input(second);
    const RecordedBoundaryTensors recorded(recorded_args);
    ASSERT_TRUE(recorded.accepted);

    simpler::hbg::Tensor held = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor moved = make_boundary_tensor(0x80000, 64, 16);
    GraphTaskArgs args;
    args.add_input(held);
    args.add_input(moved);

    EXPECT_FALSE(graph_boundary_arrangement_matches(recorded.match.data(), args));
}

// A partition that split drops the edges between the views that shared a buffer.
TEST(GraphBoundaryArrangementCondition, APartitionThatSplitIsRefused) {
    simpler::hbg::Tensor first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor second = make_boundary_tensor(0x80000, 64, 0);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(first);
    recorded_args.add_input(second);
    const RecordedBoundaryTensors recorded(recorded_args);
    ASSERT_TRUE(recorded.accepted);

    simpler::hbg::Tensor apart_first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor apart_second = make_boundary_tensor(0xA0000, 64, 0);
    GraphTaskArgs args;
    args.add_input(apart_first);
    args.add_input(apart_second);

    EXPECT_FALSE(graph_boundary_arrangement_matches(recorded.match.data(), args));
}

// A partition that merged invents edges the body never had.
TEST(GraphBoundaryArrangementCondition, APartitionThatMergedIsRefused) {
    simpler::hbg::Tensor first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor second = make_boundary_tensor(0xA0000, 64, 0);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(first);
    recorded_args.add_input(second);
    const RecordedBoundaryTensors recorded(recorded_args);
    ASSERT_TRUE(recorded.accepted);

    simpler::hbg::Tensor together_first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor together_second = make_boundary_tensor(0x80000, 64, 0);
    GraphTaskArgs args;
    args.add_input(together_first);
    args.add_input(together_second);

    EXPECT_FALSE(graph_boundary_arrangement_matches(recorded.match.data(), args));
}

// A boundary the partition refuses cannot have an arrangement either: the check settles the
// candidate's own partition before comparing it, and there is nothing to compare against a
// boundary that has none.
TEST(GraphBoundaryArrangementCondition, ACandidateWithNoPartitionIsRefused) {
    simpler::hbg::Tensor first = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor second = make_boundary_tensor(0xA0000, 64, 0);
    GraphTaskArgs recorded_args;
    recorded_args.add_input(first);
    recorded_args.add_input(second);
    const RecordedBoundaryTensors recorded(recorded_args);
    ASSERT_TRUE(recorded.accepted);

    simpler::hbg::Tensor occupied = make_boundary_tensor(0x80000, 64, 0);
    simpler::hbg::Tensor empty = make_boundary_tensor(0xA0000, 0, 0);
    GraphTaskArgs args;
    args.add_input(occupied);
    args.add_input(empty);

    EXPECT_FALSE(graph_boundary_arrangement_matches(recorded.match.data(), args));
}

// ---------------------------------------------------------------------------------------
// graph_boundary_scalar_mismatch: one slot's declaration, and a static slot's value
// ---------------------------------------------------------------------------------------

// A dynamic parameter is refreshed out of this invocation's own payload, so a Definition
// recorded against one stays valid however its value moved. This is what a whole-list
// comparison got wrong: every boundary scalar of graph_execution's layers is dynamic and
// changes per layer, so comparing them refused a Definition that was still correct.
//
// This is the admission half only. That the refreshed value is the one the task actually
// receives is GraphExecutionReplay.ResubmissionRebuildsFromDefinition in
// test_hbg_graph_cache.cpp.
TEST(GraphBoundaryScalarCondition, EveryDynamicSlotMatchesWhateverItsValue) {
    uint64_t left = 7;
    uint64_t right = 11;
    GraphTaskArgs recorded_args;
    recorded_args.add_scalar(left, right);
    const RecordedBoundaryScalars recorded(recorded_args);

    uint64_t moved_left = 70;
    uint64_t moved_right = 110;
    GraphTaskArgs args;
    args.add_scalar(moved_left, moved_right);

    EXPECT_EQ(graph_boundary_scalar_mismatch(recorded.match.data(), args), -1);
}

TEST(GraphBoundaryScalarCondition, AStaticSlotHoldingItsValueMatches) {
    GraphTaskArgs recorded_args;
    recorded_args.add_static_scalar(uint64_t{4});
    const RecordedBoundaryScalars recorded(recorded_args);

    GraphTaskArgs args;
    args.add_static_scalar(uint64_t{4});

    EXPECT_EQ(graph_boundary_scalar_mismatch(recorded.match.data(), args), -1);
}

// Nothing refreshes a static slot on replay, so whatever the body read out of it is fixed in
// the image. A different value is therefore a different Definition, and the index is what
// tells the author which of a dozen parameters to look at.
TEST(GraphBoundaryScalarCondition, AStaticSlotThatMovedNamesItsIndex) {
    uint64_t dynamic_first = 1;
    GraphTaskArgs recorded_args;
    recorded_args.add_scalar(dynamic_first);
    recorded_args.add_static_scalar(uint64_t{4});
    const RecordedBoundaryScalars recorded(recorded_args);

    uint64_t dynamic_moved = 99;
    GraphTaskArgs args;
    args.add_scalar(dynamic_moved);
    args.add_static_scalar(uint64_t{8});

    EXPECT_EQ(graph_boundary_scalar_mismatch(recorded.match.data(), args), 1);
}

// Same value, declared the other way. The declaration decides whether replay refreshes the
// slot, so a Definition recorded under one of them says nothing about the other.
TEST(GraphBoundaryScalarCondition, ADeclarationThatDiffersNamesItsIndex) {
    uint64_t held = 4;
    GraphTaskArgs recorded_args;
    recorded_args.add_scalar(held);
    const RecordedBoundaryScalars recorded(recorded_args);

    GraphTaskArgs args;
    args.add_static_scalar(held);

    EXPECT_EQ(graph_boundary_scalar_mismatch(recorded.match.data(), args), 0);
}
