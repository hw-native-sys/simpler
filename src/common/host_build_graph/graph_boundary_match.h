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

#pragma once

#include <string.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"

// What a recorded Graph boundary is compared against when a later submission presents the
// same key. Nothing here logs or reaches orchestrator state: every function is a predicate
// over a recorded boundary and a candidate argument list, which is what lets the reuse
// decision be tested on its own.

// One boundary tensor parameter reduced to what a later invocation is checked against.
//
// The match path reads every parameter on every same-key submission, and a parameter's own
// form is a 128-byte simpler::hbg::Tensor whose compared fields straddle both of its cache
// lines. Here one parameter's comparison is one cache line, and a boundary is a contiguous
// run of them carrying only what the comparison reads.
//
// Each field keeps the type it is compared against, so a comparison is a comparison rather
// than a cast.
//
// start_offset is here but is not compared directly: an argument may slide between
// invocations, and what is pinned is its offset from its alias partition's representative
// (see graph_boundary_arrangement_matches), which needs both this and the representative's.
struct GraphBoundaryTensorMatch {
    uint64_t buffer_size;
    uint64_t start_offset;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    // The lowest-numbered parameter sharing this one's buffer, itself when none earlier
    // does. A relation between parameters rather than a property of one, carried here so
    // the arrangement check reads it from the same line as the offset it pairs with.
    uint16_t alias_rep;
    uint8_t ndims;
    TensorArgType tag;
    DataType dtype;
    bool manual_dep;
    bool is_contiguous;
};
static_assert(sizeof(GraphBoundaryTensorMatch) == 64, "one parameter's comparison must stay one cache line");

// One recorded boundary scalar, reduced to what a later invocation is checked against.
//
// `dynamic` is the caller's declaration, which gen_scalar_params_from_args carries onto the
// boundary. `value` is what the slot held when the Definition was recorded.
struct GraphBoundaryScalarMatch {
    uint64_t value;
    bool dynamic;
};

// Everything a later submission on one Graph key is checked against.
//
// The two halves travel together because they answer one question -- whether a recorded
// Definition may be replayed for these arguments -- and a holder that kept them as separate
// members had to copy both on publication, where dropping one silently retires half the
// condition. Each vector's length is the recorded count of its element kind, so a holder
// states that invariant once, about one object.
struct GraphBoundaryMatchInfo {
    std::vector<GraphBoundaryTensorMatch> tensors;
    std::vector<GraphBoundaryScalarMatch> scalars;
};

inline GraphBoundaryTensorMatch graph_boundary_tensor_match_of(const simpler::hbg::Tensor &tensor, TensorArgType tag) {
    GraphBoundaryTensorMatch match{};
    match.buffer_size = tensor.buffer.size;
    match.start_offset = tensor.start_offset;
    // Only the dimensions in use, which is also the range graph_boundary_tensor_matches
    // compares: a value-initialized record then holds zero past ndims rather than
    // whatever the argument's unused slots happened to carry.
    std::copy(std::begin(tensor.shapes), std::begin(tensor.shapes) + tensor.ndims, std::begin(match.shapes));
    std::copy(std::begin(tensor.strides), std::begin(tensor.strides) + tensor.ndims, std::begin(match.strides));
    match.ndims = static_cast<uint8_t>(tensor.ndims);
    match.tag = tag;
    match.dtype = tensor.dtype;
    match.manual_dep = tensor.manual_dep;
    match.is_contiguous = tensor.is_contiguous;
    return match;
}

// Settle the alias partition of a boundary's parameters: which of them share a buffer.
// Fills `rep_out[i]` with the lowest-numbered parameter sharing parameter i's buffer --
// itself when no earlier one does -- which names the partition uniquely, so two boundaries
// form the same partition exactly when their rep arrays are equal.
//
// Grouping is by buffer address alone, so the answer needs no ordering: each parameter's
// representative is the first parameter that reached its address, and argument order is
// the scan order, so "first to arrive" is "lowest-numbered" for free. There are two
// implementations of that one contract below -- a hash and a sort -- and they agree on
// `rep_out` for every boundary either accepts.
//
// That two parameters over *different* addresses name non-overlapping memory is a
// precondition here, not something either checks. Every buffer a boundary can name comes
// from the one allocator, which hands out disjoint blocks. Proving it instead would need
// the addresses in order, and a sort cannot be narrowed to the pairs at risk because any
// pair may be the overlapping one.
//
// Two refusals remain, and both are answerable from one address's own group:
//   - an empty buffer, which holds no tensor and whose zero-width window would share an
//     address with the parameter after it;
//   - one address carrying two sizes, which contradicts what a parameter's recording-space
//     window is: graph_boundary_relocate_params reserves one window per address and sizes
//     it from the representative, so the group has to agree on how wide that is.
//
// Any edit to one of the two must land in the other: a boundary they disagree about would
// be accepted or refused depending on how much probing it happened to cost.
inline bool graph_alias_partition_sorted(const GraphTaskArgs &args, uint16_t *rep_out) {
    const int32_t count = args.tensor_count();
    std::array<int32_t, GRAPH_MAX_TENSOR_ARGS> order{};
    for (int32_t i = 0; i < count; ++i) {
        order[i] = i;
    }
    // Address, then argument index to make the order total -- so equal addresses come out
    // in argument order and a run's first entry is its representative.
    std::sort(order.begin(), order.begin() + count, [&args](int32_t lhs, int32_t rhs) {
        const uint64_t a = args.tensor(lhs).ref().buffer.addr;
        const uint64_t b = args.tensor(rhs).ref().buffer.addr;
        if (a != b) return a < b;
        return lhs < rhs;
    });

    for (int32_t k = 0; k < count;) {
        const int32_t rep = order[k];
        const uint64_t addr = args.tensor(rep).ref().buffer.addr;
        const uint64_t size = args.tensor(rep).ref().buffer.size;
        if (size == 0) return false;
        for (; k < count; ++k) {
            const int32_t member = order[k];
            if (args.tensor(member).ref().buffer.addr != addr) break;
            if (args.tensor(member).ref().buffer.size != size) return false;
            rep_out[member] = static_cast<uint16_t>(rep);
        }
    }
    return true;
}

inline bool graph_alias_partition(const GraphTaskArgs &args, uint16_t *rep_out) {
    // A slot holds a parameter index, not a key, so the table is one byte per slot and
    // clearing it is a single small memset. The keys stay in the parameters themselves: a
    // probe re-reads one, and that read costs nothing the pass did not already pay, since
    // the compare lands on the same simpler::hbg::Tensor cache line as the buffer.size
    // test above. At half load a parameter takes barely more than one probe.
    static_assert(GRAPH_MAX_TENSOR_ARGS < 0xFF, "a parameter index must fit the slot byte");
    constexpr uint32_t SLOT_BITS = 8;
    constexpr size_t SLOT_COUNT = size_t{1} << SLOT_BITS;
    static_assert(
        SLOT_COUNT >= 2 * size_t{GRAPH_MAX_TENSOR_ARGS}, "the table must stay under half full at the boundary cap"
    );
    constexpr uint8_t SLOT_EMPTY = 0xFF;
    // Probes this may spend before conceding to the sort, per parameter. Linear probing is
    // quadratic once the addresses cluster onto one slot, and the table's half-load bound
    // caps the average probe count but not that tail; conceding is what keeps an
    // O(n log n) ceiling under the routine as a whole.
    //
    // Half load costs about 1.5 probes per parameter, so four leaves room for ordinary
    // clustering while staying far below what a degenerate boundary spends. A concession
    // costs one sort, which bounds what an over-tight budget can cost.
    constexpr int32_t MAX_PROBES_PER_PARAM = 4;

    const int32_t count = args.tensor_count();
    std::array<uint8_t, SLOT_COUNT> slots;
    memset(slots.data(), SLOT_EMPTY, slots.size());
    int32_t probes_left = count * MAX_PROBES_PER_PARAM;

    for (int32_t i = 0; i < count; ++i) {
        const PTOBufferHandle &buffer = args.tensor(i).ref().buffer;
        if (buffer.size == 0) return false;
        size_t slot = addr_to_slot(buffer.addr, SLOT_BITS);
        while (true) {
            // The sort rewrites every entry of rep_out, so the entries this pass already
            // settled are replaced rather than mixed with.
            if (--probes_left < 0) return graph_alias_partition_sorted(args, rep_out);
            const uint8_t occupant = slots[slot];
            if (occupant == SLOT_EMPTY) {
                slots[slot] = static_cast<uint8_t>(i);
                rep_out[i] = static_cast<uint16_t>(i);
                break;
            }
            const PTOBufferHandle &held = args.tensor(occupant).ref().buffer;
            if (held.addr == buffer.addr) {
                if (held.size != buffer.size) return false;
                rep_out[i] = static_cast<uint16_t>(occupant);
                break;
            }
            slot = (slot + 1) & (SLOT_COUNT - 1);
        }
    }
    return true;
}

// Whether one boundary tensor parameter is presented again as it was recorded.
//
// Deliberately not compared: `buffer.addr` and `owner_task_id`, which a captured parameter
// carries in the recording's own terms rather than the caller's; `version` and
// `address_space`, which the contract lets vary; `extent_elem_cache`, which shapes and
// strides already settle; and start_offset, which an argument may slide between
// invocations -- what may not change is its offset *within its alias partition*, which
// graph_boundary_arrangement_matches checks for the boundary as a whole.
inline bool graph_boundary_tensor_matches(
    const GraphBoundaryTensorMatch &expected, const simpler::hbg::Tensor &actual, TensorArgType actual_tag
) {
    return actual.buffer.size == expected.buffer_size && actual.ndims == expected.ndims &&
           actual.dtype == expected.dtype && actual_tag == expected.tag && actual.manual_dep == expected.manual_dep &&
           actual.is_contiguous == expected.is_contiguous &&
           std::equal(
               std::begin(actual.shapes), std::begin(actual.shapes) + actual.ndims, std::begin(expected.shapes)
           ) &&
           std::equal(
               std::begin(actual.strides), std::begin(actual.strides) + actual.ndims, std::begin(expected.strides)
           );
}

// Whether `args` presents the alias partition the recording was captured with, and each
// parameter at the same offset within its partition.
//
// Two things are checked together because they are one property. A partition's members
// share a buffer, and the body's WAR/WAW edges were inferred from where their views sat
// in it; those edges are fixed in the Definition, so the arrangement has to come back.
// What may change is where the partition as a whole sits: every recorded tensor is rebuilt
// against the origin of the parameter it came from, so a uniform slide moves them all with
// it and the distances survive. A differential slide does not, which is what the offset
// comparison refuses.
//
// The partition is compared as the whole rep array rather than parameter by parameter: an
// array names its partition uniquely, so equality catches a class that split and a class
// that merged alike, and one parameter's rep on its own answers neither.
inline bool graph_boundary_arrangement_matches(const GraphBoundaryTensorMatch *recorded, const GraphTaskArgs &args) {
    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> actual_rep{};
    if (!graph_alias_partition(args, actual_rep.data())) return false;
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        if (actual_rep[i] != recorded[i].alias_rep) return false;
        // Signed, because a partition's representative is its lowest-numbered member
        // rather than its lowest-addressed one, so a member may sit before it.
        const int32_t rep = recorded[i].alias_rep;
        const int64_t actual_delta = static_cast<int64_t>(args.tensor(i).ref().start_offset) -
                                     static_cast<int64_t>(args.tensor(rep).ref().start_offset);
        const int64_t recorded_delta =
            static_cast<int64_t>(recorded[i].start_offset) - static_cast<int64_t>(recorded[rep].start_offset);
        if (actual_delta != recorded_delta) return false;
    }
    return true;
}

// Capture a boundary's scalars in the form graph_boundary_scalar_mismatch compares against.
// `out` has room for params.scalar_count() entries.
//
// `params` is the boundary rather than the caller's arguments: gen_scalar_params_from_args
// has already resolved every value and carried every declaration across, so this is the list
// a later same-key submission is checked against.
//
// pack_scalars rather than a per-slot read, because a dynamic parameter hands out its
// parameter rather than a value, and a comparison wants the raw slot either way.
inline void graph_boundary_capture_scalars(GraphBoundaryScalarMatch *out, const GraphTaskArgs &params) {
    std::array<uint64_t, GRAPH_MAX_SCALAR_ARGS> values{};
    params.pack_scalars(values.data());
    for (int32_t i = 0; i < params.scalar_count(); ++i) {
        out[i] = {values[i], params.scalar_dynamic(i)};
    }
}

// The first boundary scalar of `args` that does not present what the recording was captured
// with, or -1 when every one of them agrees. A slot agrees when its declaration matches and,
// where that declaration is static, its value does too.
//
// Only a static slot's value is compared, and that is the whole point of the check rather
// than a weakening of it. A Definition's scalar source refs index this boundary and the
// outer task's payload carries this invocation's values, so every dynamic slot comes back
// refreshed on replay and what it held at record time binds nothing -- comparing it would
// refuse a Definition that is still valid. A static slot is refreshed by nothing: whatever
// the body read out of it is fixed in the image, so its value is part of the condition the
// Definition is reused under.
//
// `recorded` is walked over args.scalar_count(), which both call sites pin equal to the
// recorded count before reaching here.
//
// Nothing is logged: each call site words its own warning from the index, which keeps this a
// predicate rather than a diagnostic.
inline int32_t graph_boundary_scalar_mismatch(const GraphBoundaryScalarMatch *recorded, const GraphTaskArgs &args) {
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        const bool dynamic = args.scalar_dynamic(i);
        if (dynamic != recorded[i].dynamic) return i;
        if (!dynamic && args.scalar<uint64_t>(i) != recorded[i].value) return i;
    }
    return -1;
}
