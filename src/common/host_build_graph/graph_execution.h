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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cstddef>
#include <type_traits>

#include "host_build_graph/task_id.h"
#include "host_build_graph/runtime_types.h"
#include "tensor.h"

inline constexpr int32_t SUB_TASK_MAX_NUM = 1024;
// A body's producers precede their consumers, so a fanin CSR row holds at most
// task_count - 1 entries, and both of the slot's row cursors index one of them.
// A sub-task's row has no cap of its own — unlike a GLOBAL task's inline row,
// which append_fanin_or_fail holds to CHIP_MAX_FANIN — so this is what bounds
// them, and a cursor too narrow for it would report a row scanned that was not.
static_assert(SUB_TASK_MAX_NUM - 1 < 0xFFFF, "a fanin CSR row index must fit ChipTaskSlotState::wake_scan_cursor");
static_assert(
    SUB_TASK_MAX_NUM - 1 < 0xFFFF && CHIP_MAX_FANIN - 1 < 0xFFFF,
    "a fanin row index from either cohort must fit ChipTaskSlotState::ed_publish_scan_cursor"
);
inline constexpr int32_t GRAPH_MATERIALIZE_SLICE_TASKS = 4;

// Every type a Definition section holds is copied across the host-device boundary, so all
// of them are pointer-free, fixed-width and position-independent: a reference is an offset
// from its owning header, never an address.
//
// A tensor travels as simpler::hbg::TensorData, the 96-byte base simpler::hbg::Tensor
// derives from -- see its declaration for why the image cannot hold the aligned form.

// Where one sub-task scalar slot takes its value from: the Definition's own
// scalars[] entry, or the boundary parameter named by boundary_index(). It is the wire
// form of the two things recording knows about a slot -- whether it inherits, and which
// parameter it inherits -- so inherited() is the same predicate as Arg::scalar_inherited.
//
// Only a parameter of the replaying Graph's boundary can be refreshed; the index reaches
// that boundary and nothing else, and means nothing while inherited() is false. The
// fields are private so the pair can only be set together, through a factory that decides
// both: an entry carrying an index while claiming not to inherit, or the reverse, cannot
// be spelled. What the index means is still a claim about a boundary this entry cannot
// see, so the readers bound it against the boundary they do have.
class GraphScalarInheritance {
public:
    // The image's scalar section is allocated as an array, so a default-constructible
    // slot is required; the factories below are what a caller fills one with.
    GraphScalarInheritance() = default;

    static GraphScalarInheritance self_value() { return {0, false}; }
    static GraphScalarInheritance from_boundary(uint16_t index) { return {index, true}; }

    bool inherited() const { return inherited_ != 0; }
    uint16_t boundary_index() const { return boundary_index_; }

private:
    GraphScalarInheritance(uint16_t index, bool inherits) :
        boundary_index_(index),
        inherited_(inherits ? 1 : 0) {}

    // One access level for every field, which is what keeps this standard-layout and so
    // safe to memcpy to the device. inherited_ is a uint16_t rather than a bool so the
    // two fields fill the size alignof(uint16_t) rounds this type up to: there is no
    // padding byte, and so no indeterminate byte in the image's scalar section, which the
    // static_assert below pins the size of because that is what the section indexes by.
    uint16_t boundary_index_;
    uint16_t inherited_;
};

// Wire representation of a sub-task's dispatch predicate. The operand's absolute GM
// address is not replay-invariant, so the Definition names the tensor the
// operand element sits in plus its element offset within that tensor;
// materialize rebinds the tensor for the execution and resolves the pair into
// the address the scheduler reads at the dispatch point.
struct GraphPredicate {
    // A relocation record on the same terms as the off_tensors section: replay rebinds it
    // before resolving the address below.
    simpler::hbg::TensorData operand;
    // Element index into the rebound operand tensor, added to its start_offset.
    // Fixed at record time: a Graph with a variable simpler::hbg::Tensor shape is rejected
    // before recording, so the operand's strides cannot change across replays.
    uint64_t elem_offset;
    int64_t target;
    uint8_t elem_size;
    uint8_t op;
    uint8_t reserved[6];
};

struct SubTaskDefinition {
    int32_t kernel_id[SUBTASK_SLOT_COUNT];
    uint8_t active_mask;
    uint8_t task_attrs;
    // Early-dispatch verdicts (ED_FLAG_CANDIDATE / ED_FLAG_TRACKED), decided once
    // per recorded shape rather than per execution: a body's structure is fixed by
    // its Definition, so every execution of it qualifies the same tasks. Carries
    // the same meaning as ChipTaskSlotState::ed_flags, against the Definition's
    // fanin CSR instead of a payload fanin region.
    uint8_t ed_flags;
    uint8_t reserved;
    int16_t logical_block_num;
    int16_t total_required_subtasks;
    // One-based index into the Definition's predicate array; 0 means the task
    // carries no dispatch predicate. Biased so that a zeroed SubTaskDefinition
    // is a valid predicate-free task. Predicated tasks are rare, so the
    // predicates live in their own array rather than inline.
    uint16_t predicate_slot;
    int32_t tensor_count;
    int32_t scalar_count;
    int32_t total_output_size;
    // Element indices into the Definition's argument pools, not byte offsets: each
    // pairs with the count above it to form [offset, offset + count), so it carries
    // that count's type.
    int32_t tensor_offset;
    int32_t scalar_offset;
    ArgsDumpTaskMetadata dump_metadata;
};

inline constexpr uint64_t GRAPH_DEFINITION_OBJECT_MAGIC = 0x4752415048455844ULL;

// Every Definition object is [GraphDefinitionHeader][Definition image], aligned to
// this and padded to a multiple of it. A Definition section is reached as an offset
// from the image base, and the widest one may ask for max_align_t, so this is what
// an image base has to carry; the header in front preserves it because its own size
// is a multiple of it. Both the recorder that claims room for an object and the
// upload that packs them share the value.
inline constexpr size_t GRAPH_DEFINITION_OBJECT_ALIGN = alignof(std::max_align_t);

// Header prefixing each device-resident Definition object. The definition buffer
// uploaded by the host is [GraphDefinitionHeader][GraphDefinition image], and these
// three fields are the framing the device checks before it reads a section offset
// out of the image: the object is one of ours, it is the size the upload recorded,
// and it holds the Graph the outer task was bound to.
//
// alignas, not a pad field: the image starts immediately behind the header and its
// sections are offsets from that base, so the header's own size has to be a multiple
// of the object alignment.
struct alignas(GRAPH_DEFINITION_OBJECT_ALIGN) GraphDefinitionHeader {
    uint64_t magic;
    uint64_t full_key;
    uint32_t definition_bytes;
};

static_assert(std::is_trivially_copyable_v<GraphDefinitionHeader>);
static_assert(std::is_standard_layout_v<GraphDefinitionHeader>);
static_assert(sizeof(GraphDefinitionHeader) % GRAPH_DEFINITION_OBJECT_ALIGN == 0);

struct GraphDefinition {
    uint64_t full_key;
    uint64_t required_heap;
    // The header splits by range, not by name. Everything that counts *things* is
    // signed: each is capped by SUB_TASK_MAX_NUM times a per-task constant, so the
    // largest of them (edge_count, at 1024 x 1023) still clears INT32_MAX by three
    // orders of magnitude, and a signed count makes a corrupt wire value testable
    // with `< 0` instead of turning it into a huge index. Everything that counts
    // *bytes* stays unsigned because graph_layout_section dimensions the image
    // against UINT32_MAX: total_bytes and the off_ fields below reach 4 GB by design,
    // and halving that range would be a functional change.
    uint32_t total_bytes;
    int32_t task_count;
    int32_t edge_count;
    int32_t root_count;
    int32_t boundary_tensor_count;
    int32_t boundary_scalar_count;
    int32_t tensor_arg_count;
    int32_t scalar_arg_count;
    int32_t predicate_count;
    // Bytes the GraphExecution header, sub-task array and sub-task
    // argument pools need in the outer Graph task's heap tail. Invocation
    // boundaries live in the outer task payload's compact argument-pool regions
    // instead.
    uint32_t execution_storage_bytes;
    uint32_t off_fanout_offsets;
    uint32_t off_fanout_indices;
    uint32_t off_fanin_offsets;
    uint32_t off_fanin_indices;
    uint32_t off_root_indices;
    uint32_t off_sub_task_offsets;
    uint32_t off_sub_tasks;
    // Every sub-task's tensor arguments, concatenated; a task names its own run
    // through tensor_offset / tensor_count.
    //
    // These are **relocation records**, not tensors. Every other simpler::hbg::Tensor in
    // this runtime -- a boundary parameter, an invocation's argument, a materialized task
    // argument -- holds values that mean something on their own. One here does not: the
    // two fields replay has a base for are stored relative, and which base to add follows
    // the tensor's own owner_task_id, stamped by the recording.
    //
    //   field           owner PARAM                           owner SUB_TASK
    //   buffer_addr     0 -- replay takes the whole buffer     offset into the graph heap,
    //                   from this call's argument              whose base the execution has
    //   start_offset    the view's offset inside that          the view's own origin,
    //                   parameter; replay adds the             absolute in its own buffer
    //                   argument's origin
    //
    // Geometry, dtype and flags travel absolute, because graph_boundary_matches pins those
    // equal before a Definition may be reused.
    //
    // So start_offset here does not mean what it means on the boundary tensors this is
    // matched against: there it is the caller's own origin, here an offset from it. A
    // parameter's offset is relative to its own parameter rather than to its alias
    // partition's, so a tensor is rebuilt against the argument it came from; two tensors
    // over one buffer keep their recorded distance only while the arguments keep theirs,
    // which is what graph_boundary_matches checks and what the recorded WAR/WAW edges were
    // inferred from.
    uint32_t off_tensors;
    uint32_t off_scalars;
    uint32_t off_scalar_inheritance;
    uint32_t off_predicates;
};

static_assert(std::is_trivially_copyable_v<GraphScalarInheritance>);
static_assert(std::is_standard_layout_v<GraphScalarInheritance>);
static_assert(sizeof(GraphScalarInheritance) == 4, "the image's scalar section assumes this layout");
static_assert(std::is_trivially_copyable_v<SubTaskDefinition>);
static_assert(std::is_standard_layout_v<SubTaskDefinition>);
// graph_fill_definition assigns this struct field by field, so its interior padding
// is the one part of the sub-task section no writer reaches. Nothing reads it
// either, but pinning the size makes a new field's padding cost visible in the diff
// that adds it rather than silently.
static_assert(sizeof(SubTaskDefinition) == 80, "a SubTaskDefinition field changed the wire layout");
static_assert(std::is_trivially_copyable_v<GraphPredicate>);
static_assert(std::is_standard_layout_v<GraphPredicate>);
static_assert(std::is_trivially_copyable_v<GraphDefinition>);
static_assert(std::is_standard_layout_v<GraphDefinition>);

// Section starts are offsets from the image base, so a section is correctly
// aligned only if the buffer holding the image is. The builder writes each
// section through a typed pointer into a std::vector<std::byte>, whose data() is
// aligned for any type with fundamental alignment and no further — so a section
// type that asked for more would make every one of those stores undefined, with
// no diagnostic.
static_assert(
    alignof(SubTaskDefinition) <= alignof(std::max_align_t) &&
        alignof(simpler::hbg::TensorData) <= alignof(std::max_align_t) &&
        alignof(GraphScalarInheritance) <= alignof(std::max_align_t) &&
        alignof(GraphPredicate) <= alignof(std::max_align_t),
    "a Definition section type must not be over-aligned: its storage is a byte vector"
);

// A section's length is the same int32 every counting field of the header carries.
// A negative one is rejected outright rather than left to wrap through the size_t
// comparison, so a corrupt header cannot widen a section by either route.
template <typename T>
inline const T *graph_definition_array(const GraphDefinition &definition, uint32_t offset, int32_t count) {
    if (count < 0 || offset == 0 || offset > definition.total_bytes || offset % alignof(T) != 0) return nullptr;
    const size_t remaining = static_cast<size_t>(definition.total_bytes - offset);
    if (static_cast<size_t>(count) > remaining / sizeof(T)) return nullptr;
    return reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(&definition) + offset);
}

template <typename T>
inline const T *graph_definition_ptr(const GraphDefinition &definition, uint32_t offset) {
    return graph_definition_array<T>(definition, offset, 1);
}

enum class GraphExecutionState : uint8_t {
    SUBMITTED = 0,
    MATERIALIZING = 1,
    PREPARED = 2,
    ACTIVE = 3,
    COMPLETED = 4,
};

enum class GraphMaterializeResult : uint8_t {
    INVALID = 0,
    BUSY = 1,
    PENDING = 2,
    PREPARED = 3,
};

inline constexpr uint8_t GRAPH_EXECUTION_STATE_MASK = 0x7;
inline constexpr uint8_t GRAPH_EXECUTION_EXTERNAL_READY = 0x8;

struct GraphExecution {
    // The low bits hold GraphExecutionState. EXTERNAL_READY shares this byte so
    // dependency readiness can arrive before materialization without a separate
    // per-submission gate object.
    std::atomic<uint8_t> state{static_cast<uint8_t>(GraphExecutionState::SUBMITTED)};
    std::atomic<uint8_t> materialize_busy{0};
    std::atomic<int32_t> remaining_tasks{0};
    std::atomic<int32_t> retired_tasks{0};
    // Incremental activation: tasks in [0, published_tasks) are fully
    // materialized and registered, so a route pass may consider them. route_cursor
    // is the next such task index a route pass will claim; roots below it have
    // been pushed to the ready queue exactly once. Both advance monotonically and
    // reset per (re)submission.
    std::atomic<int32_t> published_tasks{0};
    std::atomic<int32_t> route_cursor{0};
    int32_t task_count{0};
    int32_t materialized_tasks{0};
    int32_t constructed_tasks{0};
    int32_t consumed_tensor_args{0};
    ChipTaskSlotState *outer_slot{nullptr};
    ChipTaskStorage *tasks{nullptr};
    ChipTaskStorage *task_storage{nullptr};
    // Polling-progress state, one ChipTaskState byte per sub-task, in the
    // storage tail. Carries the same PENDING -> PUBLISHED -> COMPLETED meaning
    // as the shared-memory task_states array a GLOBAL task uses, against
    // sub-task local ids instead of task-table slots, so both cohorts answer
    // readiness the same way.
    //
    // A byte array of its own rather than a field of ChipTaskStorage, for the
    // reason the top-level array gives: a fanin scan reads many producers'
    // states at once, which a couple of cache lines answer here and would take
    // one line per producer inside the ChipTaskStorage stride. It is also
    // read-mostly — each byte is written once per execution at completion —
    // so concurrent scanners share those lines rather than contend for them.
    std::atomic<ChipTaskState> *task_states{nullptr};
    // This execution's task argument pools, in the storage tail past task_storage.
    // Every task payload's tensor and scalar deltas point here; its pool position is
    // the Definition's tensor_offset / scalar_offset.
    simpler::hbg::Tensor *task_tensor_pool{nullptr};
    uint64_t *task_scalar_pool{nullptr};
    const GraphDefinition *definition{nullptr};
    const int32_t *fanin_offsets{nullptr};
    const uint16_t *fanin_indices{nullptr};
    // Base of the graph heap this execution was given. A body tensor's recorded offset is
    // relative to the recording's own output region, which is an affine image of this heap,
    // so the two added together are the real address.
    uintptr_t heap_base{0};
    // This invocation's actual arguments, held in the outer Graph task's own argument
    // pool. They are simpler::hbg::Tensors like every other task's, so a rebind that
    // resolves to one copies it across rather than converting: the boundary is the one
    // input that does not come from the image.
    const simpler::hbg::Tensor *boundary_tensors{nullptr};
    int32_t boundary_tensor_count{0};
    const uint64_t *boundary_scalars{nullptr};
    int32_t boundary_scalar_count{0};

    ChipTaskStorage &task_at(int32_t index) const { return task_storage[index]; }

    // Readiness accessors, named and ordered as SharedMemoryTaskHeader's so a
    // reader of one cohort reads the other the same way. The byte only ever
    // advances, so an index a scan has already cleared stays cleared.
    bool is_completed(int32_t index, std::memory_order order = std::memory_order_acquire) const {
        return task_states[index].load(order) >= CHIP_TASK_COMPLETED;
    }

    void store_completed(int32_t index, std::memory_order order = std::memory_order_release) const {
        task_states[index].store(CHIP_TASK_COMPLETED, order);
    }

    bool is_published(int32_t index, std::memory_order order = std::memory_order_acquire) const {
        return task_states[index].load(order) >= CHIP_TASK_PUBLISHED;
    }

    void store_published(int32_t index, std::memory_order order = std::memory_order_release) const {
        task_states[index].store(CHIP_TASK_PUBLISHED, order);
    }

    void reset_task_state(int32_t index) const {
        task_states[index].store(CHIP_TASK_PENDING, std::memory_order_relaxed);
    }
};

static_assert(std::is_trivially_destructible_v<ChipTaskStorage>);
// The tensor pool starts right after the sub-task array, and the scalar pool starts
// after a whole number of ChipTensors.
static_assert(
    alignof(ChipTaskStorage) % alignof(simpler::hbg::Tensor) == 0,
    "a sub-task entry must be at least simpler::hbg::Tensor-aligned: the tensor pool follows the "
    "sub-task array"
);
static_assert(
    sizeof(simpler::hbg::Tensor) % alignof(uint64_t) == 0, "the tensor stride must keep the scalar pool aligned"
);
static_assert(std::is_trivially_destructible_v<GraphExecution>);
// The whole storage is aligned for its widest member, so one base check covers the
// header as well as the sub-task array that follows it.
static_assert(
    alignof(ChipTaskStorage) % alignof(GraphExecution) == 0,
    "the sub-task array's alignment must subsume the execution header's"
);

// The outer GRAPH task's heap tail occupies
// [GraphExecution][ChipTaskStorage x task_count][simpler::hbg::Tensor x tensor_arg_count]
// [uint64_t x scalar_arg_count].
//
// The last two regions are the sub-task payloads' argument pools, indexed by the
// Definition's own tensor_offset / scalar_offset — which is why the Definition's
// arg-table counts size them rather than a per-task sum: sub-task i's arguments
// occupy [offset, offset + count) in both the table and the pool. There is no fanin
// region: a sub-task's dependencies live in the Definition's fanin CSR, so its
// fanin_count stays 0 and its fanin delta unbound.
struct GraphExecutionStorageLayout {
    size_t tasks_offset;
    size_t tensors_offset;
    size_t scalars_offset;
    size_t states_offset;
    size_t total_bytes;
};

inline bool graph_execution_storage_layout(
    int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count, GraphExecutionStorageLayout *out
) {
    if (out == nullptr || task_count <= 0 || task_count > SUB_TASK_MAX_NUM || tensor_arg_count < 0 ||
        scalar_arg_count < 0) {
        return false;
    }
    constexpr size_t ALIGNMENT = alignof(ChipTaskStorage);
    out->tasks_offset = (sizeof(GraphExecution) + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    out->tensors_offset = out->tasks_offset + static_cast<size_t>(task_count) * sizeof(ChipTaskStorage);
    out->scalars_offset = out->tensors_offset + static_cast<size_t>(tensor_arg_count) * sizeof(simpler::hbg::Tensor);
    // The state array is last because it is the one region with no alignment of
    // its own: a byte needs none, so appending it disturbs no other section's
    // offset. Every other region is entered through a typed pointer whose
    // alignment the base already guarantees.
    out->states_offset = out->scalars_offset + static_cast<size_t>(scalar_arg_count) * sizeof(uint64_t);
    out->total_bytes = out->states_offset + static_cast<size_t>(task_count) * sizeof(std::atomic<ChipTaskState>);
    return true;
}

inline bool graph_execution_storage_bytes(
    int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count, size_t *storage_bytes
) {
    GraphExecutionStorageLayout layout{};
    if (storage_bytes == nullptr ||
        !graph_execution_storage_layout(task_count, tensor_arg_count, scalar_arg_count, &layout)) {
        return false;
    }
    *storage_bytes = layout.total_bytes;
    return true;
}

GraphExecution *graph_execution_localize(ChipTaskSlotState &outer_slot);
GraphMaterializeResult graph_execution_materialize_slice(
    ChipTaskSlotState &outer_slot, GraphExecution &execution, int32_t max_tasks, int32_t *tasks_materialized = nullptr
);

// An outer GRAPH slot's graph_context holds the shared Definition's device address
// until graph_execution_localize replaces it with the execution, so this cast is only
// valid after that call. What makes it safe is the boot sequence, not this slot: every
// AICPU thread localizes a disjoint slice of the task window in classify_partition and
// all of them barrier before runtime_init_ready_ is published, so no dispatch — and
// therefore no caller of this function — observes an unlocalized GRAPH slot. A slot
// whose localization failed carries nullptr.
inline GraphExecution *graph_execution_from_outer_slot(ChipTaskSlotState &slot) {
    return slot.task_kind == TaskKind::GRAPH ? static_cast<GraphExecution *>(slot.graph_context) : nullptr;
}

inline GraphExecutionState
graph_execution_state(const GraphExecution &execution, std::memory_order order = std::memory_order_acquire) {
    return static_cast<GraphExecutionState>(execution.state.load(order) & GRAPH_EXECUTION_STATE_MASK);
}

inline bool
graph_execution_external_ready(const GraphExecution &execution, std::memory_order order = std::memory_order_acquire) {
    return (execution.state.load(order) & GRAPH_EXECUTION_EXTERNAL_READY) != 0;
}

inline void graph_execution_set_state(
    GraphExecution &execution, GraphExecutionState next, std::memory_order order = std::memory_order_release
) {
    uint8_t observed = execution.state.load(std::memory_order_relaxed);
    // Masked, so a state added past GRAPH_EXECUTION_STATE_MASK cannot reach the
    // readiness bit sharing this byte.
    const uint8_t next_state = static_cast<uint8_t>(static_cast<uint8_t>(next) & GRAPH_EXECUTION_STATE_MASK);
    while (!execution.state.compare_exchange_weak(
        observed, static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_STATE_MASK) | next_state), order,
        std::memory_order_relaxed
    )) {}
}

inline bool graph_execution_transition(
    GraphExecution &execution, GraphExecutionState expected_state, GraphExecutionState next_state
) {
    uint8_t observed = execution.state.load(std::memory_order_acquire);
    const uint8_t desired_state = static_cast<uint8_t>(static_cast<uint8_t>(next_state) & GRAPH_EXECUTION_STATE_MASK);
    while ((observed & GRAPH_EXECUTION_STATE_MASK) == static_cast<uint8_t>(expected_state)) {
        const uint8_t desired = static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_STATE_MASK) | desired_state);
        if (execution.state.compare_exchange_weak(
                observed, desired, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return true;
        }
    }
    return false;
}

inline bool graph_execution_signal_external_ready(GraphExecution &execution) {
    return (execution.state.fetch_or(GRAPH_EXECUTION_EXTERNAL_READY, std::memory_order_acq_rel) &
            GRAPH_EXECUTION_EXTERNAL_READY) == 0;
}

inline bool graph_execution_complete_sub_task(GraphExecution &execution) {
    return execution.remaining_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

inline void graph_execution_mark_completed(GraphExecution &execution) {
    graph_execution_set_state(execution, GraphExecutionState::COMPLETED);
}

inline void graph_execution_retire_sub_task(GraphExecution &execution) {
    execution.retired_tasks.fetch_add(1, std::memory_order_release);
}
