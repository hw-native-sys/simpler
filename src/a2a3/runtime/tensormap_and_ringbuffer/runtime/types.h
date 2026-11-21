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
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - TaskOutputTensors: Return value from submit containing materialized output ChipTensors
 * - Arg: Aggregated argument container for rt_submit_task API
 *
 * simpler::tmr::Tensor descriptor types (simpler::tmr::Tensor, PTOBufferHandle, TensorCreateInfo) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorLease, HostApi).
 */

#pragma once

#include "scheduler/twait_contract.h"

#include <stdint.h>
#include <string.h>

#include <string>
#include <type_traits>
#include <utility>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "aicpu/dump_arg_selection.h"
#include "common/device_phase.h"
#include "data_type.h"
#include "profiling_config.h"
#include "submit_types.h"
#include "task_args.h"
#include "tensor.h"
#include "tensor_create_info.h"                   // runtime-only TensorCreateInfo + materialization helpers
#include "tensormap_and_ringbuffer/entry_args.h"  // EntryArgsStorage

// TaskAttrs packs the timing tag into a 4-bit field and reports "untagged" as
// -1, so the tag domain must fit 0..15 and the untagged sentinel must be -1.
static_assert(NUM_TASK_TIMING_SLOTS <= 16, "timing tag must fit TaskAttrs' 4-bit field");
static_assert(TASK_TIMING_SLOT_NONE == -1, "TaskAttrs::timing_slot() reports untagged as -1");

typedef enum {
    ASYNC_ENGINE_SDMA = 0,
    ASYNC_ENGINE_ROCE = 1,
    ASYNC_ENGINE_URMA = 2,
    ASYNC_ENGINE_CCU = 3,
    NUM_ASYNC_ENGINES = 4,
} AsyncEngine;

enum class CompletionType : int32_t {
    COUNTER = 0,
};

// =============================================================================
// Task Output ChipTensors (return value from submit)
// =============================================================================

enum class ScopeMode : uint8_t {
    AUTO = 0,
    MANUAL = 1,
};

/**
 * Orthogonal dependency-edge semantics, stored per fanin edge.
 *
 *   DEP_WAIT   — readiness/ordering: the consumer cannot become ready until the
 *                producer completes. Contributes to the consumer's fanin count
 *                and to the producer's fanout notification list.
 *   DEP_RETAIN — lifetime: the producer's slot and packed output buffer must
 *                stay alive until the consumer releases it. Holds a
 *                fanout_count reference from submit until on_task_release.
 *
 * The two are independent; an edge may carry either, both, or (transiently
 * during construction) neither. A creator edge is DEP_WAIT|DEP_RETAIN (the
 * consumer reads the producer's allocated buffer). A tensormap modifier edge is
 * DEP_WAIT only (the buffer was allocated elsewhere, so only ordering is
 * needed). When one (producer, consumer) pair is discovered for several
 * reasons, the flags are OR-accumulated.
 */
enum DepFlags : uint8_t {
    DEP_NONE = 0,
    DEP_WAIT = 1u << 0,
    DEP_RETAIN = 1u << 1,
};

constexpr DepFlags operator|(DepFlags a, DepFlags b) {
    return static_cast<DepFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
constexpr DepFlags operator&(DepFlags a, DepFlags b) {
    return static_cast<DepFlags>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}
constexpr DepFlags &operator|=(DepFlags &a, DepFlags b) {
    a = a | b;
    return a;
}
constexpr bool dep_has_wait(DepFlags f) { return (f & DEP_WAIT) != DEP_NONE; }
constexpr bool dep_has_retain(DepFlags f) { return (f & DEP_RETAIN) != DEP_NONE; }

/**
 * TaskOutputTensors — returned by submit, holds materialized output ChipTensors.
 *
 * Only runtime-created outputs are stored here, indexed in add_output order.
 *
 * The underlying storage is uninitialized; only output_count elements are
 * valid after submit returns.  This avoids default-constructing simpler::tmr::Tensor[]
 * on the hot path (2 KB of unnecessary zeroing per submit).
 *
 * Users must hold a named TaskOutputTensors variable and borrow via get_ref();
 * binding get_ref() on an rvalue is compile-time rejected to prevent dangling.
 *
 * LIFETIME — single-scope only:
 *   Internally this class stores pointers into the submitting task's payload
 *   (TaskPayload::tensors[]), which lives in a ring-buffer slot. After
 *   scope_end the slot becomes eligible for reuse, and a later submit will
 *   overwrite the same simpler::tmr::Tensor storage in place. Therefore the
 *   TaskOutputTensors instance, the const simpler::tmr::Tensor& returned by get_ref(), and
 *   any pointer derived from either MUST NOT outlive the SIMPLER_SCOPE in which
 *   submit was called — do not move/copy them to outer-scope variables, do
 *   not capture references by std::reference_wrapper or raw pointers across
 *   scope boundaries.
 *
 *   This invariant is intentionally not enforced at runtime: a reused slot
 *   simply carries a different but valid owner_task_id, so checking
 *   owner_task_id cannot distinguish "still mine" from "silently aliased to
 *   an unrelated task". Misuse manifests as a wrong-tensor read with no
 *   diagnostic.
 */
class TaskOutputTensors {
public:
    TaskOutputTensors() :
        task_id_(TaskId::invalid()),
        output_count_(0) {}

    bool empty() const { return output_count_ == 0; }
    uint32_t size() const { return output_count_; }

    /// Borrow a materialized output tensor by index (lvalue only).
    const simpler::tmr::Tensor &get_ref(uint32_t index) const & {
        always_assert(index < output_count_);
        return *tensors_[index];
    }
    const simpler::tmr::Tensor &get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: append one materialized output simpler::tmr::Tensor.
    void materialize_output(const simpler::tmr::Tensor &tensor) {
        always_assert(output_count_ < MAX_TENSOR_ARGS);
        tensors_[output_count_++] = &tensor;
    }

    void set_task_id(TaskId id) { task_id_ = id; }

    TaskId task_id() const { return task_id_; }

private:
    TaskId task_id_;
    uint32_t output_count_;
    // Upper bound: a task cannot have more outputs than total tensor args
    // (every OUTPUT/OUTPUT_EXISTING slot is one of the Arg's tensor slots).
    const simpler::tmr::Tensor *tensors_[MAX_TENSOR_ARGS];
};

// =============================================================================
// Argument Types (for rt_submit_task API)
// =============================================================================

// TensorArgType is defined in tensor.h (included via task_args.h above)

/**
 * Tagged reference to a single Arg slot — either a simpler::tmr::Tensor* or a
 * TensorCreateInfo*. The active member is determined by the slot's
 * TensorArgType tag (OUTPUT → create_info, else → tensor pointer).
 *
 * Minimal-permission: the union members are private; content is set only via
 * operator=(ptr) and read via ref()/create_info(). Copy/move are deleted — a
 * TensorRef is written in place inside an Arg's slot array, never passed by
 * value.
 */
class TensorRef {
    union {
        const simpler::tmr::Tensor *ptr_;
        const TensorCreateInfo *create_info_;
    };

public:
    TensorRef() :
        ptr_(nullptr) {}
    TensorRef(const TensorRef &) = delete;
    TensorRef(TensorRef &&) = delete;
    TensorRef &operator=(const TensorRef &) = delete;
    TensorRef &operator=(TensorRef &&) = delete;

    TensorRef &operator=(const simpler::tmr::Tensor *p) {
        ptr_ = p;
        return *this;
    }
    TensorRef &operator=(const TensorCreateInfo *ci) {
        create_info_ = ci;
        return *this;
    }

    const simpler::tmr::Tensor &ref() const { return *ptr_; }
    const TensorCreateInfo &create_info() const { return *create_info_; }
    bool refers_to(const simpler::tmr::Tensor *t) const { return ptr_ == t; }
    bool refers_to(const TensorCreateInfo *ci) const { return create_info_ == ci; }
};

/**
 * Aggregated argument container for rt_submit_task
 *
 * Inherits storage from TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType>.
 * Each tensor slot stores a TensorRef union (simpler::tmr::Tensor* or TensorCreateInfo)
 * discriminated by the corresponding tag().
 * ChipTensors are dispatched first in kernel args, followed by scalars.
 *
 * Output arguments follow two distinct ownership models:
 * - add_output(const TensorCreateInfo&): OUTPUT — runtime allocates buffer
 *   and materializes a new simpler::tmr::Tensor, returned via TaskOutputTensors.
 * - add_inout(const simpler::tmr::Tensor&): INOUT — reuses an existing simpler::tmr::Tensor as the write target.
 *
 * Example:
 *   simpler::tmr::Tensor x = simpler::tmr::make_tensor_external(dev_a, shapes, 2);
 *   TensorCreateInfo ci(shapes, 2);  // must outlive submit
 *   Arg args;
 *   args.add_input(x);
 *   args.add_output(ci);
 *   args.add_scalar(some_value);
 *   TaskOutputTensors outs = rt_submit_aic_task(kernel_id, args);
 *   const simpler::tmr::Tensor& y = outs.get_ref(0);
 */

// Operand of a dispatch predicate (L0 layer): locates one element of a tensor —
// tensor + ndims + indices, mirroring get_tensor_data. The tensor is borrowed and
// must outlive submit; its buffer must be allocated by then, and its producer must
// be a dependency of the predicated task so the value is current at dispatch.
struct CorePredicateOperand {
    const simpler::tmr::Tensor *tensor{nullptr};
    uint32_t ndims{0};
    uint32_t indices[MAX_TENSOR_DIMS]{};
};

// Dispatch predicate carried on an Arg: operand OP target (e.g. count[i] > 0).
// op == NONE means "no predicate — always dispatch". Submit resolves the operand
// into the payload's DispatchPredicate (an absolute GM address). Read in-process;
// never crosses the wire.
struct CoreTaskPredicate {
    CorePredicateOperand operand;
    PredicateOp op{PredicateOp::NONE};
    int64_t target{0};
};

template <size_t MaxT, size_t MaxS>
struct Arg : TaskArgsTpl<TensorRef, uint64_t, MaxT, MaxS, TensorArgType> {
    using Base = TaskArgsTpl<TensorRef, uint64_t, MaxT, MaxS, TensorArgType>;
    // Make dependent-base members visible for unqualified use (two-phase lookup
    // does not search a dependent base in a class template).
    using Base::scalar_count_;
    using Base::scalars_;
    using Base::tags_;
    using Base::tensor_count_;
    using Base::tensors_;

    // Minimal-permission: an Arg is built in place and consumed by reference;
    // it is never copied/moved (it is a large object, and its TensorRef slots
    // are non-copyable by design).
    Arg() = default;
    Arg(const Arg &) = delete;
    Arg(Arg &&) = delete;
    Arg &operator=(const Arg &) = delete;
    Arg &operator=(Arg &&) = delete;

    bool has_error{false};
    bool blocking_wait_{false};
    bool capture_wait_proof_{false};
    void capture_wait_proof(bool enabled = true) { capture_wait_proof_ = enabled; }
    bool captures_wait_proof() const { return capture_wait_proof_; }
    twait::Declaration blocking_declaration_{};
    void set_blocking_wait(const twait::Declaration &declaration) {
        blocking_wait_ = true;
        blocking_declaration_ = declaration;
    }
    bool has_blocking_wait() const { return blocking_wait_; }
    const twait::Declaration &blocking_wait() const { return blocking_declaration_; }
    const char *error_msg{nullptr};
    LaunchSpec launch_spec;  // SPMD launch parameters (block_num, etc.)

    // Early-dispatch hint (codegen-author set, off by default). When
    // true, the scheduler may stage this task on an idle core before its producer
    // finishes, gating execution on the DATA_MAIN_BASE doorbell — only safe when
    // the author knows the task's data dependencies allow it. Read in-process by
    // the runtime; never crosses the wire format.
    bool allow_early_resolve_{false};
    void set_allow_early_resolve(bool v = true) { allow_early_resolve_ = v; }
    bool allow_early_resolve() const { return allow_early_resolve_; }

    // Dispatch predicate (codegen-author set; default op == NONE = always
    // dispatch). A FALSE result at the dispatch point retires the task inline
    // through the dep-only path — never dispatched to an AICore — while still
    // resolving fanin/fanout so consumers unlock. The predicate tensor's producer
    // MUST be a dependency of this task so the value is current when the task
    // becomes ready. Read in-process; never crosses the wire.
    CoreTaskPredicate predicate_;
    void set_predicate(const CoreTaskPredicate &pred) { predicate_ = pred; }
    const CoreTaskPredicate &predicate() const { return predicate_; }

    // Selective task-timing slot: tag this task to have the scheduler record its
    // AICPU dispatch/finish cycles into fixed slot `slot` (0..15). Untagged by
    // default. An out-of-range id fails through the standard invalid-arg path so
    // the scheduler never stamps out of bounds.
    int32_t task_timing_slot_{TASK_TIMING_SLOT_NONE};
    void set_task_timing_slot(int32_t slot) {
        if (slot < 0 || slot >= NUM_TASK_TIMING_SLOTS) {
            set_error("task_timing_slot out of range (valid: 0..15)");
            return;
        }
        task_timing_slot_ = slot;
    }
    int32_t task_timing_slot() const { return task_timing_slot_; }

    void clear() {
        Base::clear();
#if SIMPLER_DFX
        dump_arg_selection_.clear();
#endif
        explicit_deps_ = nullptr;
        explicit_dep_count_ = 0;
        explicit_dep_kinds_ = nullptr;
        blocking_wait_ = false;
        capture_wait_proof_ = false;
        blocking_declaration_ = twait::Declaration{};
        allow_early_resolve_ = false;
        predicate_ = CoreTaskPredicate{};
        task_timing_slot_ = TASK_TIMING_SLOT_NONE;
    }

    void reset() {
        clear();
        has_error = false;
        error_msg = nullptr;
    }

    void set_error(const char *msg) {
        if (!has_error) {
            has_error = true;
            error_msg = msg;
        }
    }

    template <typename... Args>
    void dump(Args &&...args) {
#if SIMPLER_DFX
        static_assert(
            (std::is_lvalue_reference_v<Args> && ...),
            "dump: temporaries are not allowed — pass tensors/scalars already added to this Arg"
        );
        static_assert(
            (is_supported_dump_arg_v<Args> && ...),
            "dump: all arguments must be simpler::tmr::Tensor, TensorCreateInfo, or scalar lvalues"
        );
        if constexpr (sizeof...(Args) == 0) {
            mark_all_dump_args();
        } else {
            (mark_dump_arg(args), ...);
        }
#else
        ((void)args, ...);
#endif
    }

#if SIMPLER_DFX
    uint64_t dump_arg_mask() const { return dump_arg_selection_.dump_arg_mask(); }
    uint64_t dump_arg_index_ambiguous_mask() const { return dump_arg_selection_.dump_arg_index_ambiguous_mask(); }
#else
    uint64_t dump_arg_mask() const { return 0; }
    uint64_t dump_arg_index_ambiguous_mask() const { return 0; }
#endif

    template <typename... Args>
    void add_input(Args &&...args) {
        assert_add_tensor_args<false, Args...>();
        if (!check_add_tensor_capacity(static_cast<int32_t>(sizeof...(Args)))) {
            return;
        }
        ((tensors_[tensor_count_] = &args, tags_[tensor_count_] = TensorArgType::INPUT, tensor_count_++), ...);
    }

    /// Batch add outputs — all simpler::tmr::Tensor or all TensorCreateInfo:
    ///   add_output(ci1, ci2)         — runtime allocates buffers (OUTPUT)
    ///   add_output(t1, t2)           — write-only existing tensors (OUTPUT_EXISTING)
    template <typename... Args>
    void add_output(Args &&...args) {
        assert_add_tensor_args<true, Args...>();
        if (!check_add_tensor_capacity(static_cast<int32_t>(sizeof...(Args)))) return;
        if constexpr ((std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...)) {
            ((tensors_[tensor_count_] = &args, tags_[tensor_count_] = TensorArgType::OUTPUT, tensor_count_++), ...);
        } else {
            ((tensors_[tensor_count_] = &args, tags_[tensor_count_] = TensorArgType::OUTPUT_EXISTING, tensor_count_++),
             ...);
        }
    }

    template <typename... Args>
    void add_inout(Args &&...args) {
        assert_add_tensor_args<false, Args...>();
        if (!check_add_tensor_capacity(static_cast<int32_t>(sizeof...(Args)))) {
            return;
        }
        ((tensors_[tensor_count_] = &args, tags_[tensor_count_] = TensorArgType::INOUT, tensor_count_++), ...);
    }

    /// No-dependency existing tensor: skips OverlapMap lookup, depends on creator only.
    template <typename... Args>
    void add_no_dep(Args &&...args) {
        assert_add_tensor_args<false, Args...>();
        if (!check_add_tensor_capacity(static_cast<int32_t>(sizeof...(Args)))) return;
        ((tensors_[tensor_count_] = &args, tags_[tensor_count_] = TensorArgType::NO_DEP, tensor_count_++), ...);
    }

    /**
     * Attach an explicit dependency array. The Arg stores (ptr, count) without
     * copying — the caller's array must outlive the submit (same lifetime rule
     * as add_input/add_output, which also store pointers).
     *
     * count == 0 is a valid "set empty" — it clears any previously stored deps
     * and returns. This lets callers that build the dep set conditionally pass
     * the result through unguarded, including in the no-dep branch:
     *   TaskId deps[3];
     *   uint32_t n = 0;
     *   if (have_prev) deps[n++] = prev;
     *   if (is_last)   deps[n++] = alloc;
     *   args.set_dependencies(deps, n);    // safe even if n == 0
     *
     * For count > 0, the call is single-shot: a second non-empty call after
     * deps are already set will fail with set_error(). Use count == 0 first
     * if you need to re-set.
     */
    void set_dependencies(const TaskId *deps, uint32_t count) {
        if (count == 0) {
            explicit_deps_ = nullptr;
            explicit_dep_count_ = 0;
            explicit_dep_kinds_ = nullptr;
            return;
        }
        if (deps == nullptr) {
            set_error("set_dependencies: deps must not be null when count > 0");
            return;
        }
        if (explicit_deps_ != nullptr) {
            set_error("set_dependencies: may be called at most once per Arg");
            return;
        }
        explicit_deps_ = deps;
        explicit_dep_count_ = count;
        explicit_dep_kinds_ = nullptr;
    }

    /**
     * Same as set_dependencies() but attaches a per-dep DepFlags array so the
     * caller can mark an explicit dep as ordering-only (DEP_WAIT) rather than the
     * conservative DEP_WAIT|DEP_RETAIN default. The kinds array is caller-owned
     * and must outlive submit (same lifetime rule as deps). A null kinds array
     * with count > 0 is rejected; use set_dependencies() for the
     * conservative DEP_WAIT|DEP_RETAIN default.
     */
    void set_dependencies_with_kinds(const TaskId *deps, const DepFlags *kinds, uint32_t count) {
        if (count == 0) {
            explicit_deps_ = nullptr;
            explicit_dep_count_ = 0;
            explicit_dep_kinds_ = nullptr;
            return;
        }
        if (deps == nullptr || kinds == nullptr) {
            set_error("set_dependencies_with_kinds: deps and kinds must not be null when count > 0");
            return;
        }
        if (explicit_deps_ != nullptr) {
            set_error("set_dependencies_with_kinds: may be called at most once per Arg");
            return;
        }
        explicit_deps_ = deps;
        explicit_dep_kinds_ = kinds;
        explicit_dep_count_ = count;
    }

    uint32_t explicit_dep_count() const { return explicit_dep_count_; }

    TaskId explicit_dep(uint32_t index) const {
        always_assert(index < explicit_dep_count_);
        return explicit_deps_[index];
    }

    /**
     * Flags of the i-th explicit dep. Returns DEP_WAIT|DEP_RETAIN when no kinds
     * array is attached (plain set_dependencies), so an explicit dep protecting a
     * runtime-created output is never silently weakened to ordering-only.
     */
    DepFlags explicit_dep_kind(uint32_t index) const {
        always_assert(index < explicit_dep_count_);
        return explicit_dep_kinds_ ? explicit_dep_kinds_[index] : (DEP_WAIT | DEP_RETAIN);
    }

    const TaskId *explicit_deps_data() const { return explicit_deps_; }
    const DepFlags *explicit_dep_kinds_data() const { return explicit_dep_kinds_; }

    /**
     * Add scalar values. Types are deduced per argument; each value is
     * bit-cast to uint64_t for storage. Mixed types are allowed:
     *
     *   args.add_scalar(uint64_val);                  // single
     *   args.add_scalar(3.14f, int32_t(42), 7u);     // mixed batch
     */
    template <typename... Args>
    void add_scalar(Args &&...args) {
        static_assert(sizeof...(Args) >= 1, "add_scalar: at least one argument required");
        static_assert((is_supported_scalar_arg_v<Args> && ...), "add_scalar: all types must be arithmetic or enum");
        if (scalar_count_ + sizeof...(Args) > MaxS) {
            set_error(scalar_cap_msg());
            return;
        }
        (add_scalar_one(std::forward<Args>(args)), ...);
    }

    void add_scalars(const uint64_t *values, int count) {
        if (count < 0 || scalar_count_ + count > MaxS) {
            set_error(scalar_cap_msg());
            return;
        }
        memcpy(&scalars_[scalar_count_], values, count * sizeof(uint64_t));
#if SIMPLER_DFX
        dump_arg_selection_.clear_scalar_metadata(scalar_count_, count);
#endif
        scalar_count_ += count;
    }

    /**
     * Zero-extend int32 bit patterns into uint64 scalar slots.
     * Negative values are treated as their unsigned 32-bit representation
     * (e.g., -1 → 0x00000000FFFFFFFF, not 0xFFFFFFFFFFFFFFFF).
     * Uses NEON to process 4 elements per iteration on aarch64.
     */
    void add_scalars_i32(const int32_t *values, int count) {
        if (count < 0 || scalar_count_ + count > MaxS) {
            set_error(scalar_cap_msg());
            return;
        }
        uint64_t *dst = &scalars_[scalar_count_];
#if defined(__aarch64__)
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            uint32x4_t v = vld1q_u32(reinterpret_cast<const uint32_t *>(values + i));
            uint64x2_t lo = vmovl_u32(vget_low_u32(v));
            uint64x2_t hi = vmovl_u32(vget_high_u32(v));
            vst1q_u64(dst + i, lo);
            vst1q_u64(dst + i + 2, hi);
        }
        for (; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
#else
        for (int i = 0; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
#endif
#if SIMPLER_DFX
        dump_arg_selection_.clear_scalar_metadata(scalar_count_, count);
#endif
        scalar_count_ += count;
    }

    /**
     * Copy scalars from another Arg's scalar array.
     * Useful when multiple tasks share the same scalar data (e.g., block indices).
     */
    void copy_scalars_from(const Arg &src, int src_offset, int count) {
        if (src_offset < 0 || count < 0 || src_offset + count > src.scalar_count_) {
            set_error("Source scalar range out of bounds in copy_scalars_from");
            return;
        }
        if (scalar_count_ + count > MaxS) {
            set_error(scalar_cap_msg());
            return;
        }
        memcpy(&scalars_[scalar_count_], &src.scalars_[src_offset], count * sizeof(uint64_t));
#if SIMPLER_DFX
        dump_arg_selection_.copy_scalar_dtypes_from(src.dump_arg_selection_, scalar_count_, src_offset, count);
#endif
        scalar_count_ += count;
    }

#if SIMPLER_DFX
    const uint8_t *scalar_dtypes() const { return dump_arg_selection_.scalar_dtypes(); }
#else
    const uint8_t *scalar_dtypes() const { return nullptr; }
#endif

private:
    // Caller-owned dependency array; lifetime must extend through submit.
#if SIMPLER_DFX
    DumpArgSelection dump_arg_selection_;
#endif
    const TaskId *explicit_deps_{nullptr};
    const DepFlags *explicit_dep_kinds_{nullptr};
    uint32_t explicit_dep_count_{0};
#if SIMPLER_DFX
    template <typename T>
    static constexpr bool is_supported_dump_arg_v =
        std::is_same_v<std::decay_t<T>, simpler::tmr::Tensor> || std::is_same_v<std::decay_t<T>, TensorCreateInfo> ||
        is_supported_scalar_arg_v<T>;
#endif

    // Capacity-overflow messages — spell the actual limit (MaxS/MaxT, whatever
    // the instantiation is) into the text via std::to_string. Built once into a
    // function-local static so set_error() can hold the const char* safely.
    static const char *scalar_cap_msg() {
        static const std::string msg = "Too many scalar args (max " + std::to_string(MaxS) + ")";
        return msg.c_str();
    }
    static const char *tensor_cap_msg() {
        static const std::string msg = "Too many tensor args (max " + std::to_string(MaxT) + ")";
        return msg.c_str();
    }

    template <typename T>
    void add_scalar_one(T &&value) {
        scalars_[scalar_count_] = to_u64(value);
#if SIMPLER_DFX
        uintptr_t scalar_source_ptr = 0;
        if constexpr (std::is_lvalue_reference_v<T>) {
            scalar_source_ptr = reinterpret_cast<uintptr_t>(&value);
        }
        dump_arg_selection_.record_scalar_source(
            scalar_count_, scalar_source_ptr, dtype_of<std::remove_cv_t<std::remove_reference_t<T>>>()
        );
#endif
        scalar_count_++;
    }

#if SIMPLER_DFX
    // No-arg dump(): mark every arg already added to this Arg.
    void mark_all_dump_args() {
        if (tensor_count_ == 0 && scalar_count_ == 0) {
            set_error("dump: no arguments added to this Arg");
            return;
        }
        dump_arg_selection_.mark_all(tensor_count_, scalar_count_);
    }

    void mark_dump_arg(const simpler::tmr::Tensor &tensor) {
        for (int32_t i = 0; i < tensor_count_; i++) {
            if (tags_[i] != TensorArgType::OUTPUT && tensors_[i].refers_to(&tensor)) {
                dump_arg_selection_.mark_index(i);
                return;
            }
        }
        set_error("dump: tensor is not part of this Arg");
    }

    void mark_dump_arg(const TensorCreateInfo &create_info) {
        for (int32_t i = 0; i < tensor_count_; i++) {
            if (tags_[i] == TensorArgType::OUTPUT && tensors_[i].refers_to(&create_info)) {
                dump_arg_selection_.mark_index(i);
                return;
            }
        }
        set_error("dump: TensorCreateInfo is not part of this Arg");
    }

    template <typename T>
    std::enable_if_t<is_supported_scalar_arg_v<T>, void> mark_dump_arg(const T &scalar) {
        uintptr_t ptr = reinterpret_cast<uintptr_t>(&scalar);
        if (dump_arg_selection_.mark_scalar_by_ptr(ptr, scalar_count_, tensor_count_)) {
            return;
        }
        set_error("dump: scalar is not part of this Arg");
    }
#endif

    // Compile-time validation: arg count, value category (reject temporaries —
    // a stored &arg would dangle after the call), and element type. Driven
    // purely by Args, with no runtime state.
    template <bool is_output, typename... Args>
    static void assert_add_tensor_args() {
        static_assert(sizeof...(Args) >= 1, "at least one argument required");
        static_assert(
            (std::is_lvalue_reference_v<Args> && ...),
            "temporaries are not allowed — stored pointers would dangle after the call"
        );
        if constexpr (is_output) {
            static_assert(
                (std::is_same_v<std::decay_t<Args>, simpler::tmr::Tensor> && ...) ||
                    (std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...),
                "add_output: all arguments must be the same type (all simpler::tmr::Tensor or all TensorCreateInfo)"
            );
        } else {
            static_assert(
                (std::is_same_v<std::decay_t<Args>, simpler::tmr::Tensor> && ...),
                "all arguments must be simpler::tmr::Tensor"
            );
        }
    }

    // Runtime validation: tensor-before-scalar ordering + slot capacity. Records
    // an error and returns false on violation.
    bool check_add_tensor_capacity(int32_t count) {
        if (scalar_count_ != 0) {
            set_error(
                "add_input/add_output/add_inout called after add_scalar: "
                "all tensors must be added before any scalars"
            );
            return false;
        }
        if (tensor_count_ + count > static_cast<int32_t>(MaxT)) {
            set_error(tensor_cap_msg());
            return false;
        }
        return true;
    }
};

// =============================================================================
// Task-args layer aliases
// =============================================================================
//
// CoreTaskArgs — core-level container used to build and submit tasks inside
//   orchestration (small, stack-friendly).
using CoreTaskArgs = Arg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>;

// ChipTaskArgs — chip-level entry-arg holding the orchestration entry's
// already-allocated inputs (capacity matches simpler::tmr::EntryArgsStorage).
// aicpu_orchestration_entry/config receive a const ChipTaskArgs&.
struct ChipTaskArgs : Arg<CHIP_MAX_TENSOR_ARGS, CHIP_MAX_SCALAR_ARGS> {
    // Build from the runtime's entry-arg storage: each input becomes a
    // TensorRef pointing into `src`, so `src` must outlive this (on the
    // executor path src is runtime->dev.orch_args_storage_, alive for the whole run).
    void create_from_entry_storage(const simpler::tmr::EntryArgsStorage &src) {
        reset();
        for (int32_t i = 0; i < src.tensor_count(); ++i) {
            // Adoption left every entry input external: no producing task, no
            // overlap version, default dependency treatment.
            const simpler::tmr::Tensor &t = src.tensor(i);
            debug_assert(!t.manual_dep && t.version == 0);
            add_input(t);
        }
        for (int32_t i = 0; i < src.scalar_count(); ++i) {
            add_scalar(src.scalar(i));
        }
    }
};
