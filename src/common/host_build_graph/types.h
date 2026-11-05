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
 * simpler::hbg::Tensor descriptor types (simpler::hbg::Tensor, PTOBufferHandle, TensorCreateInfo) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorLease, HostApi).
 */

#pragma once

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <string>
#include <type_traits>
#include <utility>

#include "assert_compat.h"
#include "aicpu/dump_arg_selection.h"
#include "common/device_phase.h"
#include "data_type.h"
#include "host_build_graph/entry_args.h"  // EntryArgsStorage
#include "profiling_config.h"
#include "host_build_graph/submit_types.h"
#include "task_args.h"
#include "tensor.h"
#include "host_build_graph/tensor_create_info.h"  // runtime-only TensorCreateInfo + materialization helpers

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
 * TaskOutputTensors — returned by submit, holds materialized output ChipTensors.
 *
 * Only runtime-created outputs are stored here, indexed in add_output order.
 *
 * The underlying storage is uninitialized; only output_count elements are
 * valid after submit returns.  This avoids default-constructing simpler::hbg::Tensor[]
 * on the hot path (2 KB of unnecessary zeroing per submit).
 *
 * Users must hold a named TaskOutputTensors variable and borrow via get_ref();
 * binding get_ref() on an rvalue is compile-time rejected to prevent dangling.
 *
 * LIFETIME — single-pass only:
 *   Internally this class stores pointers into the submitting task's tensor
 *   storage: the region named by TaskPayload::tensors for a plain submit, the
 *   GraphRecording sub-task's tensors for a submit inside a Graph body. Both belong to
 *   one orchestration pass, which the next bind rebuilds over the same bytes.
 *   Therefore the TaskOutputTensors instance, the const simpler::hbg::Tensor& returned by
 *   get_ref(), and any pointer derived from either MUST NOT outlive the
 *   orchestration entry that submitted the task — do not move/copy them into state
 *   that survives the pass, and do not capture them by std::reference_wrapper or
 *   raw pointer across that boundary.
 *
 *   This invariant is intentionally not enforced at runtime: rebuilt storage
 *   carries a different but valid owner_task_id, so checking owner_task_id cannot
 *   distinguish "still mine" from "silently aliased to an unrelated task". Misuse
 *   manifests as a wrong-tensor read with no diagnostic.
 */
class TaskOutputTensors {
public:
    TaskOutputTensors() :
        task_id_(TaskId::invalid()),
        output_count_(0) {}

    bool empty() const { return output_count_ == 0; }
    uint32_t size() const { return output_count_; }

    /// Borrow a materialized output tensor by index (lvalue only).
    const simpler::hbg::Tensor &get_ref(uint32_t index) const & {
        always_assert(index < output_count_);
        return *tensors_[index];
    }
    const simpler::hbg::Tensor &get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: append one materialized output simpler::hbg::Tensor.
    void materialize_output(const simpler::hbg::Tensor &tensor) {
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
    const simpler::hbg::Tensor *tensors_[MAX_TENSOR_ARGS];
};

using TaskSubmitResult = TaskOutputTensors;

// =============================================================================
// Argument Types (for rt_submit_task API)
// =============================================================================

// TensorArgType is defined in tensor.h (included via task_args.h above)

/**
 * Tagged reference to a single Arg slot — either a simpler::hbg::Tensor* or a
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
        const simpler::hbg::Tensor *ptr_;
        const TensorCreateInfo *create_info_;
    };

public:
    TensorRef() :
        ptr_(nullptr) {}
    TensorRef(const TensorRef &) = delete;
    TensorRef(TensorRef &&) = delete;
    TensorRef &operator=(const TensorRef &) = delete;
    TensorRef &operator=(TensorRef &&) = delete;

    TensorRef &operator=(const simpler::hbg::Tensor *p) {
        ptr_ = p;
        return *this;
    }
    TensorRef &operator=(const TensorCreateInfo *ci) {
        create_info_ = ci;
        return *this;
    }

    const simpler::hbg::Tensor &ref() const { return *ptr_; }
    const TensorCreateInfo &create_info() const { return *create_info_; }
    bool refers_to(const simpler::hbg::Tensor *t) const { return ptr_ == t; }
    bool refers_to(const TensorCreateInfo *ci) const { return create_info_ == ci; }
};

template <size_t MaxT, size_t MaxS>
class Arg;
template <size_t MaxT, size_t MaxS>
class InheritableArg;
class InheritableScalar;

/**
 * A scalar parameter handed out for forwarding: its value, and the address of the slot it
 * ultimately comes from.
 *
 * Carrying both is what lets a destination store the value without ever dereferencing the
 * origin. So the origin is allowed to dangle -- a caller's local goes out of scope once
 * submit returns -- and the value is still right. The address is only ever tested against
 * null (dynamic vs static parameter) or subtracted from a boundary's slot array base to
 * recover a parameter index (recording).
 *
 * Arg::scalar(i) folds: an already-inherited slot yields its own origin rather than
 * itself, so a chain is exactly one hop and recording resolves it without a walk.
 */
class InheritableScalar {
public:
    constexpr InheritableScalar(uint64_t bits, const void *origin) :
        bits_(bits),
        origin_(origin) {}

    constexpr const void *origin() const { return origin_; }

private:
    // A handle offers no value read at all. It cannot tell a dynamic parameter from a
    // static one -- InheritableArg::scalar(i) hands out a self-pointing origin either way --
    // so a read taken from one could not be checked against the slot's declaration, where
    // InheritableArg::scalar<T>(i) has the Arg and can. Forwarding is all a handle is for,
    // and add_scalar_one copies these bytes across as they are.
    template <size_t MaxT, size_t MaxS>
    friend class InheritableArg;

    uint64_t bits_;
    const void *origin_;
};
static_assert(sizeof(InheritableScalar) == 16, "InheritableScalar is passed by value and holds a value and an origin");

// Defined here rather than beside is_supported_scalar_arg_v in data_type.h: that trait
// is shared with the tensormap_and_ringbuffer runtime, which has no Graph and whose
// dtype_of/mark_dump_arg would silently accept a type they cannot describe.
template <typename T>
inline constexpr bool is_inheritable_scalar_v = std::is_same_v<std::decay_t<T>, InheritableScalar>;

/**
 * Aggregated argument container for rt_submit_task
 *
 * Inherits storage from TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType>.
 * Each tensor slot stores a TensorRef union (simpler::hbg::Tensor* or TensorCreateInfo)
 * discriminated by the corresponding tag(). Each scalar slot stores a value; the parallel
 * scalar_inherited_ array names where that value came from, or is null when the parameter
 * is static.
 * ChipTensors are dispatched first in kernel args, followed by scalars.
 *
 * Output arguments follow two distinct ownership models:
 * - add_output(const TensorCreateInfo&): OUTPUT — runtime allocates buffer
 *   and materializes a new simpler::hbg::Tensor, returned via TaskOutputTensors.
 * - add_inout(const simpler::hbg::Tensor&): INOUT — reuses an existing simpler::hbg::Tensor as the write target.
 *
 * Example:
 *   simpler::hbg::Tensor x = simpler::hbg::make_tensor_external(dev_a, shapes, 2);
 *   TensorCreateInfo ci(shapes, 2);  // must outlive submit
 *   Arg args;
 *   args.add_input(x);
 *   args.add_output(ci);
 *   args.add_scalar(some_value);
 *   TaskOutputTensors outs = rt_submit_aic_task(kernel_id, args);
 *   const simpler::hbg::Tensor& y = outs.get_ref(0);
 */

// Operand of a dispatch predicate (L0 layer): locates one element of a tensor —
// tensor + ndims + indices, mirroring get_tensor_data. The tensor is borrowed and
// must outlive submit; its buffer must be allocated by then, and its producer must
// be a dependency of the predicated task so the value is current at dispatch.
struct CorePredicateOperand {
    const simpler::hbg::Tensor *tensor{nullptr};
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
class Arg : private TaskArgsTpl<TensorRef, uint64_t, MaxT, MaxS, TensorArgType> {
    using Base = TaskArgsTpl<TensorRef, uint64_t, MaxT, MaxS, TensorArgType>;

public:
    // The base's own API, re-exported one name at a time. Private inheritance is what
    // makes that a choice rather than a default: a public base is reachable by an
    // implicit derived-to-base conversion, through which its members are public again
    // however this class hides their names, so `static_cast<const Base &>(args).tags_`
    // would read the tag array the accessors exist to mediate.
    using Base::scalar;
    using Base::scalar_count;
    using Base::tag;
    using Base::tag_data;
    using Base::tensor;
    using Base::tensor_count;
    using Base::tensor_data;

    // Minimal-permission: an Arg is built in place and consumed by reference;
    // it is never copied/moved (it is a large object, and its TensorRef slots
    // are non-copyable by design).
    //
    // Construction is protected: Arg is the shared implementation of the task-args layer,
    // never a type a caller names. What a caller builds is one of the aliases below --
    // CoreTaskArgs, GraphTaskArgs, ChipTaskArgs -- each of which decides what its slots mean.
    Arg(const Arg &) = delete;
    Arg(Arg &&) = delete;
    Arg &operator=(const Arg &) = delete;
    Arg &operator=(Arg &&) = delete;

    bool has_error() const { return has_error_; }
    const char *error_msg() const { return error_msg_; }

    void set_allow_early_resolve(bool v = true) { allow_early_resolve_ = v; }
    bool allow_early_resolve() const { return allow_early_resolve_; }

    void set_predicate(const CoreTaskPredicate &pred) { predicate_ = pred; }
    const CoreTaskPredicate &predicate() const { return predicate_; }

    // An out-of-range id fails through the standard invalid-arg path so the scheduler
    // never stamps out of bounds.
    void set_task_timing_slot(int32_t slot) {
        if (slot < 0 || slot >= NUM_TASK_TIMING_SLOTS) {
            set_error("task_timing_slot out of range (valid: 0..15)");
            return;
        }
        task_timing_slot_ = slot;
    }
    int32_t task_timing_slot() const { return task_timing_slot_; }

    void reset() {
        clear();
        has_error_ = false;
        error_msg_ = nullptr;
    }

    void set_error(const char *msg) {
        if (!has_error_) {
            has_error_ = true;
            error_msg_ = msg;
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
            "dump: all arguments must be simpler::hbg::Tensor, TensorCreateInfo, or scalar lvalues"
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

    /// Batch add outputs — all simpler::hbg::Tensor or all TensorCreateInfo:
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
     * the result through unguarded, including in the no-dep branch. Fill with
     * TaskId::invalid() so an entry the branches never write is rejected by the
     * orchestrator rather than read as a task id:
     *   TaskId deps[3] = {TaskId::invalid(), TaskId::invalid(), TaskId::invalid()};
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
    }

    uint32_t explicit_dep_count() const { return explicit_dep_count_; }

    TaskId explicit_dep(uint32_t index) const {
        always_assert(index < explicit_dep_count_);
        return explicit_deps_[index];
    }

    const TaskId *explicit_deps_data() const { return explicit_deps_; }

    /**
     * Add scalar values. Types are deduced per argument; each value is bit-cast to uint64_t
     * for storage. Mixed types are allowed:
     *
     *   args.add_scalar(token_pos);                   // single
     *   args.add_scalar(3.14f, int32_t(42), 7u);      // mixed batch
     *
     * Every slot of an Arg holds a value of its own. A slot that instead follows a Graph
     * boundary parameter is something only an InheritableArg has, and only its add_scalar
     * accepts the handle that makes one.
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

    // Bulk form of add_scalar.
    template <typename T>
    void add_scalars(const T *values, int count) {
        static_assert(is_supported_scalar_arg_v<T>, "add_scalars: element type must be arithmetic or enum");
        if (count < 0 || scalar_count_ + count > MaxS) {
            set_error(scalar_cap_msg());
            return;
        }
        if constexpr (std::is_same_v<std::remove_cv_t<T>, uint64_t>) {
            memcpy(&scalars_[scalar_count_], values, static_cast<size_t>(count) * sizeof(uint64_t));
        } else {
            for (int i = 0; i < count; ++i) {
                scalars_[scalar_count_ + i] = to_u64(values[i]);
            }
        }
#if SIMPLER_DFX
        dump_arg_selection_.clear_scalar_metadata(scalar_count_, count);
#endif
        scalar_count_ += count;
    }

    // Copy every value into a compact uint64 array -- what submit and payload
    // materialization need. A slot is always a value, so this is one memcpy: no
    // discriminator to consult, nothing to resolve.
    void pack_scalars(uint64_t *out) const {
        memcpy(out, scalars_, static_cast<size_t>(scalar_count_) * sizeof(uint64_t));
    }

    // Base of the slot array. Its only use is turning an origin pointer into a boundary
    // parameter index; named apart from a value accessor because what it hands out is an
    // identity baseline rather than a run of values.
    const void *scalar_slot_base() const { return scalars_; }

#if SIMPLER_DFX
    const uint8_t *scalar_dtypes() const { return dump_arg_selection_.scalar_dtypes(); }
#else
    const uint8_t *scalar_dtypes() const { return nullptr; }
#endif

protected:
    Arg() = default;

    // Empty every field an Arg carries, keeping the error state. Protected rather than
    // public: a derived Arg may add state that clearing has to reach -- InheritableArg's
    // origin array does -- and this one is not virtual, so a call through an `Arg &` would
    // run this body alone and leave that state behind. A slot still naming an origin while
    // holding a value of its own is recorded as following a boundary parameter, so the
    // Definition takes its value from the wrong place. `reset()` is the public spelling,
    // and each derived Arg overrides it to cover its own fields.
    void clear() {
        Base::clear();
#if SIMPLER_DFX
        dump_arg_selection_.clear();
#endif
        explicit_deps_ = nullptr;
        explicit_dep_count_ = 0;
        allow_early_resolve_ = false;
        task_timing_slot_ = TASK_TIMING_SLOT_NONE;
        predicate_ = CoreTaskPredicate{};
    }

    // Capacity-overflow message — spells the actual limit (MaxS, whatever the
    // instantiation is) into the text via std::to_string. Built once into a
    // function-local static so set_error() can hold the const char* safely.
    static const char *scalar_cap_msg() {
        static const std::string msg = "Too many scalar args (max " + std::to_string(MaxS) + ")";
        return msg.c_str();
    }

    // Add one value. T is still known here, which is why a slot of any width is stored
    // correctly: deferring the conversion would need the width, and the slot has nowhere
    // to keep it.
    //
    // Protected rather than private: InheritableArg builds on it, adding the origin that
    // makes a slot follow a parameter.
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
    // What dump() matches a scalar by, for a slot a derived Arg filled itself.
    void record_scalar_source_for_slot(int32_t slot, uintptr_t source_ptr, uint8_t dtype) {
        dump_arg_selection_.record_scalar_source(slot, source_ptr, dtype);
    }
#endif

private:
    // Held fully private: these two hand out the slot array with nothing to qualify it.
    using Base::scalar_data;
    using Base::scalars;

#if SIMPLER_DFX
    template <typename T>
    static constexpr bool is_supported_dump_arg_v =
        std::is_same_v<std::decay_t<T>, simpler::hbg::Tensor> || std::is_same_v<std::decay_t<T>, TensorCreateInfo> ||
        is_supported_scalar_arg_v<T>;
#endif

    static const char *tensor_cap_msg() {
        static const std::string msg = "Too many tensor args (max " + std::to_string(MaxT) + ")";
        return msg.c_str();
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

    void mark_dump_arg(const simpler::hbg::Tensor &tensor) {
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
                (std::is_same_v<std::decay_t<Args>, simpler::hbg::Tensor> && ...) ||
                    (std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...),
                "add_output: all arguments must be the same type (all simpler::hbg::Tensor or all TensorCreateInfo)"
            );
        } else {
            static_assert(
                (std::is_same_v<std::decay_t<Args>, simpler::hbg::Tensor> && ...),
                "all arguments must be simpler::hbg::Tensor"
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

public:
    LaunchSpec launch_spec;  // SPMD launch parameters (block_num, etc.)

protected:
    // The dependent base's storage. These declarations are what makes it reachable by
    // unqualified lookup inside this template, which does not search a dependent base.
    //
    // Protected, not public: a derived Arg writes these slot by slot
    // (GraphTaskArgs::gen_scalar_params_from_args), while every reader outside goes
    // through an accessor that pairs each array with what qualifies it -- tensor() and
    // tag() for the tensors, pack_scalars() and scalar(i) for the values and origins.
    using Base::scalar_count_;
    using Base::scalars_;
    using Base::tags_;
    using Base::tensor_count_;
    using Base::tensors_;

private:
    bool has_error_{false};
    const char *error_msg_{nullptr};

    // Speculative early-dispatch hint (codegen-author set, off by default). When true,
    // the scheduler may stage this task on an idle core before its producer finishes,
    // gating execution on the DATA_MAIN_BASE doorbell — only safe when the author knows
    // the task's data dependencies allow it. Read in-process by the runtime; never
    // crosses the wire format.
    bool allow_early_resolve_{false};

    // Dispatch predicate (codegen-author set; default op == NONE = always dispatch). A
    // FALSE result at the dispatch point retires the task inline through the dep-only
    // path — never dispatched to an AICore — while still resolving fanin/fanout so
    // consumers unlock. The predicate tensor's producer MUST be a dependency of this task
    // so the value is current when the task becomes ready. Read in-process; never crosses
    // the wire.
    CoreTaskPredicate predicate_;

    // Scheduler records this task's AICPU dispatch/finish cycles into fixed slot 0..15.
    // TASK_TIMING_SLOT_NONE leaves it untagged.
    int32_t task_timing_slot_{TASK_TIMING_SLOT_NONE};

#if SIMPLER_DFX
    DumpArgSelection dump_arg_selection_;
#endif

    // Caller-owned dependency array; lifetime must extend through submit.
    const TaskId *explicit_deps_{nullptr};
    uint32_t explicit_dep_count_{0};
};

/**
 * An Arg whose slots may follow someone else's.
 *
 * The difference from Arg is one array: where each slot's value came from. A slot that
 * names an origin is a parameter being forwarded -- a Graph boundary scalar passed through
 * to a sub-task -- and recording resolves that origin into a boundary parameter index
 * instead of freezing the value into the Definition.
 *
 * Which Args need it is decided by whether they can inherit at all, not by their position
 * in the hierarchy. A sub-task's arguments and a Graph's boundary both can; an orchestration
 * entry's arguments cannot -- those are the run's own values, with nothing above them to
 * follow -- so ChipTaskArgs stays on plain Arg and keeps a value read that is always legal.
 */
template <size_t MaxT, size_t MaxS>
class InheritableArg : private Arg<MaxT, MaxS> {
    using Base = Arg<MaxT, MaxS>;

public:
    // Arg's own API, re-exported one name at a time, for the reason Arg itself inherits
    // privately: a public base is reachable by an implicit derived-to-base conversion, and
    // every mutator here is non-virtual, so `Arg &a = args; a.reset();` would run Arg's body
    // alone and leave `scalar_inherited_` holding origins for slots that no longer exist. A
    // slot still naming an origin while holding a value of its own is recorded as following
    // a boundary parameter, so the Definition takes its value from the wrong place.
    //
    // Four names are deliberately absent and defined below instead -- clear, reset,
    // add_scalar, add_scalars -- because each has to cover the origin array too. Arg::scalar
    // is absent for a different reason: a slot here may follow a parameter, so the only
    // read this class offers is the handle.
    using Base::add_inout;
    using Base::add_input;
    using Base::add_no_dep;
    using Base::add_output;
    using Base::allow_early_resolve;
    using Base::error_msg;
    using Base::explicit_dep;
    using Base::explicit_dep_count;
    using Base::explicit_deps_data;
    using Base::has_error;
    using Base::pack_scalars;
    using Base::predicate;
    using Base::scalar_count;
    using Base::scalar_dtypes;
    using Base::scalar_slot_base;
    using Base::set_allow_early_resolve;
    using Base::set_dependencies;
    using Base::set_error;
    using Base::set_predicate;
    using Base::set_task_timing_slot;
    using Base::tag;
    using Base::tag_data;
    using Base::task_timing_slot;
    using Base::tensor;
    using Base::tensor_count;
    using Base::tensor_data;
#if SIMPLER_DFX
    using Base::dump;
#endif
    using Base::dump_arg_index_ambiguous_mask;
    using Base::dump_arg_mask;

    void reset() {
        Base::reset();
        scalar_inherited_.fill(nullptr);
    }

    /**
     * Add scalar values. Types are deduced per argument; each value is bit-cast to uint64_t
     * for storage. Mixed types are allowed:
     *
     *   args.add_scalar(token_pos);                            // a value of this slot's own
     *   args.add_scalar(boundary.scalar(0));                   // follows a boundary parameter
     *   args.add_scalar(3.14f, boundary.scalar(1), 7u);        // mixed batch
     *
     * **What a slot follows is decided by what was passed, not by its value category.** An
     * InheritableScalar names a parameter, so the slot follows it and is refreshed on every
     * replay; anything else is a value of this slot's own, whether it arrived as an lvalue
     * or a literal.
     */
    template <typename... Args>
    void add_scalar(Args &&...args) {
        static_assert(sizeof...(Args) >= 1, "add_scalar: at least one argument required");
        static_assert(
            ((is_supported_scalar_arg_v<Args> || is_inheritable_scalar_v<Args>) && ...),
            "add_scalar: all types must be arithmetic, enum, or a Graph boundary scalar"
        );
        if (this->scalar_count_ + sizeof...(Args) > MaxS) {
            this->set_error(Base::scalar_cap_msg());
            return;
        }
        (add_scalar_one(std::forward<Args>(args)), ...);
    }

    // Bulk form of add_scalar, dispatching on the element type the same way: a run of values
    // fills value slots, a run of handles fills slots that follow their parameters. One call
    // is necessarily homogeneous -- `const T *` has one T -- so mixing the two takes the
    // variadic form above.
    template <typename T>
    void add_scalars(const T *values, int count) {
        if constexpr (is_inheritable_scalar_v<T>) {
            if (count < 0 || this->scalar_count_ + count > MaxS) {
                this->set_error(Base::scalar_cap_msg());
                return;
            }
            for (int i = 0; i < count; ++i) {
                add_scalar_one(values[i]);
            }
        } else {
            const int32_t first = this->scalar_count_;
            Base::add_scalars(values, count);
            // A refused batch left the count where it was, so nothing below is reached.
            for (int i = 0; first + i < this->scalar_count_; ++i) {
                scalar_inherited_[first + i] = nullptr;
            }
        }
    }

    // Hand out parameter i for forwarding: its value, plus the slot that value comes from.
    // Passing the result to another InheritableArg's add_scalar makes that slot follow this
    // one.
    //
    // There is no value read. A slot here may follow a parameter the caller refreshes per
    // invocation, and a number taken out of one becomes ordinary host data: whatever the
    // body computes from it is fixed in the Definition, and every later invocation replays
    // that number instead of the one it passed. A Graph's fixed values belong in its Config,
    // which the body receives alongside these.
    //
    // An already-inherited slot yields its own origin rather than itself, so the chain a
    // destination records is always one hop: "C inherits B, B inherits A" records C -> A.
    InheritableScalar scalar(int32_t i) const {
        return {this->scalars_[i], scalar_inherited_[i] != nullptr ? scalar_inherited_[i] : &this->scalars_[i]};
    }

    // A parameter follows another exactly when it names where its value came from. The
    // address is never dereferenced -- see scalar_origin.
    bool scalar_dynamic(int32_t i) const { return scalar_inherited_[i] != nullptr; }

    // Where parameter i's value came from, or null when it is static.
    //
    // This address MUST NOT be dereferenced: it may already dangle, since a caller's local
    // goes out of scope once submit returns. It is only ever tested against null and
    // subtracted from a slot array base, and that is safe precisely because the value was
    // copied at add_scalar time.
    const void *scalar_origin(int32_t i) const { return scalar_inherited_[i]; }

protected:
    // The base's protected functions. Private inheritance would otherwise end the chain
    // here: Arg's protected members are private in this class, so a class derived from this
    // one could not reach them at all.
    using Base::scalar_cap_msg;

    // Covers the origin array the base cannot see. Protected for the same reason the base's
    // is, and overriding it is what keeps a derived Arg's own clear() correct.
    void clear() {
        Base::clear();
        // All-null is a correct empty state: every slot reads as its own value, so nothing
        // below scalar_count_ can be mistaken for one that follows a source.
        scalar_inherited_.fill(nullptr);
    }

private:
    // Add one slot. What it ends up being is decided by T alone: a handle names a parameter
    // and the slot follows it, anything else is a value of this slot's own.
    template <typename T>
    void add_scalar_one(T &&value) {
        if constexpr (is_inheritable_scalar_v<T>) {
            // The value travels with the handle, so following the parameter costs no
            // dereference of the origin -- and lets the origin dangle harmlessly. A slot is
            // 8 bytes whatever the source type was, so the bits copy across as they are.
            this->scalars_[this->scalar_count_] = value.bits_;
            scalar_inherited_[this->scalar_count_] = value.origin();
#if SIMPLER_DFX
            // No host address to identify this slot by: it names a boundary parameter,
            // not a caller variable, so dump() cannot match it by pointer. The dtype is
            // u64 for the same reason -- an InheritableScalar carries bits and origin and
            // nothing else -- so a forwarded float or int32 slot prints as its bit
            // pattern rather than as its source type.
            this->record_scalar_source_for_slot(this->scalar_count_, 0, dtype_of<uint64_t>());
#endif
            this->scalar_count_++;
        } else {
            const int32_t slot = this->scalar_count_;
            Base::add_scalar_one(std::forward<T>(value));
            scalar_inherited_[slot] = nullptr;
        }
    }

    // ---- Data members ----------------------------------------------------------------
    //
    // Below this line the access specifiers run a second time, for storage rather than for
    // functions. The base's `launch_spec` is assigned through rather than called, which is
    // what puts it here and not among the re-exports above.

public:
    using Base::launch_spec;

protected:
    // Slot storage, carried through for the same reason as the protected functions above: a
    // derived Arg writes it directly, GraphTaskArgs::gen_scalar_params_from_args filling a
    // whole parameter list at once.
    using Base::scalar_count_;
    using Base::scalars_;
    using Base::tags_;
    using Base::tensor_count_;
    using Base::tensors_;

    // One more column of those same slots, and this class's own rather than the base's --
    // which is why it is a declaration where the five above are using-declarations.
    //
    // Where each parameter's value came from, or null when the slot holds a value of its
    // own. These addresses are compared and subtracted, never dereferenced -- one may point
    // at a caller local that has already gone out of scope. That is safe precisely because
    // the value was copied at add_scalar time.
    std::array<const void *, MaxS> scalar_inherited_{};
};

// =============================================================================
// Task-args layer aliases
// =============================================================================
//
// CoreTaskArgs — core-level container used to build and submit tasks inside
//   orchestration (small, stack-friendly). On the inheritable layer: a sub-task's
//   slot may forward a Graph boundary parameter.
using CoreTaskArgs = InheritableArg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>;

// Tensor and scalar capacity of a Graph boundary.
inline constexpr int32_t GRAPH_MAX_TENSOR_ARGS = 128;
inline constexpr int32_t GRAPH_MAX_SCALAR_ARGS = 64;

/**
 * A Graph's argument list.
 *
 * One type serves both sides of the call. A caller fills one with the arguments it is
 * passing; the in-flight entry holds one that is the Graph's formal parameters. The two
 * are separated by gen_scalar_params_from_args below -- before it, a slot's origin names
 * wherever the caller's value came from; after it, a dynamic parameter names itself, and
 * that is what a recorded body resolves against.
 *
 * Sized independently of CoreTaskArgs because the outer GRAPH payload carries the whole
 * boundary, while materialize stages only one sub-task's arguments at a time. The
 * compact boundary values live in that payload's argument-pool regions, so widening these
 * caps costs pool bytes only for Graphs that use them; TaskPayload itself stays
 * fixed-size.
 *
 * A type of its own rather than an alias of Arg, because generating a parameter list is
 * something only a Graph does: the one method below writes the slot arrays directly,
 * which is why they are protected rather than private.
 */
struct GraphTaskArgs : InheritableArg<GRAPH_MAX_TENSOR_ARGS, GRAPH_MAX_SCALAR_ARGS> {
    /**
     * Existing tensors only: a Graph boundary cannot take a runtime-allocated output.
     *
     * A Definition records the device addresses its body resolved against, so every
     * boundary tensor must already own its buffer when the body is recorded and again on
     * every replay. A TensorCreateInfo names a buffer the runtime would allocate at
     * submit, which is a different address each time and none at record time.
     */
    template <typename... Args>
    void add_output(Args &&...args) {
        static_assert(
            !(std::is_same_v<std::decay_t<Args>, TensorCreateInfo> || ...),
            "a Graph boundary cannot take a runtime-allocated output (TensorCreateInfo); "
            "allocate the tensor before the Graph and pass it as an existing Tensor"
        );
        InheritableArg<GRAPH_MAX_TENSOR_ARGS, GRAPH_MAX_SCALAR_ARGS>::add_output(std::forward<Args>(args)...);
    }

    /**
     * Build this Graph's scalar parameters from the arguments a caller passed.
     *
     * Values are resolved straight into the slots, because an argument's origin does not
     * outlive the submit call that lent it. **Every parameter then names itself**, which is
     * what keeps this list the basis recording resolves against: scalar(i) folds to
     * &scalars_[i], so a task slot that follows parameter i reports this array's i-th slot
     * and graph_classify_scalars turns that into the index i.
     *
     * Naming the caller's variable instead would hand out an address outside this array, the
     * task slot would be recorded as the body's own value, and the parameter would silently
     * stop being refreshed on replay.
     *
     * Unconditional because a boundary parameter is refreshed per invocation, with no second
     * kind to tell apart: a Graph's fixed values live in its Config. Carrying the caller
     * slot's own origin across would make a parameter the caller happened to pass by value
     * stop being refreshed, which is the distinction this boundary no longer has.
     */
    void gen_scalar_params_from_args(const GraphTaskArgs &args) {
        // The whole list is generated at once, into an Arg that has none yet -- graph_begin
        // calls this on an object it has just built. Appending to an existing list has no
        // meaning here: these are the Graph's parameters, not values accumulated by a
        // caller. Both sides are GraphTaskArgs, so the source can never overflow this.
        debug_assert(scalar_count_ == 0 && "a parameter list is generated whole, not appended to");
        const int32_t count = args.scalar_count();
        args.pack_scalars(scalars_);
        for (int32_t i = 0; i < count; ++i) {
            scalar_inherited_[i] = &scalars_[i];
        }
        scalar_count_ = count;
    }
};

// ChipTaskArgs — chip-level entry-arg holding the orchestration entry's
// already-allocated inputs (capacity matches simpler::hbg::EntryArgsStorage).
// aicpu_orchestration_entry/config receive a const ChipTaskArgs&.
struct ChipTaskArgs : Arg<CHIP_MAX_TENSOR_ARGS, CHIP_MAX_SCALAR_ARGS> {
    // Build from the runtime's entry-arg storage: each input becomes a TensorRef
    // pointing into `src`, so `src` must outlive this (on the executor path src is
    // runtime->orch_args_storage_, alive for the whole run).
    void create_from_entry_storage(const simpler::hbg::EntryArgsStorage &src) {
        reset();
        for (int32_t i = 0; i < src.tensor_count(); ++i) {
            // Adoption left every entry input external: no producing task, no
            // overlap version, default dependency treatment.
            const simpler::hbg::Tensor &t = src.tensor(i);
            debug_assert(!t.manual_dep && t.version == 0);
            add_input(t);
        }
        // An orchestration entry's scalars are the run's own values, not a Graph boundary's
        // parameters, so no slot here follows anything -- which is what every Arg slot
        // already is.
        for (int32_t i = 0; i < src.scalar_count(); ++i) {
            add_scalar(src.scalar(i));
        }
    }
};
