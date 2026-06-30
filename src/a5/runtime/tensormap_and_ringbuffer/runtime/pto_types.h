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
 * - TaskOutputTensors: Return value from submit containing materialized output Tensors
 * - Arg: Aggregated argument container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, TensorCreateInfo) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_

#include <stdint.h>
#include <string.h>

#include <string>
#include <type_traits>
#include <utility>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "aicpu/dump_arg_selection.h"
#include "data_type.h"
#include "profiling_config.h"
#include "pto_submit_types.h"
#include "task_args.h"
#include "tensor.h"
#include "tensor_create_info.h"  // runtime-only TensorCreateInfo + materialization helpers

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
// Task Output Tensors (return value from submit)
// =============================================================================

enum class PTO2ScopeMode : uint8_t {
    AUTO = 0,
    MANUAL = 1,
};

/**
 * TaskOutputTensors — returned by submit, holds materialized output Tensors.
 *
 * Only runtime-created outputs are stored here, indexed in add_output order.
 *
 * The underlying storage is uninitialized; only output_count elements are
 * valid after submit returns.  This avoids default-constructing Tensor[]
 * on the hot path (2 KB of unnecessary zeroing per submit).
 *
 * Users must hold a named TaskOutputTensors variable and borrow via get_ref();
 * binding get_ref() on an rvalue is compile-time rejected to prevent dangling.
 *
 * LIFETIME — single-scope only:
 *   Internally this class stores pointers into the submitting task's payload
 *   (PTO2TaskPayload::tensors[]), which lives in a ring-buffer slot. After
 *   scope_end the slot becomes eligible for reuse, and a later submit will
 *   overwrite the same Tensor storage in place. Therefore the
 *   TaskOutputTensors instance, the const Tensor& returned by get_ref(), and
 *   any pointer derived from either MUST NOT outlive the PTO2_SCOPE in which
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
        task_id_(PTO2TaskId::invalid()),
        output_count_(0) {}

    bool empty() const { return output_count_ == 0; }
    uint32_t size() const { return output_count_; }

    /// Borrow a materialized output tensor by index (lvalue only).
    const Tensor &get_ref(uint32_t index) const & {
        always_assert(index < output_count_);
        return *tensors_[index];
    }
    const Tensor &get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: append one materialized output Tensor.
    void materialize_output(const Tensor &tensor) {
        always_assert(output_count_ < MAX_TENSOR_ARGS);
        tensors_[output_count_++] = &tensor;
    }

    void set_task_id(PTO2TaskId id) { task_id_ = id; }

    PTO2TaskId task_id() const { return task_id_; }

private:
    PTO2TaskId task_id_;
    uint32_t output_count_;
    // Upper bound: a task cannot have more outputs than total tensor args
    // (every OUTPUT/OUTPUT_EXISTING slot is one of the Arg's tensor slots).
    const Tensor *tensors_[MAX_TENSOR_ARGS];
};

// =============================================================================
// Argument Types (for pto_submit_task API)
// =============================================================================

// TensorArgType is defined in tensor.h (included via task_args.h above)

/**
 * Tagged reference to a single Arg slot — either a Tensor* or a
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
        const Tensor *ptr_;
        const TensorCreateInfo *create_info_;
    };

public:
    TensorRef() :
        ptr_(nullptr) {}
    TensorRef(const TensorRef &) = delete;
    TensorRef(TensorRef &&) = delete;
    TensorRef &operator=(const TensorRef &) = delete;
    TensorRef &operator=(TensorRef &&) = delete;

    TensorRef &operator=(const Tensor *p) {
        ptr_ = p;
        return *this;
    }
    TensorRef &operator=(const TensorCreateInfo *ci) {
        create_info_ = ci;
        return *this;
    }

    const Tensor &ref() const { return *ptr_; }
    const TensorCreateInfo &create_info() const { return *create_info_; }
    bool refers_to(const Tensor *t) const { return ptr_ == t; }
    bool refers_to(const TensorCreateInfo *ci) const { return create_info_ == ci; }
};

/**
 * Aggregated argument container for pto_submit_task
 *
 * Inherits storage from TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType>.
 * Each tensor slot stores a TensorRef union (Tensor* or TensorCreateInfo)
 * discriminated by the corresponding tag().
 * Tensors are dispatched first in kernel args, followed by scalars.
 *
 * Output arguments follow two distinct ownership models:
 * - add_output(const TensorCreateInfo&): OUTPUT — runtime allocates buffer
 *   and materializes a new Tensor, returned via TaskOutputTensors.
 * - add_inout(const Tensor&): INOUT — reuses an existing Tensor as the write target.
 *
 * Example:
 *   Tensor x = make_tensor_external(dev_a, shapes, 2);
 *   TensorCreateInfo ci(shapes, 2);  // must outlive submit
 *   Arg args;
 *   args.add_input(x);
 *   args.add_output(ci);
 *   args.add_scalar(some_value);
 *   TaskOutputTensors outs = rt_submit_aic_task(kernel_id, args);
 *   const Tensor& y = outs.get_ref(0);
 */
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
    const char *error_msg{nullptr};
    PTO2LaunchSpec launch_spec;  // SPMD launch parameters (block_num, etc.)

    void clear() {
        Base::clear();
#if PTO2_PROFILING
        dump_arg_selection_.clear();
#endif
        explicit_deps_ = nullptr;
        explicit_dep_count_ = 0;
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
#if PTO2_PROFILING
        static_assert(
            (std::is_lvalue_reference_v<Args> && ...),
            "dump: temporaries are not allowed — pass tensors/scalars already added to this Arg"
        );
        static_assert(
            (is_supported_dump_arg_v<Args> && ...),
            "dump: all arguments must be Tensor, TensorCreateInfo, or scalar lvalues"
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

#if PTO2_PROFILING
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

    /// Batch add outputs — all Tensor or all TensorCreateInfo:
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
     *   PTO2TaskId deps[3];
     *   uint32_t n = 0;
     *   if (have_prev) deps[n++] = prev;
     *   if (is_last)   deps[n++] = alloc;
     *   args.set_dependencies(deps, n);    // safe even if n == 0
     *
     * For count > 0, the call is single-shot: a second non-empty call after
     * deps are already set will fail with set_error(). Use count == 0 first
     * if you need to re-set.
     */
    void set_dependencies(const PTO2TaskId *deps, uint32_t count) {
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

    PTO2TaskId explicit_dep(uint32_t index) const {
        always_assert(index < explicit_dep_count_);
        return explicit_deps_[index];
    }

    const PTO2TaskId *explicit_deps_data() const { return explicit_deps_; }

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
#if PTO2_PROFILING
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
#if PTO2_PROFILING
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
#if PTO2_PROFILING
        dump_arg_selection_.copy_scalar_dtypes_from(src.dump_arg_selection_, scalar_count_, src_offset, count);
#endif
        scalar_count_ += count;
    }

#if PTO2_PROFILING
    const uint8_t *scalar_dtypes() const { return dump_arg_selection_.scalar_dtypes(); }
#else
    const uint8_t *scalar_dtypes() const { return nullptr; }
#endif

private:
    // Caller-owned dependency array; lifetime must extend through submit.
#if PTO2_PROFILING
    DumpArgSelection dump_arg_selection_;
#endif
    const PTO2TaskId *explicit_deps_{nullptr};
    uint32_t explicit_dep_count_{0};
#if PTO2_PROFILING
    template <typename T>
    static constexpr bool is_supported_dump_arg_v =
        std::is_same_v<std::decay_t<T>, Tensor> || std::is_same_v<std::decay_t<T>, TensorCreateInfo> ||
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
#if PTO2_PROFILING
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

#if PTO2_PROFILING
    // No-arg dump(): mark every arg already added to this Arg.
    void mark_all_dump_args() {
        if (tensor_count_ == 0 && scalar_count_ == 0) {
            set_error("dump: no arguments added to this Arg");
            return;
        }
        dump_arg_selection_.mark_all(tensor_count_, scalar_count_);
    }

    void mark_dump_arg(const Tensor &tensor) {
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
                (std::is_same_v<std::decay_t<Args>, Tensor> && ...) ||
                    (std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...),
                "add_output: all arguments must be the same type (all Tensor or all TensorCreateInfo)"
            );
        } else {
            static_assert((std::is_same_v<std::decay_t<Args>, Tensor> && ...), "all arguments must be Tensor");
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
// L0TaskArgs — core-level container used to build and submit tasks inside
//   orchestration (small, stack-friendly).
using L0TaskArgs = Arg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>;

// L2TaskArgs — chip-level entry-arg holding the orchestration entry's
// already-allocated inputs (capacity matches ChipStorageTaskArgs).
// aicpu_orchestration_entry/config receive a const L2TaskArgs&.
struct L2TaskArgs : Arg<CHIP_MAX_TENSOR_ARGS, CHIP_MAX_SCALAR_ARGS> {
    // Build from the executor's ChipStorageTaskArgs: each input becomes a
    // TensorRef pointing at src's Tensor, so `src` must outlive this (on the
    // executor path src is runtime->dev.orch_args_storage_, alive for the whole run).
    void create_from_chip_args(const ChipStorageTaskArgs &src) {
        reset();
        for (int32_t i = 0; i < src.tensor_count(); ++i) {
            // Entry inputs are external submit-time tensors; the entry binds them
            // by const Tensor& (replacing from_tensor_arg's old version/manual_dep
            // reset), so this invariant is what keeps that binding behavior-preserving.
            const Tensor &t = src.tensor(i);
            debug_assert(!t.manual_dep && t.version == 0);
            add_input(t);
        }
        for (int32_t i = 0; i < src.scalar_count(); ++i) {
            add_scalar(src.scalar(i));
        }
    }
};

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
