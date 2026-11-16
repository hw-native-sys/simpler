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
 * The L3+ argument form and everything that speaks its wire format.
 *
 * The element here is the self-describing `Tensor` from buffer.h — a buffer
 * descriptor plus a strided view, carrying no materialized address. That is what
 * an L3+ submit builds and what crosses the L3->L2 boundary:
 *
 *   - TaskArgs             — the unified builder Orchestrator.submit_* takes:
 *                            vector-backed, TensorArgType-tagged, with host-graph
 *                            explicit dependency handles
 *   - TaskArgsView         — zero-copy view over a blob (no tags; they are consumed
 *                            at submit time and never travel)
 *   - write_blob/read_blob — length-prefixed serialization for PROCESS-mode mailbox
 *                            transport
 *   - validate_submit_args — the submit-time access and overlap checks
 *
 * Separate from task_args.h because that header is on the include path of every
 * kernel and orchestration translation unit, and none of them resolves a buffer
 * descriptor: they work in `ChipTensor` and their runtime's own `Tensor`. Reaching
 * buffer.h from there would also put a second `Tensor` — this one at global scope —
 * in their scope. Include this header only where the L3+ form is actually handled:
 * the host orchestrator under src/common/hierarchical/, the Python bindings, and the
 * platform host code that decodes a blob into ChipStorageTaskArgs.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "buffer.h"  // Tensor wire type, AccessMode, validate_tensor, tensors_overlap
#include "task_args.h"

// ============================================================================
// TaskArgs — view values plus per-call request metadata
// ============================================================================

// Unified user-facing builder: vector-backed with TensorArgType tags.
// Used by Orchestrator.submit_*; tags drive dependency inference at submit
// time and are stripped before the args cross the dispatch boundary. The element
// is Tensor (self-describing view; L3+ holds no C++ ChipTensor) — the L3→L2 wire
// carries views plus requests, materialized to ChipStorageTaskArgs on the L2 child.
struct CcuTaskBinding {
    uint64_t allocation_id;
    uint64_t registration_id;
    uint32_t ccu_id;
    uint32_t reserved{0};
};
static_assert(sizeof(CcuTaskBinding) == 24 && std::is_trivially_copyable_v<CcuTaskBinding>);

class TaskArgs : private TaskArgsTpl<Tensor, uint64_t, 0, 0, TensorArgType> {
    using Base = TaskArgsTpl<Tensor, uint64_t, 0, 0, TensorArgType>;
    std::vector<TensorTransfer> transfers_;
    std::vector<CcuTaskBinding> ccu_bindings_;

public:
    using Base::add_dep;
    using Base::add_dep_wait;
    using Base::add_scalar;
    using Base::explicit_dep;
    using Base::explicit_dep_count;
    using Base::explicit_dep_retain;
    using Base::scalar;
    using Base::scalar_count;
    using Base::scalar_data;
    using Base::tag;
    using Base::tensor;
    using Base::tensor_count;
    using Base::tensor_data;

    void add_tensor(const Tensor &t, TensorArgType tag = TensorArgType::INPUT) {
        add_tensor(t, tag, legacy_tensor_transfer(static_cast<AddressSpace>(t.buffer.address_space)));
    }

    void add_tensor(const Tensor &t, TensorArgType tag, TensorTransfer transfer) {
        if (const char *error = tensor_transfer_error(static_cast<AddressSpace>(t.buffer.address_space), transfer)) {
            throw std::invalid_argument(error);
        }
        if (!scalars_.empty()) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        const Tensor value = t;  // t may refer to an existing entry invalidated by reserve.
        // Every append below is non-allocating, so allocation failure cannot separate a view
        // from its tag or request. Growth remains geometric for large host argument lists.
        if (tensors_.size() == tensors_.capacity() || tensors_.size() == tags_.capacity() ||
            tensors_.size() == transfers_.capacity()) {
            const size_t capacity = (tensors_.size() + 1) * 2;
            tensors_.reserve(capacity);
            tags_.reserve(capacity);
            transfers_.reserve(capacity);
        }
        Base::add_tensor(value, tag);
        transfers_.push_back(transfer);
    }

    TensorTransfer transfer(int32_t i) const {
        const auto request = transfers_.at(static_cast<size_t>(i));
        if (const char *error =
                tensor_transfer_error(static_cast<AddressSpace>(tensor(i).buffer.address_space), request)) {
            throw std::invalid_argument(error);
        }
        return request;
    }

    const std::vector<CcuTaskBinding> &ccu_bindings() const { return ccu_bindings_; }
    void set_ccu_bindings(std::vector<CcuTaskBinding> bindings) {
        if (bindings.size() > UINT32_MAX) throw std::invalid_argument("too many CCU bindings");
        for (const auto &binding : bindings) {
            if (!binding.registration_id || binding.reserved)
                throw std::invalid_argument("invalid CCU binding identity");
        }
        ccu_bindings_ = std::move(bindings);
    }

    void clear() {
        ccu_bindings_.clear();
        Base::clear();
        transfers_.clear();
    }
};

// The mailbox argument slot reserves one byte of the view padding for the invocation request.
inline constexpr size_t TASK_ARG_TRANSFER_OFFSET = offsetof(Tensor, _pad);
static_assert(TASK_ARG_TRANSFER_OFFSET == 141, "mailbox transfer offset is wire ABI");

// ============================================================================
// TaskArgsView — zero-copy view over a wire blob
// ============================================================================
//
// View-only: refers to externally owned tensor + scalar arrays. No tags
// (tags are consumed by Orchestrator at submit time and never travel further).

struct TaskArgsView {
    int32_t tensor_count;
    int32_t scalar_count;
    // Raw bytes of the tensor array, NOT a `const Tensor *`. The blob's tensor region starts at the
    // 8-byte header boundary, so a `Tensor *` formed onto it would carry an alignment the type does
    // not promise. Copy a tensor out with tensors(i).
    const uint8_t *tensor_bytes;
    const uint64_t *scalars;  // 8-byte aligned by blob construction; safe to address as uint64_t*

    uint32_t ccu_binding_count{0};
    const uint8_t *ccu_binding_bytes{nullptr};

    CcuTaskBinding ccu_binding(uint32_t index) const {
        if (index >= ccu_binding_count) throw std::out_of_range("CCU binding index");
        CcuTaskBinding result{};
        std::memcpy(&result, ccu_binding_bytes + index * sizeof(result), sizeof(result));
        if (!result.registration_id || result.reserved) throw std::runtime_error("invalid wire CCU binding");
        return result;
    }

    // Copy the i-th tensor into a properly-aligned local and gate it. Bounds-checked: a negative
    // index would otherwise wrap to a huge offset once cast to size_t. This is the ONLY validation
    // a blob element ever gets — nothing downstream re-checks magic, tag, body_len, the view's
    // containment in its backing, or the FORK_COW read-only rule.
    Tensor tensors(int32_t i) const {
        if (i < 0 || i >= tensor_count) {
            throw std::out_of_range("TaskArgsView::tensors: index out of range");
        }
        Tensor t;
        std::memcpy(&t, tensor_bytes + static_cast<size_t>(i) * sizeof(Tensor), sizeof(Tensor));
        t._pad[0] = 0;
        validate_tensor(t);
        return t;
    }

    TensorTransfer transfer(int32_t i) const {
        const Tensor t = tensors(i);
        const auto request = static_cast<TensorTransfer>(
            tensor_bytes[static_cast<size_t>(i) * sizeof(Tensor) + TASK_ARG_TRANSFER_OFFSET]
        );
        if (const char *error = tensor_transfer_error(static_cast<AddressSpace>(t.buffer.address_space), request)) {
            throw std::invalid_argument(error);
        }
        return request;
    }
};

// ============================================================================
// Wire format — length-prefixed blob for PROCESS-mode mailbox transport
// ============================================================================
//
// Byte layout (tags stripped):
//   offset 0:                 int32 tensor_count = T
//   offset 4:                 int32 scalar_count = S
//   offset 8:                 argument slots[T]             (144 B each; view + transfer at byte 141)
//   offset 8 + 144T:          uint64_t scalars[S]           (8 B each)
// total bytes used:           8 + 144T + 8S
//
// Each slot carries the self-describing `Tensor` view and its invocation transfer, so a
// consumer resolves it with no prior handshake. A chip child materializes each one to a
// `ChipTensor` (address-bearing) and assembles a `ChipStorageTaskArgs` for the runtime.so ABI.

inline constexpr size_t TASK_ARGS_BLOB_HEADER_SIZE = 8;

// CCU submissions carry a versioned envelope; ordinary submissions retain the 8-byte header.
inline constexpr int32_t TASK_ARGS_CCU_MAGIC = INT32_MIN;
inline constexpr uint32_t TASK_ARGS_CCU_VERSION = 1;
inline constexpr size_t TASK_ARGS_CCU_HEADER_SIZE = 24;

inline size_t task_args_blob_size(const TaskArgs &a) {
    return (a.ccu_bindings().empty() ? TASK_ARGS_BLOB_HEADER_SIZE : TASK_ARGS_CCU_HEADER_SIZE) +
           static_cast<size_t>(a.tensor_count()) * sizeof(Tensor) +
           static_cast<size_t>(a.scalar_count()) * sizeof(uint64_t) + a.ccu_bindings().size() * sizeof(CcuTaskBinding);
}

// Serialize a TaskArgs into `dst`. Caller must ensure `dst` has room for
// task_args_blob_size(a) bytes. Tags are not written.
inline void write_blob(uint8_t *dst, const TaskArgs &a) {
    const int32_t T = a.tensor_count();
    const int32_t S = a.scalar_count();
    const bool ccu = !a.ccu_bindings().empty();
    const size_t header = ccu ? TASK_ARGS_CCU_HEADER_SIZE : TASK_ARGS_BLOB_HEADER_SIZE;
    if (ccu) {
        const uint32_t count = static_cast<uint32_t>(a.ccu_bindings().size());
        const uint32_t reserved = 0;
        std::memcpy(dst, &TASK_ARGS_CCU_MAGIC, 4);
        std::memcpy(dst + 4, &TASK_ARGS_CCU_VERSION, 4);
        std::memcpy(dst + 8, &T, 4);
        std::memcpy(dst + 12, &S, 4);
        std::memcpy(dst + 16, &count, 4);
        std::memcpy(dst + 20, &reserved, 4);
    } else {
        std::memcpy(dst, &T, 4);
        std::memcpy(dst + 4, &S, 4);
    }
    if (T > 0) {
        std::memcpy(dst + header, a.tensor_data(), static_cast<size_t>(T) * sizeof(Tensor));
        for (int32_t i = 0; i < T; ++i)
            dst[header + static_cast<size_t>(i) * sizeof(Tensor) + TASK_ARG_TRANSFER_OFFSET] =
                static_cast<uint8_t>(a.transfer(i));
    }
    const size_t scalar_offset = header + static_cast<size_t>(T) * sizeof(Tensor);
    if (S > 0) std::memcpy(dst + scalar_offset, a.scalar_data(), static_cast<size_t>(S) * sizeof(uint64_t));
    if (ccu)
        std::memcpy(
            dst + scalar_offset + static_cast<size_t>(S) * sizeof(uint64_t), a.ccu_bindings().data(),
            a.ccu_bindings().size() * sizeof(CcuTaskBinding)
        );
}

// Zero-copy view into a blob written by write_blob. The returned view is only
// valid as long as `src` stays alive in mapped/shm memory.
//
// `capacity` is the maximum number of bytes the reader is allowed to consume
// from `src` (e.g. MAILBOX_ARGS_CAPACITY when reading from the IPC mailbox).
// Throws std::runtime_error if the header reports counts that would walk past
// `capacity` — defends against shared-memory corruption or a writer-side bug
// that slipped past the writer's own bounds check. This bounds the envelope
// only; each element is gated by TaskArgsView::tensors.
inline TaskArgsView read_blob(const uint8_t *src, size_t capacity) {
    if (capacity < TASK_ARGS_BLOB_HEADER_SIZE) {
        throw std::runtime_error(
            "read_blob: capacity " + std::to_string(capacity) + " < header size " +
            std::to_string(TASK_ARGS_BLOB_HEADER_SIZE)
        );
    }
    int32_t T;
    int32_t S;
    std::memcpy(&T, src + 0, sizeof(T));
    std::memcpy(&S, src + 4, sizeof(S));
    size_t header = TASK_ARGS_BLOB_HEADER_SIZE;
    uint32_t bindings = 0;
    if (T == TASK_ARGS_CCU_MAGIC) {
        if (capacity < TASK_ARGS_CCU_HEADER_SIZE || static_cast<uint32_t>(S) != TASK_ARGS_CCU_VERSION)
            throw std::runtime_error("invalid CCU argument envelope version or size");
        uint32_t reserved = 0;
        std::memcpy(&T, src + 8, 4);
        std::memcpy(&S, src + 12, 4);
        std::memcpy(&bindings, src + 16, 4);
        std::memcpy(&reserved, src + 20, 4);
        if (reserved || !bindings) throw std::runtime_error("invalid CCU argument envelope");
        header = TASK_ARGS_CCU_HEADER_SIZE;
    }
    if (T < 0 || S < 0) {
        throw std::runtime_error(
            "read_blob: negative counts — tensors=" + std::to_string(T) + ", scalars=" + std::to_string(S)
        );
    }
    const size_t needed = header + static_cast<size_t>(T) * sizeof(Tensor) + static_cast<size_t>(S) * sizeof(uint64_t) +
                          static_cast<size_t>(bindings) * sizeof(CcuTaskBinding);
    if (needed > capacity) {
        throw std::runtime_error(
            "read_blob: header reports " + std::to_string(needed) + " bytes (T=" + std::to_string(T) +
            ", S=" + std::to_string(S) + ") but capacity is " + std::to_string(capacity) +
            " — likely shm corruption or a writer-side bug"
        );
    }
    return TaskArgsView{
        T,
        S,
        src + header,
        reinterpret_cast<const uint64_t *>(src + header + static_cast<size_t>(T) * sizeof(Tensor)),
        bindings,
        src + header + static_cast<size_t>(T) * sizeof(Tensor) + static_cast<size_t>(S) * sizeof(uint64_t),
    };
}

// ============================================================================
// Submit-time argument validation
// ============================================================================

// access ⊆ granted: an arg's TensorArgType may only request what the backing grants.
//   INPUT -> READ, OUTPUT_EXISTING -> WRITE, INOUT -> READWRITE; READWRITE grants everything.
//   NO_DEP / OUTPUT are unconstrained.
// Catches e.g. a READ-only copy-on-write backing tagged OUTPUT_EXISTING, whose writes in a forked
// child would silently never reach the parent.
inline bool access_permits(uint8_t granted, TensorArgType tag) {
    auto granted_has = [&](AccessMode need) {
        return granted == static_cast<uint8_t>(AccessMode::READWRITE) || granted == static_cast<uint8_t>(need);
    };
    switch (tag) {
    case TensorArgType::INPUT:
        return granted_has(AccessMode::READ);
    case TensorArgType::OUTPUT_EXISTING:
        return granted_has(AccessMode::WRITE);
    case TensorArgType::INOUT:
        return granted == static_cast<uint8_t>(AccessMode::READWRITE);
    default:
        return true;
    }
}

// Does this tag declare a write? NO_DEP is excluded deliberately: it opts out of dependency
// tracking altogether, so its ordering is the caller's to arrange.
inline bool tag_writes(TensorArgType tag) {
    return tag == TensorArgType::OUTPUT || tag == TensorArgType::OUTPUT_EXISTING || tag == TensorArgType::INOUT;
}

/**
 * Validate one submit's whole argument set, at the point where the values are final.
 *
 * `access ⊆ granted` is re-checked here rather than trusted from add time because a tag is mutable
 * after its element is added — the pair that governs the dispatch is the one present now.
 *
 * Overlapping writes WITHIN one TaskArgs are rejected because no later layer can catch them: the two
 * args belong to one task node, so there is no order between them to express, and a device-staged
 * copy of a host backing does not even alias on the device for the L2 overlap map to notice.
 * Disjoint slices of one backing stay legal.
 *
 * Members of a group are NOT compared against each other. A group is one DAG node whose members
 * deliberately share their tags — naming one buffer as every member's OUTPUT is how a group
 * publishes a single completion token for a downstream task to depend on. Whether such a shared
 * write carries data or only ordering is not visible here, so the caller owns it.
 */
inline void validate_submit_args(const std::vector<TaskArgs> &args_list) {
    for (const TaskArgs &args : args_list) {
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            (void)args.transfer(i);  // Mutable C++ views may have changed location since add_tensor.
            if (!access_permits(args.tensor(i).buffer.access, args.tag(i))) {
                throw std::invalid_argument(
                    "submit: an argument's TensorArgType requests access the backing does not grant"
                );
            }
        }
    }
    for (const TaskArgs &args : args_list) {
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            if (!tag_writes(args.tag(i))) continue;
            for (int32_t j = i + 1; j < args.tensor_count(); ++j) {
                if (!tensors_overlap(args.tensor(i), args.tensor(j))) continue;
                throw std::invalid_argument(
                    "submit: two arguments of one task write overlapping bytes of the same buffer; "
                    "give them disjoint ranges, or order them as separate tasks"
                );
            }
        }
    }
}
