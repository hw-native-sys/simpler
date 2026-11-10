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

#include <cstddef>
#include <cstdint>
#include <type_traits>

enum class CcuStatus : uint32_t {
    OK,
    INVALID_ARGUMENT,
    UNSUPPORTED,
    ABI_MISMATCH,
    STALE_HANDLE,
    BUSY,
    INIT_FAILED,
    EXECUTION_FAILED,
    TIMEOUT,
    RECLAMATION_UNCONFIRMED
};
enum class CcuRequestState : uint32_t { PENDING, SUCCEEDED, FAILED, UNCONFIRMED };
enum class CcuCloseMode : uint32_t { NORMAL, ABORT_UNPUBLISHED };

struct CcuProgramKey {
    uint8_t bytes[32];
};
struct CcuProgramRef {
    uint64_t context_addr;
    uint64_t session_epoch;
    uint64_t binding_offset;
};
struct CcuDeviceContextView {
    uint64_t addr;
    uint64_t size;
    uint32_t abi_version;
    uint32_t reserved;
};
struct CcuRequestRef {
    uint64_t record_addr;
    uint64_t sequence;
};
struct CcuPollResult {
    CcuRequestState state;
    CcuStatus status;
    int32_t native_error;
};

inline constexpr uint32_t CCU_DEVICE_ABI_VERSION = 1;
inline constexpr uint32_t CCU_BUILD_ABI_VERSION = 1;

enum class CcuArgumentKind : uint32_t { TENSOR, SCALAR };
enum class CcuArgumentType : uint32_t { FLOAT32, UINT64, INT64 };
enum class CcuSlotState : uint32_t { FREE, RESERVED, SUBMITTED };
enum class CcuSessionState : uint32_t { READY, QUIESCING, UNCONFIRMED, CLOSED };

// All internal references in this image are byte offsets from CcuDeviceContext.
// The host freezes the image's binding/range tables before acquiring a run lease.
struct alignas(128) CcuDeviceContext {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t image_size;
    uint64_t allocation_id;
    uint64_t session_epoch;
    volatile uint64_t next_sequence;
    uint64_t bindings_offset;
    uint64_t ranges_offset;
    uint64_t record_offset;
    uint64_t arguments_offset;
    uint32_t binding_count;
    uint32_t range_count;
    uint32_t arguments_size;
    volatile CcuSessionState state;
    uint32_t rank;
    uint32_t rank_count;
    uint64_t timeout_ns;
    uint64_t admission_offset;
    uint64_t reserved[2];
};

// Host owns this cache line; device readers never write it back with request state.
struct alignas(128) CcuAdmissionRecord {
    volatile uint64_t closed;
    uint64_t reserved[15];
};

struct CcuProgramBinding {
    CcuProgramKey key;
    uint64_t session_epoch;
    uint64_t signature_offset;
    uint64_t gate_addr;
    uint64_t program_index;
    uint32_t parameter_count;
    uint32_t argument_bytes;
    uint64_t reserved;
};

// One backend parameter layout is shared by reserve and the CCU build prologue.
// Tensor words are address, access token and view bytes; scalar words hold bits.
struct CcuParameterLayout {
    CcuArgumentKind kind;
    CcuArgumentType type;
    uint32_t word_offset;
    uint32_t reserved;
    uint64_t allowed_aliases;
};
struct CcuMemoryRange {
    uint64_t memory_id;
    uint64_t addr;
    uint64_t byte_size;
    uint64_t token;
};

struct alignas(128) CcuRequestRecord {
    uint64_t session_epoch;
    uint64_t sequence;
    uint64_t binding_offset;
    uint64_t record_offset;
    uint64_t program_index;
    uint64_t argument_bytes;
    volatile CcuSlotState slot_state;
    uint32_t reserved0;
    uint64_t submitted_at_ns;
    CcuRequestState local_state;
    CcuStatus local_status;
    uint64_t reserved_control[7];
    // CCU alone publishes this line; completed_sequence is the final commit store.
    volatile uint64_t completed_sequence;
    uint64_t completed_epoch;
    uint64_t completed_program;
    CcuRequestState completed_state;
    CcuStatus status;
    int32_t native_error;
    uint32_t reserved1;
    uint64_t reserved_completion[11];
};

static_assert(sizeof(CcuProgramKey) == 32);
static_assert(sizeof(CcuProgramRef) == 24);
static_assert(sizeof(CcuDeviceContextView) == 24);
static_assert(sizeof(CcuRequestRef) == 16);
static_assert(sizeof(CcuPollResult) == 12);
static_assert(sizeof(CcuDeviceContext) == 128);
static_assert(sizeof(CcuAdmissionRecord) == 128);
static_assert(sizeof(CcuProgramBinding) == 80);
static_assert(sizeof(CcuParameterLayout) == 24);
static_assert(sizeof(CcuMemoryRange) == 32);
static_assert(sizeof(CcuRequestRecord) == 256);
static_assert(offsetof(CcuRequestRecord, completed_sequence) == 128);
static_assert(std::is_standard_layout_v<CcuDeviceContext> && std::is_trivially_copyable_v<CcuDeviceContext>);
static_assert(std::is_standard_layout_v<CcuRequestRecord> && std::is_trivially_copyable_v<CcuRequestRecord>);
static_assert(std::is_standard_layout_v<CcuProgramBinding> && std::is_trivially_copyable_v<CcuProgramBinding>);
static_assert(std::is_standard_layout_v<CcuParameterLayout> && std::is_trivially_copyable_v<CcuParameterLayout>);
static_assert(std::is_standard_layout_v<CcuMemoryRange> && std::is_trivially_copyable_v<CcuMemoryRange>);
static_assert(std::is_standard_layout_v<CcuAdmissionRecord> && std::is_trivially_copyable_v<CcuAdmissionRecord>);
static_assert(std::is_standard_layout_v<CcuProgramKey> && std::is_trivially_copyable_v<CcuProgramKey>);
static_assert(std::is_standard_layout_v<CcuProgramRef> && std::is_trivially_copyable_v<CcuProgramRef>);
static_assert(std::is_standard_layout_v<CcuDeviceContextView> && std::is_trivially_copyable_v<CcuDeviceContextView>);
static_assert(std::is_standard_layout_v<CcuRequestRef> && std::is_trivially_copyable_v<CcuRequestRef>);
static_assert(std::is_standard_layout_v<CcuPollResult> && std::is_trivially_copyable_v<CcuPollResult>);
