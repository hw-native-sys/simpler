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

/**
 * BufferHandle / BufferRef ABI — typed, versioned, opaque cross-layer buffer identity.
 *
 * Layout implements the frozen logical schema in
 * .docs/L3-new/worker-memory-model/bufferhandle-abi.md (that doc is authoritative for field set /
 * widths / enum values / endianness / evolution; exact byte offsets here are this P1 wire's choice).
 *
 * Three types:
 *   - BufferHandleDescriptor : the owner's self-describing wire descriptor. Carries backing
 *                              properties + a versioned length-delimited backend body. abi_version
 *                              (u16) leads so a decoder rejects an unknown version before trusting the
 *                              rest. Embedded whole in every BufferRef built over the handle.
 *   - BufferRef              : the blob-carried wire element. Embeds the full BufferHandleDescriptor
 *                              plus a view (byte_offset, shape, strides, dtype) — self-describing, so
 *                              a consumer materializes it lazily on receipt with no prior handshake.
 *                              No materialized address.
 *   - CanonicalIdentity      : owner_instance_id + owner_worker_path + buffer_id + generation. The
 *                              key both the owner registry and every consumer import cache use.
 *
 * Endianness: all multi-byte integers little-endian. owner_instance_id is an opaque byte sequence
 * (bytewise-compared, no integer/endianness meaning). Unknown version / backend / descriptor_version
 * is rejected, never silently accepted.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "data_type.h"

// Wire version of the handle schema (u16). A decoder rejects an unknown abi_version rather than
// misreading a future layout. 0 is reserved (illegal).
inline constexpr uint16_t BUFFER_ABI_VERSION = 1;

// Version of a backend_descriptor body. Unknown descriptor_version is rejected like abi_version.
inline constexpr uint8_t BUFFER_DESCRIPTOR_VERSION = 1;

// owner_instance_id is a fixed-width opaque nonce (compared bytewise; no integer/endianness meaning).
inline constexpr uint32_t OWNER_INSTANCE_ID_BYTES = 16;

// Bounded length-delimited limits (single constants; revisit before the final ABI freeze).
// owner_worker_path is a UTF-8 tree path like "L4/L3[2]/L2[5]"; backend body is per-backend.
inline constexpr uint32_t PATH_MAX_BYTES = 64;
inline constexpr uint32_t DESC_MAX_BYTES = 96;

// AddressSpace (HOST/DEVICE) is shared with Tensor and lives in data_type.h.

// Which workers may see a backing. Single-hop/multi-hop visibility is explicit, not by convention.
enum class Visibility : uint8_t {
    PRIVATE = 0,
    SHARED = 1,
};

// The backing's granted permission. A per-arg TensorArgType requests read/write and is validated
// against this at submit (requested must be a subset of granted).
enum class AccessMode : uint8_t {
    READ = 0,
    WRITE = 1,
    READWRITE = 2,
};

// Materialization backend of a handle. The consumer resolves a BufferRef to a local address via the
// import registry keyed by canonical identity; this tag selects how. REMOTE_SIDECAR is reserved for
// P2 and rejected on decode in P1. Values are frozen; 5.. reserved (unknown tag => reject).
enum class BackendKind : uint8_t {
    FORK_SHM = 0,
    POSIX_SHM = 1,
    VMM_WINDOW = 2,
    REMOTE_SIDECAR = 3,
    DEVICE_MALLOC = 4,
};

/**
 * Canonical allocation identity — globally unique across owner incarnations, unchanged across every
 * edge. `buffer_id` is unique only within one owner incarnation; `owner_instance_id` (a 16-byte
 * per-incarnation nonce) and `owner_worker_path` disambiguate it, and `generation` detects buffer_id
 * reuse (ABA). The key of both the owner registry and every consumer import registry.
 *
 * `owner_worker_path` is a bounded length-delimited UTF-8 tree path ("L4/L3[2]/L2[5]"): `path_len`
 * valid bytes in `owner_worker_path[0, path_len)`; bytes beyond `path_len` and `_pad` are zero.
 */
struct CanonicalIdentity {
    uint8_t owner_instance_id[OWNER_INSTANCE_ID_BYTES];
    uint64_t buffer_id;
    uint32_t generation;
    uint16_t path_len;
    uint8_t _pad[2];
    char owner_worker_path[PATH_MAX_BYTES];
};

static_assert(std::is_trivially_copyable_v<CanonicalIdentity>);
static_assert(sizeof(CanonicalIdentity) == 96, "CanonicalIdentity is wire ABI");
static_assert(offsetof(CanonicalIdentity, owner_instance_id) == 0);
static_assert(offsetof(CanonicalIdentity, buffer_id) == 16);
static_assert(offsetof(CanonicalIdentity, generation) == 24);
static_assert(offsetof(CanonicalIdentity, path_len) == 28);
static_assert(offsetof(CanonicalIdentity, owner_worker_path) == 32);

inline bool operator==(const CanonicalIdentity &a, const CanonicalIdentity &b) {
    return a.buffer_id == b.buffer_id && a.generation == b.generation && a.path_len == b.path_len &&
           std::memcmp(a.owner_instance_id, b.owner_instance_id, OWNER_INSTANCE_ID_BYTES) == 0 &&
           std::memcmp(a.owner_worker_path, b.owner_worker_path, a.path_len) == 0;
}
inline bool operator!=(const CanonicalIdentity &a, const CanonicalIdentity &b) { return !(a == b); }

// Hash for use as an unordered_map key (consumer import registry). Folds the significant identity
// bytes; unused path bytes past path_len are excluded so they never perturb the hash.
struct CanonicalIdentityHash {
    size_t operator()(const CanonicalIdentity &k) const {
        auto mix = [](size_t h, uint64_t v) {
            return (h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)));
        };
        size_t h = 0;
        for (uint32_t i = 0; i < OWNER_INSTANCE_ID_BYTES; ++i)
            h = mix(h, k.owner_instance_id[i]);
        h = mix(h, k.buffer_id);
        h = mix(h, k.generation);
        h = mix(h, k.path_len);
        for (uint16_t i = 0; i < k.path_len; ++i)
            h = mix(h, static_cast<uint8_t>(k.owner_worker_path[i]));
        return h;
    }
};

/**
 * The owner's self-describing handle descriptor — embedded whole in every BufferRef built over the
 * handle. A consumer materializes it lazily on first receipt (no separate export handshake) and
 * caches `canonical identity -> local base` (map-once). `backend_kind` + `descriptor_version` +
 * `body[0, body_len)` carry the per-backend materialization (POSIX/fork shm name, VMM
 * shareable-handle, device VA, ...). Version-prefixed (`abi_version` u16 leads); unknown abi_version /
 * backend_kind / descriptor_version is rejected before trusting the rest. `address_space` /
 * `visibility` / `access` / `backend_kind` are raw u8 so an unknown value can be rejected without
 * invoking undefined enum behavior.
 */
struct BufferHandleDescriptor {
    uint16_t abi_version;
    uint8_t address_space;
    uint8_t visibility;
    uint8_t access;
    uint8_t backend_kind;
    uint8_t descriptor_version;
    uint8_t _pad0;
    CanonicalIdentity identity;
    uint64_t nbytes;
    uint16_t body_len;
    uint8_t _pad1[6];
    char body[DESC_MAX_BYTES];
};

static_assert(std::is_trivially_copyable_v<BufferHandleDescriptor>);
static_assert(sizeof(BufferHandleDescriptor) == 216, "BufferHandleDescriptor is wire ABI");
static_assert(offsetof(BufferHandleDescriptor, abi_version) == 0);
static_assert(offsetof(BufferHandleDescriptor, address_space) == 2);
static_assert(offsetof(BufferHandleDescriptor, visibility) == 3);
static_assert(offsetof(BufferHandleDescriptor, access) == 4);
static_assert(offsetof(BufferHandleDescriptor, backend_kind) == 5);
static_assert(offsetof(BufferHandleDescriptor, descriptor_version) == 6);
static_assert(offsetof(BufferHandleDescriptor, identity) == 8);
static_assert(offsetof(BufferHandleDescriptor, nbytes) == 104);
static_assert(offsetof(BufferHandleDescriptor, body_len) == 112);
static_assert(offsetof(BufferHandleDescriptor, body) == 120);

/**
 * The blob-carried, self-describing wire element: a full embedded handle descriptor plus a strided
 * view onto it. Because the descriptor travels with the ref, a consumer needs no prior handshake —
 * it materializes the embedded `handle` (backend selects how) on first receipt, keyed by
 * `handle.identity`, and reuses the cached base for later refs to the same identity.
 *
 * Invariants:
 *   - Carries NO materialized address. The consumer materializes `handle` to a local base, then
 *     `Tensor.buffer.addr = base`, `Tensor.start_offset = byte_offset / dtype_bytes`.
 *   - `byte_offset` is a BYTE offset of the view origin; a multiple of the dtype size (validated at
 *     materialization).
 *   - `strides[i] > 0` strictly (broadcast / negative step unsupported), carried explicitly — a
 *     singleton dimension's stride is never normalized away.
 */
struct BufferRef {
    BufferHandleDescriptor handle;
    uint64_t byte_offset;
    uint32_t ndims;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    DataType dtype;
    uint8_t _pad[3];
};

static_assert(std::is_trivially_copyable_v<BufferRef>, "BufferRef must be trivially copyable for blob memcpy");
static_assert(sizeof(BufferRef) == 272, "BufferRef is wire ABI");
static_assert(offsetof(BufferRef, handle) == 0);
static_assert(offsetof(BufferRef, byte_offset) == 216);
static_assert(offsetof(BufferRef, ndims) == 224);
static_assert(offsetof(BufferRef, shapes) == 228);
static_assert(offsetof(BufferRef, strides) == 248);
static_assert(offsetof(BufferRef, dtype) == 268);
