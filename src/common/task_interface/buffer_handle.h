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
 * Three types (see .docs/L3/P1-B-buffer-handle-abi.md):
 *   - BufferHandleDescriptor : the owner's exported wire descriptor, sent once per edge in the
 *                              export/import handshake and registered into the consumer's import
 *                              registry. Carries the handle's backing properties (address space,
 *                              visibility, access, nbytes, backend). Version-prefixed: abi_version
 *                              leads so a decoder rejects an unknown version before trusting the rest.
 *   - BufferRef              : the blob-carried view. Holds only the canonical identity (a reference
 *                              to a handle) plus a view (byte_offset, shape, strides, dtype). Carries
 *                              NO materialized address and NO copy of handle properties — the consumer
 *                              resolves those from its import registry by canonical identity.
 *   - CanonicalIdentity      : owner_instance_id + owner_worker_path + buffer_id + generation. The
 *                              key both the owner registry and every consumer import registry use.
 *
 * Every field width, enum value, and offset below is wire ABI: pinned by static_assert and versioned
 * by BUFFER_ABI_VERSION. Unknown version / backend / non-zero reserved is rejected, never
 * silently accepted.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "data_type.h"

// Wire version. A decoder rejects an unknown abi_version rather than misreading a future layout.
inline constexpr uint32_t BUFFER_ABI_VERSION = 1u;

// owner_worker_path bound. Current L4 topology is depth 3 (L4/L3/L2 = at most 2 hops from root);
// 4 leaves one level of headroom. Frozen: P2 must not extend the handle header.
inline constexpr uint32_t MAX_WORKER_PATH_DEPTH = 4;

// Bounded, NUL-terminated backend token (POSIX/fork shm name). POSIX shm names are far shorter.
inline constexpr uint32_t BACKEND_TOKEN_BYTES = 64;

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
// P2 and rejected on decode in P1. INVALID (0) is the zero-value sentinel, never a valid backing.
enum class BackendKind : uint8_t {
    INVALID = 0,
    FORK_SHM = 1,
    POSIX_SHM = 2,
    VMM_WINDOW = 3,
    DEVICE_MALLOC = 4,
    REMOTE_SIDECAR = 5,
};

// BufferRef.flags bitmask (bitwise use — plain enum per codestyle).
enum BufferRefFlag : uint8_t {
    BUFFER_REF_MANUAL_DEP = 1u << 0,
    BUFFER_REF_CONTIGUOUS = 1u << 1,
};

/**
 * Owner's position in the worker tree: `depth` hops from root, `hop[i]` the child index at hop i.
 * depth == 0 is root. Bytes beyond `depth` and the padding are reserved and must be zero.
 */
struct OwnerWorkerPath {
    uint8_t depth;
    uint8_t _reserved[3];
    uint32_t hop[MAX_WORKER_PATH_DEPTH];
};

static_assert(std::is_trivially_copyable_v<OwnerWorkerPath>, "OwnerWorkerPath must be trivially copyable for wire");
static_assert(sizeof(OwnerWorkerPath) == 20, "OwnerWorkerPath is wire ABI (4 + 4*MAX_WORKER_PATH_DEPTH)");
static_assert(offsetof(OwnerWorkerPath, hop) == 4);

/**
 * Canonical allocation identity — globally unique across owner incarnations. `buffer_id` is unique
 * only within one owner incarnation; `owner_instance_id` (a per-incarnation nonce) and
 * `owner_worker_path` disambiguate it, and `generation` detects buffer_id reuse (ABA). The key of
 * both the owner registry and every consumer import registry.
 */
struct CanonicalIdentity {
    uint64_t owner_instance_id;
    uint64_t buffer_id;
    OwnerWorkerPath owner_worker_path;
    uint32_t generation;
};

static_assert(std::is_trivially_copyable_v<CanonicalIdentity>);
static_assert(sizeof(CanonicalIdentity) == 40, "CanonicalIdentity is wire ABI");
static_assert(offsetof(CanonicalIdentity, owner_instance_id) == 0);
static_assert(offsetof(CanonicalIdentity, buffer_id) == 8);
static_assert(offsetof(CanonicalIdentity, owner_worker_path) == 16);
static_assert(offsetof(CanonicalIdentity, generation) == 36);

inline bool operator==(const CanonicalIdentity &a, const CanonicalIdentity &b) {
    return a.owner_instance_id == b.owner_instance_id && a.buffer_id == b.buffer_id && a.generation == b.generation &&
           a.owner_worker_path.depth == b.owner_worker_path.depth &&
           a.owner_worker_path.hop[0] == b.owner_worker_path.hop[0] &&
           a.owner_worker_path.hop[1] == b.owner_worker_path.hop[1] &&
           a.owner_worker_path.hop[2] == b.owner_worker_path.hop[2] &&
           a.owner_worker_path.hop[3] == b.owner_worker_path.hop[3];
}
inline bool operator!=(const CanonicalIdentity &a, const CanonicalIdentity &b) { return !(a == b); }

// Hash for use as an unordered_map key (consumer import registry). Mixes the identity's integer
// fields; the path's reserved bytes are intentionally excluded so padding never perturbs the hash.
struct CanonicalIdentityHash {
    size_t operator()(const CanonicalIdentity &k) const {
        auto mix = [](size_t h, uint64_t v) {
            return (h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)));
        };
        size_t h = 0;
        h = mix(h, k.owner_instance_id);
        h = mix(h, k.buffer_id);
        h = mix(h, k.generation);
        h = mix(h, k.owner_worker_path.depth);
        for (uint32_t i = 0; i < MAX_WORKER_PATH_DEPTH; ++i)
            h = mix(h, k.owner_worker_path.hop[i]);
        return h;
    }
};

/**
 * The blob-carried view: a reference to a handle (canonical identity) plus a strided view onto it.
 *
 * Invariants:
 *   - Carries NO materialized address. The consumer resolves the canonical identity against its
 *     import registry to a local base, then `Tensor.buffer.addr = base`,
 *     `Tensor.start_offset = byte_offset / dtype_bytes`.
 *   - `byte_offset` is a BYTE offset of the view origin and must be a multiple of the dtype size
 *     (validated at materialization; non-aligned byte-views are not supported in this ABI version).
 *   - `strides[i] > 0` strictly (broadcast / negative step unsupported), carried explicitly — a
 *     singleton dimension's stride is never normalized away.
 *   - `reserved` must be zero; a non-zero reserved is rejected on decode.
 */
struct BufferRef {
    CanonicalIdentity identity;
    uint64_t byte_offset;
    uint32_t ndims;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    DataType dtype;
    uint8_t flags;
    uint8_t reserved[2];
};

static_assert(std::is_trivially_copyable_v<BufferRef>, "BufferRef must be trivially copyable for blob memcpy");
static_assert(sizeof(BufferRef) == 96, "BufferRef is wire ABI");
static_assert(offsetof(BufferRef, identity) == 0);
static_assert(offsetof(BufferRef, byte_offset) == 40);
static_assert(offsetof(BufferRef, ndims) == 48);
static_assert(offsetof(BufferRef, shapes) == 52);
static_assert(offsetof(BufferRef, strides) == 72);
static_assert(offsetof(BufferRef, dtype) == 92);
static_assert(offsetof(BufferRef, flags) == 93);

/**
 * The owner's exported handle descriptor — the export/import handshake payload. Registered once per
 * edge into the consumer's import registry, keyed by canonical identity, valued by the local
 * materialization the consumer derives from `backend_kind` + `token`/`backend_handle`.
 *
 * Version-prefixed: `abi_version` leads so a decoder rejects an unknown version before trusting the
 * rest. `token` is a NUL-terminated shm name for FORK_SHM/POSIX_SHM ("" else); `backend_handle` is a
 * VMM shareable-handle / device pointer for VMM_WINDOW/DEVICE_MALLOC (0 else). Adding a field is an
 * abi_version bump (no reserved slot).
 */
struct BufferHandleDescriptor {
    uint32_t abi_version;
    AddressSpace address_space;
    Visibility visibility;
    AccessMode access;
    BackendKind backend_kind;
    CanonicalIdentity identity;
    uint64_t nbytes;
    uint64_t backend_handle;
    char token[BACKEND_TOKEN_BYTES];
};

static_assert(std::is_trivially_copyable_v<BufferHandleDescriptor>);
static_assert(sizeof(BufferHandleDescriptor) == 128, "BufferHandleDescriptor is wire ABI");
static_assert(offsetof(BufferHandleDescriptor, abi_version) == 0);
static_assert(offsetof(BufferHandleDescriptor, address_space) == 4);
static_assert(offsetof(BufferHandleDescriptor, visibility) == 5);
static_assert(offsetof(BufferHandleDescriptor, access) == 6);
static_assert(offsetof(BufferHandleDescriptor, backend_kind) == 7);
static_assert(offsetof(BufferHandleDescriptor, identity) == 8);
static_assert(offsetof(BufferHandleDescriptor, nbytes) == 48);
static_assert(offsetof(BufferHandleDescriptor, backend_handle) == 56);
static_assert(offsetof(BufferHandleDescriptor, token) == 64);
