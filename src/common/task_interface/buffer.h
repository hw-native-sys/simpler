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
 * Buffer / Tensor ABI — typed, opaque cross-layer buffer identity.
 *
 * Three types:
 *   - CanonicalIdentity      : owner_instance_id + buffer_id + generation. The key both the owner
 *                              registry and every consumer import cache use, invariant across every
 *                              edge. Fixed-length with no length field, so hashing and comparison
 *                              cannot read past it.
 *   - BufferDescriptor       : the owner's self-describing wire descriptor — backing properties plus
 *                              a length-delimited backend body. Embedded whole in every Tensor
 *                              built over the buffer.
 *   - Tensor                 : the argument an L3+ TaskArgs carries, and the element its mailbox
 *                              blob transports. Embeds the full BufferDescriptor plus a view
 *                              (byte_offset, shape, strides, dtype) — self-describing, so a consumer
 *                              materializes it lazily on receipt with no prior handshake. Carries no
 *                              address; the consuming endpoint materializes it into the
 *                              GM-address-bearing `ChipTensor` of tensor.h, the distinct device POD
 *                              the L2 runtime reads.
 *
 * The blob that carries these lives in task_args.h — it is the existing TaskArgs mailbox format,
 * with Tensor as its element. This header owns the types and the gate every receive boundary runs,
 * not their transport.
 *
 * There is no wire version. Every endpoint of a run comes from one `pip install`, so a version field
 * would guard a skew that cannot arise; the skew that CAN arise (a stale compiled extension against
 * newer Python) is caught by SIMPLER_BUILD_COMMIT. The descriptor's leading `magic` is a
 * discriminator against non-descriptor bytes, not a version.
 *
 * Endianness: all multi-byte integers little-endian. owner_instance_id is an opaque byte sequence
 * (bytewise-compared, no integer/endianness meaning). An unknown backend / address_space / access
 * value is rejected, never silently accepted.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "data_type.h"

// Leading sentinel of a BufferDescriptor, NOT a version. A descriptor decoder needs a cheap
// leading discriminator because `TaskArgs.add_tensor` accepts raw bytes; the sentinel rejects most
// non-descriptor input before any field is trusted. It does not by itself prove the bytes are a
// descriptor — every other field is still validated. There is no multi-version wire: every endpoint
// of a run is built from one `pip install`, and build skew is caught by SIMPLER_BUILD_COMMIT.
inline constexpr uint16_t BUFFER_DESCRIPTOR_MAGIC = 0x5342;  // 'BS' little-endian

// owner_instance_id is a fixed-width opaque nonce (compared bytewise; no integer/endianness meaning).
// It is the SOLE source of cross-incarnation uniqueness, so it must be generated with a full-width
// random draw — a structured value (timestamp/pid) collides between two Workers built in the same
// process and second.
inline constexpr uint32_t OWNER_INSTANCE_ID_BYTES = 8;

// Backend body upper bound. POSIX_SHM uses a shm name; VMM_SHAREABLE uses a fixed 24-byte overlay;
// every address-bearing backend stores a single u64 address.
inline constexpr uint32_t DESC_MAX_BYTES = 32;

// Body width of every address-bearing backend: one u64 little-endian base. Exact, not a maximum —
// a shorter body reads as a truncated pointer indistinguishable from a real one.
inline constexpr uint32_t BACKEND_ADDRESS_BODY_BYTES = 8;

// VMM_SHAREABLE body: int32 device_id, uint32 reserved 0, uint64 shareable_handle, uint64 mapping_bytes.
inline constexpr uint32_t VMM_SHAREABLE_BODY_BYTES = 24;

// The backing's granted permission. A per-arg TensorArgType requests read/write and is validated
// against this at submit (requested must be a subset of granted).
enum class AccessMode : uint8_t {
    READ = 0,
    WRITE = 1,
    READWRITE = 2,
};

// Materialization backend of a buffer. The consumer resolves a Tensor to a local address via the
// import registry keyed by canonical identity; this tag selects how. REMOTE_SIDECAR is reserved for
// P2 and rejected on decode in P1. Values 0–5 are frozen; VMM_SHAREABLE is 6; 7.. reserved
// (unknown tag => reject).
//
// FORK_SHM and FORK_COW materialize identically — the body is a base VA the child already has,
// inherited across the fork — but the kernel's write semantics are opposite, so they are separate
// tags rather than one tag plus a hint. A child's write to a MAP_SHARED page lands in the physical
// page the parent reads; a write to a copy-on-write page splits it into a private copy the parent
// never sees, silently. FORK_COW therefore grants READ only, and that is enforced on decode.
//
// VMM_WINDOW and VMM_SHAREABLE are distinct tags: the former's body is an already-valid owner-chip
// device VA that cannot leave that context, while the latter carries a shareable handle an importer
// maps for itself.
enum class BackendKind : uint8_t {
    FORK_SHM = 0,
    POSIX_SHM = 1,
    VMM_WINDOW = 2,
    REMOTE_SIDECAR = 3,
    DEVICE_MALLOC = 4,
    FORK_COW = 5,
    VMM_SHAREABLE = 6,
};

/**
 * Canonical allocation identity — globally unique across owner incarnations, unchanged across every
 * edge. `buffer_id` is unique only within one owner incarnation; `owner_instance_id` (a per-incarnation
 * nonce) disambiguates it, and `generation` detects buffer_id slot reuse (ABA). The key of both the
 * owner registry and every consumer import registry.
 *
 * FIXED-LENGTH BY DESIGN: no field here bounds a read. Hashing and comparison therefore cannot run
 * off the end of the struct whatever bytes arrive on the wire — the property is structural, not
 * something a validator has to enforce.
 *
 * `_pad` is excluded from comparison and hashing, so a decoded identity with dirty padding still
 * matches the same backing (two views of one backing must never key differently).
 *
 * `generation` starts at 1; every reuse of a `buffer_id` slot increments it. 0 is reserved to mean
 * uninitialized and is rejected on decode.
 */
struct CanonicalIdentity {
    uint8_t owner_instance_id[OWNER_INSTANCE_ID_BYTES];
    uint64_t buffer_id;
    uint32_t generation;
    uint8_t _pad[12];
};

static_assert(std::is_trivially_copyable_v<CanonicalIdentity>);
static_assert(sizeof(CanonicalIdentity) == 32, "CanonicalIdentity is wire ABI");
static_assert(offsetof(CanonicalIdentity, owner_instance_id) == 0);
static_assert(offsetof(CanonicalIdentity, buffer_id) == 8);
static_assert(offsetof(CanonicalIdentity, generation) == 16);

inline bool operator==(const CanonicalIdentity &a, const CanonicalIdentity &b) {
    return a.buffer_id == b.buffer_id && a.generation == b.generation &&
           std::memcmp(a.owner_instance_id, b.owner_instance_id, OWNER_INSTANCE_ID_BYTES) == 0;
}
inline bool operator!=(const CanonicalIdentity &a, const CanonicalIdentity &b) { return !(a == b); }

// Hash for use as an unordered_map key (consumer import registry). Folds exactly the fields
// `operator==` compares — `_pad` is excluded so padding can never perturb the bucket.
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
        return h;
    }
};

/**
 * The owner's self-describing buffer descriptor — embedded whole in every Tensor built over the
 * buffer. A consumer materializes it lazily on first receipt (no separate export handshake) and
 * caches `canonical identity -> local base` (map-once). `backend_kind` + `body[0, body_len)` carry
 * the per-backend materialization (POSIX shm name, fork-inherited VA, device VA, ...).
 * `magic` leads as a cheap discriminator; `address_space` / `access` / `backend_kind` are raw u8 so
 * an unknown value can be rejected without invoking undefined enum behavior.
 *
 * `owner_worker_path_id` is a DIAGNOSTIC id only — it names the owning worker in logs and
 * post-mortems and takes part in no routing, visibility or identity decision. Its side table lives in
 * the owning process; an id a consumer cannot resolve prints as `<path#N>` and is never an error.
 */
struct BufferDescriptor {
    uint16_t magic;
    uint8_t address_space;
    uint8_t access;
    uint8_t backend_kind;
    uint8_t _pad0[3];
    CanonicalIdentity identity;
    uint64_t nbytes;
    uint32_t owner_worker_path_id;
    uint16_t body_len;
    uint8_t _pad1[2];
    char body[DESC_MAX_BYTES];
};

static_assert(std::is_trivially_copyable_v<BufferDescriptor>);
static_assert(sizeof(BufferDescriptor) == 88, "BufferDescriptor is wire ABI");
static_assert(offsetof(BufferDescriptor, magic) == 0);
static_assert(offsetof(BufferDescriptor, address_space) == 2);
static_assert(offsetof(BufferDescriptor, access) == 3);
static_assert(offsetof(BufferDescriptor, backend_kind) == 4);
static_assert(offsetof(BufferDescriptor, identity) == 8);
static_assert(offsetof(BufferDescriptor, nbytes) == 40);
static_assert(offsetof(BufferDescriptor, owner_worker_path_id) == 48);
static_assert(offsetof(BufferDescriptor, body_len) == 52);
static_assert(offsetof(BufferDescriptor, body) == 56);

// Field-wise, so neither the struct padding nor the unused tail of `body` takes part: two decodes of
// one backing describe the same buffer whatever bytes sit outside the fields that carry meaning.
// `body_len` is clamped rather than trusted: this is the one length field in the header, and an
// unvalidated descriptor reaching a comparison must not be able to bound a read past `body`.
inline bool operator==(const BufferDescriptor &a, const BufferDescriptor &b) {
    const size_t body_len = a.body_len < DESC_MAX_BYTES ? a.body_len : DESC_MAX_BYTES;
    return a.identity == b.identity && a.address_space == b.address_space && a.access == b.access &&
           a.backend_kind == b.backend_kind && a.nbytes == b.nbytes &&
           a.owner_worker_path_id == b.owner_worker_path_id && a.body_len == b.body_len &&
           std::memcmp(a.body, b.body, body_len) == 0;
}
inline bool operator!=(const BufferDescriptor &a, const BufferDescriptor &b) { return !(a == b); }

/**
 * The blob-carried, self-describing wire element: a full embedded buffer descriptor plus a strided
 * view onto it. Because the descriptor travels with the tensor, a consumer needs no prior handshake
 * — it materializes the embedded `buffer` (backend selects how) on first receipt, keyed by
 * `buffer.identity`, and reuses the cached base for later tensors over the same identity.
 *
 * Invariants:
 *   - Carries NO materialized address. The consumer materializes `buffer` to a local base, then
 *     `Tensor.buffer.addr = base`, `Tensor.start_offset = byte_offset / dtype_bytes`.
 *   - `byte_offset` is a BYTE offset of the view origin; a multiple of the dtype size (validated at
 *     materialization).
 *   - `strides[i] > 0` strictly (broadcast / negative step unsupported), carried explicitly — a
 *     singleton dimension's stride is never normalized away.
 */
struct Tensor {
    BufferDescriptor buffer;
    uint64_t byte_offset;
    uint32_t ndims;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    DataType dtype;
    uint8_t _pad[3];
};

// Saturating u64 arithmetic. Every input below is wire-supplied, and `shapes[i]` and `strides[i]`
// are each u32, so one product alone reaches ~2^64: an unsaturated sum would wrap to a SMALL extent
// and pass the bound check in validate_tensor while the view really spans exabytes.
inline uint64_t tensor_mul_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_mul_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return (a != 0 && b > UINT64_MAX / a) ? UINT64_MAX : a * b;
#endif
}
inline uint64_t tensor_add_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_add_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
#endif
}

// Sentinel `tensor_extent_bytes` returns when a view's extent does not fit 64 bits. No real backing
// is 16 EiB, so a Tensor whose extent lands here describes no representable view and is rejected.
inline constexpr uint64_t TENSOR_EXTENT_UNREPRESENTABLE = UINT64_MAX;

// Byte extent of a (possibly strided) view: the last addressable element, plus one element.
// Saturates at TENSOR_EXTENT_UNREPRESENTABLE rather than wrapping, so an extent a hostile
// shape/stride cannot express is rejected instead of comparing as a small, in-bounds value.
// Callers that have not yet validated the view must not trust the result for anything but a bound.
inline uint64_t view_extent_bytes(const uint32_t *shapes, const uint32_t *strides, uint32_t ndims, DataType dtype) {
    uint64_t last_elem = 0;
    for (uint32_t i = 0; i < ndims && i < static_cast<uint32_t>(MAX_TENSOR_DIMS); ++i) {
        if (shapes[i] == 0) continue;
        last_elem = tensor_add_sat(
            last_elem, tensor_mul_sat(static_cast<uint64_t>(shapes[i] - 1), static_cast<uint64_t>(strides[i]))
        );
    }
    return tensor_mul_sat(tensor_add_sat(last_elem, 1), get_element_size(dtype));
}

inline uint64_t tensor_extent_bytes(const Tensor &r) {
    return view_extent_bytes(r.shapes, r.strides, r.ndims, r.dtype);
}

/**
 * Reject any BufferDescriptor whose fields are not self-consistent, BEFORE any of them is trusted.
 *
 * The descriptor half of the tensor gate below, split out because a descriptor also arrives on its
 * own — construction from Python and blob descriptor extraction both hand one over with no view
 * attached. `body_len` is bounded against `DESC_MAX_BYTES` here, mirroring what the fixed-length
 * `CanonicalIdentity` gets structurally, and the body itself is checked against the schema its
 * `backend_kind` implies. Throws `std::invalid_argument` naming the field.
 *
 * `REMOTE_SIDECAR` is a legal wire value and passes: an arg bound for a remote worker rides the wire
 * with no local backing, and it is *materialization* that refuses it in P1.
 */
inline void validate_buffer_descriptor(const BufferDescriptor &h) {
    auto reject = [](const char *what) {
        throw std::invalid_argument(what);
    };

    if (h.magic != BUFFER_DESCRIPTOR_MAGIC) reject("invalid BufferDescriptor: magic");
    if (h.address_space > static_cast<uint8_t>(AddressSpace::DEVICE))
        reject("invalid BufferDescriptor: address_space out of range");
    if (h.access > static_cast<uint8_t>(AccessMode::READWRITE)) reject("invalid BufferDescriptor: access out of range");
    if (h.backend_kind > static_cast<uint8_t>(BackendKind::VMM_SHAREABLE))
        reject("invalid BufferDescriptor: backend_kind out of range");
    if (h.body_len > DESC_MAX_BYTES) reject("invalid BufferDescriptor: body_len exceeds DESC_MAX_BYTES");
    if (h.identity.generation == 0) reject("invalid BufferDescriptor: generation 0 is reserved (uninitialized)");

    // address_space x backend_kind capability gate. REMOTE_SIDECAR is legal in either space.
    const auto backend = static_cast<BackendKind>(h.backend_kind);
    const bool device_space = h.address_space == static_cast<uint8_t>(AddressSpace::DEVICE);
    if (backend != BackendKind::REMOTE_SIDECAR) {
        const bool device_backend = backend == BackendKind::VMM_WINDOW || backend == BackendKind::DEVICE_MALLOC ||
                                    backend == BackendKind::VMM_SHAREABLE;
        if (device_backend != device_space)
            reject("invalid BufferDescriptor: unsupported address_space x backend_kind (capability matrix)");
    }

    // A copy-on-write page splits on the consumer's first write into a private copy the owner never
    // sees, so a write grant over FORK_COW would be silently unobservable rather than an error.
    if (backend == BackendKind::FORK_COW && h.access != static_cast<uint8_t>(AccessMode::READ)) {
        reject("invalid BufferDescriptor: FORK_COW grants READ only (a child's write would not reach the owner)");
    }

    // Per-backend body schema. `backend_kind` says how the consumer reads `body`, so a body that
    // does not fit the reading is not a smaller backing — it is a different value. A DEVICE_MALLOC
    // body of 3 bytes reads as a TRUNCATED pointer with nothing to distinguish it from a real one.
    switch (backend) {
    case BackendKind::FORK_SHM:
    case BackendKind::FORK_COW:
    case BackendKind::DEVICE_MALLOC:
    case BackendKind::VMM_WINDOW: {
        if (h.body_len != BACKEND_ADDRESS_BODY_BYTES)
            reject("invalid BufferDescriptor: an address-bearing backend body must be exactly 8 bytes");
        uint64_t base = 0;
        std::memcpy(&base, h.body, BACKEND_ADDRESS_BODY_BYTES);
        // No backing is mapped or allocated at 0, so a zero body is an unfilled one.
        if (base == 0) reject("invalid BufferDescriptor: address-bearing backend body is a null base");
        break;
    }
    case BackendKind::POSIX_SHM: {
        // The consumer opens this by name, and the Python side decodes it as UTF-8 before the open.
        // Restricting the name to printable ASCII other than '/' makes that decode total instead of
        // a second, weaker schema: it is a UTF-8 subset, so bytes that pass here always decode. '/'
        // is excluded because shm_open forbids it inside a name; space and the control range because
        // the name is rendered into paths and diagnostics.
        if (h.body_len == 0) reject("invalid BufferDescriptor: POSIX_SHM body must carry a shm name");
        for (uint16_t i = 0; i < h.body_len; ++i) {
            const auto c = static_cast<unsigned char>(h.body[i]);
            if (c < 0x21 || c > 0x7E || c == '/') {
                reject("invalid BufferDescriptor: POSIX_SHM shm name must be printable ASCII without '/'");
            }
        }
        break;
    }
    case BackendKind::VMM_SHAREABLE: {
        if (h.body_len != VMM_SHAREABLE_BODY_BYTES)
            reject("invalid BufferDescriptor: VMM_SHAREABLE body must be exactly 24 bytes");
        int32_t device_id = 0;
        uint32_t reserved = 0;
        uint64_t shareable_handle = 0;
        uint64_t mapping_bytes = 0;
        std::memcpy(&device_id, h.body, sizeof(device_id));
        std::memcpy(&reserved, h.body + 4, sizeof(reserved));
        std::memcpy(&shareable_handle, h.body + 8, sizeof(shareable_handle));
        std::memcpy(&mapping_bytes, h.body + 16, sizeof(mapping_bytes));
        if (device_id < 0) reject("invalid BufferDescriptor: VMM_SHAREABLE device_id out of range");
        if (reserved != 0) reject("invalid BufferDescriptor: VMM_SHAREABLE reserved bytes must be zero");
        if (shareable_handle == 0) reject("invalid BufferDescriptor: VMM_SHAREABLE shareable_handle must be nonzero");
        if (mapping_bytes == 0) reject("invalid BufferDescriptor: VMM_SHAREABLE mapping_bytes must be nonzero");
        if (mapping_bytes < h.nbytes) reject("invalid BufferDescriptor: VMM_SHAREABLE mapping_bytes must cover nbytes");
        break;
    }
    case BackendKind::REMOTE_SIDECAR:
        // The authoritative descriptor rides in the per-task sidecar, so nothing belongs here.
        if (h.body_len != 0) reject("invalid BufferDescriptor: REMOTE_SIDECAR carries no body in P1");
        break;
    }

    // The unused tail of `body` crosses a process boundary with the descriptor, so whatever the
    // owner's memory held there would cross with it.
    //
    // The rule covers the regions a PRODUCER chooses not to fill — this tail, and a Tensor's
    // shape/stride slots past `ndims` (see validate_tensor). It does NOT cover the `_pad` fields:
    // those are alignment slack no producer selects, and `CanonicalIdentity::_pad` in particular is
    // deliberately tolerated so two decodes of one backing still key alike. Equality stays field-wise
    // for that reason, so a descriptor is never canonical byte-for-byte, only field-for-field.
    for (uint32_t i = h.body_len; i < DESC_MAX_BYTES; ++i) {
        if (h.body[i] != 0) reject("invalid BufferDescriptor: reserved bytes past body_len must be zero");
    }
}

/**
 * Reject any Tensor whose fields are not self-consistent, BEFORE any of them is trusted.
 *
 * This is the single implementation behind the trust boundaries that build or decode a Tensor —
 * construction and the builder (`TaskArgs.add_tensor`, which accepts raw bytes) today, blob decode
 * on receipt when the wire carries a Tensor — so they can never drift apart. Throws
 * `std::invalid_argument` naming the field.
 *
 * Materialization is NOT behind this gate yet: `ImportRegistry.materialize` takes an already-decoded
 * descriptor and adds no endpoint check, so nothing there refuses a DEVICE backing handed to a host
 * endpoint. That gate is a separate change; do not read this comment as covering it.
 *
 * `ndims` is bounded against `MAX_TENSOR_DIMS` here; the descriptor's own length-like field is
 * bounded by `validate_buffer_descriptor` above, which this runs first.
 */
inline void validate_tensor(const Tensor &r) {
    auto reject = [](const char *what) {
        throw std::invalid_argument(what);
    };
    const BufferDescriptor &h = r.buffer;

    validate_buffer_descriptor(h);

    if (r.ndims == 0 || r.ndims > static_cast<uint32_t>(MAX_TENSOR_DIMS)) reject("invalid Tensor: ndims out of range");
    if (r.dtype >= DataType::DATA_TYPE_NUM) reject("invalid Tensor: unknown dtype");
    const uint64_t elem = get_element_size(r.dtype);
    if (elem == 0) reject("invalid Tensor: unknown dtype");
    if (r.byte_offset % elem != 0) reject("invalid Tensor: byte_offset is not a multiple of the dtype size");

    for (uint32_t i = 0; i < r.ndims; ++i) {
        if (r.shapes[i] == 0) reject("invalid Tensor: shape dimension is zero");
        if (r.strides[i] == 0) reject("invalid Tensor: stride must be > 0 (broadcast and negative step unsupported)");
    }
    // Slots past `ndims` are the same case as the descriptor's body tail: a producer chose not to
    // fill them, and the blob memcpy carries all 144 bytes across the process boundary regardless.
    // Requiring them zero is also what makes two encodings of one view agree over the active fields
    // AND the inactive ones, so a decoded element can be compared as bytes. (`_pad` is excluded, as
    // in the descriptor — it is alignment slack, not a slot anyone selects.)
    for (uint32_t i = r.ndims; i < static_cast<uint32_t>(MAX_TENSOR_DIMS); ++i) {
        if (r.shapes[i] != 0 || r.strides[i] != 0) {
            reject("invalid Tensor: shape/stride slots past ndims must be zero");
        }
    }
    const uint64_t extent_bytes = tensor_extent_bytes(r);
    if (extent_bytes == TENSOR_EXTENT_UNREPRESENTABLE) {
        reject("invalid Tensor: view extent overflows 64 bits (shape/stride not representable)");
    }
    if (r.byte_offset > h.nbytes || extent_bytes > h.nbytes - r.byte_offset) {
        reject("invalid Tensor: view extends past the backing (byte_offset + extent > nbytes)");
    }
}

/**
 * Half-open byte range [begin, end) within one backing — a view's bounding box.
 *
 * A strided view's gaps count as occupied, so this is only a bound: `x[0:4, 0:4]` and
 * `x[0:4, 8:12]` of one 16x16 matrix are disjoint yet their bounding ranges intersect. It is the
 * O(1) reject stage of `tensor_overlap` below, which refines an intersection per dimension.
 *
 * `end` saturates at UINT64_MAX, so an extent that does not fit 64 bits cannot wrap a range into
 * a false "disjoint". A default-constructed range spans the whole backing.
 */
struct TensorByteRange {
    uint64_t begin{0};
    uint64_t end{UINT64_MAX};

    bool overlaps(const TensorByteRange &o) const { return begin < o.end && o.begin < end; }
    bool covers(const TensorByteRange &o) const { return begin <= o.begin && o.end <= end; }
};

// The bounding range a view occupies in its backing.
inline TensorByteRange tensor_byte_range(const Tensor &r) {
    return TensorByteRange{r.byte_offset, tensor_add_sat(r.byte_offset, tensor_extent_bytes(r))};
}

/**
 * How one view (`probe`) stands to another (`entry`) over the same backing.
 *
 * `PARTIAL` is also the answer when the pair cannot be compared exactly, so a caller may only
 * read `NONE` as proof of disjointness — never `PARTIAL` as proof of a shared byte.
 */
enum class TensorOverlap : uint8_t {
    NONE,     // provably no common byte
    PARTIAL,  // they share a byte, or the pair is not exactly comparable
    COVERED,  // every byte of `entry` is also a byte of `probe`
};

/**
 * The geometry of a view inside its backing — what decides whether two views touch a common byte.
 *
 * `ndims == 0` is the whole-backing footprint: what a caller registers when it knows a buffer is
 * touched but not which bytes (an allocation; a remote argument whose key already carries the
 * offset). No valid Tensor has `ndims == 0`, so a default-constructed footprint means exactly
 * that, and overlaps everything.
 *
 * The backing's identity is deliberately absent: a footprint is only ever compared against
 * another over the SAME backing, which the caller establishes — a shared dependency key, or the
 * identity check in `tensors_overlap`.
 */
struct TensorFootprint {
    uint64_t byte_offset{0};
    uint64_t backing_nbytes{0};
    uint32_t shapes[MAX_TENSOR_DIMS]{};
    uint32_t strides[MAX_TENSOR_DIMS]{};  // in elements, as on Tensor
    uint32_t ndims{0};
    DataType dtype{};

    bool is_whole_backing() const { return ndims == 0; }
    TensorByteRange range() const {
        if (is_whole_backing()) return TensorByteRange{};
        const uint64_t extent = view_extent_bytes(shapes, strides, ndims, dtype);
        return TensorByteRange{byte_offset, tensor_add_sat(byte_offset, extent)};
    }
};

inline TensorFootprint tensor_footprint(const Tensor &r) {
    TensorFootprint f{};
    f.byte_offset = r.byte_offset;
    f.backing_nbytes = r.buffer.nbytes;
    f.ndims = r.ndims;
    f.dtype = r.dtype;
    for (uint32_t i = 0; i < MAX_TENSOR_DIMS; ++i) {
        f.shapes[i] = r.shapes[i];
        f.strides[i] = r.strides[i];
    }
    return f;
}

/**
 * Do two views of the same backing touch a common byte, and does `probe` swallow `entry` whole?
 *
 * Three stages, mirroring the L2 `ChipTensorMap::check_overlap` this is the host-side counterpart
 * of (`src/{arch}/runtime/tensormap_and_ringbuffer/runtime/tensormap.h`):
 *
 *   1. Bounding-range intersection. O(1), and the only stage a whole-backing footprint reaches.
 *      Disjoint bounding boxes are disjoint views, so this alone can answer `NONE`.
 *   2. Per-dimension hyper-rectangle intersection, once the bounding boxes do intersect. Eligible
 *      only when both sides share one canonical row-major axis layout — same dtype and ndims,
 *      identical strides, `strides` descending as exact integer multiples down to 1, and each
 *      origin decomposing cleanly under the shape that layout implies. Disjoint on any single
 *      axis makes the views disjoint, which is what separates two column blocks whose bounding
 *      boxes interleave.
 *   3. Anything stage 2 cannot model — a transposed pair, a stepped slice, mismatched dtypes —
 *      falls out as `PARTIAL`. Exact enumeration of those is not attempted.
 *
 * Every rejection path in stage 2 yields `PARTIAL`, never `NONE`: an unmodelled pair is assumed
 * to conflict. So a false edge is possible where the truth is subtler; a real one is never lost.
 */
inline TensorOverlap tensor_overlap(const TensorFootprint &probe, const TensorFootprint &entry) {
    // ---- Stage 1: bounding-range intersection (O(1) reject) ----
    const TensorByteRange probe_range = probe.range();
    const TensorByteRange entry_range = entry.range();
    if (!probe_range.overlaps(entry_range)) return TensorOverlap::NONE;
    if (probe.is_whole_backing()) {
        return entry_range.begin >= probe_range.begin && entry_range.end <= probe_range.end ? TensorOverlap::COVERED :
                                                                                              TensorOverlap::PARTIAL;
    }
    if (entry.is_whole_backing()) return TensorOverlap::PARTIAL;

    // ---- Stage 2 prerequisites: one shared canonical row-major axis layout ----
    if (probe.dtype != entry.dtype || probe.ndims != entry.ndims) return TensorOverlap::PARTIAL;
    const uint32_t ndims = probe.ndims;
    if (ndims > static_cast<uint32_t>(MAX_TENSOR_DIMS)) return TensorOverlap::PARTIAL;
    for (uint32_t i = 0; i < ndims; ++i) {
        if (probe.strides[i] != entry.strides[i]) return TensorOverlap::PARTIAL;
        if (probe.strides[i] == 0) return TensorOverlap::PARTIAL;
    }
    // The reference-shape derivation below reads `strides[i] == prod(shape[i+1..])`. Requiring the
    // innermost stride to be 1 and each outer one to be an exact multiple of its inner neighbour is
    // what makes that true; it rejects stepped slices and any view chain that reorders the axes.
    if (probe.strides[ndims - 1] != 1) return TensorOverlap::PARTIAL;
    for (uint32_t i = 1; i < ndims; ++i) {
        if (probe.strides[i - 1] % probe.strides[i] != 0) return TensorOverlap::PARTIAL;
    }

    const uint64_t elem = get_element_size(probe.dtype);
    if (elem == 0) return TensorOverlap::PARTIAL;
    if (probe.byte_offset % elem != 0 || entry.byte_offset % elem != 0) return TensorOverlap::PARTIAL;

    // Reference shape A of the backing, recovered from the stride vector: A[i] = strides[i-1] /
    // strides[i], and A[0] from however many elements the backing holds. An allocator that rounded
    // the backing up leaves A[0] larger than the truth, which only loosens the bounds check below —
    // A[0] is never used as anything but an upper bound, so it can produce neither a false NONE nor
    // a false COVERED.
    uint32_t ref_shapes[MAX_TENSOR_DIMS] = {};
    for (uint32_t i = 1; i < ndims; ++i) {
        ref_shapes[i] = probe.strides[i - 1] / probe.strides[i];
    }
    const uint64_t numel_storage = probe.backing_nbytes / elem;
    const uint32_t stride0 = probe.strides[0];
    if (numel_storage % stride0 != 0) return TensorOverlap::PARTIAL;
    const uint64_t ref_shape0 = numel_storage / stride0;
    if (ref_shape0 > UINT32_MAX) return TensorOverlap::PARTIAL;
    ref_shapes[0] = static_cast<uint32_t>(ref_shape0);

    // Decompose each origin into per-axis coordinates. `strides[i] == prod(A[i+1..])`, so dividing
    // the remainder by strides[i] yields axis i directly. A non-zero final remainder means the
    // origin does not sit on a lattice point of this layout — not modellable, so PARTIAL.
    uint64_t probe_off[MAX_TENSOR_DIMS] = {};
    uint64_t entry_off[MAX_TENSOR_DIMS] = {};
    uint64_t probe_remain = probe.byte_offset / elem;
    uint64_t entry_remain = entry.byte_offset / elem;
    for (uint32_t i = 0; i < ndims; ++i) {
        const uint64_t s = probe.strides[i];
        probe_off[i] = probe_remain / s;
        entry_off[i] = entry_remain / s;
        probe_remain %= s;
        entry_remain %= s;
    }
    if (probe_remain != 0 || entry_remain != 0) return TensorOverlap::PARTIAL;

    for (uint32_t i = 0; i < ndims; ++i) {
        if (probe_off[i] + probe.shapes[i] > ref_shapes[i]) return TensorOverlap::PARTIAL;
        if (entry_off[i] + entry.shapes[i] > ref_shapes[i]) return TensorOverlap::PARTIAL;
    }

    // ---- Stage 2 core: per-axis line-segment intersection ----
    bool probe_contains_entry = true;
    for (uint32_t i = 0; i < ndims; ++i) {
        const uint64_t p_begin = probe_off[i];
        const uint64_t p_end = p_begin + probe.shapes[i];
        const uint64_t e_begin = entry_off[i];
        const uint64_t e_end = e_begin + entry.shapes[i];
        if (!(p_begin < e_end && e_begin < p_end)) return TensorOverlap::NONE;
        if (!(p_begin <= e_begin && e_end <= p_end)) probe_contains_entry = false;
    }
    return probe_contains_entry ? TensorOverlap::COVERED : TensorOverlap::PARTIAL;
}

// Do two views of the SAME backing touch a common byte?
inline bool tensors_overlap(const Tensor &a, const Tensor &b) {
    if (!(a.buffer.identity == b.buffer.identity)) return false;
    return tensor_overlap(tensor_footprint(a), tensor_footprint(b)) != TensorOverlap::NONE;
}

static_assert(std::is_trivially_copyable_v<Tensor>, "Tensor must be trivially copyable for blob memcpy");
static_assert(sizeof(Tensor) == 144, "Tensor is wire ABI");
static_assert(offsetof(Tensor, buffer) == 0);
static_assert(offsetof(Tensor, byte_offset) == 88);
static_assert(offsetof(Tensor, ndims) == 96);
static_assert(offsetof(Tensor, shapes) == 100);
static_assert(offsetof(Tensor, strides) == 120);
static_assert(offsetof(Tensor, dtype) == 140);
