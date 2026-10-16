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
 * WorkerManager — worker pool lifecycle and dispatch.
 *
 * Owns one endpoint lane per registered worker.
 * Provides idle-worker selection and dispatch to the Scheduler.
 * The Scheduler drives both the DAG and endpoint progress through the Manager.
 *
 * Every registered local and remote endpoint advances through non-blocking
 * submit/poll calls on the Scheduler thread.
 * A two-frame local endpoint may additionally publish one PREPARE_READY
 * successor while the active frame runs; FIFO promotion changes it to ACTIVATE.
 * The Python child resolves each frame's callable digest to a child-local slot
 * and executes it on its `ChipWorker` (NEXT_LEVEL).
 */

#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../task_interface/buffer.h"
#include "../task_interface/call_config.h"
#include "../worker/device_memory_info.h"
#include "../worker/runtime_c_api.h"
#include "remote_wire.h"
#include "types.h"

class Ring;  // forward decl — owns the slot state pool
class WorkerManager;

// =============================================================================
// Unified mailbox layout (PROCESS mode)
// =============================================================================
//
// One layout for both NEXT_LEVEL (chip) and SUB workers. SUB children
// read the callable hash digest from MAILBOX_OFF_ARGS, resolve it to a
// private slot, and decode the TaskArgs blob after the digest prefix.

enum class MailboxState : int32_t {
    IDLE = 0,
    TASK_READY = 1,
    TASK_DONE = 2,
    SHUTDOWN = 3,
    CONTROL_REQUEST = 4,
    CONTROL_DONE = 5,
    // Startup readiness handshake, driven entirely by the Python facade
    // (_await_children_ready). A child writes INIT_READY after its expensive
    // init (ChipWorker::init / inner Worker::init) succeeds, or INIT_FAILED
    // after it fails. The parent barrier waits for EVERY child to reach
    // INIT_READY before any dispatch (CTRL_PREPARE or TASK_READY) goes out,
    // which also aligns the host-side stream-sync windows across distributed
    // ranks so cross-rank init skew never charges against the per-rank
    // PLATFORM_STREAM_SYNC_TIMEOUT_MS budget (issue #897).
    INIT_READY = 6,
    INIT_FAILED = 7,
    FRAME_STAGED = 8,
    TASK_LAUNCHED = 9,
    TASK_FAILED = 10,
    ACTIVATE = 11,
    PREPARE_READY = 12,
};

enum class MailboxPreparationDisposition : int32_t {
    NONE = 0,
    VALIDATED_ONLY = 1,
    NATIVE_PREPARED = 2,
};

bool mailbox_compare_exchange_state(char *frame, MailboxState expected, MailboxState desired) noexcept;

// Sized so the args region can hold any TaskArgs the runtime itself accepts
// (CHIP_MAX_TENSOR_ARGS tensors + CHIP_MAX_SCALAR_ARGS scalars; see the
// static_assert after MAILBOX_ARGS_CAPACITY). At the 256-tensor cap, wire tensors
// occupy 36 KiB of the 64 KiB frame and leave room for scalars and protocol metadata.
static constexpr size_t MAILBOX_FRAME_SIZE = 65536;
static constexpr size_t MAILBOX_CONTROL_FRAME = 0;
static constexpr size_t MAILBOX_FIRST_TASK_FRAME = 1;
static constexpr uint32_t MAILBOX_TASK_PROTOCOL_VERSION = 7;

// Error message region. 256 B of headroom is enough for
// `<ExceptionType>: <short message>` produced by the child-side Python loops;
// anything longer is truncated + NUL-terminated.
static constexpr size_t MAILBOX_ERROR_MSG_SIZE = 256;

/**
 * Request frames: how many payload hand-offs may be outstanding at once.
 *
 * A request frame carries one submission's bytes — callable digest, config,
 * lease and the args blob — and nothing about the run's life. The child copies
 * the blob out and says so, and the frame is immediately reusable by the next
 * submission while the run it carried is still executing. So this bounds
 * *concurrent hand-off*, not concurrent runs, and two is enough to keep one
 * hand-off in flight while the next is written.
 */
static constexpr size_t MAILBOX_REQUEST_FRAMES = 2;

/**
 * Run-state entries beyond the granted sets.
 *
 * `M = granted + K`. The extra `K` is what lets a request be handed over before
 * a set is free: the entry exists from the moment the parent publishes, and the
 * run it names waits for a set without holding a 64 KiB frame to do it.
 */
constexpr size_t mailbox_run_entry_count(size_t granted_sets) { return granted_sets + MAILBOX_REQUEST_FRAMES; }

/**
 * One run's whole cross-process state, for its whole life.
 *
 * Separate from the request frame that delivered it, which is the point: a
 * 64 KiB frame is held only for the hand-off, and this ~360 B entry is held for
 * the run. POD and contiguous — it is written by a parent and read by its forked
 * child, so it carries no pointer and no C++ type with a vtable or an allocator.
 *
 * Single-writer per field, with the edges stated beside each one. `state` is the
 * only field two sides write, and they never do so at the same time: the parent
 * owns the `FRAME_STAGED -> ACTIVATE` exchange and the child owns every other
 * transition, with the terminal store as the child's final access to the entry.
 */
struct MailboxRunEntry {
    /* Child, except the parent's identity-checked FRAME_STAGED -> ACTIVATE CAS. */
    int32_t state;
    /* Child, before the terminal state. */
    int32_t error;
    /* Device, at the kernel-launch marker. Sticky; the parent clears it on allocate. */
    int32_t accepted;
    /* Child, before the terminal state. */
    int32_t preparation_disposition;
    /* Parent, on allocate and free. Odd while in use, even while free; a stale
       reference carrying an older value is refused rather than acted on. */
    uint64_t entry_generation;
    /* Child, once, after copying the args blob out of its request frame. A
       level, not a pulse: the parent compares it with the generation of the
       frame it published, so a missed observation is impossible. */
    uint64_t frame_consumed_generation;
    /* Child, on activation. */
    uint64_t run_epoch;
    /* Parent, before publishing; the identity the child checks its frame against. */
    uint64_t protocol;
    uint64_t run_id;
    uint64_t slot_id;
    uint64_t generation;
    uint64_t dispatch_id;
    uint64_t task_slot;
    uint64_t group_index;
    uint64_t group_size;
    /* Child, before the terminal state. NUL-terminated, truncated to fit. */
    char error_text[MAILBOX_ERROR_MSG_SIZE];
    /* Run-resource sets the child reports it can hold, published with this
       run's terminal state and read by the parent only after observing it. A
       count of identities, never a byte figure, and zero means "no report" —
       a child that does not size itself leaves it alone. Appended last so no
       existing field's offset moves. */
    uint32_t admissible_run_resource_sets;
};
static_assert(
    std::is_trivially_copyable_v<MailboxRunEntry> && std::is_standard_layout_v<MailboxRunEntry>,
    "a run-state entry crosses a process boundary, so it must be POD and position-independent"
);

/** Stride between entries. Rounded up so every entry keeps 8-byte field alignment. */
static constexpr size_t MAILBOX_RUN_ENTRY_STRIDE = (sizeof(MailboxRunEntry) + 63) & ~size_t{63};

/** Byte offset of the entry table, which follows the control and request frames. */
static constexpr size_t MAILBOX_RUN_ENTRY_TABLE_OFFSET = MAILBOX_FRAME_SIZE * (1 + MAILBOX_REQUEST_FRAMES);

/**
 * Bytes one endpoint's mailbox occupies for `granted_sets` in-flight runs.
 *
 * One control frame, `MAILBOX_REQUEST_FRAMES` request frames, and one run-state
 * entry per run the endpoint may hold plus the hand-offs in flight. Only the
 * entry table grows with the grant, so a deeper context costs ~360 B per run
 * rather than 64 KiB. The Python facade that creates the shared-memory region
 * reads this through the same binding, so both ends derive the size from one
 * place.
 */
constexpr size_t mailbox_size(size_t granted_sets) {
    return MAILBOX_RUN_ENTRY_TABLE_OFFSET + MAILBOX_RUN_ENTRY_STRIDE * mailbox_run_entry_count(granted_sets);
}

/**
 * The largest grant a mailbox may be sized for: a 1 TiB entry table.
 *
 * Larger grants are rejected by this conservative sizing policy; their computed
 * size is still representable in a 64-bit `size_t`. The Python binding has
 * applied the same cap since the grant became caller-chosen.
 */
static constexpr uint64_t MAILBOX_MAX_GRANTED_SETS =
    (static_cast<uint64_t>(1) << 40) / static_cast<uint64_t>(MAILBOX_RUN_ENTRY_STRIDE);

/**
 * Bytes `granted_sets` needs, or 0 when it is outside the admitted range.
 *
 * Zero is never a mailbox's size — even the smallest grant has the frames — so
 * it carries the refusal with no second out-parameter. Whoever maps a region and
 * whoever validates one both ask this, which is what keeps the bound they
 * enforce the same bound.
 */
constexpr size_t mailbox_size_for_grant(uint64_t granted_sets) {
    if (granted_sets == 0 || granted_sets > MAILBOX_MAX_GRANTED_SETS) return 0;
    return mailbox_size(static_cast<size_t>(granted_sets));
}

/** Run-state entry `index` inside a mapped mailbox region. */
inline MailboxRunEntry *mailbox_run_entry(void *mailbox, size_t index) {
    return reinterpret_cast<MailboxRunEntry *>(
        static_cast<char *>(mailbox) + MAILBOX_RUN_ENTRY_TABLE_OFFSET + MAILBOX_RUN_ENTRY_STRIDE * index
    );
}

/**
 * Run-resource sets a mailbox is built for when whoever creates it does not say.
 *
 * Not a ceiling — nothing refuses a larger region, and the endpoint's counts
 * come from its worker's grant. This is the convenience size for a caller with
 * no grant to size from, which in practice is a fixture standing in for the
 * Python facade.
 */
static constexpr size_t MAILBOX_DEFAULT_GRANTED_SETS = 3;
static constexpr size_t MAILBOX_SIZE = mailbox_size(MAILBOX_DEFAULT_GRANTED_SETS);

// CallConfig is written/read as a single packed POD block (see call_config.h).
// Both ends transfer it with one memcpy — no per-field offsets to keep in sync.
//
// The generation-safe run lease follows CallConfig. MAILBOX_OFF_ARGS is
// derived by rounding up the lease's end so the
// args blob's first ChipTensor field (buffer.addr, a uint64_t at OFF_ARGS+8) is
// 8-byte aligned, avoiding SIGBUS on strict-alignment platforms (aarch64
// atomics, some ARM cores). The base frame carries control commands; task
// frames use the same relative layout without sharing those bytes.
static constexpr ptrdiff_t MAILBOX_OFF_STATE = 0;
static constexpr ptrdiff_t MAILBOX_OFF_ERROR = 4;
static constexpr ptrdiff_t MAILBOX_OFF_CALLABLE = 8;  // also: control sub-command (uint64)
static constexpr ptrdiff_t MAILBOX_OFF_CONFIG = 16;
static constexpr ptrdiff_t MAILBOX_OFF_PIPELINE_LEASE =
    (MAILBOX_OFF_CONFIG + static_cast<ptrdiff_t>(sizeof(CallConfig)) + 7) & ~ptrdiff_t{7};
static constexpr ptrdiff_t MAILBOX_OFF_ARGS =
    (MAILBOX_OFF_PIPELINE_LEASE + static_cast<ptrdiff_t>(sizeof(PipelineSlotLease)) + 7) & ~ptrdiff_t{7};
static_assert(MAILBOX_OFF_ARGS % 8 == 0, "MAILBOX_OFF_ARGS must be 8-aligned for ChipTensor.buffer.addr");
static_assert(
    MAILBOX_OFF_CONFIG + static_cast<ptrdiff_t>(sizeof(CallConfig)) <= MAILBOX_OFF_ARGS,
    "CallConfig overflows reserved config region"
);
static_assert(
    MAILBOX_OFF_PIPELINE_LEASE + static_cast<ptrdiff_t>(sizeof(PipelineSlotLease)) <= MAILBOX_OFF_ARGS,
    "PipelineSlotLease overflows reserved lease region"
);
static constexpr ptrdiff_t MAILBOX_OFF_ERROR_MSG =
    static_cast<ptrdiff_t>(MAILBOX_FRAME_SIZE) - static_cast<ptrdiff_t>(MAILBOX_ERROR_MSG_SIZE);
// Launch acceptance is sticky, not a MailboxState: a state word carrying it
// loses the ACK whenever the child reaches TASK_DONE between two parent polls,
// which is exactly the short-task case the fence exists to pipeline. The child
// sets it only after a real native launch; the parent clears it when it
// allocates the entry, so nothing else can overwrite it in between. It lives on
// the run-state entry rather than the request frame because the device writes it
// after the frame has already been handed back.
static constexpr int32_t MAILBOX_TASK_ACCEPTED = 1;
// A request frame's own header, in the bytes the identity trailer used to hold:
// which entry this payload belongs to, that entry's generation, and this
// frame's own generation — which the child echoes to say the blob is copied.
static constexpr ptrdiff_t MAILBOX_OFF_FRAME_ENTRY_INDEX = MAILBOX_OFF_ERROR_MSG - 32;
static constexpr ptrdiff_t MAILBOX_OFF_FRAME_ENTRY_GENERATION = MAILBOX_OFF_ERROR_MSG - 24;
static constexpr ptrdiff_t MAILBOX_OFF_FRAME_GENERATION = MAILBOX_OFF_ERROR_MSG - 16;
// Termination is a sticky one-way word on the control frame, not a MailboxState:
// MAILBOX_OFF_STATE has three writers (the parent's CONTROL_REQUEST, the child's
// CONTROL_DONE, and this endpoint's return-to-IDLE), any of which overwrites a
// SHUTDOWN store. Only a terminating parent writes this word, 0 -> 1, and
// nothing ever clears it, so a child that observes it exits its serve loop no
// matter what state word a concurrent control command leaves behind.
static constexpr ptrdiff_t MAILBOX_OFF_SHUTDOWN = MAILBOX_OFF_ERROR_MSG - 80;
static constexpr int32_t MAILBOX_SHUTDOWN_REQUESTED = 1;
// The chip child's teardown observation, published once on the control frame
// after its device teardown returns. Reserved on every frame for the same
// reason the shutdown word is: a task-args blob must not be able to reach it.
//
// It is a record rather than a state word because its fields are only
// meaningful together, and `schema` at offset 0 is what makes that safe: the
// producer fills the payload first and releases `schema` last, so a reader
// that acquire-loads TEARDOWN_REPORT_SCHEMA has the whole record and a reader
// that finds anything else has none of it.
static constexpr ptrdiff_t MAILBOX_OFF_TEARDOWN_REPORT =
    MAILBOX_OFF_SHUTDOWN - static_cast<ptrdiff_t>(SIMPLER_TEARDOWN_REPORT_BYTES) - 8;
static constexpr ptrdiff_t MAILBOX_OFF_TEARDOWN_REPORT_SCHEMA = MAILBOX_OFF_TEARDOWN_REPORT;
static constexpr ptrdiff_t MAILBOX_OFF_TASK_CALLABLE_HASH = MAILBOX_OFF_ARGS;
static constexpr ptrdiff_t MAILBOX_OFF_TASK_ARGS_BLOB =
    MAILBOX_OFF_TASK_CALLABLE_HASH + static_cast<ptrdiff_t>(CALLABLE_HASH_DIGEST_SIZE);
static constexpr size_t CTRL_SHM_NAME_BYTES = 32;
static constexpr ptrdiff_t MAILBOX_OFF_CONTROL_CALLABLE_HASH =
    MAILBOX_OFF_ARGS + static_cast<ptrdiff_t>(CTRL_SHM_NAME_BYTES);
static_assert(
    MAILBOX_OFF_TASK_ARGS_BLOB < MAILBOX_OFF_TEARDOWN_REPORT,
    "mailbox task-args region must precede the teardown record, the shutdown word and the frame "
    "protocol trailer"
);
static_assert(
    MAILBOX_OFF_TEARDOWN_REPORT % 8 == 0,
    "the teardown record starts 8-aligned so its schema word is atomically storable"
);
// The teardown record and the shutdown word are reserved on every frame, not
// just the control frame, so the args region a task frame accepts can never
// reach either.
static constexpr size_t MAILBOX_ARGS_CAPACITY =
    static_cast<size_t>(MAILBOX_OFF_TEARDOWN_REPORT) - static_cast<size_t>(MAILBOX_OFF_TASK_ARGS_BLOB);
// The blob's element is the wire `Tensor` (144 B), not the device `ChipTensor` (128 B), so a frozen
// descriptor size and a frame size that cannot hold CHIP_MAX_TENSOR_ARGS of them fail the build
// rather than the first 256-tensor task.
static_assert(
    MAILBOX_ARGS_CAPACITY >= TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(CHIP_MAX_TENSOR_ARGS) * sizeof(Tensor) +
                                 static_cast<size_t>(CHIP_MAX_SCALAR_ARGS) * sizeof(uint64_t),
    "mailbox args region must hold the largest args blob the runtime accepts (issue #1024)"
);

// Control sub-commands (written at MAILBOX_OFF_CALLABLE when state == CONTROL_*)
static constexpr uint64_t CTRL_MALLOC = 0;
static constexpr uint64_t CTRL_FREE = 1;
// Host<->device copy. Both ends are handles: the payload is a ControlCopyRequest at
// MAILBOX_OFF_ARGS, and the child resolves each descriptor through the same ImportRegistry that
// resolves task arguments, so a backing is reached by canonical identity and never by an address
// minted in the parent's address space.
static constexpr uint64_t CTRL_COPY_TO = 2;
static constexpr uint64_t CTRL_COPY_FROM = 3;
// Pre-warm a chip child by callable digest; issued at end of init() so the
// first simpler_run does not pay the H2D cost.
static constexpr uint64_t CTRL_PREPARE = 4;
// Dynamic post-init register/unregister of a callable identity. CTRL_REGISTER
// carries (shm_name, blob_size, digest) with bytes staged in POSIX shm by
// the parent; CTRL_UNREGISTER carries digest only.
static constexpr uint64_t CTRL_REGISTER = 5;
static constexpr uint64_t CTRL_UNREGISTER = 6;
// Dynamic per-orch CommDomain allocation/release.  Both carry a pair of
// NUL-terminated POSIX shm names at MAILBOX_OFF_ARGS — first the request shm
// (parent → child: header + rank_ids + buffer_nbytes), second the reply shm
// (child → parent: device_ctx + local_window_base + buffer_ptrs).  RELEASE
// uses only the request shm; no payload comes back beyond status.
static constexpr uint64_t CTRL_ALLOC_DOMAIN = 7;
static constexpr uint64_t CTRL_RELEASE_DOMAIN = 8;
// Establish the base HCCL/sim communicator on this chip.  Driven lazily by
// the orch facade on the first ``allocate_domain`` call.  Payload is a single
// NUL-terminated shm name at MAILBOX_OFF_ARGS that points to a small request
// shm containing (rank: u32, nranks: u32, rootinfo_path: NUL-terminated).
// Caches the comm handle on the chip's ChipWorker so subsequent
// CTRL_ALLOC_DOMAIN calls can find it.
static constexpr uint64_t CTRL_COMM_INIT = 9;
static constexpr uint64_t CTRL_PY_REGISTER = 10;
static constexpr uint64_t CTRL_PY_UNREGISTER = 11;
// 16 and 17 are unused retired control-command numbers and must not be reassigned.
// Query a chip child's MemoryAllocator-committed HBM (bytes). The child writes
// the selected chip's value to CTRL_OFF_RESULT.
static constexpr uint64_t CTRL_COMMITTED_DEVICE_MEMORY = 18;
// 19..24 are reserved by the Python Global CommDomain chip controls and its
// L4-to-local-L3 envelope. Query a chip child's device-wide ACL_HBM_MEM
// snapshot; the child writes one DeviceMemoryInfo at CTRL_OFF_RESULT.
static constexpr uint64_t CTRL_DEVICE_MEMORY_INFO = 25;
// 26 is reserved by the Python delegated-region control (below). Publish every
// diagnostic run this chip child has closed, and report what it published. The
// child writes one DfxFlushReport at CTRL_OFF_RESULT.
static constexpr uint64_t CTRL_DFX_FLUSH = 27;
// 26 is reserved by the Python delegated-region control. It carries the DRCT
// envelope on control_payload at every hop of the recursive single-owner
// region protocol, so no C++ endpoint method claims it.

// Control args occupy the base frame's config-sized region:
//   offset 16: uint64 arg0 (size for malloc/register; ptr for free)
//   offset 40: uint64 result (returned ptr from malloc)
static constexpr ptrdiff_t CTRL_OFF_ARG0 = 16;
static constexpr ptrdiff_t CTRL_OFF_RESULT = 40;

/**
 * CTRL_DFX_FLUSH result, written by the child at CTRL_OFF_RESULT.
 *
 * Fixed width and small enough to share the control frame's existing result
 * slot, so the mailbox gains no bytes. `published` counts runs whose artifact
 * exists, which includes a partial one — an incomplete receipt that is
 * represented in a published file is a verdict, not a missing output.
 */
struct DfxFlushReport {
    uint64_t session_id;
    uint64_t watermark_epoch;
    uint64_t published;
    uint64_t failed;
};

// CTRL_REGISTER puts the NUL-terminated POSIX shm name at MAILBOX_OFF_ARGS,
// the exact staged blob size at CTRL_OFF_ARG0, and the callable digest
// immediately after the shm-name slot.
// Fixed-width so the wire layout stays simple; well above the encoded length
// of "simpler-cb-<pid>-<counter>" with pid < 32-bit max.

// The byte range a control-plane copy touches: `nbytes` starting at `dst_offset` into the
// destination backing and at `src_offset` into the source. One contiguous run at each end -- the
// underlying ChipWorker::copy_to takes a single length and a single address per side, so a strided
// or otherwise non-contiguous view has no representation here. The three travel together so a call
// site cannot pair a length with the wrong offsets.
struct CopySpan {
    uint64_t nbytes;
    uint64_t dst_offset;
    uint64_t src_offset;
};

// CTRL_COPY_TO / CTRL_COPY_FROM payload, written at MAILBOX_OFF_ARGS on the control frame. `dst`
// and `src` are in the direction the sub-command names, matching ChipWorker::copy_to /
// copy_from(dst, src, nbytes); which of the two is the device end follows from the sub-command.
// `span` travels with the pair it bounds, so neither the length nor an offset can be read from a
// slot a different sub-command last wrote.
struct ControlCopyRequest {
    BufferDescriptor dst;
    BufferDescriptor src;
    CopySpan span;
};

static_assert(std::is_trivially_copyable_v<ControlCopyRequest> && std::is_standard_layout_v<ControlCopyRequest>);
static_assert(
    MAILBOX_OFF_ARGS + static_cast<ptrdiff_t>(sizeof(ControlCopyRequest)) <= MAILBOX_OFF_SHUTDOWN,
    "control copy request overflows the control frame's args region"
);

inline void write_control_copy_request(
    char *mbox, uint64_t sub_cmd, const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span
) {
    std::memcpy(mbox + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    ControlCopyRequest request{dst, src, span};
    std::memcpy(mbox + MAILBOX_OFF_ARGS, &request, sizeof(request));
}

// Both descriptors pass `validate_buffer_descriptor` before the child can act on them, so a
// malformed or over-long backend body is refused at the same gate a task-args decode uses. The two
// spans are bounded against the descriptors they arrived with, so an offset can only ever name a
// byte inside the backing that came with it -- the same receive-side gate `validate_tensor` applies
// to a view, applied here to a copy range. Subtraction, never `offset + nbytes`, so an addition
// cannot wrap a span back inside the backing.
inline void validate_control_copy_request(const ControlCopyRequest &request) {
    auto reject = [](const char *what) {
        throw std::invalid_argument(what);
    };
    validate_buffer_descriptor(request.dst);
    validate_buffer_descriptor(request.src);
    if (request.span.dst_offset > request.dst.nbytes ||
        request.span.nbytes > request.dst.nbytes - request.span.dst_offset) {
        reject("invalid ControlCopyRequest: dst range extends past the destination backing");
    }
    if (request.span.src_offset > request.src.nbytes ||
        request.span.nbytes > request.src.nbytes - request.span.src_offset) {
        reject("invalid ControlCopyRequest: src range extends past the source backing");
    }
}

inline ControlCopyRequest read_control_copy_request(const char *mbox) {
    ControlCopyRequest request{};
    std::memcpy(&request, mbox + MAILBOX_OFF_ARGS, sizeof(request));
    validate_control_copy_request(request);
    return request;
}

struct ControlResult {
    std::string worker_type;
    int32_t worker_id{0};
    bool ok{false};
    std::string error_message;
};

// =============================================================================
// WorkerDispatch — per-dispatch handle handed to a WorkerThread.
// =============================================================================
//
// `task_slot` is the slot id; `group_index` is 0 for single tasks and
// 0..group_size-1 for group members. The thread resolves callable / args /
// config by reading `ring->slot_state(task_slot)`.

struct WorkerDispatch {
    TaskSlot task_slot{INVALID_SLOT};
    int32_t group_index{0};
    uint64_t dispatch_id{0};
    bool prepare_only{false};
};

enum class WorkerProgressKind : int32_t {
    FRAME_STAGED = 0,
    ACCEPTED = 1,
    COMPLETED = 2,
};

struct WorkerEndpointProgress {
    WorkerProgressKind kind{WorkerProgressKind::FRAME_STAGED};
    WorkerDispatch dispatch{};
    WorkerCompletion completion{};
    MailboxPreparationDisposition preparation_disposition{MailboxPreparationDisposition::NONE};
};

enum class WorkerEndpointKind : int32_t {
    LOCAL_MAILBOX = 0,
    REMOTE_L3 = 1,
    MPI_GROUP_MAILBOX = 2,
};

struct WorkerEndpointCaps {
    WorkerEndpointKind kind{WorkerEndpointKind::LOCAL_MAILBOX};
    int32_t worker_id{-1};
    bool remote{false};
    bool supports_task_dispatch{true};
    bool supports_control{true};
    uint32_t max_inflight_tasks{1};
    bool supports_frame_staging{false};
    std::string transport{"local-mailbox"};
};

class WorkerEndpoint {
public:
    virtual ~WorkerEndpoint() = default;

    virtual const WorkerEndpointCaps &caps() const = 0;

    // Every endpoint is progress-driven by the Scheduler without a blocking
    // call per frame. submit_progress publishes one complete frame;
    // poll_progress reports monotonic events; activate_progress is a sticky
    // FIFO-launch permission that may arrive before the child stages the frame.
    virtual void submit_progress(Ring *ring, const WorkerDispatch &dispatch);
    /**
     * How many more frames this endpoint can publish right now.
     *
     * Bounded request transport is a resource separate from run capacity, so a
     * grant wider than the transport makes saturation an ordinary state rather
     * than a corrupt admission. Reaching zero is backpressure: the caller must
     * not admit a publication it cannot complete, and the next poll frees what
     * the child has finished copying out.
     *
     * The count is what is free at the instant it is read, with nothing held
     * back. What keeps two gates in one round from both acting on the same last
     * frame is that all publication for all endpoints runs on one thread —
     * `Scheduler::dispatch_claimed` is the only caller of either publication
     * entry point — so a second gate in that round reads this after the first
     * has already marked its frame published. The only other party that touches
     * frame ownership is the child's acknowledgement, which can only free one.
     *
     * An endpoint whose transport is not bounded reports no limit.
     */
    virtual size_t available_request_transport();
    virtual bool poll_progress(WorkerEndpointProgress &progress);
    virtual bool activate_progress(RunId run_id);
    virtual void request_progress_stop() noexcept;
    // Called when the owning progress driver catches an endpoint exception.
    // Implementations must turn already-published work into terminal progress
    // only after their child/device resources are quiescent.
    virtual void report_progress_error(const std::string &reason);
    // The return value says whether `dispatch` was published and will therefore
    // produce terminal progress through poll_progress(). A false result leaves
    // the unpublished dispatch with the caller after endpoint quiescence.
    virtual bool report_submission_error(const WorkerDispatch &dispatch, const std::string &reason);

    virtual void shutdown_child() {}
    virtual uint64_t control_malloc(size_t size);
    virtual uint64_t control_committed_device_memory();
    virtual DeviceMemoryInfo control_device_memory_info();
    /**
     * Publish every diagnostic run this child has closed, and report.
     *
     * `timeout_s` bounds only the waits it is passed to; the mailbox mutex and
     * the caller-side leases are not timed, so the call can exceed it in
     * wall-clock terms. A timeout poisons this endpoint — see
     * run_control_command — after which no control command may reuse it.
     */
    virtual DfxFlushReport control_dfx_flush(double timeout_s);
    virtual void control_free(uint64_t ptr);
    virtual void control_copy_to(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span);
    virtual void control_copy_from(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span);
    virtual void control_prepare(const uint8_t *digest);
    virtual void control_register(const char *shm_name, size_t blob_size, const uint8_t *digest);
    virtual void control_unregister(const uint8_t *digest);
    virtual void control_remote_prepare_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest,
        const void *payload, size_t payload_size
    );
    virtual void control_remote_commit_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    );
    virtual void control_remote_abort_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    );
    virtual void control_remote_unregister(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    );
    virtual RemoteBufferHandle control_remote_malloc(size_t size);
    virtual void control_remote_free(const RemoteBufferHandle &handle);
    virtual void
    control_remote_copy_to(const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size);
    virtual void control_remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size);
    virtual RemoteBufferExport control_remote_export(
        const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
        const std::string &transport_profile
    );
    virtual RemoteBufferHandle control_remote_import(
        int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
    );
    virtual void control_remote_release_import(const RemoteBufferHandle &handle);
    virtual std::vector<uint8_t> control_remote_domain(
        remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes, bool group_target = false
    );
    virtual void control_generic(
        uint64_t sub_cmd, const char *shm_name, size_t payload_size, double timeout_s, const uint8_t *digest
    );
    virtual void control_alloc_domain(const char *request_shm_name, const char *reply_shm_name);
    virtual void control_release_domain(const char *request_shm_name);
    virtual void control_comm_init(const char *request_shm_name);
};

class LocalMailboxEndpoint : public WorkerEndpoint {
public:
    // `mailbox_bytes` is how much of the region at `mailbox` the caller actually
    // mapped. The constructor writes the whole entry table `granted_sets`
    // implies, so it is checked against that extent before the first write;
    // nothing here can derive it from the grant, since the grant is the claim
    // being checked.
    //
    // `child_pid` is the forked child servicing this mailbox, or -1 when the
    // caller does not own a waitable child. A valid pid enables liveness
    // checks that turn a child that dies before publishing TASK_DONE /
    // CONTROL_DONE into a reported failure instead of an endless spin-poll.
    LocalMailboxEndpoint(
        int32_t worker_id, void *mailbox, size_t mailbox_bytes, int child_pid = -1, uint32_t granted_sets = 1
    );

    const WorkerEndpointCaps &caps() const override { return caps_; }
    void submit_progress(Ring *ring, const WorkerDispatch &dispatch) override;
    size_t available_request_transport() override;
    bool poll_progress(WorkerEndpointProgress &progress) override;
    bool activate_progress(RunId run_id) override;
    void request_progress_stop() noexcept override;
    void report_progress_error(const std::string &reason) override;
    bool report_submission_error(const WorkerDispatch &dispatch, const std::string &reason) override;

    void shutdown_child() override;
    uint64_t control_malloc(size_t size) override;
    uint64_t control_committed_device_memory() override;
    DeviceMemoryInfo control_device_memory_info() override;
    DfxFlushReport control_dfx_flush(double timeout_s) override;
    void control_free(uint64_t ptr) override;
    void control_copy_to(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span) override;
    void control_copy_from(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span) override;
    void control_prepare(const uint8_t *digest) override;
    void control_register(const char *shm_name, size_t blob_size, const uint8_t *digest) override;
    void control_unregister(const uint8_t *digest) override;
    void control_remote_prepare_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest,
        const void *payload, size_t payload_size
    ) override;
    void control_remote_commit_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    ) override;
    void control_remote_abort_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    ) override;
    void control_remote_unregister(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    ) override;
    RemoteBufferHandle control_remote_malloc(size_t size) override;
    void control_remote_free(const RemoteBufferHandle &handle) override;
    void
    control_remote_copy_to(const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size) override;
    void control_remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size) override;
    RemoteBufferExport control_remote_export(
        const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
        const std::string &transport_profile
    ) override;
    RemoteBufferHandle control_remote_import(
        int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
    ) override;
    void control_remote_release_import(const RemoteBufferHandle &handle) override;
    void control_generic(
        uint64_t sub_cmd, const char *shm_name, size_t payload_size, double timeout_s, const uint8_t *digest
    ) override;
    void control_alloc_domain(const char *request_shm_name, const char *reply_shm_name) override;
    void control_release_domain(const char *request_shm_name) override;
    void control_comm_init(const char *request_shm_name) override;

private:
    WorkerEndpointCaps caps_;
    void *mailbox_{nullptr};
    std::mutex mailbox_mu_;
    std::mutex child_mu_;
    std::mutex progress_mu_;
    // How many runs this endpoint may hold at once: the granted sets plus the
    // hand-offs in flight, which is the entry count. Distinct from the request
    // frames, of which there are always `MAILBOX_REQUEST_FRAMES`.
    uint32_t granted_sets_{1};
    bool endpoint_poisoned_{false};
    // Only child-exit poison permits consumption of independently published task results.
    bool preserve_terminal_results_on_exit_{false};
    std::string endpoint_poison_reason_;
    size_t poll_cursor_{0};
    std::chrono::steady_clock::time_point next_liveness_check_{};
    std::atomic<bool> mailbox_control_timed_out_{false};
    int child_pid_{-1};
    // Set once the child has been reaped or observed unwaitable; the exit
    // status is only available from the reaping waitpid(), so it is retained
    // here to describe every later operation on this dead mailbox.
    bool child_dead_{false};
    std::string child_death_reason_;

    char *mbox() const { return static_cast<char *>(mailbox_); }
    MailboxState read_mailbox_state(const char *frame = nullptr) const;
    void write_mailbox_state(MailboxState s, char *frame = nullptr);
    /** The control frame, or request frame `index`. */
    char *task_frame(size_t index) const;
    /** Run-state entry `index`. The table follows the request frames. */
    MailboxRunEntry *run_entry(size_t index) const;
    void run_control_command(const char *op_name, double timeout_s = -1.0);

    /** One run's state as this endpoint reads it, in one coherent pass. */
    struct EntrySnapshot {
        MailboxState state{MailboxState::IDLE};
        bool accepted{false};
        MailboxPreparationDisposition disposition{MailboxPreparationDisposition::NONE};
    };

    /**
     * What the parent remembers about one run-state entry.
     *
     * Paired with the shared `MailboxRunEntry` it names: this side is the
     * parent's own bookkeeping — what it has already reported upward — and the
     * shared side is what the child and the device publish.
     */
    struct EntryRecord {
        bool occupied{false};
        bool staged_reported{false};
        bool accepted_reported{false};
        bool activation_requested{false};
        bool activation_published{false};
        // The disposition this run was last reported with. A staged run may be
        // prepared later than it was staged — its own predecessor had not
        // launched yet when it arrived — and that promotion is published by the
        // child into the same entry. Remembering what was reported is what lets
        // the change be reported once more instead of being lost.
        MailboxPreparationDisposition reported_disposition{MailboxPreparationDisposition::NONE};
        WorkerDispatch dispatch{};
        RunId run_id{INVALID_RUN_ID};
        uint64_t slot_id{0};
        uint64_t generation{0};
        uint64_t group_size{0};
        // Which request frame carried this run's payload, and that frame's
        // generation. The parent reclaims the frame when the child echoes the
        // generation back, which is a level rather than a pulse — so an
        // observation missed on one poll is seen on the next.
        size_t request_frame{kNoRequestFrame};
        uint64_t request_frame_generation{0};
        // This entry's own generation, odd while occupied. Every cross-side
        // reference carries it, so a stale one is refused rather than acted on.
        uint64_t entry_generation{0};
    };
    static constexpr size_t kNoRequestFrame = static_cast<size_t>(-1);
    std::vector<EntryRecord> entries_;

    /** Per request frame: whether the parent still owns it, and its generation. */
    struct RequestFrameRecord {
        bool published{false};
        uint64_t generation{0};
        size_t entry{kNoRequestFrame};
    };
    std::vector<RequestFrameRecord> request_frames_;
    uint64_t next_frame_generation_{1};
    uint64_t next_entry_generation_{1};

    /** A free run-state entry, or `entries_.size()` when every one is occupied. */
    size_t free_entry_locked() const;
    /** A request frame the parent still owns, or `request_frames_.size()`. */
    size_t free_request_frame_locked() const;
    /**
     * Give back every request frame whose payload the child has copied out.
     *
     * The decision is a level: the child stores the frame's own generation into
     * the entry it belongs to, and this compares the two. A frame reclaimed
     * twice is a no-op, and one whose echo is observed late is reclaimed on the
     * next poll instead of being lost.
     */
    void reclaim_consumed_request_frames_locked();
    /** Free entry `index` and publish the even generation that retires its references. */
    void release_entry_locked(size_t index);

    bool entry_identity_matches(const EntryRecord &record, const MailboxRunEntry *entry) const;
    bool try_publish_activation(EntryRecord &record, MailboxRunEntry *entry);
    void poison_progress(const std::string &reason);
    bool poisoned_progress_quiesced();
    WorkerCompletion poisoned_completion(const EntryRecord &record) const;
    WorkerCompletion
    terminal_completion(const EntryRecord &record, const MailboxRunEntry *entry, MailboxState state) const;

    // Returns a description of the child's death, or an empty string while it
    // is still running. Never throws: callers decide whether a dead child is
    // reported as a completion outcome or an exception.
    std::string check_child_death();
};

// =============================================================================
// WorkerThread — one Scheduler-driven endpoint lane.
// =============================================================================

class WorkerThread {
public:
    WorkerThread() = default;
    ~WorkerThread() = default;
    WorkerThread(const WorkerThread &) = delete;
    WorkerThread &operator=(const WorkerThread &) = delete;

    // Initialize the endpoint lane.
    //
    // `mailbox` points to a `mailbox_size(granted_sets)`-byte MAP_SHARED region managed
    // by the Python facade — the real worker (a `ChipWorker` for
    // NEXT_LEVEL, a Python callable for SUB) lives in the forked child
    // and consumes the mailbox via `_chip_process_loop` / `_sub_worker_loop`.
    //
    // `ring` is a borrowed pointer to the engine's slot-state pool —
    // the Scheduler thread reads callable/args/config from
    // `ring->slot_state(task_slot)` on each dispatch.
    // on_complete(completion) is called after each endpoint run.
    void start(
        Ring *ring, const std::function<void(WorkerCompletion)> &on_complete,
        const std::function<void(WorkerDispatch)> &on_accept, const std::function<void(WorkerDispatch)> &on_staged,
        std::unique_ptr<WorkerEndpoint> endpoint
    );
    /** No staging announcement: for a caller that cannot launch a staged run early. */
    void start(
        Ring *ring, const std::function<void(WorkerCompletion)> &on_complete,
        const std::function<void(WorkerDispatch)> &on_accept, std::unique_ptr<WorkerEndpoint> endpoint
    ) {
        start(ring, on_complete, on_accept, {}, std::move(endpoint));
    }

    // Submit a dispatch to the endpoint. Non-blocking.
    void dispatch(WorkerDispatch d);
    void dispatch_prepared(WorkerDispatch d);
    // Complete a slot the scheduler already claimed when publication fails.
    // No endpoint capacity was consumed, but the run-level acceptance waiter
    // still needs its conservative terminal fallback.
    void complete_unpublished(WorkerDispatch d, const std::string &error_message);
    bool has_staged_run(RunId run_id) const;
    bool activate_prepared(RunId run_id);

    /**
     * Authorize the staged run to launch its device work now, without moving it
     * out of the staged lane.
     *
     * Distinct from `activate_prepared`, which promotes a staged run into the
     * active lane and therefore requires that lane to be free. Here the
     * predecessor is still executing and still owns the active lane and its
     * identity; only the staged run's own activation is requested. Reports
     * whether this call is the one that requested it.
     */
    bool authorize_staged_launch(RunId run_id);
    void progress();

    // The active lane and staged-successor lane are intentionally distinct.
    // A staged successor must not make a second task from the active run
    // dispatchable on the same device.
    bool idle() const;
    /**
     * Whether an ordinary dispatch can be published right now.
     *
     * `idle()` alone answers only for the lane. A publication also needs a
     * request frame, and transport is bounded independently of the grant, so
     * this is the gate an admitter consults before claiming a task for the
     * ordinary lane — the same obligation `can_stage` carries for the staged
     * ones.
     */
    bool can_dispatch() const;
    bool can_stage() const;
    bool busy() const { return inflight_.load(std::memory_order_acquire) != 0; }
    const WorkerEndpointCaps &caps() const;
    int32_t worker_id() const;

    void stop();

    // Write SHUTDOWN to the mailbox so the child process exits its loop.
    // Does NOT waitpid — the Python facade owns the child PID.
    void shutdown_child();

    // Memory control — callable from the orch thread while the Scheduler thread
    // may be running a task. Commands serialize with each other on
    // `mailbox_mu_`. Task dispatch uses separate task frames, so the single
    // child progress owner may observe a control request while a task is active.
    uint64_t control_malloc(size_t size);
    uint64_t control_committed_device_memory();
    DeviceMemoryInfo control_device_memory_info();
    DfxFlushReport control_dfx_flush(double timeout_s);
    void control_free(uint64_t ptr);
    void control_copy_to(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span);
    void control_copy_from(const BufferDescriptor &dst, const BufferDescriptor &src, const CopySpan &span);

    // Pre-warm a chip child by triggering simpler_register_callable for the digest's
    // target-local slot via CTRL_PREPARE.
    void control_prepare(const uint8_t *digest);

    // Dynamic post-init register/unregister of a ChipCallable identity.
    // `shm_name` is the (NUL-terminated, ≤ CTRL_SHM_NAME_BYTES-1) POSIX shm
    // name where the ChipCallable bytes are staged; `blob_size` is the exact
    // byte span to read from that shm. Both methods hold mailbox_mu_; a child
    // additionally preserves any callable still referenced by an outstanding
    // task frame.
    void control_register(const char *shm_name, size_t blob_size, const uint8_t *digest);
    void control_unregister(const uint8_t *digest);
    void control_remote_prepare_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest,
        const void *payload, size_t payload_size
    );
    void control_remote_commit_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    );
    void control_remote_abort_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    );
    void control_remote_unregister(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    );
    RemoteBufferHandle control_remote_malloc(size_t size);
    void control_remote_free(const RemoteBufferHandle &handle);
    void control_remote_copy_to(const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size);
    void control_remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size);
    RemoteBufferExport control_remote_export(
        const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
        const std::string &transport_profile
    );
    RemoteBufferHandle control_remote_import(
        int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
    );
    void control_remote_release_import(const RemoteBufferHandle &handle);
    std::vector<uint8_t> control_remote_domain(
        remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes, bool group_target = false
    );
    void control_generic(
        uint64_t sub_cmd, const char *shm_name, size_t payload_size, double timeout_s, const uint8_t *digest
    );

    // Dynamic CommDomain allocate / release.  `request_shm_name` carries the
    // request payload (header + rank_ids + buffer_nbytes); for alloc the child
    // writes its (device_ctx, local_window_base, buffer_ptrs) into
    // `reply_shm_name`.  Both names are NUL-terminated and ≤
    // CTRL_SHM_NAME_BYTES-1. Holds mailbox_mu_ so control commands serialize;
    // task dispatch uses distinct frame state.
    void control_alloc_domain(const char *request_shm_name, const char *reply_shm_name);
    void control_release_domain(const char *request_shm_name);

    // Lazy comm_init driver — payload shm carries (rank, nranks, rootinfo_path).
    // Caller dispatches in parallel to every chip; child runs cw.comm_init.
    void control_comm_init(const char *request_shm_name);

private:
    enum class SubmitDispatchResult : uint8_t {
        SUBMITTED,
        STOPPING,
        CAPACITY_EXCEEDED,
        STAGED_IDENTITY_CHANGED,
        // Publication threw and the failure has already been delivered — by the
        // endpoint, which may own a partially published dispatch, or by this
        // worker's own terminal. Distinct from SUBMITTED on purpose: the two
        // were one value, and a caller that cannot tell them apart cannot tell
        // "the run is live" from "the run is already failed". Both mean the
        // caller reports nothing further, and neither means the lane is free.
        PUBLICATION_REPORTED,
    };

    enum class LaneKind : uint8_t {
        ACTIVE = 0,
        STAGED = 1,
    };

    struct LaneState {
        bool occupied{false};
        bool activation_requested{false};
        RunId run_id{INVALID_RUN_ID};
        uint64_t dispatch_id{0};
        // What the child reported it did with the staged frame. Staging admits
        // both dispositions, and only a natively prepared one holds work that
        // can be ordered behind another run — a validated-only frame fell back
        // to depth one and has nothing to launch early. Recorded when the
        // staging event arrives, so the authorization can refuse rather than
        // spend an activation the child will decline.
        MailboxPreparationDisposition preparation_disposition{MailboxPreparationDisposition::NONE};
    };

    Ring *ring_{nullptr};
    std::unique_ptr<WorkerEndpoint> endpoint_;
    std::function<void(WorkerCompletion)> on_complete_;
    std::function<void(WorkerDispatch)> on_accept_;
    // A run's dispatch was staged at its child. Non-throwing by contract, like
    // the two above: it runs on the progress driver.
    std::function<void(WorkerDispatch)> on_staged_;
    std::atomic<bool> shutdown_{false};
    std::atomic<uint32_t> inflight_{0};
    uint64_t next_dispatch_id_{1};
    // Linearizes stop with endpoint publication and prepared activation.
    std::mutex admission_mu_;
    mutable std::mutex lane_mu_;
    // Index 0 is the active lane; the rest are staged lanes, one per run this
    // endpoint may hold beyond the active one. A staged lane is found by the
    // identity it carries rather than by position — the endpoint picks a mailbox
    // frame from the run's own pipeline lease, so which staged lane a run
    // occupies says nothing about where its frame is.
    // Sized at `start` to the endpoint's capacity, which is the count of runs
    // its child may hold at once.
    std::vector<LaneState> lanes_;
    // Scratch for `progress`'s activation pass, sized beside `lanes_` so that
    // path allocates nothing. The progress thread is its only reader and writer.
    std::vector<RunId> activation_scratch_;
    std::unordered_set<uint64_t> accepted_dispatch_ids_;
    std::unordered_map<uint64_t, std::string> accept_errors_;

    SubmitDispatchResult submit_dispatch(WorkerDispatch d, size_t lane_index, RunId expected_run_id = INVALID_RUN_ID);
    static constexpr size_t kActiveLane = 0;
    static constexpr size_t kFirstStagedLane = 1;
    LaneState &lane(LaneKind kind) { return lanes_[static_cast<size_t>(kind)]; }
    const LaneState &lane(LaneKind kind) const { return lanes_[static_cast<size_t>(kind)]; }
    LaneState &lane_at(size_t index) { return lanes_[index]; }
    /** How many staged lanes this endpoint may use, bounded by what it admits at once. */
    size_t staged_lane_count() const {
        const uint32_t inflight_cap = caps().max_inflight_tasks;
        const size_t usable = inflight_cap > 1 ? static_cast<size_t>(inflight_cap) : 1;
        return (usable < lanes_.size() ? usable : lanes_.size()) - 1;
    }
    /** The staged lane holding `run_id`, or `lanes_.size()` when none does. */
    size_t staged_lane_of_locked(RunId run_id) const {
        for (size_t index = kFirstStagedLane; index < kFirstStagedLane + staged_lane_count(); ++index) {
            if (lanes_[index].occupied && lanes_[index].run_id == run_id) return index;
        }
        return lanes_.size();
    }
    /** The staged lane holding `dispatch_id`, or `lanes_.size()` when none does. */
    size_t staged_lane_for_dispatch_locked(uint64_t dispatch_id) const {
        for (size_t index = kFirstStagedLane; index < kFirstStagedLane + staged_lane_count(); ++index) {
            if (lanes_[index].occupied && lanes_[index].dispatch_id == dispatch_id) return index;
        }
        return lanes_.size();
    }
    /** A free staged lane, or `lanes_.size()` when every one is taken. */
    size_t free_staged_lane_locked() const {
        for (size_t index = kFirstStagedLane; index < kFirstStagedLane + staged_lane_count(); ++index) {
            if (!lanes_[index].occupied) return index;
        }
        return lanes_.size();
    }
    void release_lane(size_t lane_index, uint64_t dispatch_id);
    void release_lane_unconditional(size_t lane_index);
    void finish_progress_dispatch(const WorkerEndpointProgress &progress);
    void fail_submission(const WorkerDispatch &dispatch, const std::string &reason);
    void fail_progress_driver(const std::string &reason) noexcept;
};

// =============================================================================
// WorkerManager — worker pool lifecycle and dispatch
// =============================================================================

class WorkerManager {
public:
    using OnCompleteFn = std::function<void(WorkerCompletion)>;
    using OnAcceptFn = std::function<void(WorkerDispatch)>;
    using OnStagedFn = std::function<void(WorkerDispatch)>;

    // Register a worker. `mailbox` is a MAP_SHARED region of `mailbox_bytes`
    // bytes, which must hold the `mailbox_size(granted_sets)` the grant implies
    // — the registration is refused here, before any entry is installed, rather
    // than when the endpoint first writes into the table. The real worker (a
    // `ChipWorker` for NEXT_LEVEL, a Python callable for SUB) lives in the
    // forked child.
    // `child_pid` is the forked child servicing `mailbox`; pass -1 when the
    // caller owns no waitable child.
    void add_next_level(void *mailbox, size_t mailbox_bytes, int child_pid = -1, uint32_t granted_sets = 1);
    void add_next_level_at(
        int32_t worker_id, void *mailbox, size_t mailbox_bytes, int child_pid = -1, uint32_t granted_sets = 1
    );
    void add_next_level_endpoint(std::unique_ptr<WorkerEndpoint> endpoint);
    void add_sub(void *mailbox, size_t mailbox_bytes, int child_pid = -1);

    // `on_complete` and `on_accept` are non-throwing contracts: the endpoint lane invokes
    // `on_complete` after endpoint ownership has ended, so no lower layer can
    // reconstruct run accounting if it unwinds. `on_accept` advances the run's
    // launch fence; pass an empty function only when nothing waits on that
    // fence, since an omitted callback leaves pending_accepts non-zero forever.
    // No default: the choice is the caller's.
    // `on_staged` is optional: it announces that a child holds a prepared
    // frame, which only a caller that can launch a staged run early has any use
    // for. Non-throwing on the same terms as the other two.
    void
    start(Ring *ring, const OnCompleteFn &on_complete, const OnAcceptFn &on_accept, const OnStagedFn &on_staged = {});
    void stop_workers();
    void stop();
    void progress();

    WorkerThread *get_worker_by_id(WorkerType type, int32_t worker_id) const;
    std::vector<int32_t> next_level_worker_ids() const;

    // SUB has no worker-selection API. Pick any idle SUB worker not already
    // selected for the same group dispatch.
    WorkerThread *pick_idle_sub_excluding(const std::vector<WorkerThread *> &exclude) const;

    bool any_busy() const;
    bool has_staged_run(RunId run_id) const;
    bool activate_prepared_run(RunId run_id);
    /** @see WorkerThread::authorize_staged_launch. */
    bool authorize_staged_launch(RunId run_id);

    // Forward CTRL_PREPARE to a specific NEXT_LEVEL worker. Thin wrapper
    // over WorkerThread::control_prepare; exposed at manager level so the
    // Python facade can prewarm without reaching into individual endpoint lanes.
    void control_prepare(int worker_id, const uint8_t *digest);

    // Forward CTRL_ALLOC_DOMAIN / CTRL_RELEASE_DOMAIN to a specific NEXT_LEVEL
    // worker.  Used by the Python orch facade to drive collective domain
    // allocation across a subset of chips — caller dispatches to each
    // participating chip and joins on completion.
    void control_alloc_domain(int worker_id, const char *request_shm_name, const char *reply_shm_name);
    void control_release_domain(int worker_id, const char *request_shm_name);
    void control_comm_init(int worker_id, const char *request_shm_name);
    ControlResult
    control_digest_only(WorkerType type, int worker_id, uint64_t sub_cmd, const uint8_t *digest, double timeout_s);
    std::vector<uint8_t> control_payload(
        WorkerType type, int worker_id, uint64_t sub_cmd, const void *payload, size_t payload_size, double timeout_s
    );
    ControlResult control_remote_prepare_register(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const void *payload,
        size_t payload_size, const uint8_t *digest
    );
    ControlResult control_remote_commit_register(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind,
        const uint8_t *digest
    );
    ControlResult control_remote_abort_register(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind,
        const uint8_t *digest
    );
    ControlResult control_remote_unregister(
        int worker_id, remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind,
        const uint8_t *digest
    );
    RemoteBufferHandle control_remote_malloc(int worker_id, size_t size);
    void control_remote_free(const RemoteBufferHandle &handle);
    void control_remote_copy_to(const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size);
    void control_remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size);
    RemoteBufferExport control_remote_export(
        const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
        const std::string &transport_profile
    );
    RemoteBufferHandle control_remote_import(
        int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
    );
    void control_remote_release_import(const RemoteBufferHandle &handle);
    std::vector<uint8_t> control_remote_domain(
        int worker_id, remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes,
        bool group_target = false
    );

    // Broadcast CTRL_REGISTER for `digest` to every NEXT_LEVEL worker in
    // parallel. Stages `blob_size` bytes from `blob_ptr` into a per-call
    // POSIX shm under name "simpler-cb-<pid>-<counter>", spawns one
    // std::thread per endpoint lane, and joins. Returns one ControlResult per
    // target so the Python facade can clean up only targets that confirmed
    // install/refcount increment on a partial failure.
    std::vector<ControlResult> broadcast_register_all(const void *blob_ptr, size_t blob_size, const uint8_t *digest);

    // Best-effort: broadcast CTRL_UNREGISTER for `digest` to every NEXT_LEVEL
    // worker in parallel. Returns a vector of per-worker error strings
    // (empty on full success). Caller decides whether to log / surface.
    std::vector<std::string> broadcast_unregister_all(const uint8_t *digest);
    std::vector<ControlResult> broadcast_control_all(
        WorkerType type, uint64_t sub_cmd, const void *payload, size_t payload_size, const uint8_t *digest,
        double timeout_s
    );

private:
    struct LocalNextLevelEntry {
        int32_t worker_id{-1};
        void *mailbox{nullptr};
        // What the registering caller mapped at `mailbox`, carried so the
        // endpoint built from this entry checks the region that exists rather
        // than the one the grant implies.
        size_t mailbox_bytes{0};
        int child_pid{-1};
        uint32_t granted_sets{1};
    };
    struct LocalSubEntry {
        void *mailbox{nullptr};
        size_t mailbox_bytes{0};
        int child_pid{-1};
    };
    std::vector<LocalNextLevelEntry> next_level_entries_;
    std::vector<LocalSubEntry> sub_entries_;
    std::vector<std::unique_ptr<WorkerEndpoint>> next_level_endpoint_entries_;

    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;
};
