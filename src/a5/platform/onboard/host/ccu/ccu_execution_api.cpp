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
#include "host/ccu_execution_registry.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

namespace {
constexpr uint32_t kProtocolTimeoutMs = 5000;
enum Phase : uint32_t { BIND = 1, PREPARE = 2, ARM = 3, CANCEL = 4, RETIRE = 5 };
using Wire = std::array<uint64_t, 32>;
static_assert(sizeof(Wire) == 32 * sizeof(uint64_t));

struct Failure {
    int status;
    const char *message;
    int sdk_error{};
};
void Require(bool condition, int status, const char *message) {
    if (!condition) throw Failure{status, message};
}

void Result(CcuHostResultV1 *out, int status, CcuStageV1 stage, const char *message) {
    if (!out) return;
    *out = {};
    out->abi_version = CCU_EXECUTION_ABI_V1;
    out->struct_size = sizeof(*out);
    out->status = status;
    out->stage = stage;
    out->retained = status == CCU_STATUS_RETAINED;
    std::strncpy(out->message, message, sizeof(out->message) - 1);
}

template <typename Fn>
int Protect(CcuHostResultV1 *out, CcuStageV1 stage, Fn &&fn) noexcept {
    Result(out, CCU_STATUS_SUCCESS, stage, "Success");
    try {
        return fn();
    } catch (const Failure &e) {
        Result(out, e.status, stage, e.message);
        if (out) out->sdk_error = e.sdk_error;
        return e.status;
    } catch (const CcuSdkError &e) {
        Result(out, CCU_STATUS_SDK_ERROR, stage, e.what());
        if (out) out->sdk_error = e.code;
        return CCU_STATUS_SDK_ERROR;
    } catch (...) {
        Result(out, CCU_STATUS_INTERNAL_ERROR, stage, "Unexpected exception at C ABI boundary");
        return CCU_STATUS_INTERNAL_ERROR;
    }
}

int CurrentSdkError() noexcept {
    try {
        throw;
    } catch (const CcuSdkError &e) {
        return e.code;
    } catch (...) {
        return 0;
    }
}

// One operation owns side effects; neither SDK waits nor bootstrap callbacks hold mutex.
// 一个操作独占副作用，但 SDK 和控制面回调均不持有 context mutex。
class Operation {
public:
    explicit Operation(std::shared_ptr<CcuContextEntryV1> value) :
        ctx(std::move(value)) {
        Require(bool(ctx), CCU_STATUS_STALE_HANDLE, "Unknown context");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        Require(!ctx->busy, CCU_STATUS_BUSY, "Another context operation is active");
        ctx->busy = true;
    }
    ~Operation() {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->busy = false;
    }
    std::shared_ptr<CcuContextEntryV1> ctx;
};

std::shared_ptr<CcuContextEntryV1> Context(CcuContextId id) {
    return CcuExecutionRegistryV1::Instance().GetContext(id);
}
std::shared_ptr<CcuRequestEntryV1> Request(CcuRequestId id) {
    auto r = CcuExecutionRegistryV1::Instance().GetRequest(id);
    Require(bool(r), CCU_STATUS_STALE_HANDLE, "Unknown request");
    return r;
}
bool CleanupComplete(const CcuRequestEntryV1 &r) {
    const auto phase = r.state.Phase();
    return phase == CcuRequestPhaseV1::CANCELLED || phase == CcuRequestPhaseV1::RETIRED;
}
void Active(const CcuContextEntryV1 &ctx, const std::shared_ptr<CcuRequestEntryV1> &r, bool cleanup = false) {
    Require(ctx.request == r && (cleanup || !CleanupComplete(*r)), CCU_STATUS_STALE_HANDLE, "Request has retired");
}
bool ValidOps(const CcuOwnerOpsV1 *ops) {
    return ops && ops->abi_version == CCU_EXECUTION_ABI_V1 && ops->struct_size == sizeof(*ops) && ops->retain &&
           ops->release;
}

Wire Header(const CcuContextEntryV1 &ctx, uint64_t collective, Phase phase) {
    Wire wire{};
    wire[0] = CCU_EXECUTION_ABI_V1;
    wire[1] = sizeof(Wire);
    wire[2] = ctx.options.group_id;
    wire[3] = collective;
    wire[4] = phase;
    wire[5] = ctx.options.rank;
    wire[6] = ctx.id;
    wire[7] = ctx.generation;
    return wire;
}
std::array<Wire, 2> Exchange(CcuContextEntryV1 &ctx, const Wire &local, uint32_t timeout = kProtocolTimeoutMs) {
    std::array<Wire, 2> peers{};
    int rc = ctx.bootstrap.exchange(
        ctx.bootstrap_owner, ctx.options.group_id, local[3], local[4], local.data(), sizeof(local), peers.data(),
        sizeof(Wire), 2, timeout
    );
    if (rc != 0 && ctx.bootstrap.report_abort)
        (void)ctx.bootstrap.report_abort(ctx.bootstrap_owner, ctx.options.group_id, local[3], local[4], rc);
    Require(rc == 0, CCU_STATUS_PROTOCOL_ERROR, "Two-rank exchange failed");
    for (uint32_t rank = 0; rank < 2; ++rank) {
        const auto &w = peers[rank];
        Require(
            w[0] == local[0] && w[1] == local[1] && w[2] == local[2] && w[3] == local[3] && w[4] == local[4] &&
                w[5] == rank && w[6] != 0 && w[7] != 0,
            CCU_STATUS_PROTOCOL_ERROR, "Wrong rank, phase, group or wire identity"
        );
        if (ctx.bound)
            Require(
                w[6] == ctx.peer_contexts[rank] && w[7] == ctx.peer_generations[rank], CCU_STATUS_PROTOCOL_ERROR,
                "Peer context generation changed"
            );
    }
    Require(peers[ctx.options.rank] == local, CCU_STATUS_PROTOCOL_ERROR, "Local exchange echo differs");
    return peers;
}

std::shared_ptr<CcuWindowLeaseV1> Window(CcuContextEntryV1 &ctx, CcuWindowId id) {
    const auto it = ctx.windows.find(id);
    Require(it != ctx.windows.end(), CCU_STATUS_STALE_HANDLE, "Unknown window");
    return it->second;
}
void PutWindow(Wire &w, size_t at, const CcuWindowLeaseV1 &window) {
    const auto &i = window.import;
    w[at] = i.allocation_id;
    w[at + 1] = i.allocation_generation;
    w[at + 2] = i.allocation_base_va;
    w[at + 3] = i.allocation_size;
    w[at + 4] = i.window_offset;
    w[at + 5] = i.window_size;
    w[at + 6] = static_cast<uint64_t>(i.device_id);
    w[at + 7] = window.token;
}
CcuPeerWindowV1 ReadWindow(const Wire &w, size_t at) {
    Require(
        w[at] && w[at + 1] && w[at + 2] && w[at + 3] && w[at + 5] && w[at + 6] <= INT32_MAX && w[at + 4] <= w[at + 3] &&
            w[at + 5] <= w[at + 3] - w[at + 4] && w[at + 3] <= UINT64_MAX - w[at + 2],
        CCU_STATUS_PROTOCOL_ERROR, "Invalid peer allocation extent"
    );
    return {w[at], w[at + 1], w[at + 2], w[at + 3], w[at + 4], w[at + 5], static_cast<int32_t>(w[at + 6]),
            0,     w[at + 7]};
}

void Details(CcuHostResultV1 *out, const CcuRequestEntryV1 &r) {
    if (!out) return;
    out->expected_sequence = r.descriptor.request_seq;
    out->observed_sequence = r.observed;
    out->data_ready = r.data_ready;
    out->drained = r.drained;
    out->submission_state = static_cast<uint32_t>(r.trigger_attempted ? r.trigger_submission : r.submission);
}

int Poll(CcuContextEntryV1 &ctx, CcuRequestEntryV1 &r) {
    if (r.drained) return CCU_STATUS_SUCCESS;
    const auto phase = r.state.Phase();
    Require(
        phase == CcuRequestPhaseV1::IN_FLIGHT || phase == CcuRequestPhaseV1::DATA_READY ||
            phase == CcuRequestPhaseV1::RETAINED,
        CCU_STATUS_INVALID_STATE, "Request is not in flight"
    );
    if (!r.trigger_attempted || r.state.Poisoned()) return CCU_STATUS_RETAINED;
    const auto observation = ctx.device->Observe();
    const auto &record = observation.record;
    if (ccu_validate_completion_header_v1(record) != CcuRequestValidationError::NONE || record.context_id != ctx.id ||
        record.context_generation != ctx.generation) {
        (void)r.state.RetainFault();
        throw Failure{CCU_STATUS_RETAINED, "Completion record identity mismatch"};
    }
    r.observed = record.completed_seq;
    if (r.observed > r.descriptor.request_seq) {
        (void)r.state.RetainFault();
        throw Failure{CCU_STATUS_RETAINED, "Completion sequence is ahead of request"};
    }
    r.data_ready = r.observed == r.descriptor.request_seq;
    if (phase == CcuRequestPhaseV1::RETAINED) {
        if (r.state.RecoverDrained(r.observed, observation.trigger_drained, observation.ccu_drained) ==
            CcuStateResultV1::OK)
            r.drained = true;
    } else {
        (void)r.state.ObserveCompletion(r.observed);
        if (r.state.MarkDrained(observation.trigger_drained, observation.ccu_drained) == CcuStateResultV1::OK)
            r.drained = true;
    }
    if (r.drained) return CCU_STATUS_SUCCESS;
    if (r.state.Phase() == CcuRequestPhaseV1::RETAINED) return CCU_STATUS_RETAINED;
    return CCU_STATUS_PENDING;
}

using Clock = std::chrono::steady_clock;
int Wait(CcuContextEntryV1 &ctx, CcuRequestEntryV1 &r, Clock::time_point deadline) {
    for (;;) {
        const int status = Poll(ctx, r);
        if (status == CCU_STATUS_SUCCESS || r.state.Poisoned()) return status;
        if (Clock::now() >= deadline) {
            (void)r.state.RetainPending();
            return CCU_STATUS_RETAINED;
        }
        std::this_thread::yield();
    }
}

uint32_t Remaining(Clock::time_point deadline) {
    return static_cast<uint32_t>(
        std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count())
    );
}

void Retire(CcuContextEntryV1 &ctx, CcuRequestEntryV1 &r, Clock::time_point deadline) {
    const bool replay = r.state.Phase() == CcuRequestPhaseV1::RETIRED;
    if (!replay) {
        try {
            Require(
                Wait(ctx, r, deadline) == CCU_STATUS_SUCCESS, CCU_STATUS_RETAINED, "Cannot retire undrained request"
            );
        } catch (const CcuSdkError &e) {
            (void)r.state.RetainFault();
            throw Failure{CCU_STATUS_RETAINED, "Completion query failed during retirement", e.code};
        }
        Require(r.state.BeginRetire() == CcuStateResultV1::OK, CCU_STATUS_INVALID_STATE, "Cannot begin retire");
    }
    bool agreed = false;
    try {
        auto wire = Header(ctx, r.descriptor.collective_id, RETIRE);
        wire[8] = ctx.binding_generation;
        const auto peers = Exchange(ctx, wire, Remaining(deadline));
        agreed = peers[0][8] == wire[8] && peers[1][8] == wire[8];
    } catch (...) {}
    Require(
        replay ? agreed : r.state.FinishRetire(agreed) == CcuStateResultV1::OK, CCU_STATUS_RETAINED,
        "Peer retirement unconfirmed"
    );
}

void Cancel(CcuContextEntryV1 &ctx, CcuRequestEntryV1 &r, uint32_t timeout) {
    Require(!r.launch_attempted, CCU_STATUS_INVALID_STATE, "Only unsubmitted requests can cancel");
    const bool replay = r.state.Phase() == CcuRequestPhaseV1::CANCELLED;
    bool agreed = false;
    try {
        (void)Exchange(ctx, Header(ctx, r.descriptor.collective_id, CANCEL), timeout);
        agreed = true;
    } catch (...) {}
    Require(
        replay ? agreed : r.state.CancelPrepared(agreed) == CcuStateResultV1::OK, CCU_STATUS_RETAINED,
        "Peer cancellation unconfirmed"
    );
}
}  // namespace

extern "C" int ccu_query_capabilities(CcuCapabilitiesV1 *out) {
    return Protect(nullptr, CCU_STAGE_CAPABILITY, [&] {
        Require(out && out->struct_size >= sizeof(*out), CCU_STATUS_INVALID_ARGUMENT, "Capabilities buffer");
        *out = {};
        out->abi_version = CCU_EXECUTION_ABI_V1;
        out->struct_size = sizeof(*out);
        out->arch = 5;
        out->max_rank_count = 2;
        out->max_element_count = 1024;
        out->max_inflight = 1;
        if (ccu_device_available()) {
            out->flags = CCU_CAPABILITY_SUPPORTED | CCU_CAPABILITY_UBC_CTP | CCU_CAPABILITY_COMPLETION_STORE;
            out->supported_operation_mask = 1U << CCU_OPERATION_BROADCAST;
            out->supported_dtype_mask = 1U << CCU_DATA_TYPE_FLOAT32;
        }
        return CCU_STATUS_SUCCESS;
    });
}
extern "C" int ccu_get_root_info(int32_t device, void *bytes, uint32_t capacity, uint32_t *size_out) {
    return Protect(nullptr, CCU_STAGE_CREATE, [&] {
        Require(size_out, CCU_STATUS_INVALID_ARGUMENT, "Missing root-info size output");
        *size_out = ccu_device_root_info_size();
        Require(
            device >= 0 && bytes && capacity >= *size_out, CCU_STATUS_INVALID_ARGUMENT, "Root-info buffer too small"
        );
        Require(ccu_device_available(), CCU_STATUS_UNSUPPORTED, "Missing CCU SDK symbols");
        ccu_device_root_info(device, bytes);
        return CCU_STATUS_SUCCESS;
    });
}
extern "C" int ccu_context_create(const CcuCreateOptionsV1 *o, CcuContextId *out, CcuHostResultV1 *result) {
    if (out) *out = 0;
    return Protect(result, CCU_STAGE_CREATE, [&] {
        Require(
            o && out && o->abi_version == CCU_EXECUTION_ABI_V1 && o->struct_size == sizeof(*o) && o->device_id >= 0 &&
                o->rank_count == 2 && o->rank < 2 && o->group_id && o->root_info &&
                o->root_info_size == ccu_device_root_info_size() && o->bootstrap_ops &&
                o->bootstrap_ops->abi_version == CCU_EXECUTION_ABI_V1 &&
                o->bootstrap_ops->struct_size == sizeof(CcuBootstrapOpsV1) && o->bootstrap_ops->exchange &&
                o->bootstrap_owner && ValidOps(o->bootstrap_owner_ops),
            CCU_STATUS_INVALID_ARGUMENT, "Invalid create options"
        );
        Require(ccu_device_available(), CCU_STATUS_UNSUPPORTED, "Missing CCU SDK symbols");
        auto ctx = std::make_shared<CcuContextEntryV1>();
        auto &registry = CcuExecutionRegistryV1::Instance();
        ctx->id = registry.NextId();
        Require(ctx->id != 0, CCU_STATUS_INTERNAL_ERROR, "Handle space exhausted");
        ctx->options = *o;
        ctx->root_info.assign(
            static_cast<const uint8_t *>(o->root_info), static_cast<const uint8_t *>(o->root_info) + o->root_info_size
        );
        if (o->trigger_binary_path) ctx->trigger_path = o->trigger_binary_path;
        ctx->options.root_info = ctx->root_info.data();
        ctx->options.trigger_binary_path = ctx->trigger_path.empty() ? nullptr : ctx->trigger_path.c_str();
        ctx->bootstrap = *o->bootstrap_ops;
        ctx->bootstrap_owner = o->bootstrap_owner;
        ctx->options.bootstrap_ops = nullptr;
        ctx->options.bootstrap_owner_ops = nullptr;
        ctx->bootstrap_lease.owner = o->bootstrap_owner;
        ctx->bootstrap_lease.ops = *o->bootstrap_owner_ops;
        ctx->device = ccu_make_device();
        Require(registry.AddContext(ctx), CCU_STATUS_INTERNAL_ERROR, "Context registry failure");
        Operation operation(ctx);
        bool open_attempted = false;
        try {
            Require(ctx->bootstrap_lease.Retain(), CCU_STATUS_INVALID_ARGUMENT, "Bootstrap retain failed");
            open_attempted = true;
            ctx->device->Open(ctx->options, ctx->id, ctx->generation);
            ctx->ready = true;
        } catch (...) {
            try {
                if (open_attempted) ctx->device->Close();
                Require(ctx->bootstrap_lease.Release(), CCU_STATUS_RETAINED, "Bootstrap release failed");
            } catch (...) {
                ctx->closing = ctx->retained = true;
                *out = ctx->id;
                throw Failure{CCU_STATUS_RETAINED, "Initialization rollback incomplete; close returned context"};
            }
            registry.RemoveContext(ctx->id);
            throw;
        }
        *out = ctx->id;
        return CCU_STATUS_SUCCESS;
    });
}

extern "C" int
ccu_window_register(CcuContextId id, const CcuWindowImportV1 *i, CcuWindowId *out, CcuHostResultV1 *result) {
    if (out) *out = 0;
    return Protect(result, CCU_STAGE_WINDOW_REGISTER, [&] {
        Operation operation(Context(id));
        auto &ctx = *operation.ctx;
        Require(
            ctx.ready && !ctx.closing && !ctx.bound && !ctx.request, CCU_STATUS_INVALID_STATE,
            "Context cannot register windows"
        );
        Require(
            i && out && i->abi_version == CCU_EXECUTION_ABI_V1 && i->struct_size == sizeof(*i) &&
                i->device_id == ctx.options.device_id && !i->flags && i->allocation_id && i->allocation_generation &&
                i->allocation_base_va && i->allocation_size && i->window_size &&
                i->window_offset <= i->allocation_size && i->window_size <= i->allocation_size - i->window_offset &&
                i->allocation_size <= UINT64_MAX - i->allocation_base_va && i->allocation_owner &&
                ValidOps(i->allocation_owner_ops),
            CCU_STATUS_INVALID_ARGUMENT, "Invalid allocation lease or extent"
        );
        auto w = std::make_shared<CcuWindowLeaseV1>();
        w->id = CcuExecutionRegistryV1::Instance().NextId();
        w->import = *i;
        w->import.allocation_owner_ops = nullptr;
        w->owner.owner = i->allocation_owner;
        w->owner.ops = *i->allocation_owner_ops;
        Require(w->id != 0, CCU_STATUS_INTERNAL_ERROR, "Window handle space exhausted");
        // Ownership is registered before retain; rollback has no subsequent allocating insertion.
        // 先登记 owner，再获取租约，回滚时不需要新的容器分配。
        ctx.windows.emplace(w->id, w);
        try {
            Require(w->owner.Retain(), CCU_STATUS_INVALID_ARGUMENT, "Allocation retain failed");
            w->token = ctx.device->Token(i->allocation_base_va, i->allocation_size);
        } catch (...) {
            if (w->owner.Release()) ctx.windows.erase(w->id);
            else {
                ctx.closing = ctx.retained = true;
                throw Failure{CCU_STATUS_RETAINED, "Allocation rollback failed; retry Close", CurrentSdkError()};
            }
            throw;
        }
        *out = w->id;
        return CCU_STATUS_SUCCESS;
    });
}

extern "C" int ccu_bind_windows(CcuContextId id, const CcuWindowBindingV1 *b, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_BIND, [&] {
        Operation operation(Context(id));
        auto &ctx = *operation.ctx;
        Require(
            b && b->abi_version == CCU_EXECUTION_ABI_V1 && b->struct_size == sizeof(*b) && b->binding_generation &&
                !b->reserved[0] && !b->reserved[1] && !b->reserved[2],
            CCU_STATUS_INVALID_ARGUMENT, "Invalid binding"
        );
        Require(
            ctx.ready && !ctx.closing && !ctx.request && !ctx.bound, CCU_STATUS_INVALID_STATE,
            "Binding requires idle unbound context"
        );
        const auto input = Window(ctx, b->input_window), output = Window(ctx, b->output_window);
        auto wire = Header(ctx, b->binding_generation, BIND);
        wire[8] = b->binding_generation;
        PutWindow(wire, 9, *input);
        PutWindow(wire, 17, *output);
        const auto peers = Exchange(ctx, wire);
        for (uint32_t rank = 0; rank < 2; ++rank) {
            Require(peers[rank][8] == b->binding_generation, CCU_STATUS_PROTOCOL_ERROR, "Binding generation mismatch");
            ctx.peer_inputs[rank] = ReadWindow(peers[rank], 9);
            ctx.peer_windows[rank] = ReadWindow(peers[rank], 17);
            ctx.peer_contexts[rank] = peers[rank][6];
            ctx.peer_generations[rank] = peers[rank][7];
        }
        ctx.input_id = input->id;
        ctx.output_id = output->id;
        ctx.binding_generation = b->binding_generation;
        ctx.bound = true;
        return CCU_STATUS_SUCCESS;
    });
}

extern "C" int ccu_prepare(CcuContextId id, const CcuRequestSpecV1 *s, CcuRequestId *out, CcuHostResultV1 *result) {
    if (out) *out = 0;
    return Protect(result, CCU_STAGE_PREPARE, [&] {
        Operation operation(Context(id));
        auto &ctx = *operation.ctx;
        Require(
            s && out && s->abi_version == CCU_EXECUTION_ABI_V1 && s->struct_size == sizeof(*s) && s->collective_id &&
                !s->reserved0,
            CCU_STATUS_INVALID_ARGUMENT, "Invalid request specification"
        );
        Require(
            ctx.bound && !ctx.closing && (!ctx.request || CleanupComplete(*ctx.request)), CCU_STATUS_BUSY,
            "Context is unbound, closing or busy"
        );
        Require(ctx.next_sequence && ctx.next_sequence < UINT64_MAX, CCU_STATUS_INVALID_STATE, "Sequence exhausted");
        auto input = Window(ctx, ctx.input_id), output = Window(ctx, ctx.output_id);
        auto r = std::make_shared<CcuRequestEntryV1>();
        r->id = CcuExecutionRegistryV1::Instance().NextId();
        auto &d = r->descriptor;
        d.abi_version = CCU_EXECUTION_ABI_V1;
        d.struct_size = sizeof(d);
        d.context_id = ctx.id;
        d.context_generation = ctx.generation;
        d.request_seq = ctx.next_sequence++;
        d.program_id = s->root < 2 ? ctx.device->Program(s->root) : 0;
        d.operation = s->operation;
        d.dtype = s->dtype;
        d.root = s->root;
        d.rank_count = 2;
        d.input_window_id = input->id;
        d.output_window_id = output->id;
        d.input_offset = s->input_offset;
        d.output_offset = s->output_offset;
        d.element_count = s->element_count;
        d.collective_id = s->collective_id;
        const auto range = [](const CcuWindowLeaseV1 &w) {
            return CcuWindowRangeV1{
                w.id,
                w.import.allocation_id,
                w.import.allocation_generation,
                w.import.window_offset,
                w.import.window_size,
                w.import.device_id
            };
        };
        Require(
            ccu_validate_window_ranges_v1(d, range(*input), range(*output)) == CcuRequestValidationError::NONE,
            CCU_STATUS_INVALID_ARGUMENT, "Request is outside leased windows or unsupported"
        );
        const uint64_t local_input = input->import.allocation_base_va + input->import.window_offset + s->input_offset;
        const uint64_t local_output =
            output->import.allocation_base_va + output->import.window_offset + s->output_offset;
        const uint64_t bytes = s->element_count * sizeof(float);
        Require(
            local_input % sizeof(float) == 0 && local_output % sizeof(float) == 0 &&
                !(local_input < local_output + bytes && local_output < local_input + bytes),
            CCU_STATUS_INVALID_ARGUMENT, "Misaligned or overlapping device ranges"
        );
        auto wire = Header(ctx, s->collective_id, PREPARE);
        wire[8] = s->operation;
        wire[9] = s->dtype;
        wire[10] = s->root;
        wire[11] = s->element_count;
        wire[12] = ctx.binding_generation;
        wire[13] = d.request_seq;
        wire[14] = input->import.allocation_generation;
        wire[15] = output->import.allocation_generation;
        wire[16] = s->input_offset;
        wire[17] = s->output_offset;
        wire[18] = input->import.allocation_id;
        wire[19] = output->import.allocation_id;
        const auto peers = Exchange(ctx, wire);
        for (const auto &peer : peers) {
            for (size_t at = 8; at <= 12; ++at)
                Require(peer[at] == wire[at], CCU_STATUS_PROTOCOL_ERROR, "Prepared collective contract mismatch");
            Require(peer[13] != 0, CCU_STATUS_PROTOCOL_ERROR, "Invalid peer sequence");
        }
        for (uint32_t rank = 0; rank < 2; ++rank) {
            const auto &p = peers[rank];
            const auto &in = ctx.peer_inputs[rank];
            const auto &out_window = ctx.peer_windows[rank];
            Require(
                p[14] == in.allocation_generation && p[18] == in.allocation_id &&
                    p[15] == out_window.allocation_generation && p[19] == out_window.allocation_id &&
                    p[16] <= in.window_size && bytes <= in.window_size - p[16] && p[16] % sizeof(float) == 0 &&
                    p[17] <= out_window.window_size && bytes <= out_window.window_size - p[17] &&
                    p[17] % sizeof(float) == 0,
                CCU_STATUS_PROTOCOL_ERROR, "Peer input/output lease or offsets mismatch"
            );
        }
        const auto &peer = peers[1 - ctx.options.rank];
        const auto &pw = ctx.peer_windows[1 - ctx.options.rank];
        Require(
            peer[15] == pw.allocation_generation && peer[19] == pw.allocation_id && peer[17] <= pw.window_size &&
                bytes <= pw.window_size - peer[17] && peer[17] % sizeof(float) == 0,
            CCU_STATUS_PROTOCOL_ERROR, "Peer output generation or offset mismatch"
        );
        r->input_id = input->id;
        r->output_id = output->id;
        r->task_args = {
            input->import.allocation_base_va + input->import.window_offset + s->input_offset,
            output->import.allocation_base_va + output->import.window_offset + s->output_offset,
            input->token,
            output->token,
            pw.base_va + pw.window_offset + peer[17],
            pw.token,
            bytes,
            d.request_seq
        };
        Require(
            r->id && r->state.Prepare(d.request_seq) == CcuStateResultV1::OK &&
                CcuExecutionRegistryV1::Instance().AddRequest(r),
            CCU_STATUS_INTERNAL_ERROR, "Request registry failure"
        );
        // The previous cleanup receipt survives until a new request is successfully prepared.
        // 上一请求的清理记录保留到新请求准备成功。
        if (ctx.request) CcuExecutionRegistryV1::Instance().RemoveRequest(ctx.request->id);
        ctx.request = r;
        *out = r->id;
        Details(result, *r);
        return CCU_STATUS_SUCCESS;
    });
}

extern "C" int ccu_arm(CcuRequestId id, CcuTriggerInfoV1 *out, CcuHostResultV1 *result) {
    if (out) *out = {};
    return Protect(result, CCU_STAGE_ARM, [&] {
        auto r = Request(id);
        Operation operation(Context(r->descriptor.context_id));
        auto &ctx = *operation.ctx;
        Active(ctx, r);
        Require(out && !ctx.closing, CCU_STATUS_INVALID_ARGUMENT, "Missing trigger output or context closing");
        Require(r->state.BeginArm() == CcuStateResultV1::OK, CCU_STATUS_INVALID_STATE, "Request cannot arm twice");
        int sdk_error = 0;
        r->launch_attempted = true;
        r->submission = CcuSubmissionV1::UNKNOWN;
        try {
            ctx.device->Launch(r->descriptor.root, r->task_args);
            r->submission = CcuSubmissionV1::SUBMITTED;
        } catch (const CcuSdkError &e) {
            sdk_error = e.code;
        } catch (...) {
            sdk_error = -1;
        }
        bool peers_armed = false;
        try {
            auto wire = Header(ctx, r->descriptor.collective_id, ARM);
            wire[8] = sdk_error == 0;
            wire[9] = ctx.binding_generation;
            const auto peers = Exchange(ctx, wire);
            peers_armed = peers[0][8] == 1 && peers[1][8] == 1 && peers[0][9] == wire[9] && peers[1][9] == wire[9];
        } catch (...) {}
        const auto submission = sdk_error == 0 ? CcuSubmissionV1::SUBMITTED : CcuSubmissionV1::UNKNOWN;
        if (r->state.FinishArm(submission, peers_armed) != CcuStateResultV1::OK) {
            Result(
                result, CCU_STATUS_RETAINED, CCU_STAGE_ARM, "Launch or peer ARMED outcome uncertain; owners retained"
            );
            Details(result, *r);
            if (result) result->sdk_error = sdk_error;
            return CCU_STATUS_RETAINED;
        }
        out->abi_version = CCU_EXECUTION_ABI_V1;
        out->struct_size = sizeof(*out);
        out->context_id = ctx.id;
        out->context_generation = ctx.generation;
        out->request_seq = r->descriptor.request_seq;
        out->gate_va = ctx.device->Gate();
        out->completion_record_va = ctx.device->Completion();
        out->gate_mask = 1;
        out->trigger_owner = CCU_TRIGGER_OWNER_HOST;
        Details(result, *r);
        return CCU_STATUS_SUCCESS;
    });
}
extern "C" int ccu_trigger_host(CcuRequestId id, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_TRIGGER, [&] {
        auto r = Request(id);
        Operation operation(Context(r->descriptor.context_id));
        auto &ctx = *operation.ctx;
        Active(ctx, r);
        Require(
            r->state.BeginTrigger() == CcuStateResultV1::OK, CCU_STATUS_INVALID_STATE,
            "Request is not armed or already triggered"
        );
        r->trigger_attempted = true;
        r->trigger_submission = CcuSubmissionV1::UNKNOWN;
        try {
            ctx.device->Trigger();
        } catch (...) {
            (void)r->state.FinishTrigger(CcuSubmissionV1::UNKNOWN);
            Result(result, CCU_STATUS_RETAINED, CCU_STAGE_TRIGGER, "AIV submission uncertain; owners retained");
            if (result) result->sdk_error = CurrentSdkError();
            Details(result, *r);
            return CCU_STATUS_RETAINED;
        }
        (void)r->state.FinishTrigger(CcuSubmissionV1::SUBMITTED);
        r->trigger_submission = CcuSubmissionV1::SUBMITTED;
        Details(result, *r);
        return CCU_STATUS_SUCCESS;
    });
}
extern "C" int ccu_poll(CcuRequestId id, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_POLL, [&]() -> int {
        auto r = Request(id);
        Operation operation(Context(r->descriptor.context_id));
        auto &ctx = *operation.ctx;
        Active(ctx, r);
        int status;
        try {
            status = Poll(ctx, *r);
        } catch (const Failure &e) {
            if (e.status == CCU_STATUS_INVALID_STATE) throw;
            (void)r->state.RetainFault();
            Result(result, CCU_STATUS_RETAINED, CCU_STAGE_POLL, e.message);
            Details(result, *r);
            return CCU_STATUS_RETAINED;
        } catch (...) {
            (void)r->state.RetainFault();
            Result(result, CCU_STATUS_RETAINED, CCU_STAGE_POLL, "Completion query failed; owners retained");
            if (result) result->sdk_error = CurrentSdkError();
            Details(result, *r);
            return CCU_STATUS_RETAINED;
        }
        Result(
            result, status, CCU_STAGE_POLL,
            status == 0 ? "Completion and both streams drained" : "Completion pending or retained"
        );
        Details(result, *r);
        return status;
    });
}
extern "C" int ccu_wait(CcuRequestId id, uint32_t timeout, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_WAIT, [&]() -> int {
        auto r = Request(id);
        Operation operation(Context(r->descriptor.context_id));
        auto &ctx = *operation.ctx;
        Active(ctx, r);
        int status;
        try {
            status = Wait(ctx, *r, Clock::now() + std::chrono::milliseconds(timeout));
        } catch (const Failure &e) {
            if (e.status == CCU_STATUS_INVALID_STATE) throw;
            (void)r->state.RetainFault();
            Result(result, CCU_STATUS_RETAINED, CCU_STAGE_WAIT, e.message);
            Details(result, *r);
            return CCU_STATUS_RETAINED;
        } catch (...) {
            (void)r->state.RetainFault();
            Result(result, CCU_STATUS_RETAINED, CCU_STAGE_WAIT, "Completion query failed; owners retained");
            if (result) result->sdk_error = CurrentSdkError();
            Details(result, *r);
            return CCU_STATUS_RETAINED;
        }
        Result(
            result, status, CCU_STAGE_WAIT,
            status == 0 ? "Completion and both streams drained" : "Deadline or retained work"
        );
        Details(result, *r);
        return status;
    });
}
extern "C" int ccu_cancel_prepared(CcuRequestId id, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_CANCEL, [&] {
        auto r = Request(id);
        Operation operation(Context(r->descriptor.context_id));
        auto &ctx = *operation.ctx;
        Active(ctx, r, r->state.Phase() == CcuRequestPhaseV1::CANCELLED);
        Cancel(ctx, *r, kProtocolTimeoutMs);
        return CCU_STATUS_SUCCESS;
    });
}
extern "C" int ccu_retire(CcuRequestId id, uint32_t timeout, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_RETIRE, [&] {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        auto r = Request(id);
        Operation operation(Context(r->descriptor.context_id));
        auto &ctx = *operation.ctx;
        Active(ctx, r, r->state.Phase() == CcuRequestPhaseV1::RETIRED);
        Retire(ctx, *r, deadline);
        Details(result, *r);
        return CCU_STATUS_SUCCESS;
    });
}
extern "C" int ccu_context_close(CcuContextId id, uint32_t timeout, CcuHostResultV1 *result) {
    return Protect(result, CCU_STAGE_CLOSE, [&] {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        Operation operation(Context(id));
        auto &ctx = *operation.ctx;
        ctx.closing = true;
        if (auto r = ctx.request; r && !CleanupComplete(*r)) {
            if (!r->launch_attempted) Cancel(ctx, *r, Remaining(deadline));
            else {
                Require(
                    r->trigger_attempted && !r->state.Poisoned(), CCU_STATUS_RETAINED,
                    "Untriggered or uncertain device work remains owned"
                );
                Retire(ctx, *r, deadline);
            }
        }
        try {
            ctx.device->Close();
        } catch (...) {
            ctx.retained = true;
            throw Failure{CCU_STATUS_RETAINED, "SDK cleanup incomplete; retry Close", CurrentSdkError()};
        }
        for (auto it = ctx.windows.begin(); it != ctx.windows.end();) {
            Require(it->second->owner.Release(), CCU_STATUS_RETAINED, "Allocation release failed; retry Close");
            it = ctx.windows.erase(it);
        }
        Require(ctx.bootstrap_lease.Release(), CCU_STATUS_RETAINED, "Bootstrap release failed; retry Close");
        if (ctx.request) CcuExecutionRegistryV1::Instance().RemoveRequest(ctx.request->id);
        CcuExecutionRegistryV1::Instance().RemoveContext(id);
        return CCU_STATUS_SUCCESS;
    });
}
