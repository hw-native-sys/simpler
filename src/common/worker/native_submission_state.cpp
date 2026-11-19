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

#include "native_submission_state.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>

#include "call_argument_snapshot.h"
#include "chip_worker.h"
#include "queued_submission.h"

struct AclCaptureModelNotice {
    AclCaptureModelNotice(void *model, const ChipWorker *worker, void *stream) :
        model(model),
        worker(worker),
        stream(stream) {}
    void *const model;
    const ChipWorker *const worker;
    void *const stream;
    std::atomic<bool> destroyed{false};
};

namespace {
static_assert(std::atomic<bool>::is_always_lock_free);

extern "C" {
static void mark_acl_capture_destroyed(void *cookie) noexcept {
    // RTS can hold its model and callback locks. This last access to the
    // cookie only publishes the notice; release runs on the checked owner.
    static_cast<AclCaptureModelNotice *>(cookie)->destroyed.store(true, std::memory_order_release);
}
}

bool model_notice_observed(const AclCaptureModelNotice &notice) noexcept {
    return notice.destroyed.load(std::memory_order_acquire);
}

struct ModelNotices {
    const pid_t process = getpid();
    std::mutex mutex;
    std::unordered_map<void *, std::shared_ptr<AclCaptureModelNotice>> pending;
};

template <typename Register>
std::shared_ptr<AclCaptureModelNotice>
register_model_notice(void *model, const ChipWorker *worker, void *stream, Register subscribe) {
    // RTS owns a raw cookie until it calls the one-shot destroy callback.
    // Missing notice cannot make that cookie reclaimable at owner teardown.
    static auto *const notices = new ModelNotices;
    if (notices->process != getpid()) throw RunPreparationRefused("ACL model notices belong to another process");
    std::lock_guard<std::mutex> lock(notices->mutex);
    for (auto it = notices->pending.begin(); it != notices->pending.end();) {
        if (model_notice_observed(*it->second)) it = notices->pending.erase(it);
        else ++it;
    }
    if (notices->pending.count(model) != 0)
        throw RunPreparationRefused("ACL model handle already has an unconfirmed destroy notice");
    auto notice = std::make_shared<AclCaptureModelNotice>(model, worker, stream);
    notices->pending.emplace(model, notice);
    const int rc = subscribe(model, mark_acl_capture_destroyed, notice.get());
    if (rc != 0)
        throw RunPreparationRefused("ACL model destroy callback registration failed with code " + std::to_string(rc));
    if (model_notice_observed(*notice))
        throw RunPreparationRefused("ACL model was destroyed during callback registration");
    return notice;
}
}  // namespace

struct NativeSubmissionState::Backing {
    Backing(
        ChipWorker &worker, std::shared_ptr<ChipWorker> owner, int32_t callable_id, const CallArgumentSnapshot &values,
        const CallConfig &config, void *caller_stream, bool framework_eligible
    ) :
        worker(&worker),
        owner(std::move(owner)),
        callable_id(callable_id),
        values(values),
        config(config),
        caller_stream(caller_stream),
        submission(
            [this](bool framework) {
                if (purpose == SIMPLER_PREPARE_CAPTURE) {
                    prepared->activate();
                    return *prepared;
                }
                try {
                    this->worker->validate_execution_context();
                    this->worker->validate_caller_stream(this->caller_stream);
                } catch (const std::exception &error) {
                    throw RunPreparationRefused(error.what());
                }
                if (framework) {
                    if (!this->worker->execution_environment_ || !this->worker->execution_environment_->borrowed ||
                        this->config.diagnostics_any() || this->worker->transfer_completed_run_fn_ == nullptr ||
                        this->worker->observe_framework_submissions_fn_ == nullptr)
                        throw RunPreparationRefused(
                            "framework observation transfer requires matched eager Kernel runtime"
                        );
                    for (int i = 0; i < this->values.args().tensor_count(); ++i) {
                        const auto &tensor = this->values.args().tensor(i);
                        if (tensor.address_space != AddressSpace::DEVICE || tensor.transfer != TensorTransfer::NONE)
                            throw RunPreparationRefused(
                                "framework observation transfer requires DEVICE/NONE arguments"
                            );
                    }
                }
                return this->worker->submit_chip_run(
                    this->callable_id, this->values.args(), this->config, nullptr, 0, this->caller_stream,
                    SIMPLER_PREPARE_EAGER, nullptr, framework
                );
            },
            framework_eligible
        ) {}

    const pid_t process = getpid();
    ChipWorker *worker;
    std::shared_ptr<ChipWorker> owner;
    const int32_t callable_id;
    CallArgumentSnapshot values;
    const CallConfig config;
    void *const caller_stream;
    // Selection and native-operation admission share the submission mutex.
    std::optional<NativeRunPreparePurpose> purpose;
    bool registration_claimed{false};
    bool operation_in_progress{false};
    std::optional<ChipRun> prepared;
    std::unique_ptr<ChipRunResources> resources;
    std::shared_ptr<AclCaptureModelNotice> model_notice;
    std::exception_ptr preparation_error;
    QueuedSubmission<ChipRun> submission;
};

NativeSubmissionState::NativeSubmissionState(
    ChipWorker &worker, std::shared_ptr<ChipWorker> owner, int32_t callable_id, const CallArgumentSnapshot &values,
    const CallConfig &config, void *caller_stream, bool framework_eligible
) :
    backing_(
        std::make_unique<Backing>(
            worker, std::move(owner), callable_id, values, config, caller_stream, framework_eligible
        )
    ) {}

NativeSubmissionState::~NativeSubmissionState() noexcept {
    if (!backing_) return;
    if (backing_->process != getpid()) {
        (void)backing_.release();
        return;
    }
    if (backing_->purpose != SIMPLER_PREPARE_CAPTURE || !backing_->registration_claimed) return;
    try {
        // Unknown cleanup retains the exact native storage and HOST values.
        if (!carrier_retired() || (backing_->resources && backing_->resources->has_reference()))
            (void)backing_.release();
    } catch (...) {
        (void)backing_.release();
    }
}

void NativeSubmissionState::validate_process() const {
    if (backing_->process != getpid()) throw ExecutionContextRefused("native submission belongs to another process");
}

void NativeSubmissionState::finish_operation() noexcept {
    std::lock_guard<std::mutex> lock(backing_->submission.mutex_);
    backing_->operation_in_progress = false;
}

bool NativeSubmissionState::carrier_retired() const {
    return !backing_->prepared || backing_->prepared->carrier_retired_for_submission();
}

void NativeSubmissionState::submit() {
    auto &state = *backing_;
    state.submission.submit(
        [&] {
            if (state.operation_in_progress) throw RunPreparationRefused("native resource operation is in progress");
            if (!state.purpose) state.purpose = SIMPLER_PREPARE_EAGER;
        },
        [&] {
            if (state.purpose == SIMPLER_PREPARE_CAPTURE) {
                if (state.preparation_error) std::rethrow_exception(state.preparation_error);
                if (!state.registration_claimed || !state.prepared || !state.resources ||
                    !state.resources->has_reference())
                    throw RunPreparationRefused("capture submission requires registered native resources");
                if (!state.model_notice || model_notice_observed(*state.model_notice) ||
                    active_acl_model() != state.model_notice->model || model_notice_observed(*state.model_notice))
                    throw RunPreparationRefused("ACL capture model changed before submission");
            }
        }
    );
}

void *NativeSubmissionState::active_acl_model() const {
    const auto &state = *backing_;
    state.owner->validate_execution_context();
    const auto query = state.owner->acl_capture_info_fn_;
    if (query == nullptr || state.caller_stream == nullptr)
        throw RunPreparationRefused("ACL model capture query is unavailable");
    // ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE in the existing CANN capture ABI.
    constexpr uint32_t active = 1;
    alignas(uint32_t) unsigned char status_bytes[sizeof(uint32_t)]{};
    void *model = nullptr;
    const int rc = query(state.caller_stream, status_bytes, &model);
    if (rc != 0) throw RunPreparationRefused("ACL capture query failed with code " + std::to_string(rc));
    uint32_t status;
    std::memcpy(&status, status_bytes, sizeof(status));
    if (status != active || model == nullptr)
        throw RunPreparationRefused("ACL model retention requires an ACTIVE capture");
    return model;
}

void NativeSubmissionState::retain_model_resources(
    int &registration_started, const NativeSubmissionState *model_anchor
) {
    registration_started = 0;
    std::shared_ptr<AclCaptureModelNotice> anchor;
    if (model_anchor != nullptr) {
        model_anchor->validate_process();
        auto &state = *model_anchor->backing_;
        std::lock_guard<std::mutex> lock(state.submission.mutex_);
        if (state.operation_in_progress || !state.model_notice || model_notice_observed(*state.model_notice))
            throw RunPreparationRefused("ACL model anchor has no live destroy registration");
        anchor = state.model_notice;
    }
    validate_process();
    auto &state = *backing_;
    std::shared_ptr<ChipWorker> owner;
    auto admissible = [&] {
        if (state.operation_in_progress) throw RunPreparationRefused("native resource operation is in progress");
        if (state.registration_claimed)
            throw RunPreparationRefused("native submission resources are already registered");
        if (state.submission.phase_ != QueuedSubmission<ChipRun>::Phase::Pending)
            throw RunPreparationRefused("native resource registration requires a pending token");
        if (state.submission.framework_claimed_)
            throw RunPreparationRefused("capture registration cannot take a framework eager token");
        if (!state.owner) throw RunPreparationRefused("capture registration requires a native shared Worker owner");
        if (state.config.diagnostics_any())
            throw RunPreparationRefused("framework capture does not prepare diagnostic resources during recording");
    };
    {
        std::lock_guard<std::mutex> lock(state.submission.mutex_);
        admissible();
        owner = state.owner;
    }
    // A refused environment has not claimed registration and can be retried.
    owner->validate_execution_context();
    {
        std::lock_guard<std::mutex> lock(state.submission.mutex_);
        admissible();
        state.purpose = SIMPLER_PREPARE_CAPTURE;
        state.registration_claimed = true;
        state.operation_in_progress = true;
        registration_started = 1;
    }
    try {
        state.prepared.emplace();
        (void)owner->submit_chip_run(
            state.callable_id, state.values.args(), state.config, nullptr, 0, state.caller_stream,
            SIMPLER_PREPARE_CAPTURE, &*state.prepared
        );
        state.prepared->retain_resources_into(state.resources);
        // The retained run pins this Worker and its loaded ACL provider.
        // Neither this preparation nor registration emits a captured run.
        void *const model = active_acl_model();
        if (anchor) {
            if (anchor->worker != owner.get() || anchor->stream != state.caller_stream || anchor->model != model ||
                model_notice_observed(*anchor))
                throw RunPreparationRefused("ACL model anchor does not match the ACTIVE capture owner and stream");
            state.model_notice = std::move(anchor);
        } else {
            const auto subscribe = owner->acl_model_destroy_register_fn_;
            if (subscribe == nullptr)
                throw RunPreparationRefused("ACL model destroy callback registration is unavailable");
            state.model_notice = register_model_notice(model, owner.get(), state.caller_stream, subscribe);
        }
    } catch (...) {
        state.preparation_error = std::current_exception();
        finish_operation();
        throw;
    }
    finish_operation();
}

bool NativeSubmissionState::model_destroyed() const {
    validate_process();
    auto &state = *backing_;
    std::lock_guard<std::mutex> lock(state.submission.mutex_);
    return !state.operation_in_progress && state.model_notice && model_notice_observed(*state.model_notice);
}

void NativeSubmissionState::cancel_pending(NativeSubmissionCleanup &outcome) {
    validate_process();
    auto &state = *backing_;
    bool cancelled = false;
    {
        std::lock_guard<std::mutex> lock(state.submission.mutex_);
        if (state.operation_in_progress || state.submission.phase_ == QueuedSubmission<ChipRun>::Phase::Submitting) {
            outcome = {NativeSubmissionDisposition::Submitting, false};
            return;
        }
        const auto reason =
            state.preparation_error ?
                state.preparation_error :
                std::make_exception_ptr(RunPreparationRefused("native submission cancelled before submit"));
        cancelled = state.submission.cancel_locked(reason);
        outcome = {static_cast<NativeSubmissionDisposition>(state.submission.phase_), false};
        if (state.purpose != SIMPLER_PREPARE_CAPTURE) {
            if (!state.purpose) {
                outcome.carrier_retired = true;
            } else {
                throw RunPreparationRefused("native carrier cleanup requires capture registration");
            }
        } else {
            state.operation_in_progress = true;
        }
    }
    if (cancelled) state.submission.changed_.notify_all();
    if (state.purpose != SIMPLER_PREPARE_CAPTURE) return;
    try {
        try {
            if (state.prepared) state.prepared->abandon();
        } catch (...) {
            outcome.carrier_retired = carrier_retired();
            if (!outcome.carrier_retired) throw;
            // Historical submission errors remain in their observation.
        }
        outcome.carrier_retired = carrier_retired();
    } catch (...) {
        finish_operation();
        throw;
    }
    finish_operation();
}

void NativeSubmissionState::release_resources() {
    validate_process();
    auto &state = *backing_;
    {
        std::lock_guard<std::mutex> lock(state.submission.mutex_);
        if (state.operation_in_progress) throw RunPreparationRefused("native resource operation is in progress");
        if (state.submission.phase_ == QueuedSubmission<ChipRun>::Phase::Pending ||
            state.submission.phase_ == QueuedSubmission<ChipRun>::Phase::Submitting)
            throw RunPreparationRefused("native resource release requires settled submission or cancellation");
        state.operation_in_progress = true;
    }
    try {
        if (!carrier_retired())
            throw RunPreparationRefused("native resource release requires original carrier retirement");
        if (state.resources) {
            if (state.model_notice && !model_notice_observed(*state.model_notice))
                throw RunPreparationRefused("native resource release requires the ACL model destroy notice");
            state.resources->release();
            state.resources.reset();
        }
    } catch (...) {
        finish_operation();
        throw;
    }
    finish_operation();
}

bool NativeSubmissionState::cancel(std::exception_ptr reason) {
    validate_process();
    if (!reason) throw std::invalid_argument("cancellation requires a reason");
    bool cancelled;
    {
        std::lock_guard<std::mutex> lock(backing_->submission.mutex_);
        if (backing_->operation_in_progress) return false;
        cancelled = backing_->submission.cancel_locked(reason);
    }
    if (cancelled) backing_->submission.changed_.notify_all();
    return cancelled;
}
bool NativeSubmissionState::done() { return backing_->submission.done(); }
bool NativeSubmissionState::wait_until(ChipRun::Deadline deadline) { return backing_->submission.wait_until(deadline); }
void NativeSubmissionState::claim_framework() {
    auto &state = *backing_;
    state.submission.claim_framework([&] {
        if (state.operation_in_progress || state.purpose == SIMPLER_PREPARE_CAPTURE)
            throw RunPreparationRefused("framework eager ownership cannot take a capture token");
    });
}
void NativeSubmissionState::note_export() { backing_->submission.note_export(); }
bool NativeSubmissionState::framework_claimed() { return backing_->submission.framework_claimed(); }
bool NativeSubmissionState::framework_transferred() { return backing_->submission.framework_transferred(); }
bool NativeSubmissionState::wait_framework_until(ChipRun::Deadline deadline) {
    return backing_->submission.wait_framework_until(deadline);
}
bool NativeSubmissionState::retire_framework(bool wait_for_own_completion, ChipRun::Deadline deadline) {
    auto &submission = backing_->submission;
    return submission.with_framework_run(
        wait_for_own_completion,
        [&](ChipRun &run) {
            try {
                const bool retired = run.retire_framework(wait_for_own_completion, deadline);
                if (run.framework_transferred()) submission.note_framework_transfer();
                return retired;
            } catch (...) {
                if (run.framework_transferred()) submission.note_framework_transfer();
                throw;
            }
        },
        deadline
    );
}
bool NativeSubmissionState::owner_recovery_required() {
    validate_process();
    std::optional<ChipRun> prepared;
    bool capture;
    {
        std::lock_guard<std::mutex> lock(backing_->submission.mutex_);
        if (backing_->operation_in_progress) return true;
        capture = backing_->purpose == SIMPLER_PREPARE_CAPTURE;
        if (capture) {
            if (backing_->resources && backing_->resources->has_reference()) return true;
            prepared = backing_->prepared;
        }
    }
    if (capture) return prepared && !prepared->carrier_retired_for_submission();
    return backing_->submission.owner_recovery_required();
}
void NativeSubmissionState::retire() {
    validate_process();
    auto &state = *backing_;
    {
        std::lock_guard<std::mutex> lock(state.submission.mutex_);
        if (state.operation_in_progress) throw RunPreparationRefused("native resource operation is in progress");
        state.operation_in_progress = true;
    }
    try {
        if (state.purpose == SIMPLER_PREPARE_CAPTURE &&
            (!carrier_retired() || (state.resources && state.resources->has_reference())))
            throw RunPreparationRefused("cannot retire native capture owners before checked cleanup");
        state.submission.retire();
        state.owner.reset();
    } catch (...) {
        finish_operation();
        throw;
    }
    finish_operation();
}

namespace {
template <class Action>
int native_result(char *message, size_t capacity, Action action) noexcept {
    try {
        action();
        return 0;
    } catch (const std::exception &error) {
        if (message && capacity) std::snprintf(message, capacity, "%s", error.what());
    } catch (...) {
        if (message && capacity) std::snprintf(message, capacity, "unknown native submission failure");
    }
    return -1;
}
}  // namespace

NativeSubmissionCapsule::NativeSubmissionCapsule(std::shared_ptr<NativeSubmissionState> state) :
    state(std::move(state)) {
    api = {this, retain, release, submit};
}
void NativeSubmissionCapsule::retain(void *context) noexcept {
    static_cast<NativeSubmissionCapsule *>(context)->references.fetch_add(1, std::memory_order_relaxed);
}
void NativeSubmissionCapsule::release(void *context) noexcept {
    auto *self = static_cast<NativeSubmissionCapsule *>(context);
    if (self->references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete self;
}
int NativeSubmissionCapsule::submit(void *context, char *message, size_t capacity) noexcept {
    return native_result(message, capacity, [&] {
        static_cast<NativeSubmissionCapsule *>(context)->state->submit();
    });
}
