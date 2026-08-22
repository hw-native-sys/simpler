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

#include "chip_run_lane.h"

#include "chip_worker.h"

#include <algorithm>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct ChipRunState {
    enum class Phase : uint8_t { QUEUED, PREPARED, LAUNCHED, TERMINAL };

    int32_t callable_id{0};
    ChipStorageTaskArgs args{};
    CallConfig config{};
    PipelineSlotLease lease{};
    uint64_t run_id{0};
    uint64_t dispatch_id{0};
    volatile int32_t *accepted_state{nullptr};
    int32_t accepted_value{0};
    ChipWorkerNativeRun native_run{};
    Phase phase{Phase::QUEUED};
    ChipRunPreparationDisposition disposition{ChipRunPreparationDisposition::VALIDATED_ONLY};
    bool pipeline_leased{true};
    bool activated{false};
    bool crossed_launch_fence{false};
    bool depth_one_fallback{false};
    std::exception_ptr error;
    std::exception_ptr poison_error;
};

struct ChipRunLaneState {
    explicit ChipRunLaneState(ChipWorker &worker) :
        worker(&worker),
        generations(worker.pipeline_depth(), 0) {}

    [[noreturn]] static void rethrow_as_poisoned(const std::exception_ptr &error) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chip run lane is poisoned: ") + e.what());
        } catch (...) {
            throw std::runtime_error("chip run lane is poisoned by an unknown native failure");
        }
    }

    void require_usable() const {
        if (closed) throw std::runtime_error("chip run lane is closed");
        if (poison != nullptr) rethrow_as_poisoned(poison);
    }

    void rethrow_run_error(const std::shared_ptr<ChipRunState> &run) const {
        if (run->error == nullptr) return;
        if (run->poison_error != nullptr) rethrow_as_poisoned(run->poison_error);
        std::rethrow_exception(run->error);
    }

    bool permits_native_successor(const ChipRunState &predecessor, const CallConfig &successor_config) const {
        return worker->supports_concurrent_native_prepare() && !predecessor.config.diagnostics_any() &&
               !successor_config.diagnostics_any() && predecessor.phase == ChipRunState::Phase::LAUNCHED;
    }

    bool permits_native_successor(const ChipRunState &predecessor, const ChipRunState &successor) const {
        return permits_native_successor(predecessor, successor.config);
    }

    void prepare(const std::shared_ptr<ChipRunState> &run) {
        run->native_run = worker->prepare_native_run_for_lane(
            run->callable_id, &run->args, run->config, run->lease, run->run_id, run->dispatch_id, run->accepted_state,
            run->accepted_value, run->pipeline_leased
        );
        run->phase = ChipRunState::Phase::PREPARED;
        run->disposition = ChipRunPreparationDisposition::NATIVE_PREPARED;
    }

    void poison_with(const std::shared_ptr<ChipRunState> &run, std::exception_ptr error) {
        if (run->poison_error == nullptr) run->poison_error = error;
        if (poison == nullptr) poison = error;
    }

    void finish(const std::shared_ptr<ChipRunState> &run) noexcept {
        try {
            worker->finalize_native_run(run->native_run);
        } catch (...) {
            const std::exception_ptr finalize_error = std::current_exception();
            if (run->error == nullptr) run->error = finalize_error;
            poison_with(run, finalize_error);
        }
        run->phase = ChipRunState::Phase::TERMINAL;
        if (!fifo.empty() && fifo.front() == run) fifo.pop_front();
    }

    void fail_launch(const std::shared_ptr<ChipRunState> &run) noexcept {
        run->error = std::current_exception();
        finish(run);
    }

    void launch_front() noexcept {
        if (fifo.empty()) return;
        const auto run = fifo.front();
        if (poison != nullptr) {
            if (run->phase == ChipRunState::Phase::PREPARED) {
                run->error = poison;
                run->poison_error = poison;
                finish(run);
            } else if (run->phase == ChipRunState::Phase::QUEUED) {
                run->error = poison;
                run->poison_error = poison;
                run->phase = ChipRunState::Phase::TERMINAL;
                fifo.pop_front();
            }
            return;
        }
        if (!run->activated || run->phase == ChipRunState::Phase::LAUNCHED ||
            run->phase == ChipRunState::Phase::TERMINAL) {
            return;
        }
        if (run->phase == ChipRunState::Phase::QUEUED) {
            try {
                prepare(run);
            } catch (...) {
                run->error = std::current_exception();
                run->phase = ChipRunState::Phase::TERMINAL;
                fifo.pop_front();
                return;
            }
        }
        try {
            worker->launch_native_run(run->native_run);
            run->phase = ChipRunState::Phase::LAUNCHED;
            run->crossed_launch_fence = true;
        } catch (...) {
            fail_launch(run);
        }
    }

    void prepare_successor_if_eligible(const std::shared_ptr<ChipRunState> &run) {
        if (fifo.size() != 2 || fifo.back() != run || fifo.front() == run) return;
        if (run->phase != ChipRunState::Phase::QUEUED || run->depth_one_fallback) return;
        if (!permits_native_successor(*fifo.front(), *run)) return;
        try {
            prepare(run);
        } catch (const ChipWorker::PreparedRunIncompatible &) {
            run->depth_one_fallback = true;
            return;
        } catch (...) {
            run->error = std::current_exception();
            run->phase = ChipRunState::Phase::TERMINAL;
            fifo.pop_back();
        }
    }

    bool progress(const std::shared_ptr<ChipRunState> &target) {
        if (target->phase == ChipRunState::Phase::TERMINAL) return true;
        if (fifo.empty()) throw std::runtime_error("chip run lane lost a nonterminal run");
        if (fifo.front() != target) {
            (void)progress(fifo.front());
            if (target->phase == ChipRunState::Phase::TERMINAL) return true;
            if (fifo.empty() || fifo.front() != target) return false;
        }

        launch_front();
        if (fifo.size() == 2 && fifo.front() == target) prepare_successor_if_eligible(fifo.back());
        if (target->phase == ChipRunState::Phase::TERMINAL) {
            launch_front();
            return true;
        }
        if (target->phase != ChipRunState::Phase::LAUNCHED) return false;

        try {
            if (!worker->poll_native_run(target->native_run)) return false;
        } catch (...) {
            const std::exception_ptr poll_error = std::current_exception();
            finish(target);
            if (target->error == nullptr) target->error = poll_error;
            poison_with(target, target->error);
            launch_front();
            return true;
        }
        finish(target);
        launch_front();
        return true;
    }

    void drain_front() noexcept {
        if (fifo.empty()) return;
        const auto run = fifo.front();
        if (poison != nullptr && run->phase == ChipRunState::Phase::QUEUED) {
            run->error = poison;
            run->poison_error = poison;
            run->phase = ChipRunState::Phase::TERMINAL;
            fifo.pop_front();
            return;
        }
        if (run->phase == ChipRunState::Phase::QUEUED && !run->activated) {
            run->phase = ChipRunState::Phase::TERMINAL;
            fifo.pop_front();
            return;
        }
        launch_front();
        if (run->phase == ChipRunState::Phase::TERMINAL) return;
        if (run->phase == ChipRunState::Phase::PREPARED && (!run->activated || poison != nullptr)) {
            if (poison != nullptr && run->error == nullptr) {
                run->error = poison;
                run->poison_error = poison;
            }
            finish(run);
            return;
        }
        if (run->phase == ChipRunState::Phase::LAUNCHED) {
            try {
                worker->wait_native_run(run->native_run);
            } catch (...) {
                run->error = std::current_exception();
                poison_with(run, run->error);
            }
            finish(run);
        }
    }

    // Block on the device for the launched front, for waiters with no deadline
    // to bound them. Only the front can be LAUNCHED, so its completion is what
    // lets any waiter in the FIFO advance. Re-polling instead would hold a core
    // for the whole run — the case codestyle rule 5 sends to a wakeup primitive
    // rather than a busy loop. Reports whether it actually blocked, so a caller
    // that cannot be unblocked this way does not spin on it.
    bool block_on_front() noexcept {
        if (fifo.empty()) return false;
        const auto front = fifo.front();
        if (front->phase != ChipRunState::Phase::LAUNCHED) return false;
        try {
            worker->wait_native_run(front->native_run);
        } catch (...) {
            const std::exception_ptr wait_error = std::current_exception();
            if (front->error == nullptr) front->error = wait_error;
            poison_with(front, front->error);
        }
        finish(front);
        launch_front();
        return true;
    }

    ChipWorker *worker;
    mutable std::mutex mu;
    std::deque<std::shared_ptr<ChipRunState>> fifo;
    std::vector<uint64_t> generations;
    uint64_t direct_generation{0};
    std::exception_ptr poison;
    bool closed{false};
};

ChipRun::ChipRun(std::shared_ptr<ChipRunLaneState> lane, std::shared_ptr<ChipRunState> run) :
    lane_(std::move(lane)),
    run_(std::move(run)) {}

bool ChipRun::done() {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return lane_->progress(run_);
}

bool ChipRun::wait_until(Deadline deadline) {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    const bool unbounded = deadline == Deadline::max();
    while (true) {
        {
            std::lock_guard<std::mutex> lk(lane_->mu);
            if (lane_->progress(run_)) {
                lane_->rethrow_run_error(run_);
                return true;
            }
            // An unbounded waiter has no deadline to end its loop, so polling
            // here would spin for the whole run. Block on the device instead;
            // if nothing is blockable yet the poll loop below still applies.
            if (unbounded && lane_->block_on_front()) continue;
        }
        if (Clock::now() >= deadline) return false;
    }
}

void ChipRun::activate() {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    if (run_->phase == ChipRunState::Phase::TERMINAL) {
        lane_->rethrow_run_error(run_);
        return;
    }
    run_->activated = true;
    lane_->launch_front();
    if (lane_->fifo.size() == 2 && lane_->fifo.front() == run_) {
        lane_->prepare_successor_if_eligible(lane_->fifo.back());
    }
}

void ChipRun::abandon() {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    if (run_->phase == ChipRunState::Phase::TERMINAL) return;
    if (run_->phase == ChipRunState::Phase::LAUNCHED) {
        throw std::runtime_error("cannot abandon a launched ChipRun");
    }
    auto it = std::find(lane_->fifo.begin(), lane_->fifo.end(), run_);
    if (it == lane_->fifo.end()) throw std::runtime_error("chip run lane lost an unlaunched run");
    if (run_->phase == ChipRunState::Phase::PREPARED) {
        try {
            lane_->worker->finalize_native_run(run_->native_run);
        } catch (...) {
            run_->error = std::current_exception();
            lane_->poison_with(run_, run_->error);
        }
    }
    run_->phase = ChipRunState::Phase::TERMINAL;
    lane_->fifo.erase(it);
    if (run_->error != nullptr) std::rethrow_exception(run_->error);
}

bool ChipRun::launched() const {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return run_->crossed_launch_fence;
}

bool ChipRun::lane_poisoned() const {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return lane_->poison != nullptr;
}

ChipRunPreparationDisposition ChipRun::preparation_disposition() const {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return run_->disposition;
}

ChipRunLane::ChipRunLane(ChipWorker &worker) :
    state_(std::make_shared<ChipRunLaneState>(worker)) {}

ChipRunLane::~ChipRunLane() {
    try {
        close();
    } catch (...) {}
}

ChipRun ChipRunLane::submit(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value, bool activated
) {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->require_usable();
    if (lease.reserved != 0 || lease.generation == 0 || lease.slot_id >= state_->generations.size()) {
        throw std::runtime_error("chip run lane received an invalid pipeline lease");
    }
    if (lease.generation < state_->generations[lease.slot_id]) {
        throw std::runtime_error("chip run lane pipeline lease generation is stale");
    }
    for (const auto &candidate : state_->fifo) {
        if ((run_id != 0 && candidate->run_id == run_id) ||
            (dispatch_id != 0 && candidate->dispatch_id == dispatch_id)) {
            throw std::runtime_error("chip run lane received a duplicate run or dispatch identity");
        }
        if (candidate->lease.slot_id == lease.slot_id) {
            throw std::runtime_error("chip run lane pipeline slot is already occupied");
        }
    }
    if (state_->fifo.size() >= 2) {
        throw std::runtime_error("chip run lane capacity exceeded before native preparation");
    }
    if (!state_->fifo.empty() && state_->fifo.front()->phase == ChipRunState::Phase::LAUNCHED &&
        dispatch_id < state_->fifo.front()->dispatch_id) {
        throw std::runtime_error("chip run lane cannot reorder before an active dispatch");
    }
    auto run = std::make_shared<ChipRunState>();
    run->callable_id = callable_id;
    run->args = args;
    run->config = config;
    run->lease = lease;
    run->run_id = run_id;
    run->dispatch_id = dispatch_id;
    run->accepted_state = accepted_state;
    run->accepted_value = accepted_value;
    run->pipeline_leased = true;
    run->activated = activated;
    state_->generations[lease.slot_id] = lease.generation;
    auto position = std::upper_bound(
        state_->fifo.begin(), state_->fifo.end(), dispatch_id,
        [](uint64_t id, const std::shared_ptr<ChipRunState> &candidate) {
            return id < candidate->dispatch_id;
        }
    );
    state_->fifo.insert(position, run);

    try {
        if (state_->fifo.front() == run) {
            state_->launch_front();
            if (state_->fifo.size() == 2) state_->prepare_successor_if_eligible(state_->fifo.back());
        } else {
            state_->prepare_successor_if_eligible(run);
        }
    } catch (...) {
        run->error = std::current_exception();
        run->phase = ChipRunState::Phase::TERMINAL;
        auto it = std::find(state_->fifo.begin(), state_->fifo.end(), run);
        if (it != state_->fifo.end()) state_->fifo.erase(it);
    }
    return ChipRun(state_, std::move(run));
}

ChipRun ChipRunLane::submit(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, volatile int32_t *accepted_state,
    int32_t accepted_value
) {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->require_usable();
    if (state_->generations.empty()) throw std::runtime_error("chip run lane has no runtime slots");

    // Direct admission follows the runtime contract but keeps the lane as its
    // only authority. A compatible active run may own one prepared successor;
    // otherwise admission drains the front and retains depth-one behavior.
    // In particular, the third submit waits here before a slot generation is
    // minted or native preparation begins.
    while (!state_->fifo.empty()) {
        const bool has_successor_capacity =
            state_->fifo.size() == 1 && state_->permits_native_successor(*state_->fifo.front(), config);
        if (has_successor_capacity) break;
        state_->drain_front();
        state_->require_usable();
        state_->launch_front();
    }

    uint32_t slot_id = 0;
    for (; slot_id < state_->generations.size(); ++slot_id) {
        const bool occupied = std::any_of(
            state_->fifo.begin(), state_->fifo.end(), [slot_id](const std::shared_ptr<ChipRunState> &candidate) {
                return candidate->lease.slot_id == slot_id;
            }
        );
        if (!occupied) break;
    }
    if (slot_id == state_->generations.size()) {
        throw std::runtime_error("chip run lane has no free direct runtime slot after admission");
    }
    if (state_->direct_generation == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("chip run lane exhausted its direct generation space");
    }
    ++state_->direct_generation;

    auto run = std::make_shared<ChipRunState>();
    run->callable_id = callable_id;
    run->args = args;
    run->config = config;
    run->lease = PipelineSlotLease{slot_id, 0, state_->direct_generation};
    run->accepted_state = accepted_state;
    run->accepted_value = accepted_value;
    run->pipeline_leased = false;
    run->activated = true;
    state_->fifo.push_back(run);
    if (state_->fifo.front() == run) {
        state_->launch_front();
    } else {
        state_->prepare_successor_if_eligible(run);
    }
    return ChipRun(state_, std::move(run));
}

void ChipRunLane::drain() {
    std::lock_guard<std::mutex> lk(state_->mu);
    while (!state_->fifo.empty())
        state_->drain_front();
    if (state_->poison != nullptr) std::rethrow_exception(state_->poison);
}

void ChipRunLane::close() {
    std::lock_guard<std::mutex> lk(state_->mu);
    if (state_->closed) {
        if (state_->poison != nullptr) std::rethrow_exception(state_->poison);
        return;
    }
    while (!state_->fifo.empty())
        state_->drain_front();
    state_->closed = true;
    if (state_->poison != nullptr) std::rethrow_exception(state_->poison);
}

bool ChipRunLane::poisoned() const {
    std::lock_guard<std::mutex> lk(state_->mu);
    return state_->poison != nullptr;
}
