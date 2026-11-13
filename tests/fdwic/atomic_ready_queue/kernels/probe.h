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
#include <pto/pto-inst.hpp>

#include "inner_kernel.h"
#include "intrinsic.h"
#include "scheduler/scheduler_scalar_policy.h"

namespace {
// This independently linked callable has no resident scalar-trace binding.
template <>
struct SchedulerScalarCompilation<void> {
    static constexpr bool enabled = false;
};
}  // namespace

#include "scheduler/scheduler_ticket_queue.h"
#include "tensor.h"

namespace atomic_ready_queue {

constexpr uint32_t kMaxWorkers = 96;
constexpr uint32_t kPerProducer = 16;
constexpr uint32_t kMaxTasks = kMaxWorkers / 3 * kPerProducer;

struct alignas(64) AtomicLine {
    volatile uint64_t value{0};
};

struct alignas(64) Payload {
    uint64_t words[8]{};
};

struct alignas(SCHEDULER_STATE_BASE_ALIGNMENT) State {
    SchedulerTicketQueue queue;
    SchedulerTicketQueue zero_queue;
    SchedulerTicketQueue last_queue;
    SchedulerTicketEntry entries[kMaxTasks + 1];
    SchedulerTicketEntry last_entries[2];
    AtomicLine arrivals[6][kMaxWorkers];
    AtomicLine initialized;
    AtomicLine hole_reserved;
    AtomicLine late_published;
    AtomicLine allow_early;
    AtomicLine early_returned;
    AtomicLine late_returned;
    AtomicLine hole_finished;
    AtomicLine stress_ready;
    AtomicLine last_ready;
    AtomicLine consumed;
    AtomicLine last_winners;
    AtomicLine error;
    Payload payloads[kMaxTasks];
    AtomicLine claims[kMaxTasks];
};

struct alignas(64) Record {
    AtomicLine visits;
    uint64_t values[16]{};
};

static_assert(sizeof(AtomicLine) == 64 && sizeof(Payload) == 64);
static_assert(sizeof(SchedulerTicketEntry) == 64 && sizeof(SchedulerTicketQueue) == 256);
constexpr uint64_t state_bytes = (12 + kMaxTasks + 1 + 2 + 6 * kMaxWorkers + 12 + 2 * kMaxTasks) * 64;
static_assert(
    sizeof(State) ==
    (state_bytes + SCHEDULER_STATE_BASE_ALIGNMENT - 1) / SCHEDULER_STATE_BASE_ALIGNMENT * SCHEDULER_STATE_BASE_ALIGNMENT
);
static_assert(sizeof(Record) == 192 && offsetof(Record, values) == 64);

static inline __aicore__ __gm__ int64_t *data(int64_t arg) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
}

static inline __aicore__ uint64_t load(__gm__ AtomicLine &line) {
    return scheduler_gm_fetch_add(line.value, UINT64_C(0));
}

static inline __aicore__ bool check(__gm__ State *state, bool valid, uint64_t code, uint64_t &error) {
    if (!valid) {
        if (error == 0) error = code;
        scheduler_gm_compare_exchange(state->error.value, UINT64_C(0), code);
    }
    return valid;
}

static inline __aicore__ bool
wait(__gm__ State *state, __gm__ AtomicLine &line, uint64_t epoch, uint64_t code, uint64_t &error) {
    const uint64_t start = get_sys_cnt_aicore();
    while (true) {
        if (load(state->error) != 0) return false;
        const uint64_t value = load(line);
        if (value >= epoch) return check(state, value == epoch, code, error);
        if (get_sys_cnt_aicore() - start > PLATFORM_PROF_SYS_CNT_FREQ * 5) return check(state, false, code, error);
    }
}

static inline __aicore__ bool
barrier(__gm__ State *state, uint32_t phase, uint32_t rank, uint32_t workers, uint64_t epoch, uint64_t &error) {
    scheduler_cache_barrier();
    scheduler_gm_store(state->arrivals[phase][rank].value, epoch);
    for (uint32_t peer = 0; peer < workers; ++peer)
        if (!wait(state, state->arrivals[phase][peer], epoch, 100 + phase, error)) return false;
    return true;
}

static inline __aicore__ uint64_t pattern(uint64_t epoch, uint32_t ref, uint32_t word) {
    return (epoch << 48) ^ (static_cast<uint64_t>(ref) << 16) ^ (UINT64_C(0x12340000) + word);
}

static inline __aicore__ void write_payload(__gm__ State *state, uint64_t epoch, uint32_t ref) {
    auto *payload = &state->payloads[ref];
    scheduler_observe_cache_line(payload);
    for (uint32_t word = 0; word < 8; ++word)
        payload->words[word] = pattern(epoch, ref, word);
    scheduler_publish_cache_line(payload);
}

static inline __aicore__ bool verify_payload(__gm__ State *state, uint64_t epoch, uint32_t ref, uint64_t &error) {
    auto *payload = &state->payloads[ref];
    scheduler_observe_cache_line(payload);
    bool valid = true;
    for (uint32_t word = 0; word < 8; ++word)
        valid = valid && payload->words[word] == pattern(epoch, ref, word);
    return check(state, valid, 20, error);
}

static inline __aicore__ void
reset_queue(__gm__ SchedulerTicketQueue *queue, __gm__ SchedulerTicketEntry *entries, uint32_t capacity) {
    scheduler_gm_store(queue->head, UINT64_C(0));
    scheduler_gm_store(queue->reserved_tail, UINT64_C(0));
    // The extra entry is a guard outside the supplied view's capacity.
    for (uint32_t index = 0; index <= capacity; ++index)
        scheduler_gm_store(entries[index].task_ref, SCHEDULER_TASK_REF_INVALID);
    scheduler_cache_barrier();
}

static inline __aicore__ void store_record(__gm__ Record *record, const uint64_t (&values)[16]) {
    for (uint32_t word = 0; word < 16; ++word)
        record->values[word] = values[word];
    scheduler_writeback_cache_line(&record->values[0]);
    scheduler_writeback_cache_line(&record->values[8]);
    scheduler_cache_barrier();
}

static inline __aicore__ bool hole_test(
    __gm__ State *state, uint32_t rank, uint32_t consumer, uint64_t epoch, uint64_t (&values)[16], uint64_t &error
) {
    SchedulerTicketQueueView queue{&state->queue, state->entries, 4, 4};
    if (rank == 0) {
        SchedulerTicketRange range{};
        if (!check(
                state,
                scheduler_ticket_reserve(queue, 2, &range) == SchedulerTicketResult::SUCCESS && range.begin == 0 &&
                    range.count == 2,
                1, error
            ))
            return false;
        scheduler_gm_store(state->hole_reserved.value, epoch);
        if (!wait(state, state->allow_early, epoch, 2, error)) return false;
        const int64_t refs[2]{0, 1};
        write_payload(state, epoch, 0);
        write_payload(state, epoch, 1);
        if (!check(state, scheduler_ticket_publish(queue, range, refs) == SchedulerTicketResult::SUCCESS, 3, error))
            return false;
        scheduler_gm_store(state->early_returned.value, epoch);
    } else if (rank == 1) {
        if (!wait(state, state->hole_reserved, epoch, 4, error)) return false;
        SchedulerTicketRange range{};
        if (!check(
                state,
                scheduler_ticket_reserve(queue, 2, &range) == SchedulerTicketResult::SUCCESS && range.begin == 2 &&
                    range.count == 2,
                5, error
            ))
            return false;
        const int64_t refs[2]{2, 3};
        write_payload(state, epoch, 2);
        write_payload(state, epoch, 3);
        if (!check(state, scheduler_ticket_publish(queue, range, refs) == SchedulerTicketResult::SUCCESS, 6, error))
            return false;
        scheduler_gm_store(state->late_returned.value, epoch);
        scheduler_gm_store(state->late_published.value, epoch);
    } else if (rank == consumer) {
        if (!wait(state, state->late_published, epoch, 7, error)) return false;
        int64_t ref = 123;
        if (!check(
                state,
                scheduler_ticket_try_pop(queue, &ref) == SchedulerTicketResult::NOT_READY &&
                    ref == SCHEDULER_TASK_REF_INVALID &&
                    scheduler_gm_fetch_add(queue.control->head, UINT64_C(0)) == 0 &&
                    scheduler_gm_fetch_add(queue.control->reserved_tail, UINT64_C(0)) == 4,
                8, error
            ))
            return false;
        values[6] = 1;
        scheduler_gm_store(state->allow_early.value, epoch);
        if (!wait(state, state->early_returned, epoch, 9, error)) return false;
        for (int64_t expected = 0; expected < 4; ++expected) {
            if (!check(
                    state, scheduler_ticket_try_pop(queue, &ref) == SchedulerTicketResult::SUCCESS && ref == expected,
                    10, error
                ) ||
                !verify_payload(state, epoch, static_cast<uint32_t>(ref), error))
                return false;
            values[7] += static_cast<uint64_t>(ref);
            values[8] ^= static_cast<uint64_t>(ref);
        }
        if (!wait(state, state->late_returned, epoch, 11, error)) return false;
        if (!check(state, scheduler_ticket_try_pop(queue, &ref) == SchedulerTicketResult::NOT_READY, 12, error))
            return false;
        scheduler_gm_store(state->hole_finished.value, epoch);
    }
    return wait(state, state->hole_finished, epoch, 13, error);
}

static inline __aicore__ bool stress_test(
    __gm__ State *state, uint32_t rank, uint32_t producers, uint64_t epoch, uint64_t (&values)[16], uint64_t &error
) {
    const uint32_t tasks = producers * kPerProducer;
    SchedulerTicketQueueView queue{&state->queue, state->entries, tasks, tasks};
    if (rank < producers) {
        for (uint32_t offset = 0; offset < kPerProducer; offset += 4) {
            if (load(state->error) != 0) return false;
            SchedulerTicketRange range{};
            if (!check(state, scheduler_ticket_reserve(queue, 4, &range) == SchedulerTicketResult::SUCCESS, 21, error))
                return false;
            if (offset == 0 && !barrier(state, 4, rank, producers, epoch, error)) return false;
            int64_t refs[4]{};
            for (uint32_t index = 0; index < 4; ++index) {
                refs[index] = rank * kPerProducer + offset + index;
                write_payload(state, epoch + 16, static_cast<uint32_t>(refs[index]));
            }
            if (!check(
                    state, scheduler_ticket_publish(queue, range, refs) == SchedulerTicketResult::SUCCESS, 22, error
                ))
                return false;
        }
    } else {
        const uint64_t start = get_sys_cnt_aicore();
        while (load(state->consumed) < tasks) {
            if (load(state->error) != 0) return false;
            if (get_sys_cnt_aicore() - start > PLATFORM_PROF_SYS_CNT_FREQ * 5) return check(state, false, 23, error);
            int64_t ref = -1;
            const auto result = scheduler_ticket_try_pop(queue, &ref, 2);
            if (result == SchedulerTicketResult::NOT_READY || result == SchedulerTicketResult::RETRY) continue;
            if (!check(state, result == SchedulerTicketResult::SUCCESS && ref >= 0 && ref < tasks, 24, error))
                return false;
            if (!verify_payload(state, epoch + 16, static_cast<uint32_t>(ref), error)) return false;
            if (!check(state, scheduler_gm_fetch_add(state->claims[ref].value, UINT64_C(1)) == 0, 25, error))
                return false;
            ++values[3];
            values[4] += static_cast<uint64_t>(ref);
            values[5] ^= static_cast<uint64_t>(ref);
            scheduler_gm_fetch_add(state->consumed.value, UINT64_C(1));
        }
    }
    return true;
}

}  // namespace atomic_ready_queue

static inline __aicore__ void atomic_ready_queue_probe(__gm__ int64_t *args, uint32_t lane) {
    using namespace atomic_ready_queue;
    const uint32_t block = static_cast<uint32_t>(get_block_idx(args));
    const uint32_t blocks = static_cast<uint32_t>(args[2]);
    const uint32_t mask = static_cast<uint32_t>(args[3]);
    const uint64_t epoch = static_cast<uint64_t>(args[4]);
    const uint32_t workers = blocks * (mask == 7 ? 3 : 1);
    const uint32_t rank = mask == 7 ? block * 3 + lane : block;
    if (workers < 6 || workers > kMaxWorkers || rank >= workers || epoch == 0 || epoch > 8) return;
    auto *state_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *record_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    if (state_tensor->buffer.size < sizeof(State) ||
        record_tensor->buffer.size < epoch * (workers + 1) * sizeof(Record))
        return;
    auto *state = reinterpret_cast<__gm__ State *>(data(args[0]));
    auto *records = reinterpret_cast<__gm__ Record *>(data(args[1])) + (epoch - 1) * (workers + 1);
    auto *record = &records[rank];
    const uint32_t producers = workers / 3;
    const uint32_t tasks = producers * kPerProducer;
    uint64_t values[16]{};
    uint64_t error = 0, hole_head = 0, hole_tail = 0, boundary_checks = 0;
    values[1] = rank;
    values[2] = epoch;
    values[10] = static_cast<uint32_t>(get_coreid()) & 0xfffU;
    values[11] = lane;
    values[12] = get_sys_cnt_aicore();
    values[14] = mask;
    values[15] = workers;
    bool ok = check(state, scheduler_gm_fetch_add(record->visits.value, UINT64_C(1)) == 0, 30, error);
    if (ok) ok = barrier(state, 0, rank, workers, epoch, error);
    if (ok && rank == 0) {
        reset_queue(&state->queue, state->entries, 4);
        scheduler_gm_store(state->initialized.value, epoch);
    }
    if (ok) ok = wait(state, state->initialized, epoch, 31, error);
    if (ok) ok = hole_test(state, rank, producers, epoch, values, error);
    if (ok) ok = barrier(state, 1, rank, workers, epoch, error);
    if (ok && rank == 0) {
        hole_head = scheduler_gm_fetch_add(state->queue.head, UINT64_C(0));
        hole_tail = scheduler_gm_fetch_add(state->queue.reserved_tail, UINT64_C(0));
        ok = check(state, hole_head == 4 && hole_tail == 4, 32, error);
        reset_queue(&state->queue, state->entries, tasks);
        scheduler_gm_store(state->consumed.value, UINT64_C(0));
        scheduler_gm_store(state->last_winners.value, UINT64_C(0));
        for (uint32_t ref = 0; ref < tasks; ++ref)
            scheduler_gm_store(state->claims[ref].value, UINT64_C(0));
        scheduler_cache_barrier();
        if (ok) scheduler_gm_store(state->stress_ready.value, epoch);
    }
    if (ok) ok = wait(state, state->stress_ready, epoch, 33, error);
    if (ok) ok = stress_test(state, rank, producers, epoch, values, error);
    if (ok) ok = barrier(state, 2, rank, workers, epoch, error);
    if (ok && rank == 0) {
        scheduler_gm_store(state->zero_queue.head, UINT64_C(0));
        scheduler_gm_store(state->zero_queue.reserved_tail, UINT64_C(0));
        SchedulerTicketQueueView zero{&state->zero_queue, nullptr, 0, 0};
        SchedulerTicketRange empty{};
        int64_t ref = 123;
        ok = check(
            state,
            scheduler_ticket_reserve(zero, 0, &empty) == SchedulerTicketResult::SUCCESS &&
                scheduler_ticket_publish(zero, empty, nullptr) == SchedulerTicketResult::SUCCESS &&
                scheduler_ticket_try_pop(zero, &ref) == SchedulerTicketResult::NOT_READY &&
                ref == SCHEDULER_TASK_REF_INVALID &&
                scheduler_ticket_reserve(zero, 1, &empty) == SchedulerTicketResult::INVALID,
            34, error
        );
        if (ok) boundary_checks |= 1;
        reset_queue(&state->last_queue, state->last_entries, 1);
        SchedulerTicketQueueView last{&state->last_queue, state->last_entries, 1, 1};
        SchedulerTicketRange range{};
        const int64_t only[]{0};
        ok = ok && check(
                       state,
                       scheduler_ticket_reserve(last, 1, &range) == SchedulerTicketResult::SUCCESS &&
                           scheduler_ticket_publish(last, range, only) == SchedulerTicketResult::SUCCESS,
                       35, error
                   );
        if (ok) scheduler_gm_store(state->last_ready.value, epoch);
    }
    if (ok) ok = wait(state, state->last_ready, epoch, 36, error);
    if (ok) ok = check(state, scheduler_gm_fetch_add(state->last_queue.head, UINT64_C(0)) == 0, 42, error);
    if (ok) ok = barrier(state, 3, rank, workers, epoch, error);
    if (ok) {
        SchedulerTicketQueueView last{&state->last_queue, state->last_entries, 1, 1};
        int64_t ref = -1;
        const auto result = scheduler_ticket_try_pop(last, &ref, 2);
        if (result == SchedulerTicketResult::SUCCESS) {
            ok = check(state, ref == 0, 37, error);
            values[9] = 1;
            scheduler_gm_fetch_add(state->last_winners.value, UINT64_C(1));
        } else {
            ok = check(
                state, result == SchedulerTicketResult::NOT_READY || result == SchedulerTicketResult::RETRY, 38, error
            );
        }
    }
    if (ok) ok = barrier(state, 5, rank, workers, epoch, error);
    if (ok && rank == 0) {
        SchedulerTicketQueueView last{&state->last_queue, state->last_entries, 1, 1};
        SchedulerTicketRange overflow{};
        const int64_t only[]{0};
        ok = check(
            state, load(state->last_winners) == 1 && scheduler_gm_fetch_add(state->last_queue.head, UINT64_C(0)) == 1,
            39, error
        );
        if (ok) boundary_checks |= 2;
        ok = ok && check(
                       state,
                       scheduler_ticket_reserve(last, 1, &overflow) == SchedulerTicketResult::INVALID &&
                           scheduler_ticket_publish(last, {1, 1}, only) == SchedulerTicketResult::INVALID &&
                           scheduler_gm_load(state->last_entries[1].task_ref) == SCHEDULER_TASK_REF_INVALID,
                       40, error
                   );
        if (ok) boundary_checks |= 4;
        bool exactly_once = load(state->consumed) == tasks;
        for (uint32_t ref = 0; ref < tasks; ++ref)
            exactly_once = exactly_once && load(state->claims[ref]) == 1;
        ok = ok &&
             check(
                 state, exactly_once && scheduler_gm_load(state->entries[tasks].task_ref) == SCHEDULER_TASK_REF_INVALID,
                 41, error
             );
        if (ok) boundary_checks |= 8;
        uint64_t summary[16]{
            load(state->error),
            epoch,
            scheduler_gm_fetch_add(state->queue.head, UINT64_C(0)),
            scheduler_gm_fetch_add(state->queue.reserved_tail, UINT64_C(0)),
            load(state->consumed),
            tasks,
            hole_head,
            hole_tail,
            scheduler_gm_fetch_add(state->last_queue.head, UINT64_C(0)),
            scheduler_gm_fetch_add(state->last_queue.reserved_tail, UINT64_C(0)),
            boundary_checks,
            sizeof(SchedulerTicketEntry),
            sizeof(SchedulerTicketQueue),
            sizeof(State),
            static_cast<uint64_t>(scheduler_gm_load(state->entries[tasks].task_ref)),
            workers
        };
        scheduler_gm_fetch_add(records[workers].visits.value, UINT64_C(1));
        store_record(&records[workers], summary);
    }
    values[0] = error == 0 ? load(state->error) : error;
    values[13] = get_sys_cnt_aicore();
    store_record(record, values);
}
