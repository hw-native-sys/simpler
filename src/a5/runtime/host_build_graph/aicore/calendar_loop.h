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

template <typename T>
inline __aicore__ const __gm__ T *
percounter_calendar_table(const __gm__ uint8_t *image, uint64_t bytes, uint32_t offset, uint64_t count) {
    if (offset < sizeof(CalendarRangeHeader) || offset % alignof(T) != 0 || offset > bytes ||
        count > (bytes - offset) / sizeof(T))
        return nullptr;
    return reinterpret_cast<const __gm__ T *>(image + offset);
}

template <typename Ready>
inline __aicore__ bool percounter_calendar_wait(const LaneConfig &cfg, Ready ready, const uint32_t &retry_cycles = 0) {
    uint32_t pending = 0;
    for (; !ready(); ++pending) {
        // Check errors on the first miss. Sample the shared EXIT word only
        // on a sustained wait; the row loop also checks it during ready work.
        if (pending % PERCOUNTER_SPIN_CHECK_INTERVAL == 0 && percounter_should_stop(cfg, pending != 0)) {
            if constexpr (profile_compiled) {
                if (cfg.scalar != nullptr) cfg.scalar->poll_loops += static_cast<uint64_t>(pending) + 1;
            }
            return false;
        }
#if defined(__CCE_AICORE__)
        if (retry_cycles != 0 && pending >= 2) {
            const uint64_t retry_start = get_sys_cnt_aicore();
            while (get_sys_cnt_aicore() - retry_start < retry_cycles) {}
        }
#else
        (void)retry_cycles;
#endif
        SPIN_WAIT_HINT();
    }
    if constexpr (profile_compiled) {
        if (cfg.scalar != nullptr) cfg.scalar->poll_loops += pending;
    }
    return true;
}

inline __aicore__ uint32_t
percounter_calendar_retry_cycles(const LaneConfig &cfg, uint32_t blocks, bool completion_helper) {
    if (blocks < 16) return 0;
    const auto offset = (static_cast<uint32_t>(cfg.cluster_index) & 7U) * 12U;
    return (completion_helper ? 128U : 256U) + offset;
}

inline __aicore__ uint32_t percounter_calendar_retry_scale(uint32_t remaining) {
    return (remaining > 2U) + (remaining > 8U) + (remaining > 16U);
}

inline __aicore__ bool percounter_calendar_wait_threshold(
    const LaneConfig &cfg, uint64_t address, uint32_t target, uint32_t base_retry_cycles
) {
    auto *counter = reinterpret_cast<__gm__ volatile uint32_t *>(address);
    uint32_t retry_cycles = 0;
    return percounter_calendar_wait(
        cfg,
        [&]() __aicore__ {
            const uint32_t observed = percounter_soc_sram_read(*counter) & CALENDAR_COUNT_MASK;
            if (observed >= target) return true;
            const uint32_t remaining = target - observed;
            // A large distance to the last credit can tolerate a longer retry
            // interval. Observe the final credits with no added delay.
            retry_cycles = base_retry_cycles * percounter_calendar_retry_scale(remaining);
            return false;
        },
        retry_cycles
    );
}

inline __aicore__ void percounter_calendar_add(uint64_t address) {
    if (address != 0) percounter_soc_sram_increment(address);
}

inline __aicore__ bool percounter_calendar_seal(
    const LaneConfig &cfg, const __gm__ CalendarExecutionExtra *extra, bool sink, __gm__ CalendarTiming *sample,
    uint64_t entry, bool *deferred
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::AsyncSeal, entry);
    auto *scratch = cfg.async_group;
    const uint32_t count = scratch->count;
    const int32_t error = scratch->error_code;
    if (error != SIMPLER_ERROR_NONE || count > PERCOUNTER_ASYNC_CAPACITY || scratch->reservation_limit != 0) {
        percounter_fail(
            cfg, error == SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW      ? SchedulerGraphResult::ASYNC_WAIT_OVERFLOW :
                 error == SIMPLER_ERROR_ASYNC_COMPLETION_INVALID ? SchedulerGraphResult::ASYNC_COMPLETION_INVALID :
                                                                   SchedulerGraphResult::ASYNC_REGISTRATION_FAILED
        );
        return false;
    }
    if (count == 0) return true;
    if (!cfg.async_inputs_active) {
        percounter_soc_sram_or(reinterpret_cast<uint64_t>(cfg.lifecycle_notify), AICORE_LIFECYCLE_ASYNC_INPUT_BIT);
        percounter_memory_barrier();
        cfg.async_inputs_active = true;
    }
    const uint32_t task = extra->task_id;
    if (task >= cfg.task_count) {
        percounter_fail(cfg, SchedulerGraphResult::ASYNC_REGISTRATION_FAILED, task);
        return false;
    }
    auto &counter = cfg.counters[task];
    const uint64_t address = counter.completion_notify_address;
    const uint32_t bit = counter.calendar_defer_bit;
    if (address == 0 || bit == 0 || (bit & CALENDAR_COUNT_MASK) != 0 || (bit & (bit - 1)) != 0) {
        percounter_fail(cfg, SchedulerGraphResult::ASYNC_REGISTRATION_FAILED, task);
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto &source = cfg.async_entries[i];
        const CompletionToken token{
            source.addr, source.expected_value, source.engine, source.completion_type, source.backend_cookie
        };
        if (source._pad != 0 || !percounter_completion_token_valid(token)) {
            percounter_fail(cfg, SchedulerGraphResult::ASYNC_COMPLETION_INVALID, task);
            return false;
        }
        // A single monotonic counter can already be ready when the kernel
        // returns. Its read is non-consuming, so pending values still use the
        // regular registration and any successor can settle that registration.
        if (count == 1 && token.completion_type == COMPLETION_TYPE_COUNTER &&
            percounter_poll_valid_completion(token).state == CompletionPollState::READY) {
            scratch->count = 0;
            dcci(scratch, SINGLE_CACHE_LINE, CACHELINE_OUT);
            percounter_memory_barrier();
            if (sample != nullptr) sample->end = get_sys_cnt_aicore();
            *deferred = false;
            return true;
        }
    }

    const uint64_t index = percounter_gm_fetch_add(cfg.control->async_registered, uint64_t{1});
    if (index >= cfg.async_completion_capacity) {
        percounter_fail(cfg, SchedulerGraphResult::ASYNC_REGISTRATION_FAILED, task);
        return false;
    }
    auto *group = &cfg.async_completions[index];
    group->calendar_defer_owner = 0;
    group->pending_mask = UINT64_MAX >> (64 - count);
    group->completion_add_cmd = percounter_soc_sram_add_address(address);
    group->calendar_sample = reinterpret_cast<uint64_t>(sample);
    group->calendar_shell_add_cmd = sink ? extra->shell_add_cmd : 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto &source = cfg.async_entries[i];
        auto &target = group->entries[i];
        target.addr = source.addr;
        target.backend_cookie = source.backend_cookie;
        target.expected_value = source.expected_value;
        target.engine = source.engine;
        target.completion_type = source.completion_type;
        target._pad = 0;
    }

    auto &head = counter.calendar_defer_head;
    uint64_t previous = percounter_gm_read(head);
    group->next = previous;
    const uint64_t bytes = offsetof(PercounterAsyncCompletion, entries) + count * sizeof(DeferredCompletionEntry);
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        dcci(reinterpret_cast<__gm__ uint8_t *>(group) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    scratch->count = 0;
    dcci(scratch, SINGLE_CACHE_LINE, CACHELINE_OUT);
    if (sample != nullptr) {
        sample->defer_register = get_sys_cnt_aicore();
        dcci(sample, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    percounter_memory_barrier();

    // The head only grows. A later registration can be observed immediately
    // through an already-set defer bit, so its record is visible before CAS.
    const uint64_t ref = index + 1;
    for (;;) {
        const uint64_t observed = percounter_gm_compare_exchange(head, previous, ref);
        if (observed == previous) break;
        previous = observed;
        percounter_gm_store(group->next, previous);
        percounter_memory_barrier();
    }
    percounter_memory_barrier();
    const uint64_t issued = percounter_task_event_clock(cfg);
    percounter_soc_sram_or(address, bit);
    *deferred = true;
    percounter_record_task_event(cfg, PercounterTaskEventKind::DeferRegistered, entry, issued, ref);
    return true;
}

inline __aicore__ void percounter_calendar_record_defer_completion(uint64_t address) {
    if (address == 0) return;
    auto *sample = reinterpret_cast<__gm__ CalendarTiming *>(address);
    const uint64_t cycles = get_sys_cnt_aicore();
    percounter_scalar_memory_issue(ProfilePhase::BypassStoreIssue, 1, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
        __builtin_cce_st_dev(cycles, &sample->defer_completion, 0);
#else
        __atomic_store_n(&sample->defer_completion, cycles, __ATOMIC_RELEASE);
#endif
    });
}

inline __aicore__ bool percounter_calendar_service_defer(const LaneConfig &cfg, uint32_t task, uint64_t *cursor) {
    const uint64_t ref = *cursor != 0 ? *cursor : percounter_gm_read(cfg.counters[task].calendar_defer_head);
    if (ref == 0) return true;
    if (ref > cfg.async_completion_capacity) {
        percounter_fail(cfg, SchedulerGraphResult::ASYNC_COMPLETION_INVALID, task);
        return false;
    }
    auto &group = cfg.async_completions[ref - 1];
    *cursor = percounter_gm_read(group.next);
    const uint64_t lane = 1 + cfg.cluster_index * PLATFORM_CORES_PER_BLOCKDIM + cfg.cluster_lane;
    auto &owner_word = group.calendar_defer_owner;
    uint64_t owner = percounter_gm_read(owner_word);
    if (owner == 0) {
        owner = percounter_gm_compare_exchange(owner_word, uint64_t{0}, lane);
        if (owner == 0) {
            owner = lane;
            percounter_record_task_event(
                cfg, PercounterTaskEventKind::DeferPollBegin, 0, percounter_task_event_clock(cfg), ref
            );
        }
    }
    if (owner != lane) return true;

    ProfileScope profile_scope(cfg.scalar, ProfilePhase::AsyncPoll);
    const uint64_t pending = percounter_gm_read(group.pending_mask);
    uint64_t remaining = pending;
    for (uint64_t conditions = pending; conditions != 0; conditions &= conditions - 1) {
        const uint32_t index = static_cast<uint32_t>(__builtin_ctzll(conditions));
        auto &entry = group.entries[index];
        percounter_observe_cache_line(&entry);
        const CompletionToken token{
            entry.addr, entry.expected_value, entry.engine, entry.completion_type, entry.backend_cookie
        };
        const CompletionPollResult result = percounter_poll_valid_completion(token);
        if (result.state == CompletionPollState::FAILED) {
            percounter_fail(cfg, SchedulerGraphResult::ASYNC_COMPLETION_INVALID, task);
            return false;
        }
        if (result.state == CompletionPollState::READY) remaining &= ~(uint64_t{1} << index);
    }
    if (remaining != pending) percounter_gm_store(group.pending_mask, remaining);
    if (remaining != 0) return true;

    percounter_record_task_event(
        cfg, PercounterTaskEventKind::DeferSignalReady, 0, percounter_task_event_clock(cfg), ref
    );
    percounter_calendar_record_defer_completion(percounter_gm_read(group.calendar_sample));
    percounter_memory_barrier();
    percounter_calendar_add(percounter_gm_read(group.completion_add_cmd));
    const uint64_t issued = percounter_task_event_clock(cfg);
    percounter_calendar_add(percounter_gm_read(group.calendar_shell_add_cmd));
    percounter_memory_barrier();
    percounter_gm_fetch_add_issue(cfg.control->async_settled, uint64_t{1});
    percounter_gm_store(owner_word, UINT64_MAX);
    percounter_record_task_event(cfg, PercounterTaskEventKind::DeferredCompletionIssue, 0, issued, ref);
    return true;
}

inline __aicore__ CompletionPollState percounter_calendar_poll_completion(
    const LaneConfig &cfg, uint64_t command, uint64_t target, uint64_t *cursor, uint32_t *remaining = nullptr
) {
    const uint64_t tag = command & CALENDAR_WAIT_TAG_MASK;
    auto *counter = reinterpret_cast<__gm__ volatile uint32_t *>(command & ~CALENDAR_WAIT_TAG_MASK);
    if (tag == CALENDAR_WAIT_NOTIFY)
        return (percounter_soc_sram_read(*counter) & CALENDAR_COUNT_MASK) >= target ? CompletionPollState::READY :
                                                                                      CompletionPollState::PENDING;
    const uint32_t task = static_cast<uint32_t>(target >> 32);
    const uint32_t bit_index = static_cast<uint8_t>(target >> 24);
    if (tag != CALENDAR_WAIT_DEFER || task >= cfg.task_count || bit_index < 24 || bit_index >= 32) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, task, bit_index);
        return CompletionPollState::FAILED;
    }
    const uint32_t word = percounter_soc_sram_read(*counter);
    const uint32_t observed = word & CALENDAR_COUNT_MASK;
    const uint32_t required = static_cast<uint32_t>(target) & CALENDAR_COUNT_MASK;
    if (observed >= required) return CompletionPollState::READY;
    const bool needs_service = (word & (uint32_t{1} << bit_index)) != 0;
    if (remaining != nullptr) *remaining = needs_service ? 0 : required - observed;
    if (needs_service && !percounter_calendar_service_defer(cfg, task, cursor)) return CompletionPollState::FAILED;
    return CompletionPollState::PENDING;
}

inline __aicore__ bool percounter_calendar_wait_list(
    const LaneConfig &cfg, const __gm__ CalendarWait *waits, uint32_t count, uint64_t entry = 0,
    uint64_t range_key = UINT64_MAX, uint32_t retry_cycles = 0
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Fanin, entry);
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t command = waits[i].read_cmd;
        const uint64_t target = waits[i].target;
        const uint64_t tag = command & CALENDAR_WAIT_TAG_MASK;
        const uint64_t address = command & ~CALENDAR_WAIT_TAG_MASK;
        auto *notify = reinterpret_cast<__gm__ volatile uint32_t *>(address);
        if (tag == CALENDAR_WAIT_NOTIFY) {
            if (!percounter_calendar_wait(
                    cfg,
                    [&]() __aicore__ {
                        return (percounter_soc_sram_read(*notify) & CALENDAR_COUNT_MASK) >= target;
                    },
                    retry_cycles
                ))
                return false;
        } else if (tag == CALENDAR_WAIT_GM_TARGET) {
            auto *watermark = reinterpret_cast<__gm__ volatile uint64_t *>(address);
            if (!percounter_calendar_wait(cfg, [&]() __aicore__ {
                    return percounter_gm_read(*watermark) >= target;
                }))
                return false;
            if (range_key != UINT64_MAX)
                percounter_record_task_event(
                    cfg, PercounterTaskEventKind::GraphReadyObserved, range_key, percounter_task_event_clock(cfg),
                    target
                );
        } else if (tag == CALENDAR_WAIT_DEFER) {
            uint64_t cursor = 0;
            bool failed = false;
            uint32_t retry_delay = 0;
            if (!percounter_calendar_wait(
                    cfg,
                    [&]() __aicore__ {
                        uint32_t remaining = 0;
                        const auto state =
                            percounter_calendar_poll_completion(cfg, command, target, &cursor, &remaining);
                        failed = state == CompletionPollState::FAILED;
                        retry_delay = retry_cycles * percounter_calendar_retry_scale(remaining);
                        return state != CompletionPollState::PENDING;
                    },
                    retry_delay
                ) ||
                failed)
                return false;
        } else {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, -1, command);
            return false;
        }
    }
    return true;
}

inline __aicore__ CompletionPollState percounter_calendar_poll_fused_fanin(
    const LaneConfig &cfg, uint64_t address, uint32_t target, uint32_t task, uint32_t bit, uint64_t *cursor,
    uint32_t *remaining = nullptr
) {
    auto *counter = reinterpret_cast<__gm__ volatile uint32_t *>(address);
    const uint32_t word = percounter_soc_sram_read(*counter);
    const uint32_t observed = word & CALENDAR_COUNT_MASK;
    if (observed >= target) return CompletionPollState::READY;
    const bool has_defer = (word & bit) != 0;
    if (remaining != nullptr) *remaining = has_defer ? 0 : target - observed;
    if (has_defer && !percounter_calendar_service_defer(cfg, task, cursor)) return CompletionPollState::FAILED;
    return CompletionPollState::PENDING;
}

inline __aicore__ bool percounter_calendar_wait_fused_fanin(
    const LaneConfig &cfg, const __gm__ CalendarWait *waits, uint64_t address, uint64_t group_target, uint32_t begin,
    uint32_t count, uint64_t entry, uint32_t retry_cycles
) {
    const auto target = static_cast<uint32_t>(group_target);
    const auto producers = static_cast<uint32_t>(group_target >> 32);
    if (address == 0 || target == 0 || target > CALENDAR_COUNT_MASK || producers < 2 ||
        producers > CALENDAR_FUSED_FANIN_MAX_PRODUCERS || count > 1 || (count != 0 && begin >= producers)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
        return false;
    }

    if (count == 0) return percounter_calendar_wait_threshold(cfg, address, target, retry_cycles);

    uint32_t task = 0;
    uint32_t bit = 0;
    if (count != 0) {
        // Each original helper observes its own producer and defer descriptor.
        const auto *member = waits + 1 + begin;
        const uint64_t member_command = member->read_cmd;
        const uint64_t member_target = member->target;
        task = static_cast<uint32_t>(member_target >> 32);
        const uint32_t bit_index = 24 + begin;
        if (member_command != (address | CALENDAR_WAIT_DEFER) || task >= cfg.task_count ||
            static_cast<uint32_t>(member_target) != ((bit_index << 24) | target)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, task);
            return false;
        }
        bit = uint32_t{1} << bit_index;
    }

    uint64_t cursor = 0;
    bool failed = false;
    uint32_t retry_delay = 0;
    const bool finished = percounter_calendar_wait(
        cfg,
        [&]() __aicore__ {
            uint32_t remaining = 0;
            const auto state =
                percounter_calendar_poll_fused_fanin(cfg, address, target, task, bit, &cursor, &remaining);
            retry_delay = remaining > 2 ? retry_cycles : 0;
            failed = state == CompletionPollState::FAILED;
            return state != CompletionPollState::PENDING;
        },
        retry_delay
    );
    return finished && !failed;
}

inline __aicore__ bool percounter_calendar_wait_fanin(
    const LaneConfig &cfg, const __gm__ CalendarWait *waits, uint32_t begin, uint32_t count, uint64_t entry,
    uint32_t retry_cycles = 0
) {
    ProfileScope scope(cfg.scalar, ProfilePhase::Fanin, entry);
    const uint64_t command = waits[0].read_cmd;
    const uint64_t target = waits[0].target;
    const uint64_t tag = command & CALENDAR_WAIT_TAG_MASK;
    const uint64_t address = command & ~CALENDAR_WAIT_TAG_MASK;
    if (tag == CALENDAR_WAIT_FUSED_FANIN)
        return percounter_calendar_wait_fused_fanin(cfg, waits, address, target, begin, count, entry, retry_cycles);
    if (count > CALENDAR_FANIN_TASKS_PER_LANE || begin > target || count > target - begin ||
        tag != CALENDAR_WAIT_NOTIFY) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
        return false;
    }
    if (count != 0) {
        uint64_t cursors[CALENDAR_FANIN_TASKS_PER_LANE]{};
        uint32_t pending = (1U << count) - 1;
        bool failed = false;
        const uint64_t add = percounter_soc_sram_add_address(address);
        const auto *assigned = waits + 1 + begin;
        if (!percounter_calendar_wait(
                cfg,
                [&]() __aicore__ {
                    for (uint32_t work = pending; work != 0; work &= work - 1) {
                        const uint32_t i = static_cast<uint32_t>(__builtin_ctz(work));
                        const auto state = percounter_calendar_poll_completion(
                            cfg, assigned[i].read_cmd, assigned[i].target, &cursors[i]
                        );
                        if (state == CompletionPollState::FAILED) {
                            failed = true;
                            return true;
                        }
                        if (state != CompletionPollState::READY) continue;
                        // Publish the dependencies observed by this lane before
                        // followers start on the shared completion threshold.
                        percounter_memory_barrier();
                        percounter_calendar_add(add);
                        pending &= ~(1U << i);
                    }
                    return pending == 0;
                },
                retry_cycles
            ) ||
            failed)
            return false;
    }
    return percounter_calendar_wait_threshold(cfg, address, static_cast<uint32_t>(target), retry_cycles);
}

inline __aicore__ bool percounter_calendar_wait_range(
    const LaneConfig &cfg, const __gm__ uint8_t *image, uint64_t bytes, uint32_t offset, uint32_t count,
    uint64_t range_key = UINT64_MAX
) {
    if (count == 0) return true;
    const auto *waits = percounter_calendar_table<CalendarWait>(image, bytes, offset, count);
    if (waits == nullptr) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return false;
    }
    if (!percounter_calendar_wait_list(cfg, waits, count, 0, range_key)) return false;
    percounter_memory_barrier();
    return true;
}

// The observation clock supplies the counted wait's retry deadline, including at L0.
inline __aicore__ uint32_t percounter_read_dmb_sample(uint64_t &cycles) {
#if defined(__CCE_AICORE__)
    uint64_t token;
    asm volatile("MOV %0, DATA_MAIN_BASE\nMOV %1, SYS_CNT" : "=&l"(token), "=&l"(cycles) : : "memory");
    return static_cast<uint32_t>(token);
#else
    const auto token = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    cycles = get_sys_cnt_aicore();
    return token;
#endif
}

inline __aicore__ void percounter_record_dmb_sample(
    const LaneConfig &cfg, uint64_t entry, uint64_t cycles, uint64_t previous, uint32_t token, uint32_t sequence
) {
    if constexpr (task_profile_compiled) {
        const auto kind =
            previous == 0 ? PercounterTaskEventKind::PeerDmbMovAtEntry : PercounterTaskEventKind::PeerDmbMovChanged;
        if (auto *sample = percounter_record_task_event(cfg, kind, entry, cycles, (uint64_t{token} << 32) | sequence)) {
            const auto delta = cycles - previous;
            sample->previous_poll_delta =
                previous != 0 && delta < UINT32_MAX ? static_cast<uint32_t>(delta) : UINT32_MAX;
        }
    }
}

// A return hint prompts a check of the original DONE and defer states.
inline constexpr uint64_t CALENDAR_COMPLETION_RECHECK_CYCLES = 300;

template <uint32_t Inputs, bool ReadDmb, bool Spmd = false>
inline __aicore__ bool percounter_wait_paced_inputs(
    const LaneConfig &cfg, const __gm__ CalendarWait *waits, uint32_t token_base, uint64_t entry
) {
    uint64_t commands[Inputs], targets[Inputs];
    for (uint32_t i = 0; i < Inputs; ++i) {
        commands[i] = waits[i].read_cmd;
        targets[i] = waits[i].target;
        const uint64_t tag = commands[i] & CALENDAR_WAIT_TAG_MASK;
        if ((tag != CALENDAR_WAIT_NOTIFY && tag != CALENDAR_WAIT_DEFER) ||
            (commands[i] & ~CALENDAR_WAIT_TAG_MASK) == 0 || (targets[i] & CALENDAR_COUNT_MASK) == 0) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
            return false;
        }
    }

    const uint64_t begin = percounter_task_event_clock(cfg);
    uint64_t mov_cycles = 0;
    uint64_t last_check = 0;
    uint64_t seen = UINT64_MAX;
    uint64_t cursor = 0;
    uint32_t ready_inputs = 0;
    bool failed = false;
    const bool ready = percounter_calendar_wait(cfg, [&]() __aicore__ {
        const uint64_t previous_poll = mov_cycles;
        uint32_t token = 0;
        bool changed = false;
        const bool initial = last_check == 0;
        if constexpr (ReadDmb) {
            token = percounter_read_dmb_sample(mov_cycles);
            changed = token != seen && (Spmd || token - token_base < Inputs);
            seen = token;
        } else {
            mov_cycles = get_sys_cnt_aicore();
        }
        const uint64_t elapsed = mov_cycles - last_check;
        if (!initial && !changed && elapsed < CALENDAR_COMPLETION_RECHECK_CYCLES) return false;

        while (ready_inputs < Inputs) {
            const auto state =
                percounter_calendar_poll_completion(cfg, commands[ready_inputs], targets[ready_inputs], &cursor);
            if (state == CompletionPollState::FAILED) {
                failed = true;
                return true;
            }
            if (state == CompletionPollState::PENDING) break;
            ++ready_inputs;
            cursor = 0;
        }
        const bool satisfied = ready_inputs == Inputs;
        if constexpr (!task_profile_compiled) {
            if (satisfied) return true;
        }
        last_check = get_sys_cnt_aicore();
        if constexpr (task_profile_compiled && ReadDmb) {
            if (cfg.task_events != nullptr) {
                if (initial || changed) {
                    percounter_record_dmb_sample(
                        cfg, entry, mov_cycles, previous_poll, token, Spmd ? token ^ token_base : 0
                    );
                    const auto kind =
                        initial ? PercounterTaskEventKind::PeerInitialProof : PercounterTaskEventKind::PeerObserved;
                    auto *observation = percounter_record_task_event(
                        cfg, kind, entry, last_check, (uint64_t{token} << 32) | ready_inputs
                    );
                    if (observation != nullptr && !satisfied) {
                        // Publish the unsatisfied observation while the receiver is still waiting.
                        dcci(observation, SINGLE_CACHE_LINE, CACHELINE_OUT);
                        percounter_memory_barrier();
                    }
                } else if (satisfied) {
                    const uint64_t interval = elapsed > UINT32_MAX ? UINT32_MAX : elapsed;
                    percounter_record_task_event(
                        cfg, PercounterTaskEventKind::PeerPeriodicReady, entry, last_check,
                        (interval << 32) | ready_inputs
                    );
                }
            }
        }
        return satisfied;
    });
    if constexpr (ReadDmb)
        percounter_record_task_event(cfg, PercounterTaskEventKind::PeerWaitBegin, entry, begin, Inputs);
    return ready && !failed;
}

inline __aicore__ bool percounter_wait_spmd_input(
    const LaneConfig &cfg, const __gm__ CalendarWait *waits, uint32_t count, uint64_t entry, uint32_t worker
) {
    if (count != 1 || cfg.peer_endpoints == nullptr ||
        cfg.peer_endpoints[worker].input_count != CALENDAR_PEER_SPMD_ENDPOINT) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
        return false;
    }
    return percounter_wait_paced_inputs<1, true, true>(cfg, waits, cfg.peer_endpoints[worker].token_base, entry);
}

inline __aicore__ bool percounter_calendar_enter(
    const LaneConfig &cfg, const __gm__ CalendarLocalJoin *join, const __gm__ CalendarWait *waits, uint32_t wait_count,
    uint32_t sequence_base, uint32_t blocks, bool sync, uint64_t entry, uint16_t flags, uint32_t worker
) {
    // Keep the first probes immediate. Longer groups use a staggered
    // retry interval so idle lanes do not flood one completion word.
    const uint32_t retry_cycles = wait_count == 1 && !sync ? percounter_calendar_retry_cycles(cfg, blocks, true) : 0;
    const auto enter = [&]() __aicore__ {
        if ((flags & CALENDAR_EXEC_SPMD_WAIT) != 0) {
            if (!percounter_wait_spmd_input(cfg, waits, wait_count, entry, worker)) return false;
        } else if ((flags & CALENDAR_EXEC_PACED_WAIT) != 0) {
            if (wait_count != 1 && wait_count != 2) {
                percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
                return false;
            }
            const bool ready = wait_count == 1 ? percounter_wait_paced_inputs<1, false>(cfg, waits, 0, entry) :
                                                 percounter_wait_paced_inputs<2, false>(cfg, waits, 0, entry);
            if (!ready) return false;
        } else if (!percounter_calendar_wait_list(cfg, waits, wait_count, entry, UINT64_MAX, retry_cycles)) {
            return false;
        }
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::InputsReady, entry, percounter_task_event_clock(cfg)
        );
        return true;
    };
    if (join == nullptr) return enter();

    ProfileScope mix_scope(cfg.scalar, ProfilePhase::MixRendezvous, entry);
    const uint32_t lane = static_cast<uint32_t>(cfg.cluster_lane);
    const uint32_t lead = join->leader_lane;
    uint32_t arrivals[PLATFORM_CORES_PER_BLOCKDIM]{};
    for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer) {
        const uint32_t ordinal = join->arrive[peer];
        if (ordinal != 0) arrivals[peer] = sequence_base + ordinal;
    }
    auto *signal = &cfg.ssbuf->mix;
    // Tokens are per-lane ordinals across phases. A newer token also
    // satisfies an observer still polling an older arrival or release.
    if (lead != lane) {
        percounter_ssbuf_store_release(&signal->arrive[lane], arrivals[lane]);
        if (!percounter_calendar_wait(cfg, [&]() __aicore__ {
                return percounter_ssbuf_load_field(&signal->release[lead]) >= arrivals[lead];
            }))
            return false;
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::MixReleaseObserved, entry, percounter_task_event_clock(cfg)
        );
        return true;
    }
    if (!percounter_calendar_wait(cfg, [&]() __aicore__ {
            for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
                if (peer != lane && arrivals[peer] != 0 &&
                    percounter_ssbuf_load_field(&signal->arrive[peer]) < arrivals[peer])
                    return false;
            return true;
        }))
        return false;
    // Only the selected role checks data dependencies after all local roles
    // have reached the same calendar block.
    if (!enter()) return false;
    percounter_ssbuf_store_release(&signal->release[lane], arrivals[lane]);
    return true;
}

template <bool Summarized = false>
inline __aicore__ bool percounter_calendar_wait_cluster_input(
    const LaneConfig &cfg, const __gm__ CalendarWait &wait, uint32_t sequence, uint64_t entry
) {
    const uint32_t peer = 3 - cfg.cluster_lane;
    const uint64_t command = wait.read_cmd;
    const uint64_t target = wait.target;
    auto *summary = reinterpret_cast<__gm__ volatile uint32_t *>(command & ~CALENDAR_WAIT_TAG_MASK);
    const uint32_t threshold = static_cast<uint32_t>(target);
    if constexpr (Summarized) {
        const uint64_t tag = command & CALENDAR_WAIT_TAG_MASK;
        const uint32_t writers = static_cast<uint32_t>(target >> 32);
        if (summary == nullptr || threshold == 0 || threshold > CALENDAR_COUNT_MASK ||
            (tag == CALENDAR_WAIT_FUSED_FANIN ? (writers < 2 || writers > CALENDAR_FUSED_FANIN_MAX_PRODUCERS) :
                                                (tag != CALENDAR_WAIT_NOTIFY || writers != 0))) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
            return false;
        }
    }
    uint32_t attempts = 0;
    uint64_t cursor = 0;
    bool failed = false;
    if (!percounter_calendar_wait(
            cfg,
            [&]() __aicore__ {
                if (percounter_ssbuf_load_field(&cfg.ssbuf->mix.release[peer]) >= sequence) return true;
                // Device communication may have dependencies outside this rank's
                // calendar. Keep the original DONE observer as an escape path.
                constexpr uint32_t interval =
                    Summarized ? CALENDAR_FOLLOWER_FANIN_INTERVAL : CALENDAR_FOLLOWER_DONE_INTERVAL;
                if (++attempts % interval != 0) return false;
                if constexpr (Summarized) {
                    // A follower has no assigned producer to service.
                    return (percounter_soc_sram_read(*summary) & CALENDAR_COUNT_MASK) >= threshold;
                } else {
                    const auto state = percounter_calendar_poll_completion(cfg, command, target, &cursor);
                    failed = state == CompletionPollState::FAILED;
                    return state != CompletionPollState::PENDING;
                }
            }
        ) ||
        failed)
        return false;
    // The memory barrier before kernel entry acquires either observation.
    percounter_record_task_event(cfg, PercounterTaskEventKind::InputsReady, entry, percounter_task_event_clock(cfg));
    return true;
}

inline __aicore__ bool percounter_wait_peer_inputs(
    const LaneConfig &cfg, const __gm__ CalendarPeerExtra &peer, const __gm__ CalendarWait *waits, uint32_t wait_count,
    uint32_t worker, uint64_t entry
) {
    const uint32_t inputs = peer.input_count;
    if (cfg.peer_endpoints == nullptr || inputs == 0 || inputs > 2 || inputs != wait_count ||
        inputs != cfg.peer_endpoints[worker].input_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, peer.execution.task_id);
        return false;
    }
    const uint32_t base = cfg.peer_endpoints[worker].token_base;
    return inputs == 1 ? percounter_wait_paced_inputs<1, true>(cfg, waits, base, entry) :
                         percounter_wait_paced_inputs<2, true>(cfg, waits, base, entry);
}

#if !defined(__DAV_CUBE__)
inline __aicore__ void percounter_write_peer_dmb(
    const LaneConfig &cfg, __gm__ uint32_t *dmb, uint32_t worker, uint32_t token, uint64_t entry
) {
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    percounter_release_dmb_source(cfg);
    pipe_barrier(PIPE_ALL);
    set_atomic_none();
    set_loop_size_ubtoout((1U << 21) | 1U);
    auto *word = reinterpret_cast<__ubuf__ uint32_t *>(32);
    *word = token;
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
#endif
    const uint64_t write_begin = percounter_task_event_clock(cfg);
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    copy_ubuf_to_gm_align_v2(dmb, word, 0, 1, sizeof(token), 0, 0, 0);
#else
    __atomic_store_n(dmb, token, __ATOMIC_RELEASE);
#endif
    const uint64_t issued = percounter_task_event_clock(cfg);
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    cfg.peer_dmb_source_in_use = true;
    constexpr auto published = PercounterTaskEventKind::PeerMtePublished;
#else
    constexpr auto published = PercounterTaskEventKind::PeerPublished;
#endif
    const uint64_t detail = (uint64_t{worker} << 32) | token;
    percounter_record_task_event(cfg, PercounterTaskEventKind::PeerWriteBegin, entry, write_begin, detail);
    percounter_record_task_event(cfg, published, entry, issued, detail);
}

// A DMB wakeup can make its receiver read CntNotify immediately.
inline __aicore__ void percounter_order_credit_before_dmb() {
#if defined(__CCE_AICORE__)
    dsb(DSB_DDR);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

#endif  // !__DAV_CUBE__

#if !defined(__DAV_CUBE__)
inline __aicore__ bool percounter_send_peer_notifications(
    const LaneConfig &cfg, const __gm__ uint8_t *image, uint64_t bytes, const __gm__ CalendarPeerExtra &peer,
    uint64_t entry
) {
    const uint32_t count = peer.send_count;
    const auto *sends = percounter_calendar_table<CalendarPeerSend>(image, bytes, peer.sends_offset, count);
    if (sends == nullptr || cfg.peer_endpoints == nullptr) return false;
    percounter_order_credit_before_dmb();
    for (uint32_t i = 0; i < count; ++i) {
        const auto &send = sends[i];
        const uint32_t worker = send.worker;
        if (worker >= cfg.cluster_count * PLATFORM_CORES_PER_BLOCKDIM) return false;
        const auto &target = cfg.peer_endpoints[worker];
        const uint32_t ordinal = send.ordinal;
        if (ordinal >= target.input_count || target.address == 0) return false;
        auto *dmb = reinterpret_cast<__gm__ uint32_t *>(target.address);
        percounter_write_peer_dmb(cfg, dmb, worker, target.token_base + ordinal, entry);
    }
    return true;
}

#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
inline __aicore__ void percounter_finish_dmb_batch(
    const LaneConfig &cfg, uint64_t entry, uint64_t begin, uint32_t count, uint32_t commands, uint32_t sequence
) {
    const uint64_t issued = percounter_task_event_clock(cfg);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    cfg.peer_dmb_source_in_use = true;
    const uint64_t detail = (uint64_t{commands} << 48) | (uint64_t{count} << 32) | sequence;
    percounter_record_task_event(cfg, PercounterTaskEventKind::PeerBatchBegin, entry, begin, detail);
    percounter_record_task_event(cfg, PercounterTaskEventKind::PeerMteBatchIssued, entry, issued, detail);
}

template <bool SharedWord, uint32_t Width = 0>
inline __aicore__ void percounter_issue_dmb_batch(
    const LaneConfig &cfg, const __ubuf__ uint64_t *addresses, uint32_t count, __ubuf__ uint32_t *words, uint64_t entry,
    uint32_t sequence
) {
    constexpr uint32_t words_per_line = 32 / sizeof(uint32_t);
    const uint64_t first = addresses[0];
    if constexpr (task_profile_compiled) asm volatile("" : : "l"(first) : "memory");
    const uint64_t begin = percounter_task_event_clock(cfg);
    copy_ubuf_to_gm_align_v2(reinterpret_cast<__gm__ uint32_t *>(first), words, 0, 1, sizeof(uint32_t), 0, 0, 0);
    const auto write = [&](uint32_t i) __aicore__ {
        auto *destination = reinterpret_cast<__gm__ uint32_t *>(addresses[i]);
        auto *source = SharedWord ? words : words + i * words_per_line;
        copy_ubuf_to_gm_align_v2(destination, source, 0, 1, sizeof(uint32_t), 0, 0, 0);
    };
    if constexpr (Width != 0) {
#pragma clang loop unroll(full)
        for (uint32_t i = 1; i < Width; ++i)
            write(i);
    } else {
        for (uint32_t i = 1; i < count; ++i)
            write(i);
    }
    percounter_finish_dmb_batch(cfg, entry, begin, count, count, sequence);
}

#endif

__attribute__((noinline)) __aicore__ bool percounter_send_prepared_fanout(
    const LaneConfig &cfg, const __gm__ uint8_t *image, uint64_t bytes, uint32_t offset, uint32_t count,
    uint32_t sequence, uint64_t entry
) {
    const auto *fanout = percounter_calendar_table<CalendarPeerFanout>(image, bytes, offset, 1);
    if (fanout == nullptr || count > SCHEDULER_WORKER_CAPACITY || fanout->count != count ||
        sizeof(*fanout) + uint64_t{count} * CALENDAR_FANOUT_TARGET_BYTES > bytes - offset)
        return false;
    const auto *addresses = reinterpret_cast<const __gm__ uint64_t *>(fanout + 1);
    const auto *bases = reinterpret_cast<const __gm__ uint32_t *>(addresses + count);
    const uint64_t common = fanout->common_token_base;
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    constexpr uint32_t words_per_line = 32 / sizeof(uint32_t);
    auto *words = reinterpret_cast<__ubuf__ uint32_t *>(32);

    percounter_release_dmb_source(cfg);
    pipe_barrier(PIPE_ALL);
    set_atomic_none();
    set_loop_size_ubtoout((1U << 21) | 1U);
    const uint32_t commands = fanout->burst_count;
    if (commands != 0) {
        if (common == UINT64_MAX || commands > count * sizeof(uint32_t) / sizeof(CalendarPeerBurst)) return false;
        const auto *bursts = reinterpret_cast<const __gm__ CalendarPeerBurst *>(bases);
        *words = static_cast<uint32_t>(common) ^ sequence;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        const uint64_t first_address = bursts[0].start;
        const uint32_t first_count = bursts[0].targets;
        const uint32_t first_stride = bursts[0].stride_bytes;
        if constexpr (task_profile_compiled)
            asm volatile("" : : "l"(first_address), "l"(uint64_t{first_count}), "l"(uint64_t{first_stride}) : "memory");
        const uint64_t begin = percounter_task_event_clock(cfg);
        copy_ubuf_to_gm_align_v2(
            reinterpret_cast<__gm__ uint32_t *>(first_address), words, 0, first_count, sizeof(uint32_t), 0,
            first_stride, 0
        );
        for (uint32_t i = 1; i < commands; ++i) {
            auto *destination = reinterpret_cast<__gm__ uint32_t *>(bursts[i].start);
            copy_ubuf_to_gm_align_v2(
                destination, words, 0, bursts[i].targets, sizeof(uint32_t), 0, bursts[i].stride_bytes, 0
            );
        }
        percounter_finish_dmb_batch(cfg, entry, begin, count, commands, sequence);
        return true;
    }
    // Copy the immutable multicast list in one transfer while Scalar prepares the payload.
    auto *local_addresses = reinterpret_cast<__ubuf__ uint64_t *>(32 * (1 + SCHEDULER_WORKER_CAPACITY));
    set_loop_size_outtoub((1U << 21) | 1U);
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ uint8_t *>(local_addresses),
        reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ uint64_t *>(addresses)), 0, 1, count * sizeof(uint64_t), 0,
        0, false, 0, 0, 0
    );
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    if (common != UINT64_MAX) {
        *words = static_cast<uint32_t>(common) ^ sequence;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        // A full cube wave, vector wave and mixed wave have fixed multicast widths on A5.
        if (count == 32) percounter_issue_dmb_batch<true, 32>(cfg, local_addresses, count, words, entry, sequence);
        else if (count == 64) percounter_issue_dmb_batch<true, 64>(cfg, local_addresses, count, words, entry, sequence);
        else if (count == 96) percounter_issue_dmb_batch<true, 96>(cfg, local_addresses, count, words, entry, sequence);
        else percounter_issue_dmb_batch<true>(cfg, local_addresses, count, words, entry, sequence);
    } else {
        // Distinct source lines stay immutable while the whole batch is in flight.
        for (uint32_t i = 0; i < count; ++i)
            words[i * words_per_line] = bases[i] ^ sequence;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        percounter_issue_dmb_batch<false>(cfg, local_addresses, count, words, entry, sequence);
    }
#else
    const uint32_t shared_word = static_cast<uint32_t>(common) ^ sequence;
    const uint64_t begin = percounter_task_event_clock(cfg);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t token = common == UINT64_MAX ? bases[i] ^ sequence : shared_word;
        auto *destination = reinterpret_cast<__gm__ uint32_t *>(addresses[i]);
        __atomic_store_n(destination, token, __ATOMIC_RELEASE);
    }
    const uint64_t issued = percounter_task_event_clock(cfg);
    const uint64_t detail = (uint64_t{count} << 32) | sequence;
    percounter_record_task_event(cfg, PercounterTaskEventKind::PeerBatchBegin, entry, begin, detail);
    percounter_record_task_event(cfg, PercounterTaskEventKind::PeerScalarBatchIssued, entry, issued, detail);
#endif
    return true;
}

inline __aicore__ bool percounter_send_spmd_return_hint(
    const LaneConfig &cfg, const __gm__ uint8_t *image, uint64_t bytes, const __gm__ CalendarSpmdPeerExtra &peer,
    uint32_t block, uint32_t role, uint64_t entry, bool deferred
) {
    const uint32_t count = peer.target_count;
    const uint32_t sequence = peer.token_begin + PLATFORM_CORES_PER_BLOCKDIM * block + role;
    percounter_order_credit_before_dmb();
    if (count > 1) {
        // Ordinary returns may broadcast once the cohort reaches its DONE threshold.
        // A returning deferred member can still prompt subscribers to service its registration.
        if (!deferred) {
            const uint64_t command = peer.completion.read_cmd;
            const uint64_t tag = command & CALENDAR_WAIT_TAG_MASK;
            const uint64_t address = command & ~CALENDAR_WAIT_TAG_MASK;
            const uint32_t target = peer.completion.target & CALENDAR_COUNT_MASK;
            if ((tag != CALENDAR_WAIT_NOTIFY && tag != CALENDAR_WAIT_DEFER) || address == 0 || target == 0)
                return false;
            auto *counter = reinterpret_cast<__gm__ volatile uint32_t *>(address);
            if ((percounter_soc_sram_read(*counter) & CALENDAR_COUNT_MASK) < target) return true;
        }
        return percounter_send_prepared_fanout(cfg, image, bytes, peer.fanout_offset, count, sequence, entry);
    }
    const uint32_t worker = peer.single_worker;
    if (count != 1 || cfg.peer_endpoints == nullptr || worker >= cfg.cluster_count * PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    const auto &target = cfg.peer_endpoints[worker];
    if (target.input_count != CALENDAR_PEER_SPMD_ENDPOINT || target.address == 0) return false;
    percounter_write_peer_dmb(
        cfg, reinterpret_cast<__gm__ uint32_t *>(target.address), worker, target.token_base ^ sequence, entry
    );
    return true;
}

#endif  // !__DAV_CUBE__

inline __aicore__ bool percounter_calendar_execute_range(
    const LaneConfig &cfg, const __gm__ uint8_t *image, uint64_t bytes, const __gm__ CalendarRange &range,
    uint32_t sample_offset, uint32_t worker_row_begin, uint32_t range_index, uint32_t worker
) {
    const uint32_t row_count = range.row_count;
    const uint32_t binding_offset = range.binding_offset;
    const uint32_t sequence_base = range.sequence_base;
    const auto *rows = percounter_calendar_table<CalendarExecutionRow>(image, bytes, range.rows_offset, row_count);
    if ((row_count != 0 && (rows == nullptr || binding_offset < sizeof(CalendarRangeHeader) ||
                            binding_offset >= bytes || binding_offset % alignof(CalendarDispatch) != 0))) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return false;
    }
    __gm__ CalendarPmuCell *pmu = nullptr;
    if (range.pmu_offset != 0) {
        pmu = const_cast<__gm__ CalendarPmuCell *>(
            percounter_calendar_table<CalendarPmuCell>(image, bytes, range.pmu_offset, row_count)
        );
        if (pmu == nullptr || cfg.pmu_reg_base == 0) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
    }
    const __gm__ uint8_t *binding = image + binding_offset;
    __gm__ CalendarTiming *samples = nullptr;
    if (sample_offset != 0) {
        samples = const_cast<__gm__ CalendarTiming *>(
            percounter_calendar_table<CalendarTiming>(image, bytes, sample_offset, row_count)
        );
        if (samples == nullptr) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
    }
    uint64_t range_key = UINT64_MAX;
    if constexpr (task_profile_compiled) {
        if (cfg.task_events != nullptr) {
            range_key = (uint64_t{range.graph_shell_id} << 32) | range_index;
            percounter_record_task_event(
                cfg, PercounterTaskEventKind::RangeEnterBegin, range_key, percounter_task_event_clock(cfg)
            );
        }
    }
    {
        ProfileScope scope(cfg.scalar, ProfilePhase::RangeEnter);
        if (!percounter_calendar_wait_range(
                cfg, image, bytes, range.entry_wait_offset, range.entry_wait_count, range_key
            ))
            return false;
    }
    if (range_key != UINT64_MAX)
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::RangeEntered, range_key, percounter_task_event_clock(cfg)
        );

    const bool profile_tasks = cfg.profile != nullptr || cfg.profile_buffer != nullptr;
    for (uint32_t i = 0; i < row_count; ++i) {
        if (((worker_row_begin + i) % PERCOUNTER_SPIN_CHECK_INTERVAL) == 0 && percounter_should_stop(cfg)) return false;
        const uint64_t row_begin = percounter_task_event_clock(cfg);
        const auto *row = rows + i;
        const uint16_t flags = row->flags;
        const uint32_t block = row->block_index;
        const uint32_t blocks = row->block_count;
        const uint32_t role = flags & CALENDAR_EXEC_ROLE_MASK;
        const bool sync = (flags & CALENDAR_EXEC_SYNC_START) != 0;
        const bool sink = (flags & CALENDAR_EXEC_SINK) != 0;
        const bool control_row = (flags & CALENDAR_EXEC_CONTROL) != 0;
        const bool has_predicate = (flags & CALENDAR_EXEC_PREDICATE) != 0;
        const auto *dispatch = reinterpret_cast<const __gm__ CalendarDispatch *>(binding + row->dispatch_offset);
        const auto *waits = reinterpret_cast<const __gm__ CalendarWait *>(binding + dispatch->waits_offset);
        const auto *extra = reinterpret_cast<const __gm__ CalendarExecutionExtra *>(binding + dispatch->extra_offset);
        const auto *peer =
            (flags & CALENDAR_EXEC_PEER) != 0 ? reinterpret_cast<const __gm__ CalendarPeerExtra *>(extra) : nullptr;
        bool deferred = false;
        const uint32_t mix_delta = row->mix_delta;
        constexpr uint16_t non_join_flags =
            CALENDAR_EXEC_FANIN | CALENDAR_EXEC_INPUT_OWNER | CALENDAR_EXEC_INPUT_FOLLOWER;
        const auto *join = mix_delta != 0 && (flags & non_join_flags) == 0 ?
                               reinterpret_cast<const __gm__ CalendarLocalJoin *>(
                                   reinterpret_cast<const __gm__ uint8_t *>(row) + mix_delta
                               ) :
                               nullptr;
        const __gm__ volatile uint64_t &done_command = dispatch->done_add_cmd;
        const uint64_t done_add_cmd = done_command;
        const uint64_t entry = profile_tasks ? percounter_order_entry(extra->task_id, block, blocks) : 0;
        if constexpr (profile_compiled) {
            if (cfg.scalar != nullptr)
                cfg.scalar->append_function(
                    ProfilePhase::ResolveRow, row_begin, percounter_scalar_clock(), static_cast<int64_t>(entry)
                );
        }
        percounter_record_task_event(cfg, PercounterTaskEventKind::RowBegin, entry, row_begin);
        if (peer != nullptr && peer->input_count != 0) {
            if (!percounter_wait_peer_inputs(cfg, *peer, waits, row->wait_count, worker, entry)) return false;
            percounter_record_task_event(
                cfg, PercounterTaskEventKind::InputsReady, entry, percounter_task_event_clock(cfg)
            );
        } else if ((flags & CALENDAR_EXEC_INPUT_FOLLOWER) != 0) {
            if (row->wait_count != 0) {
                percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(entry));
                return false;
            }
            const uint32_t sequence = sequence_base + mix_delta;
            const bool ready = (flags & CALENDAR_EXEC_FANIN) != 0 ?
                                   percounter_calendar_wait_cluster_input<true>(cfg, waits[0], sequence, entry) :
                                   percounter_calendar_wait_cluster_input(cfg, waits[0], sequence, entry);
            if (!ready) return false;
        } else if ((flags & CALENDAR_EXEC_FANIN) != 0) {
            const auto retry_cycles = percounter_calendar_retry_cycles(cfg, blocks, row->wait_count != 0);
            if (!percounter_calendar_wait_fanin(cfg, waits, mix_delta, row->wait_count, entry, retry_cycles))
                return false;
            percounter_record_task_event(
                cfg, PercounterTaskEventKind::InputsReady, entry, percounter_task_event_clock(cfg)
            );
        } else if (!percounter_calendar_enter(
                       cfg, join, waits, row->wait_count, sequence_base, blocks, sync, entry, flags, worker
                   )) {
            return false;
        }
        if ((flags & CALENDAR_EXEC_INPUT_OWNER) != 0) {
            const uint32_t sequence = sequence_base + range.phase_row_offset + i + 1;
            percounter_ssbuf_store_release(&cfg.ssbuf->mix.release[cfg.cluster_lane], sequence);
        }
        percounter_record_task_event(cfg, PercounterTaskEventKind::Ready, entry, percounter_task_event_clock(cfg));
        __gm__ CalendarTiming *sample = samples != nullptr ? samples + i : nullptr;
        __gm__ CalendarPmuCell *measurement = pmu != nullptr && !control_row ? pmu + i : nullptr;
        bool skip_kernel = control_row;
        if (has_predicate) {
            const uint64_t address = percounter_gm_read_immutable_word(&extra->predicate_address);
            const auto verdict =
                scheduler_evaluate_task_predicate(reinterpret_cast<__gm__ SchedulerDispatchPredicate *>(address));
            if (verdict == SchedulerPredicateResult::MALFORMED) {
                percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, extra->task_id);
                return false;
            }
            skip_kernel = verdict == SchedulerPredicateResult::FAIL;
        }
        const bool capture = !control_row && cfg.args_dump_enabled;
        if (capture && !calendar_args_dump_boundary(cfg, extra->task_id, ArgsDumpStage::BEFORE_DISPATCH)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, extra->task_id);
            return false;
        }
        if (percounter_exit_signalled(cfg)) return false;
        if (skip_kernel) {
            if (measurement != nullptr) measurement->publication = CalendarPmuPublication::Skipped;
            percounter_memory_barrier();
            if (sample != nullptr) {
                const uint64_t cycles = get_sys_cnt_aicore();
                sample->start = cycles;
                sample->end = cycles;
            }
        } else {
            if (!percounter_run_task(cfg, *dispatch, extra, block, blocks, role, entry, sample, measurement, flags))
                return false;
            // A long kernel can return after the supervisor signals EXIT.
            // The rate-limited EXIT poll then suppresses late completion credit.
            if (percounter_exit_signalled(cfg)) return false;
            auto *touched = &cfg.ssbuf->contexts[cfg.cluster_lane].value.async_touched;
            if (percounter_ssbuf_load_field(touched) != 0) {
                if (!percounter_calendar_seal(cfg, extra, sink, sample, entry, &deferred)) return false;
                percounter_ssbuf_store_field(touched, uint32_t{0});
            }
        }
        if (measurement != nullptr) {
            dcci(measurement, SINGLE_CACHE_LINE, CACHELINE_OUT);
            percounter_memory_barrier();
        }
        if (capture && !calendar_args_dump_boundary(cfg, extra->task_id, ArgsDumpStage::AFTER_COMPLETION)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, extra->task_id);
            return false;
        }
        if (!deferred) {
            ProfileScope profile_scope(cfg.scalar, ProfilePhase::Complete, entry);
            percounter_calendar_add(done_add_cmd);
            const uint64_t issued = percounter_task_event_clock(cfg);
            if (sink) percounter_calendar_add(extra->shell_add_cmd);
            percounter_record_task_event(
                cfg, PercounterTaskEventKind::CompletionIssue, entry, issued, done_add_cmd != 0
            );
        }
#if !defined(__DAV_CUBE__)
        if (!skip_kernel) {
            bool notified = true;
            if ((flags & CALENDAR_EXEC_SPMD_SEND) != 0) {
                notified = percounter_send_spmd_return_hint(
                    cfg, image, bytes, *reinterpret_cast<const __gm__ CalendarSpmdPeerExtra *>(extra), block, role,
                    entry, deferred
                );
            } else if (peer != nullptr && peer->send_count != 0) {
                notified = percounter_send_peer_notifications(cfg, image, bytes, *peer, entry);
            }
            if (!notified) {
                percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, extra->task_id);
                return false;
            }
        }
#endif
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::LocalRetired, entry, percounter_task_event_clock(cfg)
        );
        percounter_commit_task_timing(cfg);
    }

    {
        ProfileScope scope(cfg.scalar, ProfilePhase::RangeExit);
        if (!percounter_calendar_wait_range(cfg, image, bytes, range.exit_wait_offset, range.exit_wait_count))
            return false;
        percounter_calendar_add(range.tail_add_cmd);
    }
    return true;
}

inline __aicore__ void percounter_calendar_loop(const LaneConfig &cfg) {
    const uint64_t offset = cfg.control->calendar_offset;
    const uint64_t image_bytes = cfg.control->image_bytes;
    if (cfg.aiv_per_cluster != 2 || offset < sizeof(PercounterControl) || offset % alignof(CalendarRangeHeader) != 0 ||
        offset > image_bytes || image_bytes - offset < sizeof(CalendarRangeHeader)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return;
    }
    const auto *calendar = scheduler_state_at<CalendarRangeHeader>(cfg.control, offset);
    const auto *image = reinterpret_cast<const __gm__ uint8_t *>(calendar);
    const uint64_t bytes = image_bytes - offset;
    const uint32_t workers = calendar->worker_count;
    const uint32_t range_count = calendar->range_count;
    const auto *indices =
        percounter_calendar_table<uint32_t>(image, bytes, calendar->worker_index_offset, uint64_t{workers} + 1);
    const auto *ranges = percounter_calendar_table<CalendarRange>(image, bytes, calendar->range_offset, range_count);
    const uint32_t sample_offset = calendar->sample_index_offset;
    const auto *sample_indices =
        sample_offset == 0 ? nullptr : percounter_calendar_table<uint32_t>(image, bytes, sample_offset, range_count);
    if (workers != cfg.cluster_count * PLATFORM_CORES_PER_BLOCKDIM || indices == nullptr || ranges == nullptr ||
        (sample_offset != 0 && sample_indices == nullptr)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return;
    }

    const uint32_t lane = static_cast<uint32_t>(cfg.cluster_lane);
    const uint32_t cluster = static_cast<uint32_t>(cfg.cluster_index);
    const uint32_t worker = lane == 0 ? cluster : cfg.cluster_count + cluster * cfg.aiv_per_cluster + lane - 1;
    if (calendar->peer_endpoints_offset != 0) {
        cfg.peer_endpoints =
            percounter_calendar_table<CalendarPeerEndpoint>(image, bytes, calendar->peer_endpoints_offset, workers);
        if (cfg.peer_endpoints == nullptr) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return;
        }
    }
    const uint32_t begin = indices[worker];
    const uint32_t end = indices[worker + 1];
    if (begin > end || end > range_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return;
    }
    uint32_t worker_row_begin = 0;
    for (uint32_t i = begin; i < end; ++i) {
        if (!percounter_calendar_execute_range(
                cfg, image, bytes, ranges[i], sample_indices != nullptr ? sample_indices[i] : 0, worker_row_begin, i,
                worker
            ))
            return;
        worker_row_begin += ranges[i].row_count;
    }
    if (sample_indices == nullptr && cfg.pmu_reg_base == 0) return;
    // Diagnostic writes must reach GM before retirement, off the phase gates.
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t sample_span = sample_indices == nullptr ? 0 : sample_indices[i];
        if (sample_span != 0) {
            auto *samples =
                reinterpret_cast<__gm__ CalendarTiming *>(const_cast<__gm__ uint8_t *>(image) + sample_span);
            for (uint32_t j = 0; j < ranges[i].row_count; ++j)
                dcci(samples + j, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        const uint32_t pmu_span = ranges[i].pmu_offset;
        if (pmu_span != 0) {
            auto *cells = reinterpret_cast<__gm__ CalendarPmuCell *>(const_cast<__gm__ uint8_t *>(image) + pmu_span);
            for (uint32_t j = 0; j < ranges[i].row_count; ++j)
                dcci(cells + j, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
    }
    dsb((mem_dsb_t)0);
}
