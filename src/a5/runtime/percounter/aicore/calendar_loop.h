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
            const uint32_t observed = percounter_notify_query(*counter) & CALENDAR_COUNT_MASK;
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
    if (address == 0) return;
#if defined(__CPU_SIM)
    // Simulated notifications keep the counter at the SET/read address.
    address -= percounter_notify_add_address(1) - 1;
#endif
    percounter_notify_add(*reinterpret_cast<__gm__ volatile uint32_t *>(address), 1);
}

inline __aicore__ void percounter_calendar_set_defer_bit(uint64_t address, uint32_t bit) {
    percounter_scalar_notify_issue(ProfilePhase::CntNotifyDeferIssue, address, bit, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
        __builtin_cce_st_dev(bit, reinterpret_cast<__gm__ uint32_t *>(address + 16), 0);
#else
        auto *notify = reinterpret_cast<uint32_t *>(address);
        (void)__atomic_fetch_or(notify, bit, __ATOMIC_RELEASE);
#endif
    });
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
    group->pending_mask = (uint64_t{1} << count) - 1;
    group->completion_add_cmd = percounter_notify_add_address(address);
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
    percounter_calendar_set_defer_bit(address, bit);
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
        return (percounter_notify_query(*counter) & CALENDAR_COUNT_MASK) >= target ? CompletionPollState::READY :
                                                                                     CompletionPollState::PENDING;
    const uint32_t task = static_cast<uint32_t>(target >> 32);
    const uint32_t bit_index = static_cast<uint8_t>(target >> 24);
    if (tag != CALENDAR_WAIT_DEFER || task >= cfg.task_count || bit_index < 24 || bit_index >= 32) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS, task, bit_index);
        return CompletionPollState::FAILED;
    }
    const uint32_t word = percounter_notify_query(*counter);
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
                        return (percounter_notify_query(*notify) & CALENDAR_COUNT_MASK) >= target;
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
    const uint32_t word = percounter_notify_query(*counter);
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
        const uint64_t add = percounter_notify_add_address(address);
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

inline __aicore__ bool percounter_calendar_enter(
    const LaneConfig &cfg, const __gm__ CalendarLocalJoin *join, const __gm__ CalendarWait *waits, uint32_t wait_count,
    uint32_t sequence_base, uint32_t blocks, bool sync, uint64_t entry
) {
    // Keep the first probes immediate. Longer groups use a staggered
    // retry interval so idle lanes do not flood one completion word.
    const uint32_t retry_cycles = wait_count == 1 && !sync ? percounter_calendar_retry_cycles(cfg, blocks, true) : 0;
    if (join == nullptr) {
        if (!percounter_calendar_wait_list(cfg, waits, wait_count, entry, UINT64_MAX, retry_cycles)) return false;
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::InputsReady, entry, percounter_task_event_clock(cfg)
        );
        return true;
    }

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
    if (!percounter_calendar_wait_list(cfg, waits, wait_count, entry, UINT64_MAX, retry_cycles)) return false;
    percounter_record_task_event(cfg, PercounterTaskEventKind::InputsReady, entry, percounter_task_event_clock(cfg));
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
                    return (percounter_notify_query(*summary) & CALENDAR_COUNT_MASK) >= threshold;
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

inline __aicore__ bool percounter_calendar_execute_range(
    const LaneConfig &cfg, const __gm__ uint8_t *image, uint64_t bytes, const __gm__ CalendarRange &range,
    uint32_t sample_offset, uint32_t worker_row_begin, uint32_t range_index
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
        if ((flags & CALENDAR_EXEC_INPUT_FOLLOWER) != 0) {
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
        } else if (!percounter_calendar_enter(cfg, join, waits, row->wait_count, sequence_base, blocks, sync, entry)) {
            return false;
        }
        if ((flags & CALENDAR_EXEC_INPUT_OWNER) != 0) {
            const uint32_t sequence = sequence_base + range.phase_row_offset + i + 1;
            percounter_ssbuf_store_release(&cfg.ssbuf->mix.release[cfg.cluster_lane], sequence);
        }
        percounter_record_task_event(cfg, PercounterTaskEventKind::Ready, entry, percounter_task_event_clock(cfg));
        __gm__ CalendarTiming *sample = samples != nullptr ? samples + i : nullptr;
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
        if (skip_kernel) {
            percounter_memory_barrier();
            if (sample != nullptr) {
                const uint64_t cycles = get_sys_cnt_aicore();
                sample->start = cycles;
                sample->end = cycles;
            }
        } else {
            if (!percounter_run_task(cfg, *dispatch, extra, block, blocks, role, entry, sample)) return false;
            auto *touched = &cfg.ssbuf->contexts[cfg.cluster_lane].value.async_touched;
            if (percounter_ssbuf_load_field(touched) != 0) {
                if (!percounter_calendar_seal(cfg, extra, sink, sample, entry, &deferred)) return false;
                percounter_ssbuf_store_field(touched, uint32_t{0});
            }
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
    const uint32_t begin = indices[worker];
    const uint32_t end = indices[worker + 1];
    if (begin > end || end > range_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return;
    }
    uint32_t worker_row_begin = 0;
    for (uint32_t i = begin; i < end; ++i) {
        if (!percounter_calendar_execute_range(
                cfg, image, bytes, ranges[i], sample_indices != nullptr ? sample_indices[i] : 0, worker_row_begin, i
            ))
            return;
        worker_row_begin += ranges[i].row_count;
    }
    if (sample_indices == nullptr) return;
    // Sample writeback must complete before the lane retires, off the phase gates.
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t sample_span = sample_indices[i];
        if (sample_span == 0) continue;
        const auto *samples = reinterpret_cast<const __gm__ CalendarTiming *>(image + sample_span);
        for (uint32_t j = 0; j < ranges[i].row_count; ++j)
            dcci(const_cast<__gm__ CalendarTiming *>(samples + j), SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    dsb((mem_dsb_t)0);
}
