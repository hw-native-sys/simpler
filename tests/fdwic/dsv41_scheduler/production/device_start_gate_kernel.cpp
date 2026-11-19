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

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>

namespace {
// Each signal owns an A5 L2 cache line. Host alone writes release; the peer
// alone increments arrivals. Neither line is reset while a run can use it.
constexpr uint64_t kArrivalOffset = 128;
constexpr uint64_t kResultOffset = 256;
constexpr uint64_t kTimeoutTicks = 30'000'000'000ULL;  // A5 get_sys_cnt: 1 GHz.

__aicore__ __attribute__((always_inline)) inline void wait_epoch(pto::comm::Signal &signal, int32_t epoch) {
    const uint64_t start = get_sys_cnt();
    while (!pto::comm::TTEST(signal, epoch, pto::comm::WaitCmp::GE)) {
        // An unsuccessful gate must fail the stream, never continue into a
        // benchmark run and present an unsynchronized duration as a sample.
        if (get_sys_cnt() - start >= kTimeoutTicks) trap();
    }
    pto::comm::TWAIT(signal, epoch, pto::comm::WaitCmp::GE);
}
}  // namespace

// aclrtLaunchKernelWithHostArgs passes three consecutive uint64_t values.
// The launcher specifies engine=AIV and exactly one block.
extern "C" __global__[aicore] void dsv41_device_start_gate(uint64_t local_base, uint64_t peer_base, uint64_t epoch) {
    if (get_block_idx() != 0 || get_subblockid() != 0) return;
    pto::comm::Signal release(reinterpret_cast<__gm__ int32_t *>(local_base));
    wait_epoch(release, static_cast<int32_t>(epoch));
    pto::comm::Signal peer_arrival(reinterpret_cast<__gm__ int32_t *>(peer_base + kArrivalOffset));
    pto::comm::TNOTIFY(peer_arrival, 1, pto::comm::NotifyOp::AtomicAdd);
    const uint64_t begin = get_sys_cnt();
    pto::comm::Signal local_arrival(reinterpret_cast<__gm__ int32_t *>(local_base + kArrivalOffset));
    wait_epoch(local_arrival, static_cast<int32_t>(epoch));
    const uint64_t end = get_sys_cnt();
    auto *result = reinterpret_cast<__gm__ uint64_t *>(local_base + kResultOffset);
    result[0] = epoch;
    result[1] = static_cast<uint64_t>(*reinterpret_cast<volatile __gm__ int32_t *>(local_base + kArrivalOffset));
    result[2] = begin;
    result[3] = end;
    dcci(reinterpret_cast<__gm__ void *>(result), cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
