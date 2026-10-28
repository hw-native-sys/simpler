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

#include "aicpu/args_dump_aicpu.h"
#include "scheduler/scheduler_args_dump.h"
#include "scheduler/scheduler_graph.h"

// One object per actual AICPU participant. It owns one stable shared-producer
// reservation and a disjoint worker request partition for this invocation.
class ResidentArgsDumpService {
public:
    ResidentArgsDumpService() = default;
    ResidentArgsDumpService(const ResidentArgsDumpService &) = delete;
    ResidentArgsDumpService &operator=(const ResidentArgsDumpService &) = delete;
    bool initialize(
        void *image, const SchedulerGraphView &graph, uint64_t callable_objects, uint64_t callable_count,
        int32_t thread, int32_t threads
    );
    // At most one 64 KiB copy, including strided gather. True means actual
    // publication/copy progress, never an observation or a WouldBlock retry.
    bool step();
    // Called on the owning thread after its last step, before run retirement.
    // A pending record is abandoned only here or at its explicit abort boundary.
    void stop();
    bool has_open_record() const { return reservation_.open(); }
    bool enabled() const { return image_ != nullptr; }

private:
    void acknowledge(
        uint32_t worker, const SchedulerArgsDumpRequest &request, SchedulerArgsDumpStatus status, uint64_t cursor,
        uint64_t bytes
    );
    bool service(uint32_t worker, const SchedulerArgsDumpRequest &request);
    uint64_t callable(int32_t function_id) const;
    SchedulerArgsDumpHeader *image_{nullptr};
    SchedulerGraphView graph_{};
    uint64_t callable_objects_{0};
    uint64_t callable_count_{0};
    uint32_t lo_{0};
    uint32_t hi_{0};
    uint32_t head_{0};
    uint32_t pending_{0};
    int32_t thread_{-1};
    uint32_t owner_{UINT32_MAX};
    SchedulerArgsDumpRequest owner_request_{};
    DumpArgReservation reservation_{};
};

// Runs once on the init leader before any context/READY publication. Only this
// initialization may flush a task line; runtime phase words use GM atomics.
bool initialize_resident_args_dump_outputs(
    void *image, const SchedulerGraphView &graph, uint64_t callable_objects, uint64_t callable_count
);
