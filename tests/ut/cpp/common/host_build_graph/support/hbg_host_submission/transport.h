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
// Allocation and transport faults; all admission decisions stay in production code.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace hbg_host_transport {
struct Launch {
    std::string name;
    void *stream;
    std::vector<uint8_t> bytes;
};
struct Operation {
    std::string name;
    void *stream;
    void *event;
};
struct Control {
    std::vector<Launch> launches;
    std::vector<Operation> operations;
    std::vector<void *> synchronized;
    size_t query_calls{0};
    size_t step{0};
    size_t fail_step{0};
    int sync_rc{0};
    size_t sync_calls{0};
    size_t fail_sync{0};
    int query_rc{0};
    bool complete{true};
    bool throw_cpu{false};
    size_t allocated_bytes{0};
    size_t module_registrations{0};
    std::vector<void *> released_orchestration_modules;
    size_t forbidden_launch_calls{0};
    bool launching{false};
    int enqueue(std::string operation, void *stream = nullptr, void *event = nullptr);
};
Control &control();
void reset_trace();
bool live_allocation(const void *address, size_t bytes);
}  // namespace hbg_host_transport
