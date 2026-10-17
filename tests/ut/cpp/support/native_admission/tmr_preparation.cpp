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
#include "fixture.h"
#include "host/host_graph_exporter.h"

namespace native_admission_test {
void *primary_entry_points() { return nullptr; }
void *alternate_entry_points() { return nullptr; }
void set_dep_gen_capture(bool) {}
uint32_t arena_bank(uint32_t) { return 0; }
}  // namespace native_admission_test

extern "C" bool dep_gen_host_graph_active() { return false; }
extern "C" int dep_gen_host_graph_emit(const char *) { return -1; }
extern "C" int dep_gen_host_graph_take(simpler::dfx::host_graph::HostGraphExport *) { return -1; }
