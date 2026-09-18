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
/**
 * @file device_fault_monitor_binding.cpp
 * @brief This host runtime's link to the process's device-fault monitor.
 *
 * A runtime `host_runtime.so` does not own the monitor and does not create
 * one: it is opened `RTLD_LOCAL` and `dlclose`d, so an instance here would be
 * one of several in a process, each with its own refcount, fighting over the
 * driver's single callback slot — and its code would be unmapped under a
 * notification still in flight.
 *
 * So this module holds only a pointer, which the loader fills in. Nothing here
 * names the driver: the trampoline the driver retains lives with the instance,
 * in the module that is loaded once per process and never unloaded.
 *
 * Unbound is a supported state, not an error. A test that `dlopen`s a host
 * runtime directly reaches no binder, sees no monitor, and installs nothing.
 *
 * The pointer is atomic because this variable is **shared per image, not per
 * loader**: `dlopen` of one path refcounts a single mapping, so two
 * `ChipWorker`s using the same runtime bind into the same word, and the
 * second's bind can land while the first's runs are reading it.
 */

#include "host/device_fault_monitor.h"

#include <atomic>

namespace {

// Atomic, not a plain pointer. `dlopen` of one path refcounts a single image,
// so two `ChipWorker`s on the same runtime share this variable: the second
// one's bind races the first one's runs. The value is the same address every
// time, which makes the race benign in practice and unprovable in principle —
// and an atomic pointer costs nothing to read.
std::atomic<DeviceFaultMonitor *> g_bound_monitor{nullptr};

}  // namespace

extern "C" __attribute__((visibility("default"))) void simpler_bind_device_fault_monitor(void *monitor) {
    // Idempotent by construction: a process has one monitor, so a repeat bind
    // stores the value already there. Release-ordered so a reader that sees
    // the pointer sees a constructed monitor behind it.
    g_bound_monitor.store(static_cast<DeviceFaultMonitor *>(monitor), std::memory_order_release);
}

DeviceFaultMonitor *device_fault_monitor() { return g_bound_monitor.load(std::memory_order_acquire); }
