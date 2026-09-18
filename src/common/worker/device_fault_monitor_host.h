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

#include "host/device_fault_monitor.h"

/**
 * The process's one device-fault monitor.
 *
 * Lives in the module that also compiles `chip_worker.cpp` — the extension the
 * interpreter loads once and never unloads — because everything the driver
 * retains has to outlive a runtime `host_runtime.so`:
 *
 *   - **The instance**, so the refcount is one number for the whole process
 *     rather than one per loaded runtime SO fighting over a single driver slot.
 *   - **The trampoline the driver holds a pointer to**, and the code it calls
 *     to decode a notification. A runtime SO is opened `RTLD_LOCAL` and
 *     `dlclose`d by `ChipWorker::finalize`, so a trampoline living there would
 *     be unmapped under a notification still in flight.
 *
 * A loaded runtime SO reaches this instance because `ChipWorker::init` binds
 * its address in, the same way it binds the host-log state. A runtime SO that
 * nobody bound sees no monitor and installs nothing.
 */
DeviceFaultMonitor &resident_device_fault_monitor();

/**
 * Hand `resident_device_fault_monitor()` to a freshly loaded module.
 *
 * Returns 0 on success, or non-zero with `*error` set when the module does not
 * export the setter. Every host runtime built from this source tree does.
 */
int bind_loaded_device_fault_monitor(void *dl_handle, const char **error);
