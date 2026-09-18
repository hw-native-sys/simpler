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
 * A stand-in for one loaded host runtime.
 *
 * Built twice, under two names, so a test can hold two distinct `RTLD_LOCAL`
 * images at once — which is what a process running two runtimes has, and the
 * shape that an instance-per-runtime-SO owner gets wrong. It compiles the real
 * binding translation unit, so what the test exercises is the production
 * pointer-holding code rather than a re-implementation of it.
 */

#include "host/device_fault_monitor.h"

extern "C" {

__attribute__((visibility("default"))) int client_has_monitor() { return device_fault_monitor() != nullptr ? 1 : 0; }

__attribute__((visibility("default"))) int client_acquire() {
    DeviceFaultMonitor *monitor = device_fault_monitor();
    if (monitor == nullptr) return -1;
    return monitor->acquire();
}

__attribute__((visibility("default"))) void client_release() {
    DeviceFaultMonitor *monitor = device_fault_monitor();
    if (monitor != nullptr) monitor->release();
}

__attribute__((visibility("default"))) int client_installed() {
    DeviceFaultMonitor *monitor = device_fault_monitor();
    return (monitor != nullptr && monitor->installed()) ? 1 : 0;
}

__attribute__((visibility("default"))) unsigned client_references() {
    DeviceFaultMonitor *monitor = device_fault_monitor();
    return monitor != nullptr ? monitor->references() : 0u;
}

__attribute__((visibility("default"))) void client_report(unsigned device_id, unsigned stream_id, unsigned error_code) {
    DeviceFaultMonitor *monitor = device_fault_monitor();
    if (monitor == nullptr) return;
    DeviceFaultNotice notice;
    notice.device_id = device_id;
    notice.stream_id = stream_id;
    notice.error_code = error_code;
    monitor->report(notice);
}

}  // extern "C"
