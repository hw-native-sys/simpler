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

#include "host_log.h"

extern "C" __attribute__((visibility("default"))) void test_host_log_consumer_emit() {
    HostLogger::get_instance().log(simpler::log::LogLevel::ERROR, "consumer", "cross-dso-record");
}

extern "C" __attribute__((visibility("default"))) int test_host_log_consumer_start_writer() {
    return HostLogger::get_instance().start_writer() ? 1 : 0;
}

extern "C" __attribute__((visibility("default"))) int test_host_log_consumer_flush() {
    return HostLogger::get_instance().flush() ? 1 : 0;
}
