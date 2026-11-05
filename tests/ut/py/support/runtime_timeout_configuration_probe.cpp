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
#include <acl/acl.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include "device_runner.h"
#include "common/host_log_state.h"
#include "common/log_level.h"

namespace {
int setter_calls;
int reset_calls;
int stream_calls;
int setter_error;
uint64_t actual_timeout;
SimplerHostLogState log_state{};
}  // namespace

extern "C" rtError_t rtSetDevice(int32_t) { return 0; }
extern "C" rtError_t rtDeviceReset(int32_t) {
    ++reset_calls;
    return 0;
}
extern "C" aclError aclrtSetOpExecuteTimeOutV2(uint64_t requested, uint64_t *actual) {
    ++setter_calls;
    if (setter_error != 0) return setter_error;
    *actual = actual_timeout != 0 ? actual_timeout : requested;
    return 0;
}
extern "C" rtError_t rtStreamCreate(rtStream_t *, int32_t) {
    ++stream_calls;
    return -776;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    log_state.threshold = static_cast<int>(simpler::log::LogLevel::WARN);
    assert(simpler_host_log_bind_state(&log_state) == 0);
    auto get_error =
        reinterpret_cast<const char *(*)(DeviceContextHandle)>(dlsym(RTLD_DEFAULT, "simpler_get_init_error_ctx"));
    assert(get_error != nullptr);
    assert(get_error(nullptr) == nullptr);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 1);
    setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", "2000", 1);
    setenv("SIMPLER_OP_EXECUTE_TIMEOUT_US", "3000000", 1);
    const uint8_t binary[] = {1};
    for (int arm = 0; arm != 3; ++arm) {
        DeviceRunner runner;
        setter_calls = reset_calls = stream_calls = 0;
        setter_error = arm == 1 ? -4321 : 0;
        actual_timeout = 4294967;
        setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", arm == 2 ? "6000" : "4000", 1);
        const int expected = arm == 1 ? -4321 : -776;
        auto initialize = [&]() {
            return simpler_init(
                &runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0
            );
        };
        const int init_rc = initialize();
        std::printf(
            "timeout arm=%d init_rc=%d expected=%d invalid_state=%d\n", arm, init_rc, expected,
            PTO_RUNTIME_ERR_INVALID_STATE
        );
        assert(init_rc == expected);
        assert(runner.device_id() == 0);
        assert(!runner.program_initialized());
        assert(setter_calls == 1);
        assert(stream_calls == (arm == 1 ? 0 : 1));
        if (arm == 1) {
            const char *detail = get_error(&runner);
            assert(detail != nullptr);
            assert(std::strstr(detail, "aclrtSetOpExecuteTimeOutV2") != nullptr);
            assert(std::strstr(detail, "-4321") != nullptr);
        } else {
            assert(get_error(&runner) == nullptr);
        }
        if (arm == 1) {
            for (int attempt = 0; attempt != 2; ++attempt) {
                assert(runner.attach_current_thread(0) == expected);
                assert(runner.ensure_device_initialized() == expected);
            }
            assert(setter_calls == 1);
            assert(stream_calls == 0);
        } else {
            assert(runner.attach_current_thread(0) == 0);
            assert(setter_calls == 1);
        }
        assert(initialize() == PTO_RUNTIME_ERR_INVALID_STATE);
        assert(setter_calls == 1);
        assert(finalize_device(&runner) == 0);
        assert(runner.device_id() == -1);
        assert(reset_calls == 1);
        assert(setter_calls == 1);
        setter_error = 0;
        setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "6000", 1);
        const int retry_rc = initialize();
        assert(retry_rc == -776);
        assert(get_error(&runner) == nullptr);
        assert(setter_calls == 2);
        assert(finalize_device(&runner) == 0);
        assert(reset_calls == 2);
        assert(setter_calls == 2);
        assert(finalize_device(&runner) == 0);
        assert(reset_calls == 2);
    }
    std::puts("PASS: actual timeout warning, original setter and stream errors, owned close and reinitialization");
}
