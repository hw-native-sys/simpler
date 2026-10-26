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
#include "callable.h"
#include "fixture.h"
#include "acl/acl.h"

constexpr int ACL_ERROR_RT_FEATURE_NOT_SUPPORT = 999;
extern "C" int rtEventWorkModeGet(uint8_t *mode) {
    *mode = 1;
    return 0;
}
extern "C" int rtEventWorkModeSet(uint8_t) { return 0; }
extern "C" int aclrtMemsetAsync(void *p, size_t n, int value, size_t bytes, void *) {
    return aclrtMemset(p, n, value, bytes);
}
// Compile the real close path with its private ownership records visible to the
// fixture. Only the argument allocator and device transport inject failures.
#include "hbg_kernel_launch.cpp"  // NOLINT(bugprone-suspicious-include)

static int registration_calls = 0;
extern "C" int __wrap_register_callable_impl(const ChipCallable *, const HostApi *, CallableArtifacts *) {
    ++registration_calls;
    return -4243;
}

namespace native_admission_test {
thread_local bool refuse_next_host_allocation = false;
namespace {
int fake_current_device(void *, int *device_id) noexcept {
    *device_id = 0;
    return 0;
}
int fake_current_context(void *, void **context) noexcept {
    static int token;
    *context = &token;
    return 0;
}
int fake_create_hidden_stream(void *, void **stream) noexcept {
    static int tokens[2];
    static int next;
    *stream = &tokens[next++ % 2];
    return 0;
}
int fake_destroy_hidden_stream(void *, void *) noexcept { return 0; }
int fake_create_event(void *, void **event) noexcept {
    static int tokens[8];
    static int next;
    *event = &tokens[next++ % 8];
    return 0;
}
int fake_destroy_event(void *, void *) noexcept { return 0; }
}  // namespace

TEST_F(NativeAdmissionTest, RegistrationRefusesAfterFailedKernelClose) {
    class ClosingRunner : public Runner {
    public:
        bool fail_release = true;
        void setup_binding() {
            hbg_launch_context_ = std::make_shared<hbg::KernelContextLaunchState>();
            auto state = std::make_shared<hbg::KernelCallableLaunchState>();
            PersistentArgsOps ops;
            ops.context = this;
            ops.alloc = [](void *, size_t n) -> void * {
                return std::calloc(1, n);
            };
            ops.copy_h2d = [](void *, void *d, size_t, const void *s, size_t n) -> int {
                std::memcpy(d, s, n);
                return 0;
            };
            ops.fill_arch_fields = [](void *, KernelArgs *, uint64_t) -> int {
                return 0;
            };
            ops.free_ = [](void *p, void *block) -> int {
                if (std::exchange(static_cast<ClosingRunner *>(p)->fail_release, false)) return -4242;
                std::free(block);
                return 0;
            };
            ASSERT_EQ(state->arguments.prepare_once(state->runtime, ops, 0), 0);
            callables_[1].hbg_launch_state = state;
        }
    } runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops;
    ops.context = &runner;
    ops.get_current_device = fake_current_device;
    ops.get_current_context = fake_current_context;
    ops.create_hidden_stream = fake_create_hidden_stream;
    ops.destroy_hidden_stream = fake_destroy_hidden_stream;
    ops.create_event = fake_create_event;
    ops.destroy_event = fake_destroy_event;
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    runner.setup_binding();
    uint64_t live_lease = 0;
    ASSERT_EQ(simpler_kernel_mode_retain_graph(&runner, &live_lease), 0);
    EXPECT_EQ(finalize_device(&runner), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.kernel_execution_state().accepts_dispatch());
    EXPECT_TRUE(runner.fail_release);
    ASSERT_EQ(simpler_kernel_mode_release_graph(&runner, live_lease), 0);
    ASSERT_EQ(finalize_device(&runner), -4242);
    ASSERT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
    uint64_t token = 99;
    EXPECT_EQ(simpler_kernel_mode_retain_graph(&runner, &token), PTO_RUNTIME_ERR_INVALID_STATE);
    const uint8_t binary[] = {1};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    int32_t id = 99;
    const int rc = simpler_register_callable_v2(&runner, image.data(), image.size(), &id);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Closing);
    EXPECT_FALSE(runner.kernel_execution_state().accepts_dispatch());
    EXPECT_EQ(id, -1);
    EXPECT_EQ(rc, PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(registration_calls, 0);
    EXPECT_EQ(finalize_device(&runner), 0);
}
}  // namespace native_admission_test
