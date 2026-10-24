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
#include <gtest/gtest.h>

#include <string>
#include <stdexcept>
#include <vector>

#include "host/kernel_capture_binder.h"

namespace {
class CaptureBinder : public ::testing::Test {
protected:
    void SetUp() override {
        handles = {&tokens[0], &tokens[1], &tokens[2], &tokens[3], &tokens[4], &tokens[5], &tokens[6], &tokens[7]};
        ops.context = this;
        ops.record = [](void *p, void *event, void *stream) {
            auto &self = *static_cast<CaptureBinder *>(p);
            return self.step("record:" + self.name(event) + ":" + self.name(stream));
        };
        ops.wait = [](void *p, void *stream, void *event) {
            auto &self = *static_cast<CaptureBinder *>(p);
            return self.step("wait:" + self.name(stream) + ":" + self.name(event));
        };
        ops.admit = [](void *p, void *stream) {
            auto &self = *static_cast<CaptureBinder *>(p);
            return self.step("admit:" + self.name(stream));
        };
        ops.launch_core = [](void *p, void *stream) {
            auto &self = *static_cast<CaptureBinder *>(p);
            return self.step("core:" + self.name(stream));
        };
        ops.launch_cpu = [](void *p, void *stream) {
            auto &self = *static_cast<CaptureBinder *>(p);
            return self.step("cpu:" + self.name(stream));
        };
    }
    std::string name(void *p) {
        const char *names[] = {"caller", "aicpu", "aicore", "start", "fork", "core_done", "cpu_done", "tail"};
        for (int i = 0; i < 8; ++i)
            if (p == &tokens[i]) return names[i];
        return "unknown";
    }
    int step(std::string event) {
        trace.push_back(event);
        if (throw_at != 0 && static_cast<int>(trace.size()) == throw_at) throw std::runtime_error("capture callback");
        return fail_at != 0 && static_cast<int>(trace.size()) == fail_at ? -77 : 0;
    }
    int tokens[8]{};
    int fail_at{0};
    int throw_at{0};
    KernelCaptureHandles handles{};
    KernelCaptureOps ops{};
    std::vector<std::string> trace;
};

TEST_F(CaptureBinder, AdmissionPrecedesCoreAndCompletionCoversBothProcessors) {
    LaunchProgressSink sink;
    const auto result = record_kernel_capture(handles, ops, sink);
    ASSERT_EQ(result.rc, 0);
    EXPECT_EQ(result.progress, LaunchProgress::Complete);
    EXPECT_TRUE(sink.submitted());
    EXPECT_EQ(
        trace, (std::vector<std::string>{
                   "record:start:caller", "wait:aicpu:start", "admit:aicpu", "record:fork:aicpu", "wait:aicore:fork",
                   "core:aicore", "record:core_done:aicore", "cpu:aicpu", "wait:aicpu:core_done",
                   "record:cpu_done:aicpu", "wait:caller:cpu_done", "record:tail:caller"
               })
    );
}

TEST_F(CaptureBinder, EveryAttemptedSubmissionFailureRetainsPartialOwnership) {
    for (fail_at = 1; fail_at <= 12; ++fail_at) {
        trace.clear();
        LaunchProgressSink sink;
        const auto result = record_kernel_capture(handles, ops, sink);
        EXPECT_EQ(result.rc, -77) << fail_at;
        EXPECT_EQ(result.progress, LaunchProgress::Partial) << fail_at;
        EXPECT_TRUE(sink.submitted());
        EXPECT_EQ(trace.size(), static_cast<size_t>(fail_at));
    }
}

TEST_F(CaptureBinder, InvalidTopologyAndIncompleteOpsAreCleanRefusals) {
    handles.aicore = handles.caller;
    LaunchProgressSink sink;
    auto result = record_kernel_capture(handles, ops, sink);
    EXPECT_EQ(result.progress, LaunchProgress::NotStarted);
    EXPECT_FALSE(sink.submitted());
    EXPECT_TRUE(trace.empty());
    handles.aicore = &tokens[2];
    ops.admit = nullptr;
    result = record_kernel_capture(handles, ops, sink);
    EXPECT_EQ(result.progress, LaunchProgress::NotStarted);
    EXPECT_FALSE(sink.submitted());
    EXPECT_TRUE(trace.empty());
}

TEST_F(CaptureBinder, EveryCallbackExceptionBecomesAnErrorWithPartialOwnership) {
    for (throw_at = 1; throw_at <= 12; ++throw_at) {
        trace.clear();
        LaunchProgressSink sink;
        KernelCaptureRecordResult result;
        bool escaped = false;
        try {
            result = record_kernel_capture(handles, ops, sink);
        } catch (...) {
            escaped = true;
        }
        EXPECT_FALSE(escaped) << throw_at;
        EXPECT_EQ(result.rc, PTO_RUNTIME_ERR_INTERNAL) << throw_at;
        EXPECT_EQ(result.progress, LaunchProgress::Partial) << throw_at;
        EXPECT_TRUE(sink.submitted()) << throw_at;
        EXPECT_EQ(trace.size(), static_cast<size_t>(throw_at));
    }
}
}  // namespace
