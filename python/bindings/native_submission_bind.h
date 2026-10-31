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

#include <atomic>
#include <cmath>
#include <nanobind/nanobind.h>
#include <cstdio>
#include <memory>

#include "chip_run_lane.h"
#include "chip_worker.h"
#include "native_submission.h"
#include "queued_submission.h"

using NativeSubmissionState = QueuedSubmission<ChipRun>;

struct NativeSubmissionCapsule {
    explicit NativeSubmissionCapsule(std::shared_ptr<NativeSubmissionState> state) :
        state(std::move(state)) {
        api = {1, this, retain, release, submit};
    }
    static void retain(void *context) noexcept {
        static_cast<NativeSubmissionCapsule *>(context)->references.fetch_add(1, std::memory_order_relaxed);
    }
    static void release(void *context) noexcept {
        auto *self = static_cast<NativeSubmissionCapsule *>(context);
        if (self->references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete self;
    }
    static int submit(void *context, char *message, size_t capacity) noexcept {
        try {
            static_cast<NativeSubmissionCapsule *>(context)->state->submit();
            return 0;
        } catch (const std::exception &error) {
            if (message && capacity) std::snprintf(message, capacity, "%s", error.what());
        } catch (...) {
            if (message && capacity) std::snprintf(message, capacity, "unknown native submission failure");
        }
        return -1;
    }
    SimplerNativeSubmissionV1 api{};
    std::atomic<size_t> references{1};
    std::shared_ptr<NativeSubmissionState> state;
};

inline void bind_native_submission(nanobind::module_ &m) {
    nanobind::class_<NativeSubmissionState>(m, "_NativeSubmission")
        .def("done", &NativeSubmissionState::done, nanobind::call_guard<nanobind::gil_scoped_release>())
        .def("_retire", &NativeSubmissionState::retire, nanobind::call_guard<nanobind::gil_scoped_release>())
        .def(
            "cancel",
            [](NativeSubmissionState &self) {
                return self.cancel(
                    std::make_exception_ptr(RunPreparationRefused("native submission cancelled before submit"))
                );
            },
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def(
            "wait",
            [](NativeSubmissionState &self, double timeout) {
                if (std::isnan(timeout)) throw std::invalid_argument("submission wait timeout must not be NaN");
                auto deadline = ChipRun::Deadline::max();
                const auto available = deadline - ChipRun::Clock::now();
                if (timeout >= 0 && timeout < std::chrono::duration<double>(available).count())
                    deadline =
                        ChipRun::Clock::now() +
                        std::chrono::duration_cast<ChipRun::Clock::duration>(std::chrono::duration<double>(timeout));
                return self.wait_until(deadline);
            },
            nanobind::arg("timeout") = -1.0, nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def("capsule", [](NativeSubmissionState &self) {
            // The control block comes from make_shared, never nanobind's
            // Python-owner shared_ptr caster (whose deleter acquires GIL).
            auto holder = std::make_unique<NativeSubmissionCapsule>(self.shared_from_this());
            auto capsule =
                nanobind::capsule(&holder->api, SIMPLER_NATIVE_SUBMISSION_CAPSULE, [](void *pointer) noexcept {
                    auto *api = static_cast<SimplerNativeSubmissionV1 *>(pointer);
                    api->release(api->context);
                });
            holder.release();
            return capsule;
        });
}
