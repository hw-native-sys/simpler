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
#include "native_submission_state.h"

inline ChipRun::Deadline native_submission_wait_deadline(double timeout) {
    if (std::isnan(timeout)) throw std::invalid_argument("submission wait timeout must not be NaN");
    auto deadline = ChipRun::Deadline::max();
    const auto available = deadline - ChipRun::Clock::now();
    if (timeout >= 0 && timeout < std::chrono::duration<double>(available).count())
        deadline = ChipRun::Clock::now() +
                   std::chrono::duration_cast<ChipRun::Clock::duration>(std::chrono::duration<double>(timeout));
    return deadline;
}

inline void bind_native_submission(nanobind::module_ &m) {
    nanobind::class_<NativeSubmissionState>(m, "_NativeSubmission")
        .def_prop_ro(
            "_owner_recovery_required", &NativeSubmissionState::owner_recovery_required,
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def("done", &NativeSubmissionState::done, nanobind::call_guard<nanobind::gil_scoped_release>())
        .def(
            "_claim_framework", &NativeSubmissionState::claim_framework,
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def_prop_ro(
            "_framework_claimed", &NativeSubmissionState::framework_claimed,
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def_prop_ro(
            "_framework_transferred", &NativeSubmissionState::framework_transferred,
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def(
            "_retire_framework",
            [](NativeSubmissionState &self, bool wait_for_own_completion, double timeout) {
                const auto deadline = native_submission_wait_deadline(timeout);
                return self.retire_framework(wait_for_own_completion, deadline);
            },
            nanobind::arg("wait_for_own_completion") = false, nanobind::arg("timeout") = -1.0,
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def("_retire", &NativeSubmissionState::retire, nanobind::call_guard<nanobind::gil_scoped_release>())
        .def(
            "_retain_model_resources",
            [](NativeSubmissionState &self, const NativeSubmissionState *anchor) {
                int started = 0;
                self.retain_model_resources(started, anchor);
            },
            nanobind::arg("anchor").none() = nullptr, nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def(
            "_model_destroyed", &NativeSubmissionState::model_destroyed,
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def(
            "_release_model_resources",
            [](NativeSubmissionState &self) {
                NativeSubmissionCleanup outcome;
                self.cancel_pending(outcome);
                if (!outcome.carrier_retired) throw RunPreparationRefused("native capture carrier is still live");
                self.release_resources();
                return outcome.disposition == NativeSubmissionDisposition::Cancelled;
            },
            nanobind::call_guard<nanobind::gil_scoped_release>()
        )
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
                return self.wait_until(native_submission_wait_deadline(timeout));
            },
            nanobind::arg("timeout") = -1.0, nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def(
            "_wait_framework",
            [](NativeSubmissionState &self, double timeout) {
                return self.wait_framework_until(native_submission_wait_deadline(timeout));
            },
            nanobind::arg("timeout") = -1.0, nanobind::call_guard<nanobind::gil_scoped_release>()
        )
        .def("capsule", [](NativeSubmissionState &self) {
            // The control block comes from make_shared, never nanobind's
            // Python-owner shared_ptr caster (whose deleter acquires GIL).
            auto holder = std::make_unique<NativeSubmissionCapsule>(self.shared_from_this());
            self.note_export();
            auto capsule =
                nanobind::capsule(&holder->api, SIMPLER_NATIVE_SUBMISSION_CAPSULE, [](void *pointer) noexcept {
                    auto *api = static_cast<SimplerNativeSubmission *>(pointer);
                    api->release(api->context);
                });
            holder.release();
            return capsule;
        });
}
