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

#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "native_submission.h"

namespace py = pybind11;

static void mark(const char *message) {
    std::fprintf(stderr, "GIL_PROBE %s\n", message);
    std::fflush(stderr);
}

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool ready = false;
    bool released = false;
    bool complete = false;
    std::string error;
};

struct RetainedSubmission {
    explicit RetainedSubmission(const SimplerNativeSubmissionV1 &api) :
        reference(std::in_place, api) {}
    std::optional<NativeSubmissionRef> reference;
    void release_in_thread() {
        if (!reference) throw std::logic_error("reference already released");
        std::thread destroy([ref = std::move(*reference)] {});
        reference.reset();
        // The caller holds GIL throughout this join.
        destroy.join();
    }
};

PYBIND11_MODULE(_framework_queue_gil_probe, module) {
    py::class_<RetainedSubmission>(module, "RetainedSubmission")
        .def("release_in_thread", &RetainedSubmission::release_in_thread);
    module.def("retain_submission", [](py::capsule capsule) {
        auto *api = static_cast<SimplerNativeSubmissionV1 *>(
            PyCapsule_GetPointer(capsule.ptr(), SIMPLER_NATIVE_SUBMISSION_CAPSULE)
        );
        if (!api) throw py::error_already_set();
        return RetainedSubmission(*api);
    });
    module.def("warmup", [] {
        py::gil_scoped_release unlocked;
        at_npu::native::OpCommand::RunOpApi(
            "SimplerGILProbeWarmup",
            [] {
                return 0;
            },
            false
        );
        c10_npu::getCurrentNPUStream().synchronize();
    });
    py::class_<Gate, std::shared_ptr<Gate>>(module, "Gate");
    module.def("enqueue_submission", [](py::capsule capsule) {
        auto *api = static_cast<SimplerNativeSubmissionV1 *>(
            PyCapsule_GetPointer(capsule.ptr(), SIMPLER_NATIVE_SUBMISSION_CAPSULE)
        );
        if (!api) throw py::error_already_set();
        NativeSubmissionRef submission(*api);
        auto gate = std::make_shared<Gate>();
        auto invoke = [gate, submission]() -> int {
            {
                std::unique_lock<std::mutex> lock(gate->mutex);
                gate->ready = true;
                gate->changed.notify_all();
                gate->changed.wait(lock, [&] {
                    return gate->released;
                });
            }
            std::string error;
            try {
                submission.submit();
            } catch (const std::exception &failure) {
                error = failure.what();
            } catch (...) {
                error = "unknown native submission failure";
            }
            {
                std::lock_guard<std::mutex> lock(gate->mutex);
                gate->error = std::move(error);
                gate->complete = true;
            }
            return 0;
        };
        py::gil_scoped_release unlocked;
        at_npu::native::OpCommand::RunOpApi("SimplerGatedNativeSubmit", invoke, false);
        return gate;
    });
    module.def("enqueue", [](py::function callback, bool needs_python) {
        auto gate = std::make_shared<Gate>();
        // The Python caller retains callback until the queue drains. The queue
        // owns only native state and a borrowed pointer, never a Python owner.
        PyObject *function = callback.ptr();
        auto invoke = [gate, function, needs_python]() -> int {
            mark("CALLBACK_ENTER_GATE");
            {
                std::unique_lock<std::mutex> lock(gate->mutex);
                gate->ready = true;
                gate->changed.notify_all();
                gate->changed.wait(lock, [&] {
                    return gate->released;
                });
            }
            if (needs_python) {
                mark("CALLBACK_WAIT_GIL");
                py::gil_scoped_acquire lock;
                mark("CALLBACK_ACQUIRED_GIL");
                py::reinterpret_borrow<py::function>(function)();
            } else {
                mark("NATIVE_CALLBACK");
            }
            {
                std::lock_guard<std::mutex> lock(gate->mutex);
                gate->complete = true;
            }
            mark("CALLBACK_COMPLETE");
            return 0;
        };
        py::gil_scoped_release unlocked;
        mark("NATIVE_ENQUEUE_ENTER");
        at_npu::native::OpCommand::RunOpApi("SimplerGILProbe", invoke, false);
        mark("NATIVE_ENQUEUE_RETURN");
        return gate;
    });
    module.def("wait_ready", [](const std::shared_ptr<Gate> &gate) {
        py::gil_scoped_release unlocked;
        std::unique_lock<std::mutex> lock(gate->mutex);
        gate->changed.wait(lock, [&] {
            return gate->ready;
        });
    });
    module.def("drain", [](const std::shared_ptr<Gate> &gate, py::function drain) {
        mark("DRAIN_ENTER_GIL_HELD");
        {
            std::lock_guard<std::mutex> lock(gate->mutex);
            gate->released = true;
        }
        gate->changed.notify_all();
        drain();
        mark("DRAIN_RETURN");
        std::lock_guard<std::mutex> lock(gate->mutex);
        if (!gate->complete) throw std::runtime_error("drain returned before callback completion");
        if (!gate->error.empty()) throw std::runtime_error(gate->error);
    });
    module.def("query_released", [] {
        auto stream = c10_npu::getCurrentNPUStream();
        py::gil_scoped_release unlocked;
        return stream.query();
    });
    module.def("empty_cache_released", [] {
        py::gil_scoped_release unlocked;
        c10_npu::NPUCachingAllocator::emptyCache();
    });
}
