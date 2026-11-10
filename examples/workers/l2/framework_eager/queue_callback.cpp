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
// This module is a bounded reference consumer, not a Simpler submission API.
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "native_submission.h"

namespace py = pybind11;

struct QueuedCall {
    explicit QueuedCall(py::function function) :
        callback(function.release().ptr()) {}
    ~QueuedCall() {
        py::gil_scoped_acquire lock;
        Py_DECREF(callback);
    }
    PyObject *callback;
    std::atomic<bool> complete{false};
    uint64_t thread_id{0};
    std::string error;
    void check() const {
        if (!complete.load(std::memory_order_acquire)) throw std::runtime_error("framework callback is not complete");
        if (!error.empty()) throw std::runtime_error(error);
    }
};

struct QueuedNativeCall {
    explicit QueuedNativeCall(const SimplerNativeSubmission &api) :
        submission(api) {}
    NativeSubmissionRef submission;
    std::atomic<bool> complete{false};
    std::string error;
    void check() const {
        if (!complete.load(std::memory_order_acquire)) throw std::runtime_error("native callback is not complete");
        if (!error.empty()) throw std::runtime_error(error);
    }
};

// Holds the framework queue thread inside one queued operation until released.
// Valid only with the host queue enabled; disabled, the operation runs inline.
struct QueueGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool released{false};
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        changed.notify_all();
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<QueueGate, std::shared_ptr<QueueGate>>(module, "QueueGate")
        .def("release", &QueueGate::release, py::call_guard<py::gil_scoped_release>());
    module.def("enqueue_gate", [] {
        auto gate = std::make_shared<QueueGate>();
        auto invoke = [gate]() -> int {
            std::unique_lock<std::mutex> lock(gate->mutex);
            gate->changed.wait(lock, [&gate] {
                return gate->released;
            });
            return 0;
        };
        py::gil_scoped_release unlocked;
        at_npu::native::OpCommand::RunOpApi("SimplerQueueGate", invoke, false);
        return gate;
    });
    py::class_<QueuedNativeCall, std::shared_ptr<QueuedNativeCall>>(module, "QueuedNativeCall")
        .def("check", &QueuedNativeCall::check);
    module.def("enqueue_native", [](py::capsule capsule) {
        auto *api = static_cast<SimplerNativeSubmission *>(
            PyCapsule_GetPointer(capsule.ptr(), SIMPLER_NATIVE_SUBMISSION_CAPSULE)
        );
        if (!api) throw py::error_already_set();
        auto call = std::make_shared<QueuedNativeCall>(*api);
        // A non-zero return is the framework's operator-failure channel: with the
        // host queue enabled, torch_npu discards every later queued operation and
        // reports the error at its next synchronization, so no successor reads an
        // unwritten output. Discarded callbacks never submit; their tickets remain
        // pending until the producer cancels them.
        auto invoke = [call]() -> int {
            try {
                call->submission.submit();
            } catch (const std::exception &error) {
                call->error = error.what();
            } catch (...) {
                call->error = "unknown native submission exception";
            }
            call->complete.store(true, std::memory_order_release);
            return call->error.empty() ? 0 : -1;
        };
        py::gil_scoped_release unlocked;
        try {
            at_npu::native::OpCommand::RunOpApi("SimplerNativeSubmit", invoke, false);
        } catch (const std::exception &) {
            // With the host queue disabled the callback ran inline and torch_npu
            // raised a generic operator error; report the Simpler error instead.
            if (call->complete.load(std::memory_order_acquire) && !call->error.empty())
                throw std::runtime_error(call->error);
            throw;
        }
        return call;
    });
    module.def("submit_native", [](py::capsule capsule) {
        // Synchronous submission on the caller thread, outside the framework queue.
        auto *api = static_cast<SimplerNativeSubmission *>(
            PyCapsule_GetPointer(capsule.ptr(), SIMPLER_NATIVE_SUBMISSION_CAPSULE)
        );
        if (!api) throw py::error_already_set();
        NativeSubmissionRef submission(*api);
        py::gil_scoped_release unlocked;
        submission.submit();
    });
    py::class_<QueuedCall, std::shared_ptr<QueuedCall>>(module, "QueuedCall")
        .def("check", &QueuedCall::check)
        .def(
            "complete",
            [](const QueuedCall &call) {
                return call.complete.load(std::memory_order_acquire);
            }
        )
        .def_property_readonly("thread_id", [](const QueuedCall &call) {
            call.check();
            return call.thread_id;
        });
    module.def("enqueue", [](py::function function) {
        auto call = std::make_shared<QueuedCall>(std::move(function));
        // The queue owns the callable/frame until execution; callback errors
        // remain observable after the final framework completion boundary.
        auto invoke = [call]() -> int {
            py::gil_scoped_acquire lock;
            call->thread_id = static_cast<uint64_t>(syscall(SYS_gettid));
            try {
                py::reinterpret_borrow<py::function>(call->callback)();
            } catch (py::error_already_set &error) {
                call->error = error.what();
            } catch (const std::exception &error) {
                call->error = error.what();
            } catch (...) {
                call->error = "unknown framework callback exception";
            }
            call->complete.store(true, std::memory_order_release);
            return 0;
        };
        py::gil_scoped_release unlocked;
        at_npu::native::OpCommand::RunOpApi("SimplerReferenceSubmit", invoke, false);
        return call;
    });
    module.def("finish_current_stream", [] {
        // A queued Python callback needs the GIL. The reference's explicit
        // completion boundary releases it while the framework drains.
        auto stream = c10_npu::getCurrentNPUStream();
        py::gil_scoped_release unlocked;
        stream.synchronize();
    });
}
