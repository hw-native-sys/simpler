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
 * Nanobind bindings for the distributed runtime (DistWorker and helpers).
 *
 * Compiled into the same _task_interface extension module as task_interface.cpp.
 * Call bind_dist_worker(m) from the NB_MODULE definition in task_interface.cpp.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "dist_chip_process.h"
#include "dist_chip_bootstrap_channel.h"
#include "dist_orchestrator.h"
#include "dist_sub_worker.h"
#include "dist_types.h"
#include "dist_worker.h"
#include "chip_worker.h"

namespace nb = nanobind;

inline void bind_dist_worker(nb::module_ &m) {
    // --- WorkerType ---
    nb::enum_<WorkerType>(m, "WorkerType")
        .value("CHIP", WorkerType::CHIP)
        .value("SUB", WorkerType::SUB)
        .value("DIST", WorkerType::DIST);

    // --- TaskState ---
    nb::enum_<TaskState>(m, "TaskState")
        .value("FREE", TaskState::FREE)
        .value("PENDING", TaskState::PENDING)
        .value("READY", TaskState::READY)
        .value("RUNNING", TaskState::RUNNING)
        .value("COMPLETED", TaskState::COMPLETED)
        .value("CONSUMED", TaskState::CONSUMED);

    nb::class_<DistTensorKey>(m, "DistTensorKey")
        .def(nb::init<>())
        .def(
            "__init__",
            [](DistTensorKey *self, uint64_t base_ptr, int32_t worker_index) {
                new (self) DistTensorKey{worker_index, base_ptr};
            },
            nb::arg("base_ptr"),
            nb::arg("worker_index") = -1
        )
        .def_rw("worker_index", &DistTensorKey::worker_index)
        .def_rw("base_ptr", &DistTensorKey::base_ptr);

    // --- WorkerPayload ---
    nb::class_<WorkerPayload>(m, "WorkerPayload")
        .def(nb::init<>())
        .def_rw("task_slot", &WorkerPayload::task_slot)
        .def_rw("worker_type", &WorkerPayload::worker_type)
        .def_prop_rw(
            "callable",
            [](const WorkerPayload &p) {
                return reinterpret_cast<uint64_t>(p.callable);
            },
            [](WorkerPayload &p, uint64_t v) {
                p.callable = reinterpret_cast<const void *>(v);
            },
            "Callable buffer pointer as uint64_t address."
        )
        .def_prop_rw(
            "args",
            [](const WorkerPayload &p) {
                return reinterpret_cast<uint64_t>(p.args);
            },
            [](WorkerPayload &p, uint64_t v) {
                p.args = reinterpret_cast<const void *>(v);
            },
            "Args pointer as uint64_t address."
        )
        .def_rw("block_dim", &WorkerPayload::block_dim)
        .def_rw("aicpu_thread_num", &WorkerPayload::aicpu_thread_num)
        .def_rw("enable_profiling", &WorkerPayload::enable_profiling)
        .def_rw("callable_id", &WorkerPayload::callable_id);

    // --- DistInputSpec ---
    nb::class_<DistInputSpec>(m, "DistInputSpec")
        .def(nb::init<>())
        .def(
            "__init__",
            [](DistInputSpec *self, uint64_t base_ptr, int32_t worker_index) {
                new (self) DistInputSpec{DistTensorKey{worker_index, base_ptr}};
            },
            nb::arg("base_ptr"),
            nb::arg("worker_index") = -1
        )
        .def_rw("key", &DistInputSpec::key)
        .def_prop_rw(
            "base_ptr",
            [](const DistInputSpec &self) {
                return self.key.base_ptr;
            },
            [](DistInputSpec &self, uint64_t base_ptr) {
                self.key.base_ptr = base_ptr;
            }
        )
        .def_prop_rw(
            "worker_index",
            [](const DistInputSpec &self) {
                return self.key.worker_index;
            },
            [](DistInputSpec &self, int32_t worker_index) {
                self.key.worker_index = worker_index;
            }
        );

    nb::enum_<DistOutputOwnership>(m, "DistOutputOwnership")
        .value("ALLOCATED", DistOutputOwnership::ALLOCATED)
        .value("EXTERNAL", DistOutputOwnership::EXTERNAL);

    // --- DistOutputSpec ---
    nb::class_<DistOutputSpec>(m, "DistOutputSpec")
        .def(nb::init<>())
        .def(
            "__init__",
            [](DistOutputSpec *self, size_t size) {
                new (self) DistOutputSpec{DistOutputOwnership::ALLOCATED, size, DistTensorKey{}, nullptr};
            },
            nb::arg("size")
        )
        .def_static(
            "external",
            [](uint64_t ptr, size_t size, int32_t worker_index) {
                DistOutputSpec spec;
                spec.ownership = DistOutputOwnership::EXTERNAL;
                spec.size = size;
                spec.key = DistTensorKey{worker_index, ptr};
                spec.external_ptr = reinterpret_cast<void *>(ptr);
                return spec;
            },
            nb::arg("ptr"),
            nb::arg("size"),
            nb::arg("worker_index") = -1
        )
        .def_rw("ownership", &DistOutputSpec::ownership)
        .def_rw("size", &DistOutputSpec::size)
        .def_rw("key", &DistOutputSpec::key)
        .def_prop_rw(
            "base_ptr",
            [](const DistOutputSpec &self) {
                return self.key.base_ptr;
            },
            [](DistOutputSpec &self, uint64_t base_ptr) {
                self.key.base_ptr = base_ptr;
            }
        )
        .def_prop_rw(
            "worker_index",
            [](const DistOutputSpec &self) {
                return self.key.worker_index;
            },
            [](DistOutputSpec &self, int32_t worker_index) {
                self.key.worker_index = worker_index;
            }
        )
        .def_prop_rw(
            "ptr",
            [](const DistOutputSpec &self) {
                return reinterpret_cast<uint64_t>(self.external_ptr);
            },
            [](DistOutputSpec &self, uint64_t ptr) {
                self.external_ptr = reinterpret_cast<void *>(ptr);
            }
        );

    // --- DistSubmitOutput ---
    nb::class_<DistSubmitOutput>(m, "DistSubmitOutput")
        .def_prop_ro(
            "ptr",
            [](const DistSubmitOutput &o) {
                return reinterpret_cast<uint64_t>(o.ptr);
            }
        )
        .def_prop_ro("size", [](const DistSubmitOutput &o) {
            return o.size;
        });

    // --- DistSubmitResult ---
    nb::class_<DistSubmitResult>(m, "DistSubmitResult")
        .def_prop_ro(
            "task_slot",
            [](const DistSubmitResult &r) {
                return r.task_slot;
            }
        )
        .def_prop_ro("outputs", [](const DistSubmitResult &r) {
            return r.outputs;
        });

    // --- DistSubWorker ---
    // The fork + Python callable loop are managed from Python (HostWorker.__init__).
    // This class only handles dispatch/poll via the shared-memory mailbox.
    nb::class_<DistSubWorker>(m, "DistSubWorker")
        .def(
            "__init__",
            [](DistSubWorker *self, uint64_t mailbox_ptr) {
                new (self) DistSubWorker(reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"), "Wrap a shared-memory mailbox pointer (uint64_t address)."
        )
        .def("shutdown", &DistSubWorker::shutdown);

    // Python can use this constant to allocate mailboxes of the right size.
    m.attr("DIST_SUB_MAILBOX_SIZE") = static_cast<int>(DIST_SUB_MAILBOX_SIZE);

    nb::enum_<ChipBootstrapMailboxState>(m, "ChipBootstrapMailboxState")
        .value("IDLE", ChipBootstrapMailboxState::IDLE)
        .value("SUCCESS", ChipBootstrapMailboxState::SUCCESS)
        .value("ERROR", ChipBootstrapMailboxState::ERROR);

    nb::class_<DistChipBootstrapChannel>(m, "DistChipBootstrapChannel")
        .def(
            "__init__",
            [](DistChipBootstrapChannel *self, uint64_t mailbox_ptr, size_t max_buffer_count) {
                new (self) DistChipBootstrapChannel(reinterpret_cast<void *>(mailbox_ptr), max_buffer_count);
            },
            nb::arg("mailbox_ptr"), nb::arg("max_buffer_count"),
            "Wrap a chip-bootstrap mailbox pointer. max_buffer_count must match the bootstrap buffer list length."
        )
        .def("reset", &DistChipBootstrapChannel::reset)
        .def(
            "write_success", &DistChipBootstrapChannel::write_success, nb::arg("device_ctx"), nb::arg("local_window_base"),
            nb::arg("actual_window_size"), nb::arg("buffer_ptrs")
        )
        .def("write_error", &DistChipBootstrapChannel::write_error, nb::arg("error_code"), nb::arg("message"))
        .def_prop_ro("state", &DistChipBootstrapChannel::state)
        .def_prop_ro("error_code", &DistChipBootstrapChannel::error_code)
        .def_prop_ro("device_ctx", &DistChipBootstrapChannel::device_ctx)
        .def_prop_ro("local_window_base", &DistChipBootstrapChannel::local_window_base)
        .def_prop_ro("actual_window_size", &DistChipBootstrapChannel::actual_window_size)
        .def_prop_ro("buffer_ptrs", &DistChipBootstrapChannel::buffer_ptrs)
        .def_prop_ro("error_message", &DistChipBootstrapChannel::error_message);

    m.attr("DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE") = static_cast<int>(DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE);

    // --- DistChipProcess ---
    // Fork + host_runtime.so init are managed from Python (Worker.__init__).
    // This class handles dispatch/poll via the chip mailbox (4096 bytes).
    nb::class_<DistChipProcess>(m, "DistChipProcess")
        .def(
            "__init__",
            [](DistChipProcess *self, uint64_t mailbox_ptr, size_t args_size) {
                new (self) DistChipProcess(reinterpret_cast<void *>(mailbox_ptr), args_size);
            },
            nb::arg("mailbox_ptr"), nb::arg("args_size"),
            "Wrap a chip mailbox pointer. args_size = sizeof(ChipStorageTaskArgs)."
        )
        .def("shutdown", &DistChipProcess::shutdown);

    m.attr("DIST_CHIP_MAILBOX_SIZE") = static_cast<int>(DIST_CHIP_MAILBOX_SIZE);

    // --- DistWorker ---
    nb::class_<DistWorker>(m, "DistWorker")
        .def(
            nb::init<int32_t>(), nb::arg("level"), "Create a DistWorker for the given hierarchy level (3=L3, 4=L4, …)."
        )

        .def(
            "add_chip_worker",
            [](DistWorker &self, DistWorker &w) {
                self.add_worker(WorkerType::CHIP, &w);
            },
            nb::arg("worker"), "Add a lower-level DistWorker as a CHIP sub-worker (for L4+)."
        )

        .def(
            "add_chip_worker_native",
            [](DistWorker &self, ChipWorker &w) {
                self.add_worker(WorkerType::CHIP, &w);
            },
            nb::arg("worker"), "Add a ChipWorker (_ChipWorker) as a CHIP sub-worker (for L3)."
        )

        .def(
            "add_chip_process",
            [](DistWorker &self, DistChipProcess &w) {
                self.add_worker(WorkerType::CHIP, &w);
            },
            nb::arg("worker"), "Add a forked ChipProcess as a CHIP sub-worker (process-isolated)."
        )

        .def(
            "add_sub_worker",
            [](DistWorker &self, DistSubWorker &w) {
                self.add_worker(WorkerType::SUB, &w);
            },
            nb::arg("worker"), "Add a SubWorker (fork/shm) as a SUB sub-worker."
        )

        .def("init", &DistWorker::init, "Start the Scheduler thread.")
        .def("close", &DistWorker::close, "Stop the Scheduler thread.")

        .def(
            "drain", &DistWorker::drain, nb::call_guard<nb::gil_scoped_release>(),
            "Block until all submitted tasks are consumed (releases GIL)."
        )

        .def("scope_begin", &DistWorker::scope_begin)
        .def("scope_end", &DistWorker::scope_end)

        .def(
            "submit",
            [](DistWorker &self, WorkerType worker_type, const WorkerPayload &base_payload,
               const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs) {
                return self.submit(worker_type, base_payload, inputs, outputs);
            },
            nb::arg("worker_type"), nb::arg("payload"), nb::arg("inputs") = std::vector<DistInputSpec>{},
            nb::arg("outputs") = std::vector<DistOutputSpec>{}
        )

        .def(
            "submit_group",
            [](DistWorker &self, WorkerType worker_type, const WorkerPayload &base_payload,
               const std::vector<uint64_t> &args_addrs, const std::vector<DistInputSpec> &inputs,
               const std::vector<DistOutputSpec> &outputs) {
                std::vector<const void *> args_list;
                args_list.reserve(args_addrs.size());
                for (uint64_t addr : args_addrs)
                    args_list.push_back(reinterpret_cast<const void *>(addr));
                return self.submit_group(worker_type, base_payload, args_list, inputs, outputs);
            },
            nb::arg("worker_type"), nb::arg("payload"), nb::arg("args_list"),
            nb::arg("inputs") = std::vector<DistInputSpec>{}, nb::arg("outputs") = std::vector<DistOutputSpec>{},
            "Submit a group task: N args -> N workers, 1 DAG node."
        )

        .def_prop_ro("level", &DistWorker::level)
        .def_prop_ro("idle", &DistWorker::idle);
}
