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
 * Nanobind bindings for the distributed runtime (DistWorker, DistOrchestrator).
 *
 * Compiled into the same _task_interface extension module as task_interface.cpp.
 * Call bind_dist_worker(m) from the NB_MODULE definition in task_interface.cpp.
 *
 * PR-D-2: `DistChipProcess` and `DistSubWorker` bindings are removed; their
 * PROCESS-mode dispatch logic now lives inside `WorkerThread`. Python callers
 * register PROCESS-mode workers via `add_next_level_process(mailbox_ptr)` /
 * `add_sub_process(mailbox_ptr)` instead of wrapping an IWorker subclass.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "chip_worker.h"
#include "dist_ring.h"
#include "dist_orchestrator.h"
#include "dist_types.h"
#include "dist_worker.h"
#include "dist_worker_manager.h"

namespace nb = nanobind;

inline void bind_dist_worker(nb::module_ &m) {
    // --- WorkerType ---
    nb::enum_<WorkerType>(m, "WorkerType").value("NEXT_LEVEL", WorkerType::NEXT_LEVEL).value("SUB", WorkerType::SUB);

    // --- TaskState ---
    nb::enum_<TaskState>(m, "TaskState")
        .value("FREE", TaskState::FREE)
        .value("PENDING", TaskState::PENDING)
        .value("READY", TaskState::READY)
        .value("RUNNING", TaskState::RUNNING)
        .value("COMPLETED", TaskState::COMPLETED)
        .value("CONSUMED", TaskState::CONSUMED);

    // --- DistSubmitResult ---
    nb::class_<DistSubmitResult>(m, "DistSubmitResult").def_prop_ro("task_slot", [](const DistSubmitResult &r) {
        return r.task_slot;
    });

    // --- DistOrchestrator (DAG builder, exposed via DistWorker.get_orchestrator()) ---
    nb::class_<DistOrchestrator>(m, "DistOrchestrator")
        .def(
            "submit_next_level",
            [](DistOrchestrator &self, uint64_t callable, const TaskArgs &args, const ChipCallConfig &config) {
                return self.submit_next_level(callable, args, config);
            },
            nb::arg("callable"), nb::arg("args"), nb::arg("config"),
            "Submit a NEXT_LEVEL (chip) task. Tags inside `args` drive dependency inference."
        )
        .def(
            "submit_next_level_group",
            [](DistOrchestrator &self, uint64_t callable, const std::vector<TaskArgs> &args_list,
               const ChipCallConfig &config) {
                return self.submit_next_level_group(callable, args_list, config);
            },
            nb::arg("callable"), nb::arg("args_list"), nb::arg("config"),
            "Submit a group of NEXT_LEVEL tasks: N args -> N workers, 1 DAG node."
        )
        .def(
            "submit_sub",
            [](DistOrchestrator &self, int32_t callable_id, const TaskArgs &args) {
                return self.submit_sub(callable_id, args);
            },
            nb::arg("callable_id"), nb::arg("args"),
            "Submit a SUB task by registered callable id. Tags drive dependency inference."
        )
        .def(
            "submit_sub_group",
            [](DistOrchestrator &self, int32_t callable_id, const std::vector<TaskArgs> &args_list) {
                return self.submit_sub_group(callable_id, args_list);
            },
            nb::arg("callable_id"), nb::arg("args_list"),
            "Submit a group of SUB tasks: N args -> N workers, 1 DAG node."
        )
        .def(
            "alloc",
            [](DistOrchestrator &self, const std::vector<uint32_t> &shape, DataType dtype) {
                return self.alloc(shape, dtype);
            },
            nb::arg("shape"), nb::arg("dtype"),
            "Allocate an intermediate ContinuousTensor from the orchestrator's MAP_SHARED "
            "pool (visible to forked child workers). Lifetime: until the next Worker.run() call."
        )
        .def(
            "scope_begin", &DistOrchestrator::scope_begin,
            "Open a nested scope. Max nesting depth = DIST_MAX_SCOPE_DEPTH (64)."
        )
        .def("scope_end", &DistOrchestrator::scope_end, "Close the innermost scope. Non-blocking.")
        .def("_scope_begin", &DistOrchestrator::scope_begin)
        .def("_scope_end", &DistOrchestrator::scope_end)
        .def(
            "_drain", &DistOrchestrator::drain, nb::call_guard<nb::gil_scoped_release>(),
            "Block until all submitted tasks are CONSUMED (releases GIL)."
        );

    // --- DistWorker ---
    nb::class_<DistWorker>(m, "DistWorker")
        .def(
            nb::init<int32_t, uint64_t>(), nb::arg("level"), nb::arg("heap_ring_size") = DIST_DEFAULT_HEAP_RING_SIZE,
            "Create a DistWorker for the given hierarchy level (3=L3, 4=L4, …). "
            "`heap_ring_size` selects the per-ring MAP_SHARED heap mmap'd in the ctor "
            "(default 1 GiB; total VA = 4 × heap_ring_size)."
        )

        // THREAD-mode registration (parent calls worker->run directly).
        .def(
            "add_next_level_worker",
            [](DistWorker &self, DistWorker &w) {
                self.add_worker(WorkerType::NEXT_LEVEL, &w);
            },
            nb::arg("worker"), "Add a lower-level DistWorker as a NEXT_LEVEL sub-worker (THREAD mode)."
        )
        .def(
            "add_next_level_worker",
            [](DistWorker &self, ChipWorker &w) {
                self.add_worker(WorkerType::NEXT_LEVEL, &w);
            },
            nb::arg("worker"), "Add a ChipWorker as a NEXT_LEVEL sub-worker (THREAD mode)."
        )

        // PROCESS-mode registration (parent writes unified mailbox; child runs
        // the real IWorker in its own address space).
        .def(
            "add_next_level_process",
            [](DistWorker &self, uint64_t mailbox_ptr) {
                self.add_process_worker(WorkerType::NEXT_LEVEL, reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"),
            "Add a PROCESS-mode NEXT_LEVEL worker. `mailbox_ptr` is the address of a "
            "DIST_MAILBOX_SIZE-byte MAP_SHARED region. The child process loop is "
            "Python-managed (fork + _chip_process_loop)."
        )
        .def(
            "add_sub_process",
            [](DistWorker &self, uint64_t mailbox_ptr) {
                self.add_process_worker(WorkerType::SUB, reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"),
            "Add a PROCESS-mode SUB worker. `mailbox_ptr` is the address of a "
            "DIST_MAILBOX_SIZE-byte MAP_SHARED region. The child process loop is "
            "Python-managed (fork + _sub_worker_loop)."
        )

        .def("init", &DistWorker::init, "Start the Scheduler thread.")
        .def("close", &DistWorker::close, "Stop the Scheduler thread.")

        .def(
            "get_orchestrator", &DistWorker::get_orchestrator, nb::rv_policy::reference_internal,
            "Return the Orchestrator handle (lifetime tied to this DistWorker)."
        );

    m.attr("DIST_DEFAULT_HEAP_RING_SIZE") = static_cast<uint64_t>(DIST_DEFAULT_HEAP_RING_SIZE);
    m.attr("DIST_MAILBOX_SIZE") = static_cast<int>(DIST_MAILBOX_SIZE);
    m.attr("DIST_MAX_RING_DEPTH") = static_cast<int32_t>(DIST_MAX_RING_DEPTH);
    m.attr("DIST_MAX_SCOPE_DEPTH") = static_cast<int32_t>(DIST_MAX_SCOPE_DEPTH);
}
