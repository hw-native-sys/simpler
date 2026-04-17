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
 * Nanobind bindings for the distributed runtime (Worker, Orchestrator).
 *
 * Compiled into the same _task_interface extension module as task_interface.cpp.
 * Call bind_worker(m) from the NB_MODULE definition in task_interface.cpp.
 *
 * PR-D-2: `ChipProcess` and `SubWorker` bindings are removed; their
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
#include "ring.h"
#include "dist_chip_bootstrap_channel.h"
#include "orchestrator.h"
#include "types.h"
#include "worker.h"
#include "worker_manager.h"

namespace nb = nanobind;

inline int32_t load_i32_acquire(const volatile int32_t *ptr) {
    int32_t value;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(value) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    value = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &value, __ATOMIC_ACQUIRE);
#endif
    return value;
}

inline void store_i32_release(volatile int32_t *ptr, int32_t value) {
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(value), "r"(ptr) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *ptr = value;
#else
    __atomic_store(ptr, &value, __ATOMIC_RELEASE);
#endif
}

inline void bind_worker(nb::module_ &m) {
    m.def(
        "_mailbox_load_i32",
        [](uint64_t addr) -> int32_t {
            return load_i32_acquire(reinterpret_cast<const volatile int32_t *>(addr));
        },
        nb::arg("addr"), "Internal: acquire-load an int32 mailbox field."
    );
    m.def(
        "_mailbox_store_i32",
        [](uint64_t addr, int32_t value) {
            store_i32_release(reinterpret_cast<volatile int32_t *>(addr), value);
        },
        nb::arg("addr"), nb::arg("value"), "Internal: release-store an int32 mailbox field."
    );

    nb::enum_<WorkerType>(m, "WorkerType").value("NEXT_LEVEL", WorkerType::NEXT_LEVEL).value("SUB", WorkerType::SUB);

    nb::enum_<TaskState>(m, "TaskState")
        .value("FREE", TaskState::FREE)
        .value("PENDING", TaskState::PENDING)
        .value("READY", TaskState::READY)
        .value("RUNNING", TaskState::RUNNING)
        .value("COMPLETED", TaskState::COMPLETED)
        .value("CONSUMED", TaskState::CONSUMED);

    nb::class_<SubmitResult>(m, "SubmitResult").def_prop_ro("task_slot", [](const SubmitResult &r) {
        return r.task_slot;
    });

    m.attr("DIST_MAILBOX_SIZE") = static_cast<int>(MAILBOX_SIZE);
    m.attr("DIST_SUB_MAILBOX_SIZE") = static_cast<int>(MAILBOX_SIZE);
    m.attr("DIST_CHIP_MAILBOX_SIZE") = static_cast<int>(MAILBOX_SIZE);
    m.attr("DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE") = static_cast<int>(DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE);

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
            nb::arg("mailbox_ptr"), nb::arg("max_buffer_count")
        )
        .def("reset", &DistChipBootstrapChannel::reset)
        .def(
            "write_success", &DistChipBootstrapChannel::write_success, nb::arg("device_ctx"),
            nb::arg("local_window_base"), nb::arg("actual_window_size"), nb::arg("buffer_ptrs")
        )
        .def("write_error", &DistChipBootstrapChannel::write_error, nb::arg("error_code"), nb::arg("message"))
        .def_prop_ro("state", &DistChipBootstrapChannel::state)
        .def_prop_ro("error_code", &DistChipBootstrapChannel::error_code)
        .def_prop_ro("device_ctx", &DistChipBootstrapChannel::device_ctx)
        .def_prop_ro("local_window_base", &DistChipBootstrapChannel::local_window_base)
        .def_prop_ro("actual_window_size", &DistChipBootstrapChannel::actual_window_size)
        .def_prop_ro("buffer_ptrs", &DistChipBootstrapChannel::buffer_ptrs)
        .def_prop_ro("error_message", &DistChipBootstrapChannel::error_message);

    nb::class_<Orchestrator>(m, "_Orchestrator")
        .def(
            "submit_next_level",
            [](Orchestrator &self, uint64_t callable, const TaskArgs &args, const ChipCallConfig &config) {
                return self.submit_next_level(callable, args, config);
            },
            nb::arg("callable"), nb::arg("args"), nb::arg("config"),
            "Submit a NEXT_LEVEL (chip) task. Tags inside `args` drive dependency inference."
        )
        .def(
            "submit_next_level_group",
            [](Orchestrator &self, uint64_t callable, const std::vector<TaskArgs> &args_list,
               const ChipCallConfig &config) {
                return self.submit_next_level_group(callable, args_list, config);
            },
            nb::arg("callable"), nb::arg("args_list"), nb::arg("config"),
            "Submit a group of NEXT_LEVEL tasks: N args -> N workers, 1 DAG node."
        )
        .def(
            "submit_sub",
            [](Orchestrator &self, int32_t callable_id, const TaskArgs &args) {
                return self.submit_sub(callable_id, args);
            },
            nb::arg("callable_id"), nb::arg("args"),
            "Submit a SUB task by registered callable id. Tags drive dependency inference."
        )
        .def(
            "submit_sub_group",
            [](Orchestrator &self, int32_t callable_id, const std::vector<TaskArgs> &args_list) {
                return self.submit_sub_group(callable_id, args_list);
            },
            nb::arg("callable_id"), nb::arg("args_list"),
            "Submit a group of SUB tasks: N args -> N workers, 1 DAG node."
        )
        .def(
            "alloc",
            [](Orchestrator &self, const std::vector<uint32_t> &shape, DataType dtype) {
                return self.alloc(shape, dtype);
            },
            nb::arg("shape"), nb::arg("dtype"),
            "Allocate an intermediate ContinuousTensor from the orchestrator's MAP_SHARED "
            "pool (visible to forked child workers). Lifetime: until the next Worker.run() call."
        )
        .def(
            "scope_begin", &Orchestrator::scope_begin, "Open a nested scope. Max nesting depth = MAX_SCOPE_DEPTH (64)."
        )
        .def("scope_end", &Orchestrator::scope_end, "Close the innermost scope. Non-blocking.")
        .def("_scope_begin", &Orchestrator::scope_begin)
        .def("_scope_end", &Orchestrator::scope_end)
        .def(
            "_drain", &Orchestrator::drain, nb::call_guard<nb::gil_scoped_release>(),
            "Block until all submitted tasks are CONSUMED (releases GIL)."
        );

    // --- Worker ---
    // Bound as `_Worker` because the Python user-facing `Worker` factory
    // (simpler.worker.Worker) wraps this C++ class.
    nb::class_<Worker>(m, "_Worker")
        .def(
            nb::init<int32_t, uint64_t>(), nb::arg("level"), nb::arg("heap_ring_size") = DEFAULT_HEAP_RING_SIZE,
            "Create a Worker for the given hierarchy level (3=L3, 4=L4, …). "
            "`heap_ring_size` selects the per-ring MAP_SHARED heap mmap'd in the ctor "
            "(default 1 GiB; total VA = 4 × heap_ring_size)."
        )

        // THREAD-mode registration (parent calls worker->run directly).
        .def(
            "add_next_level_worker",
            [](Worker &self, Worker &w) {
                self.add_worker(WorkerType::NEXT_LEVEL, &w);
            },
            nb::arg("worker"), "Add a lower-level Worker as a NEXT_LEVEL sub-worker (THREAD mode)."
        )
        .def(
            "add_next_level_worker",
            [](Worker &self, ChipWorker &w) {
                self.add_worker(WorkerType::NEXT_LEVEL, &w);
            },
            nb::arg("worker"), "Add a ChipWorker as a NEXT_LEVEL sub-worker (THREAD mode)."
        )

        // PROCESS-mode registration (parent writes unified mailbox; child runs
        // the real IWorker in its own address space).
        .def(
            "add_next_level_process",
            [](Worker &self, uint64_t mailbox_ptr) {
                self.add_process_worker(WorkerType::NEXT_LEVEL, reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"),
            "Add a PROCESS-mode NEXT_LEVEL worker. `mailbox_ptr` is the address of a "
            "MAILBOX_SIZE-byte MAP_SHARED region. The child process loop is "
            "Python-managed (fork + _chip_process_loop)."
        )
        .def(
            "add_sub_process",
            [](Worker &self, uint64_t mailbox_ptr) {
                self.add_process_worker(WorkerType::SUB, reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"),
            "Add a PROCESS-mode SUB worker. `mailbox_ptr` is the address of a "
            "MAILBOX_SIZE-byte MAP_SHARED region. The child process loop is "
            "Python-managed (fork + _sub_worker_loop)."
        )

        .def("init", &Worker::init, "Start the Scheduler thread.")
        .def("close", &Worker::close, "Stop the Scheduler thread.")

        // THREAD-mode callback for L4+ recursion (approach b: Python callback).
        // The lambda captures the Python callable and wraps it with GIL
        // acquisition + TaskArgsView→TaskArgs reconstruction so the Python
        // side receives normal objects.
        .def(
            "set_run_callback",
            [](Worker &self, nb::object cb) {
                self.set_run_callback(
                    [cb_stored = nb::object(cb)](uint64_t callable, TaskArgsView view, const ChipCallConfig &config) {
                        nb::gil_scoped_acquire gil;
                        TaskArgs args;
                        for (int32_t i = 0; i < view.tensor_count; i++) {
                            args.add_tensor(view.tensors[i]);
                        }
                        for (int32_t i = 0; i < view.scalar_count; i++) {
                            args.add_scalar(view.scalars[i]);
                        }
                        cb_stored(callable, &args, &config);
                    }
                );
            },
            nb::arg("callback"),
            "Set the Python callback for THREAD-mode L4+ dispatch. The callback "
            "receives (callable_id, TaskArgs, ChipCallConfig) with the GIL held."
        )

        .def(
            "get_orchestrator", &Worker::get_orchestrator, nb::rv_policy::reference_internal,
            "Return the Orchestrator handle (lifetime tied to this Worker)."
        );

    m.attr("DEFAULT_HEAP_RING_SIZE") = static_cast<uint64_t>(DEFAULT_HEAP_RING_SIZE);
    m.attr("MAILBOX_SIZE") = static_cast<int>(MAILBOX_SIZE);
    m.attr("MAX_RING_DEPTH") = static_cast<int32_t>(MAX_RING_DEPTH);
    m.attr("MAX_SCOPE_DEPTH") = static_cast<int32_t>(MAX_SCOPE_DEPTH);
}
