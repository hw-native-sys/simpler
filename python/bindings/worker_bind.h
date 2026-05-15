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
 * Python callers register sub-workers via `add_next_level_worker(mailbox_ptr)`
 * / `add_sub_worker(mailbox_ptr)`. Each mailbox addresses a MAILBOX_SIZE-byte
 * MAP_SHARED region; the real IWorker lives in a forked Python child consuming
 * the mailbox via `_chip_process_loop` / `_sub_worker_loop`.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>

#include "chip_bootstrap_channel.h"
#include "ring.h"
#include "orchestrator.h"
#include "types.h"
#include "worker.h"
#include "worker_manager.h"

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Mailbox acquire/release helpers (exposed to Python as _mailbox_load_i32 /
// _mailbox_store_i32). Mirror WorkerThread::read_mailbox_state /
// write_mailbox_state in worker_manager.cpp so the Python side of the mailbox
// handshake uses the same memory order as the C++ side. Without these, a
// plain struct.pack_into("i", ...) on the Python child followed by the parent
// C++ acquire-load on aarch64 can observe the state flip before the
// preceding error-field writes are visible.
inline int32_t mailbox_load_i32(uint64_t addr) {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(addr);
    int32_t v;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    v = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
#endif
    return v;
}

inline void mailbox_store_i32(uint64_t addr, int32_t v) {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(addr);
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(v), "r"(ptr) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *ptr = v;
#else
    __atomic_store(ptr, &v, __ATOMIC_RELEASE);
#endif
}

inline void bind_worker(nb::module_ &m) {
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

    // --- SubmitResult ---
    nb::class_<SubmitResult>(m, "SubmitResult").def_prop_ro("task_slot", [](const SubmitResult &r) {
        return r.task_slot;
    });

    // --- Orchestrator (DAG builder, exposed via Worker.get_orchestrator()) ---
    // Bound as `_Orchestrator` because the Python user-facing `Orchestrator`
    // wrapper (simpler.orchestrator.Orchestrator) holds a borrowed reference
    // to this C++ type.
    nb::class_<Orchestrator>(m, "_Orchestrator")
        .def(
            "submit_next_level",
            [](Orchestrator &self, int32_t callable_id, const TaskArgs &args, const CallConfig &config, int8_t worker) {
                return self.submit_next_level(callable_id, args, config, worker);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"), nb::arg("worker") = int8_t(-1),
            "Submit a NEXT_LEVEL (chip) task by registered callable id. "
            "worker= pins to a specific next-level worker (-1 = any)."
        )
        .def(
            "submit_next_level_group",
            [](Orchestrator &self, int32_t callable_id, const std::vector<TaskArgs> &args_list,
               const CallConfig &config, const std::vector<int8_t> &workers) {
                return self.submit_next_level_group(callable_id, args_list, config, workers);
            },
            nb::arg("callable_id"), nb::arg("args_list"), nb::arg("config"), nb::arg("workers") = std::vector<int8_t>{},
            "Submit a group of NEXT_LEVEL tasks by registered callable id. "
            "workers= per-args affinity (empty = any)."
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
            "malloc",
            [](Orchestrator &self, int worker_id, size_t size) {
                return self.malloc(worker_id, size);
            },
            nb::arg("worker_id"), nb::arg("size"), "Allocate memory on next-level worker."
        )
        .def(
            "free",
            [](Orchestrator &self, int worker_id, uint64_t ptr) {
                self.free(worker_id, ptr);
            },
            nb::arg("worker_id"), nb::arg("ptr"), "Free memory on next-level worker."
        )
        .def(
            "copy_to",
            [](Orchestrator &self, int worker_id, uint64_t dst, uint64_t src, size_t size) {
                self.copy_to(worker_id, dst, src, size);
            },
            nb::arg("worker_id"), nb::arg("dst"), nb::arg("src"), nb::arg("size"), "Copy host src to worker dst."
        )
        .def(
            "copy_from",
            [](Orchestrator &self, int worker_id, uint64_t dst, uint64_t src, size_t size) {
                self.copy_from(worker_id, dst, src, size);
            },
            nb::arg("worker_id"), nb::arg("dst"), nb::arg("src"), nb::arg("size"), "Copy worker src to host dst."
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
            "Block until all submitted tasks are CONSUMED (releases GIL). "
            "Rethrows the first dispatch failure seen in this run, if any."
        )
        .def(
            "_clear_error", &Orchestrator::clear_error, "Clear any stored dispatch error so the next run can proceed."
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

        .def(
            "add_next_level_worker",
            [](Worker &self, uint64_t mailbox_ptr) {
                self.add_worker(WorkerType::NEXT_LEVEL, reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"),
            "Add a NEXT_LEVEL sub-worker. `mailbox_ptr` is the address of a "
            "MAILBOX_SIZE-byte MAP_SHARED region; the child process loop is "
            "Python-managed (fork + _chip_process_loop)."
        )
        .def(
            "add_sub_worker",
            [](Worker &self, uint64_t mailbox_ptr) {
                self.add_worker(WorkerType::SUB, reinterpret_cast<void *>(mailbox_ptr));
            },
            nb::arg("mailbox_ptr"),
            "Add a SUB sub-worker. `mailbox_ptr` is the address of a "
            "MAILBOX_SIZE-byte MAP_SHARED region; the child process loop is "
            "Python-managed (fork + _sub_worker_loop)."
        )

        .def("init", &Worker::init, "Start the Scheduler thread.")
        .def("close", &Worker::close, "Stop the Scheduler thread.")

        .def(
            "get_orchestrator", &Worker::get_orchestrator, nb::rv_policy::reference_internal,
            "Return the Orchestrator handle (lifetime tied to this Worker)."
        )

        // --- Mailbox control plane (parent side) ---
        // These hold the per-WorkerThread mailbox_mu_ inside C++, so they
        // serialize against dispatch_process without any Python-side lock.
        // Release the GIL during the spin-poll wait so other Python threads
        // (e.g. a concurrent Worker.run) can keep running.
        .def(
            "control_prepare", &Worker::control_prepare, nb::arg("worker_id"), nb::arg("cid"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Prewarm a NEXT_LEVEL child for `cid` by sending CTRL_PREPARE. "
            "Blocks until the child publishes CONTROL_DONE."
        )
        .def(
            "broadcast_register_all", &Worker::broadcast_register_all, nb::arg("cid"), nb::arg("blob_ptr"),
            nb::arg("blob_size"), nb::call_guard<nb::gil_scoped_release>(),
            "Stage `blob_size` bytes from `blob_ptr` into a POSIX shm and broadcast "
            "CTRL_REGISTER to every NEXT_LEVEL child in parallel. Throws on any failure."
        )
        .def(
            "broadcast_unregister_all", &Worker::broadcast_unregister_all, nb::arg("cid"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Best-effort broadcast of CTRL_UNREGISTER to every NEXT_LEVEL child in parallel. "
            "Returns a list of per-child error strings (empty on full success)."
        );

    m.attr("DEFAULT_HEAP_RING_SIZE") = static_cast<uint64_t>(DEFAULT_HEAP_RING_SIZE);
    m.attr("MAILBOX_SIZE") = static_cast<int>(MAILBOX_SIZE);
    m.attr("MAILBOX_OFF_ERROR_MSG") = static_cast<int>(MAILBOX_OFF_ERROR_MSG);
    m.attr("MAILBOX_ERROR_MSG_SIZE") = static_cast<int>(MAILBOX_ERROR_MSG_SIZE);
    m.attr("MAX_RING_DEPTH") = static_cast<int32_t>(MAX_RING_DEPTH);
    m.attr("MAX_SCOPE_DEPTH") = static_cast<int32_t>(MAX_SCOPE_DEPTH);

    // --- ChipBootstrapChannel ---
    m.attr("CHIP_BOOTSTRAP_MAILBOX_SIZE") = static_cast<int>(CHIP_BOOTSTRAP_MAILBOX_SIZE);

    nb::enum_<ChipBootstrapMailboxState>(m, "ChipBootstrapMailboxState")
        .value("IDLE", ChipBootstrapMailboxState::IDLE)
        .value("SUCCESS", ChipBootstrapMailboxState::SUCCESS)
        .value("ERROR", ChipBootstrapMailboxState::ERROR);

    nb::class_<ChipBootstrapChannel>(m, "ChipBootstrapChannel")
        .def(
            "__init__",
            [](ChipBootstrapChannel *self, uint64_t mailbox_ptr, size_t max_buffer_count) {
                new (self) ChipBootstrapChannel(reinterpret_cast<void *>(mailbox_ptr), max_buffer_count);
            },
            nb::arg("mailbox_ptr"), nb::arg("max_buffer_count")
        )
        .def("reset", &ChipBootstrapChannel::reset)
        .def(
            "write_success",
            [](ChipBootstrapChannel &self, uint64_t device_ctx, uint64_t local_window_base, uint64_t actual_window_size,
               const std::vector<uint64_t> &buffer_ptrs) {
                self.write_success(device_ctx, local_window_base, actual_window_size, buffer_ptrs);
            },
            nb::arg("device_ctx"), nb::arg("local_window_base"), nb::arg("actual_window_size"), nb::arg("buffer_ptrs")
        )
        .def(
            "write_error",
            [](ChipBootstrapChannel &self, int32_t error_code, const std::string &message) {
                self.write_error(error_code, message);
            },
            nb::arg("error_code"), nb::arg("message")
        )
        .def_prop_ro("state", &ChipBootstrapChannel::state)
        .def_prop_ro("error_code", &ChipBootstrapChannel::error_code)
        .def_prop_ro("device_ctx", &ChipBootstrapChannel::device_ctx)
        .def_prop_ro("local_window_base", &ChipBootstrapChannel::local_window_base)
        .def_prop_ro("actual_window_size", &ChipBootstrapChannel::actual_window_size)
        .def_prop_ro("buffer_ptrs", &ChipBootstrapChannel::buffer_ptrs)
        .def_prop_ro("error_message", &ChipBootstrapChannel::error_message);

    // Private mailbox acquire/release helpers — only for simpler.worker. The
    // underscore prefix keeps them out of the public surface; they do not
    // appear in task_interface.__all__.
    m.def(
        "_mailbox_load_i32",
        [](uint64_t addr) -> int32_t {
            return mailbox_load_i32(addr);
        },
        nb::arg("addr"), "Acquire-load a 32-bit mailbox word at `addr`."
    );
    m.def(
        "_mailbox_store_i32",
        [](uint64_t addr, int32_t value) {
            mailbox_store_i32(addr, value);
        },
        nb::arg("addr"), nb::arg("value"), "Release-store a 32-bit mailbox word at `addr`."
    );
}
