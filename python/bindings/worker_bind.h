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
 * MAP_SHARED region; the real worker (a `ChipWorker` for NEXT_LEVEL, a Python
 * callable for SUB) lives in a forked Python child consuming the mailbox via
 * `_chip_process_loop` / `_sub_worker_loop`.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ring.h"
#include "orchestrator.h"
#include "types.h"
#include "worker.h"
#include "worker_manager.h"

namespace nb = nanobind;

class PyBufferView {
public:
    explicit PyBufferView(nb::handle obj) {
        if (PyObject_GetBuffer(obj.ptr(), &view_, PyBUF_CONTIG_RO) != 0) {
            throw nb::python_error();
        }
    }

    ~PyBufferView() { PyBuffer_Release(&view_); }

    PyBufferView(const PyBufferView &) = delete;
    PyBufferView &operator=(const PyBufferView &) = delete;

    const void *data() const { return view_.buf; }
    Py_ssize_t size() const { return view_.len; }

private:
    Py_buffer view_{};
};

inline uint64_t checked_remote_range_end(const char *op_name, uint64_t base, uint64_t offset, uint64_t size) {
    uint64_t max = std::numeric_limits<uint64_t>::max();
    if (offset > max - size || base > max - offset - size) {
        throw std::invalid_argument(std::string(op_name) + ": remote buffer range overflows");
    }
    return base + offset + size;
}

inline uint64_t
validated_handle_nbytes(uint64_t nbytes, const char *op_name, uint64_t base, uint64_t offset, uint64_t size) {
    uint64_t minimum = checked_remote_range_end(op_name, base, offset, size);
    if (nbytes < minimum) {
        throw std::invalid_argument(std::string(op_name) + ": handle_nbytes is smaller than requested range");
    }
    return nbytes;
}

inline std::string buffer_to_string(nb::handle obj, const char *field_name) {
    PyBufferView view(obj);
    if (view.size() < 0) {
        throw std::invalid_argument(std::string(field_name) + " must not have negative length");
    }
    if (view.size() == 0) return {};
    return std::string(static_cast<const char *>(view.data()), static_cast<size_t>(view.size()));
}

inline CallableKind parse_callable_kind(const std::string &kind) {
    if (kind == "CHIP_CALLABLE") return CallableKind::CHIP_CALLABLE;
    if (kind == "PYTHON_SERIALIZED") return CallableKind::PYTHON_SERIALIZED;
    if (kind == "PYTHON_IMPORT") return CallableKind::PYTHON_IMPORT;
    throw std::invalid_argument("CALLABLE_KIND_UNSUPPORTED: " + kind);
}

inline TargetNamespace parse_target_namespace(const std::string &target_namespace) {
    if (target_namespace == "LOCAL_CHIP") return TargetNamespace::LOCAL_CHIP;
    if (target_namespace == "LOCAL_PYTHON") return TargetNamespace::LOCAL_PYTHON;
    if (target_namespace == "REMOTE_TASK_DISPATCHER") return TargetNamespace::REMOTE_TASK_DISPATCHER;
    throw std::invalid_argument("unsupported callable target namespace: " + target_namespace);
}

inline remote_l3::RemoteRegistryTarget parse_remote_registry_target(const std::string &target_registry) {
    if (target_registry == "REMOTE_TASK_DISPATCHER") return remote_l3::RemoteRegistryTarget::REMOTE_TASK_DISPATCHER;
    if (target_registry == "INNER_L3_WORKER") return remote_l3::RemoteRegistryTarget::INNER_L3_WORKER;
    throw std::invalid_argument("unsupported remote registry target: " + target_registry);
}

inline CallableIdentity
make_callable_identity(nb::bytes digest, const std::string &kind, const std::string &target_namespace) {
    PyBufferView view(digest);
    if (view.size() != static_cast<Py_ssize_t>(CALLABLE_HASH_DIGEST_SIZE)) {
        throw std::invalid_argument("callable digest must be exactly 32 bytes");
    }
    CallableIdentity out;
    std::memcpy(out.digest.data(), view.data(), CALLABLE_HASH_DIGEST_SIZE);
    out.kind = parse_callable_kind(kind);
    out.target_namespace = parse_target_namespace(target_namespace);
    return out;
}

inline std::string bytes_from_digest_arg(nb::object digest) {
    std::string out = buffer_to_string(digest, "callable digest");
    if (out.size() != CALLABLE_HASH_DIGEST_SIZE) {
        throw std::invalid_argument("callable digest must be exactly 32 bytes");
    }
    return out;
}

inline std::vector<uint8_t> bytes_to_u8_vector(nb::handle obj, const char *field_name) {
    PyBufferView view(obj);
    if (view.size() < 0) {
        throw std::invalid_argument(std::string(field_name) + " must not have negative length");
    }
    if (view.size() == 0) return {};
    auto *begin = static_cast<const uint8_t *>(view.data());
    return std::vector<uint8_t>(begin, begin + static_cast<size_t>(view.size()));
}

inline RemoteAddressSpace parse_remote_address_space(nb::handle value) {
    int v = nb::cast<int>(value);
    switch (v) {
    case static_cast<int>(RemoteAddressSpace::HOST_INLINE):
        return RemoteAddressSpace::HOST_INLINE;
    case static_cast<int>(RemoteAddressSpace::REMOTE_DEVICE):
        return RemoteAddressSpace::REMOTE_DEVICE;
    case static_cast<int>(RemoteAddressSpace::REMOTE_WINDOW):
        return RemoteAddressSpace::REMOTE_WINDOW;
    case static_cast<int>(RemoteAddressSpace::UB_LDST):
        return RemoteAddressSpace::UB_LDST;
    default:
        throw std::invalid_argument("unknown RemoteAddressSpace value: " + std::to_string(v));
    }
}

inline RemoteTensorSidecar parse_remote_tensor_sidecar(nb::handle obj) {
    RemoteTensorSidecar sidecar{};
    if (obj.is_none()) return sidecar;

    sidecar.present = nb::cast<bool>(obj.attr("present"));
    if (!sidecar.present) return sidecar;

    nb::object desc = obj.attr("desc");
    sidecar.desc.address_space = parse_remote_address_space(desc.attr("address_space"));
    sidecar.desc.owner_endpoint_id = nb::cast<int32_t>(desc.attr("owner_endpoint_id"));
    sidecar.desc.buffer_id = nb::cast<uint64_t>(desc.attr("buffer_id"));
    sidecar.desc.offset = nb::cast<uint64_t>(desc.attr("offset"));
    sidecar.desc.nbytes = nb::cast<uint64_t>(desc.attr("nbytes"));
    sidecar.desc.remote_addr = nb::cast<uint64_t>(desc.attr("remote_addr"));
    sidecar.desc.rkey_or_token = nb::cast<uint64_t>(desc.attr("rkey_or_token"));
    sidecar.desc.generation = nb::cast<uint64_t>(desc.attr("generation"));
    sidecar.desc.inline_payload_offset = nb::cast<uint64_t>(desc.attr("inline_payload_offset"));
    sidecar.desc.inline_payload_len = nb::cast<uint64_t>(desc.attr("inline_payload_len"));
    sidecar.desc.flags = nb::cast<uint64_t>(desc.attr("flags"));
    return sidecar;
}

inline RemoteTaskArgsSidecar parse_remote_task_args_sidecar(nb::handle obj) {
    RemoteTaskArgsSidecar sidecar{};
    if (obj.is_none()) return sidecar;

    nb::object tensors_obj = obj.attr("tensors");
    for (nb::handle item : nb::borrow<nb::iterable>(tensors_obj)) {
        sidecar.tensors.push_back(parse_remote_tensor_sidecar(item));
    }
    sidecar.inline_payload = bytes_to_u8_vector(obj.attr("inline_payload"), "remote sidecar inline_payload");
    return sidecar;
}

inline std::vector<RemoteTaskArgsSidecar> parse_remote_task_args_sidecars(nb::handle obj) {
    std::vector<RemoteTaskArgsSidecar> sidecars;
    if (obj.is_none()) return sidecars;
    for (nb::handle item : nb::borrow<nb::iterable>(obj)) {
        sidecars.push_back(parse_remote_task_args_sidecar(item));
    }
    return sidecars;
}

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

    nb::class_<ControlResult>(m, "ControlResult")
        .def_ro("worker_type", &ControlResult::worker_type)
        .def_ro("worker_index", &ControlResult::worker_index)
        .def_ro("ok", &ControlResult::ok)
        .def_ro("error_message", &ControlResult::error_message);

    // --- TaskState ---
    nb::enum_<TaskState>(m, "TaskState")
        .value("FREE", TaskState::FREE)
        .value("PENDING", TaskState::PENDING)
        .value("READY", TaskState::READY)
        .value("RUNNING", TaskState::RUNNING)
        .value("COMPLETED", TaskState::COMPLETED)
        .value("FAILED", TaskState::FAILED)
        .value("CONSUMED", TaskState::CONSUMED);

    // --- Orchestrator (DAG builder, exposed via Worker.get_orchestrator()) ---
    // Bound as `_Orchestrator` because the Python user-facing `Orchestrator`
    // wrapper (simpler.orchestrator.Orchestrator) holds a borrowed reference
    // to this C++ type.
    nb::class_<Orchestrator>(m, "_Orchestrator")
        .def(
            "submit_next_level",
            [](Orchestrator &self, nb::bytes digest, const std::string &kind, const std::string &target_namespace,
               const TaskArgs &args, const CallConfig &config, int8_t worker,
               const std::vector<int32_t> &eligible_endpoint_ids, nb::object remote_sidecar) {
                self.submit_next_level(
                    make_callable_identity(digest, kind, target_namespace), args, config, worker, eligible_endpoint_ids,
                    parse_remote_task_args_sidecar(remote_sidecar)
                );
            },
            nb::arg("digest"), nb::arg("kind"), nb::arg("target_namespace"), nb::arg("args"), nb::arg("config"),
            nb::arg("worker") = int8_t(-1), nb::arg("eligible_endpoint_ids") = std::vector<int32_t>{},
            nb::arg("remote_sidecar") = nb::none(),
            "Submit a NEXT_LEVEL task by registered callable digest. "
            "worker= pins to a specific next-level worker (-1 = any)."
        )
        .def(
            "submit_next_level_group",
            [](Orchestrator &self, nb::bytes digest, const std::string &kind, const std::string &target_namespace,
               const std::vector<TaskArgs> &args_list, const CallConfig &config, const std::vector<int8_t> &workers,
               const std::vector<std::vector<int32_t>> &eligible_endpoint_ids, nb::object remote_sidecars) {
                self.submit_next_level_group(
                    make_callable_identity(digest, kind, target_namespace), args_list, config, workers,
                    eligible_endpoint_ids, parse_remote_task_args_sidecars(remote_sidecars)
                );
            },
            nb::arg("digest"), nb::arg("kind"), nb::arg("target_namespace"), nb::arg("args_list"), nb::arg("config"),
            nb::arg("workers") = std::vector<int8_t>{},
            nb::arg("eligible_endpoint_ids") = std::vector<std::vector<int32_t>>{},
            nb::arg("remote_sidecars") = nb::none(),
            "Submit a group of NEXT_LEVEL tasks by registered callable digest. "
            "workers= per-args affinity (empty = any)."
        )
        .def(
            "submit_sub",
            [](Orchestrator &self, nb::bytes digest, const std::string &kind, const std::string &target_namespace,
               const TaskArgs &args) {
                self.submit_sub(make_callable_identity(digest, kind, target_namespace), args);
            },
            nb::arg("digest"), nb::arg("kind"), nb::arg("target_namespace"), nb::arg("args"),
            "Submit a SUB task by registered callable digest. Tags drive dependency inference."
        )
        .def(
            "submit_sub_group",
            [](Orchestrator &self, nb::bytes digest, const std::string &kind, const std::string &target_namespace,
               const std::vector<TaskArgs> &args_list) {
                self.submit_sub_group(make_callable_identity(digest, kind, target_namespace), args_list);
            },
            nb::arg("digest"), nb::arg("kind"), nb::arg("target_namespace"), nb::arg("args_list"),
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
        .def(
            "add_remote_l3_socket",
            [](Worker &self, int32_t endpoint_id, uint64_t session_id, const std::string &transport_name,
               const std::string &host, uint16_t port, const std::string &health_host, uint16_t health_port,
               double timeout_s) {
                nb::gil_scoped_release release;
                self.add_remote_l3_socket(
                    endpoint_id, session_id, transport_name, host, port, health_host, health_port, timeout_s
                );
            },
            nb::arg("endpoint_id"), nb::arg("session_id"), nb::arg("transport_name"), nb::arg("host"), nb::arg("port"),
            nb::arg("health_host"), nb::arg("health_port"), nb::arg("timeout_s") = 30.0,
            "Register a REMOTE_L3 endpoint after the session reports HELLO READY."
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
            "control_prepare",
            [](Worker &self, int worker_id, nb::object digest) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                self.control_prepare(worker_id, reinterpret_cast<const uint8_t *>(digest_bytes.data()));
            },
            nb::arg("worker_id"), nb::arg("digest"), "Prewarm a NEXT_LEVEL child by callable digest."
        )
        .def(
            "broadcast_register_all",
            [](Worker &self, uint64_t blob_ptr, uint64_t blob_size, nb::object digest) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                return self.broadcast_register_all(
                    blob_ptr, blob_size, reinterpret_cast<const uint8_t *>(digest_bytes.data())
                );
            },
            nb::arg("blob_ptr"), nb::arg("blob_size"), nb::arg("digest"),
            "Stage `blob_size` bytes from `blob_ptr` into a POSIX shm and broadcast "
            "CTRL_REGISTER to every NEXT_LEVEL child in parallel. Returns per-child status."
        )
        .def(
            "control_digest_only",
            [](Worker &self, WorkerType worker_type, int worker_id, uint64_t sub_cmd, nb::object digest,
               nb::object timeout_s) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                double timeout_val = timeout_s.is_none() ? -1.0 : nb::cast<double>(timeout_s);
                nb::gil_scoped_release release;
                return self.control_digest_only(
                    worker_type, worker_id, sub_cmd, reinterpret_cast<const uint8_t *>(digest_bytes.data()), timeout_val
                );
            },
            nb::arg("worker_type"), nb::arg("worker_id"), nb::arg("sub_cmd"), nb::arg("digest"),
            nb::arg("timeout_s") = nb::none(),
            "Drive one selected worker through a digest-only CONTROL_REQUEST. "
            "Used by registration cleanup after partial broadcast failures."
        )
        .def(
            "remote_prepare_register",
            [](Worker &self, int endpoint_id, const std::string &target_registry, const std::string &kind,
               nb::object payload, nb::object digest) {
                std::string payload_bytes;
                if (!payload.is_none()) {
                    payload_bytes = buffer_to_string(payload, "payload");
                }
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                return self.remote_prepare_register(
                    endpoint_id, parse_remote_registry_target(target_registry), parse_callable_kind(kind),
                    payload_bytes.data(), payload_bytes.size(), reinterpret_cast<const uint8_t *>(digest_bytes.data())
                );
            },
            nb::arg("endpoint_id"), nb::arg("target_registry"), nb::arg("kind"), nb::arg("payload"), nb::arg("digest"),
            "Send PREPARE_REGISTER_CALLABLE to one remote endpoint."
        )
        .def(
            "remote_commit_register",
            [](Worker &self, int endpoint_id, const std::string &target_registry, const std::string &kind,
               nb::object digest) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                return self.remote_commit_register(
                    endpoint_id, parse_remote_registry_target(target_registry), parse_callable_kind(kind),
                    reinterpret_cast<const uint8_t *>(digest_bytes.data())
                );
            },
            nb::arg("endpoint_id"), nb::arg("target_registry"), nb::arg("kind"), nb::arg("digest"),
            "Send COMMIT_REGISTER_CALLABLE to one remote endpoint."
        )
        .def(
            "remote_abort_register",
            [](Worker &self, int endpoint_id, const std::string &target_registry, const std::string &kind,
               nb::object digest) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                return self.remote_abort_register(
                    endpoint_id, parse_remote_registry_target(target_registry), parse_callable_kind(kind),
                    reinterpret_cast<const uint8_t *>(digest_bytes.data())
                );
            },
            nb::arg("endpoint_id"), nb::arg("target_registry"), nb::arg("kind"), nb::arg("digest"),
            "Send ABORT_REGISTER_CALLABLE to one remote endpoint."
        )
        .def(
            "remote_unregister",
            [](Worker &self, int endpoint_id, const std::string &target_registry, const std::string &kind,
               nb::object digest) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                return self.remote_unregister(
                    endpoint_id, parse_remote_registry_target(target_registry), parse_callable_kind(kind),
                    reinterpret_cast<const uint8_t *>(digest_bytes.data())
                );
            },
            nb::arg("endpoint_id"), nb::arg("target_registry"), nb::arg("kind"), nb::arg("digest"),
            "Send UNREGISTER_CALLABLE to one remote endpoint."
        )
        .def(
            "remote_malloc",
            [](Worker &self, int endpoint_id, size_t size) {
                RemoteBufferHandle h;
                {
                    nb::gil_scoped_release release;
                    h = self.remote_malloc(endpoint_id, size);
                }
                return nb::make_tuple(
                    h.endpoint_id, h.buffer_id, h.generation, static_cast<int32_t>(h.address_space), h.nbytes,
                    h.remote_addr, h.rkey_or_token, h.ub_ldst_va
                );
            },
            nb::arg("endpoint_id"), nb::arg("size"), "Allocate a remote buffer and return handle fields."
        )
        .def(
            "remote_free",
            [](Worker &self, int endpoint_id, uint64_t buffer_id, uint64_t generation) {
                RemoteBufferHandle h;
                h.endpoint_id = endpoint_id;
                h.buffer_id = buffer_id;
                h.generation = generation;
                nb::gil_scoped_release release;
                self.remote_free(h);
            },
            nb::arg("endpoint_id"), nb::arg("buffer_id"), nb::arg("generation"), "Free a remote buffer."
        )
        .def(
            "remote_copy_to",
            [](Worker &self, int endpoint_id, uint64_t buffer_id, uint64_t generation, uint64_t offset, uint64_t src,
               size_t size, uint64_t handle_nbytes) {
                RemoteBufferHandle h;
                h.endpoint_id = endpoint_id;
                h.buffer_id = buffer_id;
                h.generation = generation;
                h.nbytes = validated_handle_nbytes(handle_nbytes, "remote_copy_to", 0, offset, size);
                nb::gil_scoped_release release;
                self.remote_copy_to(h, offset, reinterpret_cast<const void *>(src), size);
            },
            nb::arg("endpoint_id"), nb::arg("buffer_id"), nb::arg("generation"), nb::arg("offset"), nb::arg("src"),
            nb::arg("size"), nb::arg("handle_nbytes"), "Copy host bytes into a remote buffer."
        )
        .def(
            "remote_copy_from",
            [](Worker &self, uint64_t dst, int endpoint_id, uint64_t buffer_id, uint64_t generation, uint64_t offset,
               size_t size, uint64_t handle_nbytes) {
                RemoteBufferHandle h;
                h.endpoint_id = endpoint_id;
                h.buffer_id = buffer_id;
                h.generation = generation;
                h.nbytes = validated_handle_nbytes(handle_nbytes, "remote_copy_from", 0, offset, size);
                nb::gil_scoped_release release;
                self.remote_copy_from(reinterpret_cast<void *>(dst), h, offset, size);
            },
            nb::arg("dst"), nb::arg("endpoint_id"), nb::arg("buffer_id"), nb::arg("generation"), nb::arg("offset"),
            nb::arg("size"), nb::arg("handle_nbytes"), "Copy remote buffer bytes into host memory."
        )
        .def(
            "remote_export",
            [](Worker &self, int owner_endpoint_id, uint64_t buffer_id, uint64_t generation, uint64_t handle_offset,
               uint64_t offset, uint64_t size, uint32_t access_flags, const std::string &transport_profile,
               uint64_t handle_nbytes) {
                RemoteBufferHandle h;
                h.endpoint_id = owner_endpoint_id;
                h.owner_endpoint_id = owner_endpoint_id;
                h.buffer_id = buffer_id;
                h.generation = generation;
                h.offset = handle_offset;
                h.nbytes = validated_handle_nbytes(handle_nbytes, "remote_export", handle_offset, offset, size);
                RemoteBufferExport e;
                {
                    nb::gil_scoped_release release;
                    e = self.remote_export(h, offset, size, access_flags, transport_profile);
                }
                return nb::make_tuple(
                    e.owner_endpoint_id, e.buffer_id, e.generation, static_cast<int32_t>(e.address_space), e.offset,
                    e.nbytes, e.export_id, e.remote_addr, e.rkey_or_token, e.ub_ldst_va, e.access_flags,
                    e.transport_profile,
                    nb::bytes(
                        reinterpret_cast<const char *>(e.transport_descriptor.data()), e.transport_descriptor.size()
                    )
                );
            },
            nb::arg("owner_endpoint_id"), nb::arg("buffer_id"), nb::arg("generation"), nb::arg("handle_offset"),
            nb::arg("offset"), nb::arg("size"), nb::arg("access_flags"), nb::arg("transport_profile"),
            nb::arg("handle_nbytes"), "Export a remote buffer range and return export descriptor fields."
        )
        .def(
            "remote_import",
            [](Worker &self, int importer_endpoint_id, int owner_endpoint_id, uint64_t buffer_id, uint64_t generation,
               int address_space, uint64_t offset, uint64_t size, uint64_t export_id, uint64_t remote_addr,
               uint64_t rkey_or_token, uint64_t ub_ldst_va, uint32_t access_flags, const std::string &transport_profile,
               nb::object transport_descriptor, uint32_t requested_access_flags) {
                RemoteBufferExport e;
                e.owner_endpoint_id = owner_endpoint_id;
                e.buffer_id = buffer_id;
                e.generation = generation;
                e.address_space = parse_remote_address_space(nb::int_(address_space));
                e.offset = offset;
                e.nbytes = size;
                e.export_id = export_id;
                e.remote_addr = remote_addr;
                e.rkey_or_token = rkey_or_token;
                e.ub_ldst_va = ub_ldst_va;
                e.access_flags = access_flags;
                e.transport_profile = transport_profile;
                e.transport_descriptor = bytes_to_u8_vector(transport_descriptor, "transport_descriptor");
                RemoteBufferHandle h;
                {
                    nb::gil_scoped_release release;
                    h = self.remote_import(importer_endpoint_id, e, requested_access_flags);
                }
                return nb::make_tuple(
                    h.endpoint_id, h.owner_endpoint_id, h.buffer_id, h.generation, h.import_id,
                    static_cast<int32_t>(h.address_space), h.nbytes, h.offset, h.remote_addr, h.rkey_or_token,
                    h.ub_ldst_va, h.access_flags
                );
            },
            nb::arg("importer_endpoint_id"), nb::arg("owner_endpoint_id"), nb::arg("buffer_id"), nb::arg("generation"),
            nb::arg("address_space"), nb::arg("offset"), nb::arg("size"), nb::arg("export_id"), nb::arg("remote_addr"),
            nb::arg("rkey_or_token"), nb::arg("ub_ldst_va"), nb::arg("access_flags"), nb::arg("transport_profile"),
            nb::arg("transport_descriptor"), nb::arg("requested_access_flags"),
            "Import a remote buffer export into an endpoint."
        )
        .def(
            "remote_release_import",
            [](Worker &self, int importer_endpoint_id, int owner_endpoint_id, uint64_t buffer_id, uint64_t generation,
               uint64_t import_id) {
                RemoteBufferHandle h;
                h.endpoint_id = importer_endpoint_id;
                h.owner_endpoint_id = owner_endpoint_id;
                h.buffer_id = buffer_id;
                h.generation = generation;
                h.import_id = import_id;
                nb::gil_scoped_release release;
                self.remote_release_import(h);
            },
            nb::arg("importer_endpoint_id"), nb::arg("owner_endpoint_id"), nb::arg("buffer_id"), nb::arg("generation"),
            nb::arg("import_id"), "Release an imported remote buffer mapping."
        )
        .def(
            "broadcast_unregister_all",
            [](Worker &self, nb::object digest) {
                std::string digest_bytes = bytes_from_digest_arg(digest);
                nb::gil_scoped_release release;
                return self.broadcast_unregister_all(reinterpret_cast<const uint8_t *>(digest_bytes.data()));
            },
            nb::arg("digest"),
            "Best-effort broadcast of CTRL_UNREGISTER to every NEXT_LEVEL child in parallel. "
            "Returns a list of per-child error strings (empty on full success)."
        )
        .def(
            "broadcast_control_all",
            [](Worker &self, WorkerType worker_type, uint64_t sub_cmd, nb::object payload, nb::object digest,
               nb::object timeout_s) {
                std::string payload_bytes;
                const void *payload_ptr = nullptr;
                size_t payload_size = 0;
                if (!payload.is_none()) {
                    payload_bytes = buffer_to_string(payload, "payload");
                    payload_ptr = payload_bytes.data();
                    payload_size = payload_bytes.size();
                }
                std::string digest_bytes;
                const uint8_t *digest_ptr = nullptr;
                if (!digest.is_none()) {
                    digest_bytes = bytes_from_digest_arg(digest);
                    digest_ptr = reinterpret_cast<const uint8_t *>(digest_bytes.data());
                }
                double timeout_val = timeout_s.is_none() ? -1.0 : nb::cast<double>(timeout_s);
                nb::gil_scoped_release release;
                return self.broadcast_control_all(
                    worker_type, sub_cmd, payload_ptr, payload_size, digest_ptr, timeout_val
                );
            },
            nb::arg("worker_type"), nb::arg("sub_cmd"), nb::arg("payload") = nb::none(), nb::arg("digest") = nb::none(),
            nb::arg("timeout_s") = nb::none(),
            "Broadcast an arbitrary CONTROL_REQUEST to the selected worker pool. "
            "If payload is a Python buffer, C++ stages it in POSIX shm and writes the shm name "
            "into the mailbox. Returns per-child ControlResult entries."
        )
        .def(
            "control_alloc_domain", &Worker::control_alloc_domain, nb::arg("worker_id"), nb::arg("request_shm_name"),
            nb::arg("reply_shm_name"), nb::call_guard<nb::gil_scoped_release>(),
            "Drive one NEXT_LEVEL chip child through CTRL_ALLOC_DOMAIN.  Holds mailbox_mu_ "
            "so it serialises with task dispatch on the same mailbox.  Caller fans out to all "
            "participating chips in parallel (one Python thread per chip)."
        )
        .def(
            "control_release_domain", &Worker::control_release_domain, nb::arg("worker_id"),
            nb::arg("request_shm_name"), nb::call_guard<nb::gil_scoped_release>(),
            "Drive one NEXT_LEVEL chip child through CTRL_RELEASE_DOMAIN.  Same serialisation "
            "semantics as control_alloc_domain."
        )
        .def(
            "control_comm_init", &Worker::control_comm_init, nb::arg("worker_id"), nb::arg("request_shm_name"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Drive one NEXT_LEVEL chip child through CTRL_COMM_INIT (lazy base comm init)."
        );

    m.attr("DEFAULT_HEAP_RING_SIZE") = static_cast<uint64_t>(DEFAULT_HEAP_RING_SIZE);
    m.attr("MAILBOX_SIZE") = static_cast<int>(MAILBOX_SIZE);
    m.attr("MAILBOX_OFF_ERROR_MSG") = static_cast<int>(MAILBOX_OFF_ERROR_MSG);
    m.attr("MAILBOX_ERROR_MSG_SIZE") = static_cast<int>(MAILBOX_ERROR_MSG_SIZE);
    m.attr("MAX_RING_DEPTH") = static_cast<int32_t>(MAX_RING_DEPTH);
    m.attr("MAX_SCOPE_DEPTH") = static_cast<int32_t>(MAX_SCOPE_DEPTH);

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
