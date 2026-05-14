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
 * Nanobind Python extension for task_interface headers.
 *
 * Wraps DataType, ContinuousTensor, ChipStorageTaskArgs, TaskArgs (unified
 * vector-backed builder with per-tensor TensorArgType tags), TensorArgType,
 * ArgDirection, CoreCallable, ChipCallable, and helper functions from
 * data_type.h / tensor_arg.h / task_args.h / arg_direction.h / callable.h.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "arg_direction.h"
#include "callable.h"
#include "callable_protocol.h"
#include "chip_worker.h"
#include "data_type.h"
#include "worker_bind.h"
#include "task_args.h"
#include "tensor_arg.h"

namespace nb = nanobind;

// ============================================================================
// Module definition
// ============================================================================

NB_MODULE(_task_interface, m) {
    m.doc() = "Nanobind bindings for task_interface (DataType, ContinuousTensor, TaskArgs variants)";

    // --- DataType enum ---
    nb::enum_<DataType>(m, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT16", DataType::FLOAT16)
        .value("INT32", DataType::INT32)
        .value("INT16", DataType::INT16)
        .value("INT8", DataType::INT8)
        .value("UINT8", DataType::UINT8)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("INT64", DataType::INT64)
        .value("UINT64", DataType::UINT64)
        .value("UINT16", DataType::UINT16)
        .value("UINT32", DataType::UINT32);

    // --- Free functions ---
    m.def(
        "get_element_size", &get_element_size, nb::arg("dtype"),
        "Return the byte size of a single element of the given DataType."
    );

    m.def(
        "get_dtype_name",
        [](DataType dt) -> std::string {
            return get_dtype_name(dt);
        },
        nb::arg("dtype"), "Return the string name of a DataType."
    );

    // --- Constants ---
    m.attr("CONTINUOUS_TENSOR_MAX_DIMS") = CONTINUOUS_TENSOR_MAX_DIMS;
    m.attr("MAX_REGISTERED_CALLABLE_IDS") = MAX_REGISTERED_CALLABLE_IDS;

    // --- ContinuousTensor ---
    nb::class_<ContinuousTensor>(m, "ContinuousTensor")
        .def(nb::init<>())

        .def_static(
            "make",
            [](uint64_t data, nb::tuple shapes, DataType dtype, bool child_memory) -> ContinuousTensor {
                size_t n = nb::len(shapes);
                if (n > CONTINUOUS_TENSOR_MAX_DIMS)
                    throw std::invalid_argument("shapes length exceeds CONTINUOUS_TENSOR_MAX_DIMS");
                ContinuousTensor arg{};
                arg.data = data;
                arg.dtype = dtype;
                arg.ndims = static_cast<uint32_t>(n);
                arg.child_memory = child_memory ? 1 : 0;
                for (size_t i = 0; i < n; ++i)
                    arg.shapes[i] = nb::cast<uint32_t>(shapes[i]);
                return arg;
            },
            nb::arg("data"), nb::arg("shapes"), nb::arg("dtype"), nb::arg("child_memory") = false,
            "Create a ContinuousTensor. Set child_memory=True when data is a device pointer "
            "allocated by the child process (skips H2D copy in init_runtime_impl)."
        )

        .def_prop_rw(
            "data",
            [](const ContinuousTensor &self) -> uint64_t {
                return self.data;
            },
            [](ContinuousTensor &self, uint64_t v) {
                self.data = v;
            }
        )

        .def_prop_rw(
            "shapes",
            [](const ContinuousTensor &self) -> nb::tuple {
                uint32_t n = self.ndims;
                if (n > CONTINUOUS_TENSOR_MAX_DIMS) n = CONTINUOUS_TENSOR_MAX_DIMS;
                nb::list lst;
                for (uint32_t i = 0; i < n; ++i)
                    lst.append(self.shapes[i]);
                return nb::tuple(lst);
            },
            [](ContinuousTensor &self, nb::tuple t) {
                size_t n = nb::len(t);
                if (n > CONTINUOUS_TENSOR_MAX_DIMS)
                    throw std::invalid_argument(
                        "shapes tuple length exceeds CONTINUOUS_TENSOR_MAX_DIMS (" +
                        std::to_string(CONTINUOUS_TENSOR_MAX_DIMS) + ")"
                    );
                for (size_t i = 0; i < n; ++i)
                    self.shapes[i] = nb::cast<uint32_t>(t[i]);
                self.ndims = static_cast<uint32_t>(n);
            }
        )

        .def_prop_rw(
            "ndims",
            [](const ContinuousTensor &self) -> uint32_t {
                return self.ndims;
            },
            [](ContinuousTensor &self, uint32_t v) {
                self.ndims = v;
            }
        )

        .def_prop_rw(
            "dtype",
            [](const ContinuousTensor &self) -> DataType {
                return self.dtype;
            },
            [](ContinuousTensor &self, DataType dt) {
                self.dtype = dt;
            }
        )

        .def_prop_rw(
            "child_memory",
            [](const ContinuousTensor &self) -> bool {
                return self.is_child_memory();
            },
            [](ContinuousTensor &self, bool v) {
                self.child_memory = v ? 1 : 0;
            }
        )

        .def(
            "nbytes",
            [](const ContinuousTensor &self) -> uint64_t {
                return self.nbytes();
            },
            "Compute total bytes (product of shapes * element_size)."
        )

        .def("__repr__", [](const ContinuousTensor &self) -> std::string {
            std::ostringstream os;
            os << "ContinuousTensor(data=0x" << std::hex << self.data << std::dec << ", shapes=(";
            for (uint32_t i = 0; i < self.ndims; ++i) {
                if (i) os << ", ";
                os << self.shapes[i];
            }
            os << "), dtype=" << get_dtype_name(self.dtype);
            if (self.is_child_memory()) os << ", child_memory=True";
            os << ")";
            return os.str();
        });

    // --- ChipStorageTaskArgs (fixed-size TaskArgs) ---
    nb::class_<ChipStorageTaskArgs>(m, "ChipStorageTaskArgs")
        .def(nb::init<>())

        .def(
            "add_tensor", &ChipStorageTaskArgs::add_tensor, nb::arg("t"),
            "Add a ContinuousTensor. Must be called before any add_scalar()."
        )

        .def(
            "add_scalar", &ChipStorageTaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const ChipStorageTaskArgs &self, int32_t i) -> const ContinuousTensor & {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("ChipStorageTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the ContinuousTensor at index i."
        )

        .def(
            "scalar",
            [](const ChipStorageTaskArgs &self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count())
                    throw std::out_of_range("ChipStorageTaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"), "Return the scalar at index i."
        )

        .def("tensor_count", &ChipStorageTaskArgs::tensor_count)
        .def("scalar_count", &ChipStorageTaskArgs::scalar_count)

        .def("clear", &ChipStorageTaskArgs::clear)

        .def(
            "__len__",
            [](const ChipStorageTaskArgs &self) {
                return self.tensor_count() + self.scalar_count();
            },
            "Return total number of arguments (tensors + scalars)."
        )

        .def(
            "__ptr__",
            [](const ChipStorageTaskArgs &self) -> uint64_t {
                return reinterpret_cast<uint64_t>(&self);
            },
            "Return the memory address of the underlying C++ object."
        )

        .def_static(
            "sizeof",
            []() -> size_t {
                return sizeof(ChipStorageTaskArgs);
            },
            "Return sizeof(ChipStorageTaskArgs) in bytes."
        );

    // --- TensorArgType enum ---
    nb::enum_<TensorArgType>(m, "TensorArgType")
        .value("INPUT", TensorArgType::INPUT)
        .value("OUTPUT", TensorArgType::OUTPUT)
        .value("INOUT", TensorArgType::INOUT)
        .value("OUTPUT_EXISTING", TensorArgType::OUTPUT_EXISTING)
        .value("NO_DEP", TensorArgType::NO_DEP);

    // --- TaskArgs (unified vector-backed builder with per-tensor TensorArgType tags) ---
    nb::class_<TaskArgs>(m, "TaskArgs")
        .def(nb::init<>())

        .def(
            "add_tensor",
            [](TaskArgs &self, const ContinuousTensor &t, TensorArgType tag) {
                self.add_tensor(t, tag);
            },
            nb::arg("t"), nb::arg("tag") = TensorArgType::INPUT,
            "Add a ContinuousTensor with an optional TensorArgType tag (default INPUT)."
        )

        .def(
            "add_scalar", &TaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const TaskArgs &self, int32_t i) -> const ContinuousTensor & {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the ContinuousTensor at index i."
        )

        .def(
            "scalar",
            [](const TaskArgs &self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count()) throw std::out_of_range("TaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"), "Return the scalar at index i."
        )

        .def(
            "tag",
            [](const TaskArgs &self, int32_t i) -> TensorArgType {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tag index out of range");
                return self.tag(i);
            },
            nb::arg("i"), "Return the TensorArgType tag for the tensor at index i."
        )

        .def(
            "set_tag",
            [](TaskArgs &self, int32_t i, TensorArgType tag) {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs set_tag index out of range");
                self.tag(i) = tag;
            },
            nb::arg("i"), nb::arg("tag"), "Set the TensorArgType tag for the tensor at index i."
        )

        .def("tensor_count", &TaskArgs::tensor_count)
        .def("scalar_count", &TaskArgs::scalar_count)

        .def("clear", &TaskArgs::clear)

        .def(
            "__len__",
            [](const TaskArgs &self) {
                return self.tensor_count() + self.scalar_count();
            },
            "Return total number of arguments (tensors + scalars)."
        );

    // --- ArgDirection enum ---
    nb::enum_<ArgDirection>(m, "ArgDirection")
        .value("SCALAR", ArgDirection::SCALAR)
        .value("IN", ArgDirection::IN)
        .value("OUT", ArgDirection::OUT)
        .value("INOUT", ArgDirection::INOUT);

    m.def(
        "arg_direction_name",
        [](ArgDirection d) -> std::string {
            return arg_direction_name(d);
        },
        nb::arg("direction"), "Return the string name of an ArgDirection."
    );

    // --- PyCoreCallable wrapper ---
    struct PyCoreCallable {
        std::vector<uint8_t> buffer_;
        const CoreCallable &get() const { return *reinterpret_cast<const CoreCallable *>(buffer_.data()); }
    };

    nb::class_<PyCoreCallable>(m, "CoreCallable")
        .def_static(
            "build",
            [](std::vector<ArgDirection> signature, nb::bytes binary) -> PyCoreCallable {
                auto bin_ptr = reinterpret_cast<const void *>(binary.c_str());
                auto bin_size = static_cast<uint32_t>(binary.size());
                auto buf = make_callable<CORE_MAX_TENSOR_ARGS>(
                    signature.data(), static_cast<int32_t>(signature.size()), bin_ptr, bin_size
                );
                return PyCoreCallable{std::move(buf)};
            },
            nb::arg("signature"), nb::arg("binary"), "Build a CoreCallable from a signature list and binary bytes."
        )

        .def(
            "sig",
            [](const PyCoreCallable &self, int32_t i) -> ArgDirection {
                return self.get().sig(i);
            },
            nb::arg("i"), "Return the ArgDirection at signature index i."
        )

        .def_prop_ro(
            "sig_count",
            [](const PyCoreCallable &self) -> int32_t {
                return self.get().sig_count();
            },
            "Number of signature entries."
        )

        .def_prop_ro(
            "binary_size",
            [](const PyCoreCallable &self) -> uint32_t {
                return self.get().binary_size();
            },
            "Size of the binary payload in bytes."
        )

        .def(
            "buffer_ptr",
            [](const PyCoreCallable &self) -> uint64_t {
                return reinterpret_cast<uint64_t>(self.buffer_.data());
            },
            "Return the memory address of the underlying buffer."
        )

        .def(
            "buffer_size",
            [](const PyCoreCallable &self) -> size_t {
                return self.buffer_.size();
            },
            "Return the total size of the underlying buffer in bytes."
        )

        .def("__repr__", [](const PyCoreCallable &self) -> std::string {
            const auto &c = self.get();
            std::ostringstream os;
            os << "CoreCallable(sig_count=" << c.sig_count() << ", binary_size=" << c.binary_size() << ")";
            return os.str();
        });

    // --- PyChipCallable wrapper ---
    struct PyChipCallable {
        std::vector<uint8_t> buffer_;
        const ChipCallable &get() const { return *reinterpret_cast<const ChipCallable *>(buffer_.data()); }
    };

    nb::class_<PyChipCallable>(m, "ChipCallable")
        .def_static(
            "build",
            [](std::vector<ArgDirection> signature, std::string func_name, nb::bytes binary,
               std::vector<std::tuple<int32_t, PyCoreCallable>> children, std::string config_name) -> PyChipCallable {
                auto bin_ptr = reinterpret_cast<const void *>(binary.c_str());
                auto bin_size = static_cast<uint32_t>(binary.size());
                auto child_count = static_cast<int32_t>(children.size());

                std::vector<int32_t> func_ids(children.size());
                std::vector<std::vector<uint8_t>> child_bufs(children.size());
                for (size_t i = 0; i < children.size(); ++i) {
                    func_ids[i] = std::get<0>(children[i]);
                    child_bufs[i] = std::get<1>(children[i]).buffer_;
                }

                auto buf = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
                    signature.data(), static_cast<int32_t>(signature.size()), func_name.c_str(), bin_ptr, bin_size,
                    func_ids.data(), child_bufs.data(), child_count, config_name.c_str()
                );
                return PyChipCallable{std::move(buf)};
            },
            nb::arg("signature"), nb::arg("func_name"), nb::arg("binary"), nb::arg("children"),
            nb::arg("config_name") = "",
            "Build a ChipCallable from signature, func_name, binary, and list of (func_id, CoreCallable) children."
        )

        .def(
            "sig",
            [](const PyChipCallable &self, int32_t i) -> ArgDirection {
                return self.get().sig(i);
            },
            nb::arg("i"), "Return the ArgDirection at signature index i."
        )

        .def_prop_ro(
            "sig_count",
            [](const PyChipCallable &self) -> int32_t {
                return self.get().sig_count();
            },
            "Number of signature entries."
        )

        .def_prop_ro(
            "binary_size",
            [](const PyChipCallable &self) -> uint32_t {
                return self.get().binary_size();
            },
            "Size of the binary payload in bytes."
        )

        .def_prop_ro(
            "func_name",
            [](const PyChipCallable &self) -> std::string {
                const auto &c = self.get();
                return std::string(c.func_name(), c.func_name_len());
            },
            "The orchestration function name."
        )

        .def_prop_ro(
            "config_name",
            [](const PyChipCallable &self) -> std::string {
                const auto &c = self.get();
                return std::string(c.config_name(), c.config_name_len());
            },
            "The optional orchestration config function name."
        )

        .def_prop_ro(
            "child_count",
            [](const PyChipCallable &self) -> int32_t {
                return self.get().child_count();
            },
            "Number of child callables."
        )

        .def(
            "child_func_id",
            [](const PyChipCallable &self, int32_t i) -> int32_t {
                return self.get().child_func_id(i);
            },
            nb::arg("i"), "Return the func_id for child at index i."
        )

        .def(
            "child",
            [](const PyChipCallable &self, int32_t i) -> PyCoreCallable {
                const auto &parent = self.get();
                const auto &c = parent.child(i);
                // Reconstruct a PyCoreCallable by copying the child's raw bytes
                auto offset = parent.child_offset(i);
                const uint8_t *child_start = reinterpret_cast<const uint8_t *>(parent.storage_ + offset);
                // Determine child size: from offset to next child or end of buffer
                size_t child_size;
                if (i + 1 < parent.child_count()) {
                    child_size = parent.child_offset(i + 1) - offset;
                } else {
                    size_t header_size = offsetof(ChipCallable, storage_);
                    child_size = self.buffer_.size() - header_size - offset;
                }
                std::vector<uint8_t> child_buf(child_start, child_start + child_size);
                return PyCoreCallable{std::move(child_buf)};
            },
            nb::arg("i"), "Return the CoreCallable child at index i."
        )

        .def(
            "child_offset",
            [](const PyChipCallable &self, int32_t i) -> uint32_t {
                return self.get().child_offset(i);
            },
            nb::arg("i"), "Return the byte offset of child i within storage (must be multiple of 64)."
        )

        .def(
            "buffer_ptr",
            [](const PyChipCallable &self) -> uint64_t {
                return reinterpret_cast<uint64_t>(self.buffer_.data());
            },
            "Return the memory address of the underlying buffer."
        )

        .def(
            "buffer_size",
            [](const PyChipCallable &self) -> size_t {
                return self.buffer_.size();
            },
            "Return the total size of the underlying buffer in bytes."
        )

        .def("__repr__", [](const PyChipCallable &self) -> std::string {
            const auto &c = self.get();
            std::ostringstream os;
            os << "ChipCallable(func_name=\"" << std::string(c.func_name(), c.func_name_len()) << "\", config_name=\""
               << std::string(c.config_name(), c.config_name_len()) << "\", sig_count=" << c.sig_count()
               << ", binary_size=" << c.binary_size() << ", child_count=" << c.child_count() << ")";
            return os.str();
        });

    // --- CallConfig ---
    // The two enable_* fields are stored as int32 on the wire (see call_config.h).
    // Expose them as Python `bool` via def_property so user-facing API is unchanged.
    nb::class_<CallConfig>(m, "CallConfig")
        .def(nb::init<>())
        .def_rw("block_dim", &CallConfig::block_dim)
        .def_rw("aicpu_thread_num", &CallConfig::aicpu_thread_num)
        .def_prop_rw(
            "enable_l2_swimlane",
            [](const CallConfig &c) {
                return static_cast<bool>(c.enable_l2_swimlane);
            },
            [](CallConfig &c, bool v) {
                c.enable_l2_swimlane = v ? 1 : 0;
            }
        )
        .def_prop_rw(
            "enable_dump_tensor",
            [](const CallConfig &c) {
                return static_cast<bool>(c.enable_dump_tensor);
            },
            [](CallConfig &c, bool v) {
                c.enable_dump_tensor = v ? 1 : 0;
            }
        )
        .def_rw("enable_pmu", &CallConfig::enable_pmu)
        .def_prop_rw(
            "enable_dep_gen",
            [](const CallConfig &c) {
                return static_cast<bool>(c.enable_dep_gen);
            },
            [](CallConfig &c, bool v) {
                c.enable_dep_gen = v ? 1 : 0;
            }
        )
        .def_prop_rw(
            "output_prefix",
            [](const CallConfig &c) -> std::string {
                return std::string(c.output_prefix, ::strnlen(c.output_prefix, sizeof(c.output_prefix)));
            },
            [](CallConfig &c, const std::string &s) {
                if (s.size() >= sizeof(c.output_prefix)) {
                    throw std::invalid_argument(
                        "CallConfig.output_prefix length " + std::to_string(s.size()) + " exceeds buffer (" +
                        std::to_string(sizeof(c.output_prefix) - 1) + " bytes)"
                    );
                }
                std::memset(c.output_prefix, 0, sizeof(c.output_prefix));
                std::memcpy(c.output_prefix, s.data(), s.size());
            }
        )
        .def("__repr__", [](const CallConfig &self) -> std::string {
            std::ostringstream os;
            os << "CallConfig(block_dim=" << self.block_dim << ", aicpu_thread_num=" << self.aicpu_thread_num
               << ", enable_l2_swimlane=" << (self.enable_l2_swimlane ? "True" : "False")
               << ", enable_dump_tensor=" << (self.enable_dump_tensor ? "True" : "False")
               << ", enable_pmu=" << self.enable_pmu << ", enable_dep_gen=" << (self.enable_dep_gen ? "True" : "False");
            if (self.output_prefix_set()) {
                os << ", output_prefix='" << self.output_prefix << "'";
            }
            os << ")";
            return os.str();
        });

    // Log default constant — single source. Mirrored in
    // src/common/log/host_log.h::simpler::log::kDefaultThreshold; if you change
    // one, change the other.
    m.attr("DEFAULT_LOG_THRESHOLD") = 20;  // V5 = Python INFO

    // --- ChipWorker ---
    nb::class_<ChipWorker>(m, "_ChipWorker")
        .def(nb::init<>())
        .def(
            "init", &ChipWorker::init, nb::arg("host_lib_path"), nb::arg("aicpu_path"), nb::arg("aicore_path"),
            nb::arg("device_id")
        )
        .def("finalize", &ChipWorker::finalize)
        .def(
            "prepare_callable",
            [](ChipWorker &self, int32_t callable_id, const PyChipCallable &callable) {
                self.prepare_callable(callable_id, callable.buffer_.data());
            },
            nb::arg("callable_id"), nb::arg("callable"),
            "Stage a ChipCallable under callable_id for cheap repeated launches "
            "via run_prepared. Variants without per-callable_id support raise."
        )
        .def(
            "prepare_callable_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t blob_ptr) {
                self.prepare_callable(callable_id, reinterpret_cast<const void *>(blob_ptr));
            },
            nb::arg("callable_id"), nb::arg("blob_ptr"),
            "Stage a ChipCallable from a raw contiguous-buffer pointer (used by "
            "post-fork dynamic register handlers that receive the ChipCallable "
            "bytes via shared memory; see docs/callable-ipc-dynamic-register.md). "
            "Equivalent to prepare_callable(cid, ChipCallable) but accepts the "
            "ChipCallable layout pointer directly so chip-child loops can prepare "
            "from shm without rebuilding a PyChipCallable wrapper."
        )
        .def(
            "run_prepared",
            [](ChipWorker &self, int32_t callable_id, ChipStorageTaskArgs &args, const CallConfig &config) {
                self.run_prepared(callable_id, &args, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"),
            "Launch a callable_id previously staged via prepare_callable."
        )
        .def(
            "run_prepared",
            [](ChipWorker &self, int32_t callable_id, TaskArgs &args, const CallConfig &config) {
                TaskArgsView view = make_view(args);
                self.run_prepared(callable_id, view, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"),
            "Launch a callable_id from a TaskArgs (used for in-process callers)."
        )
        .def(
            "run_prepared_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t args_blob_ptr, size_t blob_capacity,
               const CallConfig &config) {
                // The mailbox region is the on-wire format `write_blob` produced;
                // `read_blob` is the matching reader that returns a zero-copy
                // TaskArgsView into the caller-owned bytes. Forwards to the
                // existing `run_prepared(cid, view, config)` path so chip-child
                // loops never re-implement the tensor/scalar layout in Python
                // (where it has historically dropped fields like child_memory).
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), blob_capacity);
                self.run_prepared(callable_id, view, config);
            },
            nb::arg("callable_id"), nb::arg("args_blob_ptr"), nb::arg("blob_capacity"), nb::arg("config"),
            "Launch a callable_id from a raw mailbox-blob pointer + capacity "
            "(used by chip-child mailbox loops to avoid Python-side re-deserialisation "
            "of the per-task tensor/scalar layout). The blob must be in the format "
            "produced by `write_blob`; read_blob enforces capacity bounds against shm corruption."
        )
        .def(
            "unregister_callable",
            [](ChipWorker &self, int32_t callable_id) {
                self.unregister_callable(callable_id);
            },
            nb::arg("callable_id"),
            "Drop the prepared state for callable_id; releases the per-id share "
            "of the device orch SO buffer (kernel binaries stay resident until "
            "finalize)."
        )
        .def_prop_ro("device_id", &ChipWorker::device_id)
        .def_prop_ro("initialized", &ChipWorker::initialized)
        .def_prop_ro(
            "aicpu_dlopen_count", &ChipWorker::aicpu_dlopen_count,
            "Number of distinct callable_ids the AICPU has dlopened for on the "
            "bound device. Equals 0 when not initialized or the runtime "
            "variant lacks per-cid registration. Tests assert this to verify "
            "prepare_callable + repeated run_prepared do not redundantly dlopen."
        )
        .def_prop_ro(
            "host_dlopen_count", &ChipWorker::host_dlopen_count,
            "Number of host-side dlopens triggered by prepare_callable on "
            "host_build_graph variants. Mirrors aicpu_dlopen_count for the "
            "host-orchestration path; 0 on device-orch variants."
        )
        .def("malloc", &ChipWorker::malloc, nb::arg("size"))
        .def("free", &ChipWorker::free, nb::arg("ptr"))
        .def("copy_to", &ChipWorker::copy_to, nb::arg("dst"), nb::arg("src"), nb::arg("size"))
        .def("copy_from", &ChipWorker::copy_from, nb::arg("dst"), nb::arg("src"), nb::arg("size"))
        .def(
            "comm_init", &ChipWorker::comm_init, nb::arg("rank"), nb::arg("nranks"), nb::arg("rootinfo_path"),
            "Initialize a communicator for this rank.  ChipWorker owns ACL + stream "
            "lifetime internally (onboard drives ensure_acl_ready + aclrtCreateStream; "
            "sim ignores both).  Pair with comm_destroy for cleanup."
        )
        .def(
            "comm_alloc_windows", &ChipWorker::comm_alloc_windows, nb::arg("comm_handle"), nb::arg("win_size"),
            "Allocate per-rank windows and return the device CommContext pointer."
        )
        .def(
            "comm_get_local_window_base", &ChipWorker::comm_get_local_window_base, nb::arg("comm_handle"),
            "Return this rank's local window base address."
        )
        .def(
            "comm_get_window_size", &ChipWorker::comm_get_window_size, nb::arg("comm_handle"),
            "Return the actual per-rank window size (may differ from the hint)."
        )
        .def("comm_barrier", &ChipWorker::comm_barrier, nb::arg("comm_handle"), "Synchronize all ranks.")
        .def(
            "comm_destroy", &ChipWorker::comm_destroy, nb::arg("comm_handle"),
            "Destroy the communicator and release its resources."
        );

    // --- Standalone blob helpers ---
    m.def(
        "read_args_from_blob",
        [](uint64_t blob_ptr) {
            TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(blob_ptr), MAILBOX_ARGS_CAPACITY);
            TaskArgs args;
            for (int32_t i = 0; i < view.tensor_count; i++) {
                args.add_tensor(view.tensors[i]);
            }
            for (int32_t i = 0; i < view.scalar_count; i++) {
                args.add_scalar(view.scalars[i]);
            }
            return args;
        },
        nb::arg("blob_ptr"),
        "Reconstruct a TaskArgs from a length-prefixed blob at blob_ptr. "
        "Tags are not preserved (blob wire format strips them)."
    );

    bind_worker(m);
}
