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
 * Wraps DataType, ContinuousTensor, ChipStorageTaskArgs, DynamicTaskArgs,
 * TaggedTaskArgs, TensorArgType, ArgDirection, CoreCallable, ChipCallable,
 * and helper functions from
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
#include "data_type.h"
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
        .value("UINT64", DataType::UINT64);

    // --- Free functions ---
    m.def("get_element_size",
        &get_element_size,
        nb::arg("dtype"),
        "Return the byte size of a single element of the given DataType.");

    m.def(
        "get_dtype_name",
        [](DataType dt) -> std::string { return get_dtype_name(dt); },
        nb::arg("dtype"),
        "Return the string name of a DataType.");

    // --- Constants ---
    m.attr("CONTINUOUS_TENSOR_MAX_DIMS") = CONTINUOUS_TENSOR_MAX_DIMS;

    // --- ContinuousTensor ---
    nb::class_<ContinuousTensor>(m, "ContinuousTensor")
        .def(nb::init<>())

        .def_static(
            "make",
            [](uint64_t data, nb::tuple shapes, DataType dtype) -> ContinuousTensor {
                size_t n = nb::len(shapes);
                if (n > CONTINUOUS_TENSOR_MAX_DIMS)
                    throw std::invalid_argument("shapes length exceeds CONTINUOUS_TENSOR_MAX_DIMS");
                ContinuousTensor arg{};
                arg.data = data;
                arg.dtype = dtype;
                arg.ndims = static_cast<uint32_t>(n);
                for (size_t i = 0; i < n; ++i) arg.shapes[i] = nb::cast<uint32_t>(shapes[i]);
                return arg;
            },
            nb::arg("data"),
            nb::arg("shapes"),
            nb::arg("dtype"),
            "Create a ContinuousTensor from a data pointer, shape tuple, and dtype.")

        .def_prop_rw(
            "data",
            [](const ContinuousTensor& self) -> uint64_t { return self.data; },
            [](ContinuousTensor& self, uint64_t v) { self.data = v; })

        .def_prop_rw(
            "shapes",
            [](const ContinuousTensor& self) -> nb::tuple {
                uint32_t n = self.ndims;
                if (n > CONTINUOUS_TENSOR_MAX_DIMS) n = CONTINUOUS_TENSOR_MAX_DIMS;
                nb::list lst;
                for (uint32_t i = 0; i < n; ++i) lst.append(self.shapes[i]);
                return nb::tuple(lst);
            },
            [](ContinuousTensor& self, nb::tuple t) {
                size_t n = nb::len(t);
                if (n > CONTINUOUS_TENSOR_MAX_DIMS)
                    throw std::invalid_argument("shapes tuple length exceeds CONTINUOUS_TENSOR_MAX_DIMS (" +
                                                std::to_string(CONTINUOUS_TENSOR_MAX_DIMS) + ")");
                for (size_t i = 0; i < n; ++i) self.shapes[i] = nb::cast<uint32_t>(t[i]);
                self.ndims = static_cast<uint32_t>(n);
            })

        .def_prop_rw(
            "ndims",
            [](const ContinuousTensor& self) -> uint32_t { return self.ndims; },
            [](ContinuousTensor& self, uint32_t v) { self.ndims = v; })

        .def_prop_rw(
            "dtype",
            [](const ContinuousTensor& self) -> DataType { return self.dtype; },
            [](ContinuousTensor& self, DataType dt) { self.dtype = dt; })

        .def(
            "nbytes",
            [](const ContinuousTensor& self) -> uint64_t { return self.nbytes(); },
            "Compute total bytes (product of shapes * element_size).")

        .def("__repr__", [](const ContinuousTensor& self) -> std::string {
            std::ostringstream os;
            os << "ContinuousTensor(data=0x" << std::hex << self.data << std::dec << ", shapes=(";
            for (uint32_t i = 0; i < self.ndims; ++i) {
                if (i) os << ", ";
                os << self.shapes[i];
            }
            os << "), dtype=" << get_dtype_name(self.dtype) << ")";
            return os.str();
        });

    // --- ChipStorageTaskArgs (fixed-size TaskArgs) ---
    nb::class_<ChipStorageTaskArgs>(m, "ChipStorageTaskArgs")
        .def(nb::init<>())

        .def("add_tensor",
            &ChipStorageTaskArgs::add_tensor,
            nb::arg("t"),
            "Add a ContinuousTensor. Must be called before any add_scalar().")

        .def("add_scalar",
            &ChipStorageTaskArgs::add_scalar,
            nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed.")

        .def(
            "tensor",
            [](const ChipStorageTaskArgs& self, int32_t i) -> const ContinuousTensor& {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("ChipStorageTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"),
            nb::rv_policy::reference_internal,
            "Return the ContinuousTensor at index i.")

        .def(
            "scalar",
            [](const ChipStorageTaskArgs& self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count())
                    throw std::out_of_range("ChipStorageTaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"),
            "Return the scalar at index i.")

        .def("tensor_count", &ChipStorageTaskArgs::tensor_count)
        .def("scalar_count", &ChipStorageTaskArgs::scalar_count)

        .def("clear", &ChipStorageTaskArgs::clear)

        .def(
            "__len__",
            [](const ChipStorageTaskArgs& self) { return self.tensor_count() + self.scalar_count(); },
            "Return total number of arguments (tensors + scalars).")

        .def(
            "__ptr__",
            [](const ChipStorageTaskArgs& self) -> uint64_t { return reinterpret_cast<uint64_t>(&self); },
            "Return the memory address of the underlying C++ object.");

    // --- TensorArgType enum ---
    nb::enum_<TensorArgType>(m, "TensorArgType")
        .value("INPUT", TensorArgType::INPUT)
        .value("OUTPUT", TensorArgType::OUTPUT)
        .value("INOUT", TensorArgType::INOUT);

    // --- DynamicTaskArgs (vector-backed, no capacity limit) ---
    nb::class_<DynamicTaskArgs>(m, "DynamicTaskArgs")
        .def(nb::init<>())

        .def("add_tensor",
            &DynamicTaskArgs::add_tensor,
            nb::arg("t"),
            "Add a ContinuousTensor. Must be called before any add_scalar().")

        .def("add_scalar",
            &DynamicTaskArgs::add_scalar,
            nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed.")

        .def(
            "tensor",
            [](const DynamicTaskArgs& self, int32_t i) -> const ContinuousTensor& {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("DynamicTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"),
            nb::rv_policy::reference_internal,
            "Return the ContinuousTensor at index i.")

        .def(
            "scalar",
            [](const DynamicTaskArgs& self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count())
                    throw std::out_of_range("DynamicTaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"),
            "Return the scalar at index i.")

        .def("tensor_count", &DynamicTaskArgs::tensor_count)
        .def("scalar_count", &DynamicTaskArgs::scalar_count)

        .def("clear", &DynamicTaskArgs::clear)

        .def(
            "__len__",
            [](const DynamicTaskArgs& self) { return self.tensor_count() + self.scalar_count(); },
            "Return total number of arguments (tensors + scalars).");

    // --- TaggedTaskArgs (fixed-size with per-tensor TensorArgType tags) ---
    nb::class_<TaggedTaskArgs>(m, "TaggedTaskArgs")
        .def(nb::init<>())

        .def(
            "add_tensor",
            [](TaggedTaskArgs& self, const ContinuousTensor& t, TensorArgType tag) {
                self.add_tensor(t);
                self.tag(self.tensor_count() - 1) = tag;
            },
            nb::arg("t"),
            nb::arg("tag") = TensorArgType::INPUT,
            "Add a ContinuousTensor with an optional TensorArgType tag (default INPUT).")

        .def("add_scalar",
            &TaggedTaskArgs::add_scalar,
            nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed.")

        .def(
            "tensor",
            [](const TaggedTaskArgs& self, int32_t i) -> const ContinuousTensor& {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("TaggedTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"),
            nb::rv_policy::reference_internal,
            "Return the ContinuousTensor at index i.")

        .def(
            "scalar",
            [](const TaggedTaskArgs& self, int32_t i) -> uint64_t {
                if (i < 0 || i >= self.scalar_count())
                    throw std::out_of_range("TaggedTaskArgs scalar index out of range");
                return self.scalar(i);
            },
            nb::arg("i"),
            "Return the scalar at index i.")

        .def(
            "tag",
            [](const TaggedTaskArgs& self, int32_t i) -> TensorArgType {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaggedTaskArgs tag index out of range");
                return self.tag(i);
            },
            nb::arg("i"),
            "Return the TensorArgType tag for the tensor at index i.")

        .def(
            "set_tag",
            [](TaggedTaskArgs& self, int32_t i, TensorArgType tag) {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("TaggedTaskArgs set_tag index out of range");
                self.tag(i) = tag;
            },
            nb::arg("i"),
            nb::arg("tag"),
            "Set the TensorArgType tag for the tensor at index i.")

        .def("tensor_count", &TaggedTaskArgs::tensor_count)
        .def("scalar_count", &TaggedTaskArgs::scalar_count)

        .def("clear", &TaggedTaskArgs::clear)

        .def(
            "__len__",
            [](const TaggedTaskArgs& self) { return self.tensor_count() + self.scalar_count(); },
            "Return total number of arguments (tensors + scalars).");

    // --- ArgDirection enum ---
    nb::enum_<ArgDirection>(m, "ArgDirection")
        .value("SCALAR", ArgDirection::SCALAR)
        .value("IN", ArgDirection::IN)
        .value("OUT", ArgDirection::OUT)
        .value("INOUT", ArgDirection::INOUT);

    m.def(
        "arg_direction_name",
        [](ArgDirection d) -> std::string { return arg_direction_name(d); },
        nb::arg("direction"),
        "Return the string name of an ArgDirection.");

    // --- PyCoreCallable wrapper ---
    struct PyCoreCallable {
        std::vector<uint8_t> buffer_;
        const CoreCallable& get() const { return *reinterpret_cast<const CoreCallable*>(buffer_.data()); }
    };

    nb::class_<PyCoreCallable>(m, "CoreCallable")
        .def_static(
            "build",
            [](std::vector<ArgDirection> signature, nb::bytes binary) -> PyCoreCallable {
                auto bin_ptr = reinterpret_cast<const void*>(binary.c_str());
                auto bin_size = static_cast<uint32_t>(binary.size());
                auto buf = make_callable<CORE_MAX_TENSOR_ARGS>(
                    signature.data(), static_cast<int32_t>(signature.size()), bin_ptr, bin_size);
                return PyCoreCallable{std::move(buf)};
            },
            nb::arg("signature"),
            nb::arg("binary"),
            "Build a CoreCallable from a signature list and binary bytes.")

        .def(
            "sig",
            [](const PyCoreCallable& self, int32_t i) -> ArgDirection { return self.get().sig(i); },
            nb::arg("i"),
            "Return the ArgDirection at signature index i.")

        .def_prop_ro(
            "sig_count",
            [](const PyCoreCallable& self) -> int32_t { return self.get().sig_count(); },
            "Number of signature entries.")

        .def_prop_ro(
            "binary_size",
            [](const PyCoreCallable& self) -> uint32_t { return self.get().binary_size(); },
            "Size of the binary payload in bytes.")

        .def(
            "buffer_ptr",
            [](const PyCoreCallable& self) -> uint64_t { return reinterpret_cast<uint64_t>(self.buffer_.data()); },
            "Return the memory address of the underlying buffer.")

        .def(
            "buffer_size",
            [](const PyCoreCallable& self) -> size_t { return self.buffer_.size(); },
            "Return the total size of the underlying buffer in bytes.")

        .def("__repr__", [](const PyCoreCallable& self) -> std::string {
            const auto& c = self.get();
            std::ostringstream os;
            os << "CoreCallable(sig_count=" << c.sig_count() << ", binary_size=" << c.binary_size() << ")";
            return os.str();
        });

    // --- PyChipCallable wrapper ---
    struct PyChipCallable {
        std::vector<uint8_t> buffer_;
        const ChipCallable& get() const { return *reinterpret_cast<const ChipCallable*>(buffer_.data()); }
    };

    nb::class_<PyChipCallable>(m, "ChipCallable")
        .def_static(
            "build",
            [](std::vector<ArgDirection> signature,
                std::string func_name,
                nb::bytes binary,
                std::vector<std::tuple<int32_t, PyCoreCallable>> children) -> PyChipCallable {
                auto bin_ptr = reinterpret_cast<const void*>(binary.c_str());
                auto bin_size = static_cast<uint32_t>(binary.size());
                auto child_count = static_cast<int32_t>(children.size());

                std::vector<int32_t> func_ids(children.size());
                std::vector<std::vector<uint8_t>> child_bufs(children.size());
                for (size_t i = 0; i < children.size(); ++i) {
                    func_ids[i] = std::get<0>(children[i]);
                    child_bufs[i] = std::get<1>(children[i]).buffer_;
                }

                auto buf = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 32>(signature.data(),
                    static_cast<int32_t>(signature.size()),
                    func_name.c_str(),
                    bin_ptr,
                    bin_size,
                    func_ids.data(),
                    child_bufs.data(),
                    child_count);
                return PyChipCallable{std::move(buf)};
            },
            nb::arg("signature"),
            nb::arg("func_name"),
            nb::arg("binary"),
            nb::arg("children"),
            "Build a ChipCallable from signature, func_name, binary, and list of (func_id, CoreCallable) children.")

        .def(
            "sig",
            [](const PyChipCallable& self, int32_t i) -> ArgDirection { return self.get().sig(i); },
            nb::arg("i"),
            "Return the ArgDirection at signature index i.")

        .def_prop_ro(
            "sig_count",
            [](const PyChipCallable& self) -> int32_t { return self.get().sig_count(); },
            "Number of signature entries.")

        .def_prop_ro(
            "binary_size",
            [](const PyChipCallable& self) -> uint32_t { return self.get().binary_size(); },
            "Size of the binary payload in bytes.")

        .def_prop_ro(
            "func_name",
            [](const PyChipCallable& self) -> std::string {
                const auto& c = self.get();
                return std::string(c.func_name(), c.func_name_len());
            },
            "The orchestration function name.")

        .def_prop_ro(
            "child_count",
            [](const PyChipCallable& self) -> int32_t { return self.get().child_count(); },
            "Number of child callables.")

        .def(
            "child_func_id",
            [](const PyChipCallable& self, int32_t i) -> int32_t { return self.get().child_func_id(i); },
            nb::arg("i"),
            "Return the func_id for child at index i.")

        .def(
            "child",
            [](const PyChipCallable& self, int32_t i) -> PyCoreCallable {
                const auto& parent = self.get();
                const auto& c = parent.child(i);
                // Reconstruct a PyCoreCallable by copying the child's raw bytes
                auto offset = parent.child_offset(i);
                const uint8_t* child_start = reinterpret_cast<const uint8_t*>(parent.storage_ + offset);
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
            nb::arg("i"),
            "Return the CoreCallable child at index i.")

        .def(
            "child_offset",
            [](const PyChipCallable& self, int32_t i) -> uint32_t { return self.get().child_offset(i); },
            nb::arg("i"),
            "Return the byte offset of child i within storage (must be multiple of 64).")

        .def(
            "buffer_ptr",
            [](const PyChipCallable& self) -> uint64_t { return reinterpret_cast<uint64_t>(self.buffer_.data()); },
            "Return the memory address of the underlying buffer.")

        .def(
            "buffer_size",
            [](const PyChipCallable& self) -> size_t { return self.buffer_.size(); },
            "Return the total size of the underlying buffer in bytes.")

        .def("__repr__", [](const PyChipCallable& self) -> std::string {
            const auto& c = self.get();
            std::ostringstream os;
            os << "ChipCallable(func_name=\"" << std::string(c.func_name(), c.func_name_len())
               << "\", sig_count=" << c.sig_count() << ", binary_size=" << c.binary_size()
               << ", child_count=" << c.child_count() << ")";
            return os.str();
        });
}
