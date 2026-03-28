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
 * TaggedTaskArgs, TensorArgType, and helper functions from
 * data_type.h / tensor_arg.h / separated_args.h.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include "data_type.h"       // NOLINT(build/include_subdir)
#include "separated_args.h"  // NOLINT(build/include_subdir)
#include "tensor_arg.h"      // NOLINT(build/include_subdir)

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
            "Return total number of arguments (tensors + scalars).");

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
                self.tensor_tag(self.tensor_count() - 1) = tag;
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
            "tensor_tag",
            [](const TaggedTaskArgs& self, int32_t i) -> TensorArgType {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("TaggedTaskArgs tensor_tag index out of range");
                return self.tensor_tag(i);
            },
            nb::arg("i"),
            "Return the TensorArgType tag for the tensor at index i.")

        .def(
            "set_tensor_tag",
            [](TaggedTaskArgs& self, int32_t i, TensorArgType tag) {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("TaggedTaskArgs set_tensor_tag index out of range");
                self.tensor_tag(i) = tag;
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
}
