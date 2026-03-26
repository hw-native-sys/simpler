/**
 * Nanobind Python extension for task_interface headers.
 *
 * Wraps DataType, TaskArgKind, TaskArg, and helper functions from
 * data_type.h / task_arg.h so Python can use the canonical C++ types
 * instead of maintaining parallel ctypes definitions.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include "data_type.h"
#include "task_arg.h"
#include "task_arg_array.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;

// ============================================================================
// Module definition
// ============================================================================

NB_MODULE(_task_interface, m) {
    m.doc() = "Nanobind bindings for task_interface (DataType, TaskArg)";

    // --- DataType enum ---
    nb::enum_<DataType>(m, "DataType")
        .value("FLOAT32",  DataType::FLOAT32)
        .value("FLOAT16",  DataType::FLOAT16)
        .value("INT32",    DataType::INT32)
        .value("INT16",    DataType::INT16)
        .value("INT8",     DataType::INT8)
        .value("UINT8",    DataType::UINT8)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("INT64",    DataType::INT64)
        .value("UINT64",   DataType::UINT64);

    // --- TaskArgKind enum ---
    nb::enum_<TaskArgKind>(m, "TaskArgKind")
        .value("TENSOR", TaskArgKind::TENSOR)
        .value("SCALAR", TaskArgKind::SCALAR);

    // --- Constants ---
    m.attr("TASK_ARG_MAX_DIMS") = TASK_ARG_MAX_DIMS;

    // --- Free functions ---
    m.def("get_element_size", &get_element_size,
          nb::arg("dtype"),
          "Return the byte size of a single element of the given DataType.");

    m.def("get_dtype_name",
          [](DataType dt) -> std::string { return get_dtype_name(dt); },
          nb::arg("dtype"),
          "Return the string name of a DataType.");

    // --- TaskArg ---
    nb::class_<TaskArg>(m, "TaskArg")
        .def(nb::init<>())

        // kind
        .def_prop_rw("kind",
            [](const TaskArg& self) { return self.kind; },
            [](TaskArg& self, TaskArgKind k) { self.kind = k; })

        // scalar (union field)
        .def_prop_rw("scalar",
            [](const TaskArg& self) -> uint64_t { return self.scalar; },
            [](TaskArg& self, uint64_t v) { self.scalar = v; })

        // tensor.data
        .def_prop_rw("tensor_data",
            [](const TaskArg& self) -> uint64_t { return self.tensor.data; },
            [](TaskArg& self, uint64_t v) { self.tensor.data = v; })

        // tensor.shapes — getter returns tuple[:ndims], setter writes + updates ndims
        .def_prop_rw("tensor_shapes",
            [](const TaskArg& self) -> nb::tuple {
                uint32_t n = self.tensor.ndims;
                if (n > TASK_ARG_MAX_DIMS) n = TASK_ARG_MAX_DIMS;
                nb::list lst;
                for (uint32_t i = 0; i < n; ++i)
                    lst.append(self.tensor.shapes[i]);
                return nb::tuple(lst);
            },
            [](TaskArg& self, nb::tuple t) {
                size_t n = nb::len(t);
                if (n > TASK_ARG_MAX_DIMS)
                    throw std::invalid_argument(
                        "shapes tuple length exceeds TASK_ARG_MAX_DIMS ("
                        + std::to_string(TASK_ARG_MAX_DIMS) + ")");
                for (size_t i = 0; i < n; ++i)
                    self.tensor.shapes[i] = nb::cast<uint32_t>(t[i]);
                self.tensor.ndims = static_cast<uint32_t>(n);
            })

        // tensor.ndims
        .def_prop_rw("tensor_ndims",
            [](const TaskArg& self) -> uint32_t { return self.tensor.ndims; },
            [](TaskArg& self, uint32_t v) { self.tensor.ndims = v; })

        // tensor.dtype
        .def_prop_rw("tensor_dtype",
            [](const TaskArg& self) -> DataType { return self.tensor.dtype; },
            [](TaskArg& self, DataType dt) { self.tensor.dtype = dt; })

        // set_tensor_shape(dim, val)
        .def("set_tensor_shape",
            [](TaskArg& self, uint32_t dim, uint32_t val) {
                if (dim >= TASK_ARG_MAX_DIMS)
                    throw std::out_of_range("dim >= TASK_ARG_MAX_DIMS");
                self.tensor.shapes[dim] = val;
            },
            nb::arg("dim"), nb::arg("val"))

        // nbytes()
        .def("nbytes",
            [](const TaskArg& self) -> uint64_t { return self.nbytes(); },
            "Compute total bytes for this tensor (product of shapes * element_size).")

        // Static factory: make_tensor
        .def_static("make_tensor",
            [](uint64_t data, nb::tuple shapes, DataType dtype) -> TaskArg {
                size_t n = nb::len(shapes);
                if (n > TASK_ARG_MAX_DIMS)
                    throw std::invalid_argument(
                        "shapes length exceeds TASK_ARG_MAX_DIMS");
                TaskArg arg{};
                arg.kind = TaskArgKind::TENSOR;
                arg.tensor.data = data;
                arg.tensor.dtype = dtype;
                arg.tensor.ndims = static_cast<uint32_t>(n);
                for (size_t i = 0; i < n; ++i)
                    arg.tensor.shapes[i] = nb::cast<uint32_t>(shapes[i]);
                return arg;
            },
            nb::arg("data"), nb::arg("shapes"), nb::arg("dtype"),
            "Create a TENSOR TaskArg from a data pointer, shape tuple, and dtype.")

        // Static factory: make_scalar
        .def_static("make_scalar",
            [](uint64_t value) -> TaskArg {
                TaskArg arg{};
                arg.kind = TaskArgKind::SCALAR;
                arg.scalar = value;
                return arg;
            },
            nb::arg("value"),
            "Create a SCALAR TaskArg with the given uint64 value.")

        // __repr__
        .def("__repr__",
            [](const TaskArg& self) -> std::string {
                std::ostringstream os;
                if (self.kind == TaskArgKind::TENSOR) {
                    os << "TaskArg(TENSOR, data=0x"
                       << std::hex << self.tensor.data << std::dec
                       << ", shapes=(";
                    for (uint32_t i = 0; i < self.tensor.ndims; ++i) {
                        if (i) os << ", ";
                        os << self.tensor.shapes[i];
                    }
                    os << "), dtype=" << get_dtype_name(self.tensor.dtype)
                       << ")";
                } else {
                    os << "TaskArg(SCALAR, value=" << self.scalar << ")";
                }
                return os.str();
            });

    // --- TaskArgArray ---
    nb::class_<TaskArgArray>(m, "TaskArgArray")
        .def(nb::init<>())

        .def("append", &TaskArgArray::append, nb::arg("arg"),
             "Append a TaskArg to the array.")

        .def("clear", &TaskArgArray::clear,
             "Remove all elements.")

        .def("__len__", &TaskArgArray::size)

        .def("__getitem__",
            static_cast<TaskArg& (TaskArgArray::*)(size_t)>(&TaskArgArray::get),
            nb::arg("idx"),
            nb::rv_policy::reference_internal,
            "Return a reference to the TaskArg at the given index.")

        .def("ctypes_ptr", &TaskArgArray::data_ptr,
             "Return the raw memory address of the contiguous TaskArg array "
             "(for passing to ctypes CDLL calls).");
}
