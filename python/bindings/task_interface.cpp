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
 * Wraps DataType, Tensor, ChipStorageTaskArgs, TaskArgs (unified
 * vector-backed builder with per-tensor TensorArgType tags), TensorArgType,
 * ArgDirection, CoreCallable, ChipCallable, and helper functions from
 * data_type.h / tensor.h / task_args.h / arg_direction.h / callable.h.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
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
#include "tensor.h"

namespace nb = nanobind;

namespace {

struct L3HostMappedSimRegion {
    int fd{-1};
    void *base{nullptr};
    size_t size{0};
};

std::mutex g_l3_host_mapped_region_mu;
std::unordered_map<uint64_t, L3HostMappedSimRegion> g_l3_host_mapped_regions;
uint64_t g_next_l3_host_mapped_region_handle = 1;

L3HostMappedSimRegion &l3_host_mapped_region(uint64_t handle) {
    std::lock_guard<std::mutex> lk(g_l3_host_mapped_region_mu);
    auto it = g_l3_host_mapped_regions.find(handle);
    if (it == g_l3_host_mapped_regions.end()) {
        throw std::runtime_error("L3-L2 L3 Host mapped-region handle is closed or unknown");
    }
    return it->second;
}

void validate_mapping_range(const L3HostMappedSimRegion &mapping, uint64_t offset, uint64_t nbytes) {
    if (nbytes == 0 || offset > mapping.size || nbytes > mapping.size - static_cast<size_t>(offset)) {
        throw std::out_of_range("L3-L2 L3 Host mapped-region access is out of range");
    }
}

int32_t load_counter(const L3HostMappedSimRegion &mapping, uint64_t offset) {
    validate_mapping_range(mapping, offset, sizeof(int32_t));
    int32_t value = 0;
    std::memcpy(&value, static_cast<const uint8_t *>(mapping.base) + offset, sizeof(value));
    return value;
}

void store_counter(const L3HostMappedSimRegion &mapping, uint64_t offset, int32_t value) {
    validate_mapping_range(mapping, offset, sizeof(int32_t));
    std::memcpy(static_cast<uint8_t *>(mapping.base) + offset, &value, sizeof(value));
}

bool compare_counter(int32_t observed, int32_t operand, int cmp) {
    switch (cmp) {
    case 0:
        return observed == operand;
    case 1:
        return observed != operand;
    case 2:
        return observed > operand;
    case 3:
        return observed >= operand;
    case 4:
        return observed < operand;
    case 5:
        return observed <= operand;
    default:
        return false;
    }
}

std::string shm_name_for_open(const std::string &token) {
    if (token.empty()) {
        throw std::invalid_argument("L3-L2 sim backing shm token must be non-empty");
    }
    if (token[0] == '/') {
        return token;
    }
    return "/" + token;
}

}  // namespace

// ============================================================================
// Module definition
// ============================================================================

NB_MODULE(_task_interface, m) {
    m.doc() = "Nanobind bindings for task_interface (DataType, Tensor, TaskArgs variants)";

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
    m.attr("MAX_TENSOR_DIMS") = MAX_TENSOR_DIMS;
    m.attr("MAX_REGISTERED_CALLABLE_IDS") = MAX_REGISTERED_CALLABLE_IDS;
    m.attr("RUNTIME_ENV_RING_COUNT") = RUNTIME_ENV_RING_COUNT;

    // --- Tensor ---
    // The unified strided tensor descriptor. Constructed contiguous via make()
    // (row-major strides, start_offset == 0); see src/common/task_interface/tensor.h.
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<>())

        .def_static(
            "make",
            [](uint64_t data, nb::tuple shapes, DataType dtype, bool child_memory) -> Tensor {
                size_t n = nb::len(shapes);
                if (n == 0 || n > MAX_TENSOR_DIMS)
                    throw std::invalid_argument("Tensor.make: shapes length must be in [1, MAX_TENSOR_DIMS]");
                uint32_t shp[MAX_TENSOR_DIMS];
                for (size_t i = 0; i < n; ++i)
                    shp[i] = nb::cast<uint32_t>(shapes[i]);
                // make_tensor_external yields a contiguous Tensor: row-major strides,
                // start_offset == 0, buffer.size == numel * element_size.
                return make_tensor_external(
                    reinterpret_cast<void *>(static_cast<uintptr_t>(data)), shp, static_cast<uint32_t>(n), dtype,
                    /*manual_dep=*/false, /*version=*/0, child_memory ? 1 : 0
                );
            },
            nb::arg("data"), nb::arg("shapes"), nb::arg("dtype"), nb::arg("child_memory") = false,
            "Create a contiguous Tensor over pre-allocated memory. Set child_memory=True when "
            "data is a device pointer allocated by the child process (skips H2D copy in "
            "init_runtime_impl)."
        )

        // `data` is the tensor's memory address — i.e. Tensor::buffer.addr.
        .def_prop_rw(
            "data",
            [](const Tensor &self) -> uint64_t {
                return self.buffer.addr;
            },
            [](Tensor &self, uint64_t v) {
                self.buffer.addr = v;
            }
        )

        .def_prop_rw(
            "shapes",
            [](const Tensor &self) -> nb::tuple {
                uint32_t n = self.ndims;
                if (n > MAX_TENSOR_DIMS) n = MAX_TENSOR_DIMS;
                nb::list lst;
                for (uint32_t i = 0; i < n; ++i)
                    lst.append(self.shapes[i]);
                return nb::tuple(lst);
            },
            [](Tensor &self, nb::tuple t) {
                size_t n = nb::len(t);
                if (n == 0 || n > MAX_TENSOR_DIMS)
                    throw std::invalid_argument(
                        "shapes tuple length must be in [1, MAX_TENSOR_DIMS] (" + std::to_string(MAX_TENSOR_DIMS) + ")"
                    );
                uint32_t shp[MAX_TENSOR_DIMS];
                for (size_t i = 0; i < n; ++i)
                    shp[i] = nb::cast<uint32_t>(t[i]);
                uint64_t numel = 1;
                for (size_t i = 0; i < n; ++i)
                    numel *= shp[i];
                // Re-establish a contiguous layout over the same buffer base.
                self.init_external(
                    reinterpret_cast<void *>(self.buffer.addr), numel * get_element_size(self.dtype), shp,
                    static_cast<uint32_t>(n), self.dtype, self.version, self.manual_dep, self.child_memory
                );
            }
        )

        // Read-only: a raw `ndims` write would desync shapes/strides/buffer.size
        // and could index past the fixed MAX_TENSOR_DIMS arrays. Rank changes go
        // through the `shapes` setter, which rebuilds a valid contiguous layout.
        .def_prop_ro(
            "ndims",
            [](const Tensor &self) -> uint32_t {
                return self.ndims;
            }
        )

        .def_prop_rw(
            "dtype",
            [](const Tensor &self) -> DataType {
                return self.dtype;
            },
            [](Tensor &self, DataType dt) {
                self.dtype = dt;
                self.buffer.size = self.numel() * get_element_size(dt);
            }
        )

        .def_prop_rw(
            "child_memory",
            [](const Tensor &self) -> bool {
                return self.is_child_memory();
            },
            [](Tensor &self, bool v) {
                self.child_memory = v ? 1 : 0;
            }
        )

        // Read-only views of the strided metadata (always contiguous for make()).
        .def_prop_ro(
            "strides",
            [](const Tensor &self) -> nb::tuple {
                nb::list lst;
                for (uint32_t i = 0; i < self.ndims && i < MAX_TENSOR_DIMS; ++i)
                    lst.append(self.strides[i]);
                return nb::tuple(lst);
            }
        )
        .def_prop_ro(
            "start_offset",
            [](const Tensor &self) -> uint64_t {
                return self.start_offset;
            }
        )
        .def_prop_ro(
            "is_contiguous",
            [](const Tensor &self) -> bool {
                return self.is_contiguous;
            }
        )

        .def(
            "nbytes",
            [](const Tensor &self) -> uint64_t {
                return self.nbytes();
            },
            "Compute total bytes (product of shapes * element_size)."
        )

        .def("__repr__", [](const Tensor &self) -> std::string {
            std::ostringstream os;
            os << "Tensor(data=0x" << std::hex << self.buffer.addr << std::dec << ", shapes=(";
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
            "Add a Tensor. Must be called before any add_scalar()."
        )

        .def(
            "add_scalar", &ChipStorageTaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const ChipStorageTaskArgs &self, int32_t i) -> const Tensor & {
                if (i < 0 || i >= self.tensor_count())
                    throw std::out_of_range("ChipStorageTaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the Tensor at index i."
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
    nb::class_<TaskArgs>(m, "TaskArgs", nb::is_weak_referenceable())
        .def(nb::init<>())

        .def(
            "add_tensor",
            [](TaskArgs &self, const Tensor &t, TensorArgType tag) {
                self.add_tensor(t, tag);
            },
            nb::arg("t"), nb::arg("tag") = TensorArgType::INPUT,
            "Add a Tensor with an optional TensorArgType tag (default INPUT)."
        )

        .def(
            "add_scalar", &TaskArgs::add_scalar, nb::arg("s"),
            "Add a uint64_t scalar. After this, add_tensor() is no longer allowed."
        )

        .def(
            "tensor",
            [](const TaskArgs &self, int32_t i) -> const Tensor & {
                if (i < 0 || i >= self.tensor_count()) throw std::out_of_range("TaskArgs tensor index out of range");
                return self.tensor(i);
            },
            nb::arg("i"), nb::rv_policy::reference_internal, "Return the Tensor at index i."
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
            nb::arg("signature"), nb::arg("binary"),
            "Build a CoreCallable from a signature list and binary bytes. The dump "
            "maps signature entry i to payload slot i positionally."
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

        .def_static(
            "from_bytes",
            [](nb::bytes raw) -> PyChipCallable {
                // Reconstruct a ChipCallable wrapper from the contiguous
                // serialised representation produced by `buffer_ptr()` /
                // `buffer_size()`. Used by the L4 cascade in
                // _child_worker_loop, which receives CTRL_REGISTER bytes
                // through shared memory and needs a typed ChipCallable for
                // digest-owned registration on the child Worker; see
                // docs/callable-identity-registration.md.
                std::vector<uint8_t> buf(
                    reinterpret_cast<const uint8_t *>(raw.c_str()),
                    reinterpret_cast<const uint8_t *>(raw.c_str()) + raw.size()
                );
                return PyChipCallable{std::move(buf)};
            },
            nb::arg("raw"),
            "Reconstruct a ChipCallable from the contiguous bytes that "
            "buffer_ptr() points to (size buffer_size()). Inverse of the "
            "serialisation used to ship a ChipCallable across the L4 "
            "cascade IPC channel."
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

    // --- RuntimeEnv (per-task PTO2_RING_* overrides; nested under CallConfig.runtime_env) ---
    // Each ring resource is exposed as ONE property that accepts either an int
    // (broadcast to every ring) or a list of RUNTIME_ENV_RING_COUNT ints
    // (per-ring). The value always reads back as a list — the wire layout is the
    // four-entry array, so a broadcast scalar is stored as [v, v, v, v].
    auto get_ring_values = [](const uint64_t values[RUNTIME_ENV_RING_COUNT]) -> std::vector<uint64_t> {
        std::vector<uint64_t> out;
        out.reserve(RUNTIME_ENV_RING_COUNT);
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            out.push_back(values[i]);
        }
        return out;
    };
    auto set_ring_values = [](uint64_t values[RUNTIME_ENV_RING_COUNT], nb::handle obj, const char *name) {
        uint64_t scalar = 0;
        if (nb::try_cast<uint64_t>(obj, scalar)) {  // int -> broadcast to every ring
            for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
                values[i] = scalar;
            }
            return;
        }
        std::vector<uint64_t> input;
        if (nb::try_cast<std::vector<uint64_t>>(obj, input)) {  // list -> per-ring
            if (input.size() != RUNTIME_ENV_RING_COUNT) {
                throw std::invalid_argument(
                    std::string("RuntimeEnv.") + name + " list must contain exactly " +
                    std::to_string(RUNTIME_ENV_RING_COUNT) + " entries"
                );
            }
            for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
                values[i] = input[static_cast<size_t>(i)];
            }
            return;
        }
        throw std::invalid_argument(
            std::string("RuntimeEnv.") + name + " must be an int (broadcast) or a list of " +
            std::to_string(RUNTIME_ENV_RING_COUNT) + " ints"
        );
    };
    auto append_ring_values = [](std::ostringstream &os, const char *name, bool leading_comma,
                                 const uint64_t values[RUNTIME_ENV_RING_COUNT]) {
        if (leading_comma) {
            os << ", ";
        }
        os << name << "=[";
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << values[i];
        }
        os << "]";
    };

    nb::class_<RuntimeEnv>(m, "RuntimeEnv")
        .def(nb::init<>())
        .def_prop_rw(
            "ring_task_window",
            [get_ring_values](const RuntimeEnv &self) {
                return get_ring_values(self.ring_task_window);
            },
            [set_ring_values](RuntimeEnv &self, nb::handle value) {
                set_ring_values(self.ring_task_window, value, "ring_task_window");
            }
        )
        .def_prop_rw(
            "ring_heap",
            [get_ring_values](const RuntimeEnv &self) {
                return get_ring_values(self.ring_heap);
            },
            [set_ring_values](RuntimeEnv &self, nb::handle value) {
                set_ring_values(self.ring_heap, value, "ring_heap");
            }
        )
        .def_prop_rw(
            "ring_dep_pool",
            [get_ring_values](const RuntimeEnv &self) {
                return get_ring_values(self.ring_dep_pool);
            },
            [set_ring_values](RuntimeEnv &self, nb::handle value) {
                set_ring_values(self.ring_dep_pool, value, "ring_dep_pool");
            }
        )
        .def("__repr__", [append_ring_values](const RuntimeEnv &self) -> std::string {
            std::ostringstream os;
            os << "RuntimeEnv(";
            append_ring_values(os, "ring_task_window", false, self.ring_task_window);
            append_ring_values(os, "ring_heap", true, self.ring_heap);
            append_ring_values(os, "ring_dep_pool", true, self.ring_dep_pool);
            os << ")";
            return os.str();
        });

    // --- CallConfig ---
    nb::class_<CallConfig>(m, "CallConfig")
        .def(nb::init<>())
        .def_rw("block_dim", &CallConfig::block_dim)
        .def_rw("aicpu_thread_num", &CallConfig::aicpu_thread_num)
        // runtime_env returns an internal reference so `cfg.runtime_env.ring_heap = X`
        // writes through to the owning CallConfig (rv_policy::reference_internal).
        .def_prop_rw(
            "runtime_env",
            [](CallConfig &c) -> RuntimeEnv & {
                return c.runtime_env;
            },
            [](CallConfig &c, const RuntimeEnv &re) {
                c.runtime_env = re;
            },
            nb::rv_policy::reference_internal
        )
        .def_prop_rw(
            "enable_l2_swimlane",
            [](const CallConfig &c) {
                return c.enable_l2_swimlane;
            },
            // Accept either an int perf_level (0-4) or a Python bool. `True` maps to
            // level 4 (full collection) to preserve the pre-perf_level semantics for
            // callers that still pass a boolean; `False` maps to 0.
            [](CallConfig &c, nb::object v) {
                if (PyBool_Check(v.ptr())) {
                    c.enable_l2_swimlane = nb::cast<bool>(v) ? 4 : 0;
                } else {
                    int level = nb::cast<int>(v);
                    c.enable_l2_swimlane = (level < 0) ? 0 : (level > 4) ? 4 : level;
                }
            }
        )
        // Accept either an int dump level (0=off, 1=partial, 2=full,
        // 3=full_json_only) or a Python bool. `True` maps to level 1
        // (partial) — the default when --dump-args is passed without a
        // value; `False` maps to 0.
        .def_prop_rw(
            "enable_dump_args",
            [](const CallConfig &c) {
                return c.enable_dump_args;
            },
            [](CallConfig &c, nb::object v) {
                if (PyBool_Check(v.ptr())) {
                    c.enable_dump_args = nb::cast<bool>(v) ? 1 : 0;
                } else {
                    int level = nb::cast<int>(v);
                    c.enable_dump_args = (level < 0) ? 0 : (level > 3) ? 3 : level;
                }
            }
        )
        .def_rw("enable_pmu", &CallConfig::enable_pmu)
        .def("validate", &CallConfig::validate)
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
            "enable_scope_stats",
            [](const CallConfig &c) {
                return static_cast<bool>(c.enable_scope_stats);
            },
            [](CallConfig &c, bool v) {
                c.enable_scope_stats = v ? 1 : 0;
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
        .def("__repr__", [append_ring_values](const CallConfig &self) -> std::string {
            std::ostringstream os;
            os << "CallConfig(block_dim=" << self.block_dim << ", aicpu_thread_num=" << self.aicpu_thread_num
               << ", enable_l2_swimlane=" << self.enable_l2_swimlane << ", enable_dump_args=" << self.enable_dump_args
               << ", enable_pmu=" << self.enable_pmu << ", enable_dep_gen=" << (self.enable_dep_gen ? "True" : "False")
               << ", enable_scope_stats=" << (self.enable_scope_stats ? "True" : "False");
            if (self.runtime_env.any()) {
                append_ring_values(os, "runtime_env.ring_task_window", true, self.runtime_env.ring_task_window);
                append_ring_values(os, "runtime_env.ring_heap", true, self.runtime_env.ring_heap);
                append_ring_values(os, "runtime_env.ring_dep_pool", true, self.runtime_env.ring_dep_pool);
            }
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

    // Per-stage run timing (host wall, on-NPU device wall + AICPU phase
    // breakdown) is no longer returned from run(); the platform emits it as
    // `[STRACE]` log markers — parse with simpler_setup.tools.strace_timing.

    // --- ChipWorker ---
    nb::class_<ChipWorker>(m, "_ChipWorker")
        .def(nb::init<>())
        .def(
            "init",
            [](ChipWorker &self, const std::string &host_lib_path, const std::string &aicpu_path,
               const std::string &aicore_path, const std::string &dispatcher_path, int device_id,
               std::optional<CallConfig> prewarm_config) {
                self.init(
                    host_lib_path, aicpu_path, aicore_path, dispatcher_path, device_id,
                    prewarm_config.has_value() ? &(*prewarm_config) : nullptr
                );
            },
            nb::arg("host_lib_path"), nb::arg("aicpu_path"), nb::arg("aicore_path"), nb::arg("dispatcher_path"),
            nb::arg("device_id"), nb::arg("prewarm_config") = nb::none(),
            "Bind the runtime library and attach to device_id. When prewarm_config is "
            "given, its ring sizing is built + cached inside init (fork-constant, no "
            "cross-process control command). A no-op for runtimes without a prebuilt arena."
        )
        .def("finalize", &ChipWorker::finalize)
        .def(
            "register_callable",
            [](ChipWorker &self, int32_t callable_id, const PyChipCallable &callable) {
                self.register_callable(callable_id, callable.buffer_.data());
            },
            nb::arg("callable_id"), nb::arg("callable"),
            "Stage a ChipCallable under callable_id for cheap repeated launches "
            "via run. Variants without per-callable_id support raise."
        )
        .def(
            "register_callable_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t blob_ptr) {
                self.register_callable(callable_id, reinterpret_cast<const void *>(blob_ptr));
            },
            nb::arg("callable_id"), nb::arg("blob_ptr"),
            "Stage a ChipCallable from a raw contiguous-buffer pointer (used by "
            "post-fork dynamic register handlers that receive the ChipCallable "
            "bytes via shared memory; see docs/callable-identity-registration.md). "
            "Equivalent to register_callable(callable_id, ChipCallable) but accepts the "
            "ChipCallable layout pointer directly so chip-child loops can prepare "
            "from shm without rebuilding a PyChipCallable wrapper."
        )
        .def(
            "run",
            [](ChipWorker &self, int32_t callable_id, ChipStorageTaskArgs &args, const CallConfig &config) {
                self.run(callable_id, &args, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"),
            "Launch a callable_id previously staged via register_callable. Returns "
            "None; per-stage timing is emitted as `[STRACE]` log markers."
        )
        .def(
            "run",
            [](ChipWorker &self, int32_t callable_id, TaskArgs &args, const CallConfig &config) {
                TaskArgsView view = make_view(args);
                self.run(callable_id, view, config);
            },
            nb::arg("callable_id"), nb::arg("args"), nb::arg("config"),
            "Launch a callable_id from a TaskArgs (used for in-process callers). "
            "Returns None; timing is emitted as `[STRACE]` log markers."
        )
        .def(
            "run_from_blob",
            [](ChipWorker &self, int32_t callable_id, uint64_t args_blob_ptr, size_t blob_capacity,
               const CallConfig &config) {
                // The mailbox region is the on-wire format `write_blob` produced;
                // `read_blob` is the matching reader that returns a zero-copy
                // TaskArgsView into the caller-owned bytes. Forwards to the
                // existing `run(cid, view, config)` path so chip-child
                // loops never re-implement the tensor/scalar layout in Python
                // (where it has historically dropped fields like child_memory).
                TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(args_blob_ptr), blob_capacity);
                self.run(callable_id, view, config);
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
            "Number of distinct callable entries the AICPU has dlopened for on the "
            "bound device. Equals 0 when not initialized or the runtime "
            "variant lacks prepared-callable registration. Tests assert this to verify "
            "register_callable + repeated run do not redundantly dlopen."
        )
        .def_prop_ro(
            "host_dlopen_count", &ChipWorker::host_dlopen_count,
            "Number of host-side dlopens triggered by register_callable on "
            "host_build_graph variants. Mirrors aicpu_dlopen_count for the "
            "host-orchestration path; 0 on device-orch variants."
        )
        .def("malloc", &ChipWorker::malloc, nb::arg("size"))
        .def("free", &ChipWorker::free, nb::arg("ptr"))
        .def("copy_to", &ChipWorker::copy_to, nb::arg("dst"), nb::arg("src"), nb::arg("size"))
        .def("copy_from", &ChipWorker::copy_from, nb::arg("dst"), nb::arg("src"), nb::arg("size"))
        .def(
            "l3_l2_orch_comm_init_from_addr", &ChipWorker::l3_l2_orch_comm_init, nb::arg("control_block_addr"),
            nb::arg("control_block_size"),
            "Start the independent L3-L2 orchestrator communication service from a mapped control-block address."
        )
        .def(
            "l3_l2_orch_comm_shutdown", &ChipWorker::l3_l2_orch_comm_shutdown,
            "Stop the independent L3-L2 orchestrator communication service."
        )
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
        .def(
            "comm_derive_context", &ChipWorker::comm_derive_context, nb::arg("comm_handle"), nb::arg("rank_ids"),
            nb::arg("domain_rank"), nb::arg("window_offset"), nb::arg("window_size"),
            "Derive a domain-local CommContext from an allocated base communicator."
        )
        .def(
            "comm_alloc_domain_windows",
            [](ChipWorker &self, uint64_t comm_handle, uint64_t allocation_id, const std::vector<uint32_t> &rank_ids,
               uint32_t domain_rank, size_t window_size) {
                auto [device_ctx, local_window_base] =
                    self.comm_alloc_domain_windows(comm_handle, allocation_id, rank_ids, domain_rank, window_size);
                return nb::make_tuple(device_ctx, local_window_base);
            },
            nb::arg("comm_handle"), nb::arg("allocation_id"), nb::arg("rank_ids"), nb::arg("domain_rank"),
            nb::arg("window_size"),
            "Collectively allocate a fresh per-rank pool for a subset; returns "
            "(device_ctx, local_window_base) for this rank."
        )
        .def(
            "comm_release_domain_windows", &ChipWorker::comm_release_domain_windows, nb::arg("comm_handle"),
            nb::arg("allocation_id"), nb::arg("rank_count"), nb::arg("domain_rank"),
            "Pair to comm_alloc_domain_windows: collectively release the per-rank pool."
        )
        .def("comm_barrier", &ChipWorker::comm_barrier, nb::arg("comm_handle"), "Synchronize all ranks.")
        .def(
            "comm_destroy", &ChipWorker::comm_destroy, nb::arg("comm_handle"),
            "Destroy the communicator and release its resources."
        )
        .def("comm_destroy_all", &ChipWorker::comm_destroy_all, "Destroy all owned communicators in LIFO order.");

    // --- Standalone blob helpers ---
    m.def(
        "read_args_from_blob",
        [](uint64_t blob_ptr) {
            TaskArgsView view = read_blob(reinterpret_cast<const uint8_t *>(blob_ptr), MAILBOX_ARGS_CAPACITY);
            TaskArgs args;
            for (int32_t i = 0; i < view.tensor_count; i++) {
                args.add_tensor(view.tensors(i));
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

    m.def(
        "_l3_host_mapped_region_import_sim",
        [](const std::string &token, uint64_t mapping_bytes) -> uint64_t {
            if (mapping_bytes == 0 || mapping_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::invalid_argument("L3-L2 sim L3 Host mapped-region import requires a positive mapping size");
            }
            std::string name = shm_name_for_open(token);
            int fd = shm_open(name.c_str(), O_RDWR, 0);
            if (fd < 0) {
                throw std::runtime_error("L3-L2 sim L3 Host mapped-region import shm_open failed");
            }
            void *base = mmap(nullptr, static_cast<size_t>(mapping_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (base == MAP_FAILED) {
                int err = errno;
                ::close(fd);
                throw std::runtime_error(
                    std::string("L3-L2 sim L3 Host mapped-region import mmap failed: ") + std::strerror(err)
                );
            }
            std::lock_guard<std::mutex> lk(g_l3_host_mapped_region_mu);
            uint64_t handle = g_next_l3_host_mapped_region_handle++;
            g_l3_host_mapped_regions.emplace(
                handle, L3HostMappedSimRegion{
                            fd,
                            base,
                            static_cast<size_t>(mapping_bytes),
                        }
            );
            return handle;
        },
        nb::arg("token"), nb::arg("mapping_bytes"),
        "Import a sim L3-L2 POSIX shm region for L3 Host mapped-region access."
    );
    m.def(
        "_l3_host_mapped_region_close",
        [](uint64_t handle) {
            L3HostMappedSimRegion mapping{};
            {
                std::lock_guard<std::mutex> lk(g_l3_host_mapped_region_mu);
                auto it = g_l3_host_mapped_regions.find(handle);
                if (it == g_l3_host_mapped_regions.end()) {
                    return;
                }
                mapping = it->second;
                g_l3_host_mapped_regions.erase(it);
            }
            if (mapping.base != nullptr) {
                munmap(mapping.base, mapping.size);
            }
            if (mapping.fd >= 0) {
                ::close(mapping.fd);
            }
        },
        nb::arg("handle"), "Close an L3 Host mapped-region handle."
    );
    m.def(
        "_l3_host_mapped_payload_write",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_write host_ptr must be nonzero");
            }
            L3HostMappedSimRegion &mapping = l3_host_mapped_region(handle);
            validate_mapping_range(mapping, payload_offset, nbytes);
            std::memcpy(
                static_cast<uint8_t *>(mapping.base) + payload_offset,
                reinterpret_cast<const void *>(static_cast<uintptr_t>(host_ptr)), static_cast<size_t>(nbytes)
            );
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        "Copy L3 Host bytes into an imported L3-L2 payload range."
    );
    m.def(
        "_l3_host_mapped_payload_read",
        [](uint64_t handle, uint64_t payload_offset, uint64_t host_ptr, uint64_t nbytes) {
            if (host_ptr == 0) {
                throw std::invalid_argument("L3-L2 payload_read host_ptr must be nonzero");
            }
            L3HostMappedSimRegion &mapping = l3_host_mapped_region(handle);
            validate_mapping_range(mapping, payload_offset, nbytes);
            std::memcpy(
                reinterpret_cast<void *>(static_cast<uintptr_t>(host_ptr)),
                static_cast<const uint8_t *>(mapping.base) + payload_offset, static_cast<size_t>(nbytes)
            );
        },
        nb::arg("handle"), nb::arg("payload_offset"), nb::arg("host_ptr"), nb::arg("nbytes"),
        "Copy imported L3-L2 payload bytes into L3 Host memory."
    );
    m.def(
        "_l3_host_mapped_counter_notify",
        [](uint64_t handle, uint64_t counter_offset, int32_t value, int op) {
            L3HostMappedSimRegion &mapping = l3_host_mapped_region(handle);
            if (counter_offset % sizeof(int32_t) != 0) {
                throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
            }
            if (op == 1) {
                value = load_counter(mapping, counter_offset) + value;
            } else if (op != 0) {
                throw std::invalid_argument("L3-L2 counter notify op is invalid");
            }
            store_counter(mapping, counter_offset, value);
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("value"), nb::arg("op"),
        "Store or add one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_l3_host_mapped_counter_test",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp) {
            L3HostMappedSimRegion &mapping = l3_host_mapped_region(handle);
            if (counter_offset % sizeof(int32_t) != 0) {
                throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
            }
            int32_t observed = load_counter(mapping, counter_offset);
            return nb::make_tuple(compare_counter(observed, operand, cmp), observed);
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"),
        "Load and compare one L3 Host-side L3-L2 signal counter."
    );
    m.def(
        "_l3_host_mapped_counter_wait",
        [](uint64_t handle, uint64_t counter_offset, int32_t operand, int cmp, uint64_t timeout_ns) {
            L3HostMappedSimRegion &mapping = l3_host_mapped_region(handle);
            if (counter_offset % sizeof(int32_t) != 0) {
                throw std::invalid_argument("L3-L2 counter offset must be 4-byte aligned");
            }
            auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
            int32_t observed = 0;
            while (true) {
                observed = load_counter(mapping, counter_offset);
                bool matched = compare_counter(observed, operand, cmp);
                if (matched) {
                    return nb::make_tuple(0, 0, observed, true, std::string{});
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    return nb::make_tuple(-1, 7, observed, false, std::string{"SIGNAL_WAIT timed out"});
                }
                std::this_thread::sleep_for(std::chrono::nanoseconds(50000));
            }
        },
        nb::arg("handle"), nb::arg("counter_offset"), nb::arg("operand"), nb::arg("cmp"), nb::arg("timeout_ns"),
        "Poll one L3 Host-side L3-L2 signal counter until match or timeout."
    );

    bind_worker(m);
}
