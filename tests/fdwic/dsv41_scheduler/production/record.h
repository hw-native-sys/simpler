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
#pragma once

#include <cstdint>
#include <istream>
#include <map>
#include <ostream>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "host_build_graph/tensor_data.h"

namespace dsv41 {
// A local, same-build capture of Host submissions, never an on-device protocol.
// Keeping the original TensorData retains views and aliases that deps.json lacks.
constexpr uint64_t record_magic = 0x4453563431523031;
enum class Operation : uint32_t { BOUNDARY, ALLOCATE, SUBMIT, DUMMY, SCOPE_BEGIN, SCOPE_END };
struct Header {
    Operation operation{};
    uint32_t tensor_count{}, scalar_count{}, dep_count{};
    uint64_t task_id{};
    int32_t kernels[3]{-1, -1, -1};
    int32_t block_num{1}, sync_start{}, early_resolve{}, timing_slot{-1}, scope_mode{};
};
struct Operand {
    simpler::hbg::TensorData tensor{};
    uint32_t tag{};
};
struct Record {
    Header header;
    std::vector<Operand> tensors;
    std::vector<uint64_t> scalars, dependencies;
};

template <class T>
void write(std::ostream &stream, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("Cannot write original submission capture");
}

template <class T>
T read(std::istream &stream) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    stream.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("Truncated original submission capture");
    return value;
}

inline std::vector<Record> read_records(std::istream &stream) {
    if (read<uint64_t>(stream) != record_magic || read<uint64_t>(stream) != sizeof(Operand)) {
        throw std::runtime_error("Submission capture requires its original Simpler build");
    }
    std::vector<Record> records;
    while (stream.peek() != std::char_traits<char>::eof()) {
        Record record;
        record.header = read<Header>(stream);
        if (record.header.tensor_count > 1024 || record.header.scalar_count > 1024 ||
            record.header.dep_count > 100000) {
            throw std::runtime_error("Invalid submission capture counts");
        }
        for (uint32_t i = 0; i < record.header.tensor_count; ++i)
            record.tensors.push_back(read<Operand>(stream));
        for (uint32_t i = 0; i < record.header.scalar_count; ++i)
            record.scalars.push_back(read<uint64_t>(stream));
        for (uint32_t i = 0; i < record.header.dep_count; ++i)
            record.dependencies.push_back(read<uint64_t>(stream));
        records.push_back(std::move(record));
    }
    return records;
}

struct Slice {
    size_t begin{}, end{};
    std::map<uint64_t, uint64_t> boundary_buffers;
    std::vector<int32_t> scopes;
};

inline Slice select_slice(const std::vector<Record> &records, uint64_t first, uint64_t last) {
    if (records.empty() || records.front().header.operation != Operation::BOUNDARY) {
        throw std::runtime_error("Missing original forward boundary");
    }
    Slice slice{};
    std::set<uint64_t> external;
    for (const auto &operand : records.front().tensors)
        external.insert(operand.tensor.buffer.addr);
    std::map<uint64_t, size_t> allocations;
    for (size_t i = 1; i < records.size(); ++i) {
        const auto &record = records[i];
        const auto operation = record.header.operation;
        if (operation == Operation::SUBMIT || operation == Operation::DUMMY) {
            if (record.header.task_id == first) slice.begin = i;
            if (record.header.task_id == last) slice.end = i + 1;
        }
        for (const auto &operand : record.tensors) {
            if (operand.tag == 1) allocations[operand.tensor.buffer.addr] = i;
        }
    }
    if (slice.begin == 0 || slice.end <= slice.begin)
        throw std::runtime_error("Original task boundaries were not found");
    for (size_t i = 1; i < slice.begin; ++i) {
        if (records[i].header.operation == Operation::SCOPE_BEGIN) slice.scopes.push_back(records[i].header.scope_mode);
        if (records[i].header.operation == Operation::SCOPE_END) {
            if (slice.scopes.empty()) throw std::runtime_error("Unbalanced original scope");
            slice.scopes.pop_back();
        }
    }
    for (size_t i = slice.begin; i < slice.end; ++i) {
        for (const auto &operand : records[i].tensors) {
            const auto &buffer = operand.tensor.buffer;
            if (external.count(buffer.addr) != 0) continue;
            const auto allocation = allocations.find(buffer.addr);
            if (allocation == allocations.end()) throw std::runtime_error("Tensor allocation was not captured");
            if (allocation->second < slice.begin) slice.boundary_buffers[buffer.addr] = buffer.size;
        }
    }
    return slice;
}
}  // namespace dsv41
