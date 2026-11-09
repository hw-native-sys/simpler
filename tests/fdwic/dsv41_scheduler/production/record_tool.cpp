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
#include <fstream>
#include <iostream>
#include <string>

#include "record.h"

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: record-tool CAPTURE FIRST_TASK LAST_TASK");
        std::ifstream stream(argv[1], std::ios::binary);
        const auto records = dsv41::read_records(stream);
        const auto slice = dsv41::select_slice(records, std::stoull(argv[2]), std::stoull(argv[3]));
        size_t tasks = 0, executions = 0;
        for (size_t i = slice.begin; i < slice.end; ++i) {
            const auto &record = records[i];
            if (record.header.operation == dsv41::Operation::SUBMIT ||
                record.header.operation == dsv41::Operation::DUMMY) {
                ++tasks;
                for (int lane = 0; lane < 3; ++lane) {
                    if (record.header.kernels[lane] >= 0) executions += record.header.block_num;
                }
            }
        }
        std::cout << "{\"tasks\":" << tasks << ",\"executions\":" << executions << ",\"begin\":" << slice.begin
                  << ",\"end\":" << slice.end << ",\"boundary_buffers\":[";
        bool comma = false;
        for (const auto &[address, bytes] : slice.boundary_buffers) {
            if (comma) std::cout << ',';
            comma = true;
            std::cout << "{\"source_address\":" << address << ",\"bytes\":" << bytes << '}';
        }
        std::cout << "],\"external_bytes\":[";
        comma = false;
        for (const auto &entry : records.front().tensors) {
            uint64_t bytes = 0;
            for (size_t i = 1; i < slice.end; ++i) {
                for (const auto &operand : records[i].tensors) {
                    const auto &tensor = operand.tensor;
                    if (tensor.buffer.addr == entry.tensor.buffer.addr) {
                        bytes = std::max(
                            bytes, (tensor.start_offset + tensor.extent_elem_cache) * get_element_size(tensor.dtype)
                        );
                    }
                }
            }
            if (comma) std::cout << ',';
            comma = true;
            std::cout << bytes;
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
