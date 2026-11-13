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
        if (argc == 7 && std::string(argv[1]) == "--force-sync-start") {
            std::ifstream input(argv[2], std::ios::binary);
            auto records = dsv41::read_records(input);
            const auto slice = dsv41::select_slice(records, std::stoull(argv[3]), std::stoull(argv[4]));
            const int max_blocks = std::stoi(argv[6]);
            if (max_blocks < 1) throw std::runtime_error("MAX_BLOCKS must be positive");
            size_t spmd = 0, changed = 0, over_limit = 0, enabled = 0;
            for (size_t i = slice.begin; i < slice.end; ++i) {
                auto &header = records[i].header;
                const bool has_kernel = header.kernels[0] >= 0 || header.kernels[1] >= 0 || header.kernels[2] >= 0;
                if (header.operation == dsv41::Operation::SUBMIT && has_kernel && header.block_num > 1) {
                    ++spmd;
                    if (header.block_num > max_blocks) {
                        ++over_limit;
                    } else {
                        changed += header.sync_start == 0;
                        header.sync_start = 1;
                    }
                    enabled += header.sync_start != 0;
                }
            }
            std::ofstream output(argv[5], std::ios::binary);
            dsv41::write(output, dsv41::record_magic);
            dsv41::write(output, uint64_t{sizeof(dsv41::Operand)});
            for (const auto &record : records) {
                dsv41::write(output, record.header);
                for (const auto &operand : record.tensors)
                    dsv41::write(output, operand);
                for (const auto scalar : record.scalars)
                    dsv41::write(output, scalar);
                for (const auto dependency : record.dependencies)
                    dsv41::write(output, dependency);
            }
            std::cout << "{\"spmd\":" << spmd << ",\"changed\":" << changed << ",\"over_limit\":" << over_limit
                      << ",\"sync_start_after\":" << enabled << "}\n";
            return 0;
        }
        if (argc != 4) throw std::runtime_error("usage: record-tool CAPTURE FIRST_TASK LAST_TASK");
        std::ifstream stream(argv[1], std::ios::binary);
        const auto records = dsv41::read_records(stream);
        const auto slice = dsv41::select_slice(records, std::stoull(argv[2]), std::stoull(argv[3]));
        size_t tasks = 0, executions = 0;
        bool uses_comm_context = false;
        for (size_t i = slice.begin; i < slice.end; ++i) {
            const auto &record = records[i];
            if (records.front().scalars.size() > 1 && records.front().scalars[1] != 0) {
                for (const auto scalar : record.scalars)
                    uses_comm_context |= scalar == records.front().scalars[1];
            }
            if (record.header.operation == dsv41::Operation::SUBMIT ||
                record.header.operation == dsv41::Operation::DUMMY) {
                ++tasks;
                for (int lane = 0; lane < 3; ++lane) {
                    if (record.header.kernels[lane] >= 0) executions += record.header.block_num;
                }
            }
        }
        bool prefix_uses_comm_context = false;
        if (records.front().scalars.size() > 1 && records.front().scalars[1] != 0) {
            for (size_t i = 1; i < slice.begin; ++i) {
                for (const auto scalar : records[i].scalars)
                    prefix_uses_comm_context |= scalar == records.front().scalars[1];
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
        std::cout << "],\"uses_comm_context\":" << (uses_comm_context ? "true" : "false");
        std::cout << ",\"prefix_uses_comm_context\":" << (prefix_uses_comm_context ? "true" : "false");
        for (const bool selected_only : {false, true}) {
            std::cout << (selected_only ? ",\"case_external_bytes\":[" : ",\"external_bytes\":[");
            comma = false;
            for (const auto &entry : records.front().tensors) {
                uint64_t bytes = 0;
                for (size_t i = selected_only ? slice.begin : 1; i < slice.end; ++i) {
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
            std::cout << "]";
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
