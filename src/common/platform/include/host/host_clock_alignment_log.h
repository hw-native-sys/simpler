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

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "host_log.h"

/** Written-file offset sampled before this invocation; PID/invocation filtering also excludes late older records. */
inline uint64_t host_clock_alignment_log_offset() {
    const char *directory = HostLogger::get_instance().log_directory();
    if (directory == nullptr) return 0;
    std::error_code error;
    const auto size = std::filesystem::file_size(
        std::filesystem::path(directory) / ("host." + std::to_string(getpid()) + ".log"), error
    );
    return error ? 0 : size;
}

/** Archive this invocation's original alignment markers after its final chip.run span. */
inline bool export_host_clock_alignment_log(const std::string &output_prefix, uint64_t invocation, uint64_t offset) {
    auto &logger = HostLogger::get_instance();
    if (!logger.is_enabled(simpler::log::LogLevel::TIMING)) {
        LOG_WARN("Host clock alignment log export skipped: requires TIMING-or-finer logging");
        return true;
    }
    if (!logger.flush()) {
        LOG_WARN("Host clock alignment log export: process log flush failed");
        return false;
    }
    const char *directory = logger.log_directory();
    if (directory == nullptr) {
        LOG_WARN("Host clock alignment log export: no process log directory");
        return false;
    }
    const auto pid = std::to_string(getpid());
    const auto target = std::filesystem::path(output_prefix) / ("host_clock_alignment." + pid + ".log");
    const auto temporary = target.string() + ".tmp." + std::to_string(invocation);
    try {
        std::ifstream input(std::filesystem::path(directory) / ("host." + pid + ".log"));
        if (!input) throw std::runtime_error("cannot read process log");
        input.seekg(static_cast<std::streamoff>(offset));
        const std::string pid_field = " pid=" + pid + " ";
        const std::string inv_field = " inv=" + std::to_string(invocation) + " ";
        std::vector<std::string> records;
        bool run = false, runner = false, wall = false;
        for (std::string line; std::getline(input, line);) {
            const auto marker = line.find("[STRACE] ");
            if (marker == std::string::npos) continue;
            const auto record = line.substr(marker);
            if (record.find(pid_field) == std::string::npos || record.find(inv_field) == std::string::npos) continue;
            const bool is_run = record.find(" name=chip.run ") != std::string::npos;
            const bool is_runner = record.find(" name=chip.run.runner_run ") != std::string::npos;
            const bool is_wall = record.find(" name=chip.run.runner_run.device_wall ") != std::string::npos;
            const bool launch = record.find(" name=chip.run.runner_run.aicpu_launch ") != std::string::npos;
            const bool phase = record.find(" name=chip.run.runner_run.device_wall.") != std::string::npos;
            if (!(is_run || is_runner || is_wall || launch || phase)) continue;
            run |= is_run;
            runner |= is_runner;
            wall |= is_wall;
            records.push_back(record);
            if (is_run) break;  // The root span is emitted after all alignment markers.
        }
        if (!(run && runner && wall)) throw std::runtime_error("incomplete invocation markers");
        std::ofstream output(temporary);
        for (const auto &record : records)
            output << record << '\n';
        output.close();
        if (!output) throw std::runtime_error("cannot write timing log");
        std::filesystem::rename(temporary, target);
        return true;
    } catch (const std::exception &error) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        LOG_WARN("Host clock alignment log export failed: %s", error.what());
        return false;
    }
}
