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

#include "affinity_allowed_file.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "common/platform_config.h"

namespace pto::a5 {
namespace {

constexpr char kEnvAffinityPlanPath[] = "SIMPLER_AICPU_AFFINITY_PLAN";
constexpr char kDefaultAffinityPlanRelative[] = "build/config/aicpu_affinity_plan.json";
constexpr int32_t kAffinityPlanSchemaVersion = 3;
constexpr int32_t kAffinityPlanActiveCount = 5;

bool is_supported_affinity_plan_source(const std::string &source) {
    return source == "atomic-flag-orch+cond-die-v1" || source == "manual" || source == "auto-first-run" ||
           source == "probe-failed-contiguous";
}

bool parse_i32(const std::string &text, int32_t &out) {
    if (text.empty()) return false;
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

bool parse_u64(const std::string &text, uint64_t &out) {
    if (text.empty() || text[0] == '-') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 0);
    if (errno != 0 || end == text.c_str() || *end != '\0') return false;
    out = static_cast<uint64_t>(value);
    return true;
}

bool parse_csv_int_list(const std::string &text, std::vector<int32_t> &out) {
    out.clear();
    if (text.empty()) return false;
    const char *p = text.c_str();
    while (true) {
        while (std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        if (*p == '\0') return false;
        char *end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end == p || value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        out.push_back(static_cast<int32_t>(value));
        p = end;
        while (std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        if (*p == '\0') return true;
        if (*p != ',') return false;
        ++p;
    }
}

}  // namespace

std::string affinity_cpus_side_path(int32_t device_id) {
    const char *env = std::getenv(kEnvAffinityPlanPath);
    std::string plan = (env != nullptr && env[0] != '\0') ? env : kDefaultAffinityPlanRelative;
    const std::string id = std::to_string(device_id);
    const std::string placeholder = "{device}";
    bool used_placeholder = false;
    for (size_t pos = plan.find(placeholder); pos != std::string::npos; pos = plan.find(placeholder, pos + id.size())) {
        plan.replace(pos, placeholder.size(), id);
        used_placeholder = true;
    }

    std::filesystem::path plan_path(plan);
    if (!used_placeholder) {
        const std::string device_suffix = "." + id;
        std::string extension = plan_path.extension().string();
        std::string stem = plan_path.filename().string();
        if (!extension.empty()) stem.resize(stem.size() - extension.size());
        if (stem.size() < device_suffix.size() ||
            stem.compare(stem.size() - device_suffix.size(), device_suffix.size(), device_suffix) != 0) {
            stem += device_suffix;
        }
        if (extension.empty()) extension = ".json";
        plan_path = plan_path.parent_path() / (stem + extension);
    }
    plan_path.replace_extension(".cpus");
    return plan_path.string();
}

bool resolve_aicpu_active_count(
    int32_t requested_active_count, int32_t launch_count, int32_t &active_count, bool &use_rtt_plan
) {
    active_count = 0;
    use_rtt_plan = false;
    if (launch_count < 2 || launch_count > PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH || requested_active_count < 0 ||
        requested_active_count == 1 || requested_active_count > PLATFORM_MAX_AICPU_THREADS) {
        return false;
    }
    const int32_t desired = requested_active_count == 0 ? std::min(PLATFORM_DEFAULT_AICPU_THREAD_NUM, launch_count) :
                                                          requested_active_count;
    if (desired > launch_count) return false;
    active_count = desired;
    use_rtt_plan = active_count == PLATFORM_DEFAULT_AICPU_THREAD_NUM;
    return true;
}

bool build_occupy_contiguous_allowed(uint64_t occupy, int32_t active_count, std::vector<int32_t> &allowed_cpus) {
    allowed_cpus.clear();
    if (occupy == 0 || active_count < 2 || active_count > kAffinityPlanActiveCount) return false;
    allowed_cpus.reserve(static_cast<size_t>(active_count));
    for (int32_t cpu = 0; cpu < 64 && static_cast<int32_t>(allowed_cpus.size()) < active_count; ++cpu) {
        if (((occupy >> cpu) & 1ULL) != 0) allowed_cpus.push_back(cpu);
    }
    if (allowed_cpus.size() == static_cast<size_t>(active_count)) return true;
    allowed_cpus.clear();
    return false;
}

bool load_affinity_cpus_side_file(
    const char *soc_name, int32_t device_id, uint64_t current_occupy, std::vector<int32_t> &allowed_cpus,
    std::string &plan_source
) {
    allowed_cpus.clear();
    plan_source.clear();
    if (soc_name == nullptr || soc_name[0] == '\0' || device_id < 0 || current_occupy == 0) return false;

    std::ifstream input(affinity_cpus_side_path(device_id));
    if (!input) return false;

    std::string file_soc;
    std::string cpus_text;
    int32_t schema_version = -1;
    int32_t file_device_id = -1;
    int32_t active_count = -1;
    uint64_t file_occupy = 0;
    std::set<std::string> keys;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) return false;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        if (!keys.insert(key).second) return false;
        if (key == "schema_version") {
            if (!parse_i32(value, schema_version)) return false;
        } else if (key == "device_id") {
            if (!parse_i32(value, file_device_id)) return false;
        } else if (key == "soc") {
            file_soc = value;
        } else if (key == "source") {
            plan_source = value;
        } else if (key == "occupy_mask") {
            if (!parse_u64(value, file_occupy)) return false;
        } else if (key == "active_count") {
            if (!parse_i32(value, active_count)) return false;
        } else if (key == "cpus") {
            cpus_text = value;
        } else {
            return false;
        }
    }

    if (keys.size() != 7 || schema_version != kAffinityPlanSchemaVersion || file_device_id != device_id ||
        active_count != kAffinityPlanActiveCount || file_soc != soc_name || file_occupy != current_occupy) {
        return false;
    }
    if (!is_supported_affinity_plan_source(plan_source)) return false;
    if (!parse_csv_int_list(cpus_text, allowed_cpus) || allowed_cpus.size() != kAffinityPlanActiveCount) return false;
    {
        std::vector<int32_t> unique = allowed_cpus;
        std::sort(unique.begin(), unique.end());
        if (std::unique(unique.begin(), unique.end()) != unique.end()) return false;
    }
    for (int32_t cpu : allowed_cpus) {
        if (cpu < 0 || cpu >= 64 || ((current_occupy >> cpu) & 1ULL) == 0) return false;
    }
    return true;
}

}  // namespace pto::a5
