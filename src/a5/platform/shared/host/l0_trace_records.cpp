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

#include "host/l0_trace_records.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "common/platform_config.h"
#include "common/l0_profiling.h"

namespace {
bool write_l0_events_json(
    const std::filesystem::path &run_dir, uint32_t physical_core_id, uint64_t run_epoch, uint64_t raw_bytes,
    uint64_t core_frequency_hz
) {
    const std::string stem = "core_" + std::to_string(physical_core_id);
    const auto raw_path = run_dir / (stem + ".bin");
    const auto json_path = run_dir / (stem + ".events.json");
    const auto temp_path = run_dir / (stem + ".events.json.tmp");
    std::error_code ec;
    if (std::filesystem::file_size(raw_path, ec) != raw_bytes || ec) return false;
    std::ifstream raw(raw_path, std::ios::binary);
    std::ofstream out(temp_path, std::ios::trunc);
    if (!raw || !out) return false;

    out << "{\"format\":\"simpler_a5_l0_events_v1\",\"run_epoch\":" << run_epoch
        << ",\"physical_core_id\":" << physical_core_id << ",\"raw_file\":\"" << stem
        << ".bin\",\"raw_bytes\":" << raw_bytes << ",\"system_counter_frequency_hz\":" << PLATFORM_PROF_SYS_CNT_FREQ
        << ",\"core_frequency_hz\":" << core_frequency_hz << ",\"core_frequency_source\":\""
        << (core_frequency_hz != 0 ? "hal_rated" : "unavailable") << '"'
        << ",\"time_domains\":{\"anchor_syscnt\":\"system_counter_ticks\",\"elapsed_cycles\":\"core_cycles_since_"
           "anchor\"}"
        << ",\"event_kinds\":{\"0\":\"start_stamp\",\"1\":\"state\",\"2\":\"mark_stamp\",\"3\":\"unknown\",\"4\":\"end_"
           "flag\"}"
        << ",\"state_lanes\":[\"SCALAR\",\"VECTOR\",\"CUBE\",\"MTE1\",\"MTE2\",\"MTE3\",\"FIXP\"]"
        << ",\"payload_semantics\":{\"state\":\"busy_mask_low_7_bits\",\"mark_stamp\":\"mark_id\"}"
        << ",\"event_columns\":[\"anchor_syscnt\",\"elapsed_cycles\",\"kind\",\"block_id\",\"payload\",\"ctrl_type\","
           "\"raw_word\"]"
        << ",\"busy_interval_columns\":[\"anchor_syscnt\",\"start_cycles\",\"end_cycles\",\"lane\",\"block_id\",\"clip_"
           "flags\"]"
           ",\"busy_intervals\":[";
    std::array<uint8_t, 64u << 10> bytes{};
    const auto decode_pass = [&](simpler::dfx::l0::TraceDecoder &decoder, auto &&emit) {
        while (raw) {
            raw.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            const auto count = raw.gcount();
            if (count <= 0) break;
            decoder.feed(bytes.data(), static_cast<size_t>(count), emit);
            if (!out) break;
        }
        return raw.eof() && out.good();
    };
    simpler::dfx::l0::BusyIntervalBuilder intervals;
    simpler::dfx::l0::TraceDecoder interval_decoder;
    uint64_t interval_count = 0;
    const auto emit_interval = [&](const simpler::dfx::l0::BusyInterval &interval) {
        if (interval_count++ != 0) out << ',';
        out << '[' << interval.anchor_syscnt << ',' << interval.start_cycles << ',' << interval.end_cycles << ','
            << static_cast<unsigned int>(interval.lane) << ',' << interval.block_id << ','
            << static_cast<unsigned int>(interval.clip_flags) << ']';
    };
    const bool first_pass_ok = decode_pass(interval_decoder, [&](const simpler::dfx::l0::TraceEvent &event) {
        intervals.accept(event, emit_interval);
    });
    interval_decoder.finish();
    const auto &interval_stats = intervals.finish(emit_interval);
    if (!first_pass_ok) {
        out.close();
        std::filesystem::remove(temp_path, ec);
        return false;
    }
    out << "],\"interval_count\":" << interval_count
        << ",\"interval_stats\":{\"clipped_start\":" << interval_stats.clipped_start
        << ",\"clipped_end\":" << interval_stats.clipped_end << "},\"events\":[";
    raw.clear();
    raw.seekg(0);
    if (!raw || !out) {
        out.close();
        std::filesystem::remove(temp_path, ec);
        return false;
    }
    simpler::dfx::l0::TraceDecoder decoder;
    uint64_t event_count = 0;
    const bool second_pass_ok = decode_pass(decoder, [&](const simpler::dfx::l0::TraceEvent &event) {
        if (event_count++ != 0) out << ',';
        out << '[' << event.anchor_syscnt << ',' << event.elapsed_cycles << ',' << static_cast<unsigned int>(event.kind)
            << ',' << event.block_id << ',' << event.payload << ',' << static_cast<unsigned int>(event.ctrl_type) << ','
            << event.raw_word << ']';
    });
    const auto &stats = decoder.finish();
    out << "],\"event_count\":" << event_count << ",\"stats\":{\"start_stamps\":" << stats.start_stamps
        << ",\"states\":" << stats.states << ",\"mark_stamps\":" << stats.mark_stamps
        << ",\"unknown\":" << stats.unknown << ",\"fillers\":" << stats.fillers
        << ",\"unexpected_fillers\":" << stats.unexpected_fillers << ",\"end_flags\":" << stats.end_flags
        << ",\"malformed_start_groups\":" << stats.malformed_start_groups
        << ",\"words_without_start\":" << stats.words_without_start << ",\"trailing_bytes\":" << stats.trailing_bytes
        << ",\"incomplete_start_groups\":" << stats.incomplete_start_groups
        << ",\"structurally_complete\":" << (stats.structurally_complete() ? "true" : "false") << "}}\n";
    out.close();
    if (!second_pass_ok || !out) {
        std::filesystem::remove(temp_path, ec);
        return false;
    }
    std::filesystem::rename(temp_path, json_path, ec);
    if (ec) {
        std::error_code cleanup_ec;
        std::filesystem::remove(temp_path, cleanup_ec);
        return false;
    }
    return true;
}

std::optional<uint32_t> raw_core_id(const std::filesystem::path &path) {
    const std::string name = path.filename().string();
    if (name.size() <= 9 || name.compare(0, 5, "core_") != 0 || name.compare(name.size() - 4, 4, ".bin") != 0)
        return std::nullopt;
    const std::string digits = name.substr(5, name.size() - 9);
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](unsigned char c) {
            return std::isdigit(c);
        }))
        return std::nullopt;
    try {
        const unsigned long id = std::stoul(digits);
        if (id < PLATFORM_MAX_CORES && id <= std::numeric_limits<uint32_t>::max()) return static_cast<uint32_t>(id);
    } catch (const std::exception &) {}
    return std::nullopt;
}

}  // namespace

bool export_l0_event_records(
    const std::filesystem::path &run_dir, uint64_t run_epoch, uint64_t aic_frequency_hz, uint64_t aiv_frequency_hz,
    const std::vector<uint32_t> &physical_core_types
) {
    std::vector<uint32_t> cores;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(run_dir, ec)) {
        if (ec) break;
        const bool regular = entry.is_regular_file(ec);
        if (ec) break;
        if (!regular) continue;
        const auto id = raw_core_id(entry.path());
        if (id) cores.push_back(*id);
    }
    if (ec) return false;
    std::sort(cores.begin(), cores.end());
    if (std::adjacent_find(cores.begin(), cores.end()) != cores.end()) return false;
    bool ok = true;
    for (uint32_t core : cores) {
        const auto raw_path = run_dir / ("core_" + std::to_string(core) + ".bin");
        const uint64_t bytes = std::filesystem::file_size(raw_path, ec);
        if (ec) return false;
        if (bytes == 0) continue;
        if (core >= physical_core_types.size()) return false;
        const uint32_t type = physical_core_types[core];
        if (type != L0_DFX_CORE_TYPE_AIC && type != L0_DFX_CORE_TYPE_AIV) return false;
        const uint64_t hz = type == L0_DFX_CORE_TYPE_AIC ? aic_frequency_hz : aiv_frequency_hz;
        if (!write_l0_events_json(run_dir, core, run_epoch, bytes, hz)) ok = false;
    }
    return ok;
}
