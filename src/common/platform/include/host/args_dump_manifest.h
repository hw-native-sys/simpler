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

#include <cmath>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

#include "data_type.h"
#include "host/args_dump_collector.h"

/**
 * The `args_dump.json` manifest's shape, in one place.
 *
 * Both output paths compose the same three pieces — prologue, one object per
 * arg, epilogue — so the single-run path and the retained background path
 * cannot drift into two dialects of the same file. The per-arg object is the
 * unit a reader parses (`simpler_setup/tools/dump_viewer.py`,
 * `core_swimlane.py`), and the prologue's `bin_file` is how either path names
 * the payload file the offsets below refer to.
 */
namespace simpler::dfx::args_dump {

inline const char *role_name(ArgsDumpRole role) {
    switch (role) {
    case ArgsDumpRole::INPUT:
        return "input";
    case ArgsDumpRole::OUTPUT:
        return "output";
    case ArgsDumpRole::INOUT:
        return "inout";
    }
    return "unknown";
}

inline const char *stage_name(ArgsDumpStage stage) {
    switch (stage) {
    case ArgsDumpStage::BEFORE_DISPATCH:
        return "before_dispatch";
    case ArgsDumpStage::AFTER_COMPLETION:
        return "after_completion";
    }
    return "unknown";
}

inline const char *kind_name(ArgsDumpKind kind) {
    switch (kind) {
    case ArgsDumpKind::TENSOR:
        return "tensor";
    case ArgsDumpKind::SCALAR:
        return "scalar";
    }
    return "unknown";
}

inline void write_scalar_json_value(std::ostream &json, const DumpedArg &dt) {
    uint64_t raw = dt.scalar_value;
    if (dt.dtype == static_cast<uint8_t>(DataType::FLOAT32)) {
        float f;
        memcpy(&f, &raw, sizeof(float));
        if (std::isnan(f)) {
            json << ", \"value\": null";
        } else if (std::isinf(f)) {
            json << ", \"value\": " << (f < 0 ? "\"-$Inf\"" : "\"$Inf\"");
        } else {
            std::ostringstream val_ss;
            val_ss << f;
            std::string val_str = val_ss.str();
            if (val_str.find('.') == std::string::npos && val_str.find('e') == std::string::npos) {
                val_str += ".0";
            }
            json << ", \"value\": " << val_str;
        }
    } else if (dt.dtype == static_cast<uint8_t>(DataType::INT32)) {
        int32_t val;
        memcpy(&val, &raw, sizeof(int32_t));
        json << ", \"value\": " << val;
    } else if (dt.dtype == static_cast<uint8_t>(DataType::UINT32)) {
        uint32_t val;
        memcpy(&val, &raw, sizeof(uint32_t));
        json << ", \"value\": " << val;
    } else if (dt.dtype == static_cast<uint8_t>(DataType::BOOL)) {
        json << ", \"value\": " << (raw != 0 ? "true" : "false");
    } else if (dt.dtype == static_cast<uint8_t>(DataType::INT64)) {
        int64_t val;
        memcpy(&val, &raw, sizeof(int64_t));
        json << ", \"value\": " << val;
    } else {
        json << ", \"value\": " << raw;
    }
}

inline std::string dims_to_string(const uint32_t dims[], int ndims) {
    std::ostringstream ss;
    ss << "[";
    for (int d = 0; d < ndims; d++) {
        if (d > 0) ss << ", ";
        ss << dims[d];
    }
    ss << "]";
    return ss.str();
}

inline std::string dtype_name_from_raw(uint8_t dtype) { return get_dtype_name(static_cast<DataType>(dtype)); }

inline uint64_t num_elements(const DumpedArg &dt) {
    uint64_t numel = 1;
    for (int d = 0; d < dt.ndims; d++) {
        numel *= dt.shapes[d];
    }
    return (dt.ndims == 0) ? 1 : numel;
}

/** The per-run counts and file names the prologue publishes. */
struct ManifestMeta {
    std::string run_dir_name;
    uint32_t dump_args_level{0};
    size_t total_args{0};
    uint32_t before_dispatch{0};
    uint32_t after_completion{0};
    uint32_t input_args{0};
    uint32_t output_args{0};
    uint32_t inout_args{0};
    uint64_t truncated_args{0};
    uint64_t dropped_records{0};
    // Empty means `null`: the level selected no payload, so no payload file
    // was produced and there is nothing for the offsets to refer to.
    std::string bin_file;
    // Retained path only, and diagnostic only: they explain an incomplete
    // result and never excuse one, because each also fails that run's flush.
    bool retained{false};
    uint64_t host_discarded_args{0};
    uint64_t metadata_discarded_records{0};
    bool counts_unknown{false};
    std::string verdict;
};

inline void write_manifest_prologue(std::ostream &json, const ManifestMeta &meta) {
    json << "{\n";
    json << "  \"run_dir\": \"" << meta.run_dir_name << "\",\n";
    json << "  \"bin_format\": {\n";
    json << "    \"type\": \"logical_contiguous\",\n";
    json << "    \"byte_order\": \"little_endian\"\n";
    json << "  },\n";
    json << "  \"dump_args_level\": " << meta.dump_args_level << ",\n";
    json << "  \"total_args\": " << meta.total_args << ",\n";
    json << "  \"before_dispatch\": " << meta.before_dispatch << ",\n";
    json << "  \"after_completion\": " << meta.after_completion << ",\n";
    json << "  \"input_args\": " << meta.input_args << ",\n";
    json << "  \"output_args\": " << meta.output_args << ",\n";
    json << "  \"inout_args\": " << meta.inout_args << ",\n";
    json << "  \"truncated_args\": " << meta.truncated_args << ",\n";
    json << "  \"dropped_records\": " << meta.dropped_records << ",\n";
    if (meta.retained) {
        json << "  \"host_discarded_args\": " << meta.host_discarded_args << ",\n";
        json << "  \"metadata_discarded_records\": " << meta.metadata_discarded_records << ",\n";
        json << "  \"counts_unknown\": " << (meta.counts_unknown ? "true" : "false") << ",\n";
        json << "  \"collection_verdict\": \"" << meta.verdict << "\",\n";
    }
    if (meta.bin_file.empty()) {
        json << "  \"bin_file\": null,\n";
    } else {
        json << "  \"bin_file\": \"" << meta.bin_file << "\",\n";
    }
    json << "  \"args\": [\n";
}

/** One arg object, without its separator. */
inline void write_arg_json(std::ostream &json, const DumpedArg &dt) {
    std::string dtype = dtype_name_from_raw(dt.dtype);
    uint64_t numel = num_elements(dt);
    std::string shape_str = dims_to_string(dt.shapes, dt.ndims);
    std::string strides_str = dims_to_string(dt.strides, dt.ndims);

    json << "    {\"run_epoch\": " << dt.run_epoch << ", \"task_id\": \"0x" << std::hex << std::setfill('0')
         << std::setw(16) << dt.task_id << std::dec << "\"";
    json << ", \"func_id\": [";
    for (int32_t f = 0; f < dt.func_count; f++) {
        if (f) json << ", ";
        json << dt.func_ids[f];
    }
    json << "]";
    json << ", \"arg_index\": " << dt.arg_index << ", \"role\": \"" << role_name(dt.role) << "\", \"stage\": \""
         << stage_name(dt.stage) << "\", \"kind\": \"" << kind_name(dt.kind) << "\", \"dtype\": \"" << dtype << "\"";
    if (dt.kind == ArgsDumpKind::SCALAR) {
        write_scalar_json_value(json, dt);
    }
    json << ", \"is_contiguous\": " << (dt.is_contiguous ? "true" : "false") << ", \"shape\": " << shape_str
         << ", \"strides\": " << strides_str << ", \"start_offset\": " << dt.start_offset << ", \"numel\": " << numel;
    if ((dt.flags & ARGS_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS) != 0) {
        json << ", \"arg_index_ambiguous\": true";
    }
    json << ", \"bin_offset\": " << dt.bin_offset << ", \"bin_size\": " << dt.payload_size
         << ", \"truncated\": " << (dt.truncated ? "true" : "false");
    if (dt.host_discarded) {
        // The record is here and its bytes are not: stated per arg so a reader
        // cannot mistake a zero-size payload for an empty tensor.
        json << ", \"host_discarded\": true";
    }
    json << "}";
}

inline void write_manifest_epilogue(std::ostream &json) { json << "\n  ]\n}\n"; }

}  // namespace simpler::dfx::args_dump
