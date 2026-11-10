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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "chip_worker.h"
#include "chip_run_lane.h"
#include "callable.h"
#include "host_log.h"

namespace fs = std::filesystem;

namespace {
std::vector<uint8_t> read_binary(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

template <class T>
T read(std::istream &stream) {
    T value;
    if (!(stream >> value)) throw std::runtime_error("Incomplete native input manifest");
    return value;
}

// Host launch alignment only. Device dependencies and defer_wait remain owned
// by the original kernels and production scheduler.
void host_barrier(const fs::path &output, const std::string &phase, int rank, int ranks) {
    const auto prefix = output.parent_path() / phase;
    std::ofstream(prefix.string() + "." + std::to_string(rank)).put('1');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    for (int peer = 0; peer < ranks; ++peer) {
        while (!fs::exists(prefix.string() + "." + std::to_string(peer))) {
            if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Peer did not reach " + phase);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

struct Buffer {
    uint64_t address;
    size_t bytes;
    bool owned;
    bool repeat;
    std::vector<uint8_t> initial;
    std::vector<uint8_t> host;
};

void initialize(ChipWorker &worker, uint64_t address, size_t bytes, const std::vector<uint8_t> &value, bool repeat) {
    if (value.empty() || (repeat ? bytes % value.size() != 0 : bytes != value.size())) {
        throw std::runtime_error("Initializer size does not match original tensor");
    }
    std::vector<uint8_t> chunk(std::min(bytes, size_t{64} << 20));
    if (repeat) {
        for (size_t i = 0; i < chunk.size(); ++i)
            chunk[i] = value[i % value.size()];
    }
    for (size_t offset = 0; offset < bytes; offset += chunk.size()) {
        const size_t count = std::min(chunk.size(), bytes - offset);
        if (!repeat) {
            std::copy_n(value.data() + offset, count, chunk.data());
        }
        worker.copy_to(address + offset, reinterpret_cast<uint64_t>(chunk.data()), count);
    }
}

int run(int argc, char **argv) {
    if (argc == 5 && std::string(argv[1]) == "--assemble-orchestration") {
        std::ifstream manifest(argv[2]);
        auto signature = [&] {
            std::vector<ArgDirection> values;
            const auto count = read<size_t>(manifest);
            for (size_t i = 0; i < count; ++i)
                values.push_back(static_cast<ArgDirection>(read<int>(manifest)));
            return values;
        };
        const auto entry_signature = signature();
        const auto count = read<size_t>(manifest);
        std::vector<int32_t> ids;
        std::vector<std::vector<uint8_t>> children;
        for (size_t i = 0; i < count; ++i) {
            ids.push_back(read<int32_t>(manifest));
            const auto directions = signature();
            std::string path;
            if (!(manifest >> std::quoted(path))) throw std::runtime_error("Missing cached kernel path");
            const auto binary = read_binary(path);
            children.push_back(
                make_callable<CORE_MAX_TENSOR_ARGS>(directions.data(), directions.size(), binary.data(), binary.size())
            );
        }
        const auto orchestration = read_binary(argv[3]);
        const auto packed = make_chip_callable(
            entry_signature.data(), entry_signature.size(), "aicpu_orchestration_entry", orchestration.data(),
            orchestration.size(), ids.data(), children.data(), children.size(), "aicpu_orchestration_config"
        );
        std::ofstream target(argv[4], std::ios::binary);
        target.write(reinterpret_cast<const char *>(packed.data()), packed.size());
        if (!target) throw std::runtime_error("Cannot write original kernel bundle");
        return 0;
    }
    if ((argc == 5 || argc == 6) && std::string(argv[1]) == "--repack-orchestration") {
        const auto original = read_binary(argv[2]);
        auto orchestration = read_binary(argv[3]);
        if (original.size() < sizeof(ChipCallable)) throw std::runtime_error("Truncated callable");
        const auto *source = reinterpret_cast<const ChipCallable *>(original.data());
        // The host ELF shares this image with the model's machine code. A
        // smaller test entry must not move every kernel to new I-cache sets.
        if (orchestration.size() > source->binary_size()) {
            throw std::runtime_error(
                "Host adapter exceeds original orchestration slot; build it without debug symbols"
            );
        }
        orchestration.resize(source->binary_size(), 0);
        std::vector<std::vector<uint8_t>> children;
        for (int i = 0; i < source->child_count(); ++i) {
            const auto &child = source->child(i);
            const auto *begin = reinterpret_cast<const uint8_t *>(&child);
            const auto *end = begin + CoreCallable::binary_data_offset() + child.binary_size();
            if (begin < original.data() || end > original.data() + original.size()) {
                throw std::runtime_error("Truncated original kernel");
            }
            children.emplace_back(begin, end);
        }
        std::vector<ArgDirection> signature(source->signature_, source->signature_ + source->sig_count());
        if (argc == 6) signature.insert(signature.end(), std::stoul(argv[5]), ArgDirection::INOUT);
        const auto packed = make_chip_callable(
            signature.data(), signature.size(), source->func_name(), orchestration.data(), orchestration.size(),
            source->child_func_ids_, children.data(), children.size(), source->config_name()
        );
        const auto *replacement = reinterpret_cast<const ChipCallable *>(packed.data());
        for (int i = 0; i < source->child_count(); ++i) {
            if (replacement->child_offset(i) != source->child_offset(i)) {
                throw std::runtime_error("Original kernel image offsets changed");
            }
        }
        std::ofstream target(argv[4], std::ios::binary);
        target.write(reinterpret_cast<const char *>(packed.data()), packed.size());
        if (!target) throw std::runtime_error("Cannot write repacked callable");
        std::cout << "preserved_original_kernels=" << children.size() << '\n';
        return 0;
    }
    if (argc != 10) {
        std::cerr << "usage: native-runner BUNDLE RUNTIME_LIB DEVICE RANK ROOTINFO OUTPUT LEVEL WARMUP ROUNDS\n";
        return 2;
    }
    const fs::path bundle = argv[1], libraries = argv[2], output = argv[6];
    const int device = std::stoi(argv[3]), rank = std::stoi(argv[4]);
    const bool all_levels = std::string(argv[7]) == "all";
    const int level = all_levels ? 0 : std::stoi(argv[7]), warmup = std::stoi(argv[8]), rounds = std::stoi(argv[9]);
    const bool sliced = fs::exists(bundle / "slice.txt");
    if (sliced) {
        std::ifstream cut(bundle / "slice.txt");
        const auto first = read<std::string>(cut), last = read<std::string>(cut);
        setenv("DSV41_FIRST_TASK", first.c_str(), 1);
        setenv("DSV41_LAST_TASK", last.c_str(), 1);
        setenv("DSV41_PLAN_PATH", (bundle / ("plan-rank" + std::to_string(rank) + ".bin")).c_str(), 1);
    }
    if ((level != 0 && level != 1 && level != 3) || warmup < 0 || rounds < 1) {
        throw std::runtime_error("Invalid profiling level or iteration count");
    }
    fs::create_directories(output / "host");
    if (HostLogger::get_instance().adopt_state(HostLogger::get_instance().state()) != 0) {
        throw std::runtime_error("Cannot initialize the native process logger");
    }
    HostLogger::get_instance().set_log_directory((output / "host").c_str());
    HostLogger::get_instance().set_level(simpler::log::LogLevel::TIMING);
    std::ifstream manifest(bundle / ("rank" + std::to_string(rank) + ".txt"));
    if (read<std::string>(manifest) != "DSV41_NATIVE_V1") throw std::runtime_error("Invalid input manifest");
    const int ranks = read<int>(manifest);
    const size_t window_bytes = read<size_t>(manifest);
    if (rank < 0 || rank >= ranks) throw std::runtime_error("Rank outside original workload");
    ChipWorker worker;
    CallConfig config;
    const uint64_t heaps[] = {2ULL << 30, 2ULL << 30, 1ULL << 30, 512ULL << 20};
    for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
        config.runtime_env.ring_task_window[i] = 262144;
        config.runtime_env.ring_heap[i] = heaps[i];
        config.runtime_env.ring_dep_pool[i] = 131072;
    }
    worker.init(
        (libraries / "host_build_graph/libhost_runtime.so").string(),
        (libraries / "host_build_graph/libaicpu_kernel.so").string(),
        (libraries / "host_build_graph/aicore_kernel.o").string(),
        (libraries / "../dispatcher/libsimpler_aicpu_dispatcher.so").string(), device, &config
    );
    uint64_t comm = 0, comm_context = 0, window = 0;
    if (window_bytes != 0) {
        comm = worker.comm_init(rank, ranks, argv[5]);
        comm_context = worker.comm_alloc_windows(comm, window_bytes);
        window = worker.comm_get_local_window_base(comm);
        initialize(worker, window, window_bytes, {0}, true);
    }
    auto callable = read_binary(bundle / "callable.bin");
    const int callable_id = worker.register_callable(callable.data(), callable.size());
    ChipStorageTaskArgs args{};
    std::vector<Buffer> buffers;
    const size_t tensor_count = read<size_t>(manifest);
    buffers.reserve(tensor_count);
    for (size_t i = 0; i < tensor_count; ++i) {
        const std::string kind = read<std::string>(manifest);
        const auto dtype = static_cast<DataType>(read<int>(manifest));
        const size_t bytes = read<size_t>(manifest);
        const uint32_t ndims = read<uint32_t>(manifest);
        if (ndims == 0 || ndims > MAX_TENSOR_DIMS) throw std::runtime_error("Invalid tensor rank");
        uint32_t shape[MAX_TENSOR_DIMS]{};
        for (uint32_t dim = 0; dim < ndims; ++dim)
            shape[dim] = read<uint32_t>(manifest);
        Buffer buffer{};
        buffer.bytes = bytes;
        if (kind == "COMM") {
            const size_t offset = read<size_t>(manifest);
            if (offset > window_bytes || bytes > window_bytes - offset) {
                throw std::runtime_error("Communication view outside original window");
            }
            buffer.address = window + offset;
        } else {
            const bool repeat = read<std::string>(manifest) == "REPEAT";
            auto value = read_binary(bundle / read<std::string>(manifest));
            buffer.repeat = repeat;
            buffer.initial = value;
            if (kind == "HOST") {
                if (value.empty() || (repeat ? bytes % value.size() != 0 : bytes != value.size())) {
                    throw std::runtime_error("Host initializer size mismatch");
                }
                buffer.host.resize(bytes);
                for (size_t j = 0; j < bytes; ++j)
                    buffer.host[j] = value[repeat ? j % value.size() : j];
                buffer.address = reinterpret_cast<uint64_t>(buffer.host.data());
            } else if (kind == "DEVICE") {
                buffer.owned = true;
                buffer.address = worker.malloc(bytes);
                initialize(worker, buffer.address, bytes, value, repeat);
            } else {
                throw std::runtime_error("Unknown tensor storage kind");
            }
        }
        buffers.push_back(std::move(buffer));
        ChipTensor tensor;
        tensor.init_external(
            reinterpret_cast<void *>(buffers.back().address), bytes, shape, ndims, dtype,
            kind == "HOST" ? AddressSpace::HOST : AddressSpace::DEVICE, TensorTransfer::NONE
        );
        args.add_tensor(tensor);
    }
    const size_t scalar_count = read<size_t>(manifest);
    for (size_t i = 0; i < scalar_count; ++i) {
        const auto kind = read<std::string>(manifest);
        if (kind == "RANK") args.add_scalar(rank);
        else if (kind == "COMM") args.add_scalar(comm_context);
        else if (kind == "VALUE") args.add_scalar(read<uint64_t>(manifest));
        else throw std::runtime_error("Unknown scalar kind");
    }
    if (sliced) args.add_scalar(0);
    if (read<std::string>(manifest) != "VALIDATE_ZERO") throw std::runtime_error("Missing original output check");
    std::vector<size_t> checked;
    const auto check_count = read<size_t>(manifest);
    for (size_t i = 0; i < check_count; ++i) {
        const auto index = read<size_t>(manifest);
        if (index >= buffers.size()) throw std::runtime_error("Invalid output index");
        checked.push_back(index);
    }
    const auto *registered = reinterpret_cast<const ChipCallable *>(callable.data());
    auto writable = [&](size_t index) {
        return registered->sig(index) == ArgDirection::OUT || registered->sig(index) == ArgDirection::INOUT;
    };
    auto reset_inputs = [&] {
        for (size_t i = 0; i < buffers.size(); ++i) {
            const auto &buffer = buffers[i];
            if (buffer.owned && writable(i))
                initialize(worker, buffer.address, buffer.bytes, buffer.initial, buffer.repeat);
        }
        if (window_bytes) initialize(worker, window, window_bytes, {0}, true);
    };
    auto output_hashes = [&] {
        std::vector<uint64_t> hashes;
        std::vector<uint8_t> chunk(size_t{8} << 20);
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (!writable(i) || !buffers[i].host.empty()) continue;
            const auto &buffer = buffers[i];
            uint64_t hash = 14695981039346656037ULL;
            for (size_t offset = 0; offset < buffer.bytes; offset += chunk.size()) {
                const size_t count = std::min(chunk.size(), buffer.bytes - offset);
                worker.copy_from(reinterpret_cast<uint64_t>(chunk.data()), buffer.address + offset, count);
                for (size_t j = 0; j < count; ++j)
                    hash = (hash ^ chunk[j]) * 1099511628211ULL;
            }
            hashes.push_back(hash);
        }
        return hashes;
    };
    auto launch = [&](uint64_t phase, int profiling, const std::string &label) {
        config.enable_chip_swimlane = profiling;
        config.enable_dep_gen = profiling != 0;
        const auto directory = output / label;
        fs::create_directories(directory);
        if (directory.string().size() >= sizeof(config.output_prefix)) throw std::runtime_error("Output path too long");
        std::strcpy(config.output_prefix, directory.c_str());
        setenv("DSV41_MAPPING_PATH", (directory / "source-task-map.txt").c_str(), 1);
        if (sliced) args.scalars_[scalar_count] = phase;
        host_barrier(output, "launch-" + label, rank, ranks);
        worker.run(callable_id, &args, config);
        const auto mode = worker.last_scheduler_runtime_mode();
        if (mode != 1 && mode != 2) throw std::runtime_error("Production AICore scheduler was not selected");
        host_barrier(output, "complete-" + label, rank, ranks);
        return mode;
    };
    std::vector<uint64_t> reference;
    if (sliced) {
        launch(0, 0, "reference");
        reference = output_hashes();
        host_barrier(output, "reference-checked", rank, ranks);
    }
    for (int iteration = 0; iteration < warmup + rounds + (all_levels ? 2 : 0); ++iteration) {
        if (sliced) {
            reset_inputs();
            launch(1, 0, "prefix" + std::to_string(iteration));
        }
        int profiling = iteration < warmup ? 0 : level;
        if (all_levels && iteration >= warmup + rounds) profiling = iteration == warmup + rounds ? 1 : 3;
        const auto mode = launch(2, profiling, "run" + std::to_string(iteration));
        if (sliced && output_hashes() != reference)
            throw std::runtime_error("Isolated case differs from original prefix + case");
        for (auto index : checked) {
            const auto &buffer = buffers[index];
            std::vector<uint8_t> actual(buffer.bytes);
            worker.copy_from(reinterpret_cast<uint64_t>(actual.data()), buffer.address, actual.size());
            if (std::any_of(actual.begin(), actual.end(), [](uint8_t byte) {
                    return byte != 0;
                })) {
                throw std::runtime_error("Original zero-weight output check failed");
            }
        }
        std::cout << "run=" << iteration << " scheduler_mode=" << mode << " profiling=" << profiling
                  << " warmup=" << (iteration < warmup) << std::endl;
        host_barrier(output, "checked" + std::to_string(iteration), rank, ranks);
    }
    worker.unregister_callable(callable_id);
    for (const auto &buffer : buffers) {
        if (buffer.owned) worker.free(buffer.address);
    }
    if (comm != 0) {
        host_barrier(output, "finished", rank, ranks);
        worker.comm_destroy(comm);
    }
    worker.finalize();
    HostLogger::get_instance().flush();
    std::ofstream result(output / "result.json");
    result << "{\"passed\":true,\"device\":" << device << ",\"rank\":" << rank
           << ",\"original_kernels\":" << registered->child_count()
           << ",\"isolated_case\":" << (sliced ? "true" : "false")
           << ",\"validated_against_fused_prefix\":" << (sliced ? "true" : "false") << ",\"warmup\":" << warmup
           << ",\"rounds\":" << rounds << "}\n";
    return 0;
}
}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
