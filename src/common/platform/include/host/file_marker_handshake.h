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

#ifndef SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_FILE_MARKER_HANDSHAKE_H
#define SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_FILE_MARKER_HANDSHAKE_H

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>

namespace file_marker_handshake {

enum class Stage {
    COMPLETE,
    ARRIVAL_PUBLISH,
    ARRIVAL_WAIT,
    DEPARTURE_PUBLISH,
    DEPARTURE_WAIT,
    CLEANUP,
    RELEASE_PUBLISH,
    RELEASE_WAIT,
    RELEASE_REMOVE,
};

enum class Error {
    NONE,
    MARKER_WRITE_FAILED,
    MARKER_REMOVE_FAILED,
    CLEANUP_FAILED,
    TIMED_OUT,
};

struct Result {
    Error error{Error::NONE};
    Stage stage{Stage::COMPLETE};
    int rank{-1};
    std::string path;
    std::error_code system_error;
    size_t failure_count{0};
    bool rootinfo_retired{true};

    bool ok() const { return error == Error::NONE; }
};

struct WaitOptions {
    std::chrono::milliseconds timeout{std::chrono::seconds(120)};
    std::chrono::milliseconds poll_interval{std::chrono::milliseconds(50)};
};

inline const char *stage_name(Stage stage) {
    switch (stage) {
    case Stage::COMPLETE:
        return "complete";
    case Stage::ARRIVAL_PUBLISH:
        return "arrival publish";
    case Stage::ARRIVAL_WAIT:
        return "arrival wait";
    case Stage::DEPARTURE_PUBLISH:
        return "departure publish";
    case Stage::DEPARTURE_WAIT:
        return "departure wait";
    case Stage::CLEANUP:
        return "cleanup";
    case Stage::RELEASE_PUBLISH:
        return "release publish";
    case Stage::RELEASE_WAIT:
        return "release wait";
    case Stage::RELEASE_REMOVE:
        return "release remove";
    }
    return "unknown";
}

inline std::string handshake_dir(const std::string &rootinfo_path) {
    const auto last_slash = rootinfo_path.rfind('/');
    if (last_slash == std::string::npos) return ".";
    return rootinfo_path.substr(0, last_slash);
}

inline std::string handshake_prefix(const std::string &rootinfo_path) {
    const auto last_slash = rootinfo_path.rfind('/');
    return last_slash == std::string::npos ? rootinfo_path : rootinfo_path.substr(last_slash + 1);
}

inline std::string run_token_hex(uint64_t run_token) {
    std::ostringstream oss;
    oss << std::hex << run_token;
    return oss.str();
}

inline std::string marker_path(const std::string &rootinfo_path, uint64_t run_token, const std::string &tag, int rank) {
    return handshake_dir(rootinfo_path) + "/barrier_" + handshake_prefix(rootinfo_path) + "_" + tag + "_" +
           run_token_hex(run_token) + "_" + std::to_string(rank) + ".ready";
}

inline bool publish_marker(const std::string &path) {
    // Waiters treat visibility as publication, so never expose a partial
    // marker. Each token/tag/rank path has exactly one serialized writer.
    const std::string tmp_path = path + ".tmp";
    std::ofstream marker(tmp_path, std::ios::trunc);
    marker << "1";
    marker.close();
    if (!marker.good() || std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }
    return true;
}

inline void record_cleanup_error(Result &result, const std::string &path, std::error_code ec) {
    if (!ec) return;
    if (result.ok()) {
        result.error = Error::CLEANUP_FAILED;
        result.stage = Stage::CLEANUP;
        result.path = path;
        result.system_error = ec;
    }
    ++result.failure_count;
}

inline Result cleanup(const std::string &rootinfo_path) {
    Result result;
    std::error_code ec;
    std::filesystem::remove(rootinfo_path, ec);
    result.rootinfo_retired = !ec;
    record_cleanup_error(result, rootinfo_path, ec);

    const std::string prefix = "barrier_" + handshake_prefix(rootinfo_path) + "_";
    const std::string dir = handshake_dir(rootinfo_path);
    std::filesystem::directory_iterator entry(dir, ec);
    const std::filesystem::directory_iterator end;
    while (!ec && entry != end) {
        const std::string name = entry->path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name.size() >= 6 && name.substr(name.size() - 6) == ".ready") {
            std::error_code entry_ec;
            const auto status = entry->symlink_status(entry_ec);
            if (!entry_ec && !std::filesystem::is_regular_file(status)) {
                entry_ec = std::make_error_code(std::errc::invalid_argument);
            }
            if (!entry_ec) std::filesystem::remove(entry->path(), entry_ec);
            record_cleanup_error(result, entry->path().string(), entry_ec);
        }
        entry.increment(ec);
    }
    record_cleanup_error(result, dir, ec);
    return result;
}

inline Result wait_for_path(
    const std::string &path, Stage stage, int rank, std::chrono::steady_clock::time_point deadline,
    std::chrono::milliseconds poll_interval
) {
    while (true) {
        std::ifstream marker(path);
        if (marker.good()) return {};
        if (std::chrono::steady_clock::now() >= deadline) {
            return {Error::TIMED_OUT, stage, rank};
        }
        std::this_thread::sleep_for(poll_interval);
    }
}

inline Result wait_for_marker(
    const std::string &rootinfo_path, uint64_t run_token, const char *tag, int rank, Stage stage,
    std::chrono::steady_clock::time_point deadline, std::chrono::milliseconds poll_interval
) {
    return wait_for_path(marker_path(rootinfo_path, run_token, tag, rank), stage, rank, deadline, poll_interval);
}

inline std::string release_path(const std::string &rootinfo_path, uint64_t run_token, int rank) {
    // Deliberately outside cleanup()'s barrier_<rootinfo>_*.ready namespace.
    // A subsequent rank-0 init must not remove this file before its follower
    // has observed and unlinked it. A crashed follower can leave one small,
    // token-scoped file; it may only be swept after every rank has joined a
    // newer generation, never by the init-entry cleanup.
    return rootinfo_path + ".destroy_released." + run_token_hex(run_token) + "." + std::to_string(rank);
}

inline Result
destroy_barrier(const std::string &rootinfo_path, int rank, int nranks, uint64_t run_token, WaitOptions options = {}) {
    constexpr const char *kArrivalTag = "destroy";
    constexpr const char *kDepartureTag = "destroy_departed";
    const auto deadline = std::chrono::steady_clock::now() + options.timeout;

    if (!publish_marker(marker_path(rootinfo_path, run_token, kArrivalTag, rank))) {
        return {Error::MARKER_WRITE_FAILED, Stage::ARRIVAL_PUBLISH, rank};
    }
    for (int peer_rank = 0; peer_rank < nranks; ++peer_rank) {
        Result result = wait_for_marker(
            rootinfo_path, run_token, kArrivalTag, peer_rank, Stage::ARRIVAL_WAIT, deadline, options.poll_interval
        );
        if (!result.ok()) return result;
    }

    if (!publish_marker(marker_path(rootinfo_path, run_token, kDepartureTag, rank))) {
        return {Error::MARKER_WRITE_FAILED, Stage::DEPARTURE_PUBLISH, rank};
    }
    if (rank != 0) return {};

    // A departure marker proves its follower has observed every arrival
    // marker. Successful rank-0 return makes a later marker sweep safe.
    for (int follower_rank = 1; follower_rank < nranks; ++follower_rank) {
        Result result = wait_for_marker(
            rootinfo_path, run_token, kDepartureTag, follower_rank, Stage::DEPARTURE_WAIT, deadline,
            options.poll_interval
        );
        if (!result.ok()) return result;
    }
    return {};
}

inline Result release_after_cleanup(
    const std::string &rootinfo_path, int rank, int nranks, uint64_t run_token, WaitOptions options = {}
) {
    if (rank == 0) {
        Result cleanup_result = cleanup(rootinfo_path);
        if (!cleanup_result.rootinfo_retired) return cleanup_result;

        Result first_error = cleanup_result;
        for (int follower_rank = 1; follower_rank < nranks; ++follower_rank) {
            if (!publish_marker(release_path(rootinfo_path, run_token, follower_rank)) && first_error.ok()) {
                first_error = {Error::MARKER_WRITE_FAILED, Stage::RELEASE_PUBLISH, follower_rank};
            }
        }
        return first_error;
    }

    const std::string path = release_path(rootinfo_path, run_token, rank);
    Result wait_result = wait_for_path(
        path, Stage::RELEASE_WAIT, rank, std::chrono::steady_clock::now() + options.timeout, options.poll_interval
    );
    if (!wait_result.ok()) return wait_result;

    std::error_code ec;
    const bool removed = std::filesystem::remove(path, ec);
    if (ec || !removed) return {Error::MARKER_REMOVE_FAILED, Stage::RELEASE_REMOVE, rank};
    return {};
}

// All callers have completed the allocation's release barrier before entry.
// A consumed marker is the last publication by that rank for this allocation.
inline Result retire_allocation(
    const std::string &rootinfo_path, int rank, int nranks, uint64_t run_token, uint64_t allocation_id,
    WaitOptions options = {}
) {
    const std::string prefix = "alloc_" + std::to_string(allocation_id) + "_";
    const std::string consumed = prefix + "release_consumed";
    const std::string own = marker_path(rootinfo_path, run_token, consumed, rank);
    if (!publish_marker(own)) return {Error::MARKER_WRITE_FAILED, Stage::DEPARTURE_PUBLISH, rank, own};
    if (rank != 0) return {};
    const auto deadline = std::chrono::steady_clock::now() + options.timeout;
    for (int peer = 0; peer < nranks; ++peer) {
        Result result = wait_for_marker(
            rootinfo_path, run_token, consumed.c_str(), peer, Stage::DEPARTURE_WAIT, deadline, options.poll_interval
        );
        if (!result.ok()) return result;
    }
    // Every release-barrier reader has departed; no rank reads consumed markers after publishing its own.
    constexpr std::array<const char *, 7> phases{"fabric_announce", "fabric_ready", "ipc_announce",    "ipc_p2p_ready",
                                                 "p2p_ready",       "release",      "release_consumed"};
    Result result;
    for (const auto *phase : phases) {
        for (int peer = 0; peer < nranks; ++peer) {
            const auto path = marker_path(rootinfo_path, run_token, prefix + phase, peer);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            record_cleanup_error(result, path, ec);
        }
    }
    return result;
}

}  // namespace file_marker_handshake

#endif  // SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_FILE_MARKER_HANDSHAKE_H
