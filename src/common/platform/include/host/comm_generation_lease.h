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
#ifndef SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_COMM_GENERATION_LEASE_H
#define SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_COMM_GENERATION_LEASE_H

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <string>

namespace file_marker_handshake {

class GenerationLease {
public:
    GenerationLease() = default;
    GenerationLease(const GenerationLease &) = delete;
    GenerationLease &operator=(const GenerationLease &) = delete;
    ~GenerationLease() {
        if (fd_ >= 0) close(fd_);
    }

    bool acquire(const std::string &rootinfo) {
        const auto path = std::filesystem::path(rootinfo);
        const auto generation = path.parent_path();
        const auto registry = generation.parent_path();
        if (registry.filename() != "simpler-comm-" + std::to_string(getuid()) ||
            generation.filename().string().rfind("gen-", 0) != 0)
            return true;
        if (fd_ >= 0 || path.filename() != "rootinfo.bin") return false;
        const auto lease_path = (generation / ".lease").string();
        const int fd = open(lease_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return false;
        struct stat held{}, named{};
        // A reclaimer may hold EX while an arriving rank waits for SH. The
        // path identity check rejects a lease unlinked during that wait.
        if (flock(fd, LOCK_SH) != 0 || fstat(fd, &held) != 0 || lstat(lease_path.c_str(), &named) != 0 ||
            !S_ISREG(held.st_mode) || held.st_uid != getuid() || held.st_ino != named.st_ino ||
            held.st_dev != named.st_dev || held.st_nlink != 1) {
            close(fd);
            return false;
        }
        fd_ = fd;
        return true;
    }

private:
    int fd_{-1};
};

}  // namespace file_marker_handshake
#endif  // SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_COMM_GENERATION_LEASE_H
