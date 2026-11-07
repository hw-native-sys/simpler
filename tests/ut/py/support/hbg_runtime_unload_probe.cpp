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

#include "runtime_module.h"
#include <dlfcn.h>
#include <cstdio>
#include <memory>
#include <dirent.h>
#include <unistd.h>
int threads() {
    auto *dir = opendir("/proc/self/task");
    if (!dir) return -1;
    int count = 0;
    while (auto *entry = readdir(dir))
        if (entry->d_name[0] != '.') ++count;
    closedir(dir);
    return count;
}
int main(int argc, char **argv) {
    if (argc != 4 || !dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL)) return 2;
    const bool warm = argv[3][0] == '1';
    const int baseline = threads();
    for (int round = 0; round < 3; ++round) {
        auto first = std::make_unique<RuntimeModule>(argv[2]);
        auto second = std::make_unique<RuntimeModule>(argv[2]);
        auto prewarm = first->required<bool (*)()>("_Z22graph_recorder_prewarmv");
        if (warm && !prewarm()) return 3;
        first.reset();
        if (warm && !prewarm()) return 4;
        second.reset();
        Dl_info info{};
        const bool mapped = dladdr(reinterpret_cast<void *>(prewarm), &info) != 0;
        int remaining = threads();
        for (int attempt = 0; remaining != baseline && attempt < 100; ++attempt) {
            usleep(1000);
            remaining = threads();
        }
        std::printf("round=%d mapped=%d threads_before=%d threads_after=%d\n", round, mapped, baseline, remaining);
        if (mapped || remaining != baseline) return 5;
    }
}
