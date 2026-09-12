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

#include "cpu_sim_context.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <future>
#include <iostream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << message << std::endl;
    std::_Exit(1);
}

void bind(int device = 0, uint32_t cluster = 0, uint32_t lane = 0) {
    pto_cpu_sim_bind_device(device);
    sim_context_set_cluster_id(cluster);
    sim_context_set_subblock_id(lane);
}

struct Kernel {
    using Event = void (*)(int);
    void *handle;
    Event signal;
    Event wait;
    Event legacy_signal;
    Event legacy_wait;
    bool (*rejects_mode)(int);
    bool (*rejects_count)(int);

    explicit Kernel(const char *path) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) fail(dlerror());
        auto lookup = [this](const char *name) {
            auto *symbol = dlsym(handle, name);
            if (symbol == nullptr) fail(std::string("missing DSO symbol: ") + name);
            return symbol;
        };
        auto inject = reinterpret_cast<void (*)(void *, void *)>(lookup("pto_sim_register_hooks"));
        inject(
            reinterpret_cast<void *>(pto_sim_get_subblock_id), reinterpret_cast<void *>(pto_sim_get_pipe_shared_state)
        );
        signal = reinterpret_cast<Event>(lookup("signal_event"));
        wait = reinterpret_cast<Event>(lookup("wait_event"));
        legacy_signal = reinterpret_cast<Event>(lookup("legacy_signal_event"));
        legacy_wait = reinterpret_cast<Event>(lookup("legacy_wait_event"));
        rejects_mode = reinterpret_cast<bool (*)(int)>(lookup("rejects_mode"));
        rejects_count = reinterpret_cast<bool (*)(int)>(lookup("rejects_count"));
    }
};

struct Waiter {
    std::future<void> completion;
    std::thread thread;

    template <class Fn>
    explicit Waiter(Fn fn) {
        std::promise<void> entered;
        auto start = entered.get_future();
        std::promise<void> done;
        completion = done.get_future();
        thread = std::thread([fn, entered = std::move(entered), done = std::move(done)]() mutable {
            entered.set_value();
            fn();
            done.set_value();
        });
        start.wait();
    }

    void blocked() {
        if (completion.wait_for(20ms) != std::future_status::timeout)
            fail("wait consumed an unrelated or missing credit");
    }

    void finish() {
        if (completion.wait_for(5s) != std::future_status::ready) fail("wait did not receive its matching credits");
        thread.join();
    }

    ~Waiter() {
        if (thread.joinable()) thread.join();
    }
};

void credits(Kernel &cube, Kernel &vector) {
    std::cout << "broadcast and queued credits" << std::endl;
    bind();
    for (int i = 0; i < 3; ++i)
        cube.signal(0);
    for (int lane = 0; lane < 2; ++lane) {
        bind(0, 0, lane);
        for (int i = 0; i < 3; ++i)
            vector.wait(0);
    }
    Waiter lane0([&] {
        bind();
        vector.wait(0);
    });
    Waiter lane1([&] {
        bind(0, 0, 1);
        vector.wait(0);
    });
    lane0.blocked();
    lane1.blocked();
    bind();
    cube.legacy_signal(0);
    lane0.finish();
    lane1.finish();

    std::cout << "joint AIV completion and event isolation" << std::endl;
    bind();
    vector.signal(3);
    vector.signal(3);
    Waiter joined([&] {
        bind();
        cube.wait(3);
    });
    joined.blocked();
    bind(0, 0, 1);
    vector.signal(7);
    joined.blocked();
    vector.signal(3);
    joined.finish();
    Waiter joined2([&] {
        bind();
        cube.legacy_wait(3);
    });
    joined2.blocked();
    bind(0, 0, 1);
    vector.legacy_signal(3);
    joined2.finish();
    bind();
    vector.signal(7);
    cube.wait(7);
}

void isolation(Kernel &cube, Kernel &vector) {
    std::cout << "device and cluster isolation" << std::endl;
    bind();
    cube.signal(5);
    Waiter device([&] {
        bind(2);
        vector.wait(5);
    });
    Waiter cluster([&] {
        bind(0, 1);
        vector.wait(5);
    });
    device.blocked();
    cluster.blocked();
    bind(2);
    cube.signal(5);
    device.finish();
    bind(2, 0, 1);
    vector.wait(5);
    bind(0, 1);
    cube.signal(5);
    cluster.finish();
    bind(0, 1, 1);
    vector.wait(5);
    for (int lane = 0; lane < 2; ++lane) {
        bind(0, 0, lane);
        vector.legacy_wait(5);
    }

    std::cout << "run reset" << std::endl;
    bind();
    cube.signal(9);
    vector.signal(10);
    bind(0, 0, 1);
    vector.signal(10);
    bind(2);
    cube.signal(9);
    bind();
    clear_cpu_sim_shared_storage();
    Waiter reset_vector([&] {
        bind();
        vector.wait(9);
    });
    Waiter reset_cube([&] {
        bind();
        cube.wait(10);
    });
    reset_vector.blocked();
    reset_cube.blocked();
    bind();
    cube.signal(9);
    vector.signal(10);
    bind(0, 0, 1);
    vector.wait(9);
    vector.signal(10);
    reset_vector.finish();
    reset_cube.finish();
    for (int lane = 0; lane < 2; ++lane) {
        bind(2, 0, lane);
        vector.wait(9);
    }
}

void pipeline(Kernel &cube, Kernel &vector) {
    std::cout << "four-event sparse pipeline with repeated epochs" << std::endl;
    std::array<std::array<int, 2>, 3> kv{}, probability{};
    std::array<int, 3> score{}, output{};
    auto valid = [](int block) {
        return block != 1 && block != 4;
    };
    auto input = [](int epoch, int block, int lane) {
        return 100 * epoch + 10 * block + lane;
    };
    Waiter aic([&] {
        bind();
        for (int epoch = 0; epoch < 6; ++epoch) {
            for (int tick = 0; tick < 9; ++tick) {
                if (tick < 7 && valid(tick)) {
                    cube.wait(0);
                    score[tick % 3] = kv[tick % 3][0] + kv[tick % 3][1];
                    cube.signal(1);
                }
                int block = tick - 2;
                if (block >= 0 && valid(block)) {
                    cube.wait(2);
                    output[block % 3] = probability[block % 3][0] + probability[block % 3][1];
                    cube.signal(3);
                }
            }
        }
    });
    auto run_vector = [&](int lane) {
        bind(0, 0, lane);
        for (int epoch = 0; epoch < 6; ++epoch) {
            for (int tick = 0; tick < 9; ++tick) {
                if (tick < 7 && valid(tick)) {
                    kv[tick % 3][lane] = input(epoch, tick, lane);
                    vector.signal(0);
                    vector.wait(1);
                    probability[tick % 3][lane] = score[tick % 3] + lane;
                    vector.signal(2);
                }
                int block = tick - 2;
                if (block >= 0 && valid(block)) {
                    vector.wait(3);
                    int expected = 2 * (input(epoch, block, 0) + input(epoch, block, 1)) + 1;
                    if (output[block % 3] != expected) fail("pipeline observed data before both producers completed");
                }
            }
        }
    };
    Waiter aiv0([&] {
        run_vector(0);
    });
    Waiter aiv1([&] {
        run_vector(1);
    });
    aic.finish();
    aiv0.finish();
    aiv1.finish();
}

int main(int argc, char **argv) {
    if (argc != 3) fail("expected AIC and AIV DSO paths");
    pto_cpu_sim_acquire_device(0);
    pto_cpu_sim_acquire_device(2);
    Kernel cube(argv[1]);
    Kernel vector(argv[2]);
    bind();
    for (int mode : {0, 1, 3}) {
        if (!cube.rejects_mode(mode) || !vector.rejects_mode(mode)) fail("unsupported mode was accepted");
    }
    for (int count : {0, 2, 15}) {
        if (!cube.rejects_count(count) || !vector.rejects_count(count)) fail("unsupported count was accepted");
    }
    credits(cube, vector);
    isolation(cube, vector);
    pipeline(cube, vector);
    pto_cpu_sim_release_device(0);
    pto_cpu_sim_release_device(2);
    dlclose(cube.handle);
    dlclose(vector.handle);
    std::cout << "PASS" << std::endl;
}
