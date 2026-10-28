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

#include <chrono>
#include <future>
#include <thread>

#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "device_execution_owners.h"

#if defined(__linux__)
namespace {
int resident_registrations = 0;
}  // namespace
extern "C" int __real_pthread_atfork(void (*)(), void (*)(), void (*)());
extern "C" int __wrap_pthread_atfork(void (*prepare)(), void (*parent)(), void (*child)()) {
    ++resident_registrations;
    return __real_pthread_atfork(prepare, parent, child);
}
#endif

namespace {

int fork_entered_fd = -1;

void observe_fork_preparation() {
    const char entered = 'F';
    if (fork_entered_fd >= 0) (void)write(fork_entered_fd, &entered, sizeof(entered));
}

struct StartupObservation {
    int registrations = 0;
    pid_t child = -1;
    pid_t waited = -1;
    int status = 0;
};
StartupObservation startup;

class DeviceOwnerEnvironment : public testing::Environment {
public:
    void SetUp() override {
        if (observed_) return;
        observed_ = true;
#if defined(__linux__)
        startup.registrations = resident_registrations;
#endif
        // Run before any test case, independently of shuffle/filter/repeat.
        // The parent has not looked up the registry; the child reaches its
        // first lookup after resident module startup has completed.
        startup.child = fork();
        if (startup.child == 0) {
            alarm(3);
            auto &owners = simpler::worker::resident_device_execution_owners();
            const ChipWorker::ExecutionEnvironment child{static_cast<uint64_t>(getpid()), 100, 100, 0, false};
            const bool claimed = owners.claim(child);
            const bool refused = !owners.claim(child);
            owners.release(child);
            _exit(claimed && refused ? 0 : 1);
        }
        if (startup.child > 0) startup.waited = waitpid(startup.child, &startup.status, 0);
    }

private:
    bool observed_ = false;
};

[[maybe_unused]] testing::Environment *const owner_environment =
    testing::AddGlobalTestEnvironment(new DeviceOwnerEnvironment);

TEST(DeviceExecutionOwnersTest, ForkHooksPrecedeTheFirstResidentLookup) {
#if defined(__linux__)
    EXPECT_EQ(startup.registrations, 1) << "owner fork registration was deferred until concurrent first use";
#endif
    ASSERT_GT(startup.child, 0);
    ASSERT_EQ(startup.waited, startup.child);
    ASSERT_TRUE(WIFEXITED(startup.status));
    EXPECT_EQ(WEXITSTATUS(startup.status), 0);
}

TEST(DeviceExecutionOwnersTest, AContendedForkKeepsTheChildAndParentOwnerTablesUsable) {
    auto &owners = simpler::worker::resident_device_execution_owners();
    const ChipWorker::ExecutionEnvironment parent{static_cast<uint64_t>(getpid()), 101, 101, 0, false};
    ASSERT_TRUE(owners.claim(parent));

    int entered[2];
    ASSERT_EQ(pipe(entered), 0);
    fork_entered_fd = entered[1];
    // Prepare handlers run in reverse registration order. This observation
    // precedes the resident registry's attempt to take the held mutex.
    ASSERT_EQ(pthread_atfork(observe_fork_preparation, nullptr, nullptr), 0);

    owners.before_fork();
    std::promise<pid_t> fork_returned;
    auto returned = fork_returned.get_future();
    int child_status = 0;
    std::thread forking([&] {
        const pid_t pid = fork();
        if (pid == 0) {
            alarm(3);
            const ChipWorker::ExecutionEnvironment child{static_cast<uint64_t>(getpid()), 102, 102, 0, false};
            const bool claimed = owners.claim(child);
            const bool duplicate_refused = !owners.claim(child);
            owners.release(child);
            const bool reclaimed = owners.claim(child);
            owners.release(child);
            _exit(claimed && duplicate_refused && reclaimed ? 0 : 1);
        }
        fork_returned.set_value(pid);
        if (pid > 0) (void)waitpid(pid, &child_status, 0);
    });

    pollfd progress{entered[0], POLLIN, 0};
    const bool preparation_entered = poll(&progress, 1, 3000) == 1;
    const bool waited_for_owner = returned.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    owners.after_fork();
    forking.join();
    fork_entered_fd = -1;
    close(entered[0]);
    close(entered[1]);

    EXPECT_TRUE(preparation_entered);
    EXPECT_TRUE(waited_for_owner) << "fork copied the registry while another thread held its mutex";
    EXPECT_GE(returned.get(), 0);
    EXPECT_TRUE(WIFEXITED(child_status)) << "child blocked on an inherited registry mutex";
    if (WIFEXITED(child_status)) EXPECT_EQ(WEXITSTATUS(child_status), 0);
    EXPECT_FALSE(owners.claim(parent)) << "child cleanup released the parent's reservation";
    auto stale = parent;
    ++stale.incarnation;
    owners.release(stale);
    EXPECT_FALSE(owners.claim(parent)) << "a stale incarnation released the parent's reservation";
    owners.release(parent);
    EXPECT_TRUE(owners.claim(parent));
    owners.release(parent);
}

}  // namespace
