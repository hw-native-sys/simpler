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

#include <csignal>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "host_log.h"

using simpler::log::LogLevel;

extern "C" void simpler_host_log_set_queue_hooks_for_test(void (*after_claim)(size_t), void (*before_gap_wait)());

namespace {

std::mutex g_gap_mutex;
std::condition_variable g_gap_cv;
bool g_first_claimed = false;
bool g_release_first = false;
bool g_writer_waiting_for_gap = false;

void pause_first_queue_claim(size_t position) {
    if (position != 0) return;
    std::unique_lock<std::mutex> lock(g_gap_mutex);
    g_first_claimed = true;
    g_gap_cv.notify_all();
    g_gap_cv.wait(lock, [] {
        return g_release_first;
    });
}

void observe_writer_gap_wait() {
    std::scoped_lock lock(g_gap_mutex);
    g_writer_waiting_for_gap = true;
    g_gap_cv.notify_all();
}

}  // namespace

TEST(HostLogNonblockingTest, DeferredWriterPreservesInitializationRecordsAndDropCount) {
    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR, /*defer_writer=*/true);

    const uint64_t before = logger.dropped_records();
    testing::internal::CaptureStderr();
    logger.log(LogLevel::ERROR, "initialization", "before-writer");
    const std::string captured = testing::internal::GetCapturedStderr();
    EXPECT_NE(captured.find("][ERROR] initialization: before-writer\n"), std::string::npos);
    EXPECT_EQ(logger.dropped_records(), before);

    const int saved_stderr = dup(STDERR_FILENO);
    ASSERT_GE(saved_stderr, 0);
    ASSERT_EQ(close(STDERR_FILENO), 0);
    logger.log(LogLevel::ERROR, "initialization", "failed-before-writer");
    ASSERT_GE(dup2(saved_stderr, STDERR_FILENO), 0);
    close(saved_stderr);
    ASSERT_EQ(logger.dropped_records(), before + 1);

    // The first writer start records this process PID. It is not a fork-child
    // transition and must not erase failures observed during initialization.
    ASSERT_TRUE(logger.start_writer());
    EXPECT_EQ(logger.dropped_records(), before + 1);
    EXPECT_TRUE(logger.prepare_to_fork());
}

// A total without a breakdown cannot tell "the queue is too small" from "the
// destination is broken", and those want opposite fixes. Pin that each drop lands
// in exactly one bucket and that the buckets sum to the total.
TEST(HostLogNonblockingTest, OutputFailureIsAttributedToTheOutputBucket) {
    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR, /*defer_writer=*/true);

    const uint64_t before_total = logger.dropped_records();
    const uint64_t before_output = logger.dropped_records(SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED);
    const uint64_t before_queue = logger.dropped_records(SIMPLER_HOST_LOG_DROP_QUEUE_FULL);
    const uint64_t before_claim = logger.dropped_records(SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED);

    // Closing stderr is the portable way to make the final write(2) fail; there
    // is no bound directory here, so stderr is the destination.
    const int saved_stderr = dup(STDERR_FILENO);
    ASSERT_GE(saved_stderr, 0);
    ASSERT_EQ(close(STDERR_FILENO), 0);
    logger.log(LogLevel::ERROR, "attribution", "unwritable");
    ASSERT_GE(dup2(saved_stderr, STDERR_FILENO), 0);
    close(saved_stderr);

    EXPECT_EQ(logger.dropped_records(), before_total + 1);
    EXPECT_EQ(logger.dropped_records(SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED), before_output + 1);
    EXPECT_EQ(logger.dropped_records(SIMPLER_HOST_LOG_DROP_QUEUE_FULL), before_queue);
    EXPECT_EQ(logger.dropped_records(SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED), before_claim);

    uint64_t summed = 0;
    for (int reason = 0; reason < SIMPLER_HOST_LOG_DROP_REASON_COUNT; ++reason) {
        summed += logger.dropped_records(static_cast<SimplerHostLogDropReason>(reason));
    }
    EXPECT_EQ(summed, logger.dropped_records()) << "the breakdown must account for every drop";

    ASSERT_TRUE(logger.start_writer());
    EXPECT_TRUE(logger.prepare_to_fork());
}

// The counters die with the process, so a reader holding only the log has to be
// told in the log. prepare_to_fork() is the boundary every path that stops
// logging passes through, and it reports a growth rather than a running total.
TEST(HostLogNonblockingTest, QuiesceWritesTheLossBreakdownIntoTheLog) {
    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR, /*defer_writer=*/true);

    const int saved_stderr = dup(STDERR_FILENO);
    ASSERT_GE(saved_stderr, 0);
    ASSERT_EQ(close(STDERR_FILENO), 0);
    logger.log(LogLevel::ERROR, "loss_summary", "lost-one");
    ASSERT_GE(dup2(saved_stderr, STDERR_FILENO), 0);
    close(saved_stderr);

    ASSERT_TRUE(logger.start_writer());
    testing::internal::CaptureStderr();
    ASSERT_TRUE(logger.prepare_to_fork());
    const std::string first = testing::internal::GetCapturedStderr();
    EXPECT_NE(first.find("[HOSTLOG_DROPS] v=1 "), std::string::npos) << first;
    EXPECT_NE(first.find("output_failed="), std::string::npos) << first;

    // Nothing new was dropped, so a second quiesce must stay silent rather than
    // restating the same losses at every boundary.
    ASSERT_TRUE(logger.start_writer());
    testing::internal::CaptureStderr();
    ASSERT_TRUE(logger.prepare_to_fork());
    const std::string second = testing::internal::GetCapturedStderr();
    EXPECT_EQ(second.find("[HOSTLOG_DROPS]"), std::string::npos) << second;
}

// Quiescing a live writer re-enters the same no-sink window a process starts in,
// so the synchronous fallback has to be usable on the way out of it. A process
// that initializes a second hierarchical Worker after a first one would
// otherwise lose every record of that second initialization: the stop flag the
// quiesce raises rejects both the queue and the fallback.
TEST(HostLogNonblockingTest, QuiescingALiveWriterKeepsTheSynchronousFallback) {
    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR, /*defer_writer=*/true);
    ASSERT_TRUE(logger.start_writer());
    logger.log(LogLevel::ERROR, "first_worker", "owned-writer-record");
    ASSERT_TRUE(logger.flush());

    const uint64_t before = logger.dropped_records();
    ASSERT_TRUE(logger.prepare_to_fork());

    testing::internal::CaptureStderr();
    logger.log(LogLevel::ERROR, "second_worker", "after-quiesce-record");
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("][ERROR] second_worker: after-quiesce-record\n"), std::string::npos);
    EXPECT_EQ(logger.dropped_records(), before);

    ASSERT_TRUE(logger.start_writer());
    EXPECT_TRUE(logger.prepare_to_fork());
}

TEST(HostLogNonblockingTest, WriterSleepsWhenAdjacentProducersPublishOutOfOrder) {
    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR);
    {
        std::scoped_lock lock(g_gap_mutex);
        g_first_claimed = false;
        g_release_first = false;
        g_writer_waiting_for_gap = false;
    }
    simpler_host_log_set_queue_hooks_for_test(pause_first_queue_claim, observe_writer_gap_wait);

    std::thread first([&logger] {
        logger.log(LogLevel::ERROR, "producer", "first-reserved");
    });
    {
        std::unique_lock<std::mutex> lock(g_gap_mutex);
        EXPECT_TRUE(g_gap_cv.wait_for(lock, std::chrono::seconds(1), [] {
            return g_first_claimed;
        }));
    }

    std::thread second([&logger] {
        logger.log(LogLevel::ERROR, "producer", "second-published");
    });
    second.join();
    {
        std::unique_lock<std::mutex> lock(g_gap_mutex);
        EXPECT_TRUE(g_gap_cv.wait_for(lock, std::chrono::seconds(1), [] {
            return g_writer_waiting_for_gap;
        })) << "writer did not return to sem_wait for the missing earlier slot";
        g_release_first = true;
    }
    g_gap_cv.notify_all();
    first.join();
    simpler_host_log_set_queue_hooks_for_test(nullptr, nullptr);

    EXPECT_TRUE(logger.flush());
    EXPECT_TRUE(logger.prepare_to_fork());
}

TEST(HostLogNonblockingTest, FullStderrPipeDoesNotStallProducer) {
    int stderr_pipe[2];
    ASSERT_EQ(pipe(stderr_pipe), 0);
#ifdef F_SETPIPE_SZ
    ASSERT_GE(fcntl(stderr_pipe[0], F_SETPIPE_SZ, 4096), 0);
#endif

    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        close(stderr_pipe[0]);
        if (dup2(stderr_pipe[1], STDERR_FILENO) < 0) _exit(10);
        close(stderr_pipe[1]);

        alarm(2);
        HostLogger::get_instance().set_level(LogLevel::ERROR);
        const std::string payload(400, 'x');
        for (int i = 0; i < 10000; ++i) {
            HostLogger::get_instance().log(LogLevel::ERROR, "producer", "record=%d %s", i, payload.c_str());
        }
        if (HostLogger::get_instance().dropped_records() == 0) _exit(11);
        // A blocked writer cannot be joined, so a later hierarchical startup
        // must fail boundedly instead of proceeding to fork with it alive.
        _exit(HostLogger::get_instance().prepare_to_fork(10) ? 12 : 0);
    }

    close(stderr_pipe[1]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    close(stderr_pipe[0]);

    ASSERT_TRUE(WIFEXITED(status)) << "producer stalled on stderr (wait status " << status << ")";
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(HostLogNonblockingTest, HardWriteFailureIncrementsDropCounter) {
    int stderr_pipe[2];
    ASSERT_EQ(pipe(stderr_pipe), 0);
    close(stderr_pipe[0]);

    const int saved_stderr = dup(STDERR_FILENO);
    ASSERT_GE(saved_stderr, 0);
    auto previous_sigpipe = std::signal(SIGPIPE, SIG_IGN);
    ASSERT_NE(previous_sigpipe, SIG_ERR);
    ASSERT_GE(dup2(stderr_pipe[1], STDERR_FILENO), 0);
    close(stderr_pipe[1]);

    HostLogger::get_instance().set_level(LogLevel::ERROR);
    const uint64_t before = HostLogger::get_instance().dropped_records();
    HostLogger::get_instance().log(LogLevel::ERROR, "producer", "write-must-fail");
    EXPECT_TRUE(HostLogger::get_instance().flush());

    ASSERT_GE(dup2(saved_stderr, STDERR_FILENO), 0);
    close(saved_stderr);
    std::signal(SIGPIPE, previous_sigpipe);
    EXPECT_EQ(HostLogger::get_instance().dropped_records(), before + 1);
}

TEST(HostLogNonblockingTest, ExistingWriterIsJoinedBeforeALaterFork) {
    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR);
    logger.log(LogLevel::ERROR, "producer", "before-second-worker");
    ASSERT_TRUE(logger.prepare_to_fork());

    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        logger.set_level(LogLevel::NUL);
        _exit(logger.flush() ? 0 : 12);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    // The parent can install a fresh writer after the fork boundary.
    EXPECT_TRUE(logger.start_writer());
    EXPECT_TRUE(logger.prepare_to_fork());
}

TEST(HostLogNonblockingTest, ConcurrentProducersCanBeQuiescedSafely) {
    const int saved_stderr = dup(STDERR_FILENO);
    ASSERT_GE(saved_stderr, 0);
    const int null_fd = open("/dev/null", O_WRONLY);
    ASSERT_GE(null_fd, 0);
    ASSERT_GE(dup2(null_fd, STDERR_FILENO), 0);
    close(null_fd);

    HostLogger &logger = HostLogger::get_instance();
    logger.set_level(LogLevel::ERROR);
    std::atomic<bool> running{true};
    std::vector<std::thread> producers;
    for (int index = 0; index < 4; ++index) {
        producers.emplace_back([&logger, &running, index] {
            uint64_t sequence = 0;
            while (running.load(std::memory_order_acquire)) {
                logger.log(LogLevel::ERROR, "producer", "thread=%d sequence=%lu", index, sequence++);
            }
        });
    }

    for (int iteration = 0; iteration < 20; ++iteration) {
        ASSERT_TRUE(logger.prepare_to_fork(1000));
        ASSERT_TRUE(logger.start_writer());
    }
    running.store(false, std::memory_order_release);
    for (std::thread &producer : producers)
        producer.join();
    EXPECT_TRUE(logger.prepare_to_fork(1000));

    ASSERT_GE(dup2(saved_stderr, STDERR_FILENO), 0);
    close(saved_stderr);
}
