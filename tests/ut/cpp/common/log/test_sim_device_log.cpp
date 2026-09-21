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

// Sim AICPU records use the bound HostLogger threshold and host envelope. The
// short records below also stay intact when many threads or forked chip workers
// share the stderr fallback.

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "aicpu/device_log.h"
#include "common/host_log_state.h"
#include "common/log_level.h"
#include "host_log.h"

namespace {

constexpr const char *kTags[] = {"DEBUG", "INFO", "TIMING", "WARN", "ERROR"};

SimplerHostLogState g_log_state{
    static_cast<int32_t>(simpler::log::LogLevel::ERROR), 0, 0, {}, 0, 0, nullptr, nullptr, 0, 0, 0, {}, 0,
};

// level_idx selects the compatibility entry point under test.
void emit(int level_idx, const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    switch (level_idx % 5) {
    case 0:
        dev_vlog_debug(func, fmt, ap);
        break;
    case 1:
        dev_vlog_info(func, fmt, ap);
        break;
    case 2:
        dev_vlog_timing(func, fmt, ap);
        break;
    case 3:
        dev_vlog_warn(func, fmt, ap);
        break;
    default:
        dev_vlog_error(func, fmt, ap);
        break;
    }
    va_end(ap);
}

std::string record(int level_idx, const char *func, const std::string &body) {
    return std::string("[") + kTags[level_idx % 5] + "] " + func + ": " + body;
}

void bind_level(simpler::log::LogLevel level) {
    g_log_state.threshold = static_cast<int32_t>(level);
    g_log_state.clock_anchor_pid = static_cast<int32_t>(getpid());
    g_log_state.log_directory_bound = 0;
    g_log_state.log_directory[0] = '\0';
    ASSERT_EQ(set_host_log_state(&g_log_state), 0);
    // This host-only executable supplies the process state. A production sim
    // AICPU DSO remains a bound consumer and cannot create the owner writer.
    ASSERT_EQ(HostLogger::get_instance().adopt_state(&g_log_state), 0);
    set_log_level(static_cast<int>(level));
}

// A failed assertion must not leave the process-global writer stopped for the
// next test. Normal paths call restart() after stderr capture is restored;
// early returns get a best-effort restart from the destructor.
class ScopedHostWriterRestart {
public:
    ~ScopedHostWriterRestart() {
        if (armed_) (void)HostLogger::get_instance().start_writer();
    }

    bool restart() {
        if (!HostLogger::get_instance().start_writer()) return false;
        armed_ = false;
        return true;
    }

private:
    bool armed_ = true;
};

// Redirect stderr onto a fresh pipe, drained from the moment it is installed.
//
// A pipe holds a bounded amount of unread data — 16 KiB on macOS, 64 KiB on
// Linux — and these tests emit ~19-23 KiB. Draining only after the writers
// finish therefore deadlocks wherever the total exceeds the capacity: the
// writers block in write(2) while the parent waits for them. The reader thread
// runs concurrently so the writers never block, whatever they emit.
//
// The reader owns the buffer through a shared_ptr, so returning `Capture` by
// value cannot leave the thread writing into a moved-from string. No logger
// producer is active at the fork boundary in the process tests below.
struct Capture {
    int read_fd = -1;
    int saved_stderr = -1;
    std::shared_ptr<std::string> out = std::make_shared<std::string>();
    std::thread reader;
};

Capture begin_capture() {
    int fds[2];
    EXPECT_EQ(pipe(fds), 0);
    fflush(stderr);
    Capture cap;
    cap.saved_stderr = dup(STDERR_FILENO);
    EXPECT_GE(cap.saved_stderr, 0);
    EXPECT_GE(dup2(fds[1], STDERR_FILENO), 0);
    close(fds[1]);  // STDERR_FILENO is now the sole write handle
    cap.read_fd = fds[0];
    cap.reader = std::thread([fd = cap.read_fd, out = cap.out] {
        char buf[4096];
        ssize_t n;
        while ((n = read(fd, buf, sizeof(buf))) > 0) {
            out->append(buf, static_cast<size_t>(n));
        }
    });
    return cap;
}

// Restore stderr, which drops this process's last write handle; with every child
// already reaped the reader then sees EOF and finishes.
std::string end_capture(Capture &cap) {
    EXPECT_TRUE(HostLogger::get_instance().flush());
    fflush(stderr);
    EXPECT_GE(dup2(cap.saved_stderr, STDERR_FILENO), 0);
    close(cap.saved_stderr);
    cap.reader.join();
    close(cap.read_fd);
    return std::move(*cap.out);
}

// Assert the captured stream is exactly `expected`, one intact record per
// physical line — a torn (merged or split) record matches no expected string.
void expect_intact(const std::string &captured, std::multiset<std::string> expected) {
    size_t start = 0;
    while (start < captured.size()) {
        size_t nl = captured.find('\n', start);
        ASSERT_NE(nl, std::string::npos) << "record missing terminating newline";
        const std::string line = captured.substr(start, nl - start);
        const size_t monotonic_end = line.find("][");
        ASSERT_NE(monotonic_end, std::string::npos) << "host envelope missing from: '" << line << "'";
        const size_t thread_end = line.find("][", monotonic_end + 2);
        ASSERT_NE(thread_end, std::string::npos) << "host envelope missing from: '" << line << "'";
        const std::string payload = line.substr(thread_end + 1);
        auto it = expected.find(payload);
        ASSERT_NE(it, expected.end()) << "torn or unexpected record: '" << line << "'";
        expected.erase(it);
        start = nl + 1;
    }
    EXPECT_TRUE(expected.empty()) << expected.size() << " records never arrived intact";
}

}  // namespace

// set_host_log_state reports its bind result so a sim AICPU SO that cannot bind
// fails device init instead of running with an unbound logger that silently
// drops every record.
TEST(SimDeviceLogTest, RejectsUnusableHostLogState) {
    SimplerHostLogState unusable = g_log_state;
    unusable.threshold = 26;

    EXPECT_NE(set_host_log_state(nullptr), 0);
    EXPECT_NE(set_host_log_state(&unusable), 0);
    EXPECT_EQ(set_host_log_state(&g_log_state), 0);
}

TEST(SimDeviceLogTest, UsesHostEnvelopeAndLiveBoundThreshold) {
    bind_level(simpler::log::LogLevel::ERROR);

    testing::internal::CaptureStderr();
    emit(3, "worker", "warn-hidden");
    emit(4, "worker", "error-visible");
    ASSERT_TRUE(HostLogger::get_instance().flush());
    std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_EQ(captured.find("warn-hidden"), std::string::npos);
    EXPECT_NE(captured.find("][ERROR] worker: error-visible\n"), std::string::npos);
    EXPECT_NE(captured.find("[mono_ns="), std::string::npos);
    EXPECT_NE(captured.find("][T0x"), std::string::npos);

    // The adapter reads the shared state on every query; no second push into a
    // private sim flag table is required.
    g_log_state.threshold = static_cast<int32_t>(simpler::log::LogLevel::WARN);
    testing::internal::CaptureStderr();
    emit(3, "worker", "warn-visible");
    ASSERT_TRUE(HostLogger::get_instance().flush());
    captured = testing::internal::GetCapturedStderr();
    EXPECT_NE(captured.find("][WARN] worker: warn-visible\n"), std::string::npos);
}

TEST(SimDeviceLogTest, BoundLogDirectoryTakesSimRecordsInsteadOfStderr) {
    char directory_template[] = "/tmp/simpler-sim-device-log-XXXXXX";
    char *directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);

    bind_level(simpler::log::LogLevel::DEBUG);
    HostLogger::get_instance().set_log_directory(directory);

    testing::internal::CaptureStderr();
    emit(4, "chip_worker", "device-record-to-file");
    // The writer owns the destination, so the record is on disk only once this
    // returns; nothing about its level makes it synchronous.
    ASSERT_TRUE(HostLogger::get_instance().flush());
    const std::string captured = testing::internal::GetCapturedStderr();
    EXPECT_EQ(captured.find("device-record-to-file"), std::string::npos)
        << "a bound directory is the logger's destination for device records too";

    // The destination is a property of the logger, so a sim device record lands
    // in the same per-process file as every host record.
    const std::string path = std::string(directory) + "/host." + std::to_string(static_cast<int>(getpid())) + ".log";
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    EXPECT_NE(contents.find("][ERROR] chip_worker: device-record-to-file\n"), std::string::npos);

    g_log_state.log_directory_bound = 0;
    g_log_state.log_directory[0] = '\0';
    EXPECT_EQ(unlink(path.c_str()), 0);
    EXPECT_EQ(rmdir(directory), 0);
}

// A bound directory is the destination, not a preference. A record it cannot
// take is dropped and counted rather than relocated to stderr — otherwise a
// directory that cannot be opened sends every record somewhere nobody reads
// while the drop counter still says zero, and "the log is complete" stops
// meaning "dropped_record_count is zero".
TEST(SimDeviceLogTest, UnwritableBoundDirectoryDropsRatherThanRelocating) {
    bind_level(simpler::log::LogLevel::DEBUG);
    HostLogger::get_instance().set_log_directory("/nonexistent/simpler-host-log-destination");

    const uint64_t before = HostLogger::get_instance().dropped_records();
    testing::internal::CaptureStderr();
    emit(4, "chip_worker", "record-with-no-destination");
    ASSERT_TRUE(HostLogger::get_instance().flush());
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_EQ(captured.find("record-with-no-destination"), std::string::npos)
        << "an unwritable bound directory must not silently relocate records to stderr";
    EXPECT_EQ(HostLogger::get_instance().dropped_records(), before + 1);

    g_log_state.log_directory_bound = 0;
    g_log_state.log_directory[0] = '\0';
}

TEST(SimDeviceLogTest, MultiThreadedRecordsStayIntact) {
    constexpr int kThreads = 4;
    constexpr int kPerThread = 200;

    bind_level(simpler::log::LogLevel::DEBUG);
    std::multiset<std::string> expected;
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; ++i) {
            char body[32];
            snprintf(body, sizeof(body), "t%d-r%04d", t, i);
            expected.insert(record(t, "worker", body));
        }
    }

    Capture cap = begin_capture();
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < kPerThread; ++i) {
                emit(t, "worker", "t%d-r%04d", t, i);
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }
    std::string captured = end_capture(cap);

    ASSERT_NO_FATAL_FAILURE(expect_intact(captured, std::move(expected)));
}

TEST(SimDeviceLogTest, ForkedProcessesEmitWholeRecords) {
    constexpr int kChildren = 8;
    constexpr int kPerChild = 100;

    bind_level(simpler::log::LogLevel::DEBUG);
    ASSERT_TRUE(HostLogger::get_instance().prepare_to_fork());
    ScopedHostWriterRestart writer_restart;
    std::multiset<std::string> expected;
    for (int c = 0; c < kChildren; ++c) {
        for (int i = 0; i < kPerChild; ++i) {
            char body[32];
            snprintf(body, sizeof(body), "c%d-r%03d", c, i);
            expected.insert(record(c, "chip_worker", body));
        }
    }

    Capture cap = begin_capture();
    std::vector<pid_t> pids;
    pids.reserve(kChildren);
    for (int c = 0; c < kChildren; ++c) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            g_log_state.clock_anchor_pid = static_cast<int32_t>(getpid());
            set_log_level(static_cast<int>(simpler::log::LogLevel::DEBUG));
            for (int i = 0; i < kPerChild; ++i) {
                emit(c, "chip_worker", "c%d-r%03d", c, i);
            }
            if (!HostLogger::get_instance().flush()) _exit(3);
            _exit(0);  // skip gtest/atexit teardown so nothing else hits the pipe
        }
        pids.push_back(pid);
    }
    for (pid_t pid : pids) {
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFEXITED(status)) << "child terminated abnormally";
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
    std::string captured = end_capture(cap);
    ASSERT_TRUE(writer_restart.restart());

    ASSERT_NO_FATAL_FAILURE(expect_intact(captured, std::move(expected)));
}

#ifdef F_SETPIPE_SZ
// The deadlock this guards against is capacity-dependent, so Linux's 64 KiB
// default pipe hides it while macOS's 16 KiB does not. Shrinking the pipe to one
// page makes the condition reproducible on either: the writers emit an order of
// magnitude more than fits, so they can only finish if something drains while
// they run. Without the concurrent reader this test hangs rather than fails,
// which is why the ctest timeout in CMakeLists is part of the same guard.
TEST(SimDeviceLogTest, WritersOutrunASmallPipeWithoutDeadlocking) {
    constexpr int kChildren = 4;
    constexpr int kPerChild = 400;

    bind_level(simpler::log::LogLevel::DEBUG);
    ASSERT_TRUE(HostLogger::get_instance().prepare_to_fork());
    ScopedHostWriterRestart writer_restart;
    std::multiset<std::string> expected;
    for (int c = 0; c < kChildren; ++c) {
        for (int i = 0; i < kPerChild; ++i) {
            char body[32];
            snprintf(body, sizeof(body), "c%d-r%03d", c, i);
            expected.insert(record(c, "chip_worker", body));
        }
    }

    Capture cap = begin_capture();
    if (fcntl(cap.read_fd, F_SETPIPE_SZ, 4096) < 0) {
        std::string discard = end_capture(cap);
        ASSERT_TRUE(writer_restart.restart());
        GTEST_SKIP() << "cannot shrink the pipe on this kernel";
    }

    std::vector<pid_t> pids;
    pids.reserve(kChildren);
    for (int c = 0; c < kChildren; ++c) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            g_log_state.clock_anchor_pid = static_cast<int32_t>(getpid());
            set_log_level(static_cast<int>(simpler::log::LogLevel::DEBUG));
            for (int i = 0; i < kPerChild; ++i) {
                emit(c, "chip_worker", "c%d-r%03d", c, i);
            }
            if (!HostLogger::get_instance().flush(5000)) _exit(3);
            _exit(0);
        }
        pids.push_back(pid);
    }
    for (pid_t pid : pids) {
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFEXITED(status)) << "child terminated abnormally";
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
    std::string captured = end_capture(cap);
    ASSERT_TRUE(writer_restart.restart());

    ASSERT_NO_FATAL_FAILURE(expect_intact(captured, std::move(expected)));
}
#endif  // F_SETPIPE_SZ
