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

// HostLogger filtering: one Python-compatible threshold.
// Drives the singleton via a direct setter, captures stderr, and asserts on
// the buffered output.

#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "common/host_span.h"
#include "common/strace.h"
#include "host_log.h"

using simpler::log::LogLevel;

namespace {

struct CapturedStdio {
    std::string out;
    std::string err;
};

struct CannLogLevelCall {
    int count;
    int module_id;
    int level;
    int enable_event;
};

CannLogLevelCall g_cann_log_level_call{};
SimplerHostLogState g_shared_log_state{
    SIMPLER_HOST_LOG_STATE_ABI_VERSION,
    sizeof(SimplerHostLogState),
    static_cast<int32_t>(LogLevel::TIMING),
    0,
};

int capture_cann_log_level(int module_id, int level, int enable_event) {
    g_cann_log_level_call.count++;
    g_cann_log_level_call.module_id = module_id;
    g_cann_log_level_call.level = level;
    g_cann_log_level_call.enable_event = enable_event;
    return 0;
}

template <typename Fn>
CapturedStdio run_with_config(LogLevel level, Fn &&fn) {
    fflush(stdout);
    fflush(stderr);
    FILE *out_tmp = tmpfile();
    FILE *err_tmp = tmpfile();
    int saved_out = dup(fileno(stdout));
    int saved_err = dup(fileno(stderr));
    dup2(fileno(out_tmp), fileno(stdout));
    dup2(fileno(err_tmp), fileno(stderr));

    HostLogger::get_instance().set_level(level);

    fn();

    fflush(stdout);
    fflush(stderr);
    dup2(saved_out, fileno(stdout));
    dup2(saved_err, fileno(stderr));
    close(saved_out);
    close(saved_err);

    auto slurp = [](FILE *f) {
        std::string s;
        rewind(f);
        char buf[512];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            s.append(buf, n);
        }
        fclose(f);
        return s;
    };
    return {slurp(out_tmp), slurp(err_tmp)};
}

}  // namespace

TEST(HostLogTest, SharedStateBindingValidatesAbiAndOwnsThreshold) {
    SimplerHostLogState bad_version = g_shared_log_state;
    bad_version.abi_version++;
    EXPECT_NE(simpler_host_log_bind_state(nullptr), 0);
    EXPECT_NE(simpler_host_log_bind_state(&bad_version), 0);

    SimplerHostLogState bad_size = g_shared_log_state;
    bad_size.struct_size = sizeof(SimplerHostLogState) - 1;
    EXPECT_NE(simpler_host_log_bind_state(&bad_size), 0);

    SimplerHostLogState bad_threshold = g_shared_log_state;
    bad_threshold.threshold = 26;
    EXPECT_NE(simpler_host_log_bind_state(&bad_threshold), 0);

    g_shared_log_state.threshold = static_cast<int32_t>(LogLevel::ERROR);
    g_shared_log_state.clock_anchor_pid = 0;
    ASSERT_EQ(simpler_host_log_bind_state(&g_shared_log_state), 0);
    EXPECT_EQ(HostLogger::get_instance().state(), &g_shared_log_state);
    EXPECT_EQ(HostLogger::get_instance().level(), static_cast<int>(LogLevel::ERROR));
    EXPECT_FALSE(HostLogger::get_instance().is_enabled(LogLevel::WARN));

    HostLogger::get_instance().set_level(LogLevel::WARN);
    EXPECT_EQ(g_shared_log_state.threshold, static_cast<int32_t>(LogLevel::WARN));
}

TEST(HostLogTest, NulLevelMutesAllSeverities) {
    auto captured = run_with_config(LogLevel::NUL, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "err-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-msg");
        HostLogger::get_instance().log(LogLevel::INFO, "fn", "info-msg");
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "dbg-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err, "");
}

TEST(HostLogTest, HostSpanEnabledFollowsTimingVisibility) {
    for (const auto &[level, expected] : {
             std::pair{LogLevel::DEBUG, 1},
             std::pair{LogLevel::INFO, 1},
             std::pair{LogLevel::TIMING, 1},
             std::pair{LogLevel::WARN, 0},
             std::pair{LogLevel::ERROR, 0},
             std::pair{LogLevel::NUL, 0},
         }) {
        HostLogger::get_instance().set_level(level);
        EXPECT_EQ(unified_log_host_span_enabled(), expected);
    }
    HostLogger::get_instance().set_level(LogLevel::TIMING);
}

TEST(HostLogTest, ErrorLevelEmitsErrorOnly) {
    auto captured = run_with_config(LogLevel::ERROR, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "err-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_NE(captured.err.find("err-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("warn-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("timing-msg"), std::string::npos);
}

TEST(HostLogTest, TimingLevelKeepsTimingAndHigher) {
    auto captured = run_with_config(LogLevel::TIMING, [] {
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "debug-msg");
        HostLogger::get_instance().log(LogLevel::INFO, "fn", "info-msg");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "error-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err.find("debug-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("info-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("timing-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("warn-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("error-msg"), std::string::npos);
}

TEST(HostLogTest, CannLevelMappingSuppressesInfoAtDefault) {
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::DEBUG), 0);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::INFO), 1);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::TIMING), 2);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::WARN), 2);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::ERROR), 3);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::NUL), 4);
}

TEST(HostLogTest, CannConfigurationUsesGlobalModuleAndRespectsExternalOverride) {
    const char *old_env = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    const bool had_old_env = old_env != nullptr;
    const std::string old_value = had_old_env ? old_env : "";

    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    HostLogger::get_instance().set_level(LogLevel::TIMING);
    g_cann_log_level_call = {};
    HostLogger::get_instance().configure_cann_log_level(capture_cann_log_level);
    EXPECT_EQ(g_cann_log_level_call.count, 1);
    EXPECT_EQ(g_cann_log_level_call.module_id, -1);
    EXPECT_EQ(g_cann_log_level_call.level, 2);
    EXPECT_EQ(g_cann_log_level_call.enable_event, 0);

    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    g_cann_log_level_call = {};
    HostLogger::get_instance().configure_cann_log_level(capture_cann_log_level);
    EXPECT_EQ(g_cann_log_level_call.count, 0);

    if (had_old_env) {
        setenv("ASCEND_GLOBAL_LOG_LEVEL", old_value.c_str(), 1);
    } else {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    }
}

TEST(HostLogTest, EmitPrefixHasMonotonicNanosecondsAndTid) {
    auto captured = run_with_config(LogLevel::INFO, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "marker");
    });
    const size_t line_start = captured.err.rfind('\n', captured.err.find("marker"));
    const size_t prefix_start = line_start == std::string::npos ? 0 : line_start + 1;
    ASSERT_EQ(captured.err.compare(prefix_start, 9, "[mono_ns="), 0);
    const size_t prefix_end = captured.err.find(']', prefix_start);
    ASSERT_NE(prefix_end, std::string::npos);
    ASSERT_GT(prefix_end, prefix_start + 9);
    EXPECT_TRUE(
        std::all_of(
            captured.err.begin() + static_cast<std::ptrdiff_t>(prefix_start + 9),
            captured.err.begin() + static_cast<std::ptrdiff_t>(prefix_end), [](char c) {
                return c >= '0' && c <= '9';
            }
        )
    );
    // Thread-id segment "[T0x" must appear before the level tag.
    auto tid_pos = captured.err.find("][T0x", prefix_end);
    auto level_pos = captured.err.find("][ERROR]", tid_pos);
    ASSERT_NE(tid_pos, std::string::npos);
    ASSERT_NE(level_pos, std::string::npos);
    EXPECT_LT(tid_pos, level_pos);
    // Body still present.
    EXPECT_NE(captured.err.find("marker"), std::string::npos);
}

TEST(HostLogTest, TimingStartupEmitsOneClockAnchorPerProcess) {
    int log_pipe[2];
    ASSERT_EQ(pipe(log_pipe), 0);

    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        close(log_pipe[0]);
        if (dup2(log_pipe[1], STDERR_FILENO) < 0) _exit(2);
        close(log_pipe[1]);

        HostLogger::get_instance().set_level(LogLevel::TIMING);
        HostLogger::get_instance().log(LogLevel::TIMING, "child", "first-record");
        HostLogger::get_instance().log(LogLevel::TIMING, "child", "second-record");
        _exit(0);
    }

    close(log_pipe[1]);
    std::string captured;
    char buffer[1024];
    ssize_t count = 0;
    while ((count = read(log_pipe[0], buffer, sizeof(buffer))) > 0) {
        captured.append(buffer, static_cast<size_t>(count));
    }
    close(log_pipe[0]);

    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    const size_t anchor_pos = captured.find("[CLOCK_ANCHOR]");
    ASSERT_NE(anchor_pos, std::string::npos);
    EXPECT_EQ(captured.find("[CLOCK_ANCHOR]", anchor_pos + 1), std::string::npos);
    const size_t anchor_line_start = captured.rfind('\n', anchor_pos);
    const size_t anchor_level =
        captured.find("][TIMING]", anchor_line_start == std::string::npos ? 0 : anchor_line_start);
    ASSERT_NE(anchor_level, std::string::npos);
    EXPECT_LT(anchor_level, anchor_pos);
    const size_t first_record_pos = captured.find("first-record");
    const size_t second_record_pos = captured.find("second-record");
    ASSERT_NE(first_record_pos, std::string::npos);
    ASSERT_NE(second_record_pos, std::string::npos);
    EXPECT_LT(anchor_pos, first_record_pos);

    int anchor_pid = -1;
    long long mono_ns = 0;
    long long wall_ns = 0;
    ASSERT_EQ(
        sscanf(
            captured.c_str() + anchor_pos, "[CLOCK_ANCHOR] v=1 pid=%d mono_ns=%lld wall_ns=%lld", &anchor_pid, &mono_ns,
            &wall_ns
        ),
        3
    );
    EXPECT_EQ(anchor_pid, child);
    EXPECT_GT(mono_ns, 0);
    EXPECT_GT(wall_ns, 0);
}

TEST(HostLogTest, AllOutputGoesToStderr) {
    auto captured = run_with_config(LogLevel::DEBUG, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "error-output-marker");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-output-marker");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-output-marker");
        HostLogger::get_instance().log(LogLevel::INFO, "fn", "info-output-marker");
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "debug-output-marker");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_NE(captured.err.find("error-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("warn-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("timing-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("info-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("debug-output-marker"), std::string::npos);
}

TEST(HostLogTest, HostSpanEscapesDelimitersAndFitsAtomicPipeRecord) {
    const std::string name = "bad name\n[STRACE]=x";
    const std::string attributes = "run_id=7 role=worker\n[STRACE] injected=1 " + std::string(4096, 'x');
    const SimplerHostSpan span{SIMPLER_HOST_SPAN_ABI_VERSION,
                               sizeof(SimplerHostSpan),
                               7,
                               0x1234,
                               0,
                               0,
                               100,
                               25,
                               name.c_str(),
                               attributes.c_str()};

    auto captured = run_with_config(LogLevel::TIMING, [&] {
        unified_log_host_span(&span);
    });

    const size_t marker = captured.err.find("[STRACE]");
    ASSERT_NE(marker, std::string::npos);
    EXPECT_EQ(captured.err.find("[STRACE]", marker + 1), std::string::npos);
    EXPECT_NE(captured.err.find("name=bad%20name%0A%5BSTRACE%5D%3Dx"), std::string::npos);
    EXPECT_NE(captured.err.find("run_id=7 role=worker%0A%5BSTRACE%5D injected=1"), std::string::npos);
    const std::string record = captured.err.substr(marker);
    EXPECT_EQ(std::count(record.begin(), record.end(), '\n'), 1);
    EXPECT_LE(record.size(), static_cast<size_t>(_POSIX_PIPE_BUF));
    ASSERT_GE(record.size(), 2u);
    EXPECT_EQ(record[record.size() - 2], '~');
}

TEST(HostLogTest, HostSpanDirectoryWritesOneBufferedFilePerProcess) {
    char directory_template[] = "/tmp/simpler-host-strace-XXXXXX";
    char *directory = mkdtemp(directory_template);
    ASSERT_NE(directory, nullptr);
    ASSERT_EQ(setenv("SIMPLER_HOST_STRACE_DIR", directory, 1), 0);

    const SimplerHostSpan nested{
        SIMPLER_HOST_SPAN_ABI_VERSION, sizeof(SimplerHostSpan), 7, 0x1234, 1, 0, 100, 25, "chip.run.bind", "run_id=7"
    };
    const SimplerHostSpan root{
        SIMPLER_HOST_SPAN_ABI_VERSION, sizeof(SimplerHostSpan), 7, 0x1234, 0, 0, 90, 50, "chip.run", "run_id=7"
    };
    const auto captured = run_with_config(LogLevel::TIMING, [&] {
        unified_log_host_span(&nested);
        unified_log_host_span(&root);
    });
    ASSERT_EQ(unsetenv("SIMPLER_HOST_STRACE_DIR"), 0);
    EXPECT_EQ(captured.err.find("[STRACE]"), std::string::npos);

    const std::string path =
        std::string(directory) + "/host-strace." + std::to_string(static_cast<int>(getpid())) + ".log";
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find("name=chip.run.bind"), std::string::npos);
    EXPECT_NE(contents.find("name=chip.run"), std::string::npos);
    EXPECT_EQ(std::count(contents.begin(), contents.end(), '\n'), 2);
    EXPECT_EQ(unlink(path.c_str()), 0);
    EXPECT_EQ(rmdir(directory), 0);
}

TEST(HostLogTest, DisabledHostSpanProducesNoRecord) {
    const SimplerHostSpan span{
        SIMPLER_HOST_SPAN_ABI_VERSION, sizeof(SimplerHostSpan), 7, 0x1234, 0, 0, 100, 25, "host.dispatch",
        "run_id=7 role=scheduler"
    };

    auto captured = run_with_config(LogLevel::WARN, [&] {
        unified_log_host_span(&span);
    });

    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err, "");
}

TEST(HostLogTest, AllHostSpanEmitPathsPreserve64BitInvocationIds) {
    constexpr uint64_t invocation_id = (UINT64_C(1) << 32) + 7;
    constexpr uint64_t callable_hash = UINT64_C(0x1234);

    auto captured = run_with_config(LogLevel::TIMING, [] {
        simpler::strace::StraceContextScope context(invocation_id, callable_hash, 0);
        { simpler::strace::StraceScope scope("scope_path"); }
        simpler::strace::emit_host_span_at("explicit_path", 100, 25, 0);

        const SimplerHostSpan span{SIMPLER_HOST_SPAN_ABI_VERSION,
                                   sizeof(SimplerHostSpan),
                                   invocation_id,
                                   callable_hash,
                                   0,
                                   0,
                                   200,
                                   30,
                                   "c_abi_path",
                                   ""};
        unified_log_host_span(&span);
    });

    const std::string expected_inv = "inv=" + std::to_string(invocation_id);
    for (const char *name : {"scope_path", "explicit_path", "c_abi_path"}) {
        const size_t name_pos = captured.err.find(std::string("name=") + name);
        ASSERT_NE(name_pos, std::string::npos) << name;
        const size_t record_pos = captured.err.rfind("[STRACE]", name_pos);
        ASSERT_NE(record_pos, std::string::npos) << name;
        EXPECT_NE(captured.err.substr(record_pos, name_pos - record_pos).find(expected_inv), std::string::npos) << name;
    }
}

// A `%XX` escape is three bytes that only mean anything together, so a field
// that fills its budget exactly on one must lose the whole escape to the
// truncation marker. Overwriting just the last byte would leave `%0A` as `%0~`,
// which no decoder can read back.
TEST(HostLogTest, HostSpanTruncationDropsAWholeEscapeRatherThanItsLastByte) {
    // 3 (leading escape) + 186 + 3 (trailing escape) is exactly the 192-byte
    // attribute budget, so the next byte truncates on an escape boundary.
    const std::string attributes = "\n" + std::string(186, 'x') + "\ny";
    const SimplerHostSpan span{SIMPLER_HOST_SPAN_ABI_VERSION,
                               sizeof(SimplerHostSpan),
                               7,
                               0x1234,
                               0,
                               0,
                               100,
                               25,
                               "node.dispatch",
                               attributes.c_str()};

    auto captured = run_with_config(LogLevel::TIMING, [&] {
        unified_log_host_span(&span);
    });

    EXPECT_EQ(captured.err.find("%0~"), std::string::npos) << "truncation marker landed inside an escape";
    // The leading escape survives whole; the trailing one is gone entirely.
    EXPECT_NE(captured.err.find("dur=25 %0Axxx"), std::string::npos);
    EXPECT_EQ(std::count(captured.err.begin(), captured.err.end(), '%'), 1);
    ASSERT_GE(captured.err.size(), 2u);
    EXPECT_EQ(captured.err[captured.err.size() - 2], '~');
}

TEST(HostLogTest, ForkedProcessesEmitWholePipeRecords) {
    int log_pipe[2];
    int start_pipe[2];
    ASSERT_EQ(pipe(log_pipe), 0);
    ASSERT_EQ(pipe(start_pipe), 0);

    const long pipe_buf = fpathconf(log_pipe[1], _PC_PIPE_BUF);
    ASSERT_GT(pipe_buf, 256);
    const size_t payload_size = static_cast<size_t>(std::min<long>(pipe_buf - 256, 2048));
    constexpr int child_count = 16;
    constexpr int records_per_child = 128;

    std::vector<pid_t> children;
    for (int child = 0; child < child_count; ++child) {
        const pid_t pid = fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            close(log_pipe[0]);
            close(start_pipe[1]);
            char start;
            if (read(start_pipe[0], &start, 1) != 1 || dup2(log_pipe[1], STDERR_FILENO) < 0) {
                _exit(2);
            }
            close(start_pipe[0]);
            close(log_pipe[1]);

            HostLogger::get_instance().set_level(LogLevel::DEBUG);
            const std::string payload(payload_size, static_cast<char>('a' + child));
            for (int seq = 0; seq < records_per_child; ++seq) {
                HostLogger::get_instance().log(
                    LogLevel::ERROR, "fork_writer", "child=%d seq=%d payload=%s", child, seq, payload.c_str()
                );
            }
            _exit(0);
        }
        children.push_back(pid);
    }

    close(log_pipe[1]);
    close(start_pipe[0]);
    std::string captured;
    std::thread reader([&] {
        char buffer[8192];
        while (true) {
            const ssize_t count = read(log_pipe[0], buffer, sizeof(buffer));
            if (count > 0) {
                captured.append(buffer, static_cast<size_t>(count));
            } else if (count < 0 && errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        close(log_pipe[0]);
    });

    const std::string starts(child_count, 'x');
    EXPECT_EQ(write(start_pipe[1], starts.data(), starts.size()), static_cast<ssize_t>(starts.size()));
    close(start_pipe[1]);

    for (pid_t child : children) {
        int status = 0;
        const pid_t waited = waitpid(child, &status, 0);
        EXPECT_EQ(waited, child);
        if (waited == child) {
            EXPECT_TRUE(WIFEXITED(status));
            if (WIFEXITED(status)) {
                EXPECT_EQ(WEXITSTATUS(status), 0);
            }
        }
    }
    reader.join();

    std::vector<std::vector<bool>> seen(child_count, std::vector<bool>(records_per_child, false));
    std::set<int> anchor_pids;
    std::istringstream lines(captured);
    std::string line;
    int line_count = 0;
    constexpr char payload_marker[] = " payload=";
    while (std::getline(lines, line)) {
        const size_t anchor_pos = line.find("[CLOCK_ANCHOR]");
        if (anchor_pos != std::string::npos) {
            int anchor_pid = -1;
            ASSERT_EQ(sscanf(line.c_str() + anchor_pos, "[CLOCK_ANCHOR] v=1 pid=%d", &anchor_pid), 1);
            EXPECT_TRUE(anchor_pids.insert(anchor_pid).second);
            continue;
        }
        const size_t record_pos = line.find("child=");
        const size_t payload_pos = line.find(payload_marker);
        ASSERT_NE(record_pos, std::string::npos);
        ASSERT_NE(payload_pos, std::string::npos);
        ASSERT_EQ(line.find("child=", record_pos + 1), std::string::npos);

        int child = -1;
        int seq = -1;
        ASSERT_EQ(sscanf(line.c_str() + record_pos, "child=%d seq=%d", &child, &seq), 2);
        ASSERT_GE(child, 0);
        ASSERT_LT(child, child_count);
        ASSERT_GE(seq, 0);
        ASSERT_LT(seq, records_per_child);
        ASSERT_FALSE(seen[child][seq]);
        seen[child][seq] = true;

        const std::string payload = line.substr(payload_pos + sizeof(payload_marker) - 1);
        ASSERT_EQ(payload.size(), payload_size);
        EXPECT_TRUE(std::all_of(payload.begin(), payload.end(), [child](char value) {
            return value == static_cast<char>('a' + child);
        }));
        ++line_count;
    }
    EXPECT_EQ(line_count, child_count * records_per_child);
    EXPECT_EQ(anchor_pids.size(), static_cast<size_t>(child_count));
}
