/**
 * Link-time stubs for platform APIs used by runtime headers.
 *
 * Provides x86-compatible implementations of functions declared in
 * platform headers (unified_log.h, device_time.h, common.h) so that
 * runtime data structures can be unit-tested on CI runners without
 * Ascend hardware or SDK.
 */

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// =============================================================================
// unified_log.h stubs (5 log-level functions)
// =============================================================================

extern "C" {

void unified_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[WARN]  %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_info(const char* /* func */, const char* /* fmt */, ...) {
    // Suppress info in tests
}

void unified_log_debug(const char* /* func */, const char* /* fmt */, ...) {
    // Suppress debug in tests
}

void unified_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ALWAYS] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

}  // extern "C"

// =============================================================================
// device_time.h stub
// =============================================================================

uint64_t get_sys_cnt_aicpu() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}

// =============================================================================
// common.h stubs (assert_impl, get_stacktrace, AssertionError)
// =============================================================================

std::string get_stacktrace(int /* skip_frames */) {
    return "<stacktrace not available in test stubs>";
}

class AssertionError : public std::runtime_error {
public:
    AssertionError(const char* condition, const char* file, int line)
        : std::runtime_error(std::string("Assertion failed: ") + condition + " at " + file + ":" +
                             std::to_string(line)),
          condition_(condition),
          file_(file),
          line_(line) {}

    const char* condition() const { return condition_; }
    const char* file() const { return file_; }
    int line() const { return line_; }

private:
    const char* condition_;
    const char* file_;
    int line_;
};

[[noreturn]] void assert_impl(const char* condition, const char* file, int line) {
    throw AssertionError(condition, file, line);
}
