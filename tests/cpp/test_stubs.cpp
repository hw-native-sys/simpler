/**
 * Test stubs for platform and runtime dependencies.
 *
 * Provides simple implementations so that runtime code can be compiled
 * and tested on the host without linking against platform-specific backends.
 */

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// =============================================================================
// Unified logging stubs (common/unified_log.h)
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

void unified_log_info(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO]  %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void unified_log_debug(const char* /* func */, const char* /* fmt */, ...) {
    // Suppress debug output during tests
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
// common.h stubs (assert_impl, get_stacktrace)
// =============================================================================

std::string get_stacktrace(int /*skip_frames*/) {
    return "<stacktrace unavailable in unit test>";
}

[[noreturn]] void assert_impl(const char* condition, const char* file, int line) {
    fprintf(stderr, "Assertion failed: %s at %s:%d\n", condition, file, line);
    throw std::runtime_error(std::string("Assertion failed: ") + condition);
}
