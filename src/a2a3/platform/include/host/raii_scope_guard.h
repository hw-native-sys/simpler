#ifndef PLATFORM_RAII_SCOPE_GUARD_H_
#define PLATFORM_RAII_SCOPE_GUARD_H_

#include <utility>

/**
 * RAII guard that runs a cleanup function on scope exit.
 *
 * The destructor invokes the stored callable unless dismiss() has been called,
 * allowing error paths to clean up automatically while success paths retain resources.
 *
 * Usage:
 *   auto guard = RAIIScopeGuard([&]() { allocator.free(ptr); });
 *   // ... do work that might fail ...
 *   guard.dismiss();  // success — keep the resource
 */
template <typename F>
class RAIIScopeGuard {
public:
    explicit RAIIScopeGuard(F fn) : fn_(std::move(fn)), active_(true) {}

    ~RAIIScopeGuard() {
        if (active_) {
            fn_();
        }
    }

    RAIIScopeGuard(RAIIScopeGuard&& other) noexcept : fn_(std::move(other.fn_)), active_(other.active_) {
        other.active_ = false;
    }

    RAIIScopeGuard(const RAIIScopeGuard&) = delete;
    RAIIScopeGuard& operator=(const RAIIScopeGuard&) = delete;

    void dismiss() noexcept { active_ = false; }

private:
    F fn_;
    bool active_;
};

#endif  // PLATFORM_RAII_SCOPE_GUARD_H_
