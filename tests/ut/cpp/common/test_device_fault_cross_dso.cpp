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
/**
 * Two loadable host runtimes, one process monitor.
 *
 * The property under test is the one an instance-per-runtime-SO owner cannot
 * have: the driver's callback slot is registered **once** no matter how many
 * runtimes are loaded, no runtime's teardown retires a registration another is
 * still using, and unloading one leaves the monitor — which lives in this
 * executable, standing in for the module the interpreter never unloads —
 * intact and usable.
 */

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <string>

#include "host/device_fault_monitor.h"

namespace {

struct FakeDriver {
    int installs{0};
    int uninstalls{0};

    DeviceFaultMonitor::Ops ops() {
        return DeviceFaultMonitor::Ops{
            [this]() {
                ++installs;
                return 0;
            },
            [this]() {
                ++uninstalls;
                return 0;
            },
            []() {
                return 4242L;
            },
        };
    }
};

/** One loaded runtime, reached only through its exported C entry points. */
class Client {
public:
    explicit Client(const char *path) {
        dlerror();
        // RTLD_LOCAL is what ChipWorker uses, and it is why two runtimes'
        // identically named symbols do not collide — and why an owner defined
        // in one is invisible to the other.
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            const char *error = dlerror();
            last_error_ = error != nullptr ? error : "unknown dlopen error";
        }
    }

    ~Client() { close(); }

    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;

    bool loaded() const { return handle_ != nullptr; }
    const std::string &last_error() const { return last_error_; }

    void bind(DeviceFaultMonitor *monitor) {
        auto fn = reinterpret_cast<void (*)(void *)>(dlsym(handle_, "simpler_bind_device_fault_monitor"));
        ASSERT_NE(fn, nullptr) << "every host runtime built from this tree exports the binder";
        fn(monitor);
    }

    int has_monitor() { return call<int (*)()>("client_has_monitor")(); }
    int acquire() { return call<int (*)()>("client_acquire")(); }
    void release() { call<void (*)()>("client_release")(); }
    int installed() { return call<int (*)()>("client_installed")(); }
    unsigned references() { return call<unsigned (*)()>("client_references")(); }
    void report(unsigned device, unsigned stream, unsigned code) {
        call<void (*)(unsigned, unsigned, unsigned)>("client_report")(device, stream, code);
    }

    void close() {
        if (handle_ != nullptr) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

private:
    template <typename Fn>
    Fn call(const char *name) {
        return reinterpret_cast<Fn>(dlsym(handle_, name));
    }

    void *handle_{nullptr};
    std::string last_error_;
};

}  // namespace

TEST(DeviceFaultMonitorCrossDsoTest, TwoLoadedRuntimesRegisterTheDriverSlotExactlyOnce) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());

    Client first(TEST_DEVICE_FAULT_CLIENT_A_PATH);
    ASSERT_TRUE(first.loaded()) << first.last_error();
    Client second(TEST_DEVICE_FAULT_CLIENT_B_PATH);
    ASSERT_TRUE(second.loaded()) << second.last_error();

    first.bind(&monitor);
    second.bind(&monitor);
    ASSERT_EQ(first.has_monitor(), 1);
    ASSERT_EQ(second.has_monitor(), 1);

    ASSERT_EQ(first.acquire(), 0);
    EXPECT_EQ(driver.installs, 1);
    ASSERT_EQ(second.acquire(), 0);
    // The whole point: a second loaded runtime must not overwrite the slot the
    // first one registered. An owner defined inside each runtime SO would have
    // its own refcount here and install again.
    EXPECT_EQ(driver.installs, 1) << "two loaded runtimes must share one registration";
    EXPECT_EQ(monitor.references(), 2u);
    EXPECT_EQ(first.references(), 2u) << "both runtimes see the same refcount";
    EXPECT_EQ(second.references(), 2u);

    // One runtime finishing must not retire the other's callback.
    first.release();
    EXPECT_EQ(driver.uninstalls, 0) << "a runtime's teardown must not retire a registration another still uses";
    EXPECT_EQ(second.installed(), 1);

    second.release();
    EXPECT_EQ(driver.uninstalls, 1);
    EXPECT_FALSE(monitor.installed());
}

TEST(DeviceFaultMonitorCrossDsoTest, UnloadingOneRuntimeLeavesTheMonitorAndTheOtherRegistrationIntact) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());

    auto *first = new Client(TEST_DEVICE_FAULT_CLIENT_A_PATH);
    ASSERT_TRUE(first->loaded()) << first->last_error();
    Client second(TEST_DEVICE_FAULT_CLIENT_B_PATH);
    ASSERT_TRUE(second.loaded()) << second.last_error();

    first->bind(&monitor);
    second.bind(&monitor);
    ASSERT_EQ(first->acquire(), 0);
    ASSERT_EQ(second.acquire(), 0);

    first->report(1, 44, 507018);
    ASSERT_EQ(monitor.sequence(), 1u);

    // The runtime that reported goes away entirely, as ChipWorker::finalize
    // makes it. Its notice, the monitor and the surviving registration all
    // belong to this executable, so none of them goes with it.
    first->release();
    first->close();
    delete first;

    EXPECT_TRUE(monitor.installed()) << "the surviving runtime's registration must outlive its sibling's unload";
    EXPECT_EQ(second.installed(), 1);
    DeviceFaultNotice out;
    ASSERT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok)
        << "a notice must not be lost with the runtime that happened to report it";
    EXPECT_EQ(out.device_id, 1u);
    EXPECT_EQ(out.error_code, 507018u);

    // And the survivor keeps working through the same monitor.
    second.report(2, 43, 507015);
    ASSERT_EQ(monitor.read_notice(1, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, 2u);

    second.release();
    EXPECT_EQ(driver.uninstalls, 1);
}

TEST(DeviceFaultMonitorCrossDsoTest, AnUnboundRuntimeInstallsNothing) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());

    Client client(TEST_DEVICE_FAULT_CLIENT_A_PATH);
    ASSERT_TRUE(client.loaded()) << client.last_error();
    // A host runtime opened directly rather than through a loader reaches no
    // binder. That is a supported state, not a failure.
    EXPECT_EQ(client.has_monitor(), 0);
    EXPECT_NE(client.acquire(), 0);
    EXPECT_EQ(driver.installs, 0);
    EXPECT_EQ(monitor.references(), 0u);
}

TEST(DeviceFaultMonitorCrossDsoTest, OneMonitorPerLoadedRuntimeIsTheTopologyThisDesignRejects) {
    // What an owner defined *inside* each runtime SO would produce, stated as
    // an executable fact rather than an argument: two instances over one
    // driver slot, each with its own refcount.
    FakeDriver driver;
    DeviceFaultMonitor per_first_runtime(driver.ops());
    DeviceFaultMonitor per_second_runtime(driver.ops());

    Client first(TEST_DEVICE_FAULT_CLIENT_A_PATH);
    ASSERT_TRUE(first.loaded()) << first.last_error();
    Client second(TEST_DEVICE_FAULT_CLIENT_B_PATH);
    ASSERT_TRUE(second.loaded()) << second.last_error();
    first.bind(&per_first_runtime);
    second.bind(&per_second_runtime);

    ASSERT_EQ(first.acquire(), 0);
    ASSERT_EQ(second.acquire(), 0);
    // The second runtime silently overwrites the slot the first registered:
    // the driver keeps one callback, so the loser's is simply gone.
    EXPECT_EQ(driver.installs, 2);
    EXPECT_EQ(per_first_runtime.references(), 1u);
    EXPECT_EQ(per_second_runtime.references(), 1u);

    // And the first runtime's ordinary teardown retires the registration the
    // second one is still relying on.
    first.release();
    EXPECT_EQ(driver.uninstalls, 1);
    EXPECT_EQ(second.installed(), 1) << "the survivor still believes it is registered";
}
