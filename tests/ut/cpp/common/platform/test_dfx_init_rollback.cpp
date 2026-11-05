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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <vector>
#include <set>
#include <type_traits>
#include <unordered_set>

#include "host/args_dump_collector.h"
#include "host/chip_swimlane_collector.h"
#include "host/dep_gen_collector.h"
#include "host/pmu_collector.h"
#include "host/profiling_copy.h"
#include "host/scope_stats_collector.h"
#include "profiler_test_allocator.h"

namespace allocation_fault {
thread_local int countdown = -1;
thread_local bool fired = false;
void check() {
    if (countdown == 0) {
        countdown = -1;
        fired = true;
        throw std::bad_alloc();
    }
    if (countdown > 0) --countdown;
}
struct Pause {
    int saved = countdown;
    Pause() { countdown = -1; }
    ~Pause() { countdown = saved; }
};
}  // namespace allocation_fault

void *operator new(std::size_t size) {
    allocation_fault::check();
    if (void *p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void *operator new(std::size_t size, std::align_val_t alignment) {
    allocation_fault::check();
    void *p = nullptr;
    if (posix_memalign(&p, static_cast<size_t>(alignment), size == 0 ? 1 : size) == 0) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

int test_cores = 1;
int test_threads = 1;
int copy_attempts = 0;
int fail_copy = 0;

}  // namespace

int profiling_copy_to_device(volatile void *dst, const void *src, size_t size) {
    if (++copy_attempts == fail_copy) return 29;
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
    std::memcpy(const_cast<void *>(dst), src, size);
#else
    (void)dst;
    (void)src;
    (void)size;
#endif
    return 0;
}

int profiling_copy_from_device(volatile void *dst, const volatile void *src, size_t size) {
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
    std::memcpy(const_cast<void *>(dst), const_cast<const void *>(src), size);
#else
    (void)dst;
    (void)src;
    (void)size;
#endif
    return 0;
}

std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null() {
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
    return &profiling_copy_to_device_for_ops;
#else
    return {};
#endif
}

std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null() {
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
    return &profiling_copy_from_device_for_ops;
#else
    return {};
#endif
}

namespace {

struct Ledger {
    Ledger() { copy_attempts = fail_copy = 0; }
    int allocation_attempts = 0;
    int registration_attempts = 0;
    int fail_allocation = 0;
    int fail_registration = 0;
    int frees = 0;
    int free_attempts = 0;
    int fail_free = 0;
    int unregisters = 0;
    int unregister_attempts = 0;
    int fail_unregister = 0;
    bool acquired = false;
    std::unordered_set<void *> live;
    std::unordered_set<void *> registered;

    void *allocate(size_t size) {
        allocation_fault::Pause pause;
        if (++allocation_attempts == fail_allocation) return nullptr;
        void *p = profiler_test_calloc(size);
        EXPECT_NE(p, nullptr);
        if (p != nullptr) {
            EXPECT_TRUE(live.insert(p).second);
            acquired = true;
        }
        return p;
    }
    int release(void *p) {
        allocation_fault::Pause pause;
        if (++free_attempts == fail_free) return 41;
        EXPECT_EQ(registered.count(p), 0u) << "free preceded unregister";
        if (live.erase(p) != 1) {
            ADD_FAILURE() << "allocation released twice or not owned";
            return 31;
        }
        ++frees;
        std::free(p);
        return 0;
    }
    bool clean() const { return live.empty() && registered.empty(); }
    ~Ledger() {
        for (void *p : live)
            std::free(p);
    }
};

Ledger *active = nullptr;

int register_memory(void *p, size_t, int device, void **host) {
    allocation_fault::Pause pause;
    EXPECT_EQ(device, 7);
    if (++active->registration_attempts == active->fail_registration) return 19;
    EXPECT_TRUE(active->live.count(p));
    if (!active->registered.insert(p).second) return 23;
    *host = p;
    return 0;
}

int unregister_memory(void *p, int device) {
    allocation_fault::Pause pause;
    EXPECT_EQ(device, 7);
    if (++active->unregister_attempts == active->fail_unregister) return 43;
    EXPECT_EQ(active->registered.erase(p), 1u);
    ++active->unregisters;
    return 0;
}

template <class C>
struct Api {
    static int init(
        C &c, const profiling_common::ProfAllocCallback &a, profiling_common::ProfRegisterCallback r,
        const profiling_common::ProfFreeCallback &f, profiling_common::ProfUnregisterCallback u = unregister_memory
    ) {
        return c.init(test_threads, a, r, f, 7, u);
    }
};

template <>
struct Api<PmuCollector> {
    static int init(
        PmuCollector &c, const profiling_common::ProfAllocCallback &a, profiling_common::ProfRegisterCallback r,
        const profiling_common::ProfFreeCallback &f, profiling_common::ProfUnregisterCallback u = unregister_memory
    ) {
        return c.init(test_cores, test_threads, a, r, f, 7, u);
    }
};

template <>
struct Api<ArgsDumpCollector> {
    static int init(
        ArgsDumpCollector &c, const profiling_common::ProfAllocCallback &a, profiling_common::ProfRegisterCallback r,
        const profiling_common::ProfFreeCallback &f, profiling_common::ProfUnregisterCallback u = unregister_memory
    ) {
        return c.initialize(test_threads, 7, DumpArgsLevel::HYBRID, a, r, f, u);
    }
};

template <>
struct Api<ChipSwimlaneCollector> {
    static int init(
        ChipSwimlaneCollector &c, const profiling_common::ProfAllocCallback &a,
        profiling_common::ProfRegisterCallback r, const profiling_common::ProfFreeCallback &f,
        profiling_common::ProfUnregisterCallback u = unregister_memory
    ) {
        return c.initialize(test_cores, test_threads, 7, ChipSwimlaneLevel::TASK_TIMING, a, r, f, u);
    }
};

std::set<int> failure_positions(int count) {
    std::set<int> positions;
    for (int p = 1; p <= count; ++p)
        positions.insert(p);
    return positions;
}

template <class C>
class DfxInitRollback : public ::testing::Test {
    void SetUp() override {
        test_cores = test_threads = 1;
        allocation_fault::countdown = -1;
    }
};
using Collectors =
    ::testing::Types<PmuCollector, ArgsDumpCollector, ChipSwimlaneCollector, DepGenCollector, ScopeStatsCollector>;
TYPED_TEST_SUITE(DfxInitRollback, Collectors);

TYPED_TEST(DfxInitRollback, FailedAcquisitionRollsBackBeforeReturnAndCanRetry) {
    int allocation_count = 0, registration_count = 0;
    {
        Ledger ledger;
        active = &ledger;
        auto c = std::make_unique<TypeParam>();
        auto alloc = [&](size_t n) {
            return ledger.allocate(n);
        };
        auto free = [&](void *p) {
            return ledger.release(p);
        };
        ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
        allocation_count = ledger.allocation_attempts;
        registration_count = ledger.registration_attempts;
        c->finalize(unregister_memory, free);
        ASSERT_TRUE(ledger.clean());
    }
    this->RecordProperty("allocation_count", allocation_count);
    this->RecordProperty("registration_count", registration_count);
    ASSERT_GT(registration_count, 1);
    for (bool registration_fault : {false, true}) {
        for (int position : failure_positions(registration_fault ? registration_count : allocation_count)) {
            SCOPED_TRACE(::testing::Message() << "registration=" << registration_fault << " position=" << position);
            Ledger ledger;
            active = &ledger;
            auto c = std::make_unique<TypeParam>();
            auto alloc = [&](size_t n) {
                return ledger.allocate(n);
            };
            auto free = [&](void *p) {
                return ledger.release(p);
            };
            if (registration_fault) ledger.fail_registration = position;
            else ledger.fail_allocation = position;
            const int rc = Api<TypeParam>::init(*c, alloc, register_memory, free);
            EXPECT_NE(rc, 0);
            if constexpr (std::is_same_v<TypeParam, PmuCollector>) {
                if (registration_fault) EXPECT_EQ(rc, 19);
            }
            EXPECT_EQ(registration_fault ? ledger.registration_attempts : ledger.allocation_attempts, position);
            EXPECT_FALSE(c->is_initialized());
            EXPECT_TRUE(ledger.live.empty()) << "init returned before releasing allocations";
            EXPECT_TRUE(ledger.registered.empty()) << "init returned before unregistering mappings";
            if (!ledger.clean()) return;
            const int frees = ledger.frees, unregisters = ledger.unregisters;
            c->finalize(unregister_memory, free);
            EXPECT_EQ(ledger.frees, frees);
            EXPECT_EQ(ledger.unregisters, unregisters);
            ledger.fail_allocation = ledger.fail_registration = 0;
            ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
            EXPECT_TRUE(c->is_initialized());
            c->finalize(unregister_memory, free);
            EXPECT_TRUE(ledger.clean());
        }
    }
}

TYPED_TEST(DfxInitRollback, MissingUnregisterIsRejectedBeforeAllocation) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<TypeParam>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    EXPECT_NE(Api<TypeParam>::init(*c, alloc, register_memory, free, nullptr), 0);
    EXPECT_EQ(ledger.allocation_attempts, 0);
    EXPECT_FALSE(c->is_initialized());
}

TYPED_TEST(DfxInitRollback, FailedCopyRollsBackBeforeReturnAndCanRetry) {
    int copy_count = 0;
    {
        Ledger ledger;
        active = &ledger;
        auto c = std::make_unique<TypeParam>();
        auto alloc = [&](size_t n) {
            return ledger.allocate(n);
        };
        auto free = [&](void *p) {
            return ledger.release(p);
        };
        ASSERT_EQ(Api<TypeParam>::init(*c, alloc, nullptr, free), 0);
        copy_count = copy_attempts;
        c->finalize(nullptr, free);
        ASSERT_TRUE(ledger.clean());
    }
    this->RecordProperty("copy_count", copy_count);
    for (int position : failure_positions(copy_count)) {
        SCOPED_TRACE(position);
        Ledger ledger;
        active = &ledger;
        auto c = std::make_unique<TypeParam>();
        auto alloc = [&](size_t n) {
            return ledger.allocate(n);
        };
        auto free = [&](void *p) {
            return ledger.release(p);
        };
        fail_copy = position;
        EXPECT_NE(Api<TypeParam>::init(*c, alloc, nullptr, free), 0);
        EXPECT_EQ(copy_attempts, position);
        EXPECT_FALSE(c->is_initialized());
        ASSERT_TRUE(ledger.clean());
        fail_copy = 0;
        ASSERT_EQ(Api<TypeParam>::init(*c, alloc, nullptr, free), 0);
        c->finalize(nullptr, free);
        EXPECT_TRUE(ledger.clean());
    }
}

TYPED_TEST(DfxInitRollback, UnregisteredFailureNeverCallsUnregister) {
    Ledger ledger;
    active = &ledger;
    ledger.fail_allocation = 2;
    auto c = std::make_unique<TypeParam>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    EXPECT_NE(Api<TypeParam>::init(*c, alloc, nullptr, free), 0);
    EXPECT_EQ(ledger.allocation_attempts, 2);
    EXPECT_TRUE(ledger.clean());
    EXPECT_EQ(ledger.unregisters, 0);
    ledger.fail_allocation = 0;
    ASSERT_EQ(Api<TypeParam>::init(*c, alloc, nullptr, free), 0);
    c->finalize(nullptr, free);
    EXPECT_TRUE(ledger.clean());
    EXPECT_EQ(ledger.unregisters, 0);
}

struct CopySensitiveFree {
    Ledger *ledger;
    explicit CopySensitiveFree(Ledger *p) :
        ledger(p) {}
    CopySensitiveFree(const CopySensitiveFree &other) :
        ledger(other.ledger) {
        if (ledger->acquired) throw std::bad_alloc();
    }
    int operator()(void *p) const { return ledger->release(p); }
};

TYPED_TEST(DfxInitRollback, PublishingSuccessfulInitDoesNotCopyCallbacksAfterAcquisition) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<TypeParam>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    profiling_common::ProfFreeCallback free = CopySensitiveFree(&ledger);
    EXPECT_NO_THROW(EXPECT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0));
    ledger.acquired = false;
    c->finalize(unregister_memory, free);
    EXPECT_TRUE(ledger.clean());
}

TYPED_TEST(DfxInitRollback, SuccessfulInitLeavesFinalizeUsingItsCurrentQuarantineCallbacks) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<TypeParam>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
    const size_t live = ledger.live.size(), mappings = ledger.registered.size();
    int quarantine_frees = 0;
    auto quarantine = [&](void *) {
        ++quarantine_frees;
        return 0;
    };
    c->finalize(
        +[](void *, int) {
            return 0;
        },
        quarantine
    );
    EXPECT_EQ(ledger.frees, 0);
    EXPECT_EQ(ledger.unregisters, 0);
    EXPECT_EQ(ledger.live.size(), live);
    EXPECT_EQ(ledger.registered.size(), mappings);
    EXPECT_EQ(static_cast<size_t>(quarantine_frees), live);
    c->finalize(unregister_memory, free);
    EXPECT_EQ(ledger.frees, 0);
}

TEST(InitRollbackGuard, DirectAndManagerAllocationsArePairedAndReleasedOnce) {
    Ledger ledger;
    active = &ledger;
    profiling_common::BufferPoolManager<DepGenModule> manager;
    manager.set_memory_context({}, nullptr, nullptr, 0, 7);
    void *paired = ledger.allocate(64), *direct = ledger.allocate(64), *raw = ledger.allocate(64);
    void *host = nullptr;
    ASSERT_EQ(register_memory(paired, 64, 7, &host), 0);
    manager.register_mapping(paired, host, true);
    ASSERT_EQ(register_memory(direct, 64, 7, &host), 0);
    {
        profiling_common::InitRollbackGuard<decltype(manager)> guard(
            manager,
            [&](void *p) {
                return ledger.release(p);
            },
            unregister_memory, 7
        );
        guard.add_direct_ptr(paired, true);
        guard.add_direct_ptr(direct);
        guard.add_direct_ptr(direct, true);
        guard.add_direct_ptr(raw);
    }
    EXPECT_TRUE(ledger.clean());
    EXPECT_EQ(ledger.frees, 3);
    EXPECT_EQ(ledger.unregisters, 2);
}

TEST(InitRollbackGuard, CommitTransfersBothOwnershipKindsWithoutCleanup) {
    Ledger ledger;
    active = &ledger;
    profiling_common::BufferPoolManager<DepGenModule> manager;
    manager.set_memory_context({}, nullptr, nullptr, 0, 7);
    void *paired = ledger.allocate(64), *direct = ledger.allocate(64);
    void *host = nullptr;
    ASSERT_EQ(register_memory(paired, 64, 7, &host), 0);
    manager.register_mapping(paired, host, true);
    {
        profiling_common::InitRollbackGuard<decltype(manager)> guard(
            manager,
            [&](void *p) {
                return ledger.release(p);
            },
            unregister_memory, 7
        );
        guard.add_direct_ptr(direct);
        guard.commit();
    }
    EXPECT_EQ(ledger.frees, 0);
    EXPECT_EQ(ledger.unregisters, 0);
    manager.release_all_owned([&](void *p) {
        unregister_memory(p, 7);
        ledger.release(p);
    });
    ledger.release(direct);
    EXPECT_TRUE(ledger.clean());
}

template <class C>
class Inspectable : public C {
public:
    bool release_unproven() const { return this->manager_.release_unproven(); }
};

TYPED_TEST(DfxInitRollback, MultiCoreThreadAcquisitionFailuresRollBackAndRetry) {
    test_cores = 3;
    test_threads = 2;
    int allocations = 0, registrations = 0;
    {
        Ledger ledger;
        active = &ledger;
        auto c = std::make_unique<TypeParam>();
        auto alloc = [&](size_t n) {
            return ledger.allocate(n);
        };
        auto free = [&](void *p) {
            return ledger.release(p);
        };
        ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
        allocations = ledger.allocation_attempts;
        registrations = ledger.registration_attempts;
        c->finalize(unregister_memory, free);
        ASSERT_TRUE(ledger.clean());
    }
    this->RecordProperty("multi_core_allocations", allocations);
    this->RecordProperty("multi_core_registrations", registrations);
    for (bool registration_fault : {false, true}) {
        for (int position : failure_positions(registration_fault ? registrations : allocations)) {
            SCOPED_TRACE(::testing::Message() << "registration=" << registration_fault << " position=" << position);
            Ledger ledger;
            active = &ledger;
            auto c = std::make_unique<TypeParam>();
            auto alloc = [&](size_t n) {
                return ledger.allocate(n);
            };
            auto free = [&](void *p) {
                return ledger.release(p);
            };
            if (registration_fault) ledger.fail_registration = position;
            else ledger.fail_allocation = position;
            EXPECT_NE(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
            ASSERT_TRUE(ledger.clean());
            EXPECT_FALSE(c->is_initialized());
            ledger.fail_allocation = ledger.fail_registration = 0;
            ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
            c->finalize(unregister_memory, free);
            EXPECT_TRUE(ledger.clean());
        }
    }
}

TYPED_TEST(DfxInitRollback, AllocationExceptionsReturnErrorsAfterRollbackAndCanRetry) {
    test_cores = 3;
    test_threads = 2;
    for (int position = 0; position < 1000; ++position) {
        SCOPED_TRACE(position);
        Ledger ledger;
        active = &ledger;
        auto c = std::make_unique<TypeParam>();
        profiling_common::ProfAllocCallback alloc = [&](size_t n) {
            return ledger.allocate(n);
        };
        profiling_common::ProfFreeCallback free = [&](void *p) {
            return ledger.release(p);
        };
        allocation_fault::fired = false;
        allocation_fault::countdown = position;
        int rc = 0;
        bool escaped = false;
        try {
            rc = Api<TypeParam>::init(*c, alloc, register_memory, free);
        } catch (...) {
            escaped = true;
        }
        allocation_fault::countdown = -1;
        ASSERT_FALSE(escaped);
        if (!allocation_fault::fired) {
            ASSERT_EQ(rc, 0);
            c->finalize(unregister_memory, free);
            ASSERT_TRUE(ledger.clean());
            this->RecordProperty("allocation_exception_positions", position);
            return;
        }
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        ASSERT_TRUE(ledger.clean());
        EXPECT_FALSE(c->is_initialized());
        ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
        c->finalize(unregister_memory, free);
        EXPECT_TRUE(ledger.clean());
    }
    FAIL() << "allocation fault sweep did not reach a complete init";
}

TYPED_TEST(DfxInitRollback, EveryReleaseFailureRemainsAccountedAfterFinalize) {
    int count = 0;
    for (int position = 0; position <= count; ++position) {
        Ledger ledger;
        active = &ledger;
        auto c = std::make_unique<Inspectable<TypeParam>>();
        auto alloc = [&](size_t n) {
            return ledger.allocate(n);
        };
        auto free = [&](void *p) {
            return ledger.release(p);
        };
        ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
        count = ledger.allocation_attempts;
        ledger.fail_free = position;
        c->finalize(unregister_memory, free);
        EXPECT_EQ(c->release_unproven(), position != 0);
        EXPECT_EQ(ledger.live.size(), position != 0 ? 1u : 0u);
        EXPECT_TRUE(ledger.registered.empty());
        int attempts = ledger.free_attempts;
        c->finalize(unregister_memory, free);
        EXPECT_EQ(ledger.free_attempts, attempts);
    }
}

TYPED_TEST(DfxInitRollback, MissingFinalizeUnregisterPreservesResourcesForRetry) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<Inspectable<TypeParam>>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
    const size_t live = ledger.live.size(), mappings = ledger.registered.size();
    c->finalize(nullptr, free);
    EXPECT_TRUE(c->is_initialized());
    EXPECT_EQ(ledger.live.size(), live);
    EXPECT_EQ(ledger.registered.size(), mappings);
    EXPECT_EQ(ledger.frees, 0);
    EXPECT_TRUE(c->release_unproven());
    c->finalize(unregister_memory, free);
    EXPECT_TRUE(ledger.clean());
}

TYPED_TEST(DfxInitRollback, UnregisteredFinalizeDoesNotUseSuppliedUnregister) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<TypeParam>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    ASSERT_EQ(Api<TypeParam>::init(*c, alloc, nullptr, free, unregister_memory), 0);
    c->finalize(unregister_memory, free);
    EXPECT_TRUE(ledger.clean());
    EXPECT_EQ(ledger.unregisters, 0);
}

TEST(InitRollbackGuard, MixedResourceMetadataControlsUnregisterWithoutAllocatingDuringCleanup) {
    Ledger ledger;
    active = &ledger;
    profiling_common::BufferPoolManager<DepGenModule> manager;
    manager.set_memory_context({}, nullptr, nullptr, 0, 7);
    void *paired = ledger.allocate(64), *raw = ledger.allocate(64), *identity = ledger.allocate(64);
    void *host = nullptr;
    ASSERT_EQ(register_memory(paired, 64, 7, &host), 0);
    manager.register_mapping(paired, host, true);
    manager.register_device_allocation(raw);
    manager.register_mapping(identity, identity, false);
    EXPECT_EQ(manager.resolve_host_ptr(raw), nullptr);
    {
        profiling_common::InitRollbackGuard<decltype(manager)> guard(
            manager,
            [&](void *p) {
                return ledger.release(p);
            },
            unregister_memory, 7
        );
        allocation_fault::countdown = 0;
    }
    allocation_fault::countdown = -1;
    EXPECT_TRUE(ledger.clean());
    EXPECT_EQ(ledger.unregisters, 1);
    EXPECT_EQ(ledger.frees, 3);
}

TYPED_TEST(DfxInitRollback, FailedUnregisterDoesNotFreeRegisteredMemoryAndRemainsAccounted) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<Inspectable<TypeParam>>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    ASSERT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), 0);
    ledger.fail_unregister = 1;
    c->finalize(unregister_memory, free);
    EXPECT_EQ(ledger.live.size(), 1u);
    EXPECT_EQ(ledger.registered.size(), 1u);
    EXPECT_TRUE(c->release_unproven());
}

TYPED_TEST(DfxInitRollback, RollbackUnregisterFailureIsAccountedWithoutHidingOriginalError) {
    Ledger ledger;
    active = &ledger;
    auto c = std::make_unique<Inspectable<TypeParam>>();
    auto alloc = [&](size_t n) {
        return ledger.allocate(n);
    };
    auto free = [&](void *p) {
        return ledger.release(p);
    };
    ledger.fail_allocation = 2;
    ledger.fail_unregister = 1;
    EXPECT_EQ(Api<TypeParam>::init(*c, alloc, register_memory, free), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(c->is_initialized());
    EXPECT_EQ(ledger.live.size(), 1u);
    EXPECT_EQ(ledger.registered.size(), 1u);
    EXPECT_TRUE(c->release_unproven());
}

TEST(InitRollbackGuard, DirectTrackingAllocationFailureCleansTheNewPointerExactlyOnce) {
    Ledger ledger;
    active = &ledger;
    profiling_common::BufferPoolManager<DepGenModule> manager;
    manager.set_memory_context({}, nullptr, nullptr, 0, 7);
    void *p = ledger.allocate(64), *host = nullptr;
    ASSERT_EQ(register_memory(p, 64, 7, &host), 0);
    {
        profiling_common::InitRollbackGuard<decltype(manager)> guard(
            manager,
            [&](void *ptr) {
                return ledger.release(ptr);
            },
            unregister_memory, 7
        );
        allocation_fault::countdown = 0;
        bool caught = false;
        try {
            guard.add_direct_ptr(p, true);
        } catch (const std::bad_alloc &) {
            caught = true;
        }
        allocation_fault::countdown = -1;
        EXPECT_TRUE(caught);
        EXPECT_TRUE(ledger.clean());
    }
    EXPECT_EQ(ledger.frees, 1);
    EXPECT_EQ(ledger.unregisters, 1);
}

}  // namespace
