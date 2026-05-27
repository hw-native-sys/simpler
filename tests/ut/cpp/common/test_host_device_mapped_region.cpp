#include "host_device_comm/host_device_mapped_region.h"

#include <errno.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

struct CacheOpsRecorder {
    int flush_count = 0;
    int invalidate_count = 0;
    std::vector<std::string> events;
};

int allocate_heap_region(
    DeviceContextHandle, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, static_cast<size_t>(total_bytes)) != 0) {
        return -ENOMEM;
    }
    platform->resource = ptr;
    platform->release = [](HostDeviceMappedRegionPlatform *p) {
        std::free(p->resource);
        p->resource = nullptr;
    };
    *host_base = ptr;
    *device_base = ptr;
    return 0;
}

int allocate_heap_region_with_cache_ops(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
) {
    int rc = allocate_heap_region(ctx, total_bytes, platform, host_base, device_base);
    if (rc != 0) {
        return rc;
    }
    platform->cache_ops_cookie = ctx;
    platform->flush_host_range = [](HostDeviceMappedRegionPlatform *p, void *, uint64_t) {
        auto *recorder = static_cast<CacheOpsRecorder *>(p->cache_ops_cookie);
        ++recorder->flush_count;
        recorder->events.push_back("flush");
        return 0;
    };
    platform->invalidate_host_range = [](HostDeviceMappedRegionPlatform *p, void *, uint64_t) {
        auto *recorder = static_cast<CacheOpsRecorder *>(p->cache_ops_cookie);
        ++recorder->invalidate_count;
        recorder->events.push_back("invalidate");
        return 0;
    };
    return 0;
}

struct ReleaseRecorder {
    int release_count = 0;
};

struct ReleaseState {
    ReleaseRecorder *recorder = nullptr;
    void *resource = nullptr;
};

void release_recorded_heap_region(HostDeviceMappedRegionPlatform *p) {
    auto *state = static_cast<ReleaseState *>(p->resource);
    ++state->recorder->release_count;
    std::free(state->resource);
    delete state;
    p->resource = nullptr;
}

int allocate_region_without_host_base(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
) {
    int rc = allocate_heap_region(ctx, total_bytes, platform, host_base, device_base);
    if (rc != 0) {
        return rc;
    }
    auto *recorder = static_cast<ReleaseRecorder *>(ctx);
    void *resource = platform->resource;
    platform->resource = new ReleaseState{recorder, resource};
    platform->release = release_recorded_heap_region;
    *host_base = nullptr;
    return 0;
}

int allocate_region_with_failing_initial_flush(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
) {
    int rc = allocate_heap_region(ctx, total_bytes, platform, host_base, device_base);
    if (rc != 0) {
        return rc;
    }
    auto *recorder = static_cast<ReleaseRecorder *>(ctx);
    void *resource = platform->resource;
    platform->resource = new ReleaseState{recorder, resource};
    platform->release = release_recorded_heap_region;
    platform->flush_host_range = [](HostDeviceMappedRegionPlatform *, void *, uint64_t) {
        return -EIO;
    };
    return 0;
}

}  // namespace

TEST(HostDeviceMappedRegion, PublicAbiLayoutIsStable) {
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, host_data_ptr), 0u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, device_data_ptr), 8u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, data_bytes), 16u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, host_signal_ptr), 24u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, device_signal_ptr), 32u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, signal_count), 40u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, total_bytes), 48u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionInfo, flags), 56u);
    EXPECT_EQ(sizeof(HostDeviceMappedRegionInfo), 64u);
}

TEST(HostDeviceMappedRegion, InternalLayoutIsStable) {
    EXPECT_EQ(sizeof(HostDeviceMappedRegionHeader), 64u);
    EXPECT_EQ(alignof(HostDeviceMappedRegionHeader), 64u);
    EXPECT_EQ(sizeof(HostDeviceMappedRegionSignalSlot), 64u);
    EXPECT_EQ(alignof(HostDeviceMappedRegionSignalSlot), 64u);
    EXPECT_EQ(offsetof(HostDeviceMappedRegionSignalSlot, value), 0u);
}

TEST(HostDeviceMappedRegion, RejectsInvalidConfig) {
    HostDeviceMappedRegionHandle region = reinterpret_cast<HostDeviceMappedRegionHandle>(0x1);
    HostDeviceMappedRegionConfig cfg{0, 1, 0};
    EXPECT_EQ(
        host_device_mapped_region_open_common(
            reinterpret_cast<DeviceContextHandle>(0x10), &cfg, &region, allocate_heap_region
        ),
        -EINVAL
    );
    EXPECT_EQ(region, nullptr);

    cfg = HostDeviceMappedRegionConfig{16, 0, 0};
    EXPECT_EQ(
        host_device_mapped_region_open_common(
            reinterpret_cast<DeviceContextHandle>(0x10), &cfg, &region, allocate_heap_region
        ),
        -EINVAL
    );
    EXPECT_EQ(region, nullptr);

    cfg = HostDeviceMappedRegionConfig{16, 1, 1};
    EXPECT_EQ(
        host_device_mapped_region_open_common(
            reinterpret_cast<DeviceContextHandle>(0x10), &cfg, &region, allocate_heap_region
        ),
        -EINVAL
    );
    EXPECT_EQ(region, nullptr);
}

TEST(HostDeviceMappedRegion, ReleasesBackendResourceWhenOpenValidationFailsAfterAllocate) {
    ReleaseRecorder recorder;
    auto ctx = reinterpret_cast<DeviceContextHandle>(&recorder);
    HostDeviceMappedRegionConfig cfg{16, 1, 0};
    HostDeviceMappedRegionHandle region = reinterpret_cast<HostDeviceMappedRegionHandle>(0x1);

    EXPECT_EQ(host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_region_without_host_base), -EIO);
    EXPECT_EQ(region, nullptr);
    EXPECT_EQ(recorder.release_count, 1);
}

TEST(HostDeviceMappedRegion, ReleasesBackendResourceWhenInitialFlushFails) {
    ReleaseRecorder recorder;
    auto ctx = reinterpret_cast<DeviceContextHandle>(&recorder);
    HostDeviceMappedRegionConfig cfg{16, 1, 0};
    HostDeviceMappedRegionHandle region = reinterpret_cast<HostDeviceMappedRegionHandle>(0x1);

    EXPECT_EQ(
        host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_region_with_failing_initial_flush), -EIO
    );
    EXPECT_EQ(region, nullptr);
    EXPECT_EQ(recorder.release_count, 1);
}

TEST(HostDeviceMappedRegion, OpensZeroInitializedRegionAndReportsInfo) {
    auto ctx = reinterpret_cast<DeviceContextHandle>(0x20);
    HostDeviceMappedRegionConfig cfg{17, 2, 0};
    HostDeviceMappedRegionHandle region = nullptr;
    ASSERT_EQ(host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_heap_region), 0);
    ASSERT_NE(region, nullptr);

    HostDeviceMappedRegionInfo info{};
    ASSERT_EQ(host_device_mapped_region_info_common(ctx, region, &info), 0);
    EXPECT_NE(info.host_data_ptr, 0u);
    EXPECT_EQ(info.host_data_ptr, info.device_data_ptr);
    EXPECT_NE(info.host_signal_ptr, 0u);
    EXPECT_EQ(info.host_signal_ptr, info.device_signal_ptr);
    EXPECT_EQ(info.data_bytes, 17u);
    EXPECT_EQ(info.signal_count, 2u);
    EXPECT_EQ(info.flags, 0u);
    EXPECT_EQ(info.total_bytes, 64u + 2u * 64u + 64u);

    auto *host_base = reinterpret_cast<uint8_t *>(info.host_signal_ptr - sizeof(HostDeviceMappedRegionHeader));
    auto *header = reinterpret_cast<HostDeviceMappedRegionHeader *>(host_base);
    EXPECT_EQ(header->magic, HDMR_MAGIC);
    EXPECT_EQ(header->version, HDMR_VERSION);
    EXPECT_EQ(header->flags, 0u);
    EXPECT_EQ(header->signal_count, 2u);
    EXPECT_EQ(header->signal_offset, 64u);
    EXPECT_EQ(header->data_offset, 64u + 2u * 64u);
    EXPECT_EQ(header->data_bytes, 17u);
    EXPECT_EQ(header->total_bytes, info.total_bytes);

    std::vector<uint8_t> out(17, 0xAA);
    ASSERT_EQ(host_device_mapped_region_datacopy_region2h_common(ctx, region, 0, out.data(), out.size()), 0);
    EXPECT_EQ(out, std::vector<uint8_t>(17, 0));

    EXPECT_EQ(host_device_mapped_region_close_common(ctx, region), 0);
}

TEST(HostDeviceMappedRegion, DatacopyValidatesBoundsAndRoundTripsBytes) {
    auto ctx = reinterpret_cast<DeviceContextHandle>(0x30);
    HostDeviceMappedRegionConfig cfg{8, 1, 0};
    HostDeviceMappedRegionHandle region = nullptr;
    ASSERT_EQ(host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_heap_region), 0);

    const uint8_t input[4] = {1, 2, 3, 4};
    ASSERT_EQ(host_device_mapped_region_datacopy_h2region_common(ctx, region, 2, input, sizeof(input)), 0);
    uint8_t output[8] = {};
    ASSERT_EQ(host_device_mapped_region_datacopy_region2h_common(ctx, region, 0, output, sizeof(output)), 0);
    const uint8_t expected[8] = {0, 0, 1, 2, 3, 4, 0, 0};
    EXPECT_EQ(std::memcmp(output, expected, sizeof(expected)), 0);

    EXPECT_EQ(host_device_mapped_region_datacopy_h2region_common(ctx, region, 8, input, 0), 0);
    EXPECT_EQ(host_device_mapped_region_datacopy_h2region_common(ctx, region, 8, input, 1), -EINVAL);
    EXPECT_EQ(host_device_mapped_region_datacopy_region2h_common(ctx, region, 9, output, 0), -EINVAL);

    EXPECT_EQ(host_device_mapped_region_close_common(ctx, region), 0);
}

TEST(HostDeviceMappedRegion, InvokesPlatformCacheOpsAroundHostAccess) {
    CacheOpsRecorder recorder;
    auto ctx = reinterpret_cast<DeviceContextHandle>(&recorder);
    HostDeviceMappedRegionConfig cfg{8, 2, 0};
    HostDeviceMappedRegionHandle region = nullptr;
    ASSERT_EQ(host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_heap_region_with_cache_ops), 0);
    EXPECT_GT(recorder.flush_count, 0);

    recorder.events.clear();
    const uint8_t input[4] = {1, 2, 3, 4};
    EXPECT_EQ(host_device_mapped_region_datacopy_h2region_common(ctx, region, 0, input, sizeof(input)), 0);
    ASSERT_FALSE(recorder.events.empty());
    EXPECT_EQ(recorder.events.back(), "flush");

    recorder.events.clear();
    EXPECT_EQ(host_device_mapped_region_notify_common(ctx, region, 0, 1), 0);
    ASSERT_FALSE(recorder.events.empty());
    EXPECT_EQ(recorder.events.back(), "flush");

    recorder.events.clear();
    EXPECT_EQ(host_device_mapped_region_wait_common(ctx, region, 0, 1, 0), 0);
    ASSERT_FALSE(recorder.events.empty());
    EXPECT_EQ(recorder.events.front(), "invalidate");

    recorder.events.clear();
    uint8_t output[4] = {};
    EXPECT_EQ(host_device_mapped_region_datacopy_region2h_common(ctx, region, 0, output, sizeof(output)), 0);
    ASSERT_FALSE(recorder.events.empty());
    EXPECT_EQ(recorder.events.front(), "invalidate");

    EXPECT_EQ(host_device_mapped_region_close_common(ctx, region), 0);
}

TEST(HostDeviceMappedRegion, NotifyWaitUsesMonotonicSignals) {
    auto ctx = reinterpret_cast<DeviceContextHandle>(0x40);
    HostDeviceMappedRegionConfig cfg{8, 1, 0};
    HostDeviceMappedRegionHandle region = nullptr;
    ASSERT_EQ(host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_heap_region), 0);

    EXPECT_EQ(host_device_mapped_region_wait_common(ctx, region, 0, 0, 0), 0);
    EXPECT_EQ(host_device_mapped_region_wait_common(ctx, region, 0, 1, 0), -EAGAIN);
    EXPECT_EQ(host_device_mapped_region_notify_common(ctx, region, 0, 7), 0);
    EXPECT_EQ(host_device_mapped_region_wait_common(ctx, region, 0, 7, 0), 0);
    EXPECT_EQ(host_device_mapped_region_wait_common(ctx, region, 0, 8, 100), -EAGAIN);
    EXPECT_EQ(host_device_mapped_region_notify_common(ctx, region, 0, 6), -EINVAL);
    EXPECT_EQ(host_device_mapped_region_notify_common(ctx, region, 1, 8), -EINVAL);

    EXPECT_EQ(host_device_mapped_region_close_common(ctx, region), 0);
}

TEST(HostDeviceMappedRegion, RejectsStaleAndCrossContextHandles) {
    auto ctx_a = reinterpret_cast<DeviceContextHandle>(0x50);
    auto ctx_b = reinterpret_cast<DeviceContextHandle>(0x51);
    HostDeviceMappedRegionConfig cfg{8, 1, 0};
    HostDeviceMappedRegionHandle region = nullptr;
    ASSERT_EQ(host_device_mapped_region_open_common(ctx_a, &cfg, &region, allocate_heap_region), 0);

    HostDeviceMappedRegionInfo info{};
    EXPECT_EQ(host_device_mapped_region_info_common(ctx_b, region, &info), -EINVAL);
    EXPECT_EQ(host_device_mapped_region_close_common(ctx_a, region), 0);
    EXPECT_EQ(host_device_mapped_region_close_common(ctx_a, region), -EINVAL);
    EXPECT_EQ(host_device_mapped_region_info_common(ctx_a, region, &info), -EINVAL);
}

TEST(HostDeviceMappedRegion, CloseAllReleasesContextRegions) {
    auto ctx = reinterpret_cast<DeviceContextHandle>(0x60);
    HostDeviceMappedRegionConfig cfg{8, 1, 0};
    HostDeviceMappedRegionHandle region = nullptr;
    ASSERT_EQ(host_device_mapped_region_open_common(ctx, &cfg, &region, allocate_heap_region), 0);
    host_device_mapped_region_close_all_common(ctx);
    HostDeviceMappedRegionInfo info{};
    EXPECT_EQ(host_device_mapped_region_info_common(ctx, region, &info), -EINVAL);
}
