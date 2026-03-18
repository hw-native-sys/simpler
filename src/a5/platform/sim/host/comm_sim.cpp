/**
 * Simulation backend for the comm_* distributed communication API.
 *
 * Uses POSIX shared memory (shm_open + mmap) so that multiple *processes*
 * (one per rank, spawned by DistributedCodeRunner) share the same RDMA
 * window region.  Synchronization primitives (barrier counters) live in
 * the shared region itself, using GCC __atomic builtins which are safe
 * on lock-free-capable types in mmap'd memory.
 *
 * Shared memory layout (page-aligned header + per-rank windows):
 *   [ SharedHeader (4096 bytes) ][ rank-0 window ][ rank-1 window ] ...
 */

#include "host/comm.h"
#include "common/comm_context.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr size_t HEADER_SIZE = 4096;

namespace {

struct SharedHeader {
    volatile int nranks;
    volatile int alloc_done;
    volatile int ready_count;
    volatile int barrier_count;
    volatile int barrier_phase;
    volatile int destroy_count;
    size_t per_rank_win_size;
};

std::string make_shm_name(const char* rootinfo_path) {
    size_t h = std::hash<std::string>{}(rootinfo_path ? rootinfo_path : "default");
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/simpler_comm_%zx", h);
    return buf;
}

}  // anonymous namespace

// ============================================================================
// Per-handle state (process-local)
// ============================================================================

struct CommHandle_ {
    int rank;
    int nranks;
    std::string shm_name;

    void* mmap_base = nullptr;
    size_t mmap_size = 0;
    bool is_creator = false;

    CommDeviceContext host_ctx{};
};

// ============================================================================
// API implementation
// ============================================================================

extern "C" CommHandle comm_init(int rank, int nranks, const char* rootinfo_path) {
    auto* h = new (std::nothrow) CommHandle_{};
    if (!h) return nullptr;

    h->rank = rank;
    h->nranks = nranks;
    h->shm_name = make_shm_name(rootinfo_path);
    return h;
}

extern "C" int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t* device_ctx_out) {
    if (!h || !device_ctx_out) return -1;

    size_t total = HEADER_SIZE + win_size * static_cast<size_t>(h->nranks);

    int fd = shm_open(h->shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd >= 0) {
        h->is_creator = true;
        if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
            std::perror("comm_sim: ftruncate");
            close(fd);
            shm_unlink(h->shm_name.c_str());
            return -1;
        }
    } else if (errno == EEXIST) {
        fd = shm_open(h->shm_name.c_str(), O_RDWR, 0600);
        if (fd < 0) { std::perror("comm_sim: shm_open"); return -1; }

        // Wait for creator to finish ftruncate by checking file size
        for (int i = 0; i < 5000; ++i) {
            struct stat st;
            if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= total) break;
            usleep(1000);
        }
    } else {
        std::perror("comm_sim: shm_open O_EXCL");
        return -1;
    }

    void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { std::perror("comm_sim: mmap"); return -1; }

    h->mmap_base = base;
    h->mmap_size = total;

    auto* hdr = static_cast<SharedHeader*>(base);

    if (h->is_creator) {
        hdr->per_rank_win_size = win_size;
        hdr->ready_count = 0;
        hdr->barrier_count = 0;
        hdr->barrier_phase = 0;
        hdr->destroy_count = 0;
        __atomic_store_n(&hdr->nranks, h->nranks, __ATOMIC_RELEASE);
        __atomic_store_n(&hdr->alloc_done, 1, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&hdr->alloc_done, __ATOMIC_ACQUIRE) == 0) {
            usleep(100);
        }
    }

    auto* win_base = static_cast<uint8_t*>(base) + HEADER_SIZE;

    auto& ctx = h->host_ctx;
    ctx.workSpace = 0;
    ctx.workSpaceSize = 0;
    ctx.rankId = static_cast<uint32_t>(h->rank);
    ctx.rankNum = static_cast<uint32_t>(h->nranks);
    ctx.winSize = win_size;
    for (int i = 0; i < h->nranks; ++i) {
        ctx.windowsIn[i] = reinterpret_cast<uint64_t>(
            win_base + static_cast<size_t>(i) * win_size);
    }

    *device_ctx_out = reinterpret_cast<uint64_t>(&h->host_ctx);

    __atomic_add_fetch(&hdr->ready_count, 1, __ATOMIC_ACQ_REL);
    while (__atomic_load_n(&hdr->ready_count, __ATOMIC_ACQUIRE) < h->nranks) {
        usleep(100);
    }

    return 0;
}

extern "C" int comm_get_local_window_base(CommHandle h, uint64_t* base_out) {
    if (!h || !base_out) return -1;
    *base_out = h->host_ctx.windowsIn[h->rank];
    return 0;
}

extern "C" int comm_barrier(CommHandle h) {
    if (!h || !h->mmap_base) return -1;

    auto* hdr = static_cast<SharedHeader*>(h->mmap_base);
    int phase = __atomic_load_n(&hdr->barrier_phase, __ATOMIC_ACQUIRE);
    int arrived = __atomic_add_fetch(&hdr->barrier_count, 1, __ATOMIC_ACQ_REL);

    if (arrived == h->nranks) {
        __atomic_store_n(&hdr->barrier_count, 0, __ATOMIC_RELEASE);
        __atomic_add_fetch(&hdr->barrier_phase, 1, __ATOMIC_ACQ_REL);
    } else {
        while (__atomic_load_n(&hdr->barrier_phase, __ATOMIC_ACQUIRE) == phase) {
            usleep(50);
        }
    }

    return 0;
}

extern "C" int comm_destroy(CommHandle h) {
    if (!h) return -1;

    if (h->mmap_base) {
        auto* hdr = static_cast<SharedHeader*>(h->mmap_base);
        int gone = __atomic_add_fetch(&hdr->destroy_count, 1, __ATOMIC_ACQ_REL);

        munmap(h->mmap_base, h->mmap_size);
        h->mmap_base = nullptr;

        if (gone >= h->nranks) {
            shm_unlink(h->shm_name.c_str());
        }
    }

    delete h;
    return 0;
}
