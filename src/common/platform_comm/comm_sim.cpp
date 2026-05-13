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
 * Simulation backend for the comm_* distributed communication API.
 *
 * Uses POSIX shared memory (shm_open + mmap) so that multiple *processes*
 * (one per rank) share the same window region. Synchronization primitives
 * (barrier counters) live in the shared region itself, using GCC __atomic
 * builtins which are lock-free-safe on mmap'd memory.
 *
 * Shared memory layout (page-aligned header + per-rank windows):
 *   [ SharedHeader (4096 bytes) ][ rank-0 window ][ rank-1 window ] ...
 *
 * HCCL backend contract alignment notes:
 *   - comm_init takes (int rank, int nranks, void *stream, const char *rootinfo_path).
 *     The sim backend ignores `stream` (no ACL/device in simulation).
 *   - nranks is bounds-checked against COMM_MAX_RANK_NUM (64) because the
 *     CommContext windowsIn/windowsOut arrays are fixed-size.
 *   - windowsOut[i] is filled (mirrors windowsIn[i] in sim since there is no
 *     separate remote-write channel).  Kernels that consume windowsOut on the
 *     HCCL backend must still compile-and-run on sim.
 *   - ftruncate wait + barrier + destroy all use an explicit timeout
 *     (SIM_COMM_TIMEOUT_SECONDS) so a dead peer cannot hang survivors forever.
 *   - extern "C" entry points allocate std::string so exceptions are wrapped
 *     in function-try-blocks to avoid escaping the C ABI.
 */

#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr size_t HEADER_SIZE = 4096;
constexpr int SIM_COMM_TIMEOUT_SECONDS = 120;
constexpr int FTRUNCATE_POLL_INTERVAL_US = 1000;
constexpr int BARRIER_POLL_INTERVAL_US = 50;
constexpr int DESTROY_POLL_INTERVAL_US = 1000;

// macOS's PSHMNAMLEN is 31 (name length excluding the null terminator).  Linux
// accepts up to NAME_MAX (255), but we pick the tighter value so the same
// backend runs on both.  The name layout below is fully constant-width so we
// can static_assert on it at compile time.
constexpr size_t SHM_NAME_MAX_LEN = 31;
constexpr size_t SHM_NAME_PREFIX_LEN = 9;  // "/simpler_"
constexpr size_t SHM_NAME_HEX_FIELD = 8;   // %08x: exactly 8 hex chars
constexpr size_t SHM_NAME_LEN = SHM_NAME_PREFIX_LEN + SHM_NAME_HEX_FIELD + 1 /*underscore*/ + SHM_NAME_HEX_FIELD;
static_assert(SHM_NAME_LEN <= SHM_NAME_MAX_LEN, "shm name exceeds macOS PSHMNAMLEN");

struct SharedHeader {
    volatile int nranks;
    volatile int alloc_done;
    volatile int ready_count;
    volatile int barrier_count;
    volatile int barrier_phase;
    volatile int destroy_count;
    size_t per_rank_win_size;
};

// Build a session-scoped shm name component from a caller-controlled id.
uint32_t hash_id(const char *id) {
    size_t h = std::hash<std::string>{}(id ? id : "default");
    return static_cast<uint32_t>(h ^ (h >> 32));
}

std::string make_shm_name(uint32_t process_tree_id, uint32_t session_id) {
    char buf[SHM_NAME_LEN + 1];
    int written = std::snprintf(buf, sizeof(buf), "/simpler_%08x_%08x", process_tree_id, session_id);
    // Defensive runtime check: snprintf returns -1 only on I/O / encoding
    // errors, and the static_assert above already pins the upper bound of a
    // successful write, so this is really an "impossible path" guard for the
    // libc-misbehaving edge case.
    if (written < 0 || static_cast<size_t>(written) != SHM_NAME_LEN) {
        std::fprintf(stderr, "[comm_sim] snprintf produced unexpected length %d for shm name\n", written);
        return {};
    }
    return {buf};
}

// Build a session-scoped shm name.
//
// Hashing only the rootinfo_path produces a stable name across test re-runs
// with the same path, which means a crashed prior run that left its segment
// behind in /dev/shm would collide: the new rank 0 hits EEXIST, attaches to
// the dead segment, and reads a stale alloc_done=1 / ready_count that may
// desynchronize the barrier.
//
// Mixing in getppid() disambiguates by launching process tree: every fork of
// the same parent (the canonical sim launch pattern — one driver process
// spawns N ranks) agrees on the name, while a subsequent re-launch gets a new
// parent PID and therefore a fresh name.  Cross-node / cross-parent launches
// on sim are out of scope; callers relying on those topologies must use the
// HCCL backend.
//
// Name layout is fixed-width `"/simpler_%08x_%08x"` = 26 bytes (plus NUL), well
// under macOS's PSHMNAMLEN=31.  The width is constant-propagated into
// SHM_NAME_LEN above so a future format-string change gets caught by the
// static_assert at compile time rather than by an EFILENAMEMAXEXCEEDED at
// runtime on macOS.  PID is truncated to its low 32 bits (pid_t is int32_t on
// every target we support) and the 64-bit rootinfo-path hash is xor-folded to
// 32 bits; both are still collision-resistant for the canonical
// "one driver spawns N ranks" launch pattern.
std::string make_shm_name(const char *rootinfo_path) {
    return make_shm_name(static_cast<uint32_t>(getppid()), hash_id(rootinfo_path));
}

std::string make_subcomm_shm_name(const std::string &base_shm_name, uint64_t sub_comm_id) {
    std::string id = base_shm_name + ":" + std::to_string(sub_comm_id);
    return make_shm_name(static_cast<uint32_t>(getppid()), hash_id(id.c_str()));
}

// Poll `check` until it returns true or the timeout elapses.  Uses steady_clock
// so wall-clock NTP adjustments cannot desynchronize the wait.
bool wait_until(const std::function<bool()> &check, int timeout_seconds, int poll_interval_us) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (!check()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        usleep(poll_interval_us);
    }
    return true;
}

}  // namespace

struct CommHandle_ {
    int rank;
    int nranks;
    std::string shm_name;

    void *mmap_base = nullptr;
    size_t mmap_size = 0;
    bool is_creator = false;

    CommContext host_ctx{};
    std::vector<CommContext *> derived_contexts;
};

extern "C" CommHandle comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) try {
    (void)stream;  // sim has no ACL / stream concept

    if (rootinfo_path == nullptr) {
        std::fprintf(stderr, "[comm_sim rank %d] comm_init: rootinfo_path is null\n", rank);
        return nullptr;
    }
    if (rank < 0 || nranks <= 0 || rank >= nranks) {
        std::fprintf(stderr, "[comm_sim] comm_init: invalid rank=%d nranks=%d\n", rank, nranks);
        return nullptr;
    }
    if (static_cast<uint32_t>(nranks) > COMM_MAX_RANK_NUM) {
        std::fprintf(
            stderr, "[comm_sim rank %d] comm_init: nranks=%d exceeds COMM_MAX_RANK_NUM=%u\n", rank, nranks,
            COMM_MAX_RANK_NUM
        );
        return nullptr;
    }

    auto *h = new (std::nothrow) CommHandle_{};
    if (h == nullptr) {
        std::fprintf(stderr, "[comm_sim rank %d] comm_init: allocation failed\n", rank);
        return nullptr;
    }

    h->rank = rank;
    h->nranks = nranks;
    h->shm_name = make_shm_name(rootinfo_path);
    return h;
} catch (const std::exception &e) {
    std::fprintf(stderr, "[comm_sim rank %d] comm_init: exception: %s\n", rank, e.what());
    return nullptr;
} catch (...) {
    std::fprintf(stderr, "[comm_sim rank %d] comm_init: unknown exception\n", rank);
    return nullptr;
}

extern "C" CommHandle comm_create_subcomm(
    CommHandle base, uint64_t sub_comm_id, const uint32_t *rank_ids, size_t rank_count, uint32_t sub_comm_rank_id,
    void *stream
) try {
    (void)stream;  // sim has no ACL / stream concept

    if (base == nullptr) {
        std::fprintf(stderr, "[comm_sim] comm_create_subcomm: base is null\n");
        return nullptr;
    }
    if (rank_ids == nullptr) {
        std::fprintf(stderr, "[comm_sim rank %d] comm_create_subcomm: rank_ids is null\n", base->rank);
        return nullptr;
    }
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM) {
        std::fprintf(
            stderr, "[comm_sim rank %d] comm_create_subcomm: invalid rank_count=%zu (max=%u)\n", base->rank, rank_count,
            COMM_MAX_RANK_NUM
        );
        return nullptr;
    }
    if (sub_comm_rank_id >= rank_count) {
        std::fprintf(
            stderr, "[comm_sim rank %d] comm_create_subcomm: invalid sub_comm_rank_id=%u rank_count=%zu\n", base->rank,
            sub_comm_rank_id, rank_count
        );
        return nullptr;
    }

    for (size_t i = 0; i < rank_count; ++i) {
        if (rank_ids[i] >= static_cast<uint32_t>(base->nranks)) {
            std::fprintf(
                stderr, "[comm_sim rank %d] comm_create_subcomm: rank_ids[%zu]=%u out of range [0, %d)\n", base->rank,
                i, rank_ids[i], base->nranks
            );
            return nullptr;
        }
        for (size_t j = 0; j < i; ++j) {
            if (rank_ids[j] == rank_ids[i]) {
                std::fprintf(
                    stderr, "[comm_sim rank %d] comm_create_subcomm: duplicate rank id %u\n", base->rank, rank_ids[i]
                );
                return nullptr;
            }
        }
    }
    if (rank_ids[sub_comm_rank_id] != base->rank) {
        std::fprintf(
            stderr, "[comm_sim rank %d] comm_create_subcomm: rank_ids[%u]=%u does not match base rank\n", base->rank,
            sub_comm_rank_id, rank_ids[sub_comm_rank_id]
        );
        return nullptr;
    }

    auto *h = new (std::nothrow) CommHandle_{};
    if (h == nullptr) {
        std::fprintf(stderr, "[comm_sim rank %d] comm_create_subcomm: allocation failed\n", base->rank);
        return nullptr;
    }

    h->rank = static_cast<int>(sub_comm_rank_id);
    h->nranks = static_cast<int>(rank_count);
    h->shm_name = make_subcomm_shm_name(base->shm_name, sub_comm_id);
    return h;
} catch (const std::exception &e) {
    std::fprintf(stderr, "[comm_sim] comm_create_subcomm: exception: %s\n", e.what());
    return nullptr;
} catch (...) {
    std::fprintf(stderr, "[comm_sim] comm_create_subcomm: unknown exception\n");
    return nullptr;
}

extern "C" int comm_alloc_windows(CommHandle h, size_t win_size, uint64_t *device_ctx_out) try {
    if (h == nullptr || device_ctx_out == nullptr) return -1;

    size_t total = HEADER_SIZE + win_size * static_cast<size_t>(h->nranks);

    int fd = shm_open(h->shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd >= 0) {
        h->is_creator = true;
        if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
            std::fprintf(stderr, "[comm_sim rank %d] ftruncate failed: %s\n", h->rank, std::strerror(errno));
            close(fd);
            shm_unlink(h->shm_name.c_str());
            return -1;
        }
    } else if (errno == EEXIST) {
        fd = shm_open(h->shm_name.c_str(), O_RDWR, 0600);
        if (fd < 0) {
            std::fprintf(stderr, "[comm_sim rank %d] shm_open: %s\n", h->rank, std::strerror(errno));
            return -1;
        }
        // Wait for creator to finish ftruncate by checking file size.
        bool sized = wait_until(
            [fd, total]() {
                struct stat st;
                return fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= total;
            },
            SIM_COMM_TIMEOUT_SECONDS, FTRUNCATE_POLL_INTERVAL_US
        );
        if (!sized) {
            std::fprintf(
                stderr, "[comm_sim rank %d] ftruncate wait timed out after %ds\n", h->rank, SIM_COMM_TIMEOUT_SECONDS
            );
            close(fd);
            return -1;
        }
    } else {
        std::fprintf(stderr, "[comm_sim rank %d] shm_open O_EXCL: %s\n", h->rank, std::strerror(errno));
        return -1;
    }

    void *base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[comm_sim rank %d] mmap: %s\n", h->rank, std::strerror(errno));
        return -1;
    }

    h->mmap_base = base;
    h->mmap_size = total;

    auto *hdr = static_cast<SharedHeader *>(base);

    if (h->is_creator) {
        hdr->per_rank_win_size = win_size;
        hdr->ready_count = 0;
        hdr->barrier_count = 0;
        hdr->barrier_phase = 0;
        hdr->destroy_count = 0;
        __atomic_store_n(&hdr->nranks, h->nranks, __ATOMIC_RELEASE);
        __atomic_store_n(&hdr->alloc_done, 1, __ATOMIC_RELEASE);
    } else {
        bool ready = wait_until(
            [hdr]() {
                return __atomic_load_n(&hdr->alloc_done, __ATOMIC_ACQUIRE) != 0;
            },
            SIM_COMM_TIMEOUT_SECONDS, FTRUNCATE_POLL_INTERVAL_US
        );
        if (!ready) {
            std::fprintf(
                stderr, "[comm_sim rank %d] alloc_done wait timed out after %ds\n", h->rank, SIM_COMM_TIMEOUT_SECONDS
            );
            return -1;
        }
    }

    auto *win_base = static_cast<uint8_t *>(base) + HEADER_SIZE;

    // Cross-process addressing contract (differs from HCCL's GVA model!):
    //
    // Each rank's CommContext.windowsIn/windowsOut[i] holds *this process's*
    // pointer to rank i's slice of the shared mmap region.  Because the
    // underlying fd is MAP_SHARED, each rank's own writes to its own slice
    // are visible to other ranks that read through *their own* windowsIn[i]
    // entry — but the numerical addresses do NOT match across processes
    // (ASLR + independent mmap placement).  This is fine as long as kernels
    // only dereference their own rank's CommContext; the hardware UT's
    // cross-rank address-agreement assert is specifically an HCCL-GVA
    // invariant and is not expected to hold (nor intended to run) under sim.
    auto &ctx = h->host_ctx;
    ctx.workSpace = 0;
    ctx.workSpaceSize = 0;
    ctx.rankId = static_cast<uint32_t>(h->rank);
    ctx.rankNum = static_cast<uint32_t>(h->nranks);
    ctx.winSize = win_size;
    for (int i = 0; i < h->nranks; ++i) {
        uint64_t addr = reinterpret_cast<uint64_t>(win_base + static_cast<size_t>(i) * win_size);
        ctx.windowsIn[i] = addr;
        // Sim has no separate remote-write channel; mirror windowsIn so kernels
        // that read windowsOut still see a valid per-rank address.
        ctx.windowsOut[i] = addr;
    }

    *device_ctx_out = reinterpret_cast<uint64_t>(&h->host_ctx);

    __atomic_add_fetch(&hdr->ready_count, 1, __ATOMIC_ACQ_REL);
    bool all_ready = wait_until(
        [hdr, h]() {
            return __atomic_load_n(&hdr->ready_count, __ATOMIC_ACQUIRE) >= h->nranks;
        },
        SIM_COMM_TIMEOUT_SECONDS, BARRIER_POLL_INTERVAL_US
    );
    if (!all_ready) {
        std::fprintf(
            stderr, "[comm_sim rank %d] ready_count barrier timed out after %ds\n", h->rank, SIM_COMM_TIMEOUT_SECONDS
        );
        return -1;
    }

    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "[comm_sim] comm_alloc_windows: exception: %s\n", e.what());
    return -1;
} catch (...) {
    std::fprintf(stderr, "[comm_sim] comm_alloc_windows: unknown exception\n");
    return -1;
}

extern "C" int comm_get_local_window_base(CommHandle h, uint64_t *base_out) {
    if (h == nullptr || base_out == nullptr) return -1;
    *base_out = h->host_ctx.windowsIn[h->rank];
    return 0;
}

extern "C" int comm_get_window_size(CommHandle h, size_t *size_out) {
    if (h == nullptr || size_out == nullptr) return -1;
    *size_out = static_cast<size_t>(h->host_ctx.winSize);
    return 0;
}

extern "C" int comm_derive_context(
    CommHandle h, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) try {
    if (h == nullptr || rank_ids == nullptr || device_ctx_out == nullptr) return -1;
    if (h->mmap_base == nullptr) {
        std::fprintf(stderr, "[comm_sim rank %d] comm_derive_context: base windows are not allocated\n", h->rank);
        return -1;
    }
    if (rank_count == 0 || rank_count > COMM_MAX_RANK_NUM || domain_rank >= rank_count) {
        std::fprintf(
            stderr, "[comm_sim rank %d] comm_derive_context: invalid rank_count=%zu domain_rank=%u\n", h->rank,
            rank_count, domain_rank
        );
        return -1;
    }
    if (window_offset + window_size > static_cast<size_t>(h->host_ctx.winSize)) {
        std::fprintf(
            stderr,
            "[comm_sim rank %d] comm_derive_context: window range [%zu, %zu) exceeds base window size %llu\n",
            h->rank, window_offset, window_offset + window_size, static_cast<unsigned long long>(h->host_ctx.winSize)
        );
        return -1;
    }

    auto *ctx = new (std::nothrow) CommContext{};
    if (ctx == nullptr) return -1;
    ctx->workSpace = h->host_ctx.workSpace;
    ctx->workSpaceSize = h->host_ctx.workSpaceSize;
    ctx->rankId = domain_rank;
    ctx->rankNum = static_cast<uint32_t>(rank_count);
    ctx->winSize = window_size;
    for (size_t i = 0; i < rank_count; ++i) {
        uint32_t base_rank = rank_ids[i];
        if (base_rank >= static_cast<uint32_t>(h->nranks)) {
            std::fprintf(
                stderr, "[comm_sim rank %d] comm_derive_context: rank_ids[%zu]=%u out of range [0, %d)\n", h->rank,
                i, base_rank, h->nranks
            );
            delete ctx;
            return -1;
        }
        ctx->windowsIn[i] = h->host_ctx.windowsIn[base_rank] + window_offset;
        ctx->windowsOut[i] = h->host_ctx.windowsOut[base_rank] + window_offset;
    }

    h->derived_contexts.push_back(ctx);
    *device_ctx_out = reinterpret_cast<uint64_t>(ctx);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "[comm_sim] comm_derive_context: exception: %s\n", e.what());
    return -1;
} catch (...) {
    std::fprintf(stderr, "[comm_sim] comm_derive_context: unknown exception\n");
    return -1;
}

extern "C" int comm_barrier(CommHandle h) {
    if (h == nullptr || h->mmap_base == nullptr) return -1;

    // Sense-reversing barrier.  Each caller snapshots `phase` before
    // incrementing `barrier_count`, then waits until `phase` advances.  This
    // ordering — snapshot → increment → wait-for-change — is what makes a
    // back-to-back re-entry race-free: a fast rank that returns from this
    // barrier and immediately re-enters for the NEXT one will read the
    // already-advanced phase as its snapshot, so its own count increment
    // is accounted for the new generation instead of the old.
    //
    // The last rank's (count=0 then phase+1) release-ordered pair ensures
    // that any rank exiting the wait on the phase change also sees the
    // reset count before it can contribute to the next barrier, so
    // concurrent re-entry cannot corrupt the pending generation.
    auto *hdr = static_cast<SharedHeader *>(h->mmap_base);
    int phase = __atomic_load_n(&hdr->barrier_phase, __ATOMIC_ACQUIRE);
    int arrived = __atomic_add_fetch(&hdr->barrier_count, 1, __ATOMIC_ACQ_REL);

    if (arrived == h->nranks) {
        __atomic_store_n(&hdr->barrier_count, 0, __ATOMIC_RELEASE);
        __atomic_add_fetch(&hdr->barrier_phase, 1, __ATOMIC_ACQ_REL);
        return 0;
    }

    bool advanced = wait_until(
        [hdr, phase]() {
            return __atomic_load_n(&hdr->barrier_phase, __ATOMIC_ACQUIRE) != phase;
        },
        SIM_COMM_TIMEOUT_SECONDS, BARRIER_POLL_INTERVAL_US
    );
    if (!advanced) {
        std::fprintf(
            stderr, "[comm_sim rank %d] barrier timed out after %ds (phase=%d arrived=%d nranks=%d)\n", h->rank,
            SIM_COMM_TIMEOUT_SECONDS, phase, arrived, h->nranks
        );
        return -1;
    }
    return 0;
}

extern "C" int comm_destroy(CommHandle h) try {
    if (h == nullptr) return -1;

    int rc = 0;
    for (auto *ctx : h->derived_contexts) {
        delete ctx;
    }
    h->derived_contexts.clear();
    if (h->mmap_base != nullptr) {
        auto *hdr = static_cast<SharedHeader *>(h->mmap_base);
        int gone = __atomic_add_fetch(&hdr->destroy_count, 1, __ATOMIC_ACQ_REL);

        // Last rank out unlinks the shm segment.  Earlier ranks wait a bounded
        // time so that, on the common "all ranks destroy in lockstep" path,
        // the unlink actually happens before the next test re-creates it.
        // On a dead-peer path, the timeout elapses, we still munmap and exit,
        // and the segment lingers until /dev/shm is cleared.
        if (gone >= h->nranks) {
            munmap(h->mmap_base, h->mmap_size);
            h->mmap_base = nullptr;
            shm_unlink(h->shm_name.c_str());
        } else {
            bool drained = wait_until(
                [hdr, h]() {
                    return __atomic_load_n(&hdr->destroy_count, __ATOMIC_ACQUIRE) >= h->nranks;
                },
                SIM_COMM_TIMEOUT_SECONDS, DESTROY_POLL_INTERVAL_US
            );
            munmap(h->mmap_base, h->mmap_size);
            h->mmap_base = nullptr;
            if (!drained) {
                std::fprintf(
                    stderr, "[comm_sim rank %d] destroy barrier timed out after %ds; local teardown complete\n",
                    h->rank, SIM_COMM_TIMEOUT_SECONDS
                );
                rc = -1;
            }
        }
    }

    delete h;
    return rc;
} catch (const std::exception &e) {
    std::fprintf(stderr, "[comm_sim] comm_destroy: exception: %s\n", e.what());
    return -1;
} catch (...) {
    std::fprintf(stderr, "[comm_sim] comm_destroy: unknown exception\n");
    return -1;
}
