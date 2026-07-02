// Differential correctness test for the fully_distributed_within_core
// DistTensorMap after its conversion to RING-PER-BUCKET (docs §12).
//
// It re-implements two maps side by side on the SAME reduced key space the real
// engine uses (buffer base address + byte range [lo,hi) + producer task id):
//   * Ref  — the former linked-list semantics: lookup = MAX producer among
//            entries with producer >= alive_floor, same base, byte-range overlap.
//   * Ring — the new ring-per-bucket (private mode), with the SAME bucket count,
//            shift, cap, stride and mask as dist_engine.cpp.
//
// It drives both with randomized SPMD-ordered op streams (retire-before-insert,
// interleaved fan-in lookups, monotonically increasing producer ids) and asserts
// the ring returns byte-identical lookup results to the reference on every query.
// This is the primary correctness gate for the refactor: the sim's end-to-end
// golden/perf harness is unreliable on some hosts (device_wall reads 0 on a cold
// single launch; a second sim launch in one process SIGBUSes — both pre-existing,
// affecting the original build too), so semantic equivalence is proven here.
//
// Standalone (no gtest / build-system deps):
//   g++ -O2 -std=c++17 -o /tmp/tmr test_dist_tensormap_ring.cpp && /tmp/tmr
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---- Reference: former linked-list DistTensorMap::lookup semantics -----------
struct Ref {
    struct E { uint64_t addr, lo, hi; int32_t producer; };
    std::vector<E> es;
    int32_t alive_floor = 0;
    void reset() { es.clear(); alive_floor = 0; }
    void insert(uint64_t addr, uint64_t lo, uint64_t hi, int32_t p) { es.push_back({addr, lo, hi, p}); }
    void advance_retire(int32_t N, int32_t H) { int32_t f = N - H; if (f > alive_floor) alive_floor = f; }
    int32_t lookup(uint64_t addr, uint64_t lo, uint64_t hi) const {
        int32_t best = -1;
        for (const auto &e : es) {
            if (e.producer < alive_floor) continue;
            if (e.addr == addr && lo < e.hi && e.lo < hi && e.producer > best) best = e.producer;
        }
        return best;
    }
};

// ---- New: ring-per-bucket, mirroring dist_engine.cpp DistTensorMap ------------
static constexpr int32_t kRingBuckets = 128;   // == dist_engine.cpp
static constexpr int32_t kRingBucketShift = 7; // log2(kRingBuckets)
static constexpr int32_t kBucketCapMax = 512;  // == dist_engine.cpp
struct Ring {
    struct Slot { uint64_t addr, lo, hi; int32_t producer; int32_t pad_; };
    static constexpr int32_t kStride = kBucketCapMax;
    std::vector<Slot> slots;  // kRingBuckets * kStride
    std::vector<uint64_t> head, tail;
    int32_t cap, cap_mask, alive_floor = 0;
    bool overflowed = false;
    explicit Ring(int32_t c) : cap(c), cap_mask(c - 1) { reset(); }
    void reset() {
        slots.assign((size_t)kRingBuckets * kStride, Slot{});
        head.assign(kRingBuckets, 0);
        tail.assign(kRingBuckets, 0);
        alive_floor = 0;
        overflowed = false;
    }
    static uint32_t hash(uint64_t a) { a *= 0x9E3779B97F4A7C15ULL; return (uint32_t)(a >> (64 - kRingBucketShift)); }
    void reclaim_bucket(int32_t b) {
        while (head[b] < tail[b] && slots[(size_t)b * kStride + (int32_t)(head[b] & cap_mask)].producer < alive_floor)
            head[b]++;
    }
    void advance_retire(int32_t N, int32_t H) { int32_t f = N - H; if (f > alive_floor) alive_floor = f; }
    void insert(uint64_t addr, uint64_t lo, uint64_t hi, int32_t p) {
        int32_t b = (int32_t)hash(addr);
        reclaim_bucket(b);
        if (tail[b] - head[b] >= (uint64_t)cap) { overflowed = true; return; }  // FATAL in prod
        Slot &s = slots[(size_t)b * kStride + (int32_t)(tail[b] & cap_mask)];
        s.addr = addr; s.lo = lo; s.hi = hi; s.producer = p;
        tail[b]++;
    }
    int32_t lookup(uint64_t addr, uint64_t lo, uint64_t hi) {
        int32_t b = (int32_t)hash(addr);
        for (uint64_t k = tail[b]; k > head[b]; k--) {
            const Slot &s = slots[(size_t)b * kStride + (int32_t)((k - 1) & cap_mask)];
            if (s.producer < alive_floor) continue;
            if (s.addr == addr && lo < s.hi && s.lo < hi) return s.producer;
        }
        return -1;
    }
};

// ---- Shared: SINGLE global ring, mirroring dist_engine.cpp SharedTensorMap -----
// Models the concurrent shared map's SEMANTICS in one thread: appends happen in
// task-id order (the real engine serializes them via g_dist.tm_insert_next), so a
// fast core may have already appended FUTURE tasks (producer >= N) and the global
// reclaim floor (min_progress - H - 1) may lag. lookup(N) must still return the
// SAME producer a private replica would, via the temporal filter (producer < N)
// plus the retire floor (producer >= N - H). This test drives exactly those
// off-window conditions and asserts identity with the private/reference result.
struct SharedRing {
    struct Slot { uint64_t addr, lo, hi; int32_t producer; };
    static constexpr int32_t kStride = kBucketCapMax;
    std::vector<Slot> slots;
    std::vector<uint64_t> head, tail;
    int32_t cap, cap_mask;
    bool overflowed = false;
    explicit SharedRing(int32_t c) : cap(c), cap_mask(c - 1) {
        slots.assign((size_t)kRingBuckets * kStride, Slot{});
        head.assign(kRingBuckets, 0);
        tail.assign(kRingBuckets, 0);
    }
    // Single-appender: reclaim producer <= reclaim_floor at head, then append tail.
    void append(uint64_t addr, uint64_t lo, uint64_t hi, int32_t producer, int32_t reclaim_floor) {
        int32_t b = (int32_t)Ring::hash(addr);
        while (head[b] < tail[b] && slots[(size_t)b * kStride + (int32_t)(head[b] & cap_mask)].producer <= reclaim_floor)
            head[b]++;
        if (tail[b] - head[b] >= (uint64_t)cap) { overflowed = true; return; }
        Slot &s = slots[(size_t)b * kStride + (int32_t)(tail[b] & cap_mask)];
        s.addr = addr; s.lo = lo; s.hi = hi; s.producer = producer;
        tail[b]++;
    }
    // lookup: accept producer in [floor, N) — temporal filter + retire floor.
    int32_t lookup(uint64_t addr, uint64_t lo, uint64_t hi, int32_t N, int32_t floor) {
        int32_t b = (int32_t)Ring::hash(addr);
        for (uint64_t k = tail[b]; k > head[b]; k--) {
            const Slot &s = slots[(size_t)b * kStride + (int32_t)((k - 1) & cap_mask)];
            if (s.producer >= N) continue;      // temporal filter (future producer)
            if (s.producer < floor) continue;   // retire floor (== private alive_floor)
            if (s.addr == addr && lo < s.hi && s.lo < hi) return s.producer;
        }
        return -1;
    }
};

struct Region { uint64_t base, lo, hi; };
struct TaskOps { std::vector<Region> lookups; std::vector<Region> inserts; };

int main() {
    std::mt19937_64 rng(0xC0FFEE);
    const int32_t H = 64;
    const int32_t cap = kBucketCapMax;  // production auto default for H=64
    // Enough distinct buffer bases spread across buckets that no bucket's live
    // window exceeds cap (mirrors a well-provisioned run).
    uint64_t bases[24];
    for (auto &b : bases) b = (rng() & 0xFFFFF) << 12;
    auto rand_region = [&]() {
        Region r;
        r.base = bases[rng() % 24];
        r.lo = (rng() % 64) * 128;
        r.hi = r.lo + (1 + rng() % 8) * 128;
        return r;
    };

    int64_t total_queries = 0, shared_queries = 0;
    for (int trial = 0; trial < 300; trial++) {
        const int32_t total_tasks = 4000 + (int)(rng() % 4000);

        // Precompute the per-task op stream so the SHARED pass can append future
        // tasks (simulating a fast core that has run ahead of the looking-up core).
        std::vector<TaskOps> ops((size_t)total_tasks);
        for (int32_t N = 0; N < total_tasks; N++) {
            int nq = rng() % 4;
            for (int q = 0; q < nq; q++) ops[N].lookups.push_back(rand_region());
            int no = 1 + rng() % 3;
            for (int o = 0; o < no; o++) ops[N].inserts.push_back(rand_region());
        }

        // ---- Pass 1: private reference (linked-list) == private ring ----
        Ref ref; Ring ring(cap);
        std::vector<std::vector<int32_t>> priv_results((size_t)total_tasks);
        int mism = 0;
        for (int32_t N = 0; N < total_tasks; N++) {
            ref.advance_retire(N, H);
            ring.advance_retire(N, H);  // retire BEFORE inserts (as dist_submit_impl)
            for (const auto &q : ops[N].lookups) {
                int32_t a = ref.lookup(q.base, q.lo, q.hi);
                int32_t b = ring.lookup(q.base, q.lo, q.hi);
                total_queries++;
                priv_results[N].push_back(a);  // reference == private producer for this fan-in
                if (a != b) { mism++; if (mism <= 3) printf("  MISMATCH trial=%d N=%d: ref=%d ring=%d\n", trial, N, a, b); }
            }
            for (const auto &o : ops[N].inserts) { ref.insert(o.base, o.lo, o.hi, N); ring.insert(o.base, o.lo, o.hi, N); }
        }
        if (ring.overflowed) { printf("TRIAL %d: RING OVERFLOW (cap %d too small)\n", trial, cap); return 2; }
        if (mism) { printf("TRIAL %d: %d PRIVATE MISMATCHES\n", trial, mism); return 1; }

        // ---- Pass 2: SHARED single global ring == private result ----
        // Model concurrency: a fast core has appended tasks up to (N + ahead), and
        // the global reclaim floor lags to N-H-1 (slowest core == the looking-up
        // core). lookup(N) must reproduce the private producer despite off-window
        // (>=N future and <N-H stale) entries coexisting in the shared ring.
        SharedRing sr(cap);
        int32_t appended_upto = -1;  // highest task id appended so far
        int smis = 0;
        for (int32_t N = 0; N < total_tasks; N++) {
            // reclaim floor = min_progress - H - 1; looking-up core is the slowest.
            const int32_t rfloor = N - H - 1;
            // Fast core run-ahead: append future tasks (in id order) before lookup.
            const int32_t ahead = (int32_t)(rng() % 48);  // <= 47, keeps window < cap
            const int32_t target = std::min(N + ahead, total_tasks - 1);
            while (appended_upto < target) {
                int32_t M = ++appended_upto;
                for (const auto &o : ops[M].inserts) sr.append(o.base, o.lo, o.hi, M, rfloor);
            }
            // Resolve this task's fan-in against the shared ring; compare to private.
            for (size_t j = 0; j < ops[N].lookups.size(); j++) {
                const auto &q = ops[N].lookups[j];
                int32_t got = sr.lookup(q.base, q.lo, q.hi, N, N - H);
                int32_t want = priv_results[N][j];
                shared_queries++;
                if (got != want) { smis++; if (smis <= 3) printf("  SHARED MISMATCH trial=%d N=%d: shared=%d private=%d\n", trial, N, got, want); }
            }
        }
        if (sr.overflowed) { printf("TRIAL %d: SHARED RING OVERFLOW (cap %d too small)\n", trial, cap); return 2; }
        if (smis) { printf("TRIAL %d: %d SHARED MISMATCHES\n", trial, smis); return 1; }
    }
    printf("ALL 300 TRIALS PASSED: linked-list ref == private ring (%lld q); shared ring == private (%lld q)\n",
           (long long)total_queries, (long long)shared_queries);
    return 0;
}
