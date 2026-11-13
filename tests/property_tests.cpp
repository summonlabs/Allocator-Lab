// Allocator Lab 1.0.0 randomized/property tests.
#include "tests/test_framework.hpp"
#include "tests/test_helpers.hpp"
#include "allocator_lab/detail/rng.hpp"
#include "allocator_lab/experiment.hpp"
#include "allocator_lab/error.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace allocator_lab;
using namespace al_test_helpers;

static bool alignment_ok(void* p, std::size_t al) {
    if (al == 0) return true;
    return (reinterpret_cast<std::uintptr_t>(p) % al) == 0;
}

TEST(property_randomized_alloc_free_invariants) {
    detail::SplitMix64 rng(0xABCDEF123456789ULL);
    const std::size_t kOps = 20000;
    for (const std::string& name : { "system", "aligned", "fixed_pool", "size_class_pool", "slab", "free_list", "buddy", "segregated_fit" }) {
        auto a = make_allocator(name);
        std::unordered_map<AllocationId, std::size_t> live;   // id -> size
        std::vector<AllocationId> live_ids;
        for (std::size_t op = 0; op < kOps; ++op) {
            std::uint64_t r = rng.next(100);
            if (r < 55 || live.empty()) {
                std::size_t sz = 1 + static_cast<std::size_t>(rng.next(8192));
                std::size_t al = 0;
                if (rng.next(100) < 20) al = 16;
                AllocationRequest req; req.size = sz; req.alignment = al;
                AllocationResult ar = a->allocate(req);
                if (ar.error.ok()) {
                    REQUIRE(ar.id != 0);
                    if (al) REQUIRE(alignment_ok(ar.address, al));
                    live[ar.id] = ar.size; live_ids.push_back(ar.id);
                }
            } else if (r < 80) {
                if (!live_ids.empty()) {
                    std::size_t idx = static_cast<std::size_t>(rng.next(live_ids.size()));
                    AllocationId id = live_ids[idx];
                    REQUIRE(a->free(id).ok());
                    // remove
                    for (std::size_t j = idx + 1; j < live_ids.size(); ++j) live_ids[j-1] = live_ids[j];
                    live_ids.pop_back();
                    live.erase(id);
                }
            } else {
                if (!live_ids.empty()) {
                    std::size_t idx = static_cast<std::size_t>(rng.next(live_ids.size()));
                    AllocationId id = live_ids[idx];
                    std::size_t oldsz = live[id];
                    std::size_t nsz = 1 + static_cast<std::size_t>(rng.next(8192));
                    AllocationRequest req; req.size = nsz;
                    AllocationResult rr = a->reallocate(id, req);
                    if (rr.error.ok()) { live[id] = rr.size; }
                    else { /* original preserved */ AllocationInfo info; REQUIRE(a->query(id, info).ok()); REQUIRE_EQ(info.size, oldsz); }
                }
            }
        }
        // Free all
        for (auto id : live_ids) a->free(id);
        AllocatorStatistics st = a->statistics();
        REQUIRE(st.live.live_count == 0);
        REQUIRE(st.live.live_bytes == 0);
        a->shutdown();
    }
}

TEST(property_realloc_preserves_data_for_system) {
    SystemAllocator a;
    // Write pattern, realloc grow, verify prefix preserved for system realloc.
    AllocationRequest req; req.size = 32;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    auto* p = static_cast<std::uint8_t*>(r.address);
    for (std::size_t i = 0; i < 32; ++i) p[i] = static_cast<std::uint8_t>(i);
    AllocationRequest req2; req2.size = 64;
    AllocationResult r2 = a.reallocate(r.id, req2);
    REQUIRE(r2.error.ok());
    auto* q = static_cast<std::uint8_t*>(r2.address);
    bool ok = true;
    for (std::size_t i = 0; i < 32; ++i) if (q[i] != static_cast<std::uint8_t>(i)) ok = false;
    REQUIRE(ok);
    a.free(r.id);
}

TEST(property_deterministic_workload_replay) {
    WorkloadConfig wc; wc.operations = 5000; wc.live_target = 256; wc.seed = 42; wc.size_dist = SizeDistribution::LogUniform;
    Trace t1, t2;
    REQUIRE(generate_workload(wc, t1).ok());
    wc.seed = 42;
    REQUIRE(generate_workload(wc, t2).ok());
    REQUIRE_EQ(t1.entry_count(), t2.entry_count());
    REQUIRE_EQ(t1.seed, t2.seed);
    for (std::size_t i = 0; i < t1.entry_count(); ++i) {
        REQUIRE(t1.entries[i].op == t2.entries[i].op);
        REQUIRE(t1.entries[i].id == t2.entries[i].id);
        REQUIRE(t1.entries[i].size == t2.entries[i].size);
        REQUIRE(t1.entries[i].alignment == t2.entries[i].alignment);
    }
    // different seed => different
    WorkloadConfig wc3 = wc; wc3.seed = 43;
    Trace t3; generate_workload(wc3, t3);
    bool differ = t3.entry_count() != t1.entry_count();
    if (!differ) for (std::size_t i = 0; i < t1.entry_count(); ++i) if (t3.entries[i].op != t1.entries[i].op) { differ = true; break; }
    REQUIRE(differ);
}

TEST(property_same_trace_replays_on_multiple_allocators) {
    WorkloadConfig wc; wc.operations = 3000; wc.live_target = 200; wc.seed = 7;
    wc.size_dist = SizeDistribution::Fixed; wc.min_size = 64; wc.max_size = 64;
    Trace trace; REQUIRE(generate_workload(wc, trace).ok());
    for (const std::string& name : all_freeable_names()) {
        auto a = make_allocator(name);
        // run through the workload via the experiment runner
        ExperimentConfig ec; ec.verify_payload = true;
        ExperimentResult res = run_replay(*a, trace, ec, BuildProvenance{});
        // The trace is a valid sequence: the replay must not crash or leak. Some
        // strategies do not support reallocation, so failures are expected there;
        // what matters is that every successfully-created live object is released.
        REQUIRE(res.success_count > 0);
        AllocatorStatistics st = a->statistics();
        REQUIRE(st.live.live_count == 0);
        REQUIRE(st.live.live_bytes == 0);
        a->shutdown();
    }
}

namespace {
// Hands out the same buffer for every allocation. Whatever is written through
// one allocation is seen through all the others.
class SingleBufferAllocator : public AllocatorStrategy {
public:
    SingleBufferAllocator() : buffer_(256, 0x5A) { caps_.supports_free = true; caps_.domains = { MemoryDomain::Host }; }
    AllocatorId id() const noexcept override { return 77; }
    const std::string& name() const noexcept override { return name_; }
    AllocatorKind kind() const noexcept override { return AllocatorKind::Custom; }
    const AllocatorCapabilities& capabilities() const noexcept override { return caps_; }
    AllocationResult allocate(const AllocationRequest& req) override {
        AllocationResult r;
        if (req.size > buffer_.size()) { r.error = make_error(ErrorCode::size_too_large, "too large"); return r; }
        r.id = ++next_; r.address = buffer_.data(); r.size = req.size; r.alignment = 1;
        return r;
    }
    Error free(AllocationHandle) override { return Error{}; }
    Error query(AllocationHandle, AllocationInfo&) override { return make_error(ErrorCode::not_supported, "no query"); }
    AllocatorStatistics statistics() const override { return AllocatorStatistics{}; }
    void shutdown() noexcept override {}
    bool untouched() const { for (std::uint8_t b : buffer_) if (b != 0x5A) return false; return true; }
private:
    std::vector<std::uint8_t> buffer_;
    AllocationId next_ = 0;
    std::string name_ = "single_buffer";
    AllocatorCapabilities caps_;
};

Trace two_live_then_freed() {
    Trace t; t.version = kTraceVersion; t.name = "overlap";
    auto add = [&](TraceOperationType op, AllocationId id) {
        TraceEntry e; e.seq = t.entries.size(); e.step = e.seq; e.op = op; e.id = id; e.size = 64;
        t.entries.push_back(e);
    };
    add(TraceOperationType::Allocate, 1);
    add(TraceOperationType::Allocate, 2);
    add(TraceOperationType::Free, 1);
    add(TraceOperationType::Free, 2);
    return t;
}
}  // namespace

TEST(experiment_leaves_payload_alone_unless_asked) {
    SingleBufferAllocator a;
    ExperimentConfig ec; ec.verify_payload = false;
    ExperimentResult res = run_replay(a, two_live_then_freed(), ec, BuildProvenance{});
    REQUIRE_EQ(res.success_count, 4u);
    REQUIRE(a.untouched());                     // no pattern was written into the allocations
}

TEST(experiment_reports_payload_corruption) {
    SingleBufferAllocator a;                    // the second allocation overwrites the first
    ExperimentConfig ec; ec.verify_payload = true;
    ExperimentResult res = run_replay(a, two_live_then_freed(), ec, BuildProvenance{});
    REQUIRE(!a.untouched());
    REQUIRE(res.failure_count > 0);             // the damaged payload is a failure of the run
}

TEST(experiment_warmup_is_not_timed) {
    WorkloadConfig wc; wc.operations = 400; wc.live_target = 32; wc.seed = 11;
    wc.size_dist = SizeDistribution::Fixed; wc.min_size = 64; wc.max_size = 64;
    Trace trace; REQUIRE(generate_workload(wc, trace).ok());
    SystemAllocator a;
    ExperimentConfig ec; ec.warmup_ops = trace.entry_count();   // everything is warm-up
    ExperimentResult res = run_replay(a, trace, ec, BuildProvenance{});
    REQUIRE(res.success_count > 0);             // the operations ran
    REQUIRE_EQ(res.alloc_latency.n, 0u);        // none of them was measured
    REQUIRE_EQ(res.free_latency.n, 0u);
    REQUIRE_EQ(res.elapsed_ns, 0u);
    a.shutdown();
}

#ifdef ALLOCATOR_LAB_TEST_HAS_CUDA
TEST(experiment_does_not_write_into_device_memory) {
    CudaDirectAllocator a;
    {
        AllocationRequest probe; probe.size = 64; probe.domain = MemoryDomain::Device;
        AllocationResult r = a.allocate(probe);
        if (!r.error.ok()) { std::cout << "[SKIP] no CUDA device: " << r.error.message << "\n"; return; }
        REQUIRE(a.free(r.id).ok());
    }
    WorkloadConfig wc; wc.operations = 64; wc.live_target = 8; wc.seed = 5; wc.realloc_fraction = 0.0;
    wc.size_dist = SizeDistribution::Fixed; wc.min_size = 4096; wc.max_size = 4096; wc.domain = MemoryDomain::Device;
    Trace trace; REQUIRE(generate_workload(wc, trace).ok());
    ExperimentConfig ec; ec.verify_payload = true;              // asked for, but device memory is not host memory
    ExperimentResult res = run_replay(a, trace, ec, BuildProvenance{});
    REQUIRE(res.success_count > 0);
    REQUIRE_EQ(a.statistics().live.live_count, 0u);
    a.shutdown();
}
#endif
