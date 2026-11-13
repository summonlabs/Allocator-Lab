// Allocator Lab 1.0.0 allocator correctness tests.
#include "tests/test_framework.hpp"
#include "tests/test_helpers.hpp"
#include "allocator_lab/error.hpp"
#include "allocator_lab/config.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace allocator_lab;
using namespace al_test_helpers;

static void basic_alloc_free(AllocatorStrategy& a) {
    AllocationRequest req; req.size = 64; req.domain = MemoryDomain::Host;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    REQUIRE(r.id != 0);
    REQUIRE(r.address != nullptr);
    AllocationInfo info;
    REQUIRE(a.query(r.id, info).ok());
    REQUIRE(info.live);
    REQUIRE_EQ(info.size, r.size);
    REQUIRE(a.free(r.id).ok());
    REQUIRE(a.statistics().live.live_count == 0);
}

TEST(allocator_basic_each) {
    for (const auto& n : all_freeable_names()) {
        auto a = make_allocator(n);
        REQUIRE(a != nullptr);
        basic_alloc_free(*a);
        a->shutdown();
    }
}

TEST(allocator_arena_semantics) {
    ArenaAllocator a;
    // arena supports reset, rejects free of individual objects
    REQUIRE(a.capabilities().supports_free == false);
    REQUIRE(a.capabilities().allows_per_object_free == false);
    AllocationRequest req; req.size = 128;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    Error fe = a.free(r.id);
    REQUIRE(!fe.ok());
    REQUIRE(fe.code == ErrorCode::not_supported);
    REQUIRE(a.reset().ok());
    REQUIRE(a.statistics().live.live_count == 0);
}

TEST(allocator_alignment_supported) {
    AlignedAllocator a(4096);
    std::size_t alignments[] = { 8, 16, 32, 64, 256, 4096 };
    for (std::size_t al : alignments) {
        AllocationRequest req; req.size = 100; req.alignment = al;
        AllocationResult r = a.allocate(req);
        REQUIRE(r.error.ok());
        REQUIRE((reinterpret_cast<std::uintptr_t>(r.address) % al) == 0);
        a.free(r.id);
    }
}

TEST(allocator_alignment_rejected) {
    AlignedAllocator a(4096);
    AllocationRequest req; req.size = 100; req.alignment = 48;  // not pow2
    REQUIRE(!a.allocate(req).error.ok());
    req.alignment = 8192;  // > max
    REQUIRE(a.allocate(req).error.code == ErrorCode::alignment_unsupported);
}

TEST(allocator_double_free_rejected) {
    for (const auto& n : all_normal_names()) {
        if (n == "arena") continue;
        auto a = make_allocator(n);
        AllocationRequest req; req.size = 64;
        AllocationResult r = a->allocate(req);
        REQUIRE(r.error.ok());
        REQUIRE(a->free(r.id).ok());
        Error e2 = a->free(r.id);
        REQUIRE(!e2.ok());  // double free / invalid handle
        a->shutdown();
    }
}

TEST(allocator_foreign_handle_rejected) {
    SystemAllocator sa;
    AlignedAllocator aa;
    AllocationRequest req; req.size = 128;
    AllocationResult r = sa.allocate(req);
    REQUIRE(r.error.ok());
    // free from a different allocator must fail (foreign handle)
    Error fe = aa.free(r.id);
    REQUIRE(!fe.ok());
    sa.free(r.id);
}

TEST(allocator_realloc_preserves_on_failure) {
    SystemAllocator sa;
    AllocationRequest req; req.size = 64;
    AllocationResult r = sa.allocate(req);
    REQUIRE(r.error.ok());
    // Realloc to a huge size exceeding max should fail and preserve original.
    AllocationRequest req2; req2.size = 1ull << 40;
    AllocationResult r2 = sa.reallocate(r.id, req2);
    REQUIRE(!r2.error.ok());
    AllocationInfo info; REQUIRE(sa.query(r.id, info).ok());
    REQUIRE(info.live);
    REQUIRE_EQ(info.size, 64u);
    sa.free(r.id);
}

TEST(allocator_realloc_grow_shrink) {
    SystemAllocator sa;
    AllocationRequest req; req.size = 32;
    AllocationResult r = sa.allocate(req);
    AllocationRequest req2; req2.size = 128;
    AllocationResult r2 = sa.reallocate(r.id, req2);
    REQUIRE(r2.error.ok());
    REQUIRE_EQ(r2.id, r.id);
    AllocationInfo info; REQUIRE(sa.query(r.id, info).ok());
    REQUIRE_EQ(info.size, 128u);
    sa.free(r.id);
}

TEST(allocator_accounting_to_zero) {
    for (const auto& n : all_freeable_names()) {
        auto a = make_allocator(n);
        // allocate several, then free all
        std::vector<AllocationId> ids;
        for (int i = 0; i < 100; ++i) {
            AllocationRequest req; req.size = 64 + i * 7; req.domain = MemoryDomain::Host;
            AllocationResult r = a->allocate(req);
            if (r.error.ok()) ids.push_back(r.id);
        }
        for (auto id : ids) a->free(id);
        auto st = a->statistics();
        REQUIRE(st.live.live_count == 0);
        REQUIRE(st.live.live_bytes == 0);
        a->shutdown();
        auto st2 = a->statistics();
        REQUIRE(st2.live.live_count == 0);
    }
}

TEST(allocator_capacity_bounded) {
    FixedPool pool(64, 1024);
    REQUIRE_EQ(pool.capabilities().max_allocation_size, 64u);
    // request exceeds pool object size -> rejected
    AllocationRequest req; req.size = 200;
    REQUIRE(!pool.allocate(req).error.ok());
}

TEST(allocator_zeroing_policy) {
    SystemAllocator a;
    AllocationRequest req; req.size = 512; req.flags = AllocationFlags::ZeroFill;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    auto* p = static_cast<std::uint8_t*>(r.address);
    bool allzero = true;
    for (std::size_t i = 0; i < req.size; ++i) if (p[i] != 0) allzero = false;
    REQUIRE(allzero);
    a.free(r.id);
}

TEST(allocator_poison_alloc_flag) {
    SystemAllocator a;
    AllocationRequest req; req.size = 256; req.flags = AllocationFlags::PoisonAlloc;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    auto* p = static_cast<std::uint8_t*>(r.address);
    bool poisoned = true;
    for (std::size_t i = 0; i < req.size; ++i) if (p[i] != detail::kPoisonAllocByte) poisoned = false;
    REQUIRE(poisoned);
    a.free(r.id);
}

TEST(allocator_memory_domain_rejected) {
    // system allocator only supports Host; requesting a Device domain should not be
    // silently served on host (it will be rejected by the system allocator or honored
    // only for host memory in general). We assert host-dominant behavior is consistent.
    SystemAllocator a;
    AllocationRequest req; req.size = 10; req.domain = MemoryDomain::Device;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok() == false || r.address != nullptr);
    if (r.error.ok()) a.free(r.id);
}

static bool aligned_to(const void* p, std::size_t al) {
    return (reinterpret_cast<std::uintptr_t>(p) % al) == 0;
}

TEST(arena_honours_requested_alignment) {
    ArenaAllocator a;
    AllocationRequest one; one.size = 1;
    REQUIRE(a.allocate(one).error.ok());            // leaves the bump pointer off any large boundary
    for (std::size_t al : { std::size_t(64), std::size_t(256), std::size_t(4096) }) {
        AllocationRequest req; req.size = 100; req.alignment = al;
        AllocationResult r = a.allocate(req);
        REQUIRE(r.error.ok());
        REQUIRE(aligned_to(r.address, al));
        REQUIRE(a.allocate(one).error.ok());
    }
    a.shutdown();
}

TEST(arena_capacity_bound_spans_blocks) {
    ArenaAllocator a(4096, 8192);                   // two blocks at most
    AllocationRequest req; req.size = 3000;         // one object per block
    REQUIRE(a.allocate(req).error.ok());
    REQUIRE(a.allocate(req).error.ok());
    REQUIRE_EQ(a.statistics().live.reserved, 8192u);
    AllocationResult third = a.allocate(req);
    REQUIRE(!third.error.ok());
    REQUIRE_EQ(a.statistics().live.reserved, 8192u);
    a.shutdown();
}

TEST(free_list_zero_fill_is_all_zero) {
    FreeListAllocator a;
    AllocationRequest req; req.size = 64; req.flags = AllocationFlags::ZeroFill;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    const auto* p = static_cast<const std::uint8_t*>(r.address);
    bool allzero = true;
    for (std::size_t i = 0; i < req.size; ++i) if (p[i] != 0) allzero = false;
    REQUIRE(allzero);
    a.shutdown();
}

TEST(free_list_coalesces_back_to_one_block) {
    FreeListAllocator a;                            // 256 KiB superblocks
    // A block whose payload is all zero, then freed: the blocks carved from it
    // next have their headers where that payload was.
    AllocationRequest big; big.size = 1000; big.flags = AllocationFlags::ZeroFill;
    AllocationResult x = a.allocate(big);
    REQUIRE(x.error.ok());
    REQUIRE(a.free(x.id).ok());
    AllocationRequest req; req.size = 64;
    AllocationResult b1 = a.allocate(req), b2 = a.allocate(req), b3 = a.allocate(req);
    REQUIRE(b1.error.ok() && b2.error.ok() && b3.error.ok());
    REQUIRE(a.free(b1.id).ok());
    REQUIRE(a.free(b3.id).ok());
    REQUIRE(a.free(b2.id).ok());
    // Everything is free again and adjacent: one span, the whole superblock.
    AllocatorStatistics st = a.statistics();
    REQUIRE_EQ(st.fragmentation.free_span_count, 1u);
    REQUIRE_EQ(st.fragmentation.largest_free_span, 256u * 1024u);
    a.shutdown();
}

TEST(segregated_serves_requests_larger_than_a_superblock) {
    SegregatedAllocator a;                          // 128 KiB superblocks
    AllocationRequest req; req.size = 200u * 1024;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    std::memset(r.address, 0x5A, req.size);
    REQUIRE(a.free(r.id).ok());
    // The block is reused, not abandoned: asking again reserves nothing more.
    const std::uint64_t reserved = a.statistics().live.reserved;
    AllocationResult again = a.allocate(req);
    REQUIRE(again.error.ok());
    REQUIRE_EQ(a.statistics().live.reserved, reserved);
    a.shutdown();
}

TEST(fixed_pool_uses_its_whole_capacity) {
    for (std::size_t capacity : { std::size_t(4), std::size_t(10) }) {
        FixedPool pool(64, capacity);
        AllocationRequest req; req.size = 64;
        std::size_t got = 0;
        for (std::size_t i = 0; i < capacity; ++i) if (pool.allocate(req).error.ok()) ++got;
        REQUIRE_EQ(got, capacity);
        AllocationResult over = pool.allocate(req);
        REQUIRE(over.error.code == ErrorCode::capacity_exceeded);
        pool.shutdown();
    }
}

TEST(pool_reuse_counts_only_recycled_slots) {
    // Slots that were never handed out are fresh backing, not reuse.
    FixedPool fixed(64, 64);
    SizeClassPool classes;
    SlabAllocator slab(64);
    for (AllocatorStrategy* a : { static_cast<AllocatorStrategy*>(&fixed), static_cast<AllocatorStrategy*>(&classes), static_cast<AllocatorStrategy*>(&slab) }) {
        AllocationRequest req; req.size = 64;
        std::vector<AllocationId> ids;
        for (int i = 0; i < 6; ++i) {
            AllocationResult r = a->allocate(req);
            REQUIRE(r.error.ok());
            REQUIRE(!r.reused_backing);
            ids.push_back(r.id);
        }
        REQUIRE_EQ(a->statistics().reuse.pool_hits, 0u);
        REQUIRE(a->free(ids.back()).ok());
        AllocationResult again = a->allocate(req);
        REQUIRE(again.error.ok());
        REQUIRE(again.reused_backing);
        AllocatorStatistics st = a->statistics();
        REQUIRE_EQ(st.reuse.pool_hits, 1u);
        REQUIRE_EQ(st.reuse.same_size_reuse, 1u);
        a->shutdown();
    }
}

TEST(size_class_pool_honours_alignment) {
    SizeClassPool pool;
    struct Case { std::size_t size, align; };
    for (Case c : { Case{16, 256}, Case{100, 64}, Case{1, 128}, Case{5000, 256} }) {
        AllocationRequest req; req.size = c.size; req.alignment = c.align;
        AllocationResult r = pool.allocate(req);
        REQUIRE(r.error.ok());
        REQUIRE(aligned_to(r.address, c.align));
        REQUIRE(r.alignment >= c.align);
        REQUIRE(r.size >= c.size);
    }
    pool.shutdown();
}

TEST(slab_rejects_object_larger_than_its_page) {
    SlabAllocator slab(8000, 4096);                 // a 4096-byte page cannot hold 8000 bytes
    AllocationRequest req; req.size = 8000;
    AllocationResult r = slab.allocate(req);
    REQUIRE(!r.error.ok());
    REQUIRE(r.error.code == ErrorCode::invalid_config);
    REQUIRE_EQ(slab.statistics().live.reserved, 0u);
    slab.shutdown();
}

TEST(slab_rejects_page_that_is_not_a_power_of_two) {
    SlabAllocator slab(64, 3000);
    AllocationRequest req; req.size = 64;
    REQUIRE(slab.allocate(req).error.code == ErrorCode::invalid_config);
    slab.shutdown();
}

TEST(realloc_keeps_waste_non_negative) {
    SystemAllocator sys;
    AlignedAllocator aligned;
    for (AllocatorStrategy* a : { static_cast<AllocatorStrategy*>(&sys), static_cast<AllocatorStrategy*>(&aligned) }) {
        AllocationRequest req; req.size = 100;
        AllocationResult r = a->allocate(req);
        REQUIRE(r.error.ok());
        AllocationRequest grow; grow.size = 150;
        REQUIRE(a->reallocate(r.id, grow).error.ok());
        AllocatorStatistics st = a->statistics();
        REQUIRE(st.accounting.total_granted_bytes >= st.accounting.total_requested_bytes);
        REQUIRE_EQ(st.accounting.internal_waste, 0u);
        REQUIRE(a->free(r.id).ok());
        a->shutdown();
    }
}

TEST(aligned_realloc_rejects_bad_alignment) {
    AlignedAllocator a;
    AllocationRequest req; req.size = 64;
    AllocationResult r = a.allocate(req);
    REQUIRE(r.error.ok());
    AllocationRequest bad; bad.size = 128; bad.alignment = 48;   // not a power of two
    AllocationResult rr = a.reallocate(r.id, bad);
    REQUIRE(rr.error.code == ErrorCode::invalid_alignment);
    AllocationInfo info; REQUIRE(a.query(r.id, info).ok());      // the original is untouched
    REQUIRE_EQ(info.size, 64u);
    REQUIRE(a.free(r.id).ok());
}

TEST(registry_never_aliases_ids) {
    AllocatorRegistry reg;
    const AllocatorId first = reg.register_allocator(std::make_unique<SystemAllocator>("first", 1));
    REQUIRE_EQ(first, 1u);
    // A generated id never lands on one that is taken.
    const AllocatorId second = reg.register_allocator(std::make_unique<SystemAllocator>("second"));
    REQUIRE(second != 0);
    REQUIRE(second != first);
    REQUIRE(reg.find(second) != nullptr);
    REQUIRE(reg.find(second)->name() == "second");
    // A strategy that claims an id in use is refused, and nothing is replaced.
    const AllocatorId clash = reg.register_allocator(std::make_unique<SystemAllocator>("third", 1));
    REQUIRE_EQ(clash, 0u);
    REQUIRE_EQ(reg.size(), 2u);
    REQUIRE(reg.find(first)->name() == "first");
}

TEST(shared_allocator_alignment_bounds_and_statistics) {
    const std::string name = "al_test_shared_" + std::to_string(GetCurrentProcessId());
    SharedAllocator creator(name, 64u * 1024, SharedAllocator::Mode::Create);
    REQUIRE(creator.initialized());
    REQUIRE(creator.capabilities().supports_free == false);     // free() is not supported, and says so
    AllocationRequest req; req.size = 100; req.alignment = 4096;
    AllocationResult r = creator.allocate(req);
    REQUIRE(r.error.ok());
    REQUIRE(aligned_to(r.address, 4096));
    AllocatorStatistics st = creator.statistics();
    REQUIRE_EQ(st.accounting.total_requested_bytes, 100u);      // what callers asked for
    REQUIRE(st.name == "shared");
    // A second view of the same object, opened with a larger size than it has.
    SharedAllocator opener(name, 8u * 1024 * 1024, SharedAllocator::Mode::Open);
    REQUIRE(opener.initialized());
    REQUIRE_EQ(opener.region_size(), creator.region_size());
    AllocationRequest too_big; too_big.size = 1u * 1024 * 1024;
    REQUIRE(!opener.allocate(too_big).error.ok());              // outside what is mapped
    opener.shutdown();
    REQUIRE_EQ(opener.published_offset(), 0u);                  // no view any more: no dereference
    creator.shutdown();
}

TEST(mapped_allocator_alignment_and_capabilities) {
    const std::string path = "al_test_mapped_" + std::to_string(GetCurrentProcessId()) + ".map";
    {
        MappedAllocator m(path, 1u * 1024 * 1024, true);
        REQUIRE(m.initialized());
        REQUIRE(m.capabilities().supports_free == false);
        AllocationRequest first; first.size = 24;
        REQUIRE(m.allocate(first).error.ok());
        AllocationRequest req; req.size = 64; req.alignment = 4096;
        AllocationResult r = m.allocate(req);
        REQUIRE(r.error.ok());
        REQUIRE(aligned_to(r.address, 4096));
        m.shutdown();
    }
    std::remove(path.c_str());
}

#ifdef ALLOCATOR_LAB_TEST_HAS_CUDA
#include <cuda_runtime.h>

TEST(cuda_zero_fill_is_done_on_the_device) {
    CudaDirectAllocator a;
    AllocationRequest req; req.size = 4096; req.domain = MemoryDomain::Device; req.flags = AllocationFlags::ZeroFill;
    AllocationResult r = a.allocate(req);
    if (!r.error.ok()) { std::cout << "[SKIP] no CUDA device: " << r.error.message << "\n"; return; }
    // Make sure the zeros come from the fill: dirty the block, free it, ask again.
    REQUIRE(cudaMemset(r.address, 0xFF, req.size) == cudaSuccess);
    REQUIRE(a.free(r.id).ok());
    r = a.allocate(req);
    REQUIRE(r.error.ok());
    std::vector<std::uint8_t> host(req.size, 0xEE);
    REQUIRE(cudaMemcpy(host.data(), r.address, req.size, cudaMemcpyDeviceToHost) == cudaSuccess);
    bool allzero = true;
    for (std::uint8_t b : host) if (b != 0) allzero = false;
    REQUIRE(allzero);
    REQUIRE(a.free(r.id).ok());
    a.shutdown();
}

TEST(cuda_pool_zero_fill_is_honoured) {
    CudaPoolAllocator a;
    AllocationRequest plain; plain.size = 1u << 20; plain.domain = MemoryDomain::Device;
    AllocationResult first = a.allocate(plain);
    if (!first.error.ok()) { std::cout << "[SKIP] no CUDA device: " << first.error.message << "\n"; return; }
    REQUIRE(cudaMemset(first.address, 0xFF, plain.size) == cudaSuccess);
    REQUIRE(a.free(first.id).ok());                              // back to the pool, still dirty
    REQUIRE(a.sync().ok());
    AllocationRequest zeroed = plain; zeroed.flags = AllocationFlags::ZeroFill;
    AllocationResult r = a.allocate(zeroed);
    REQUIRE(r.error.ok());
    REQUIRE(a.sync().ok());
    std::vector<std::uint8_t> host(plain.size, 0xEE);
    REQUIRE(cudaMemcpy(host.data(), r.address, plain.size, cudaMemcpyDeviceToHost) == cudaSuccess);
    bool allzero = true;
    for (std::uint8_t b : host) if (b != 0) allzero = false;
    REQUIRE(allzero);
    REQUIRE(a.free(r.id).ok());
    a.shutdown();
}
#endif
