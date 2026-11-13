// Allocator Lab 1.0.0
// Experiment runner implementation.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Summon Software Labs

#include "allocator_lab/experiment.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace allocator_lab {
namespace {

using clock_t = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

static std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<ns>(clock_t::now().time_since_epoch()).count());
}

static void write_pattern(void* p, std::size_t size, std::uint64_t id) {
    std::uint8_t* b = static_cast<std::uint8_t*>(p);
    std::uint64_t s = id * 0x9E3779B97F4A7C15ULL + 0x12345678;
    for (std::size_t i = 0; i < size; ++i) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; b[i] = static_cast<std::uint8_t>(s >> 56); }
}

static bool verify_pattern(void* p, std::size_t size, std::uint64_t id) {
    std::uint8_t* b = static_cast<std::uint8_t*>(p);
    std::uint64_t s = id * 0x9E3779B97F4A7C15ULL + 0x12345678;
    for (std::size_t i = 0; i < size; ++i) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; if (b[i] != static_cast<std::uint8_t>(s >> 56)) return false; }
    return true;
}

struct Pending {
    AllocationId alloc_id = 0;
    std::uint64_t trace_id = 0;
    void* addr = nullptr;
    std::size_t size = 0;
};

struct StreamState {
    std::unordered_map<std::uint64_t, Pending> live;  // trace_id -> pending
    LatencyCollector alloc_lat, free_lat, realloc_lat;
    std::uint64_t ops = 0, successes = 0, failures = 0;
    bool invariant_failed = false;
};

// How one entry is applied.
//   touch:  write a deterministic pattern into each allocation and check it before
//           the block is freed or reallocated (config.verify_payload). Only done for
//           memory the host can address.
//   record: count the operation in the latency and elapsed-time figures (false
//           while warming up).
// Latency covers the allocator call alone, never the pattern work around it.
struct ApplyMode { bool touch = false; bool record = true; };

void apply_entry(AllocatorStrategy& a, const TraceEntry& e, StreamState& st, std::uint64_t& total_ns,
                 const ApplyMode& mode) {
    ++st.ops;
    AllocationResult res;
    std::uint64_t spent = 0;
    if (e.op == TraceOperationType::Allocate) {
        AllocationRequest req{}; req.size = static_cast<std::size_t>(e.size); req.alignment = static_cast<std::size_t>(e.alignment); req.domain = e.domain; req.tag = e.tag;
        const std::uint64_t t0 = now_ns();
        res = a.allocate(req);
        spent = now_ns() - t0;
        if (res.error.ok()) {
            st.live[e.id] = Pending{ res.id, e.id, res.address, res.size };
            ++st.successes;
            if (mode.touch) write_pattern(res.address, res.size, e.id);
        } else { ++st.failures; }
        if (mode.record) st.alloc_lat.add(spent);
    } else if (e.op == TraceOperationType::Free) {
        auto it = st.live.find(e.id);
        if (it != st.live.end()) {
            if (mode.touch && !verify_pattern(it->second.addr, it->second.size, e.id)) {
                // The payload changed while the allocation was live: a failure of the run.
                st.invariant_failed = true; ++st.failures;
            }
            const std::uint64_t t0 = now_ns();
            Error err = a.free(it->second.alloc_id);
            spent = now_ns() - t0;
            if (err.ok()) ++st.successes; else ++st.failures;
            if (mode.record) st.free_lat.add(spent);
            st.live.erase(it);
        } else { ++st.failures; if (mode.record) st.free_lat.add(0); }
    } else if (e.op == TraceOperationType::Reallocate) {
        auto it = st.live.find(e.id);
        if (it != st.live.end()) {
            if (mode.touch && !verify_pattern(it->second.addr, it->second.size, e.id)) {
                st.invariant_failed = true; ++st.failures;
            }
            AllocationRequest req{}; req.size = static_cast<std::size_t>(e.size); req.alignment = static_cast<std::size_t>(e.alignment); req.domain = e.domain; req.tag = e.tag;
            const std::uint64_t t0 = now_ns();
            res = a.reallocate(it->second.alloc_id, req);
            spent = now_ns() - t0;
            if (res.error.ok()) {
                it->second.addr = res.address; it->second.size = res.size;
                ++st.successes;
                if (mode.touch) write_pattern(res.address, res.size, e.id);
            } else { ++st.failures; }
            if (mode.record) st.realloc_lat.add(spent);
        } else { ++st.failures; if (mode.record) st.realloc_lat.add(0); }
    } else {
        // trim / reset / barrier / phase marker
        const std::uint64_t t0 = now_ns();
        if (e.op == TraceOperationType::Trim) { Error err = a.trim(); if (!err.ok()) ++st.failures; }
        else if (e.op == TraceOperationType::Reset) { Error err = a.reset(); if (!err.ok()) ++st.failures; }
        spent = now_ns() - t0;
    }
    if (mode.record) total_ns += spent;
}

// Patterns are written from the host, so only into memory the host can address.
bool host_addressable(const AllocatorStrategy& a) {
    for (MemoryDomain d : a.capabilities().domains) if (d == MemoryDomain::Device) return false;
    return true;
}

} // namespace

ExperimentResult run_replay(AllocatorStrategy& allocator, const Trace& trace,
                            const ExperimentConfig& config, BuildProvenance provenance) {
    ExperimentResult result{};
    result.experiment_id = 1;
    result.allocator_id = allocator.id();
    result.allocator_name = allocator.name();
    result.kind = allocator.kind();
    result.capabilities = allocator.capabilities();
    result.seed = trace.seed;
    result.operations = trace.entry_count();
    result.provenance = std::move(provenance);

    StreamState st;
    st.alloc_lat = LatencyCollector(config.max_raw_samples);
    st.free_lat = LatencyCollector(config.max_raw_samples);
    st.realloc_lat = LatencyCollector(config.max_raw_samples);
    SnapshotId snap_id = 0;
    std::uint64_t elapsed = 0;
    std::uint64_t warm = std::min(config.warmup_ops, trace.entry_count());
    ApplyMode mode;
    mode.touch = config.verify_payload && host_addressable(allocator);
    for (std::uint64_t i = 0; i < trace.entry_count(); ++i) {
        mode.record = i >= warm;   // warm-up operations run, but are not timed
        apply_entry(allocator, trace.entries[i], st, elapsed, mode);
        if (config.snapshot_interval && i >= warm && (i % config.snapshot_interval == 0)) {
            ExperimentSnapshot snap{};
            snap.snapshot_id = ++snap_id; snap.experiment_id = result.experiment_id;
            snap.allocator_id = allocator.id(); snap.allocator_name = allocator.name();
            snap.step = i; snap.seed = trace.seed;
            AllocatorStatistics s = allocator.statistics();
            snap.live = s.live; snap.fragmentation = s.fragmentation; snap.accounting = s.accounting;
            snap.reuse = s.reuse; snap.success_count = s.success_count; snap.failure_count = s.failure_count;
            snap.capabilities = allocator.capabilities(); snap.provenance = result.provenance;
            snap.trace_position = std::to_string(i);
            result.snapshots.push_back(snap);
        }
    }
    if (config.sync_after_run && allocator.capabilities().supports_trim) allocator.trim();
    AllocatorStatistics final_stats = allocator.statistics();
    result.live = final_stats.live;
    result.accounting = final_stats.accounting;
    result.fragmentation = final_stats.fragmentation;
    result.reuse = final_stats.reuse;
    result.success_count = st.successes; result.failure_count = st.failures;
    result.alloc_latency = st.alloc_lat.summary();
    result.free_latency = st.free_lat.summary();
    result.realloc_latency = st.realloc_lat.summary();
    result.elapsed_ns = elapsed;
    WorkerResult wr; wr.worker = 0; wr.ops = st.ops; wr.successes = st.successes; wr.failures = st.failures;
    wr.alloc_latency = st.alloc_lat.summary(); wr.free_latency = st.free_lat.summary(); wr.realloc_latency = st.realloc_lat.summary();
    wr.elapsed_ns = static_cast<double>(elapsed);
    result.workers.push_back(wr);
    result.throughput.elapsed_seconds = static_cast<double>(elapsed) / 1e9;
    result.throughput.allocations = st.alloc_lat.count();
    result.throughput.frees = st.free_lat.count();
    result.throughput.reallocations = st.realloc_lat.count();
    result.throughput.operations = st.ops;
    result.throughput.bytes_allocated = final_stats.accounting.total_granted_bytes;
    result.throughput.bytes_freed = final_stats.accounting.total_granted_bytes - final_stats.live.live_bytes;
    return result;
}

ExperimentResult run_concurrent(AllocatorStrategy& allocator, const Trace& trace,
                                const ExperimentConfig& config, std::uint32_t threads,
                                BuildProvenance provenance) {
    ExperimentResult result{};
    result.experiment_id = 1;
    result.allocator_id = allocator.id();
    result.allocator_name = allocator.name();
    result.kind = allocator.kind();
    result.capabilities = allocator.capabilities();
    result.seed = trace.seed;
    result.operations = trace.entry_count();
    result.provenance = std::move(provenance);
    if (threads == 0) threads = 1;

    std::vector<std::vector<const TraceEntry*>> per_worker(threads);
    for (const auto& e : trace.entries) {
        WorkerId w = e.worker % threads;
        per_worker[w].push_back(&e);
    }

    std::vector<StreamState> states(threads);
    for (auto& st : states) { st.alloc_lat = LatencyCollector(config.max_raw_samples); st.free_lat = LatencyCollector(config.max_raw_samples); st.realloc_lat = LatencyCollector(config.max_raw_samples); }
    std::vector<std::uint64_t> elapsed(threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    ApplyMode mode;
    mode.touch = config.verify_payload && host_addressable(allocator);
    for (std::uint32_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            std::uint64_t el = 0;
            for (const TraceEntry* e : per_worker[t]) apply_entry(allocator, *e, states[t], el, mode);
            elapsed[t] = el;
        });
    }
    for (auto& w : workers) w.join();

    AllocatorStatistics final_stats = allocator.statistics();
    result.live = final_stats.live;
    result.accounting = final_stats.accounting;
    result.fragmentation = final_stats.fragmentation;
    result.reuse = final_stats.reuse;
    result.success_count = 0; result.failure_count = 0;
    LatencyCollector agg_alloc(config.max_raw_samples), agg_free(config.max_raw_samples), agg_realloc(config.max_raw_samples);
    for (std::uint32_t t = 0; t < threads; ++t) {
        result.success_count += states[t].successes; result.failure_count += states[t].failures;
        agg_alloc.merge(states[t].alloc_lat); agg_free.merge(states[t].free_lat); agg_realloc.merge(states[t].realloc_lat);
        WorkerResult wr; wr.worker = t; wr.ops = states[t].ops; wr.successes = states[t].successes; wr.failures = states[t].failures;
        wr.alloc_latency = states[t].alloc_lat.summary();
        wr.free_latency = states[t].free_lat.summary();
        wr.realloc_latency = states[t].realloc_lat.summary();
        wr.elapsed_ns = static_cast<double>(elapsed[t]);
        result.workers.push_back(wr);
    }
    result.alloc_latency = agg_alloc.summary();
    result.free_latency = agg_free.summary();
    result.realloc_latency = agg_realloc.summary();
    std::uint64_t max_elapsed = 0; for (auto e : elapsed) max_elapsed = std::max(max_elapsed, e);
    result.elapsed_ns = max_elapsed;
    result.throughput.elapsed_seconds = static_cast<double>(max_elapsed) / 1e9;
    result.throughput.allocations = agg_alloc.count();
    result.throughput.frees = agg_free.count();
    result.throughput.reallocations = agg_realloc.count();
    result.throughput.operations = result.success_count + result.failure_count;
    result.throughput.bytes_allocated = final_stats.accounting.total_granted_bytes;
    result.throughput.bytes_freed = final_stats.accounting.total_granted_bytes - final_stats.live.live_bytes;
    return result;
}

} // namespace allocator_lab
