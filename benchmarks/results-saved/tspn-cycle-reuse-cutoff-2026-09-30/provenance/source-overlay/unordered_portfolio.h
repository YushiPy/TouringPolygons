#pragma once
#include "unordered_geometry.h"
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <map>
#include <optional>

namespace tpp::unordered_detail {
struct CycleMemo {
    using Key=std::vector<std::pair<size_t,size_t>>;
    struct Entry {
        Polygon contacts;
        std::vector<int> features;
        double lower_bound=0,upper_bound=std::numeric_limits<double>::infinity();
    };
    std::map<Key,Entry> entries;
};
// Workers use the identical input normalization. Only validated incumbents are
// published; each receiver validates again before using the new upper bound.
class PortfolioControl {
    std::mutex mutex;
    Polygon incumbent;
    std::atomic<double> upper{std::numeric_limits<double>::infinity()};
    std::mutex memo_mutex;
public:
    // Full polygon contents avoid assuming matching local decomposition IDs.
    using CycleKey=std::vector<std::pair<size_t,std::vector<std::pair<double,double>>>>;
private:
    std::map<CycleKey,CycleMemo::Entry> cycle_entries;
public:
    const std::chrono::steady_clock::time_point began = std::chrono::steady_clock::now();
    const size_t max_calls;
    const double max_seconds;
    const bool sharing;
    std::atomic<size_t> calls{0}, publications{0};
    std::atomic<size_t> winner{std::numeric_limits<size_t>::max()};
    double proof_seconds = 0; // Written by the winner; read after joining.

    PortfolioControl(size_t limit, double seconds, bool share)
        : max_calls(limit), max_seconds(seconds), sharing(share) {}
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    }
    bool proved() const { return winner.load(std::memory_order_relaxed)!=std::numeric_limits<size_t>::max(); }
    bool stopped() const { return proved() || elapsed()>=max_seconds; }
    bool reserve_call() {
        if(stopped()) return false;
        auto count=calls.load(std::memory_order_relaxed);
        while(count<max_calls) {
            if(calls.compare_exchange_weak(count,count+1,std::memory_order_relaxed)) return true;
        }
        return false;
    }
    void publish(const Polygon &path,double value) {
        if(!sharing || !(value<upper.load(std::memory_order_relaxed))) return;
        std::lock_guard lock(mutex);
        if(value<upper.load(std::memory_order_relaxed)) {
            incumbent=path;
            upper.store(value,std::memory_order_relaxed);
            publications.fetch_add(1,std::memory_order_relaxed);
        }
    }
    bool receive(double local_upper,Polygon &path) {
        if(!sharing || !(upper.load(std::memory_order_relaxed)<local_upper)) return false;
        std::lock_guard lock(mutex);
        if(!(upper.load(std::memory_order_relaxed)<local_upper)) return false;
        path=incumbent;
        return true;
    }
    std::optional<CycleMemo::Entry> find_cycle(const CycleKey &key) {
        if(!sharing)return {};
        std::lock_guard lock(memo_mutex);
        const auto found=cycle_entries.find(key);
        if(found==cycle_entries.end())return {};
        return found->second; // Copy under lock; verify outside the lock.
    }
    // The publisher must already have independently certified these constraints.
    void store_cycle(CycleKey key,CycleMemo::Entry entry) {
        if(!sharing)return;
        std::lock_guard lock(memo_mutex);
        auto found=cycle_entries.find(key);
        if(found!=cycle_entries.end()) {
            const double lower=std::max(found->second.lower_bound,entry.lower_bound);
            if(entry.upper_bound<found->second.upper_bound)found->second=std::move(entry);
            found->second.lower_bound=lower;
        } else {
            if(cycle_entries.size()>=4096)cycle_entries.clear();
            cycle_entries.emplace(std::move(key),std::move(entry));
        }
    }
    // Called only after the worker's full original-coordinate finalization.
    void finish_proof(size_t worker) {
        size_t expected=std::numeric_limits<size_t>::max();
        if(winner.compare_exchange_strong(expected,worker,std::memory_order_relaxed)) proof_seconds=elapsed();
    }
};
}
