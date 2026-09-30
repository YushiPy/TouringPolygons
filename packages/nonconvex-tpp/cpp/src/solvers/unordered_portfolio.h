#pragma once
#include "unordered_geometry.h"
#include "unordered_bounds.h"
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
    // Enabled lazily by compatible-bound queries. Interning compares COMPLETE
    // coordinates, so IDs cannot alias by hash collision or local piece number.
    bool bound_index_enabled=false;
    std::map<std::vector<std::pair<double,double>>,size_t> region_ids;
    std::map<CycleMemo::Key,double> cycle_bounds;
    CycleMemo::Key intern_key(const CycleKey &key) {
        CycleMemo::Key compact;compact.reserve(key.size());
        for(const auto &[label,polygon]:key) {
            auto found=region_ids.find(polygon);
            if(found==region_ids.end())found=region_ids.emplace(polygon,region_ids.size()).first;
            compact.emplace_back(label,found->second);
        }
        return compact;
    }
    static CycleMemo::Key canonical_compact(const CycleMemo::Key &key) {
        CycleMemo::Key canonical;canonical.reserve(key.size());
        for(size_t i:canonical_cycle_indices(key))canonical.push_back(key[i]);
        return canonical;
    }
    void store_bound(const CycleKey &key,double lower) {
        auto &bound=cycle_bounds[canonical_compact(intern_key(key))];
        bound=std::max(bound,lower);
    }
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
    static CycleKey canonical_key(const CycleKey &key) {
        std::vector<std::pair<size_t,size_t>> labels;
        for(const auto &v:key)labels.emplace_back(v.first,0);
        CycleKey canonical;
        for(size_t i:canonical_cycle_indices(labels))canonical.push_back(key[i]);
        return canonical;
    }
    // A tour through the target constraints shortcuts to each one-deletion
    // cyclic subsequence. Remaining polygons must match by full coordinates.
    // No bound on a stronger/differently ordered subproblem is imported.
    double compatible_cycle_bound(const CycleKey &key,double cutoff,size_t &queries,size_t &hits) {
        if(!sharing||key.size()<2)return 0;
        double bound=0;
        std::lock_guard lock(memo_mutex);
        if(!bound_index_enabled) {
            bound_index_enabled=true;
            for(const auto &[stored,entry]:cycle_entries)store_bound(stored,entry.lower_bound);
        }
        const auto compact=intern_key(key);
        auto lookup=[&](const CycleMemo::Key &candidate) {
            ++queries;
            if(const auto found=cycle_bounds.find(canonical_compact(candidate));found!=cycle_bounds.end()) {
                ++hits;bound=std::max(bound,found->second);
            }
        };
        lookup(compact);
        if(compact.size()>2)for(size_t i=0;i<compact.size()&&bound<cutoff;++i) {
            auto sub=compact;sub.erase(sub.begin()+i);lookup(sub);
        }
        return bound;
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
            if(bound_index_enabled)store_bound(key,lower);
        } else {
            if(cycle_entries.size()>=4096) {cycle_entries.clear();cycle_bounds.clear();region_ids.clear();}
            if(bound_index_enabled)store_bound(key,entry.lower_bound);
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
