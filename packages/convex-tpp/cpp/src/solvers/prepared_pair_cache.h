#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace tpp::detail {
// Keys use the immutable prepared polygon identities in increasing order.
// Both representations retain exactly the same bounded set of pair proofs.
class PreparedPairCache {
public:
    using Pair=std::pair<std::uint64_t,std::uint64_t>;
    static constexpr size_t capacity=65536;
    static constexpr size_t dense_ids=256;

    std::optional<bool> find(const Pair &key) const {
#ifdef TPP_HAS_DENSE_PAIR_CACHE
        if(key.second<dense_ids) {
            const auto mask=std::uint64_t{1}<<(key.second%64);
            const auto &row=dense_[key.first];
            if(row.known[key.second/64]&mask)return bool(row.disjoint[key.second/64]&mask);
            return std::nullopt;
        }
#endif
        const auto found=sparse_.find(key);
        if(found==sparse_.end())return std::nullopt;
        return found->second;
    }
    void insert(const Pair &key,bool disjoint) {
        if(size()>=capacity)clear();
#ifdef TPP_HAS_DENSE_PAIR_CACHE
        if(key.second<dense_ids) {
            const auto mask=std::uint64_t{1}<<(key.second%64);
            auto &row=dense_[key.first];
            if(!(row.known[key.second/64]&mask)) {
                row.known[key.second/64]|=mask;
                if(disjoint)row.disjoint[key.second/64]|=mask;
                const auto reverse_mask=std::uint64_t{1}<<(key.first%64);
                dense_[key.second].known[key.first/64]|=reverse_mask;
                if(disjoint)dense_[key.second].disjoint[key.first/64]|=reverse_mask;
                ++dense_size_;
            }
            return;
        }
#endif
        sparse_.emplace(key,disjoint);
    }
    // A positive bit implies an already stored exact proof. Unknown and
    // intersecting pairs both decline this shortcut and use ordinary dispatch.
    template<class Range,class Identity>
    bool all_known_disjoint(const Range &selected,Identity identity) const {
#ifdef TPP_HAS_DENSE_PAIR_CACHE
        std::array<std::uint64_t,dense_ids/64> mask{};
        size_t words=0;
        for(const auto &entry:selected) {
            const auto id=identity(entry);
            if(id>=dense_ids)return false;
            const auto bit=std::uint64_t{1}<<(id%64);
            if(mask[id/64]&bit)return false; // Repeated polygon: use its diagonal proof.
            mask[id/64]|=bit;
            if(words<id/64+1)words=id/64+1;
        }
        for(const auto &entry:selected) {
            const auto id=identity(entry);
            for(size_t word=0;word<words;++word) {
                const auto required=word==id/64?mask[word]&~(std::uint64_t{1}<<(id%64)):mask[word];
                if((dense_[id].disjoint[word]&required)!=required)return false;
            }
        }
        return true;
#else
        return false;
#endif
    }
    size_t size() const {
#ifdef TPP_HAS_DENSE_PAIR_CACHE
        return dense_size_+sparse_.size();
#else
        return sparse_.size();
#endif
    }
    void clear() {
        sparse_.clear();
#ifdef TPP_HAS_DENSE_PAIR_CACHE
        for(auto &row:dense_)row={};
        dense_size_=0;
#endif
    }
private:
    struct PairHash {
        size_t operator()(const Pair &pair) const {
            const auto first=std::hash<std::uint64_t>{}(pair.first);
            const auto second=std::hash<std::uint64_t>{}(pair.second);
            return first^(second+0x9e3779b97f4a7c15ULL+(first<<6)+(first>>2));
        }
    };
    std::unordered_map<Pair,bool,PairHash> sparse_;
#ifdef TPP_HAS_DENSE_PAIR_CACHE
    struct Row {
        std::array<std::uint64_t,dense_ids/64> known{},disjoint{};
    };
    std::array<Row,dense_ids> dense_{}; // 16 KiB, independent of frontier size.
    size_t dense_size_=0;
#endif
};
}
