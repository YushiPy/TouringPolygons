#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tpp::unordered_detail {
    inline constexpr size_t no_sequence_index = std::numeric_limits<size_t>::max();
    struct SequenceElement {
        size_t polygon;
        size_t piece = no_sequence_index;
        bool operator==(const SequenceElement &) const = default;
    };

    // Simple polygons have a diagonal partition with at most vertices-2 pieces.
    // Reserve the largest encoded value for the hull marker. No eager partition
    // is needed, and every actual encoding also checks for overflow.
    inline size_t sequence_index_bytes(size_t polygons, size_t max_vertices) {
        const auto count = std::max(polygons, max_vertices > 2 ? max_vertices-2 : size_t(0));
        if (count <= UINT8_MAX) return 1;
        if (count <= UINT16_MAX) return 2;
        if (count <= UINT32_MAX) return 4;
        return 8;
    }
    template<class Index> Index encode_sequence_index(size_t value) {
        if (value == no_sequence_index) return std::numeric_limits<Index>::max();
        if (value >= std::numeric_limits<Index>::max())
            throw std::overflow_error("Sequence index exceeds the selected storage width.");
        return static_cast<Index>(value);
    }
    template<class Index> size_t decode_sequence_index(Index value) {
        return value == std::numeric_limits<Index>::max() ? no_sequence_index : static_cast<size_t>(value);
    }

    struct PackedSequence {
        std::vector<uint8_t> bytes;
        template<class Index> static PackedSequence encode(const std::vector<SequenceElement> &sequence) {
            PackedSequence out;
            out.bytes.resize(2*sizeof(Index)*sequence.size());
            auto *destination = out.bytes.data();
            for (auto e : sequence) {
                const Index pair[] = {encode_sequence_index<Index>(e.polygon), encode_sequence_index<Index>(e.piece)};
                std::memcpy(destination, pair, sizeof(pair));
                destination += sizeof(pair);
            }
            return out;
        }
        template<class Index> std::vector<SequenceElement> decode() const {
            std::vector<SequenceElement> out(bytes.size()/(2*sizeof(Index)));
            const auto *source = bytes.data();
            for (auto &e : out) {
                Index pair[2];
                std::memcpy(pair, source, sizeof(pair));
                source += sizeof(pair);
                e = {decode_sequence_index(pair[0]), decode_sequence_index(pair[1])};
            }
            return out;
        }
    };

    class SequenceHistory;
    // References retain sequence records only; no ancestor path or B&B node.
    class SequenceReference {
        friend class SequenceHistory;
        SequenceHistory *arena = nullptr;
        uint32_t id = UINT32_MAX, length = 0;
        SequenceReference(SequenceHistory *owner, uint32_t record, uint32_t size)
            : arena(owner), id(record), length(size) {}
    public:
        SequenceReference() = default;
        SequenceReference(const SequenceReference &other);
        SequenceReference(SequenceReference &&other) noexcept
            : arena(std::exchange(other.arena,nullptr)), id(other.id), length(other.length) {}
        SequenceReference &operator=(SequenceReference other) noexcept {
            std::swap(arena,other.arena);std::swap(id,other.id);std::swap(length,other.length);return *this;
        }
        ~SequenceReference();
        explicit operator bool() const { return arena != nullptr; }
        size_t size() const { return length; }
        std::vector<SequenceElement> expand() const;
    };

    class SequenceHistory {
        friend class SequenceReference;
        enum class Operation : uint8_t { Root, Insert, Piece };
        template<class Index> struct Record {
            uint32_t parent = UINT32_MAX, references = 1;
            Index polygon = 0, piece = 0, position = 0;
            Operation operation = Operation::Root;
        };
        using Pools = std::variant<std::vector<Record<uint8_t>>,std::vector<Record<uint16_t>>,
            std::vector<Record<uint32_t>>,std::vector<Record<uint64_t>>>;
        Pools pool;
        uint32_t free_head = UINT32_MAX;
        size_t live = 0;
        bool root_created = false;
        std::vector<SequenceElement> root;
        std::vector<size_t> latest_pieces;
        std::vector<size_t> free_slots;

        void retain(uint32_t id) {
            std::visit([&](auto &records) {
                auto &record = records[id];assert(record.references);
                if (record.references == UINT32_MAX) throw std::overflow_error("Too many sequence references.");
                ++record.references;
            },pool);
        }
        void release(uint32_t id) noexcept {
            std::visit([&](auto &records) {
                // Iterative destruction avoids recursive shared_ptr chains.
                while (id != UINT32_MAX) {
                    auto &record = records[id];assert(record.references);
                    if (--record.references) break;
                    const auto parent = record.parent;
                    record.parent = free_head;free_head = id;--live;
                    id = parent;
                }
            },pool);
        }
        template<class Records> uint32_t allocate(Records &records, typename Records::value_type record) {
            uint32_t id;
            if (free_head != UINT32_MAX) {
                id = free_head;free_head = records[id].parent;records[id] = record;
            } else {
                if (records.size() == UINT32_MAX) throw std::overflow_error("Sequence history exhausted 32-bit record IDs.");
                id = static_cast<uint32_t>(records.size());records.push_back(record);
            }
            ++live;peak_live_records = std::max(peak_live_records,live);
            return id;
        }
        // Select a zero-based rank from at most 64 vacant final positions.
        static size_t select_slot(uint64_t mask, size_t rank) {
            size_t position = 0;
            for (size_t half=32;half;half/=2) {
                const auto below = std::popcount(mask & ((uint64_t(1)<<half)-1));
                if (rank >= static_cast<size_t>(below)) { rank -= below;mask >>= half;position += half; }
                else mask &= (uint64_t(1)<<half)-1;
            }
            return position;
        }
        std::vector<SequenceElement> expand(uint32_t id, size_t length) {
            ++reconstructions;
            std::vector<SequenceElement> out(length);
            if (!length) return out;
            std::fill(latest_pieces.begin(),latest_pieces.end(),no_sequence_index);
            uint64_t vacant = length <= 64 ? (UINT64_MAX >> (64-length)) : 0;
            if (length > 64) {
                // Fenwick tree of vacant slots, reused between expansions.
                free_slots.resize(length+1);
                for (size_t i=1;i<=length;++i) free_slots[i] = i & (~i+1);
            }
            auto take = [&](size_t rank) {
                if (length <= 64) {
                    const auto position = select_slot(vacant,rank);
                    assert(position < length && (vacant & (uint64_t(1)<<position)));
                    vacant &= ~(uint64_t(1)<<position);return position;
                }
                size_t position = 0;
                for (size_t step=std::bit_floor(length);step;step/=2) {
                    const size_t next = position+step;
                    if (next <= length && free_slots[next] <= rank) {
                        position=next;rank-=free_slots[next];
                    }
                }
                assert(position < length);
                for (size_t i=position+1;i<=length;i+=i & (~i+1)) --free_slots[i];
                return position;
            };
            std::visit([&](const auto &records) {
                // Reverse insertion ranks refer to the still-vacant positions.
                // A later piece assignment wins before its insertion is reached.
                while (records[id].operation != Operation::Root) {
                    const auto &record = records[id];
                    const size_t polygon = decode_sequence_index(record.polygon);
                    if (record.operation == Operation::Piece) {
                        if (latest_pieces[polygon] == no_sequence_index)
                            latest_pieces[polygon] = decode_sequence_index(record.piece);
                    } else out[take(record.position)] = {polygon,latest_pieces[polygon]};
                    id = record.parent;
                }
                for (auto e : root) {
                    if (latest_pieces[e.polygon] != no_sequence_index) e.piece = latest_pieces[e.polygon];
                    out[take(0)] = e;
                }
            },pool);
            return out;
        }
    public:
        size_t peak_live_records = 0, reconstructions = 0;
        SequenceHistory(size_t index_bytes, size_t polygons) : latest_pieces(polygons,no_sequence_index) {
            switch(index_bytes) {
                case 1: pool.emplace<0>();break;case 2: pool.emplace<1>();break;
                case 4: pool.emplace<2>();break;case 8: pool.emplace<3>();break;
                default: throw std::invalid_argument("Invalid sequence index width.");
            }
        }
        ~SequenceHistory() { assert(live == 0); }
        SequenceHistory(const SequenceHistory &) = delete;
        SequenceHistory &operator=(const SequenceHistory &) = delete;
        size_t live_records() const { return live; }
        size_t record_bytes() const {
            return std::visit([](const auto &records) { return sizeof(typename std::decay_t<decltype(records)>::value_type); },pool);
        }
        size_t reserved_bytes() const {
            return std::visit([](const auto &records) { return records.capacity()*sizeof(typename std::decay_t<decltype(records)>::value_type); },pool)
                + root.capacity()*sizeof(SequenceElement) + latest_pieces.capacity()*sizeof(size_t) + free_slots.capacity()*sizeof(size_t);
        }
        SequenceReference snapshot(const std::vector<SequenceElement> &sequence) {
            if (root_created) throw std::logic_error("A sequence history has only one root.");
            if (sequence.size() > UINT32_MAX) throw std::overflow_error("Sequence too long for delta storage.");
            root = sequence;root_created=true;
            const auto id = std::visit([&](auto &records) {
                return allocate(records,typename std::decay_t<decltype(records)>::value_type{});
            },pool);
            return {this,id,static_cast<uint32_t>(sequence.size())};
        }
        SequenceReference child(const SequenceReference &parent, size_t polygon, size_t piece, size_t position) {
            if (parent.arena != this) throw std::logic_error("Sequence parent belongs to another search.");
            if (polygon >= latest_pieces.size()) throw std::out_of_range("Invalid sequence delta polygon.");
            const bool inserting = piece == no_sequence_index;
            if ((inserting && parent.length == UINT32_MAX) || position > parent.length || (!inserting && position == parent.length))
                throw std::out_of_range("Invalid sequence delta position.");
            const auto id = std::visit([&](auto &records) {
                typename std::decay_t<decltype(records)>::value_type record;
                using Index = decltype(record.polygon);
                record.polygon=encode_sequence_index<Index>(polygon);
                record.piece=encode_sequence_index<Index>(piece);
                record.position=encode_sequence_index<Index>(position);
                record.operation=inserting?Operation::Insert:Operation::Piece;
                record.parent=parent.id;
                retain(parent.id);
                try { return allocate(records,record); }
                catch (...) { release(parent.id);throw; }
            },pool);
            return {this,id,parent.length+uint32_t(inserting)};
        }
    };
    inline SequenceReference::SequenceReference(const SequenceReference &other)
        : arena(other.arena), id(other.id), length(other.length) { if(arena) arena->retain(id); }
    inline SequenceReference::~SequenceReference() { if(arena) arena->release(id); }
    inline std::vector<SequenceElement> SequenceReference::expand() const { return arena->expand(id,length); }

    // Oracles and current siblings use native vectors. Only retained frontier
    // sequences are packed or represented by deltas, using the same B&B code.
    class NodeSequence {
        std::variant<std::vector<SequenceElement>,PackedSequence,SequenceReference> data;
        std::vector<SequenceElement> &full() { return std::get<0>(data); }
        const std::vector<SequenceElement> &full() const { return std::get<0>(data); }
    public:
        NodeSequence() = default;
        NodeSequence(std::vector<SequenceElement> sequence) : data(std::move(sequence)) {}
        size_t size() const {
            if (const auto *history=std::get_if<2>(&data)) return history->size();
            // Packed size is never queried until restore(); the frontier does
            // not need sequence lengths for ordering or lower bounds.
            return full().size();
        }
        bool empty() const { return size()==0; }
        auto begin() { return full().begin(); }
        auto end() { return full().end(); }
        auto begin() const { return full().begin(); }
        auto end() const { return full().end(); }
        auto &operator[](size_t i) { return full()[i]; }
        const auto &operator[](size_t i) const { return full()[i]; }
        NodeSequence inserted(size_t position, SequenceElement element) const {
            std::vector<SequenceElement> out;
            out.reserve(size()+1);
            out.insert(out.end(),begin(),begin()+position);
            out.push_back(element);out.insert(out.end(),begin()+position,end());
            return out;
        }
        bool stored() const { return data.index()!=0; }
        size_t payload_bytes() const {
            if (const auto *packed=std::get_if<1>(&data)) return packed->bytes.capacity();
            if (data.index()==2) return 0;
            return full().capacity()*sizeof(SequenceElement);
        }
        void pack(size_t bytes) {
            if (stored()) return;
            PackedSequence packed;
            switch(bytes) {
                case 1: packed=PackedSequence::encode<uint8_t>(full());break;
                case 2: packed=PackedSequence::encode<uint16_t>(full());break;
                case 4: packed=PackedSequence::encode<uint32_t>(full());break;
                case 8: packed=PackedSequence::encode<uint64_t>(full());break;
                default: throw std::invalid_argument("Invalid sequence index width.");
            }
            data=std::move(packed);
        }
        void save(SequenceReference reference) { data=std::move(reference); }
        const auto &elements() const { return full(); }
        SequenceReference restore(size_t bytes) {
            SequenceReference parent;
            if (const auto *history=std::get_if<2>(&data)) {
                parent=*history;data=parent.expand();
            } else if (const auto *packed=std::get_if<1>(&data)) {
                std::vector<SequenceElement> sequence;
                switch(bytes) {
                    case 1: sequence=packed->decode<uint8_t>();break;case 2: sequence=packed->decode<uint16_t>();break;
                    case 4: sequence=packed->decode<uint32_t>();break;case 8: sequence=packed->decode<uint64_t>();break;
                    default: throw std::invalid_argument("Invalid sequence index width.");
                }
                data=std::move(sequence);
            }
            return parent;
        }
    };
}
