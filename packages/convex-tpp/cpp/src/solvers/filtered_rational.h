#pragma once
#include "tpp/convex/rational.h"
#include "cycle_interval.h"
#include <cstddef>
#include <deque>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tpp::detail {
// The directional algorithm keeps its exact rational semantics. Most signs
// are decided by binary64 enclosures; ambiguous comparisons evaluate only
// their expression DAG, caching the exact value. This is arithmetic, not a
// second geometric solver. Exported doubles remain proposals for exact replay.
class FilteredRational {
    enum class Op { Literal, Add, Subtract, Multiply, Divide };
    struct Node {
        Op op;
        double approximate;
        CycleInterval interval;
        const Node *a=nullptr,*b=nullptr;
        mutable const ConvexRational *cached=nullptr;
        explicit Node(double value):op(Op::Literal),approximate(value),interval(value) {}
        Node(Op kind,double value,CycleInterval bounds,const Node *x,const Node *y)
            :op(kind),approximate(value),interval(bounds),a(x),b(y) {}
        const ConvexRational &exact()const;
    };
    // Nodes are immutable once built and are only read while their expression
    // is in use, so they live in a per-thread bump arena released when the
    // outermost Scope ends. This replaces per-operation heap allocation and
    // reference counting; the DAG, intervals and evaluations are unchanged.
    // Values must therefore not outlive the outermost Scope that created them.
    class Arena {
        static constexpr std::size_t chunk_nodes=4096, retained_chunks=16, retained_exact=4096;
        std::vector<std::unique_ptr<std::byte[]>> chunks;
        std::size_t current=std::size_t(-1),used=chunk_nodes;
        // Exact values are assigned in place, so retained slots reuse their
        // limb storage across expressions; a deque keeps references stable.
        std::deque<ConvexRational> exact;
        std::size_t exact_used=0;
    public:
        std::size_t depth=0;
        template<class... A> const Node *make(A&&... arguments) {
            static_assert(std::is_trivially_destructible_v<Node>);
            if(used==chunk_nodes) {
                if(++current==chunks.size())
                    chunks.push_back(std::make_unique_for_overwrite<std::byte[]>(chunk_nodes*sizeof(Node)));
                used=0;
            }
            return ::new(chunks[current].get()+sizeof(Node)*used++) Node(std::forward<A>(arguments)...);
        }
        ConvexRational &slot() {
            if(exact_used==exact.size())exact.emplace_back();
            return exact[exact_used++];
        }
        // Retain a bounded amount of node storage for the next expression.
        void reset() {
            if(exact.size()>retained_exact)exact.resize(retained_exact);
            exact_used=0;
            if(chunks.size()>retained_chunks)chunks.resize(retained_chunks);
            current=std::size_t(-1);used=chunk_nodes;
        }
    };
    static Arena &arena() {thread_local Arena value;return value;}
    inline static thread_local bool filter_enabled=false;
    const Node *node;
    struct FromNode {};
    FilteredRational(FromNode,const Node *value):node(value) {}
    static CycleInterval divide(CycleInterval a,CycleInterval b) {
        if(b.lo<=0&&b.hi>=0)return {-INFINITY,INFINITY};
        volatile double l=1/b.hi,h=1/b.lo;
        return a*CycleInterval{CycleInterval::down(l),CycleInterval::up(h)};
    }
    static FilteredRational operation(Op kind,const FilteredRational &a,const FilteredRational &b) {
        const auto x=a.node->interval,y=b.node->interval;
        CycleInterval bounds;double value;
        switch(kind) {
            case Op::Add:
                if(filter_enabled&&x.zero())return b;if(filter_enabled&&y.zero())return a;
                bounds=x+y;value=a.node->approximate+b.node->approximate;break;
            case Op::Subtract:
                if(filter_enabled&&y.zero())return a;
                if(a.node==b.node)return 0;
                bounds=x-y;value=a.node->approximate-b.node->approximate;break;
            case Op::Multiply:
                if(filter_enabled&&(x.zero()||y.zero()))return 0;
                bounds=x*y;value=a.node->approximate*b.node->approximate;break;
            case Op::Divide:
                bounds=divide(x,y);value=a.node->approximate/b.node->approximate;break;
            default: throw std::logic_error("Invalid filtered arithmetic operation");
        }
        // Expressions built in an unsupported environment remain unresolved
        // even if a later scope enables filtering again.
        if(!filter_enabled)bounds={-INFINITY,INFINITY};
        return FilteredRational(FromNode{},arena().make(kind,value,bounds,a.node,b.node));
    }
    static int compare(const FilteredRational &a,const FilteredRational &b) {
        if(a.node==b.node)return 0;
        if(filter_enabled) {
            const auto x=a.node->interval,y=b.node->interval;
            if(x.hi<y.lo)return -1;
            if(x.lo>y.hi)return 1;
            if(x.lo==x.hi&&y.lo==y.hi&&x.lo==y.lo)return 0;
        }
        const auto &x=a.node->exact(),&y=b.node->exact();
        return x<y?-1:x>y?1:0;
    }
public:
    // The interval of an expression without building its DAG. Every operator
    // repeats the shortcuts and enclosures of operation(), and an intermediate
    // result never shares a node with another value, so sign() decides exactly
    // the comparisons that the materialized expression would decide by its
    // interval. An undecided sign is left to the materialized expression.
    class Virtual {
        CycleInterval interval;
        const Node *node=nullptr;
        Virtual(CycleInterval bounds):interval(bounds) {}
    public:
        explicit Virtual(const FilteredRational &value):interval(value.node->interval),node(value.node) {}
        friend Virtual operator+(const Virtual &a,const Virtual &b) {
            if(a.interval.zero())return b;
            if(b.interval.zero())return a;
            return a.interval+b.interval;
        }
        friend Virtual operator-(const Virtual &a,const Virtual &b) {
            if(b.interval.zero())return a;
            if(a.node&&a.node==b.node)return CycleInterval(0.0);
            return a.interval-b.interval;
        }
        friend Virtual operator*(const Virtual &a,const Virtual &b) {
            if(a.interval.zero()||b.interval.zero())return CycleInterval(0.0);
            return a.interval*b.interval;
        }
        // compare(*this, 0) decided by enclosures; only valid while filtering.
        std::optional<int> sign() const {
            if(interval.hi<0)return -1;
            if(interval.lo>0)return 1;
            if(interval.zero())return 0;
            return std::nullopt;
        }
    };
    // Virtual arithmetic mirrors the filtered shortcuts, which are disabled in
    // an unsupported environment; callers then use the materialized DAG.
    static bool filtering() {return filter_enabled;}
    class Scope {
        bool previous=filter_enabled;
    public:
        Scope(){filter_enabled=cycle_interval_environment();++arena().depth;}
        ~Scope(){filter_enabled=previous;if(--arena().depth==0)arena().reset();}
        Scope(const Scope &)=delete;
        Scope &operator=(const Scope &)=delete;
    };
    FilteredRational():FilteredRational(0.0) {}
    FilteredRational(double value):node(arena().make(value)) {}
    template<class T> T convert_to()const {
        if(node->cached) {
            if constexpr(std::is_same_v<T,double>)return convex_nearest_double(*node->cached);
            else return node->cached->template convert_to<T>();
        }
        return static_cast<T>(node->approximate);
    }
    FilteredRational operator+(const FilteredRational &b)const{return operation(Op::Add,*this,b);}
    FilteredRational operator-(const FilteredRational &b)const{return operation(Op::Subtract,*this,b);}
    FilteredRational operator*(const FilteredRational &b)const{return operation(Op::Multiply,*this,b);}
    FilteredRational operator/(const FilteredRational &b)const{return operation(Op::Divide,*this,b);}
    FilteredRational operator-()const{return FilteredRational(0)-*this;}
    FilteredRational &operator+=(const FilteredRational &b){*this=*this+b;return *this;}
    friend FilteredRational operator*(double a,const FilteredRational &b){return FilteredRational(a)*b;}
    bool operator==(const FilteredRational &b)const{return compare(*this,b)==0;}
    bool operator<(const FilteredRational &b)const{return compare(*this,b)<0;}
    bool operator>(const FilteredRational &b)const{return compare(*this,b)>0;}
    bool operator<=(const FilteredRational &b)const{return compare(*this,b)<=0;}
    bool operator>=(const FilteredRational &b)const{return compare(*this,b)>=0;}
};

inline const ConvexRational &FilteredRational::Node::exact()const {
    if(!cached) {
        ConvexRational &value=arena().slot();
        switch(op) {
            case Op::Literal: value=approximate;break;
            case Op::Add: value=a->exact()+b->exact();break;
            case Op::Subtract: value=a->exact()-b->exact();break;
            case Op::Multiply: value=a->exact()*b->exact();break;
            case Op::Divide: value=a->exact()/b->exact();break;
        }
        cached=&value;
    }
    return *cached;
}
}
