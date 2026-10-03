#pragma once
#include "tpp/convex/rational.h"
#include "cycle_interval.h"
#include <memory>
#include <optional>
#include <stdexcept>

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
        std::shared_ptr<Node> a,b;
        mutable std::optional<ConvexRational> cached;
        explicit Node(double value):op(Op::Literal),approximate(value),interval(value) {}
        Node(Op kind,double value,CycleInterval bounds,std::shared_ptr<Node> x,std::shared_ptr<Node> y)
            :op(kind),approximate(value),interval(bounds),a(std::move(x)),b(std::move(y)) {}
        const ConvexRational &exact()const {
            if(!cached) {
                switch(op) {
                    case Op::Literal: cached.emplace(approximate);break;
                    case Op::Add: cached.emplace(a->exact()+b->exact());break;
                    case Op::Subtract: cached.emplace(a->exact()-b->exact());break;
                    case Op::Multiply: cached.emplace(a->exact()*b->exact());break;
                    case Op::Divide: cached.emplace(a->exact()/b->exact());break;
                }
            }
            return *cached;
        }
    };
    std::shared_ptr<Node> node;
    inline static thread_local bool filter_enabled=false;
    explicit FilteredRational(std::shared_ptr<Node> value):node(std::move(value)) {}
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
        return FilteredRational(std::make_shared<Node>(kind,value,bounds,a.node,b.node));
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
    class Scope {
        bool previous=filter_enabled;
    public:
        Scope(){filter_enabled=cycle_interval_environment();}
        ~Scope(){filter_enabled=previous;}
    };
    FilteredRational():FilteredRational(0.0) {}
    FilteredRational(double value):node(std::make_shared<Node>(value)) {}
    template<class T> T convert_to()const {
        if(node->cached)return node->cached->template convert_to<T>();
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
}
