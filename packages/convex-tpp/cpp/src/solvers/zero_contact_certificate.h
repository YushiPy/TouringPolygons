#pragma once
#include "tpp/convex/rational.h"
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace tpp::detail {
using ConvexHomogeneousDirection=ConvexArithmeticPoint<ConvexInteger>;

inline ConvexHomogeneousDirection convex_reduce_direction(ConvexHomogeneousDirection p) {
    // Repeated reflections can accumulate a large common scale. This is a
    // gcd trigger, not a precision cap: coprime coordinates remain unbounded.
    static const ConvexInteger upper=ConvexInteger(1)<<512,lower=-upper;
    if(p.x>lower&&p.x<upper&&p.y>lower&&p.y<upper)return p;
    const ConvexInteger x=p.x<0?-p.x:p.x,y=p.y<0?-p.y:p.y;
    const ConvexInteger divisor=boost::multiprecision::gcd(x,y);
    if(divisor>1){p.x/=divisor;p.y/=divisor;}
    return p;
}

// Sign(p/sqrt(a2) - q/sqrt(b2)), without square roots or approximate signs.
inline int convex_normalized_difference_sign(const ConvexRational &p, const ConvexRational &a2,
                                           const ConvexRational &q, const ConvexRational &b2) {
    if(p>=0 && q<=0)return p==0 && q==0?0:1;
    if(p<=0 && q>=0)return p==0 && q==0?0:-1;
    // Compare p*p*b2 and q*q*a2 after clearing positive denominators.
    // Only a sign is needed: normalizing intermediate fractions with gcds
    // does not contribute to the certificate. Both integer backends remain
    // unbounded, including when the products exceed the binary64 range.
    using boost::multiprecision::numerator;
    using boost::multiprecision::denominator;
    const ConvexInteger p_scaled=numerator(p)*denominator(q);
    const ConvexInteger q_scaled=numerator(q)*denominator(p);
    const ConvexInteger left=p_scaled*p_scaled*numerator(b2)*denominator(a2);
    const ConvexInteger right=q_scaled*q_scaled*numerator(a2)*denominator(b2);
    if(left==right)return 0;
    if(p>0)return left>right?1:-1;
    return left<right?1:-1;
}

// The same sign for integer positive multiples of the rational vectors:
// p and q share the factor of their common feasible vector, while a2 (b2)
// carries the squared factor of p's (q's) direction, so each normalized
// quotient is unchanged.
inline int convex_normalized_difference_sign_integer(const ConvexInteger &p, const ConvexInteger &a2,
                                                     const ConvexInteger &q, const ConvexInteger &b2) {
    if(p>=0 && q<=0)return p==0 && q==0?0:1;
    if(p<=0 && q>=0)return p==0 && q==0?0:-1;
    const ConvexInteger left=p*p*b2,right=q*q*a2;
    if(left==right)return 0;
    if(p>0)return left>right?1:-1;
    return left<right?1:-1;
}

// Reachable subgradients at a zero-link block. R is the unit disk intersected
// with halfplanes n.u <= n.d/|d|. Every extreme point of R lies on the circle:
// adding a cone retains only old extreme points, and clipping by the disk
// creates new extreme points only on the circle. Each circular vertex has a
// rational direction d; its actual normalization never needs to be evaluated.
template<class Scalar>
class BasicConvexDualReachability {
    using R=Scalar;
    using P=ConvexArithmeticPoint<R>;
    static P direction(const ConvexRationalPoint &p) {
        if constexpr(std::is_same_v<R,ConvexInteger>) {
            using boost::multiprecision::numerator;
            using boost::multiprecision::denominator;
            const auto dx=denominator(p.x),dy=denominator(p.y);
            if(dx==dy)return convex_reduce_direction({numerator(p.x),numerator(p.y)});
            // Both denominators are positive. Clearing them changes only
            // the positive scale of a direction, never its unit vector.
            return convex_reduce_direction({numerator(p.x)*dy,numerator(p.y)*dx});
        } else return p;
    }
    static int normalized_sign(const R &p,const R &a2,const R &q,const R &b2) {
        if constexpr(std::is_same_v<R,ConvexInteger>) {
            if(p>=0&&q<=0)return p==0&&q==0?0:1;
            if(p<=0&&q>=0)return p==0&&q==0?0:-1;
            const R left=p*p*b2,right=q*q*a2;
            if(left==right)return 0;
            if(p>0)return left>right?1:-1;
            return left<right?1:-1;
        } else return convex_normalized_difference_sign(p,a2,q,b2);
    }
    static P reflect(const P &normal,const P &d) {
        if constexpr(std::is_same_v<R,ConvexInteger>)
            // Multiply the rational reflection by |normal|^2 > 0.
            return convex_reduce_direction(normal*(R(2)*normal.dot(d))-d*normal.dot(normal));
        else return normal*(R(2)*normal.dot(d)/normal.dot(normal))-d;
    }
    struct Halfplane {
        P normal,direction;
        R dot,squared;
        Halfplane(P n,P d):normal(std::move(n)),direction(std::move(d)),
            dot(normal.dot(direction)),squared(direction.dot(direction)) {}
    };
    std::vector<Halfplane> planes;
    size_t &predicates;
    void (*checkpoint)();
    void check()const {if(checkpoint)checkpoint();}
    bool contains(const P &d)const {
        const R squared=d.dot(d);
        for(const auto &h:planes) {
            check();
            ++predicates;
            if(normalized_sign(h.normal.dot(d),squared,h.dot,h.squared)>0)return false;
        }
        return true;
    }
    P support(const P &normal)const {
        if(contains(normal))return normal; // Unconstrained disk maximum.
        P best;R best_dot=0,best_squared=0;bool found=false;
        auto consider=[&](const P &d) {
            if(!contains(d))return;
            const R dot=normal.dot(d),squared=d.dot(d);
            if(!found||normalized_sign(dot,squared,best_dot,best_squared)>0) {
                best=d;best_dot=dot;best_squared=squared;found=true;
            }
        };
        for(const auto &h:planes) {
            check();
            consider(h.direction);
            // The other intersection of a chord line with the unit circle is
            // the reflection in its normal axis. It also has rational direction.
            consider(reflect(h.normal,h.direction));
        }
        if(!found)throw std::logic_error("Empty reachable dual set");
        return best;
    }
public:
    BasicConvexDualReachability(const ConvexRationalPoint &incoming,size_t &count,void (*stop)()=nullptr)
        :predicates(count),checkpoint(stop) {
        // The disk and its reverse tangent halfplane intersect in one point.
        const auto d=direction(incoming);
        planes.emplace_back(-d,d);
    }
    void advance(const ConvexRationalPolygon &p,const ConvexRationalPoint &q) {
        if(p.size()==1){planes.clear();return;} // A fixed contact imposes no dual constraint.
        std::vector<P> active,tangents;
        auto append_direction=[](std::vector<P> &directions,const P &d) {
            if(d.zero())return;
            if(std::none_of(directions.begin(),directions.end(),[&](const P &e) {
                return d.cross(e)==0&&d.dot(e)>0;
            }))directions.push_back(d);
        };
        if(p.size()==2) {
            append_direction(tangents,direction(p[0]-q));append_direction(tangents,direction(p[1]-q));
        } else {
            for(size_t j=0;j<p.size();++j) {
                const auto edge=p[(j+1)%p.size()]-p[j];
                ++predicates;
                if(edge.cross(q-p[j])==0)append_direction(active,direction(edge));
            }
            if(active.empty())return; // Interior: the normal cone is {0}.
        }
        auto allowed=[&](const P &n) {
            if(p.size()==2) {
                const P e=direction(p[1]-p[0]);
                if(e.cross(n)!=0)return false;
                if(q==p[0])return e.dot(n)>=0;
                if(q==p[1])return e.dot(n)<=0;
                return true;
            }
            for(const P &e:active)if(e.cross(n)<0)return false;
            return true;
        };
        for(const P &e:active) {
            if(allowed(e))append_direction(tangents,e);
            if(allowed(-e))append_direction(tangents,-e);
        }
        // R' = D intersect (R + N_P(q)), where N is the outward normal cone.
        // Keep normals in its polar (the feasible tangent cone), and add its
        // extreme rays with the OLD set's support values. Circular constraints
        // remain redundant because D is imposed throughout.
        std::vector<Halfplane> next;
        for(const auto &h:planes)if(allowed(h.normal))next.push_back(h);
        for(const P &n:tangents) {
            const P d=support(n);
            std::erase_if(next,[&](const Halfplane &h){return h.normal.cross(n)==0&&h.normal.dot(n)>0;});
            next.emplace_back(n,d);
        }
        planes=std::move(next);
    }
    bool reaches(const ConvexRationalPoint &outgoing)const {return contains(direction(outgoing));}
};

#ifdef TPP_HAS_HOMOGENEOUS_ZERO_DUAL
using ConvexDualReachability=BasicConvexDualReachability<ConvexInteger>;
#else
using ConvexDualReachability=BasicConvexDualReachability<ConvexRational>;
#endif

} // namespace tpp::detail
