#pragma once

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>

namespace tpp::detail {
// Binary64 enclosures, not error tolerances. Each primitive is rounded to a
// stored double before expanding by one representable value in each direction.
// Volatile stores prevent contraction, reassociation and excess precision.
struct CycleInterval {
    double lo=0, hi=0;
    CycleInterval() = default;
    explicit CycleInterval(double x):lo(x),hi(x) {}
    CycleInterval(double l,double h):lo(l),hi(h) {}
    bool zero() const {return lo==0&&hi==0;}
    bool finite() const {return std::isfinite(lo)&&std::isfinite(hi);}
    static double down(double x) {return std::isnan(x)?-INFINITY:std::nextafter(x,-INFINITY);}
    static double up(double x) {return std::isnan(x)?INFINITY:std::nextafter(x,INFINITY);}
    friend CycleInterval operator+(CycleInterval a,CycleInterval b) {
        if(a.zero())return b;if(b.zero())return a;
        volatile double l=a.lo+b.lo,h=a.hi+b.hi;
        return {down(l),up(h)};
    }
    friend CycleInterval operator-(CycleInterval a,CycleInterval b) {
        if(b.zero())return a;
        volatile double l=a.lo-b.hi,h=a.hi-b.lo;
        return {down(l),up(h)};
    }
    friend CycleInterval operator*(CycleInterval a,CycleInterval b) {
        if(a.zero()||b.zero())return {};
        volatile double p=a.lo*b.lo,q=a.lo*b.hi,r=a.hi*b.lo,s=a.hi*b.hi;
        if(std::isnan(p)||std::isnan(q)||std::isnan(r)||std::isnan(s))return {-INFINITY,INFINITY};
        return {down(std::min({double(p),double(q),double(r),double(s)})),
                up(std::max({double(p),double(q),double(r),double(s)}))};
    }
    CycleInterval divided_by(double positive) const {
        if(zero())return {};
        volatile double l=lo/positive,h=hi/positive;
        return {down(l),up(h)};
    }
    CycleInterval square() const {
        if(zero())return {};
        const double a=lo<=0&&hi>=0?0:std::min(std::abs(lo),std::abs(hi));
        const double b=std::max(std::abs(lo),std::abs(hi));
        volatile double l=a*a,h=b*b;
        return {std::max(0.0,down(l)),up(h)};
    }
    CycleInterval sqrt() const {
        if(zero())return {};
        volatile double l=std::sqrt(std::max(0.0,lo)),h=std::sqrt(hi);
        return {std::max(0.0,down(l)),up(h)};
    }
};
inline bool cycle_interval_environment() {
#if defined(__FAST_MATH__)
    return false;
#else
    if(!std::numeric_limits<double>::is_iec559||std::numeric_limits<double>::digits!=53||
       std::fegetround()!=FE_TONEAREST)return false;
    // Reject flush-to-zero / denormals-are-zero environments. The rational
    // certificate remains available on unsupported floating-point platforms.
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1;
    volatile double normal=std::numeric_limits<double>::min(),half=.5;
    volatile double a=tiny*one,b=normal*half;
    return a==tiny&&a>0&&b>0;
#endif
}
}
