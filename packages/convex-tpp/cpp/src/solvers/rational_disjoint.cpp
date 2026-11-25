#include "tpp/convex/detail/rational_disjoint.h"
#include "cycle_internal.h"
#include "disjoint_arithmetic.h"
#include "binary_certificate.h"
#include "tpp/convex/detail/convex_contacts.h"
#include <cmath>
#include <limits>

namespace tpp::detail {
namespace {
using Scalar=ConvexRational;
using Point=ConvexRationalPoint;
using Polygon=ConvexRationalPolygon;
static std::pair<double,double> bounds(const std::vector<Point> &path) {
        constexpr unsigned precision=96;const tpp::ConvexInteger scale=tpp::ConvexInteger(1)<<precision;
        Scalar lower=0,upper=0;
        for(size_t i=1;i<path.size();++i){const Point d=path[i]-path[i-1];const Scalar squared=d.dot(d);if(squared==0)continue;
            const auto numerator=boost::multiprecision::numerator(squared),denominator=boost::multiprecision::denominator(squared);
            const tpp::ConvexInteger scaled=(numerator<<(2*precision))/denominator,root=sqrt(scaled);
            lower+=Scalar(root)/Scalar(scale);upper+=Scalar(root+1)/Scalar(scale);}
        double lo=lower.convert_to<double>();while(Scalar(lo)>lower)lo=std::nextafter(lo,-std::numeric_limits<double>::infinity());
        double hi=upper.convert_to<double>();while(Scalar(hi)<upper)hi=std::nextafter(hi,std::numeric_limits<double>::infinity());
        return {lo,hi};
    }
}

RationalDisjointResult solve_rational_disjoint(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons){
    ConvexRationalPolygons input;
    for(const auto &p:polygons){Polygon exact;for(auto v:p)exact.emplace_back(v);input.push_back(std::move(exact));}
    const auto exact=DisjointArithmetic<Scalar>::Solver(Point(start),Point(target),input).solve();
    RationalDisjointResult result;std::tie(result.lower_bound,result.upper_bound)=bounds(exact.path);
    for(const auto &q:exact.contacts)result.contacts.push_back(q.external());
    return result;
}

RationalDisjointExactResult solve_rational_disjoint_exact(const Point &start,const Point &target,
        const ConvexRationalPolygons &polygons){
    return {DisjointArithmetic<Scalar>::Solver(start,target,polygons).solve(true).contacts};
}

bool prepare_cycle_polygons(const ConvexRationalPolygons &input,ConvexRationalPolygons &normalized,bool check_disjoint) {
    normalized.clear();normalized.reserve(input.size());
    bool degenerate=false;
    for(const auto &source:input) {
        Polygon p;
        for(const auto &q:source)if(p.empty()||!(p.back()==q))p.push_back(q);
        if(p.size()>1&&p.front()==p.back())p.pop_back();
        if(p.empty())throw std::invalid_argument("Empty cycle region");
        // Points and segments are closed convex regions of dimension zero and
        // one. They never take the disjoint fast path below, which needs area.
        if(p.size()<=2){degenerate=true;normalized.push_back(std::move(p));continue;}
        Scalar area=0;
        for(size_t i=0;i<p.size();++i)area+=p[i].cross(p[(i+1)%p.size()]);
        if(area==0)throw std::invalid_argument("Cycle polygon must be a point, a segment or have positive area");
        if(area<0)std::reverse(p.begin(),p.end());
        Polygon strict;size_t winding=0;
        for(size_t i=0;i<p.size();++i) {
            const auto before=p[i]-p[(i+p.size()-1)%p.size()],after=p[(i+1)%p.size()]-p[i];
            const Scalar turn=before.cross(after);
            if(turn<0)throw std::invalid_argument("Cycle polygon is not convex");
            if(before.y<=0&&after.y>0)++winding;
            if(turn==0) {
                if(before.dot(after)<=0)throw std::invalid_argument("Cycle boundary backtracks");
            } else strict.push_back(p[i]);
        }
        // Local turns alone accept multiply-wound stars. One complete turn of
        // the edge directions is also necessary and sufficient for convexity.
        if(winding!=1)throw std::invalid_argument("Cycle boundary winds more than once");
        if(strict.size()<3)throw std::invalid_argument("Degenerate cycle polygon");
        normalized.push_back(std::move(strict));
    }
    if(!check_disjoint||degenerate)return false;
    return cycle_polygons_disjoint(normalized);
}

namespace {
int binary_turn(Vector2 a,Vector2 b,Vector2 c) {
    if(const auto sign=dyadic_orientation(a,b,c))return *sign;
    const Scalar turn=(Point(b)-Point(a)).cross(Point(c)-Point(a));
    return turn>0?1:turn<0?-1:0;
}
}

// The rational checks compute the area's sign, orient by it and then require
// every turn nonnegative. Equivalently, the nonzero turns share one sign, which
// gives the orientation: a counterclockwise boundary with left turns only and
// one complete turn of its edge directions bounds a convex polygon of positive
// area, and the reverse of an accepted boundary has right turns only. The
// direction tests read signs of coordinate differences, which binary64
// comparisons decide exactly; at a zero turn the two edges are nonzero and
// parallel, so one nonzero coordinate gives the sign of their dot product.
BinaryRegionDefect binary_convex_region(const std::vector<Vector2> &input,std::vector<Vector2> &p,bool strict) {
    p.clear();p.reserve(input.size());
    for(const auto &q:input)if(p.empty()||!(p.back().x==q.x&&p.back().y==q.y))p.push_back(q);
    if(p.size()>1&&p.front().x==p.back().x&&p.front().y==p.back().y)p.pop_back();
    if(p.empty())return BinaryRegionDefect::Empty;
    const size_t n=p.size();
    if(n<=2)return BinaryRegionDefect::None;
    std::vector<int> turn(n);bool left=false,right=false;
    for(size_t i=0;i<n;++i) {
        turn[i]=binary_turn(p[(i+n-1)%n],p[i],p[(i+1)%n]);
        left|=turn[i]>0;right|=turn[i]<0;
    }
    if(!left&&!right)return BinaryRegionDefect::ZeroArea;
    if(left&&right)return BinaryRegionDefect::NotConvex;
    if(right) {
        std::reverse(p.begin(),p.end());std::reverse(turn.begin(),turn.end());
        for(auto &t:turn)t=-t;
    }
    std::vector<Vector2> kept;kept.reserve(n);size_t winding=0;
    for(size_t i=0;i<n;++i) {
        const Vector2 a=p[(i+n-1)%n],b=p[i],c=p[(i+1)%n];
        if(turn[i]==0&&(a.x!=b.x?(b.x>a.x)!=(c.x>b.x):(b.y>a.y)!=(c.y>b.y)))return BinaryRegionDefect::Backtracks;
        if(b.y<=a.y&&c.y>b.y)++winding;
        if(!strict||turn[i]!=0)kept.push_back(b);
    }
    if(winding!=1)return BinaryRegionDefect::Winding;
    p=std::move(kept);
    return BinaryRegionDefect::None;
}

bool prepare_cycle_polygons(const std::vector<std::vector<Vector2>> &input,ConvexRationalPolygons &normalized,bool check_disjoint,
        std::vector<std::vector<Vector2>> *binary) {
    normalized.clear();normalized.reserve(input.size());
    std::vector<std::vector<Vector2>> regions;regions.reserve(input.size());
    bool degenerate=false;std::vector<Vector2> p;
    for(const auto &source:input) {
        for(const auto &v:source)if(!v.is_finite())throw std::invalid_argument("Nonfinite cycle region");
        switch(binary_convex_region(source,p,true)) {
            case BinaryRegionDefect::None:break;
            case BinaryRegionDefect::Empty:throw std::invalid_argument("Empty cycle region");
            case BinaryRegionDefect::ZeroArea:throw std::invalid_argument("Cycle polygon must be a point, a segment or have positive area");
            case BinaryRegionDefect::NotConvex:throw std::invalid_argument("Cycle polygon is not convex");
            case BinaryRegionDefect::Backtracks:throw std::invalid_argument("Cycle boundary backtracks");
            case BinaryRegionDefect::Winding:throw std::invalid_argument("Cycle boundary winds more than once");
        }
        if(p.size()<=2)degenerate=true;
        Polygon q;q.reserve(p.size());for(const auto &v:p)q.emplace_back(v);
        normalized.push_back(std::move(q));regions.push_back(p);
    }
    const bool disjoint=check_disjoint&&!degenerate&&cycle_polygons_disjoint(regions);
    if(binary)*binary=std::move(regions);
    return disjoint;
}

// cycle_polygons_disjoint on the rationals, step by step: closed segments
// meet as closed_segments_meet decides (between collinear points, a point on
// the segment lies in its coordinate ranges); comparisons of binary64 values,
// orders and boxes are exact.
bool cycle_polygons_disjoint(const std::vector<std::vector<Vector2>> &normalized) {
    for(const auto &p:normalized)if(p.size()<=2)return false;
    auto contains=[](const std::vector<Vector2> &p,Vector2 q) {
        for(size_t i=0;i<p.size();++i)if(binary_turn(p[i],p[(i+1)%p.size()],q)<0)return false;
        return true;
    };
    auto on=[](Vector2 p,Vector2 q,Vector2 r) {
        return std::min(p.x,q.x)<=r.x&&r.x<=std::max(p.x,q.x)&&std::min(p.y,q.y)<=r.y&&r.y<=std::max(p.y,q.y);
    };
    auto meet=[&](Vector2 a,Vector2 b,Vector2 c,Vector2 d) {
        const int o1=binary_turn(a,b,c),o2=binary_turn(a,b,d),o3=binary_turn(c,d,a),o4=binary_turn(c,d,b);
        if(o1*o2<0&&o3*o4<0)return true;
        return (o1==0&&on(a,b,c))||(o2==0&&on(a,b,d))||(o3==0&&on(c,d,a))||(o4==0&&on(c,d,b));
    };
    std::vector<LexicographicChains> chains;chains.reserve(normalized.size());
    for(const auto &p:normalized)chains.push_back(lexicographic_chains(p));
    struct Box {double x0,y0,x1,y1;};
    std::vector<Box> boxes;boxes.reserve(normalized.size());
    for(const auto &p:normalized) {
        Box box{p.front().x,p.front().y,p.front().x,p.front().y};
        for(const auto &q:p){box.x0=std::min(box.x0,q.x);box.x1=std::max(box.x1,q.x);box.y0=std::min(box.y0,q.y);box.y1=std::max(box.y1,q.y);}
        boxes.push_back(box);
    }
    for(size_t i=0;i<normalized.size();++i)for(size_t j=i+1;j<normalized.size();++j) {
        const auto &a=boxes[i],&b=boxes[j];
        if(a.x1<b.x0||b.x1<a.x0||a.y1<b.y0||b.y1<a.y0)continue;
        const auto &p=normalized[i],&q=normalized[j];
        if(contains(q,p.front())||contains(p,q.front()))return false;
        if(for_each_edge_pair_that_may_touch(p,chains[i],q,chains[j],[&](size_t e,size_t f) {
                return meet(p[e],p[(e+1)%p.size()],q[f],q[(f+1)%q.size()]);
            }))return false;
    }
    return true;
}

bool cycle_polygons_disjoint(const ConvexRationalPolygons &normalized) {
    for(const auto &p:normalized)if(p.size()<=2)return false;
    // Closed polygons are disjoint, that is strictly separated, when neither
    // contains a vertex of the other and their boundaries do not meet. Touching
    // is deliberately rejected. The boundaries take O(|a|+|b|) pairs of edges
    // (convex_contacts.h).
    auto contains=[](const Polygon &p,const Point &q) {
        for(size_t i=0;i<p.size();++i)if((p[(i+1)%p.size()]-p[i]).cross(q-p[i])<0)return false;
        return true;
    };
    std::vector<LexicographicChains> chains;chains.reserve(normalized.size());
    for(const auto &p:normalized)chains.push_back(lexicographic_chains(p));
    auto separated=[&](size_t i,size_t j) {
        const auto &a=normalized[i],&b=normalized[j];
        return !contains(b,a.front()) && !contains(a,b.front()) && !convex_boundaries_meet(a,chains[i],b,chains[j]);
    };
    // Exact bounding boxes first: boxes strictly apart (in either coordinate)
    // separate their polygons strictly. Only overlapping boxes need the test.
    struct Box {Scalar x0,y0,x1,y1;};
    std::vector<Box> boxes;boxes.reserve(normalized.size());
    for(const auto &p:normalized) {
        Box box{p.front().x,p.front().y,p.front().x,p.front().y};
        for(const auto &q:p) {
            if(q.x<box.x0)box.x0=q.x;
            if(q.x>box.x1)box.x1=q.x;
            if(q.y<box.y0)box.y0=q.y;
            if(q.y>box.y1)box.y1=q.y;
        }
        boxes.push_back(std::move(box));
    }
    for(size_t i=0;i<normalized.size();++i)for(size_t j=i+1;j<normalized.size();++j) {
        const auto &a=boxes[i],&b=boxes[j];
        if(a.x1<b.x0||b.x1<a.x0||a.y1<b.y0||b.y1<a.y0)continue;
        if(!separated(i,j))return false;
    }
    return true;
}
}
