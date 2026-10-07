#include "tpp/convex/detail/rational_disjoint.h"
#include "cycle_internal.h"
#include "disjoint_arithmetic.h"
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
    // Strict separating-axis test; closed touching is deliberately rejected.
    auto separated=[](const Polygon &a,const Polygon &b) {
        for(size_t i=0;i<a.size();++i) {
            const Point edge=a[(i+1)%a.size()]-a[i];bool outside=true;
            for(const auto &q:b)if(edge.cross(q-a[i])>=0){outside=false;break;}
            if(outside)return true;
        }
        return false;
    };
    for(size_t i=0;i<normalized.size();++i)for(size_t j=i+1;j<normalized.size();++j)
        if(!separated(normalized[i],normalized[j])&&!separated(normalized[j],normalized[i]))return false;
    return true;
}
}
