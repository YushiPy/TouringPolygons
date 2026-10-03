#include "tpp/convex/hybrid.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/detail/rational_disjoint.h"
#include "tpp/convex/solver.h"
#include "common.h"
#include "certified_internal.h"
#include "zero_contact_certificate.h"
#include "binary_certificate.h"
#include "polygon_view.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace tpp {
namespace {
std::mutex aggregate_mutex;
ConvexHybridAggregate aggregate;

struct AggregateRecorder {
    ConvexHybridResult *result;
    ~AggregateRecorder() {
        std::lock_guard lock(aggregate_mutex);const auto &r=*result;
        ++aggregate.total_calls;aggregate.disjoint_calls+=r.stats.disjoint;
        aggregate.certified_double_disjoint_calls+=r.backend==ConvexHybridBackend::DoubleDisjoint&&r.stats.double_certified;
        aggregate.certified_double_intersection_calls+=r.backend==ConvexHybridBackend::DoubleIntersection&&r.stats.double_certified;
        aggregate.interval_bound_calls+=r.stats.interval_bounds_certified;
        aggregate.interval_contracted_calls+=r.stats.interval_bounds_contracted;
        aggregate.rational_disjoint_fallbacks+=r.backend==ConvexHybridBackend::RationalDisjoint;
        aggregate.rational_disjoint_directional_recoveries+=r.stats.rational_disjoint_directional_recovery;
        aggregate.rational_intersection_fallbacks+=r.backend==ConvexHybridBackend::RationalIntersection;
        ++aggregate.fallback_reasons[static_cast<size_t>(r.fallback_reason)];
        aggregate.predicate_exact_evaluations+=r.stats.predicate_exact_evaluations;
        aggregate.zero_link_witnesses+=r.stats.zero_link_witnesses;
        aggregate.dispatch_pair_queries+=r.stats.dispatch_pair_queries;
        aggregate.dispatch_pair_cache_hits+=r.stats.dispatch_pair_cache_hits;
        aggregate.dispatch_pair_exact_checks+=r.stats.dispatch_pair_exact_checks;
        aggregate.dispatch_seconds+=r.stats.dispatch_seconds;
        aggregate.bound_evaluation_seconds+=r.stats.bound_evaluation_seconds;
        aggregate.proposal_preparation_seconds+=r.stats.proposal_preparation_seconds;
        aggregate.double_solver_seconds+=r.stats.double_solver_seconds;
        aggregate.contact_materialization_seconds+=r.stats.contact_materialization_seconds;
        aggregate.certificate_seconds+=r.stats.certificate_seconds;
        aggregate.rational_fallback_seconds+=r.stats.rational_fallback_seconds;
        aggregate.total_seconds+=r.stats.total_seconds;
    }
};
}

void reset_convex_hybrid_aggregate() {std::lock_guard lock(aggregate_mutex);aggregate={};}
ConvexHybridAggregate convex_hybrid_aggregate() {std::lock_guard lock(aggregate_mutex);return aggregate;}

namespace {
using Rational = tpp::ConvexRational;
using Clock = std::chrono::steady_clock;

using Point = tpp::ConvexRationalPoint;
using Polygon=std::vector<Point>;
using ExactPolygons=detail::PolygonView<Point>;
enum class ContactFeatureKind { Interior, Edge, Vertex };
struct ContactFeature { ContactFeatureKind kind=ContactFeatureKind::Interior;size_t index=0; };

double elapsed(Clock::time_point began) {
    return std::chrono::duration<double>(Clock::now()-began).count();
}

struct PhaseTimer {
    double &seconds;
    Clock::time_point began=Clock::now();
    ~PhaseTimer(){seconds+=elapsed(began);}
};

Polygon exact_polygon(const std::vector<Vector2> &input) {
    Polygon p;
    for(auto v:input) {
        if(!v.is_finite()) throw std::invalid_argument("Nonfinite polygon coordinate");
        Point q(v);if(p.empty() || !(p.back()==q))p.push_back(std::move(q));
    }
    if(p.size()>1 && p.front()==p.back())p.pop_back();
    Rational area=0;
    for(size_t i=0;i<p.size();++i)area+=p[i].cross(p[(i+1)%p.size()]);
    if(p.size()<3 || area==0)throw std::invalid_argument("Polygon must have positive area");
    if(area<0)std::reverse(p.begin(),p.end());
    return p;
}

std::vector<Polygon> exact_polygons(const std::vector<std::vector<Vector2>> &polygons) {
    std::vector<Polygon> result;
    result.reserve(polygons.size());
    for(const auto &input:polygons) result.push_back(exact_polygon(input));
    return result;
}

bool inside(const Point &q,const Polygon &p) {
    for(size_t i=0;i<p.size();++i)
        if((p[(i+1)%p.size()]-p[i]).cross(q-p[i])<0)return false;
    return true;
}

bool clip(const Point &a,const Point &b,const Polygon &p,Rational floor,Rational &lo,Rational &hi) {
    const Point d=b-a;lo=floor;hi=1;
    for(size_t i=0;i<p.size();++i) {
        const Point edge=p[(i+1)%p.size()]-p[i];
        const Rational constant=edge.cross(a-p[i]),slope=edge.cross(d);
        if(slope>0) {const Rational t=-constant/slope;if(t>lo)lo=t;}
        else if(slope<0) {const Rational t=-constant/slope;if(t<hi)hi=t;}
        else if(constant<0)return false;
    }
    return lo<=hi && hi>=floor && lo<=1;
}

bool classify_contact(const Point &q,const Polygon &polygon,ContactFeature &feature) {
    std::optional<size_t> boundary_edge,vertex;
    for(size_t i=0;i<polygon.size();++i) {
        const Point edge=polygon[(i+1)%polygon.size()]-polygon[i];
        const Rational side=edge.cross(q-polygon[i]);
        if(side<0)return false;
        if(side==0&&!boundary_edge)boundary_edge=i;
        if(q==polygon[i])vertex=i;
    }
    if(vertex) {
        const size_t i=*vertex,n=polygon.size();
        if((polygon[i]-polygon[(i+n-1)%n]).cross(polygon[(i+1)%n]-polygon[i])!=0) {
            feature={ContactFeatureKind::Vertex,i};return true;
        }
    }
    if(boundary_edge)feature={ContactFeatureKind::Edge,*boundary_edge};
    else feature={ContactFeatureKind::Interior,0};
    return true;
}

bool materialize_path(const std::vector<Point> &path,const ExactPolygons &polygons,
                      bool last,std::vector<Point> &contacts,std::vector<ContactFeature> &features) {
    if(path.empty())return false;
    contacts.clear();contacts.reserve(polygons.size());features.clear();features.reserve(polygons.size());
    if(path.size()==1) {
        for(const auto &p:polygons) {
            ContactFeature feature;if(!classify_contact(path.front(),p,feature))return false;
            contacts.push_back(path.front());features.push_back(feature);
        }
        return true;
    }
    size_t segment=1;Rational rate=0;
    for(const auto &polygon:polygons) {
        bool found=false;
        while(segment<path.size()) {
            Rational lo,hi;
            if(clip(path[segment-1],path[segment],polygon,rate,lo,hi)) {
                rate=last?std::min(hi,Rational(1)):std::max(lo,rate);
                Point contact=path[segment-1]+(path[segment]-path[segment-1])*rate;
                ContactFeature feature;if(!classify_contact(contact,polygon,feature))return false;
                contacts.push_back(std::move(contact));features.push_back(feature);
                found=true;break;
            }
            ++segment;rate=0;
        }
        if(!found)return false;
    }
    return true;
}

bool materialize(const std::vector<Vector2> &raw_path,const ExactPolygons &polygons,
                 bool last,std::vector<Point> &contacts,std::vector<ContactFeature> &features) {
    std::vector<Point> path;path.reserve(raw_path.size());
    for(auto v:raw_path) {
        if(!v.is_finite())return false;
        Point p(v);if(path.empty() || !(path.back()==p))path.push_back(std::move(p));
    }
    return materialize_path(path,polygons,last,contacts,features);
}

template<class Scalar>
void append(std::vector<ConvexArithmeticPoint<Scalar>> &path,const ConvexArithmeticPoint<Scalar> &p) {
    if(path.empty()||!(path.back()==p))path.push_back(p);
}

template<class Scalar, class Polygons>
ConvexArithmeticPoint<Scalar> trace_vertex(const detail::DirectionalTraceStep &step,
        const Polygons &polygons) {
    using Point=ConvexArithmeticPoint<Scalar>;
    if(!step.vertex_is_edge_intersection)return Point(step.defining_point);
    if(step.level==0||step.level>polygons.size()||step.defining_polygon>=polygons.size())
        throw std::runtime_error("Invalid directional vertex provenance");
    const auto &current=polygons[step.level-1],&other=polygons[step.defining_polygon];
    if(step.original_edge>=current.size()||step.defining_edge>=other.size())
        throw std::runtime_error("Invalid directional edge provenance");
    const Point a=current[step.original_edge];
    const Point edge=current[(step.original_edge+1)%current.size()]-a;
    const Point c=other[step.defining_edge];
    const Point other_edge=other[(step.defining_edge+1)%other.size()]-c;
    const Scalar denominator=edge.cross(other_edge);
    if(denominator==0)throw std::runtime_error("Parallel directional vertex provenance");
    return a+edge*((c-a).cross(other_edge)/denominator);
}

template<class Scalar, class Polygons>
std::vector<ConvexArithmeticPoint<Scalar>> replay_trace(const Vector2 &start,const Vector2 &target,
        const Polygons &polygons,
        const std::vector<detail::DirectionalTraceStep> &trace,
        std::vector<std::optional<ConvexArithmeticPoint<Scalar>>> &bend_contacts) {
    using Point=ConvexArithmeticPoint<Scalar>;
    if(trace.size()!=polygons.size())throw std::runtime_error("Directional trace cardinality mismatch");
    bend_contacts.assign(polygons.size(),std::nullopt);
    std::vector<Point> path;size_t trace_index=0;
    auto replay=[&](auto &&self,const Point &q,size_t level)->void {
        if(level==0){append(path,Point(start));append(path,q);return;}
        if(trace_index>=trace.size())throw std::runtime_error("Directional trace ended early");
        const auto &step=trace[trace_index++];
        if(step.level!=level)throw std::runtime_error("Directional trace level mismatch");
        if(step.region==detail::DirectionalTraceRegion::Crossing){self(self,q,level-1);return;}
        const auto &polygon=polygons[level-1];
        if(step.original_edge>=polygon.size())throw std::runtime_error("Directional trace edge out of range");
        if(step.region==detail::DirectionalTraceRegion::Vertex) {
            const Point vertex=trace_vertex<Scalar>(step,polygons);
            bend_contacts[level-1]=vertex;
            self(self,vertex,level-1);append(path,q);return;
        }
        const Point a=polygon[step.original_edge];
        const Point edge=polygon[(step.original_edge+1)%polygon.size()]-a;
        const Point reflected=a+(edge*(Scalar(2)*(q-a).dot(edge)/edge.dot(edge))-(q-a));
        self(self,reflected,level-1);
        if(path.size()<2)throw std::runtime_error("Exact trace reflection has no incoming segment");
        const Point previous=path[path.size()-2],direction=reflected-previous;
        const Scalar denominator=edge.cross(direction);
        if(denominator==0)throw std::runtime_error("Exact trace reflection is parallel to edge");
        const Scalar t=edge.cross(a-previous)/denominator;
        const Point contact=previous+direction*t;
        const Scalar u=(contact-a).dot(edge)/edge.dot(edge);
        if(t<0||t>1||u<0||u>1)throw std::runtime_error("Exact trace refolding leaves finite edge");
        bend_contacts[level-1]=contact;
        path.pop_back();append(path,contact);append(path,q);
    };
    replay(replay,Point(target),polygons.size());
    if(trace_index!=trace.size())throw std::runtime_error("Directional trace has unused steps");
    return path;
}

std::vector<Point> replay_trace_exact(const Vector2 &start,const Vector2 &target,
        const ExactPolygons &polygons,const std::vector<detail::DirectionalTraceStep> &trace,
        std::vector<std::optional<Point>> &bend_contacts) {
    return replay_trace(start,target,polygons,trace,bend_contacts);
}

bool feature_on_edge(const Point &contact,const Polygon &polygon,size_t edge_index,
                     ContactFeature &feature) {
    if(edge_index>=polygon.size())return false;
    const size_t n=polygon.size(),next=(edge_index+1)%n;
    const Point edge=polygon[next]-polygon[edge_index];
    if(edge.cross(contact-polygon[edge_index])!=0)return false;
    const Rational u=(contact-polygon[edge_index]).dot(edge)/edge.dot(edge);
    if(u<0||u>1)return false;
    const std::optional<size_t> vertex=contact==polygon[edge_index]
        ?std::optional<size_t>(edge_index):contact==polygon[next]?std::optional<size_t>(next):std::nullopt;
    if(vertex) {
        const size_t i=*vertex;
        if((polygon[i]-polygon[(i+n-1)%n]).cross(polygon[(i+1)%n]-polygon[i])!=0) {
            feature={ContactFeatureKind::Vertex,i};return true;
        }
    }
    feature={ContactFeatureKind::Edge,edge_index};return true;
}

bool advance_to(const Point &contact,const std::vector<Point> &path,size_t &segment,Rational &rate) {
    while(segment<path.size()) {
        const Point a=path[segment-1],d=path[segment]-a;
        Rational t;
        if(d.x!=0)t=(contact.x-a.x)/d.x;
        else if(d.y!=0)t=(contact.y-a.y)/d.y;
        else {++segment;rate=0;continue;}
        if(t>=rate&&t<=1&&a+d*t==contact){rate=t;return true;}
        ++segment;rate=0;
    }
    return false;
}

int polar_half(const Point &v) {return v.y<0||(v.y==0&&v.x<0);}
bool polar_less(const Point &a,const Point &b) {
    const int ah=polar_half(a),bh=polar_half(b);
    return ah!=bh?ah<bh:a.cross(b)>0;
}
size_t edge_angle_rotation(const Polygon &polygon) {
    size_t result=0;
    for(size_t i=1;i<polygon.size();++i) {
        const Point edge=polygon[(i+1)%polygon.size()]-polygon[i];
        const Point best=polygon[(result+1)%polygon.size()]-polygon[result];
        if(polar_less(edge,best))result=i;
    }
    return result;
}
size_t support_max(const Polygon &polygon,const Point &direction,size_t rotation) {
    // Along a CCW convex boundary, edge polar angles are sorted cyclically.
    // The maximum of <direction,p> starts where the edge derivative changes
    // from positive to nonpositive: angle(direction)+pi/2.
    const Point key{-direction.y,direction.x};const size_t n=polygon.size();
    size_t left=0,right=n;
    while(left<right) {
        const size_t mid=left+(right-left)/2,index=(rotation+mid)%n;
        const Point edge=polygon[(index+1)%n]-polygon[index];
        if(polar_less(edge,key))left=mid+1;else right=mid;
    }
    return (rotation+(left==n?0:left))%n;
}

struct BoundaryHit {Point point;size_t edge=0;Rational rate;};

bool logarithmic_clip(const Point &a,const Point &b,const Polygon &polygon,size_t rotation,
        Rational floor,Rational &lo,Rational &hi,ContactFeature &lo_feature,ContactFeature &hi_feature) {
    const Point direction=b-a;if(direction.zero())return false;
    const size_t n=polygon.size();
    auto side=[&](size_t i){return direction.cross(polygon[i]-a);};
    const size_t maximum=support_max(polygon,Point{-direction.y,direction.x},rotation);
    const size_t minimum=support_max(polygon,Point{direction.y,-direction.x},rotation);
    if(side(minimum)>0||side(maximum)<0)return false;
    std::vector<BoundaryHit> hits;hits.reserve(4);
    auto add=[&](const Point &point,size_t edge) {
        Rational rate=direction.x!=0?(point.x-a.x)/direction.x:(point.y-a.y)/direction.y;
        if(a+direction*rate!=point)return;
        for(const auto &hit:hits)if(hit.point==point)return;
        hits.push_back({point,edge,rate});
    };
    const Rational minimum_side=side(minimum),maximum_side=side(maximum);
    if(minimum_side==0||maximum_side==0) {
        // The line supports the polygon.  Its intersection is a vertex or a
        // contiguous collinear boundary run; locate both ends of that run.
        const size_t tangent=minimum_side==0?minimum:maximum;
        add(polygon[tangent],tangent);
        for(int orientation:{-1,1}) {
            const auto index_at=[&](size_t offset) {
                const long long raw=static_cast<long long>(tangent)
                    +orientation*static_cast<long long>(offset);
                return static_cast<size_t>((raw%static_cast<long long>(n)+static_cast<long long>(n))
                    %static_cast<long long>(n));
            };
            if(side(index_at(1))!=0)continue;
            size_t left=1,right=n-1;
            if(side(index_at(right))==0)right=1;
            else while(left<right) {
                const size_t mid=left+(right-left+1)/2;
                if(side(index_at(mid))==0)left=mid;else right=mid-1;
            }
            const size_t endpoint=index_at(left);
            add(polygon[endpoint],orientation>0?(endpoint+n-1)%n:endpoint);
        }
    } else {
    auto cross_chain=[&](size_t start,size_t finish,bool increasing) {
        const size_t distance=(finish+n-start)%n;
        size_t left=0,right=distance;
        while(left<right) {
            const size_t mid=left+(right-left)/2,index=(start+mid)%n;
            const bool reached=increasing?side(index)>=0:side(index)<=0;
            if(reached)right=mid;else left=mid+1;
        }
        const size_t v=(start+left)%n,u=(v+n-1)%n;
        const Rational fu=side(u),fv=side(v);
        if(fu==0)add(polygon[u],u);
        if(fv==0)add(polygon[v],u);
        if(fu*fv<0) {
            const Point edge=polygon[v]-polygon[u];
            const Rational denominator=direction.cross(edge);
            if(denominator!=0)add(a+direction*((polygon[u]-a).cross(edge)/denominator),u);
        }
    };
    cross_chain(minimum,maximum,true);
    cross_chain(maximum,minimum,false);
    }
    if(hits.empty())return false;
    auto [minimum_hit,maximum_hit]=std::minmax_element(hits.begin(),hits.end(),
        [](const BoundaryHit &x,const BoundaryHit &y){return x.rate<y.rate;});
    const Rational entry=minimum_hit->rate,exit=maximum_hit->rate;
    lo=std::max(floor,entry);hi=std::min(Rational(1),exit);
    if(lo>hi||hi<floor||lo>1)return false;
    lo_feature={ContactFeatureKind::Interior,0};hi_feature={ContactFeatureKind::Interior,0};
    if(lo==entry&&!feature_on_edge(minimum_hit->point,polygon,minimum_hit->edge,lo_feature))return false;
    if(hi==exit&&!feature_on_edge(maximum_hit->point,polygon,maximum_hit->edge,hi_feature))return false;
    return true;
}

bool materialize_trace_path(const std::vector<Point> &path,const ExactPolygons &polygons,
        const std::vector<detail::DirectionalTraceStep> &trace,
        const std::vector<std::optional<Point>> &bend_contacts,bool last,
        std::vector<Point> &contacts,std::vector<ContactFeature> &features,
        const std::vector<size_t> *prepared_rotations=nullptr) {
    if(path.empty()||trace.size()!=polygons.size()||bend_contacts.size()!=polygons.size())return false;
    contacts.clear();features.clear();contacts.reserve(polygons.size());features.reserve(polygons.size());
    if(path.size()==1)return materialize_path(path,polygons,last,contacts,features);
    std::vector<const detail::DirectionalTraceStep*> by_level(polygons.size());
    for(const auto &step:trace)if(step.level&&step.level<=polygons.size())by_level[step.level-1]=&step;
    std::vector<size_t> imported_rotations;
    if(!prepared_rotations) {
        imported_rotations.reserve(polygons.size());
        for(const auto &polygon:polygons)imported_rotations.push_back(edge_angle_rotation(polygon));
    }
    const auto &rotations=prepared_rotations?*prepared_rotations:imported_rotations;
    size_t segment=1;Rational rate=0;
    for(size_t i=0;i<polygons.size();++i) {
        if(bend_contacts[i]) {
            if(!advance_to(*bend_contacts[i],path,segment,rate)||!by_level[i])return false;
            ContactFeature feature;
            if(!feature_on_edge(*bend_contacts[i],polygons[i],by_level[i]->original_edge,feature))return false;
            contacts.push_back(*bend_contacts[i]);features.push_back(feature);continue;
        }
        bool found=false;
        while(segment<path.size()) {
            Rational lo,hi;ContactFeature lo_feature,hi_feature;
            if(logarithmic_clip(path[segment-1],path[segment],polygons[i],rotations[i],rate,
                                lo,hi,lo_feature,hi_feature)) {
                rate=last?std::min(hi,Rational(1)):std::max(lo,rate);
                Point contact=path[segment-1]+(path[segment]-path[segment-1])*rate;
                const ContactFeature feature=last?hi_feature:lo_feature;
                contacts.push_back(std::move(contact));features.push_back(feature);found=true;break;
            }
            ++segment;rate=0;
        }
        if(!found)return false;
    }
    return true;
}

bool materialize(const std::vector<detail::DirectionalMapContact> &details,
                 const ExactPolygons &polygons,std::vector<Point> &contacts) {
    if(details.size()!=polygons.size())return false;
    contacts.clear();contacts.reserve(details.size());
    for(size_t i=0;i<details.size();++i) {
        const auto &detail=details[i];
        Point contact(detail.point);
        if(detail.has_edge) {
            const Point a(detail.segment_start),b(detail.segment_end);
            const Point v(detail.edge_start),w(detail.edge_end);
            const Point direction=b-a,edge=w-v;
            const Rational denominator=edge.cross(direction);
            if(denominator==0)return false;
            const Rational t=edge.cross(v-a)/denominator;
            const Rational u=(a+direction*t-v).dot(edge)/edge.dot(edge);
            if(t<0||t>1||u<0||u>1)return false;
            contact=a+direction*t;
        } else {
            const Point a(detail.segment_start),b(detail.segment_end);
            if(!(a==b)) {
                Rational lo,hi;
                if(!clip(a,b,polygons[i],0,lo,hi))return false;
                contact=a+(b-a)*lo;
            }
        }
        if(!inside(contact,polygons[i]))return false;
        contacts.push_back(std::move(contact));
    }
    return true;
}

bool segment_hits(const Point &a,const Point &b,const Polygon &p) {
    Rational lo,hi;return clip(a,b,p,0,lo,hi);
}

struct Bounds {Rational min_x,max_x,min_y,max_y;};
Bounds bounds(const Polygon &p) {
    Bounds b{p.front().x,p.front().x,p.front().y,p.front().y};
    for(const auto &v:p){b.min_x=std::min(b.min_x,v.x);b.max_x=std::max(b.max_x,v.x);
        b.min_y=std::min(b.min_y,v.y);b.max_y=std::max(b.max_y,v.y);}return b;
}
bool bounds_disjoint(const Bounds &a,const Bounds &b) {
    return a.max_x<b.min_x||b.max_x<a.min_x||a.max_y<b.min_y||b.max_y<a.min_y;
}

bool pair_disjoint(const Polygon &a,const Polygon &b,const Bounds &a_bounds,const Bounds &b_bounds,
                   ConvexHybridStats &stats) {
    if(bounds_disjoint(a_bounds,b_bounds))return true;
    ++stats.dispatch_pair_exact_checks;
    if(inside(a.front(),b) || inside(b.front(),a))return false;
    for(size_t e=0;e<a.size();++e) {
        Polygon edge_box_points{a[e],a[(e+1)%a.size()]};
        if(bounds_disjoint(bounds(edge_box_points),b_bounds))continue;
        if(segment_hits(a[e],a[(e+1)%a.size()],b))return false;
    }
    return true;
}

bool pairwise_disjoint(const ExactPolygons &polygons,ConvexHybridStats &stats) {
    std::vector<Bounds> polygon_bounds;polygon_bounds.reserve(polygons.size());
    for(const auto &p:polygons)polygon_bounds.push_back(bounds(p));
    for(size_t i=0;i<polygons.size();++i)for(size_t j=i+1;j<polygons.size();++j) {
        ++stats.dispatch_pair_queries;
        if(!pair_disjoint(polygons[i],polygons[j],polygon_bounds[i],polygon_bounds[j],stats))return false;
    }
    return true;
}

#ifdef TPP_HAS_TOUCHING_DISJOINT
// A cheap candidate-selection hint only. Roundoff or small area overlaps may
// pass this test; the original constraints still require an independent proof.
bool suggests_boundary_disjoint(const std::vector<std::vector<Vector2>> &polygons) {
    std::vector<int> winding;
    std::vector<std::array<double,4>> boxes;
    for(const auto &p:polygons) {
        int orientation=1;
        for(size_t i=0;i<p.size();++i) {
            const double turn=(p[(i+1)%p.size()]-p[i]).cross(p[(i+2)%p.size()]-p[i]);
            if(turn!=0){orientation=turn>0?1:-1;break;}
        }
        winding.push_back(orientation);
        std::array<double,4> b{p.front().x,p.front().x,p.front().y,p.front().y};
        for(auto v:p){b[0]=std::min(b[0],v.x);b[1]=std::max(b[1],v.x);
            b[2]=std::min(b[2],v.y);b[3]=std::max(b[3],v.y);}
        boxes.push_back(b);
    }
    auto separates=[](const auto &a,const auto &b,int orientation) {
        for(size_t i=0;i<a.size();++i) {
            const Vector2 edge=a[(i+1)%a.size()]-a[i];bool outside=true;
            if(edge.x==0&&edge.y==0)continue;
            for(const auto &v:b) {
                const Vector2 q=v-a[i];
                const long double x=(long double)edge.x*q.y,y=(long double)edge.y*q.x;
                const long double error=64*std::numeric_limits<double>::epsilon()*(std::abs(x)+std::abs(y));
                if(orientation*(x-y)>error){outside=false;break;}
            }
            if(outside)return true;
        }
        return false;
    };
    for(size_t i=0;i<polygons.size();++i)for(size_t j=i+1;j<polygons.size();++j) {
        const auto &a=boxes[i],&b=boxes[j];
        if(a[1]<=b[0]||b[1]<=a[0]||a[3]<=b[2]||b[3]<=a[2])continue;
        if(!separates(polygons[i],polygons[j],winding[i])&&
           !separates(polygons[j],polygons[i],winding[j]))return false;
    }
    return true;
}

std::vector<detail::DirectionalTraceStep> propose_boundary_trace(Vector2 start,Vector2 target,
        const std::vector<std::vector<Vector2>> &input,const ExactPolygons &polygons,bool contract) {
    auto proposal=input;
    if(contract) {
        proposal.clear();proposal.reserve(polygons.size());
        for(const auto &p:polygons) {
            std::vector<Vector2> q;Vector2 center{};
            for(const auto &v:p){q.push_back(v.external());center+=q.back();}
            center=center/double(q.size());
            for(auto &v:q)v=v+(center-v)*0x1p-20;
            proposal.push_back(std::move(q));
        }
    }
    auto trace=detail::solve_binary_search_disjoint_trace_unchecked(start,target,proposal);
    if(contract)for(auto &step:trace)if(step.region==detail::DirectionalTraceRegion::Vertex)
        step.defining_point=polygons.at(step.level-1).at(step.original_edge).external();
    return trace;
}
#endif

ConvexFallbackReason certify(const Vector2 &start,const Vector2 &target,
        const ExactPolygons &polygons,const std::vector<Point> &contacts,
        const std::vector<ContactFeature> &features,
        std::size_t &exact_predicates,std::size_t &zero_link_witnesses) {
    if(contacts.size()!=polygons.size()||features.size()!=polygons.size())
        return ConvexFallbackReason::ContactConstruction;
    std::vector<Point> chain;chain.reserve(contacts.size()+2);
    chain.emplace_back(start);chain.insert(chain.end(),contacts.begin(),contacts.end());chain.emplace_back(target);
    std::vector<Point> directions;directions.reserve(chain.size()-1);
    for(size_t i=1;i<chain.size();++i)directions.push_back(chain[i]-chain[i-1]);
    auto kkt=[&](size_t polygon_index,const Point &incoming,const Point &outgoing) {
        const Rational a2=incoming.dot(incoming),b2=outgoing.dot(outgoing);
        if(a2==0&&b2==0)return true;
        if(a2==0||b2==0)return false;
        if(incoming.cross(outgoing)==0&&incoming.dot(outgoing)>0)return true;
        const auto check=[&](const Point &feasible) {
            ++exact_predicates;
            return detail::convex_normalized_difference_sign(incoming.dot(feasible),a2,
                                               outgoing.dot(feasible),b2)>=0;
        };
        const auto &polygon=polygons[polygon_index];const auto &feature=features[polygon_index];
        if(feature.kind==ContactFeatureKind::Interior)return false;
        if(feature.kind==ContactFeatureKind::Vertex) {
            const size_t i=feature.index,n=polygon.size();
            return check(polygon[(i+n-1)%n]-contacts[polygon_index])
                &&check(polygon[(i+1)%n]-contacts[polygon_index]);
        }
        const size_t i=feature.index,n=polygon.size();
        const Point tangent=polygon[(i+1)%n]-polygon[i];
        ++exact_predicates;
        if(detail::convex_normalized_difference_sign(incoming.dot(tangent),a2,
                                      outgoing.dot(tangent),b2)!=0)return false;
        for(const Point &vertex:polygon) {
            if(tangent.cross(vertex-polygon[i])==0)continue;
            return check(vertex-contacts[polygon_index]);
        }
        return false;
    };
    std::vector<bool> block_certified(contacts.size(),false);
    for(size_t first=0;first<directions.size();) {
        if(!directions[first].zero()){++first;continue;}
        size_t last=first;while(last+1<directions.size()&&directions[last+1].zero())++last;
        const Point *before=first?&directions[first-1]:nullptr;
        const Point *after=last+1<directions.size()?&directions[last+1]:nullptr;
        if(before&&after&&(before->cross(*after)!=0||before->dot(*after)<=0)) {
            const size_t first_polygon=first-1;
            const size_t last_polygon=std::min(last,polygons.size()-1);
            // Zero links admit the entire unit disk. Propagate the exact
            // reachable subgradients through every contact in the block,
            // including both boundary contacts next to nonzero links.
            // A constant incoming (outgoing) subgradient on all zero links
            // moves the whole turn to the last (first) contact. The remaining
            // contacts have zero support difference. These sufficient exact
            // witnesses are particularly cheap on shared boundary edges.
            if(!kkt(last_polygon,*before,*after)&&!kkt(first_polygon,*before,*after)) {
                detail::ConvexDualReachability reachable(*before,exact_predicates);
                for(size_t i=first_polygon;i<=last_polygon&&!reachable.reaches(*after);++i)
                    reachable.advance(polygons[i],contacts[i]);
                if(!reachable.reaches(*after))return ConvexFallbackReason::CoincidentContact;
            }
            for(size_t i=first_polygon;i<=last_polygon;++i)block_certified[i]=true;
        } else {
            const Point witness=before?*before:after?*after:Point{};
            for(size_t j=first;j<=last;++j)directions[j]=witness;
        }
        ++zero_link_witnesses;first=last+1;
    }
    for(size_t i=0;i<contacts.size();++i) {
        if(block_certified[i])continue;
        const Point incoming=directions[i],outgoing=directions[i+1];
        if(!kkt(i,incoming,outgoing))return ConvexFallbackReason::LocalOptimality;
    }
    return ConvexFallbackReason::None;
}

double contact_length(Vector2 start,Vector2 target,const std::vector<Vector2> &contacts) {
    long double value=0;Vector2 previous=start;
    for(auto p:contacts){value+=std::hypot((long double)p.x-previous.x,(long double)p.y-previous.y);previous=p;}
    value+=std::hypot((long double)target.x-previous.x,(long double)target.y-previous.y);
    return double(value);
}

double rational_lower(const Rational &q) {
    double d=q.convert_to<double>();
    while(Rational(d)>q)d=std::nextafter(d,-std::numeric_limits<double>::infinity());
    return d;
}
double rational_upper(const Rational &q) {
    double d=q.convert_to<double>();
    while(Rational(d)<q)d=std::nextafter(d,std::numeric_limits<double>::infinity());
    return d;
}

void set_exact_bounds(ConvexHybridResult &result,Vector2 start,Vector2 target,
                      const std::vector<Point> &contacts,ConvexHybridStats *stats=nullptr) {
    PhaseTimer timer{stats?stats->bound_evaluation_seconds:result.stats.bound_evaluation_seconds};
    constexpr unsigned precision=96;
    const tpp::ConvexInteger scale=tpp::ConvexInteger(1)<<precision;
    std::vector<Point> chain;chain.reserve(contacts.size()+2);chain.emplace_back(start);
    chain.insert(chain.end(),contacts.begin(),contacts.end());chain.emplace_back(target);
    Rational lower=0,upper=0;
    for(size_t i=1;i<chain.size();++i) {
        const Point d=chain[i]-chain[i-1];
        const Rational squared=d.dot(d);if(squared==0)continue;
        const auto numerator=boost::multiprecision::numerator(squared);
        const auto denominator=boost::multiprecision::denominator(squared);
        const tpp::ConvexInteger scaled=(numerator<<(2*precision))/denominator;
        const tpp::ConvexInteger root=sqrt(scaled);
        lower+=Rational(root)/Rational(scale);upper+=Rational(root+1)/Rational(scale);
    }
    result.lower_bound=rational_lower(lower);result.upper_bound=rational_upper(upper);
}

Rational sqrt_upper(const Rational &squared) {
    if(squared==0)return 0;
    constexpr unsigned precision=96;
    const tpp::ConvexInteger scale=tpp::ConvexInteger(1)<<precision;
    const auto numerator=boost::multiprecision::numerator(squared);
    const auto denominator=boost::multiprecision::denominator(squared);
    const tpp::ConvexInteger scaled=(numerator<<(2*precision))/denominator;
    return Rational(sqrt(scaled)+1)/Rational(scale);
}

Point feasible_unit_direction(const Point &direction) {
    const Rational norm=sqrt_upper(direction.dot(direction));
    return norm==0?Point{}:direction*(Rational(1)/norm);
}

double candidate_dual_lower(Vector2 start,Vector2 target,
        const ExactPolygons &polygons,const std::vector<Point> &contacts) {
    std::vector<Point> chain;chain.reserve(contacts.size()+2);chain.emplace_back(start);
    chain.insert(chain.end(),contacts.begin(),contacts.end());chain.emplace_back(target);
    std::vector<Point> base;std::vector<bool> zero;
    base.reserve(chain.size()-1);zero.reserve(chain.size()-1);
    for(size_t i=1;i<chain.size();++i) {
        const Point difference=chain[i]-chain[i-1];
        zero.push_back(difference.zero());base.push_back(feasible_unit_direction(difference));
    }
    const Point origin(start),destination(target);
    auto evaluate=[&](int zero_policy) {
        auto directions=base;
        for(size_t i=0;i<directions.size();++i)if(zero[i]) {
            if(zero_policy==0)continue;
            if(zero_policy==1) {
                size_t j=i;while(j>0&&zero[j])--j;
                if(!zero[j]){directions[i]=base[j];continue;}
                j=i;while(j+1<base.size()&&zero[j])++j;
                if(!zero[j])directions[i]=base[j];
            } else if(zero_policy==2) {
                size_t j=i;while(j+1<base.size()&&zero[j])++j;
                if(!zero[j]){directions[i]=base[j];continue;}
                j=i;while(j>0&&zero[j])--j;
                if(!zero[j])directions[i]=base[j];
            } else directions[i]=feasible_unit_direction(destination-origin);
        }
        Rational bound=(destination-origin).dot(directions.back());
        for(size_t i=0;i<polygons.size();++i) {
            const Point coefficient=directions[i]-directions[i+1];
            Rational support=(polygons[i].front()-origin).dot(coefficient);
            for(size_t j=1;j<polygons[i].size();++j)
                support=std::min(support,(polygons[i][j]-origin).dot(coefficient));
            bound+=support;
        }
        return bound;
    };
    Rational best=evaluate(0);
    for(int policy=1;policy<4;++policy)best=std::max(best,evaluate(policy));
    const Point direct=destination-origin;
    const Rational direct_lower=direct.dot(direct)==0?Rational(0):
        direct.dot(direct)/sqrt_upper(direct.dot(direct));
    best=std::max(best,direct_lower);
    return rational_lower(best);
}

#ifdef TPP_HAS_INTERVAL_PRIMAL_DUAL
using Interval=detail::CycleInterval;
using IntervalPoint=detail::IntervalPoint;

// Directions are exact differences of binary inputs divided by a certified
// upper norm. Their interval enclosures describe one dual-feasible vector;
// independently selecting interval endpoints would not have this property.
std::optional<double> interval_dual_lower(const std::vector<Vector2> &chain,
        const std::vector<std::vector<Vector2>> &polygons,double short_link,bool interpolate) {
    std::vector<IntervalPoint> base;
    std::vector<bool> short_links;
    for(size_t i=1;i<chain.size();++i) {
        const auto d=IntervalPoint(chain[i])-IntervalPoint(chain[i-1]);
        const auto norm=(d.x.square()+d.y.square()).sqrt();
        if(!norm.finite())return {};
        const bool equal=chain[i]==chain[i-1];
        short_links.push_back(equal||norm.hi<=short_link);
        if(equal||norm.hi==0)base.emplace_back();
        else base.emplace_back(d.x.divided_by(norm.hi),d.y.divided_by(norm.hi));
    }
    const IntervalPoint origin(chain.front()),destination(chain.back());
    const auto direct=destination-origin;
    const auto direct_norm=(direct.x.square()+direct.y.square()).sqrt();
    if(!direct_norm.finite())return {};
    const IntervalPoint direct_unit=direct_norm.hi==0?IntervalPoint{}:
        IntervalPoint(direct.x.divided_by(direct_norm.hi),direct.y.divided_by(direct_norm.hi));
    double best=std::max(0.0,direct_norm.lo);
    // Without short links all four old policies are identical, including
    // their rounding. Evaluate the common dual just once.
    const bool any_short=std::any_of(short_links.begin(),short_links.end(),[](bool x){return x;});
    const int policies=any_short?(interpolate?5:4):1;
    for(int policy=0;policy<policies;++policy) {
        auto directions=base;
        if(policy==4) {
            for(size_t i=0;i<base.size();) {
                if(!short_links[i]){++i;continue;}
                size_t end=i;while(end<base.size()&&short_links[end])++end;
                const auto left=i?base[i-1]:direct_unit,right=end<base.size()?base[end]:direct_unit;
                for(size_t j=i;j<end;++j) {
                    const Interval weight(double(j-i+1)/double(end-i+1));
                    const auto other=Interval(1)-weight;
                    // A convex combination of two feasible unit-disk vectors
                    // stays feasible. Intervals enclose that combination;
                    // near-coincidence only selects a proposal, never a bound.
                    directions[j]={other*left.x+weight*right.x,other*left.y+weight*right.y};
                }
                i=end;
            }
        } else if(policy)for(size_t i=0;i<base.size();++i)if(short_links[i]) {
            if(policy==3){directions[i]=direct_unit;continue;}
            std::optional<size_t> left,right;
            for(size_t j=i;j>0;)if(!short_links[--j]){left=j;break;}
            for(size_t j=i+1;j<base.size();++j)if(!short_links[j]){right=j;break;}
            const auto selected=policy==1?(left?left:right):(right?right:left);
            directions[i]=selected?base[*selected]:direct_unit;
        }
        Interval dual=direct.dot(directions.back());
        for(size_t i=0;i<polygons.size();++i) {
            const auto normal=directions[i]-directions[i+1];
            Interval support(INFINITY);
            for(const auto &v:polygons[i]) {
                const auto term=normal.dot(IntervalPoint(v)-origin);
                support.lo=std::min(support.lo,term.lo);
                support.hi=std::min(support.hi,term.hi);
            }
            dual=dual+support;
        }
        if(dual.finite())best=std::max(best,dual.lo);
    }
    return best;
}

bool interval_bounds_enabled(const ConvexHybridOptions &options) {
    return options.mode==ConvexHybridMode::SafeCertified&&!options.shadow_rational&&
        ((std::isfinite(options.max_gap)&&options.max_gap>0)||std::isfinite(options.cutoff))&&
        detail::cycle_interval_environment();
}

// This optional proof works entirely on the original constraints. Neither the
// trace nor its geometry is trusted. A failed interval proof retains exact
// replay and the complete original support certificate.
#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
bool try_interval_trace_bound(Vector2 start,Vector2 target,const ExactPolygons &exact,
        const std::vector<detail::DirectionalTraceStep> &trace,const ConvexHybridOptions &options,
        ConvexHybridResult &result,const std::vector<std::vector<Vector2>> *prepared_binary=nullptr) {
    result.stats.interval_bounds_attempted=true;
    try {
        std::vector<std::vector<Vector2>> imported_binary;
        std::vector<std::vector<ConvexArithmeticPoint<double>>> arithmetic;
        {
            PhaseTimer timer{result.stats.proposal_preparation_seconds};
            if(prepared_binary) {
                arithmetic.reserve(prepared_binary->size());
                for(const auto &polygon:*prepared_binary) {
                    arithmetic.emplace_back();arithmetic.back().reserve(polygon.size());
                    for(const auto &v:polygon)arithmetic.back().emplace_back(v);
                }
            } else for(const auto &polygon:exact) {
                imported_binary.emplace_back();arithmetic.emplace_back();
                for(const auto &q:polygon) {
                    imported_binary.back().push_back(q.external());arithmetic.back().emplace_back(q.external());
                }
            }
        }
        const auto &binary=prepared_binary?*prepared_binary:imported_binary;
        std::vector<Vector2> chain;
        {
            PhaseTimer timer{result.stats.contact_materialization_seconds};
            std::vector<std::optional<ConvexArithmeticPoint<double>>> bends;
            const auto path=replay_trace(start,target,arithmetic,trace,bends);
            std::vector<Vector2> raw;
            for(const auto &q:path) {const auto v=q.external();if(!v.is_finite())return false;raw.push_back(v);}
            if(!certified_detail::repair_contacts(raw,binary,chain))return false;
        }
        PhaseTimer timer{result.stats.certificate_seconds};
        const auto seed=chain;
        for(size_t i=0;i<binary.size();++i) {
            auto &q=chain[i+1];
            if(detail::interval_convex_contains(q,binary[i],exact[i],result.stats.predicate_exact_evaluations))continue;
            Vector2 center{};
            for(const auto &v:binary[i])center+=v/double(binary[i].size());
            bool repaired=false;
            for(double fraction:{0x1p-45,0x1p-40,0x1p-30}) {
                const auto candidate=q+(center-q)*fraction;
                if(detail::interval_convex_contains(candidate,binary[i],exact[i],result.stats.predicate_exact_evaluations)) {
                    q=candidate;repaired=true;break;
                }
            }
            if(!repaired)return false;
        }
        Interval length;
        for(size_t i=1;i<chain.size();++i) {
            if(chain[i]==chain[i-1])continue;
            const auto d=IntervalPoint(chain[i])-IntervalPoint(chain[i-1]);
            length=length+(d.x.square()+d.y.square()).sqrt();
        }
        if(!length.finite())return false;
        // A short-link policy only proposes unit-disk dual vectors; it is
        // never a test for geometric coincidence or a feasibility tolerance.
        double proposal_scale=0;
        for(const auto &v:seed)proposal_scale=std::max({proposal_scale,std::abs(v.x),std::abs(v.y)});
        const double short_link=std::max(32*std::numeric_limits<double>::epsilon()*proposal_scale,
            std::isfinite(options.max_gap)&&options.max_gap>0?options.max_gap/(16*double(chain.size())):0);
        const auto lower=interval_dual_lower(seed,binary,short_link,options.interpolated_zero_dual);
        if(!lower||*lower>length.hi)return false;
        const auto gap=Interval(length.hi)-Interval(*lower);
        const bool cutoff=*lower>=options.cutoff;
        if(!cutoff&&!(std::isfinite(options.max_gap)&&options.max_gap>0&&gap.hi<=options.max_gap))return false;
        result.contacts.assign(chain.begin()+1,chain.end()-1);
        result.lower_bound=*lower;result.upper_bound=length.hi;
        result.cutoff_pruned=cutoff;
        result.stats.interval_bounds_certified=true;
        result.fallback_reason=ConvexFallbackReason::None;
        result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
        return true;
    } catch(const std::exception &) {return false;}
}

#ifdef TPP_HAS_TOUCHING_DISJOINT
#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
bool try_interval_boundary_bound(Vector2 start,Vector2 target,
        const std::vector<std::vector<Vector2>> &input,const ExactPolygons &polygons,
        const ConvexHybridOptions &options,ConvexHybridResult &result,
        const std::vector<std::vector<Vector2>> *prepared_binary=nullptr) {
    {
        PhaseTimer timer{result.stats.dispatch_seconds};
        if(!suggests_boundary_disjoint(input))return false;
    }
    try {
        const auto trace=[&] {
            PhaseTimer timer{result.stats.double_solver_seconds};
            return propose_boundary_trace(start,target,input,polygons,true);
        }();
        if(!try_interval_trace_bound(start,target,polygons,trace,options,result,prepared_binary))return false;
        result.stats.interval_bounds_contracted=true;
        return true;
    } catch(const std::exception &) {return false;}
}
#endif
#endif

#ifdef TPP_HAS_TOUCHING_DISJOINT
// Keep the optional recovery out of the ordinary double/cutoff stack frame.
#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
bool recover_touching_disjoint(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,const ExactPolygons &polygons,
        ConvexHybridResult &result,std::vector<Point> &exact_contacts,
        const std::vector<size_t> *prepared_rotations) {
    const bool suggested=[&] {
        PhaseTimer timer{result.stats.dispatch_seconds};return suggests_boundary_disjoint(input);
    }();
    if(!suggested)return false;
    result.stats.touching_disjoint_attempted=true;
    // Contraction only proposes a combinatorial trace, replayed on the
    // unperturbed polygons. If it fails, try the original closed geometry.
    for(int attempt=1;attempt>=0&&!result.stats.touching_disjoint_certified;--attempt) {
        try {
            auto candidate=[&] {
                PhaseTimer timer{result.stats.double_solver_seconds};
                return propose_boundary_trace(start,target,input,polygons,attempt!=0);
            }();
            std::vector<Point> contacts;std::vector<ContactFeature> features;
            std::vector<std::optional<Point>> bends;
            const bool valid=[&] {
                PhaseTimer timer{result.stats.contact_materialization_seconds};
                const auto path=replay_trace_exact(start,target,polygons,candidate,bends);
                return materialize_trace_path(path,polygons,candidate,bends,true,contacts,features,prepared_rotations);
            }();
            if(!valid)continue;
            const auto certificate_began=Clock::now();
            const auto reason=certify(start,target,polygons,contacts,features,
                result.stats.predicate_exact_evaluations,result.stats.zero_link_witnesses);
            result.stats.certificate_seconds+=elapsed(certificate_began);
            if(reason==ConvexFallbackReason::None) {
                exact_contacts=std::move(contacts);
                result.contacts.clear();for(const auto &q:exact_contacts)result.contacts.push_back(q.external());
                result.fallback_reason=ConvexFallbackReason::None;
                result.stats.touching_disjoint_certified=result.stats.double_certified=true;
                result.stats.touching_disjoint_perturbed=attempt!=0;
            }
        } catch(const std::exception &) {
            // A perturbation is never an optimization bound. Failed
            // proposals retain the ordinary intersection solver below.
        }
    }
    return result.stats.touching_disjoint_certified;
}
#endif

void set_bounds(ConvexHybridResult &result,Vector2 start,Vector2 target) {
    PhaseTimer timer{result.stats.bound_evaluation_seconds};
    const double value=contact_length(start,target,result.contacts);
    result.lower_bound=std::nextafter(value,-std::numeric_limits<double>::infinity());
    result.upper_bound=std::nextafter(value,std::numeric_limits<double>::infinity());
}
void set_value_bounds(ConvexHybridResult &result,double value) {
    PhaseTimer timer{result.stats.bound_evaluation_seconds};
    result.lower_bound=std::nextafter(value,-std::numeric_limits<double>::infinity());
    result.upper_bound=std::nextafter(value,std::numeric_limits<double>::infinity());
}
}

struct ConvexHybridCache {
    struct Entry {
        std::vector<Vector2> input;
        Polygon exact;
        std::vector<Vector2> binary;
        Bounds box;
        size_t edge_rotation;
        std::uint64_t id;
    };
    static constexpr size_t max_vertices=8192; // Bound retained exact geometry.
    static constexpr size_t max_pairs=65536;
    using Pair=std::pair<std::uint64_t,std::uint64_t>;
    struct PairHash {
        size_t operator()(const Pair &pair) const {
            const auto first=std::hash<std::uint64_t>{}(pair.first);
            const auto second=std::hash<std::uint64_t>{}(pair.second);
            return first^(second+0x9e3779b97f4a7c15ULL+(first<<6)+(first>>2));
        }
    };
    using Prepared=std::shared_ptr<const Entry>;
    std::unordered_map<std::uint64_t,std::vector<Prepared>> entries;
    std::unordered_map<Pair,bool,PairHash> pairs;
    size_t vertices=0;
    std::uint64_t next_id=0;

    static std::uint64_t key(const std::vector<Vector2> &input) {
        std::uint64_t hash=14695981039346656037ULL;
        auto mix=[&](std::uint64_t value){hash^=value;hash*=1099511628211ULL;};
        mix(input.size());
        for(const auto &v:input) {
            mix(std::bit_cast<std::uint64_t>(v.x));
            mix(std::bit_cast<std::uint64_t>(v.y));
        }
        return hash;
    }
    static bool same_input(const std::vector<Vector2> &a,const std::vector<Vector2> &b) {
        if(a.size()!=b.size())return false;
        for(size_t i=0;i<a.size();++i)
            if(std::bit_cast<std::uint64_t>(a[i].x)!=std::bit_cast<std::uint64_t>(b[i].x) ||
               std::bit_cast<std::uint64_t>(a[i].y)!=std::bit_cast<std::uint64_t>(b[i].y))return false;
        return true;
    }
    Prepared get(const std::vector<Vector2> &input) {
        const auto hash=key(input);
        if(const auto it=entries.find(hash);it!=entries.end())
            for(const auto &entry:it->second)
                if(same_input(input,entry->input))return entry;
        Polygon exact=exact_polygon(input);
        if(next_id==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Hybrid geometry cache identity overflow");
        std::vector<Vector2> binary;binary.reserve(exact.size());
        for(const auto &q:exact)binary.push_back(q.external());
        auto box=bounds(exact);
        const size_t rotation=edge_angle_rotation(exact);
        auto entry=std::make_shared<Entry>(Entry{input,std::move(exact),std::move(binary),std::move(box),rotation,next_id++});
        if(input.size()<=max_vertices) {
            if(vertices+input.size()>max_vertices){pairs.clear();entries.clear();vertices=0;}
            entries[hash].push_back(entry);
            vertices+=input.size();
        }
        return entry;
    }
    bool disjoint(const std::vector<Prepared> &selected,ConvexHybridStats &stats) {
        for(size_t i=0;i<selected.size();++i)for(size_t j=i+1;j<selected.size();++j) {
            const auto &a=*selected[i],&b=*selected[j];
            const Pair key=std::minmax(a.id,b.id);
            ++stats.dispatch_pair_queries;
            bool disjoint;
            if(const auto found=pairs.find(key);found!=pairs.end()) {
                ++stats.dispatch_pair_cache_hits;disjoint=found->second;
            } else {
                disjoint=pair_disjoint(a.exact,b.exact,a.box,b.box,stats);
                if(pairs.size()>=max_pairs)pairs.clear();
                pairs.emplace(key,disjoint);
            }
            if(!disjoint)return false;
        }
        return true;
    }
};

static ExactPolygons cached_exact_polygons(
        const std::vector<std::vector<Vector2>> &input,DynamicConvexTppWorkspace &workspace,
        ConvexHybridStats &stats,std::vector<std::vector<Vector2>> &binary,
        std::vector<ConvexHybridCache::Prepared> &selected,std::vector<Polygon> &owned) {
    // Copies of a workspace keep independent mutable caches.
    if(!workspace.hybrid_cache || workspace.hybrid_cache.use_count()!=1)
        workspace.hybrid_cache=std::make_shared<ConvexHybridCache>();
    ExactPolygons result;result.reserve(input.size());
    selected.reserve(input.size());
    // Prepared handles survive an eviction during this call. Pair identities
    // are never recycled, including when a polygon is too large to retain.
    for(const auto &polygon:input)selected.push_back(workspace.hybrid_cache->get(polygon));
    if(workspace.cache_interval_geometry)binary.reserve(input.size());
    if(!workspace.borrow_hybrid_geometry) {
        owned.reserve(selected.size());
        for(const auto &entry:selected)owned.push_back(entry->exact);
    }
    size_t index=0;
    for(const auto &entry:selected) {
        result.push_back(workspace.borrow_hybrid_geometry?entry->exact:owned[index++]);
        if(workspace.cache_interval_geometry)binary.push_back(entry->binary);
    }
    stats.disjoint=workspace.cache_disjoint_dispatch
        ?workspace.hybrid_cache->disjoint(selected,stats):pairwise_disjoint(result,stats);
    return result;
}

const char *to_string(ConvexFallbackReason reason) {
    switch(reason) {
        case ConvexFallbackReason::None:return "none";
        case ConvexFallbackReason::LocatorOrRefoldingException:return "locator_or_refolding_exception";
        case ConvexFallbackReason::Nonfinite:return "nonfinite";
        case ConvexFallbackReason::ContactConstruction:return "contact_construction";
        case ConvexFallbackReason::MembershipOrOrdering:return "membership_or_ordering";
        case ConvexFallbackReason::LocalOptimality:return "local_optimality";
        case ConvexFallbackReason::CoincidentContact:return "coincident_contact";
        case ConvexFallbackReason::ShadowMismatch:return "shadow_mismatch";
    }
    return "unknown";
}

std::vector<Vector2> reconstruct_convex_polyline(const Vector2 &start,const Vector2 &target,
        const std::vector<Vector2> &contacts,bool compact) {
    std::vector<Vector2> result;result.reserve(contacts.size()+2);result.push_back(start);
    result.insert(result.end(),contacts.begin(),contacts.end());result.push_back(target);
    if(compact)remove_collinear_points_inplace(result);
    return result;
}

static ConvexHybridResult solve_hybrid_impl(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,const ConvexHybridOptions &options,
        DynamicConvexTppWorkspace *workspace) {
    const auto began=Clock::now();ConvexHybridResult result;
    AggregateRecorder recorder{&result};
    const auto dispatch_began=Clock::now();
    if(options.mode==ConvexHybridMode::Unchecked&&!options.materialize_contacts) {
        result.stats.disjoint=detail::pairwise_disjoint_unchecked_double(input);
        result.stats.dispatch_seconds=elapsed(dispatch_began);
        if(input.empty()){set_bounds(result,start,target);result.stats.total_seconds=elapsed(began);return result;}
        result.stats.double_attempted=true;
        const auto solve_began=Clock::now();
        const double value=result.stats.disjoint
            ?tpp_convex_solve_length_binary_search_disjoint(start,target,input)
            :detail::length_intersecting_maps_unchecked_double(start,target,input);
        result.stats.double_solver_seconds=elapsed(solve_began);
        if(!std::isfinite(value))throw std::runtime_error("Unchecked convex length is nonfinite");
        result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
        set_value_bounds(result,value);result.stats.total_seconds=elapsed(began);return result;
    }
    std::vector<std::vector<Vector2>> binary;
    std::vector<ConvexHybridCache::Prepared> selected;
    std::vector<Polygon> owned;
    if(!workspace)owned=exact_polygons(input);
    const ExactPolygons polygons=workspace?cached_exact_polygons(input,*workspace,result.stats,binary,selected,owned):ExactPolygons(owned);
    const auto *prepared_binary=workspace&&workspace->cache_interval_geometry?&binary:nullptr;
    // Gather the cached indices only when rational materialization is needed.
    // The common interval return does not allocate another per-call vector.
    std::vector<size_t> rotations;
    auto contact_rotations=[&]() -> const std::vector<size_t>* {
        if(!workspace)return nullptr;
        if(rotations.size()!=selected.size()) {
            rotations.reserve(selected.size());
            for(const auto &entry:selected)rotations.push_back(entry->edge_rotation);
        }
        return &rotations;
    };
    if(!workspace)result.stats.disjoint=pairwise_disjoint(polygons,result.stats);
    result.stats.dispatch_seconds=elapsed(dispatch_began);
    if(input.empty()) {set_bounds(result,start,target);result.stats.total_seconds=elapsed(began);return result;}
    std::vector<detail::DirectionalTraceStep> trace;
    std::vector<Point> exact_contacts;std::vector<ContactFeature> contact_features;
    auto candidate_dual=[&] {
        PhaseTimer timer{result.stats.bound_evaluation_seconds};
        return candidate_dual_lower(start,target,polygons,exact_contacts);
    };
    result.stats.double_attempted=true;
    try {
        const auto solve_began=Clock::now();
        if(result.stats.disjoint)trace=detail::solve_binary_search_disjoint_trace_unchecked(start,target,input);
        else trace=detail::solve_intersecting_map_trace_unchecked_double(start,target,input);
        result.stats.double_solver_seconds=elapsed(solve_began);
#ifdef TPP_HAS_INTERVAL_PRIMAL_DUAL
        if(interval_bounds_enabled(options)) {
            bool accepted=try_interval_trace_bound(start,target,polygons,trace,options,result,prepared_binary);
#ifdef TPP_HAS_TOUCHING_DISJOINT
            if(!accepted&&!result.stats.disjoint)
                accepted=try_interval_boundary_bound(start,target,input,polygons,options,result,prepared_binary);
#endif
            if(accepted){result.stats.total_seconds=elapsed(began);return result;}
        }
#endif
        const auto contact_began=Clock::now();
        const bool finite=std::ranges::all_of(trace,[](const auto &step){return step.defining_point.is_finite();});
        if(!finite)result.fallback_reason=ConvexFallbackReason::Nonfinite;
        else {
            std::vector<std::optional<Point>> bend_contacts;
            const auto exact_path=replay_trace_exact(start,target,polygons,trace,bend_contacts);
            const bool contacts_valid=materialize_trace_path(exact_path,polygons,trace,bend_contacts,
                result.stats.disjoint,exact_contacts,contact_features,contact_rotations());
            if(!contacts_valid)result.fallback_reason=ConvexFallbackReason::ContactConstruction;
        }
        result.stats.contact_materialization_seconds+=elapsed(contact_began);
    } catch(const std::exception &) {
        result.fallback_reason=ConvexFallbackReason::LocatorOrRefoldingException;
    }
    if(options.mode==ConvexHybridMode::Unchecked&&result.fallback_reason!=ConvexFallbackReason::None)
        throw std::runtime_error(std::string("Unchecked convex solve failed: ")+to_string(result.fallback_reason));
    if(result.fallback_reason==ConvexFallbackReason::None) {
        result.contacts.reserve(exact_contacts.size());for(const auto &p:exact_contacts)result.contacts.push_back(p.external());
        if(options.mode==ConvexHybridMode::Unchecked) {
            result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
            set_bounds(result,start,target);result.stats.total_seconds=elapsed(began);return result;
        }
        if(workspace&&workspace->bound_before_optimality&&std::isfinite(options.cutoff)
                &&options.mode==ConvexHybridMode::SafeCertified&&!options.shadow_rational) {
            const auto path=reconstruct_convex_polyline(start,target,result.contacts,false);
            const double rough=[&] {
                PhaseTimer timer{result.stats.bound_evaluation_seconds};
                return certified_detail::dual_bound(path,input);
            }();
            if(rough>=options.cutoff-1e-7*std::max(1.0,std::abs(options.cutoff))) {
                const double dual=candidate_dual();
                if(dual>=options.cutoff) {
                    set_exact_bounds(result,start,target,exact_contacts);
                    result.lower_bound=dual;result.cutoff_pruned=true;
                    result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
                    result.stats.total_seconds=elapsed(began);return result;
                }
            }
        }
        const auto certificate_began=Clock::now();
        result.fallback_reason=certify(start,target,polygons,exact_contacts,contact_features,
            result.stats.predicate_exact_evaluations,result.stats.zero_link_witnesses);
        result.stats.certificate_seconds+=elapsed(certificate_began);
        result.stats.double_certified=result.fallback_reason==ConvexFallbackReason::None;
    }
    if(options.retain_rejected_double_candidate && !result.stats.double_certified &&
       result.contacts.size()==input.size()) {
        result.rejected_double_contacts=result.contacts;
        result.rejected_double_exact_feasible=true;
        for(size_t i=0;i<result.rejected_double_contacts.size();++i)
            result.rejected_double_exact_feasible&=inside(
                Point(result.rejected_double_contacts[i]),polygons[i]);
        ConvexHybridResult candidate_bounds;
        set_exact_bounds(candidate_bounds,start,target,exact_contacts,&result.stats);
        result.rejected_double_lower_bound=candidate_dual();
        result.rejected_double_upper_bound=candidate_bounds.upper_bound;
    }
    if(result.stats.double_certified && !options.shadow_rational) {
        result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
        set_exact_bounds(result,start,target,exact_contacts);result.stats.total_seconds=elapsed(began);return result;
    }
    if(options.mode==ConvexHybridMode::SafeCertified && !options.shadow_rational
       && std::isfinite(options.cutoff)
       && exact_contacts.size()==polygons.size()) {
        const auto candidate_path=reconstruct_convex_polyline(start,target,result.contacts,false);
        const double rough=[&] {
            PhaseTimer timer{result.stats.bound_evaluation_seconds};
            return certified_detail::dual_bound(candidate_path,input);
        }();
        if(rough>=options.cutoff-1e-7*std::max(1.0,std::abs(options.cutoff))) {
            const double dual=candidate_dual();
            if(dual>=options.cutoff) {
                ConvexHybridResult candidate;
                set_exact_bounds(candidate,start,target,exact_contacts,&result.stats);
                result.lower_bound=dual;
                result.upper_bound=candidate.upper_bound;
                result.cutoff_pruned=true;
                result.fallback_reason=ConvexFallbackReason::None;
                result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
                result.stats.total_seconds=elapsed(began);
                return result;
            }
        }
    }
#ifdef TPP_HAS_TOUCHING_DISJOINT
    if(options.mode==ConvexHybridMode::SafeCertified&&!result.stats.double_certified&&!result.stats.disjoint&&
       recover_touching_disjoint(start,target,input,polygons,result,exact_contacts,contact_rotations())&&!options.shadow_rational) {
        result.backend=ConvexHybridBackend::DoubleDisjoint;
        set_exact_bounds(result,start,target,exact_contacts);
        result.stats.total_seconds=elapsed(began);return result;
    }
#endif
#ifdef TPP_HAS_FILTERED_DIRECTIONAL
    // Preserve the inexpensive double/cutoff paths. Only a rejected
    // intersecting proposal pays for the same map's filtered arithmetic.
    if(!result.stats.double_certified&&!result.stats.disjoint&&options.mode==ConvexHybridMode::SafeCertified) {
        result.stats.filtered_attempted=true;
        try {
            const auto solve_began=Clock::now();
            const auto filtered=detail::solve_intersecting_map_trace_filtered(start,target,input);
            result.stats.double_solver_seconds+=elapsed(solve_began);
            const auto contact_began=Clock::now();
            std::vector<Point> contacts;
            std::vector<ContactFeature> features;
            std::vector<std::optional<Point>> bends;
            const auto path=replay_trace_exact(start,target,polygons,filtered,bends);
            const bool valid=materialize_trace_path(path,polygons,filtered,bends,false,contacts,features,contact_rotations());
            result.stats.contact_materialization_seconds+=elapsed(contact_began);
            if(valid) {
                const auto certificate_began=Clock::now();
                const auto reason=certify(start,target,polygons,contacts,features,
                    result.stats.predicate_exact_evaluations,result.stats.zero_link_witnesses);
                result.stats.certificate_seconds+=elapsed(certificate_began);
                if(reason==ConvexFallbackReason::None) {
                    exact_contacts=std::move(contacts);
                    result.contacts.clear();
                    for(const auto &p:exact_contacts)result.contacts.push_back(p.external());
                    result.stats.filtered_certified=result.stats.double_certified=true;
                    result.fallback_reason=ConvexFallbackReason::None;
                }
            }
        } catch(const std::exception &) {
            // Failed arithmetic or reconstruction retains the original
            // rejected candidate and the ordinary complete rational recovery.
        }
        if(result.stats.filtered_certified&&!options.shadow_rational) {
            result.backend=ConvexHybridBackend::DoubleIntersection;
            set_exact_bounds(result,start,target,exact_contacts);
            result.stats.total_seconds=elapsed(began);return result;
        }
    }
#endif
    const auto fast_contacts=result.contacts;
    const auto fallback_began=Clock::now();
    double rational_lower_bound=0,rational_upper_bound=0;
    if(result.stats.disjoint) {
        try {
            const auto rational=detail::solve_rational_disjoint(start,target,input);
            result.contacts=rational.contacts;rational_lower_bound=rational.lower_bound;rational_upper_bound=rational.upper_bound;
        } catch(const std::exception &) {
            // Coincident contacts and a few boundary degeneracies make the
            // established cone recurrence undefined.  The exact directional
            // construction remains a proof-grade recovery for those cases.
            const auto rational=detail::solve_disjoint_map_contacts_with_bounds(start,target,input);
            result.contacts=rational.contacts;rational_lower_bound=rational.lower_bound;rational_upper_bound=rational.upper_bound;
            result.stats.rational_disjoint_directional_recovery=true;
        }
    } else {
        const auto rational=detail::solve_intersecting_map_contacts_with_bounds(start,target,input,false);
        result.contacts=rational.contacts;rational_lower_bound=rational.lower_bound;rational_upper_bound=rational.upper_bound;
    }
    result.stats.rational_fallback_seconds=elapsed(fallback_began);
    bool rational_shadow_mismatch=false;
    if(options.shadow_rational && result.stats.disjoint &&
       !result.stats.rational_disjoint_directional_recovery) {
        const auto oracle=detail::solve_disjoint_map_contacts_with_bounds(start,target,input);
        rational_shadow_mismatch=
            rational_upper_bound<oracle.lower_bound || oracle.upper_bound<rational_lower_bound;
        if(rational_shadow_mismatch) {
            result.contacts=oracle.contacts;
            rational_lower_bound=oracle.lower_bound;
            rational_upper_bound=oracle.upper_bound;
        }
    }
    if(options.shadow_rational && result.stats.double_certified) {
        ConvexHybridResult fast_bounds;
        set_exact_bounds(fast_bounds,start,target,exact_contacts,&result.stats);
        const bool certified_intervals_overlap=
            fast_bounds.lower_bound<=rational_upper_bound &&
            rational_lower_bound<=fast_bounds.upper_bound;
        if(certified_intervals_overlap && !rational_shadow_mismatch) {
            result.contacts=fast_contacts;
            result.backend=(result.stats.disjoint||result.stats.touching_disjoint_certified)?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
            result.fallback_reason=ConvexFallbackReason::None;
            result.lower_bound=fast_bounds.lower_bound;result.upper_bound=fast_bounds.upper_bound;
            result.stats.total_seconds=elapsed(began);return result;
        }
        result.fallback_reason=ConvexFallbackReason::ShadowMismatch;
    } else if(rational_shadow_mismatch) {
        result.fallback_reason=ConvexFallbackReason::ShadowMismatch;
    }
    result.stats.rational_fallback=true;
    result.backend=result.stats.disjoint?ConvexHybridBackend::RationalDisjoint:ConvexHybridBackend::RationalIntersection;
    result.lower_bound=rational_lower_bound;result.upper_bound=rational_upper_bound;
    result.stats.total_seconds=elapsed(began);return result;
}

ConvexHybridResult tpp_convex_solve_hybrid(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,const ConvexHybridOptions &options) {
    return solve_hybrid_impl(start,target,input,options,nullptr);
}

ConvexHybridResult tpp_convex_solve_hybrid(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,const ConvexHybridOptions &options,
        DynamicConvexTppWorkspace &workspace) {
    return solve_hybrid_impl(start,target,input,options,&workspace);
}

std::vector<Vector2> tpp_convex_solve_hybrid_safe(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    const auto result=tpp_convex_solve_hybrid(start,target,polygons);
    return reconstruct_convex_polyline(start,target,result.contacts);
}
double tpp_convex_solve_length_hybrid_safe(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    return tpp_convex_solve_hybrid(start,target,polygons).lower_bound;
}
std::vector<Vector2> tpp_convex_solve_hybrid_unchecked(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    ConvexHybridOptions options;options.mode=ConvexHybridMode::Unchecked;
    const auto result=tpp_convex_solve_hybrid(start,target,polygons,options);
    return reconstruct_convex_polyline(start,target,result.contacts);
}
double tpp_convex_solve_length_hybrid_unchecked(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    ConvexHybridOptions options;options.mode=ConvexHybridMode::Unchecked;options.materialize_contacts=false;
    return tpp_convex_solve_hybrid(start,target,polygons,options).lower_bound;
}
}
