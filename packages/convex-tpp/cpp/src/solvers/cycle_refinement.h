#pragma once
#include "cycle_internal.h"
#include "cycle_execution.h"
#include "tpp/geometry/common.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <span>

namespace tpp::detail {
// Active contacts accelerate the existing cycle reduction. Construction is
// shared by exact rationals and doubles; acceptance is always independent.
template<class S> struct CycleRefinement {
    using P=ConvexArithmeticPoint<S>;
    using Polygon=std::vector<P>;
    using Polygons=std::vector<Polygon>;
    // A read-only region: a Polygon, or a view of one kept by a caller.
    using View=std::span<const P>;
    // Feature of a contact fixed at its current point (a corner of a common
    // region shared by a block of coincident contacts); see move_zero_blocks.
    static constexpr int kFixed=-3;
    static P reflect(P q,P a,P e) {return a+e*(S(2)*(q-a).dot(e)/e.dot(e))-(q-a);}
    static bool inside(P q,View p) {
        if(p.size()==1)return q==p.front();
        if(p.size()==2) {
            const P e=p[1]-p[0],d=q-p[0];
            return e.cross(d)==0&&d.dot(e)>=0&&d.dot(e)<=e.dot(e);
        }
        for(size_t i=0;i<p.size();++i)if((p[(i+1)%p.size()]-p[i]).cross(q-p[i])<0)return false;
        return true;
    }
    static int sign(S a,S aa,S b,S bb) {
        if constexpr(std::is_same_v<S,double>) {
            const S value=a/std::sqrt(aa)-b/std::sqrt(bb);
            return value>0?1:value<0?-1:0;
        } else return cycle_normalized_difference_sign(a,aa,b,bb);
    }
    static bool support(P a,P q,P b,View p,size_t vertex) {
        const P u=q-a,v=b-q;const S uu=u.dot(u),vv=v.dot(v);
        if(uu==0||vv==0)return true; // Only used for a point on [a,b].
        // The two incident edge rays generate the feasible tangent cone.
        for(size_t j:{(vertex+p.size()-1)%p.size(),(vertex+1)%p.size()}) {
            const P d=p[j]-q;if(sign(u.dot(d),uu,v.dot(d),vv)<0)return false;
        }
        return true;
    }
    static std::optional<P> intersection(P a,P d,P b,P e) {
        const S cross=d.cross(e);if(cross==0)return {};
        return a+d*((b-a).cross(e)/cross);
    }
    // Binary64 polygons with at least this many vertices try the O(log m)
    // coordinate update first; anything it cannot settle runs the loop below.
    static constexpr size_t kLogarithmicVertices=32;
    // A vertex maximizing dir.v on a counterclockwise polygon without collinear
    // vertices, by binary search (KACTL's extrVertex). Only guides searches.
    static size_t extreme(View p,P dir) {
        const long n=long(p.size());
        auto cmp=[&](long i,long j){const S v=-dir.dot(p[size_t(i%n)]-p[size_t(j%n)]);return (v>0)-(v<0);};
        auto extr=[&](long i){return cmp(i+1,i)>=0&&cmp(i,i-1+n)<0;};
        if(extr(0))return 0;
        long lo=0,hi=n;
        while(lo+1<hi) {
            const long mid=(lo+hi)/2;
            if(extr(mid))return size_t(mid);
            const int ls=cmp(lo+1,lo),ms=cmp(mid+1,mid);
            if(ls<ms||(ls==ms&&ls==cmp(lo,mid)))hi=mid;else lo=mid;
        }
        return size_t(lo);
    }
    // Whether an edge of `a` at its vertex extreme toward `direction` leaves
    // every vertex of `b` strictly outside, by binary search for the vertex
    // of b least outside, a local walk, and a margin over its rounding error.
    // Both polygons counterclockwise without collinear vertices. Only used
    // to skip a clip whose result would be empty; false decides nothing.
    static bool separated_toward(View a,View b,Vector2 direction) {
        if constexpr(!std::is_same_v<S,double>)return false;
        else {
            if(a.size()<3||b.size()<3||!direction.is_finite())return false;
            const size_t na=a.size(),nb=b.size(),t=extreme(a,P(direction));
            for(size_t j:{(t+na-1)%na,t}) {
                const P o=a[j],e=a[(j+1)%na]-o;
                auto value=[&](size_t i){return e.cross(b[i%nb]-o);};
                size_t w=extreme(b,P{-e.y,e.x});
                for(int step=0;step<64;++step) {
                    if(value(w+nb-1)>value(w))w=(w+nb-1)%nb;
                    else if(value(w+1)>value(w))w=(w+1)%nb;
                    else break;
                }
                const P d=b[w]-o;const double l=e.x*d.y,r=e.y*d.x;
                const double bound=0x1p-50*(std::abs(l)+std::abs(r))+0x1p-1069;
                if(std::isfinite(l-r)&&std::isfinite(bound)&&l-r<-bound&&value(w+nb-1)<value(w)&&value(w+1)<value(w))return true;
            }
            return false;
        }
    }
    // Floating fan test of q in the convex polygon; only a guard for proposals.
    static bool inside_fan(P q,View p) {
        const size_t m=p.size();const P o=p[0];
        if((p[1]-o).cross(q-o)<0||(p[m-1]-o).cross(q-o)>0)return false;
        size_t lo=1,hi=m-1;
        while(hi-lo>1){const size_t mid=lo+(hi-lo)/2;if((p[mid]-o).cross(q-o)>=0)lo=mid;else hi=mid;}
        return (p[lo+1]-p[lo]).cross(q-p[lo])>=0;
    }
    // Dror et al.'s last-step map of one polygon from source a, located at b
    // by binary search over the vertex cones (as in the binary-search fixed-
    // source solver, with each cone built on demand): 2j for vertex j, 2j+1
    // for edge j, nothing for the region where [a,b] already meets the polygon.
    static std::optional<size_t> locate(P a,P b,View p) {
        const size_t m=p.size();const Vector2 source=a.external(),point=b.external();
        auto vertex=[&](size_t j){return p[j%m].external();};
        auto cone=[&](size_t j) {
            const Vector2 before=vertex(j+m-1),here=vertex(j),after=vertex(j+1),diff=(here-source).normalized();
            return std::pair{diff.cross(here-before)<0?diff.reflect(here-before):diff,
                             diff.cross(here-after)>0?diff.reflect(here-after):diff};
        };
        size_t location;
        if(const auto c=cone(0);tpp::point_in_cone_plus(point,vertex(0),c.first,c.second))location=0;
        else {
            size_t left=0,right=m-1;bool found=false;
            while(left!=right) {
                const size_t mid=left+(right-left)/2,next=mid+1;const auto c=cone(next);
                if(tpp::point_in_cone_plus(point,vertex(next),c.first,c.second)){location=2*next;found=true;break;}
                if(tpp::point_in_edge_plus(point,vertex(left),vertex(next),cone(left).second,c.first))right=mid;
                else left=mid+1;
            }
            if(!found)location=2*left+1;
        }
        // Edge j is a first contact when the source is strictly outside it.
        auto first=[&](size_t j){return (vertex(j+1)-vertex(j)).cross(source-vertex(j))<0;};
        if(first(location/2)||first(location==0?m-1:(location-1)/2))return location;
        return {};
    }
    // The vertex or edge feature at `feature`, if it passes the same tests as
    // the loop in coordinate().
    static std::optional<std::pair<P,int>> checked_feature(P a,P b,View p,size_t feature) {
        const size_t m=p.size(),j=(feature/2)%m;
        if(feature%2==0)return support(a,p[j],b,p,j)?std::optional{std::pair{p[j],int(2*j)}}:std::nullopt;
        const P e=p[(j+1)%m]-p[j];
        if(e.cross(a-p[j])*e.cross(b-p[j])<=0)return {};
        const auto q=intersection(a,reflect(b,p[j],e)-a,p[j],e);
        if(!q)return {};
        const S t=(*q-p[j]).dot(e)/e.dot(e);
        if(!(t>0&&t<1))return {};
        const P u=*q-a,v=b-*q;
        if(u.zero()||v.zero())return {};
        const P inward{-e.y,e.x};
        if(sign(u.dot(inward),u.dot(u),v.dot(inward),v.dot(v))<0)return {};
        return std::pair{*q,int(2*j+1)};
    }
    // O(log m) version of coordinate() for binary64 polygons: the chord of
    // the line ab from the two edges where it crosses the boundary (found by
    // binary search between the extreme vertices on either side of the line),
    // and otherwise the located last-step feature, checked as in the loop.
    static std::optional<std::pair<P,int>> coordinate_logarithmic(P a,P b,P old,View p) {
        const size_t m=p.size();
        const P d=b-a;const S dd=d.dot(d);
        if(dd==0)return {};
        const P n{-d.y,d.x};
        const size_t top=extreme(p,n),bottom=extreme(p,P{-n.x,-n.y});
        const S high=n.dot(p[top]-a),low=n.dot(p[bottom]-a);
        if(high>=0&&low<=0) {
            auto crossing=[&](size_t from,size_t to,bool falling) {
                size_t lo=0,hi=(to+m-from)%m;
                while(hi-lo>1) {
                    const size_t mid=(lo+hi)/2;const S s=n.dot(p[(from+mid)%m]-a);
                    if(falling?s>=0:s<=0)lo=mid;else hi=mid;
                }
                return (from+lo)%m;
            };
            S lo=0,hi=1;
            for(size_t edge:{crossing(top,bottom,true),crossing(bottom,top,false)})
                for(size_t j:{(edge+m-1)%m,edge,(edge+1)%m}) {
                    const P e=p[(j+1)%m]-p[j];const S offset=e.cross(a-p[j]),slope=e.cross(d);
                    if(slope>0)lo=std::max(lo,S(-offset/slope));
                    else if(slope<0)hi=std::min(hi,S(-offset/slope));
                    else if(offset<0)return {};
                }
            if(lo<=hi) {
                const S t=std::clamp(S((old-a).dot(d)/dd),lo,hi);
                if(!(t>0&&t<1))return {}; // a or b on the chord: the loop picks the feature
                const P q=a+d*t;
                if(!inside_fan(q,p))return {};
                return std::pair{q,-1};
            }
        }
        const auto location=locate(a,b,p);
        if(!location)return {};
        for(size_t offset:{size_t(0),size_t(1),2*m-1,size_t(2),2*m-2})
            if(auto found=checked_feature(a,b,p,(*location+offset)%(2*m)))return found;
        return {};
    }
    static std::pair<P,int> coordinate(P a,P b,P old,View p) {
        cycle_checkpoint();
        // A point's zero-length "edges" give no halfplanes: its only contact
        // is the point itself, a vertex feature.
        if(p.size()==1)return {p.front(),0};
        if constexpr(std::is_same_v<S,double>)
            if(p.size()>=kLogarithmicVertices)
                if(auto found=coordinate_logarithmic(a,b,old,p))return *found;
        return coordinate_linear(a,b,old,p);
    }
    // Every edge once: the crossing chord, then the first supporting vertex or
    // reflecting edge in boundary order.
    static std::pair<P,int> coordinate_linear(P a,P b,P old,View p) {
        if(p.size()==1)return {p.front(),0};
        const P d=b-a;const S dd=d.dot(d);S lo=0,hi=1;
        bool crosses=true;
        for(size_t j=0;j<p.size();++j) {
            const P e=p[(j+1)%p.size()]-p[j];const S offset=e.cross(a-p[j]),slope=e.cross(d);
            if(slope==0){if(offset<0){crosses=false;break;}}
            else if(slope>0)lo=std::max(lo,S(-offset/slope));
            else hi=std::min(hi,S(-offset/slope));
        }
        if(p.size()==2) {
            // The two opposite edge halfplanes describe the supporting line.
            // Its endpoint bounds are needed to obtain the closed segment.
            const P e=p[1]-p[0];const S offset=(a-p[0]).dot(e),slope=d.dot(e),ee=e.dot(e);
            if(slope==0){if(offset<0||offset>ee)crosses=false;}
            else if(slope>0){lo=std::max(lo,S(-offset/slope));hi=std::min(hi,S((ee-offset)/slope));}
            else {lo=std::max(lo,S((ee-offset)/slope));hi=std::min(hi,S(-offset/slope));}
        }
        if(crosses&&lo<=hi) {
            const S t=dd==0?S(0):std::clamp(S((old-a).dot(d)/dd),lo,hi);
            const P q=a+d*t;
            // A contact strictly between its neighbours is straight-through,
            // even on a polygon edge or vertex. Reflecting at that inactive
            // boundary creates a spurious bend in the closed construction.
            if(t>0&&t<1)return {q,-1};
            for(size_t j=0;j<p.size();++j)if(q==p[j])return {q,int(2*j)};
            for(size_t j=0;j<p.size();++j)if((p[(j+1)%p.size()]-p[j]).cross(q-p[j])==0)return {q,int(2*j+1)};
            return {q,-1};
        }
        for(size_t j=0;j<p.size();++j) {
            if(support(a,p[j],b,p,j))return {p[j],int(2*j)};
            const P e=p[(j+1)%p.size()]-p[j];
            // Both endpoints are on the same side of a supporting edge.
            // Reflect one of them, then intersect the straightened segment.
            if(e.cross(a-p[j])*e.cross(b-p[j])<=0)continue;
            const auto q=intersection(a,reflect(b,p[j],e)-a,p[j],e);
            if(!q)continue;
            const S t=(*q-p[j]).dot(e)/e.dot(e);
            if(t>0&&t<1) {
                // Tangency follows from construction; checking the inward sign
                // avoids rejecting an edge solely due to roundoff in tangency.
                const P u=*q-a,v=b-*q;
                if(u.zero()||v.zero())continue;
                const P inward{-e.y,e.x};
                if(sign(u.dot(inward),u.dot(u),v.dot(inward),v.dot(v))>=0)
                    return {*q,int(2*j+1)};
            }
        }
        throw std::runtime_error("No representable local cycle contact");
    }
    template<class Regions> static bool close(const Regions &p,const std::vector<int> &feature,Polygon &q,
                      std::vector<int> *blocking=nullptr) {
        cycle_checkpoint();
        const size_t k=p.size();std::vector<size_t> pins,edges;
        // Vertex features pin a contact at that vertex; kFixed pins it at q[i].
        for(size_t i=0;i<k;++i)if((feature[i]>=0&&feature[i]%2==0)||feature[i]==kFixed)pins.push_back(i);
        auto on_edge=[&](size_t i,size_t j,const S &t) {
            if(t>=0&&t<=1)return true;
            if(blocking) {
                const size_t vertex=t<0?j:(j+1)%p[i].size();
                (*blocking)[i]=int(2*vertex);q[i]=p[i][vertex];
            }
            return false;
        };
        auto unfold=[&](P target,const std::vector<size_t> &indices) {
            for(auto it=indices.rbegin();it!=indices.rend();++it) {
        cycle_checkpoint();
                const size_t i=*it,j=size_t(feature[i])/2;
                target=reflect(target,p[i][j],p[i][(j+1)%p[i].size()]-p[i][j]);
            }
            return target;
        };
        auto trace=[&](size_t begin,size_t end,const std::vector<size_t> &indices) {
            std::vector<P> images(indices.size());P target=q[end];
            for(size_t r=indices.size();r-->0;) {
        cycle_checkpoint();
                const size_t i=indices[r],j=size_t(feature[i])/2;
                target=reflect(target,p[i][j],p[i][(j+1)%p[i].size()]-p[i][j]);images[r]=target;
            }
            P current=q[begin];
            for(size_t r=0;r<indices.size();++r) {
        cycle_checkpoint();
                const size_t i=indices[r],j=size_t(feature[i])/2;const P e=p[i][(j+1)%p[i].size()]-p[i][j];
                const auto point=intersection(current,images[r]-current,p[i][j],e);
                if(!point)return false;
                const S t=(*point-p[i][j]).dot(e)/e.dot(e);
                if(!on_edge(i,j,t))return false;
                q[i]=current=*point;
            }
            return true;
        };
        if(pins.empty()) {
            size_t anchor=0;while(anchor<k&&feature[anchor]<0)++anchor;
            if(anchor==k)return false;
            for(size_t r=1;r<k;++r){const size_t i=(anchor+r)%k;if(feature[i]>=0)edges.push_back(i);}
            const size_t j=size_t(feature[anchor])/2;const P a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            const P c=unfold(a,edges)-a,d=unfold(a+e,edges)-(a+e)-c;
            const S dd=d.dot(d);const S t=dd==0?S(1)/2:S(-c.dot(d)/dd);
            if(!on_edge(anchor,j,t))return false;
            q[anchor]=a+e*t;pins.push_back(anchor);
        }
        for(size_t r=0;r<pins.size();++r) {
            const size_t begin=pins[r],end=pins[(r+1)%pins.size()];edges.clear();
            for(size_t i=(begin+1)%k;i!=end;i=(i+1)%k)if(feature[i]>=0)edges.push_back(i);
            if(!trace(begin,end,edges))return false;
        }
        // Restore skipped straight-through contacts in their original order.
        for(size_t begin=0;begin<k;++begin)if(feature[begin]>=0||feature[begin]==kFixed) {
            size_t end=(begin+1)%k;while(feature[end]<0&&feature[end]!=kFixed)end=(end+1)%k;
            P current=q[begin];
            for(size_t i=(begin+1)%k;i!=end;i=(i+1)%k) {
                q[i]=coordinate(current,q[end],q[i],p[i]).first;current=q[i];
            }
        }
        return true;
    }
    // Keeps the part of a closed convex region where normal.(x-a) >= 0.
    static Polygon clip(Polygon region,const P &a,const P &normal) {
        Polygon next;
        for(size_t r=0;r<region.size();++r) {
            const P u=region[r],v=region[(r+1)%region.size()];
            const S su=normal.dot(u-a),sv=normal.dot(v-a);
            if(su>=0 && (next.empty()||!(next.back()==u)))next.push_back(u);
            if((su<0&&sv>0)||(su>0&&sv<0))next.push_back(u+(v-u)*(su/(su-sv)));
        }
        if(next.size()>1&&next.front()==next.back())next.pop_back();
        return next;
    }
    // Whether the edge of counterclockwise `a` most violated by the vertex mean
    // of `b` leaves every vertex of `b` strictly outside its halfplane.
    static bool separates(View a,View b) {
        P mean=b.front();for(size_t i=1;i<b.size();++i)mean=mean+b[i];
        mean=mean*(S(1)/S(b.size()));
        size_t best=0;S lowest=0;
        for(size_t j=0;j<a.size();++j) {
            const P o=a[j],e=a[(j+1)%a.size()]-o;const S side=e.cross(mean-o);
            if(j==0||side<lowest){lowest=side;best=j;}
        }
        if(!(lowest<0))return false;
        const P o=a[best],e=a[(best+1)%a.size()]-o;
        for(const auto &v:b)if(!(e.cross(v-o)<0))return false;
        return true;
    }
    // Whether every computed turn of p is strictly counterclockwise.
    static bool strictly_convex(View p) {
        const size_t m=p.size();
        if(m<3)return false;
        for(size_t i=0;i<m;++i)if(!((p[(i+1)%m]-p[i]).cross(p[(i+2)%m]-p[(i+1)%m])>0))return false;
        return true;
    }
    // Parameters along a + t d where the line enters (lo, through edge
    // `enter`) and leaves (hi, edge `leave`) a counterclockwise polygon
    // without collinear vertices: the edges where it crosses the boundary, by
    // binary search between the extremes on either side, and their
    // neighbors. Nothing if the line misses the polygon.
    struct Chord {S lo,hi;size_t enter,leave;};
    static std::optional<Chord> chord(P a,P d,View p) {
        const size_t m=p.size();const P n{-d.y,d.x};
        if(n.x==0&&n.y==0)return {};
        const size_t top=extreme(p,n),bottom=extreme(p,P{-n.x,-n.y});
        if(!(n.dot(p[top]-a)>=0&&n.dot(p[bottom]-a)<=0))return {};
        auto crossing=[&](size_t from,size_t to,bool falling) {
            size_t lo=0,hi=(to+m-from)%m;
            while(hi-lo>1) {
                const size_t mid=(lo+hi)/2;const S side=n.dot(p[(from+mid)%m]-a);
                if(falling?side>=0:side<=0)lo=mid;else hi=mid;
            }
            return (from+lo)%m;
        };
        Chord c{-std::numeric_limits<S>::infinity(),std::numeric_limits<S>::infinity(),0,0};
        for(size_t edge:{crossing(top,bottom,true),crossing(bottom,top,false)})
            for(size_t j:{(edge+m-1)%m,edge,(edge+1)%m}) {
                const P e=p[(j+1)%m]-p[j];const S offset=e.cross(a-p[j]),slope=e.cross(d);
                if(slope>0){if(-offset/slope>c.lo){c.lo=-offset/slope;c.enter=j;}}
                else if(slope<0){if(-offset/slope<c.hi){c.hi=-offset/slope;c.leave=j;}}
                else if(offset<0)return {};
            }
        if(!(c.lo<=c.hi)||!std::isfinite(c.lo)||!std::isfinite(c.hi))return {};
        return c;
    }
    // Intersection of two counterclockwise polygons without collinear
    // vertices in O((m+n) log n), binary64 only: the vertices of a in b (fan
    // tests), the crossings of a's edges with b's boundary (chords), and the
    // arcs of b from each exit of a to the next entry. A proposal for
    // move_zero_blocks and the common-region test, which the certificates
    // check; nothing (use the clip) when the crossings do not alternate.
    static std::optional<Polygon> intersect_convex(const Polygon &a,View b) {
        const size_t ma=a.size(),mb=b.size();
        std::vector<char> in(ma);
        for(size_t i=0;i<ma;++i)in[i]=inside_fan(a[i],b);
        struct Crossing {size_t edge_a;bool entering;size_t edge_b;P point;};
        std::vector<Crossing> crossings;
        for(size_t i=0;i<ma;++i) {
            const size_t next=(i+1)%ma;
            if(in[i]&&in[next])continue;
            const P u=a[i],d=a[next]-u;
            const auto c=chord(u,d,b);
            const size_t before=crossings.size();
            if(c) {
                if(!in[i]&&c->lo>=0&&c->lo<=1&&(in[next]||c->lo<c->hi))crossings.push_back({i,true,c->enter,u+d*c->lo});
                if(!in[next]&&c->hi>=0&&c->hi<=1&&(in[i]||c->lo<c->hi))crossings.push_back({i,false,c->leave,u+d*c->hi});
            }
            if(in[i]!=in[next]&&crossings.size()!=before+1)return {};
            if(!in[i]&&!in[next]&&crossings.size()!=before&&crossings.size()!=before+2)return {};
        }
        if(crossings.empty()) {
            if(std::all_of(in.begin(),in.end(),[](char v){return v!=0;}))return a;
            if(inside_fan(b[0],View(a)))return Polygon(b.begin(),b.end());
            return Polygon{};
        }
        const size_t k=crossings.size();
        if(k%2)return {};
        const size_t first=crossings[0].entering?0:1;
        for(size_t r=0;r<k;++r)if(crossings[(first+r)%k].entering!=(r%2==0))return {};
        Polygon out;
        auto push=[&](const P &v){if(out.empty()||!(out.back()==v))out.push_back(v);};
        for(size_t r=0;r<k;r+=2) {
            const auto &enter=crossings[(first+r)%k],&leave=crossings[(first+r+1)%k],&again=crossings[(first+r+2)%k];
            push(enter.point);
            for(size_t i=enter.edge_a;i!=leave.edge_a;){i=(i+1)%ma;push(a[i]);}
            push(leave.point);
            for(size_t j=leave.edge_b;j!=again.edge_b;){j=(j+1)%mb;push(b[j]);}
        }
        if(out.size()>1&&out.front()==out.back())out.pop_back();
        return out;
    }
    // Intersection of two closed convex regions; p may also be a point or a
    // segment, whose edge halfplanes alone would describe all or a whole line.
    static Polygon intersect(Polygon region,View p) {
        cycle_checkpoint();
        if(p.size()==1) {
            if(region.empty())return region;
            const P c=p.front();
            Polygon point{c};
            for(size_t j=0;j<region.size()&&region.size()>2;++j) {
                const P a=region[j],e=region[(j+1)%region.size()]-a;
                if(e.cross(c-a)<0)return {};
            }
            if(region.size()==1&&!(region.front()==c))return {};
            if(region.size()==2) {
                const P a=region[0],e=region[1]-a;
                if(e.cross(c-a)!=0||e.dot(c-a)<0||e.dot(c-region[1])>0)return {};
            }
            return point;
        }
        if(p.size()==2) {
            const P a=p[0],b=p[1],e=b-a,left{-e.y,e.x};
            for(const auto &[origin,normal]:{std::pair{a,left},std::pair{a,P{e.y,-e.x}},std::pair{a,e},std::pair{b,P{-e.x,-e.y}}}) {
                if(region.empty())break;
                region=clip(std::move(region),origin,normal);
            }
            return region;
        }
        // Clipping by every edge costs O(m) per edge. Disjoint regions are
        // usually separated by the edge of one that faces the other, so try
        // that edge of each first: all vertices strictly outside it is the
        // empty result the loop below would reach, in O(m) instead of O(m^2).
        if(region.size()>2&&(separates(p,region)||separates(region,p)))return {};
        if constexpr(std::is_same_v<S,double>)
            if(region.size()>=kLogarithmicVertices&&p.size()>=kLogarithmicVertices&&strictly_convex(region)&&strictly_convex(p))
                if(auto fast=intersect_convex(region,p))return std::move(*fast);
        for(size_t j=0;j<p.size()&&!region.empty();++j) {
            Polygon next;const P a=p[j],e=p[(j+1)%p.size()]-a;
            for(size_t r=0;r<region.size();++r) {
                const P u=region[r],v=region[(r+1)%region.size()];
                const S su=e.cross(u-a),sv=e.cross(v-a);
                if(su>=0 && (next.empty()||!(next.back()==u)))next.push_back(u);
                if((su<0&&sv>0)||(su>0&&sv<0))next.push_back(u+(v-u)*(su/(su-sv)));
            }
            if(next.size()>1&&next.front()==next.back())next.pop_back();
            region=std::move(next);
        }
        return region;
    }
    // Optional memo of the common region of two regions, by their indices in
    // the cycle, for move_zero_blocks (binary64): set by a caller around
    // run() with PairScope; nothing returned means compute it here.
    using PairRegion=std::function<const Polygon*(size_t,size_t)>;
    static inline thread_local const PairRegion *pair_region=nullptr;
    struct PairScope {
        const PairRegion *saved;
        explicit PairScope(const PairRegion *memo):saved(pair_region){pair_region=memo;}
        ~PairScope(){pair_region=saved;}
    };
    template<class Regions> static void move_zero_blocks(const Regions &p,Polygon &q,std::vector<int> &feature) {
        const size_t k=q.size();size_t first=0;
        while(first<k&&q[first]==q[(first+1)%k])++first;
        if(first==k)return;
        size_t start=(first+1)%k,used=0;
        while(used<k) {
            size_t count=1;while(used+count<k&&q[start]==q[(start+count)%k])++count;
            const size_t end=(start+count-1)%k;
            if(count>1) {
                const Polygon *known=nullptr;
                if constexpr(std::is_same_v<S,double>)if(count==2&&pair_region&&*pair_region)known=(*pair_region)(start,(start+1)%k);
                Polygon computed;
                if(!known) {
                    computed.assign(p[start].begin(),p[start].end());
                    for(size_t r=1;r<count;++r)computed=intersect(std::move(computed),p[(start+r)%k]);
                }
                const Polygon &region=known?*known:computed;
                if(!region.empty()) {
                    // A corner of the common region (e.g. two crossing edges)
                    // pins the whole block there, although it need not be a
                    // vertex of any one region: reflecting at both edges would
                    // never reconstruct it.
                    bool corner=region.size()==1;P point=region.front();
                    if(!corner) {
                        const auto [found,kind]=coordinate(q[(start+k-1)%k],q[(end+1)%k],q[start],region);
                        point=found;corner=kind>=0&&kind%2==0;
                    }
                    bool keep=true;
                    if constexpr(std::is_same_v<S,double>) {
                        // In exact arithmetic the block's point lies in the common
                        // region, so this move never lengthens the cycle. Rounding
                        // can shrink the clipped region of a shared wall to its far
                        // corner; such a move would lengthen it and pin the block.
                        const P prev=q[(start+k-1)%k],next=q[(end+1)%k];
                        const auto norm=[](P v){return std::sqrt(v.dot(v));};
                        keep=norm(point-prev)+norm(next-point)<=norm(q[start]-prev)+norm(next-q[end]);
                    }
                    if(keep)
                    for(size_t r=0;r<count;++r) {
                        const size_t i=(start+r)%k;q[i]=point;feature[i]=corner?kFixed:-1;
                        for(size_t j=0;j<p[i].size();++j) {
                            if(q[i]==p[i][j]){feature[i]=int(2*j);break;}
                            if(!corner&&(p[i][(j+1)%p[i].size()]-p[i][j]).cross(q[i]-p[i][j])==0)feature[i]=int(2*j+1);
                        }
                    }
                }
            }
            used+=count;start=(end+1)%k;
        }
    }
    // A coincident vertex pair can be stationary under separate coordinate
    // updates while improving when both contacts leave the shared vertex.
    // Propose the incident faces exposed toward the neighboring contacts,
    // then solve their reflection equations together. The certificate decides
    // whether that feature change was valid; no perturbation distance is used.
    template<class Regions> static bool release_coincident_vertices(const Regions &p,const Polygon &q,std::vector<int> &feature) {
        bool changed=false;const size_t k=p.size();
        for(size_t i=0;i<k;++i) {
            if(feature[i]<0||feature[i]%2!=0||
               (!(q[i]==q[(i+k-1)%k])&&!(q[i]==q[(i+1)%k])))continue;
            const P toward=(q[(i+k-1)%k]+q[(i+1)%k])*S(0.5);
            const size_t vertex=size_t(feature[i])/2;int edge=-1;S best=0;
            for(size_t j:{(vertex+p[i].size()-1)%p[i].size(),vertex}) {
                const P e=p[i][(j+1)%p[i].size()]-p[i][j];const S side=e.cross(toward-p[i][j]);
                if(side>=0)continue;
                const S score=side*side/e.dot(e);
                if(edge<0||score>best){edge=int(j);best=score;}
            }
            if(edge>=0){feature[i]=2*edge+1;changed=true;}
        }
        return changed;
    }
    template<class Regions,class Consider> static bool run(const Regions &p,Consider consider,Polygon q={},const std::vector<int> &inherited={}) {
        const size_t k=p.size();
        auto submit=[&](const Polygon &candidate,const std::vector<int> &features,bool closed) {
            if constexpr(std::is_invocable_r_v<bool,Consider,const Polygon&,const std::vector<int>&,bool>)
                return consider(candidate,features,closed);
            else return consider(candidate);
        };
        if(k==2) {
            // A disjoint two-region cycle is twice their distance. A closest
            // pair includes a vertex, even for parallel supporting edges.
            // Enumerate vertex/segment projections in the shared arithmetic;
            // the usual certificate still decides acceptance (also on overlap).
            Polygon pair(2);std::vector<int> features(2);S best=0;bool found=false;
            for(size_t i=0;i<2;++i)for(size_t a=0;a<p[i].size();++a)
                for(size_t b=0;b<p[1-i].size();++b) {
                    const P v=p[i][a],w=p[1-i][b],e=p[1-i][(b+1)%p[1-i].size()]-w;
                    const S t=std::clamp(S((v-w).dot(e)/e.dot(e)),S(0),S(1));
                    const P foot=t==0?w:t==1?p[1-i][(b+1)%p[1-i].size()]:w+e*t,d=v-foot;
                    const S squared=d.dot(d);
                    if(found&&squared>=best)continue;
                    found=true;best=squared;pair[i]=v;pair[1-i]=foot;
                    features[i]=int(2*a);
                    features[1-i]=t==0?int(2*b):t==1?int(2*((b+1)%p[1-i].size())):int(2*b+1);
                }
            if(found&&submit(pair,features,true))return true;
        }
        if(q.empty()) {
            q.resize(k);
            for(size_t i=0;i<k;++i){for(const auto &v:p[i])q[i]=q[i]+v;q[i]=q[i]*(S(1)/p[i].size());}
        }
        if(inherited.size()==k) {
            auto feature=inherited;auto proposal=q;std::vector<bool> repair(k,false);
            bool valid=true;
            for(size_t i=0;i<k;++i) {
                if(feature[i]<-2||feature[i]>=int(2*p[i].size()))valid=false;
                if(feature[i]==-2)repair[i]=repair[(i+k-1)%k]=repair[(i+1)%k]=true;
            }
            if(valid) {
                for(size_t i=0;i<k;++i)if(feature[i]>=0&&feature[i]%2==0)proposal[i]=p[i][size_t(feature[i])/2];
                for(size_t i=0;i<k;++i)if(repair[i])
                    std::tie(proposal[i],feature[i])=coordinate(proposal[(i+k-1)%k],proposal[(i+1)%k],proposal[i],p[i]);
                if(close(p,feature,proposal)&&submit(proposal,feature,true))return true;
            }
        }
        std::set<std::vector<int>> visited;
        for(;;) {
        cycle_checkpoint();
            std::vector<int> feature(k);
            for(size_t i=0;i<k;++i)std::tie(q[i],feature[i])=coordinate(q[(i+k-1)%k],q[(i+1)%k],q[i],p[i]);
            move_zero_blocks(p,q,feature);
            // Rational coordinate sweeps can carry much larger denominators
            // than the closed reflection construction. Certify that construction
            // first; every accepted proposal still uses the same verifier.
            if constexpr(std::is_same_v<S,double>)
                if(submit(q,feature,false))return true;
            Polygon closed=q;
            auto active=feature;
            bool constructed=false;
            // Every successful pivot replaces an edge feature by a vertex.
            // At most k such changes are possible before a fresh sweep.
            for(size_t pivot=0;pivot<=k;++pivot) {
                auto next=active;
                if(close(p,active,closed,&next)){constructed=true;break;}
                if(next==active)break;
                active=std::move(next);
            }
            if(constructed&&submit(closed,active,true))return true;
            if constexpr(!std::is_same_v<S,double>)
                if(submit(q,feature,false))return true;
            auto released=feature;Polygon unpinned=q;
            if(release_coincident_vertices(p,q,released)&&close(p,released,unpinned)&&submit(unpinned,released,true))return true;
            // A blocking endpoint or a restored skipped contact may change the
            // active type. Feed that feasible construction into the next sweep
            // instead of repeating projections between almost parallel edges.
            if(constructed){q=std::move(closed);feature=std::move(active);}
            // This is a finite feature proposal, not a convergence tolerance.
            // Failure hands control back to the exact anchored reduction.
            if(!visited.insert(feature).second||visited.size()>=k+1)return false;
            // A structural work limit bounds this acceleration pass. It is not
            // an acceptance tolerance: the anchored solver handles the rest.
        }
    }

};
} // namespace tpp::detail
