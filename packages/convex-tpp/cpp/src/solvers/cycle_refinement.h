#pragma once
#include "cycle_internal.h"
#include <algorithm>
#include <optional>
#include <set>

namespace tpp::detail {
// Active contacts accelerate the existing cycle reduction. Construction is
// shared by exact rationals and doubles; acceptance is always independent.
template<class S> struct CycleRefinement {
    using P=ConvexArithmeticPoint<S>;
    using Polygon=std::vector<P>;
    using Polygons=std::vector<Polygon>;
    static P reflect(P q,P a,P e) {return a+e*(S(2)*(q-a).dot(e)/e.dot(e))-(q-a);}
    static bool inside(P q,const Polygon &p) {
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
    static bool support(P a,P q,P b,const Polygon &p,size_t vertex) {
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
    static std::pair<P,int> coordinate(P a,P b,P old,const Polygon &p) {
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
    static bool close(const Polygons &p,const std::vector<int> &feature,Polygon &q,
                      std::vector<int> *blocking=nullptr) {
        const size_t k=p.size();std::vector<size_t> pins,edges;
        for(size_t i=0;i<k;++i)if(feature[i]>=0&&feature[i]%2==0)pins.push_back(i);
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
                const size_t i=*it,j=size_t(feature[i])/2;
                target=reflect(target,p[i][j],p[i][(j+1)%p[i].size()]-p[i][j]);
            }
            return target;
        };
        auto trace=[&](size_t begin,size_t end,const std::vector<size_t> &indices) {
            std::vector<P> images(indices.size());P target=q[end];
            for(size_t r=indices.size();r-->0;) {
                const size_t i=indices[r],j=size_t(feature[i])/2;
                target=reflect(target,p[i][j],p[i][(j+1)%p[i].size()]-p[i][j]);images[r]=target;
            }
            P current=q[begin];
            for(size_t r=0;r<indices.size();++r) {
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
        for(size_t begin=0;begin<k;++begin)if(feature[begin]>=0) {
            size_t end=(begin+1)%k;while(feature[end]<0)end=(end+1)%k;
            P current=q[begin];
            for(size_t i=(begin+1)%k;i!=end;i=(i+1)%k) {
                q[i]=coordinate(current,q[end],q[i],p[i]).first;current=q[i];
            }
        }
        return true;
    }
    static Polygon intersect(Polygon region,const Polygon &p) {
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
    static void move_zero_blocks(const Polygons &p,Polygon &q,std::vector<int> &feature) {
        const size_t k=q.size();size_t first=0;
        while(first<k&&q[first]==q[(first+1)%k])++first;
        if(first==k)return;
        size_t start=(first+1)%k,used=0;
        while(used<k) {
            size_t count=1;while(used+count<k&&q[start]==q[(start+count)%k])++count;
            const size_t end=(start+count-1)%k;
            if(count>1) {
                Polygon region=p[start];
                for(size_t r=1;r<count;++r)region=intersect(std::move(region),p[(start+r)%k]);
                if(!region.empty()) {
                    const P point=region.size()==1?region.front():coordinate(q[(start+k-1)%k],q[(end+1)%k],q[start],region).first;
                    for(size_t r=0;r<count;++r) {
                        const size_t i=(start+r)%k;q[i]=point;feature[i]=-1;
                        for(size_t j=0;j<p[i].size();++j) {
                            if(q[i]==p[i][j]){feature[i]=int(2*j);break;}
                            if((p[i][(j+1)%p[i].size()]-p[i][j]).cross(q[i]-p[i][j])==0)feature[i]=int(2*j+1);
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
    static bool release_coincident_vertices(const Polygons &p,const Polygon &q,std::vector<int> &feature) {
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
    template<class Consider> static bool run(const Polygons &p,Consider consider,Polygon q={},const std::vector<int> &inherited={}) {
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
            std::vector<int> feature(k);
            for(size_t i=0;i<k;++i)std::tie(q[i],feature[i])=coordinate(q[(i+k-1)%k],q[(i+1)%k],q[i],p[i]);
            move_zero_blocks(p,q,feature);
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
