#include "tpp/convex/disk_cycle.h"
#include "cycle_interval.h"
#include "float_chain.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tpp {
namespace {
using detail::CycleInterval;

// r^2 - |x-c|^2 >= 0 with directed rounding; a point region holds only its center.
bool proved_inside(Vector2 x,const ConvexDisk &d) {
    if(d.radius==0)return x.x==d.center.x&&x.y==d.center.y;
    const auto dx=CycleInterval(x.x)-CycleInterval(d.center.x),dy=CycleInterval(x.y)-CycleInterval(d.center.y);
    return (CycleInterval(d.radius).square()-(dx.square()+dy.square())).lo>=0;
}
// The proposal, moved toward the center until it is proved inside; the center always is.
Vector2 inside_contact(Vector2 x,const ConvexDisk &d) {
    if(d.radius==0||!x.is_finite())return d.center;
    if(proved_inside(x,d))return x;
    for(const double f:{0x1p-45,0x1p-40,0x1p-30,0x1p-20,0x1p-10,0x1p-6,0x1p-3,0.5}) {
        const Vector2 y=d.center+(x-d.center)*(1-f);
        if(proved_inside(y,d))return y;
    }
    return d.center;
}
CycleInterval link_length(Vector2 a,Vector2 b) {
    const auto dx=CycleInterval(b.x)-CycleInterval(a.x),dy=CycleInterval(b.y)-CycleInterval(a.y);
    return (dx.square()+dy.square()).sqrt();
}
// u itself if proved in the unit disk, else shrunk until it is.
Vector2 unit_disk(Vector2 u) {
    if(!u.is_finite())return {};
    for(int attempt=0;attempt<8;++attempt) {
        if((CycleInterval(u.x).square()+CycleInterval(u.y).square()).hi<=1)return u;
        u=u*(1-0x1p-40);
        if(attempt==3)u=u/std::sqrt(u.dot(u))*(1-0x1p-30);
    }
    return {};
}

// ---- Geometric construction --------------------------------------------
// A contact and its type: on its circle (a reflection), at a corner of its
// circle with the next corner's (two overlapping disks visited at one point
// of both boundaries), or neither (a straight passage, or a point region).
struct Contact {Vector2 point;bool reflect=false,corner=false;};
Vector2 polar(double th){return {std::cos(th),std::sin(th)};}

// Alhazen's problem: the point of the disk minimizing |x-a|+|x-b|. If the
// segment ab meets the disk, its point closest to the center is optimal (a
// straight passage). Otherwise the minimizer lies on the arc between the
// radial directions of a and b: at each end the distance to that end's point
// is stationary along the circle and the other decreases inward, so
// g'(theta) changes sign there; Newton on g' kept inside that bracket finds it.
Contact alhazen(Vector2 a,Vector2 b,const ConvexDisk &d) {
    if(d.radius==0)return {d.center,false};
    const Vector2 e=b-a;const double ee=e.dot(e);
    const double t=ee>0?std::clamp((d.center-a).dot(e)/ee,0.0,1.0):0.0;
    const Vector2 foot=a+e*t,f=foot-d.center;
    const double rr=d.radius*d.radius;
    if(f.dot(f)<=rr)return {foot,false};
    const Vector2 ra=a-d.center,rb=b-d.center;
    const double ta=std::atan2(ra.y,ra.x);
    double span=std::atan2(rb.y,rb.x)-ta;
    if(span>M_PI)span-=2*M_PI;else if(span<-M_PI)span+=2*M_PI;
    if(span==0)return {d.center+polar(ta)*d.radius,true}; // a and b on one ray
    double lo=std::min(ta,ta+span),hi=std::max(ta,ta+span);
    auto slope=[&](double th,double &curvature) {
        const Vector2 n=polar(th),tangent{-n.y,n.x},p=d.center+n*d.radius;
        const Vector2 da=p-a,db=p-b;const double la=std::sqrt(da.dot(da)),lb=std::sqrt(db.dot(db));
        const Vector2 sa=da/la,sb=db/lb;
        const double ga=tangent.dot(sa),gb=tangent.dot(sb);
        curvature=d.radius*(-n.dot(sa+sb))+rr*((1-ga*ga)/la+(1-gb*gb)/lb);
        return d.radius*(ga+gb);
    };
    double th=0.5*(lo+hi);
    for(int iteration=0;iteration<64;++iteration) {
        double curvature=0;const double g=slope(th,curvature);
        if(g==0)break;
        if(g<0)lo=th;else hi=th;
        double next=curvature>0?th-g/curvature:0.5*(lo+hi);
        if(!(next>lo&&next<hi))next=0.5*(lo+hi);
        if(next==th||hi-lo<=0x1p-50*std::max(1.0,std::abs(th)))break;
        th=next;
    }
    return {d.center+polar(th)*d.radius,true};
}

double distance(Vector2 a,Vector2 b){const Vector2 d=b-a;return std::sqrt(d.dot(d));}
bool near_inside(Vector2 x,const ConvexDisk &d) {
    const Vector2 r=x-d.center;return r.dot(r)<=d.radius*d.radius*(1+1e-10);
}
// The parameters t in [lo, hi] of the points p + t e inside the disk; false if none.
bool segment_interval(Vector2 p,Vector2 e,const ConvexDisk &d,double &lo,double &hi) {
    const Vector2 f=p-d.center;const double ee=e.dot(e);
    if(!(ee>0)){lo=0;hi=1;return near_inside(p,d);}
    const double half=f.dot(e)/ee,disc=half*half-(f.dot(f)-d.radius*d.radius)/ee;
    if(disc<0)return false;
    lo=-half-std::sqrt(disc);hi=-half+std::sqrt(disc);
    return true;
}
// The points where two circles cross; false if they do not.
bool crossings(const ConvexDisk &p,const ConvexDisk &q,Vector2 &first,Vector2 &second) {
    const Vector2 e=q.center-p.center;const double dd=e.dot(e),d=std::sqrt(dd);
    if(!(d>0)||d>p.radius+q.radius||d<std::abs(p.radius-q.radius))return false;
    const double along=(p.radius*p.radius-q.radius*q.radius+dd)/(2*d);
    const double h=std::sqrt(std::max(0.0,p.radius*p.radius-along*along));
    const Vector2 u=e/d,m=p.center+u*along,n{-u.y,u.x};
    first=m+n*h;second=m-n*h;
    return true;
}

// Straight passages through `members` (in cyclic order) on the segment
// p + t e, t in [0, 1], in order along it (out of order they would zigzag):
// each as close to its center as the order allows, below the last parameter
// the later ones can take. False if the segment misses one (it goes to the
// point closest to its center).
bool place_straight(const std::vector<ConvexDisk> &disks,Vector2 p,Vector2 e,const std::vector<size_t> &members,std::vector<Contact> &out) {
    const size_t m=members.size();
    std::vector<double> lo(m),hi(m);
    for(size_t j=0;j<m;++j) {
        double a,b;
        if(!segment_interval(p,e,disks[members[j]],a,b)){a=1;b=0;}
        lo[j]=std::max(a,0.0);hi[j]=std::min(b,1.0);
    }
    for(size_t j=m;j-->1;)hi[j-1]=std::min(hi[j-1],std::max(hi[j],lo[j-1]));
    const double ee=e.dot(e);
    double previous=0;bool met=true;
    out.resize(m);
    for(size_t j=0;j<m;++j) {
        const double closest=ee>0?(disks[members[j]].center-p).dot(e)/ee:0.0;
        const double from=std::max(previous,lo[j]);
        const bool inside=from<=hi[j];met&=inside;
        const double t=inside?std::clamp(closest,from,hi[j]):std::clamp(closest,previous,1.0);
        out[j]={p+e*t,false,false};previous=t;
    }
    return met;
}

// A block: consecutive disks whose contacts coincide. Alhazen updates move one
// contact at a time, and a zero-length link holds two coincident contacts
// together (moving either alone never shortens the cycle), so a cycle through
// a lens of overlapping disks stalls there. Between its neighbours a and b the
// block takes the shortest of these placements:
// - joined, one point of the intersection R of its disks minimizing
//   |x-a|+|x-b|: on the segment ab if it meets R; else on the boundary of R,
//   at the Alhazen point of one of its circles if R holds it, or at a corner
//   where two circles cross;
// - one member reflecting at its Alhazen point, the others straight on the
//   segments to a and b (one circle active, outside the others);
// - split, one Alhazen pass each way ignoring the coincidence.
// A corner is kept only if it is optimal with the members apart too (its dual
// splits, see type_duals); otherwise, unless another placement is shorter, its
// two members start reflecting a little apart, and Newton on the type
// separates them (tangent circles, as in Mennell's bubbles, meet at one point
// that no tour needs to share). The current contacts are no candidate: a
// straight passage put on its reduced segment may lie outside its disk.
void block_update(const std::vector<ConvexDisk> &disks,std::vector<Contact> &c,const std::vector<size_t> &run) {
    const size_t k=c.size(),m=run.size();
    const Vector2 a=c[(run.front()+k-1)%k].point,b=c[(run.back()+1)%k].point;
    auto in_all=[&](Vector2 x,size_t skip1,size_t skip2) {
        for(size_t i:run)if(i!=skip1&&i!=skip2&&!near_inside(x,disks[i]))return false;
        return true;
    };
    auto chain=[&](const std::vector<Contact> &cs) {
        double s=distance(a,cs.front().point)+distance(cs.back().point,b);
        for(size_t j=0;j+1<cs.size();++j)s+=distance(cs[j].point,cs[j+1].point);
        return s;
    };
    std::vector<Contact> best(m);
    for(size_t j=0;j<m;++j)best[j]=alhazen(j?best[j-1].point:a,b,disks[run[j]]);
    for(size_t j=m;j-->0;)best[j]=alhazen(j?best[j-1].point:a,j+1<m?best[j+1].point:b,disks[run[j]]);
    double shortest=chain(best);
    auto offer=[&](const std::vector<Contact> &cs){const double v=chain(cs);if(v<=shortest){shortest=v;best=cs;}};
    // One active circle.
    std::vector<Contact> one(m),part;std::vector<size_t> members;
    for(size_t j=0;j<m;++j) {
        const Contact x=alhazen(a,b,disks[run[j]]);
        if(!x.reflect)continue;
        members.assign(run.begin(),run.begin()+long(j));
        bool met=place_straight(disks,a,x.point-a,members,part);
        std::copy(part.begin(),part.end(),one.begin());
        one[j]=x;
        members.assign(run.begin()+long(j)+1,run.end());
        met&=place_straight(disks,x.point,b-x.point,members,part);
        std::copy(part.begin(),part.end(),one.begin()+long(j)+1);
        if(met)offer(one);
    }
    // Joined.
    double value=INFINITY;Vector2 x{};size_t on=k,with=k;
    {
        double lo=0,hi=1;
        for(size_t i:run) {
            double l,h;
            if(!segment_interval(a,b-a,disks[i],l,h)){hi=-1;break;}
            lo=std::max(lo,l);hi=std::min(hi,h);
        }
        if(lo<=hi){x=a+(b-a)*(0.5*(lo+hi));value=distance(a,x)+distance(x,b);}
    }
    if(!std::isfinite(value)) {
        for(size_t i:run) {
            const Contact p=alhazen(a,b,disks[i]);
            const double v=distance(a,p.point)+distance(p.point,b);
            if(p.reflect&&v<value&&in_all(p.point,i,i)){value=v;x=p.point;on=i;with=k;}
        }
        for(size_t s=0;s<m;++s)for(size_t t=s+1;t<m;++t) {
            Vector2 corners[2];
            if(!crossings(disks[run[s]],disks[run[t]],corners[0],corners[1]))continue;
            for(const Vector2 &p:corners) {
                const double v=distance(a,p)+distance(p,b);
                if(v<value&&in_all(p,run[s],run[t])){value=v;x=p;on=run[s];with=run[t];}
            }
        }
    }
    if(std::isfinite(value)) {
        std::vector<Contact> joined(m);
        for(size_t j=0;j<m;++j)joined[j]={x,run[j]==on&&with==k,with<k&&(run[j]==on||run[j]==with)};
        bool optimal=true;
        if(with<k) {
            // The corner's dual (as in type_duals): l1 n1 + l2 n2 = u_out - u_in.
            const Vector2 n1=(x-disks[on].center)/disks[on].radius,n2=(x-disks[with].center)/disks[with].radius;
            const double la=distance(a,x),lb=distance(x,b),det=n1.x*n2.y-n1.y*n2.x;
            optimal=la>0&&lb>0&&std::abs(det)>1e-9;
            if(optimal) {
                const Vector2 w=(b-x)/lb-(x-a)/la;
                const double l1=(w.x*n2.y-w.y*n2.x)/det,l2=(n1.x*w.y-n1.y*w.x)/det;
                const Vector2 mid=(x-a)/la+n1*l1;
                optimal=l1>=-1e-12&&l2>=-1e-12&&mid.dot(mid)<=1+1e-12;
            }
        }
        if(optimal)offer(joined);
        else if(!(shortest<value)) {
            // Apart along the two circles: the first toward a, the second toward b.
            auto apart=[&](size_t i,Vector2 toward) {
                const ConvexDisk &d=disks[i];const Vector2 r=x-d.center;
                const double th=std::atan2(r.y,r.x),step=1e-3;
                const Vector2 p=d.center+polar(th+step)*d.radius,q=d.center+polar(th-step)*d.radius;
                return Contact{distance(p,toward)<distance(q,toward)?p:q,true,false};
            };
            for(size_t j=0;j<m;++j) {
                if(run[j]==on)joined[j]=apart(on,a);
                else if(run[j]==with)joined[j]=apart(with,b);
                else joined[j].corner=false;
            }
            best=joined;
        }
    }
    for(size_t j=0;j<m;++j)c[run[j]]=best[j];
}

// One Gauss-Seidel sweep: Alhazen updates of the single contacts, then one
// update of each block (maximal runs of coincident contacts of disks, found
// again after the single updates, which can make contacts meet; points never
// move, so they need none). A whole cycle at one point is a common point of
// every disk: its single updates keep it.
void sweep(const std::vector<ConvexDisk> &disks,std::vector<Contact> &c,double scale) {
    const size_t k=c.size();
    std::vector<bool> joined(k,false); // contact i coincides with contact i+1
    auto runs=[&] {
        if(k>=3)for(size_t i=0;i<k;++i) {
            const size_t j=(i+1)%k;
            joined[i]=disks[i].radius>0&&disks[j].radius>0&&distance(c[i].point,c[j].point)<=1e-12*scale;
        }
        for(size_t i=0;i<k;++i)if(!joined[(i+k-1)%k])return i;
        return k;
    };
    size_t start=runs();
    for(size_t i=0;i<k;++i)if(start==k||!(joined[i]||joined[(i+k-1)%k])) {
        c[i]=alhazen(c[(i+k-1)%k].point,c[(i+1)%k].point,disks[i]);
    }
    if((start=runs())==k)return;
    std::vector<size_t> run;
    for(size_t r=0;r<k;++r) {
        const size_t i=(start+r)%k;
        run.push_back(i);
        if(!joined[i]) {
            if(run.size()>1)block_update(disks,c,run);
            run.clear();
        }
    }
}

// A x = rhs for the symmetric cyclic tridiagonal A with diagonal `diag` and
// off[j] = A(j,j+1 mod n), n >= 3: Thomas' algorithm, with the corner by
// Sherman-Morrison. False if a pivot vanishes.
bool solve_cyclic(std::vector<double> diag,const std::vector<double> &off,std::vector<double> &rhs) {
    const size_t n=diag.size();
    auto thomas=[&](std::vector<double> d,std::vector<double> &x) {
        for(size_t j=1;j<n;++j) {
            if(!(std::abs(d[j-1])>0))return false;
            const double m=off[j-1]/d[j-1];
            d[j]-=m*off[j-1];x[j]-=m*x[j-1];
        }
        if(!(std::abs(d[n-1])>0))return false;
        x[n-1]/=d[n-1];
        for(size_t j=n-1;j-->0;)x[j]=(x[j]-off[j]*x[j+1])/d[j];
        return true;
    };
    const double corner=off[n-1];
    if(corner==0)return thomas(diag,rhs);
    const double gamma=-diag[0];
    if(!(std::abs(gamma)>0))return false;
    diag[0]-=gamma;diag[n-1]-=corner*corner/gamma;
    std::vector<double> z(n,0.0);z[0]=gamma;z[n-1]=corner;
    if(!thomas(diag,rhs)||!thomas(diag,z))return false;
    const double vx=rhs[0]+corner/gamma*rhs[n-1],vz=z[0]+corner/gamma*z[n-1];
    if(!(std::abs(1+vz)>0))return false;
    for(size_t j=0;j<n;++j)rhs[j]-=z[j]*vx/(1+vz);
    return true;
}

// The bends of a type in cyclic order: reflections, corners and point
// regions. A corner's partner is the next bend, at the same point.
std::vector<size_t> bends(const std::vector<ConvexDisk> &disks,const std::vector<Contact> &c) {
    std::vector<size_t> out;
    for(size_t i=0;i<c.size();++i)if(disks[i].radius==0||c[i].reflect||c[i].corner)out.push_back(i);
    return out;
}

// The type fixed: straight contacts are dropped (they lie on the segments
// between bends), points and corners stay (a corner pair once), and each
// reflection moves on its circle by its angle. Newton on the length of the
// reduced cycle: its Hessian in the angles is cyclic tridiagonal, and the
// circle's curvature adds radius * lambda >= 0 to the diagonal at a reflection
// (lambda its multiplier), so near a stationary reflection it converges
// quadratically. When two bends meet, a reflection has reached a crossing of
// its circle with the next one's (they become a corner) or a fixed bend
// (where it stays, straight), and the new type is closed in turn. False if
// bends meet otherwise.
bool close_type(const std::vector<ConvexDisk> &disks,std::vector<Contact> &c,double scale,size_t &iterations,int depth=0) {
    std::vector<size_t> red;
    for(const size_t i:bends(disks,c))if(red.empty()||!(c[i].corner&&c[red.back()].corner&&distance(c[i].point,c[red.back()].point)<=1e-12*scale))red.push_back(i);
    if(red.size()>1&&c[red.front()].corner&&c[red.back()].corner&&distance(c[red.front()].point,c[red.back()].point)<=1e-12*scale)red.pop_back();
    const size_t n=red.size();
    if(n<2)return true;
    std::vector<double> theta(n,0.0);std::vector<bool> var(n);
    for(size_t j=0;j<n;++j) {
        var[j]=c[red[j]].reflect;
        if(var[j]){const auto r=c[red[j]].point-disks[red[j]].center;theta[j]=std::atan2(r.y,r.x);}
    }
    auto point=[&](size_t j,double th){const auto &d=disks[red[j]];return var[j]?d.center+polar(th)*d.radius:d.radius==0?d.center:c[red[j]].point;};
    auto length=[&](const std::vector<double> &th) {
        double s=0;
        for(size_t j=0;j<n;++j){const auto v=point((j+1)%n,th[(j+1)%n])-point(j,th[j]);s+=std::sqrt(v.dot(v));}
        return s;
    };
    std::vector<double> diag(n),off(n),grad(n),step(n),trial(n);
    double current=length(theta);
    for(int iteration=0;iteration<40;++iteration) {
        std::fill(diag.begin(),diag.end(),0.0);std::fill(off.begin(),off.end(),0.0);std::fill(grad.begin(),grad.end(),0.0);
        for(size_t j=0;j<n;++j) {
            const size_t b=(j+1)%n;
            const Vector2 v=point(b,theta[b])-point(j,theta[j]);
            const double len=std::sqrt(v.dot(v));
            if(!(len>1e-9*scale)) {
                if(depth>=8||!(var[j]||var[b]))return false;
                for(size_t m=0;m<n;++m)if(var[m])c[red[m]]={point(m,theta[m]),true};
                if(var[j]&&var[b]) {
                    Vector2 x[2];
                    if(!crossings(disks[red[j]],disks[red[b]],x[0],x[1]))return false;
                    const Vector2 mid=(c[red[j]].point+c[red[b]].point)*0.5;
                    const Vector2 corner=distance(x[0],mid)<=distance(x[1],mid)?x[0]:x[1];
                    c[red[j]]={corner,false,true};c[red[b]]={corner,false,true};
                } else if(var[j])c[red[j]]={c[red[b]].point,false,false};
                else c[red[b]]={c[red[j]].point,false,false};
                return close_type(disks,c,scale,iterations,depth+1);
            }
            const Vector2 u=v/len;
            auto hessian=[&](Vector2 x,Vector2 y){return (x.dot(y)-x.dot(u)*y.dot(u))/len;};
            auto tangent=[&](size_t m){const Vector2 q=polar(theta[m]);return Vector2{-q.y,q.x}*disks[red[m]].radius;};
            if(var[j]) {
                const Vector2 tj=tangent(j);
                grad[j]-=u.dot(tj);diag[j]+=hessian(tj,tj)+disks[red[j]].radius*u.dot(polar(theta[j]));
            }
            if(var[b]) {
                const Vector2 tb=tangent(b);
                grad[b]+=u.dot(tb);diag[b]+=hessian(tb,tb)-disks[red[b]].radius*u.dot(polar(theta[b]));
            }
            if(var[j]&&var[b])off[j]-=hessian(tangent(j),tangent(b));
        }
        double size=0;
        for(size_t j=0;j<n;++j)if(var[j])size=std::max(size,std::abs(grad[j]));
        if(size<=1e-15*scale)break;
        // Newton with a growing shift until it descends (far from the
        // optimum the reduced Hessian need not be positive definite).
        bool descent=false;
        for(double shift=0;shift<1e12&&!descent;shift=shift?shift*16:1e-12*scale) {
            std::vector<double> d=diag;
            for(size_t j=0;j<n;++j){d[j]=var[j]?d[j]+shift*(1+std::abs(d[j])):1;step[j]=var[j]?-grad[j]:0;}
            std::vector<double> o=off;
            for(size_t j=0;j<n;++j)if(!var[j]||!var[(j+1)%n])o[j]=0;
            bool solved;
            if(n==2) {
                const double a=d[0],bb=d[1],cc=o[0]+o[1],det=a*bb-cc*cc;
                solved=std::abs(det)>0;
                if(solved){const double s0=(bb*step[0]-cc*step[1])/det,s1=(a*step[1]-cc*step[0])/det;step[0]=s0;step[1]=s1;}
            } else solved=solve_cyclic(d,o,step);
            if(!solved)continue;
            double slope=0;for(size_t j=0;j<n;++j)slope+=grad[j]*step[j];
            descent=std::isfinite(slope)&&slope<0;
        }
        if(!descent)return false;
        double alpha=1,after=current;
        for(;alpha>1e-10;alpha*=.5) {
            for(size_t j=0;j<n;++j)trial[j]=theta[j]+alpha*step[j];
            after=length(trial);
            if(after<=current)break;
        }
        ++iterations;
        if(!(after<current)){break;} // below the length's resolution
        theta.swap(trial);current=after;
    }
    for(size_t j=0;j<n;++j)if(var[j])c[red[j]]={point(j,theta[j]),true};
    // Newton converges slowly toward a corner, where the reduced cycle is not
    // smooth: two consecutive reflections whose circles cross are merged at
    // the crossing if that shortens the reduced cycle.
    if(depth<8&&n>=3)for(size_t j=0;j<n;++j) {
        const size_t b=(j+1)%n,before=(j+n-1)%n,after=(b+1)%n;
        if(!var[j]||!var[b])continue;
        Vector2 x[2];
        if(!crossings(disks[red[j]],disks[red[b]],x[0],x[1]))continue;
        const Vector2 p=point(before,theta[before]),q=point(after,theta[after]);
        const double now=distance(p,c[red[j]].point)+distance(c[red[j]].point,c[red[b]].point)+distance(c[red[b]].point,q);
        const Vector2 corner=distance(p,x[0])+distance(x[0],q)<=distance(p,x[1])+distance(x[1],q)?x[0]:x[1];
        if(!(distance(p,corner)+distance(corner,q)<now-1e-15*now))continue;
        c[red[j]]={corner,false,true};c[red[b]]={corner,false,true};
        return close_type(disks,c,scale,iterations,depth+1);
    }
    // Straight passages onto their reduced segments (one a segment misses
    // goes to the point closest to its center, and the sweep that follows
    // makes it reflect).
    const size_t k=c.size();
    std::vector<size_t> members;std::vector<Contact> placed;
    for(size_t j=0;j<n;++j) {
        const Vector2 p=c[red[j]].point,e=c[red[(j+1)%n]].point-p;
        members.clear();
        for(size_t l=(red[j]+1)%k;l!=red[(j+1)%n];l=(l+1)%k)if(disks[l].radius>0&&!c[l].reflect&&!c[l].corner)members.push_back(l);
        place_straight(disks,p,e,members,placed);
        for(size_t q=0;q<members.size();++q)c[members[q]]=placed[q];
    }
    return true;
}

// Dual directions of a type: every link takes the direction of the segment
// between the bends around it, so straight passages bend by exactly zero and,
// at a closed type, D(u) equals the reduced length up to rounding. Between
// the two disks of a corner the link has length zero; its direction splits
// the bend: u_mid = u_in + l1 n1 with l1 n1 + l2 n2 = u_out - u_in for the
// outward normals n of the two circles there (the multipliers are >= 0, and
// |u_mid| <= 1, exactly when the corner is optimal).
std::vector<Vector2> type_duals(const std::vector<ConvexDisk> &disks,const std::vector<Contact> &c) {
    const size_t k=c.size();
    std::vector<Vector2> u(k);
    const auto red=bends(disks,c);
    const size_t n=red.size();
    if(n<2)return u;
    std::vector<Vector2> dir(n);std::vector<bool> zero(n,false);
    for(size_t j=0;j<n;++j) {
        const size_t a=red[j],b=red[(j+1)%n];
        const Vector2 v=c[b].point-c[a].point;const double len=std::sqrt(v.dot(v));
        zero[j]=!(len>0);dir[j]=zero[j]?Vector2{}:v/len;
    }
    for(size_t j=0;j<n;++j) {
        const size_t a=red[j],b=red[(j+1)%n];
        if(!zero[j]||!c[a].corner||!c[b].corner||zero[(j+n-1)%n]||zero[(j+1)%n])continue;
        const Vector2 in=dir[(j+n-1)%n],out=dir[(j+1)%n];
        const Vector2 n1=(c[a].point-disks[a].center)/disks[a].radius,n2=(c[b].point-disks[b].center)/disks[b].radius;
        const double det=n1.x*n2.y-n1.y*n2.x;
        if(!(std::abs(det)>1e-12))continue;
        const Vector2 w=out-in;
        dir[j]=in+n1*((w.x*n2.y-w.y*n2.x)/det);
    }
    for(size_t j=0;j<n;++j) {
        const size_t a=red[j],b=red[(j+1)%n];
        for(size_t l=a;;l=(l+1)%k){u[l]=dir[j];if((l+1)%k==b)break;}
    }
    return u;
}
}

DiskCycleResult tpp_convex_solve_cycle_disks(const std::vector<ConvexDisk> &disks,const DiskCycleOptions &options) {
    DiskCycleResult result;
    const size_t k=disks.size();
    if(!k||std::isnan(options.lower_bound_cutoff))return result;
    for(const auto &d:disks)if(!d.center.is_finite()||!std::isfinite(d.radius)||d.radius<0)return result;
    if(!options.initial_contacts.empty()&&options.initial_contacts.size()!=k)return result;
    if(!detail::cycle_interval_environment())return result;

    // Both bounds are proved on the input; each level of the polish only proposes.
    auto certify=[&](const std::vector<Vector2> &points,const std::vector<Vector2> &duals) {
        std::vector<Vector2> contacts(k);
        for(size_t i=0;i<k;++i)contacts[i]=inside_contact(points[i],disks[i]);
        CycleInterval upper{};
        if(k>1)for(size_t i=0;i<k;++i)upper=upper+link_length(contacts[i],contacts[(i+1)%k]);
        if(upper.hi<result.upper_bound){result.upper_bound=upper.hi;result.contacts=std::move(contacts);}
        if(duals.size()==k) {
            Vector2 r{};for(const auto &d:disks)r+=d.center/double(k);
            std::vector<Vector2> u(k);
            for(size_t l=0;l<k;++l)u[l]=unit_disk(duals[l]);
            CycleInterval lower{};
            for(size_t i=0;i<k;++i) {
                const auto &in=u[(i+k-1)%k],&out=u[i];
                const auto wx=CycleInterval(in.x)-CycleInterval(out.x),wy=CycleInterval(in.y)-CycleInterval(out.y);
                const auto support=(CycleInterval(disks[i].center.x)-CycleInterval(r.x))*wx+
                                   (CycleInterval(disks[i].center.y)-CycleInterval(r.y))*wy-
                                   CycleInterval(disks[i].radius)*(wx.square()+wy.square()).sqrt();
                lower=lower+support;
            }
            if(std::isfinite(lower.lo))result.lower_bound=std::max(result.lower_bound,lower.lo);
        }
        return result.upper_bound-result.lower_bound<=options.max_gap||result.lower_bound>=options.lower_bound_cutoff;
    };
    if(k==1) {
        certify({disks[0].center},{});
        result.status=DiskCycleStatus::GapClosed;
        return result;
    }

    // Scaled coordinates x = ref + scale*y.
    Vector2 ref{};for(const auto &d:disks)ref+=d.center/double(k);
    double scale=0;
    for(const auto &d:disks)scale=std::max({scale,std::abs(d.center.x-ref.x)+d.radius,std::abs(d.center.y-ref.y)+d.radius});
    if(!(scale>0)||!std::isfinite(scale))scale=1;

    // Geometry first: sweeps of Alhazen updates fix a type, Newton on the
    // reflection law closes it, and a sweep checks it (a straight passage the
    // segment misses starts reflecting; a reflection whose segment crosses
    // its disk becomes straight). Each round is certified; the rounds stop
    // once the gap closes or the cycle stops shortening with the type stable
    // (a passage tangent to its disk may flip between straight and
    // reflecting without changing the length).
    std::vector<Vector2> construction_points;
    auto done=[&] {
        result.construction_closed=true;
        result.lower_bound=std::max(result.lower_bound,0.0);
        result.status=result.upper_bound-result.lower_bound<=options.max_gap?DiskCycleStatus::GapClosed:DiskCycleStatus::CertifiedBound;
        return result;
    };
    if(options.construction) {
        std::vector<Contact> c(k);
        const bool warm=!options.initial_contacts.empty();
        for(size_t i=0;i<k;++i) {
            const auto &d=disks[i];
            Vector2 p=warm&&options.initial_contacts[i].is_finite()?options.initial_contacts[i]:d.center;
            const Vector2 r=p-d.center;
            c[i]={d.radius==0?d.center:p,d.radius>0&&r.dot(r)>=d.radius*d.radius*(1-1e-9)};
        }
        sweep(disks,c,scale);
        if(!warm)sweep(disks,c,scale);
        construction_points.resize(k);
        double gap=INFINITY;size_t stalled=0;
        for(size_t round=0;round<32&&stalled<3;++round) {
            ++result.construction_rounds;
            const bool closed=close_type(disks,c,scale,result.closing_iterations);
            sweep(disks,c,scale);
            for(size_t i=0;i<k;++i)construction_points[i]=c[i].point;
            if(certify(construction_points,type_duals(disks,c)))return done();
            if(!closed)break;
            const double now=result.upper_bound-result.lower_bound;
            stalled=now<0.5*gap?0:stalled+1;
            gap=std::min(gap,now);
        }
    }
    auto local=[&](Vector2 v){return (v-ref)/scale;};
    std::vector<detail::ChainNode> nodes(k);
    size_t barriers=0;
    for(size_t i=0;i<k;++i) {
        auto &node=nodes[i];const auto &d=disks[i];
        const Vector2 c=local(d.center);
        if(d.radius==0){node.kind=detail::ChainNode::Kind::Fixed;node.offset=c;continue;}
        const double radius=d.radius/scale;
        node.kind=detail::ChainNode::Kind::Free;
        node.disk=detail::ChainNode::Disk{c,radius*radius};
        node.y=c;
        // Warm start: the construction's contacts if it ran, else the caller's.
        const auto &start=!construction_points.empty()?construction_points:options.initial_contacts;
        if(!start.empty()&&start[i].is_finite()) {
            const Vector2 y=c+(local(start[i])-c)*(1-0x1p-7);
            if((y-c).dot(y-c)<radius*radius*(1-0x1p-20))node.y=y;
        }
        ++barriers;
    }
    const bool warm=!options.initial_contacts.empty()||!construction_points.empty();
    const double mu_end=std::max(1e-15,options.max_gap/(2*scale*double(k+barriers)));
    // A warm start near the boundary needs a moderate first level: starting
    // far below the central path, Newton crawls (1e-5 to 1e-3 all work).
    const double mu_start=std::max(mu_end,options.mu_start>0?options.mu_start:warm?1e-4:0.1);
    const auto stats=detail::chain_interior_point(nodes,true,mu_start,mu_end,options.max_iterations,
        [&](const detail::ChainLevel &level) {
            std::vector<Vector2> points(k);
            for(size_t i=0;i<k;++i)points[i]=ref+level.points[i]*scale;
            return certify(points,level.duals);
        });
    if(stats){result.newton_iterations=stats->iterations;result.levels=stats->levels;}
    if(result.contacts.empty()) {
        std::vector<Vector2> centers(k);for(size_t i=0;i<k;++i)centers[i]=disks[i].center;
        certify(centers,{});
    }
    result.lower_bound=std::max(result.lower_bound,0.0);
    result.status=result.upper_bound-result.lower_bound<=options.max_gap?DiskCycleStatus::GapClosed:
        result.lower_bound>=options.lower_bound_cutoff?DiskCycleStatus::CertifiedBound:DiskCycleStatus::Limit;
    return result;
}
Vector2 tpp_convex_disk_detour_point(Vector2 a,Vector2 b,const ConvexDisk &disk) {
    return alhazen(a,b,disk).point;
}

// The point of the disk nearest the segment ab: a cheap stand-in for the
// detour point (no Newton), exact when the segment meets the disk.
Vector2 nearest_to_segment(Vector2 a,Vector2 b,const ConvexDisk &d) {
    const Vector2 e=b-a;const double ee=e.dot(e);
    const double t=ee>0?std::clamp((d.center-a).dot(e)/ee,0.0,1.0):0.0;
    const Vector2 foot=a+e*t,r=foot-d.center;const double rr=r.dot(r);
    if(rr<=d.radius*d.radius)return foot;
    return d.center+r*(d.radius/std::sqrt(rr));
}
std::vector<double> tpp_convex_disk_insertion_bounds(const std::vector<ConvexDisk> &disks,
        const std::vector<Vector2> &contacts,const ConvexDisk &inserted) {
    const size_t k=disks.size();
    if(!k||contacts.size()!=k||!inserted.center.is_finite()||!(inserted.radius>=0))return {};
    for(size_t i=0;i<k;++i)if(!contacts[i].is_finite()||!disks[i].center.is_finite()||!(disks[i].radius>=0))return {};
    if(!detail::cycle_interval_environment())return {};
    Vector2 r{};for(const auto &d:disks)r+=d.center/double(k);
    auto direction=[](Vector2 from,Vector2 to)->Vector2 {
        const Vector2 v=to-from;const double n=std::sqrt(v.dot(v));
        return n>0?unit_disk(v/n):Vector2{};
    };
    // Each link's direction; a zero link (coincident contacts) keeps the last
    // nonzero direction before it, or the corner split when two circles meet
    // there (as in type_duals).
    std::vector<Vector2> u(k);
    std::vector<bool> zero(k,false);
    for(size_t l=0;l<k;++l){u[l]=direction(contacts[l],contacts[(l+1)%k]);zero[l]=u[l].x==0&&u[l].y==0;}
    if(k>=3)for(size_t l=0;l<k;++l) {
        if(!zero[l])continue;
        size_t before=(l+k-1)%k,after=(l+1)%k;
        while(zero[before]&&before!=l)before=(before+k-1)%k;
        while(zero[after]&&after!=l)after=(after+1)%k;
        if(zero[before]||zero[after])continue;
        const Vector2 in=u[before],out=u[after];
        u[l]=in;
        const size_t i=l,j=(l+1)%k;
        if(!(disks[i].radius>0&&disks[j].radius>0))continue;
        const Vector2 ni=(contacts[i]-disks[i].center)/disks[i].radius,nj=(contacts[j]-disks[j].center)/disks[j].radius;
        const double det=ni.x*nj.y-ni.y*nj.x;
        if(!(std::abs(det)>1e-12)||!(std::abs(ni.dot(ni)-1)<1e-6)||!(std::abs(nj.dot(nj)-1)<1e-6))continue;
        const Vector2 w=out-in;
        const double li=(w.x*nj.y-w.y*nj.x)/det,lj=(ni.x*w.y-ni.y*w.x)/det;
        if(li>=0&&lj>=0)u[l]=unit_disk(in+ni*li);
    }
    auto support=[&](const ConvexDisk &d,Vector2 in,Vector2 out) {
        const auto wx=CycleInterval(in.x)-CycleInterval(out.x),wy=CycleInterval(in.y)-CycleInterval(out.y);
        return (CycleInterval(d.center.x)-CycleInterval(r.x))*wx+(CycleInterval(d.center.y)-CycleInterval(r.y))*wy-
               CycleInterval(d.radius)*(wx.square()+wy.square()).sqrt();
    };
    std::vector<double> out(k,0.0);
    if(k==1) {
        const Vector2 q=nearest_to_segment(contacts[0],contacts[0],inserted);
        const Vector2 there=direction(contacts[0],q),back=direction(q,contacts[0]);
        const auto bound=support(disks[0],back,there)+support(inserted,there,back);
        out[0]=std::isfinite(bound.lo)?std::max(0.0,bound.lo):0.0;
        return out;
    }
    std::vector<CycleInterval> term(k);
    CycleInterval total{};
    for(size_t i=0;i<k;++i){term[i]=support(disks[i],u[(i+k-1)%k],u[i]);total=total+term[i];}
    for(size_t p=1;p<=k;++p) {
        const size_t a=p-1,b=p%k;
        const Vector2 q=nearest_to_segment(contacts[a],contacts[b],inserted);
        Vector2 to=direction(contacts[a],q),from=direction(q,contacts[b]);
        if(to.x==0&&to.y==0)to=u[a];
        if(from.x==0&&from.y==0)from=u[a];
        const auto bound=total-term[a]-term[b]+support(disks[a],u[(a+k-1)%k],to)+support(inserted,to,from)+support(disks[b],from,u[b]);
        out[p-1]=std::isfinite(bound.lo)?std::max(0.0,bound.lo):0.0;
    }
    return out;
}
DiskExcess tpp_convex_disk_excess(const std::vector<ConvexDisk> &disks,const std::vector<int> &members,
        const std::vector<Vector2> &cycle,bool closed) {
    DiskExcess out{std::vector<double>(disks.size(),INFINITY),std::vector<double>(disks.size(),0)};
    std::vector<bool> member(disks.size(),false);for(int i:members)member[size_t(i)]=true;
    // Squared distances to each link, one square root per disk.
    const size_t k=cycle.size(),links=closed||k<2?k:k-1;
    std::vector<Vector2> edge(k);std::vector<double> inverse(k);
    for(size_t l=0;l<k;++l){edge[l]=cycle[(l+1)%k]-cycle[l];const double ee=edge[l].dot(edge[l]);inverse[l]=ee>0?1/ee:0;}
    // The coordinates any link involves: |center - cycle[l]| + |edge[l]| is at
    // most |center| + reach (1-norms).
    double reach=0,longest=0;
    for(size_t l=0;l<k;++l) {
        reach=std::max(reach,std::abs(cycle[l].x)+std::abs(cycle[l].y));
        longest=std::max(longest,std::abs(edge[l].x)+std::abs(edge[l].y));
    }
    reach+=longest;
    constexpr double ulp=std::numeric_limits<double>::epsilon();
    for(size_t j=0;j<disks.size();++j) {
        if(member[j]){out.excess[j]=-INFINITY;continue;}
        double best=INFINITY;
        for(size_t l=0;l<links;++l) {
            const Vector2 f=disks[j].center-cycle[l];
            const double t=std::clamp(f.dot(edge[l])*inverse[l],0.0,1.0);
            const Vector2 g=f-edge[l]*t;
            best=std::min(best,g.dot(g));
        }
        // For any t in [0,1], cycle[l]+edge*t is on the link, so the distance
        // to it bounds the distance to the polyline; computing it rounds f,
        // the edge, the product and the difference, each within an ulp of the
        // coordinates involved, and the square root within one of the result
        // (16 ulps is generous).
        const double d=std::sqrt(best);
        out.excess[j]=d-disks[j].radius;
        if(k)out.error[j]=16*ulp*(std::abs(disks[j].center.x)+std::abs(disks[j].center.y)+reach+d);
    }
    return out;
}
} // namespace tpp
