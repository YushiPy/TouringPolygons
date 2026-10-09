#pragma once

#include "tpp/geometry/vec2.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace tpp::detail {
// Log-barrier Newton method for a chain of contacts in binary64, shared by the
// fixed-endpoint path polish and the cycle polish. It only proposes contacts
// and dual vectors: every bound that leaves the oracle is proved afterwards on
// the original polygons, so nothing here is trusted.
struct ChainMatrix {double a=0,b=0,c=0,d=0;}; // [[a b] [c d]]
inline ChainMatrix operator+(const ChainMatrix &m,const ChainMatrix &h) {return {m.a+h.a,m.b+h.b,m.c+h.c,m.d+h.d};}
inline ChainMatrix operator-(const ChainMatrix &m,const ChainMatrix &h) {return {m.a-h.a,m.b-h.b,m.c-h.c,m.d-h.d};}
inline ChainMatrix operator*(const ChainMatrix &m,const ChainMatrix &h) {
    return {m.a*h.a+m.b*h.c,m.a*h.b+m.b*h.d,m.c*h.a+m.d*h.c,m.c*h.b+m.d*h.d};
}
inline Vector2 operator*(const ChainMatrix &m,Vector2 v) {return {m.a*v.x+m.b*v.y,m.c*v.x+m.d*v.y};}
inline ChainMatrix transpose(const ChainMatrix &m) {return {m.a,m.c,m.b,m.d};}
inline std::optional<ChainMatrix> inverse(const ChainMatrix &m) {
    const double det=m.a*m.d-m.b*m.c;
    if(!(std::abs(det)>0)||!std::isfinite(det))return {};
    return ChainMatrix{m.d/det,-m.b/det,-m.c/det,m.a/det};
}

// Block tridiagonal symmetric system: diagonal blocks and the upper blocks
// H(i,i+1). Block elimination without pivoting; the barrier keeps H positive
// definite, so every leading principal submatrix is too.
struct ChainTridiagonal {
    std::vector<ChainMatrix> diagonal,upper,factor,inverses;
    bool factorize(size_t n) {
        factor.resize(diagonal.size());inverses.resize(diagonal.size());
        for(size_t i=0;i<n;++i) {
            if(i) {
                const auto lower=transpose(upper[i-1]);
                factor[i]=lower*inverses[i-1];
                diagonal[i]=diagonal[i]-factor[i]*upper[i-1];
            }
            const auto inv=inverse(diagonal[i]);
            if(!inv)return false;
            inverses[i]=*inv;
        }
        return true;
    }
    // Overwrites rhs with the solution (forward elimination in place).
    void solve(std::vector<Vector2> &rhs,size_t n) const {
        for(size_t i=1;i<n;++i)rhs[i]-=factor[i]*rhs[i-1];
        for(size_t i=n;i-->0;)rhs[i]=inverses[i]*(rhs[i]-(i+1<n?upper[i]*rhs[i+1]:Vector2{}));
    }
};

// One region in scaled coordinates, x = offset + B y: a polygon is free
// (B = I), a segment has y.x = t in [0,1] along its edge, a point is fixed.
// Faces n.y >= o live in y-space; unused components of y get a unit Hessian
// and a zero gradient, so their Newton step is zero.
struct ChainFace {Vector2 normal;double offset;};
struct ChainNode {
    enum class Kind {Fixed,Segment,Free};
    Kind kind=Kind::Free;
    Vector2 offset,edge; // Fixed: x = offset; Segment: x = offset + y.x*edge; Free: x = y
    std::vector<ChainFace> faces;
    Vector2 y; // start, strictly inside its faces
    Vector2 point(Vector2 v) const {
        return kind==Kind::Free?v:kind==Kind::Fixed?offset:offset+edge*v.x;
    }
    // B^T v and B^T M B' for x = offset + B y; unused components of y are zero.
    Vector2 project(Vector2 v) const {
        return kind==Kind::Free?v:kind==Kind::Fixed?Vector2{}:Vector2{edge.dot(v),0};
    }
    friend ChainMatrix project(const ChainNode &row,const ChainMatrix &m,const ChainNode &col) {
        if(row.kind==Kind::Fixed||col.kind==Kind::Fixed)return {};
        if(row.kind==Kind::Free&&col.kind==Kind::Free)return m;
        if(row.kind==Kind::Free) { // m * [edge 0]
            return {m.a*col.edge.x+m.b*col.edge.y,0,m.c*col.edge.x+m.d*col.edge.y,0};
        }
        const Vector2 left{row.edge.x*m.a+row.edge.y*m.c,row.edge.x*m.b+row.edge.y*m.d}; // edge^T m
        if(col.kind==Kind::Free)return {left.x,left.y,0,0};
        return {left.x*col.edge.x+left.y*col.edge.y,0,0,0};
    }
};
struct ChainLevel {
    std::vector<Vector2> points,variables; // per node, scaled coordinates
    std::vector<Vector2> duals;            // per link: d/sqrt(|d|^2+mu^2)
    double mu=0;
};
struct ChainPolishStats {std::size_t iterations=0,levels=0;};

// Minimizes sum_links sqrt(|x_b-x_a|^2+mu^2) - mu sum log(n.y-o) for mu from
// mu_start down to mu_end (factor 10 per level). Links join consecutive nodes,
// and node n-1 to node 0 when cyclic. At a stationary point the smoothed
// directions satisfy B_i^T(u_in-u_out) = sum (mu/slack) n_f; done() receives
// each level's point and returns true to stop.
//
// Only free and segment nodes are variables. Their Hessian is block
// tridiagonal in chain order; a cycle with a fixed node is ordered from the
// node after it, so its closing link couples no first and last variable.
// Only a cycle without fixed nodes has that corner block, eliminated with the
// Schur complement of the last variable.
inline std::optional<ChainPolishStats> chain_interior_point(const std::vector<ChainNode> &nodes,bool cyclic,
        double mu_start,double mu_end,std::size_t max_iterations,const std::function<bool(const ChainLevel&)> &done) {
    using Kind=ChainNode::Kind;
    const size_t n=nodes.size();
    if(n<2)return {};
    const size_t links=cyclic?n:n-1;
    size_t first=0;
    if(cyclic)for(size_t i=0;i<n;++i)if(nodes[i].kind==Kind::Fixed){first=(i+1)%n;break;}
    std::vector<size_t> order;
    constexpr size_t none=std::numeric_limits<size_t>::max();
    std::vector<size_t> position(n,none);
    for(size_t r=0;r<n;++r) {
        const size_t i=(first+r)%n;
        if(nodes[i].kind!=Kind::Fixed){position[i]=order.size();order.push_back(i);}
    }
    const size_t m=order.size();
    const bool corner=cyclic&&m==n&&n>=3;
    std::vector<Vector2> y(n),trial(n),x(n);
    for(size_t i=0;i<n;++i)y[i]=nodes[i].y;
    auto slack=[&](const ChainFace &f,Vector2 v){return f.normal.dot(v)-f.offset;};
    for(size_t i=0;i<n;++i)for(const auto &f:nodes[i].faces)if(!(slack(f,y[i])>0))return {};
    auto objective=[&](const std::vector<Vector2> &z,double mu)->double {
        for(size_t i=0;i<n;++i)x[i]=nodes[i].point(z[i]);
        double value=0;
        for(size_t l=0;l<links;++l) {
            const auto d=x[(l+1)%n]-x[l];value+=std::sqrt(d.dot(d)+mu*mu);
        }
        for(size_t i=0;i<n;++i)for(const auto &f:nodes[i].faces) {
            const double s=slack(f,z[i]);
            if(!(s>0))return INFINITY;
            value-=mu*std::log(s);
        }
        return value;
    };
    ChainPolishStats stats;
    ChainTridiagonal system;
    system.diagonal.resize(m);system.upper.resize(m);
    std::vector<Vector2> gradient(m),step(m),z,w0,w1;
    if(corner){z.resize(m-1);w0.resize(m-1);w1.resize(m-1);}
    ChainLevel level;
    auto emit=[&](double mu) {
        level.mu=mu;level.variables=y;level.points.clear();level.duals.clear();
        for(size_t i=0;i<n;++i)level.points.push_back(nodes[i].point(y[i]));
        for(size_t l=0;l<links;++l) {
            const auto d=level.points[(l+1)%n]-level.points[l];
            level.duals.push_back(d/std::sqrt(d.dot(d)+mu*mu));
        }
        return done(level);
    };
    if(!m){emit(mu_end);++stats.levels;return stats;}
    double mu=mu_start;
    for(;;mu=std::max(mu_end,mu*.1)) {
        ++stats.levels;
        for(;stats.iterations<max_iterations;++stats.iterations) {
            for(size_t i=0;i<n;++i)x[i]=nodes[i].point(y[i]);
            std::fill(system.diagonal.begin(),system.diagonal.end(),ChainMatrix{});
            std::fill(system.upper.begin(),system.upper.end(),ChainMatrix{});
            std::fill(gradient.begin(),gradient.end(),Vector2{});
            ChainMatrix closing{}; // H(order[0],order[m-1])
            for(size_t l=0;l<links;++l) {
                const size_t a=l,b=(l+1)%n;
                const auto d=x[b]-x[a];
                const double norm=std::sqrt(d.dot(d)+mu*mu);
                const Vector2 g=d/norm;
                const ChainMatrix h{(1-g.x*g.x)/norm,-g.x*g.y/norm,-g.x*g.y/norm,(1-g.y*g.y)/norm};
                const size_t pa=position[a],pb=position[b];
                if(pa!=none){gradient[pa]-=nodes[a].project(g);system.diagonal[pa]=system.diagonal[pa]+project(nodes[a],h,nodes[a]);}
                if(pb!=none){gradient[pb]+=nodes[b].project(g);system.diagonal[pb]=system.diagonal[pb]+project(nodes[b],h,nodes[b]);}
                if(pa==none||pb==none)continue;
                const ChainMatrix coupling=ChainMatrix{}-h;
                if(pb==pa+1)system.upper[pa]=system.upper[pa]+project(nodes[a],coupling,nodes[b]);
                else if(pa==pb+1)system.upper[pb]=system.upper[pb]+project(nodes[b],coupling,nodes[a]);
                else closing=closing+project(nodes[order[0]],coupling,nodes[order[m-1]]);
            }
            for(size_t p=0;p<m;++p) {
                const size_t i=order[p];
                const auto &node=nodes[i];
                auto &block=system.diagonal[p];
                for(const auto &f:node.faces) {
                    const double s=slack(f,y[i]),w=mu/(s*s);
                    gradient[p]-=f.normal*(mu/s);
                    block=block+ChainMatrix{w*f.normal.x*f.normal.x,w*f.normal.x*f.normal.y,w*f.normal.x*f.normal.y,w*f.normal.y*f.normal.y};
                }
                block.a+=1e-14;block.d+=1e-14;
                if(node.kind==Kind::Segment)block.d+=1;
                step[p]=gradient[p]*-1.0;
            }
            bool singular=false;
            if(!corner) {
                if(!system.factorize(m))singular=true;
                else system.solve(step,m);
            } else {
                // T = leading m-1 blocks; b = last block column (H(0,m-1),
                // H(m-2,m-1)); solve T[z W] = [r b], then the 2x2 complement.
                const ChainMatrix last=system.diagonal[m-1],tail=system.upper[m-2];
                if(!system.factorize(m-1))singular=true;
                else {
                    std::copy(step.begin(),step.end()-1,z.begin());
                    std::fill(w0.begin(),w0.end(),Vector2{});std::fill(w1.begin(),w1.end(),Vector2{});
                    w0.front()+=Vector2{closing.a,closing.c};w1.front()+=Vector2{closing.b,closing.d};
                    w0.back()+=Vector2{tail.a,tail.c};w1.back()+=Vector2{tail.b,tail.d};
                    system.solve(z,m-1);system.solve(w0,m-1);system.solve(w1,m-1);
                    auto column=[&](size_t j){return ChainMatrix{w0[j].x,w1[j].x,w0[j].y,w1[j].y};};
                    const auto complement=last-transpose(closing)*column(0)-transpose(tail)*column(m-2);
                    const auto reduced=step[m-1]-transpose(closing)*z[0]-transpose(tail)*z[m-2];
                    const auto inv=inverse(complement);
                    if(!inv)singular=true;
                    else {
                        step[m-1]=*inv*reduced;
                        for(size_t j=0;j+1<m;++j)step[j]=z[j]-column(j)*step[m-1];
                    }
                }
            }
            if(singular)break;
            double slope=0;
            for(size_t p=0;p<m;++p)slope+=gradient[p].dot(step[p]);
            if(!std::isfinite(slope)||slope>=0||-slope<1e-10*mu)break;
            const double before=objective(y,mu);
            // Below the objective's resolution Armijo cannot see progress;
            // near convergence the feasible Newton step is then taken as is.
            const bool resolved=-slope>64*std::numeric_limits<double>::epsilon()*std::abs(before);
            double alpha=1;trial=y;
            for(;alpha>1e-12;alpha*=.5) {
                for(size_t p=0;p<m;++p)trial[order[p]]=y[order[p]]+step[p]*alpha;
                const double after=objective(trial,mu);
                if(resolved?after<=before+.01*alpha*slope:std::isfinite(after))break;
            }
            if(alpha<=1e-12)break;
            y.swap(trial);
        }
        if(emit(mu)||mu<=mu_end||stats.iterations>=max_iterations)return stats;
    }
}
}
