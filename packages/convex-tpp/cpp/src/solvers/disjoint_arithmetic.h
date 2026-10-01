#pragma once
#include "tpp/convex/rational.h"
#include "cycle_execution.h"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace tpp::detail {
// Scalar-generic form of the established disjoint binary-search recurrence.
// Rational and double cycles instantiate exactly the same geometric algorithm.
template<class Scalar> struct DisjointArithmetic {
using Point=ConvexArithmeticPoint<Scalar>;
using Polygon=std::vector<Point>;
struct Result { std::vector<Point> contacts, path; };
struct Cone {Point first,second;};

static bool same_direction(const Point &a,const Point &b){return a.cross(b)==0&&a.dot(b)>0;}
static Point reflect(const Point &p,const Point &edge){return edge*(2*p.dot(edge)/edge.dot(edge))-p;}
static Point reflect_point(const Point &p,const Point &a,const Point &edge){return a+reflect(p-a,edge);}
static bool in_cone(const Point &q,const Point &v,const Point &r1,const Point &r2) {
    if(same_direction(r1,r2))return same_direction(q-v,r1);
    const bool c1=r1.cross(q-v)>=0,c2=r2.cross(q-v)<=0;
    return r1.cross(r2)<0?c1||c2:c1&&c2;
}
static bool in_edge(const Point &q,const Point &v1,const Point &v2,const Point &r1,const Point &r2) {
    if(v1==v2)return in_cone(q,v1,r1,r2);
    const Point dv=v2-v1;
    if(same_direction(r1,dv)||same_direction(r2,-dv))return false;
    const Point p1=q-v1,p2=q-v2;
    if(dv.cross(r1)<0) {
        if(dv.cross(r2)<0)return r1.cross(p1)>=0&&r2.cross(p2)<=0&&dv.cross(p1)<=0;
        return dv.cross(p1)<0?r1.cross(p1)>=0:r2.cross(p2)<=0;
    }
    if(dv.cross(r2)<0)return dv.cross(p2)<0?r2.cross(p2)<=0:r1.cross(p1)>=0;
    return r1.cross(p1)>=0||r2.cross(p2)<=0||dv.cross(p1)<=0;
}
static bool inside(const Point &q,const Polygon &p) {
    // Strict polygons use the convex fan in O(log n). Keep closed collinear
    // behavior for the older fixed-source API's redundant boundary vertices.
    if((p[1]-p[0]).cross(p[2]-p[1])==0 ||
       (p[0]-p.back()).cross(p[1]-p[0])==0 ||
       (p.back()-p[p.size()-2]).cross(p[0]-p.back())==0) {
        for(size_t i=0;i<p.size();++i)if((p[(i+1)%p.size()]-p[i]).cross(q-p[i])<0)return false;
        return true;
    }
    const Point d=q-p[0],first=p[1]-p[0],last=p.back()-p[0];
    const Scalar a=first.cross(d),b=last.cross(d);
    if(a<0||b>0)return false;
    if(a==0)return d.dot(first)>=0&&d.dot(first)<=first.dot(first);
    if(b==0)return d.dot(last)>=0&&d.dot(last)<=last.dot(last);
    size_t lo=1,hi=p.size()-1;
    while(hi-lo>1){const size_t mid=lo+(hi-lo)/2;if((p[mid]-p[0]).cross(d)>=0)lo=mid;else hi=mid;}
    return (p[hi]-p[lo]).cross(q-p[lo])>=0;
}
static void append(std::vector<Point> &path,const Point &p){if(path.empty()||!(path.back()==p))path.push_back(p);}

class Solver {
    Point start,target;std::vector<Polygon> polygons;
    std::vector<std::vector<std::optional<Cone>>> cones;
    std::vector<std::vector<bool>> first_contact;
    std::vector<std::optional<Point>> bends;

    Point query(const Point &q,size_t level) {
        cycle_checkpoint();
        if(level==0)return start;
        if(inside(q,polygons[level-1]))return query(q,level-1);
        const auto location=locate(q,level);
        if(location<0)return query(q,level-1);
        const auto &p=polygons[level-1];const size_t j=size_t(location)/2;
        if(location%2==0)return p[j];
        const Point edge=p[(j+1)%p.size()]-p[j];
        const Point reflected=reflect_point(q,p[j],edge);
        const Point previous=query(reflected,level-1);
        const Point direction=reflected-previous;
        const Scalar denominator=edge.cross(direction);
        if(denominator==0)throw std::runtime_error("Disjoint reflection is parallel to edge");
        return previous+direction*(edge.cross(p[j]-previous)/denominator);
    }
    Cone &cone(size_t i,size_t j) {
        if(cones[i][j])return *cones[i][j];
        const auto &p=polygons[i];const size_t before=(j+p.size()-1)%p.size(),after=(j+1)%p.size();
        const Point incoming=p[j]-query(p[j],i);
        if(incoming.zero())throw std::runtime_error("Disjoint cone has zero incoming direction");
        first_contact[i][before]=incoming.cross(p[j]-p[before])<0;
        first_contact[i][j]=incoming.cross(p[j]-p[after])>0;
        Point r1=first_contact[i][before]?reflect(incoming,p[j]-p[before]):incoming;
        Point r2=first_contact[i][j]?reflect(incoming,p[j]-p[after]):incoming;
        return *(cones[i][j]=Cone{r1,r2});
    }
    long long locate(const Point &q,size_t level) {
        const size_t i=level-1,n=polygons[i].size();const auto &p=polygons[i];
        if(inside(q,p))return -1;
        auto region=[&](size_t j){auto &c=cone(i,j);return in_cone(q,p[j],c.first,c.second);};
        size_t location=0;
        if(region(0))location=0;
        else {
            size_t left=0,right=n-1;
            while(left!=right) {
                const size_t mid=left+(right-left)/2,j=mid+1;
                if(region(j)){location=2*j;goto classified;}
                if(in_edge(q,p[left],p[j],cone(i,left).second,cone(i,j).first))right=mid;
                else left=mid+1;
            }
            location=2*left+1;
        }
classified:
        const size_t previous=location==0?n-1:(location-1)/2;
    return first_contact[i][location / 2] || first_contact[i][previous]
               ? static_cast<long long>(location)
               : -1;
    }
    void path_to(const Point &q,size_t level,std::vector<Point> &path) {
        cycle_checkpoint();
        if(level==0){append(path,start);append(path,q);return;}
        if(inside(q,polygons[level-1])){path_to(q,level-1,path);return;}
        const auto location=locate(q,level);
        if(location<0){path_to(q,level-1,path);return;}
        const auto &p=polygons[level-1];const size_t j=size_t(location)/2;
        if(location%2==0){path_to(p[j],level-1,path);bends[level-1]=p[j];append(path,q);return;}
        const Point edge=p[(j+1)%p.size()]-p[j],reflected=reflect_point(q,p[j],edge);
        path_to(reflected,level-1,path);
        if(path.size()<2)throw std::runtime_error("Disjoint reflection has no incoming segment");
        const Point previous=path[path.size()-2],direction=reflected-previous;
        const Scalar denominator=edge.cross(direction);
        if(denominator==0)throw std::runtime_error("Disjoint refolding is parallel to edge");
        const Scalar t=edge.cross(p[j]-previous)/denominator;
        const Point contact=previous+direction*t;
        const Scalar u=(contact-p[j]).dot(edge)/edge.dot(edge);
        if(t<0||t>1||u<0||u>1)throw std::runtime_error("Disjoint refolding leaves finite edge");
        bends[level-1]=contact;
        path.pop_back();append(path,contact);append(path,q);
    }
    static bool clip(const Point &a,const Point &b,const Polygon &p,Scalar floor,Scalar &lo,Scalar &hi) {
        const Point d=b-a;lo=floor;hi=1;
        // Materialization asks for the last contact. If the segment endpoint
        // already belongs to the region, it is that contact. This equivalent
        // shortcut avoids computing two rounded, inconsistent parameters at a
        // vertex (lo slightly above hi) in the double instantiation.
        if(inside(b,p))return floor<=1;
        for(size_t i=0;i<p.size();++i){const Point e=p[(i+1)%p.size()]-p[i];
            const Scalar c=e.cross(a-p[i]),s=e.cross(d);
            if(s>0){const Scalar t=-c/s;if(t>lo)lo=t;}
            else if(s<0){const Scalar t=-c/s;if(t<hi)hi=t;}
            else if(c<0)return false;}
        return lo<=hi&&hi>=floor&&lo<=1;
    }

public:
    Solver(const Point &s,const Point &t,const std::vector<Polygon> &input):start(s),target(t) {
        for(const auto &source:input){Polygon p;for(auto v:source){Point q(v);if(p.empty()||!(p.back()==q))p.push_back(q);}
            if(p.size()>1&&p.front()==p.back())p.pop_back();Scalar area=0;for(size_t i=0;i<p.size();++i)area+=p[i].cross(p[(i+1)%p.size()]);
            if(p.size()<3||area==0)throw std::invalid_argument("Disjoint polygon must have positive area");if(area<0)std::reverse(p.begin(),p.end());polygons.push_back(std::move(p));}
        bends.resize(polygons.size());cones.resize(polygons.size());first_contact.resize(polygons.size());for(size_t i=0;i<polygons.size();++i){cones[i].resize(polygons[i].size());first_contact[i].resize(polygons[i].size());}
    }
    Result solve(bool preserve_bend_contacts = false) {
        std::vector<Point> path;path_to(target,polygons.size(),path);
        Result result;result.contacts.reserve(polygons.size());size_t segment=path.size()==1?0:1;Scalar rate=0;
        for(size_t index=0;index<polygons.size();++index){const auto &p=polygons[index];
            // A reflected/vertex contact is already constructed by the map.
            // Re-clipping it would lose provenance and can reject a rounded
            // double endpoint lying just outside its supporting halfplane.
            if(preserve_bend_contacts && bends[index]) {
                while(segment<path.size()&&!(path[segment]==*bends[index]))++segment;
                if(segment==path.size())throw std::runtime_error("Disjoint bend missing from path");
                result.contacts.push_back(*bends[index]);rate=1;continue;
            }
            if(path.size()==1){if(!inside(path.front(),p))throw std::runtime_error("Stationary rational disjoint path misses polygon");result.contacts.push_back(path.front());continue;}
            bool found=false;while(segment<path.size()){Scalar lo,hi;if(clip(path[segment-1],path[segment],p,rate,lo,hi)){rate=std::min(hi,Scalar(1));result.contacts.push_back(path[segment-1]+(path[segment]-path[segment-1])*rate);found=true;break;}++segment;rate=0;}
            if(!found)throw std::runtime_error("Disjoint path contact missing at polygon "+std::to_string(index));}
        result.path=std::move(path);return result;
    }
};
};
} // namespace tpp::detail
