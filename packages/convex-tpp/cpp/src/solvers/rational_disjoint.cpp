#include "tpp/convex/detail/rational_disjoint.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tpp::detail {
namespace {
using Scalar=boost::multiprecision::cpp_rational;
struct Point {
    Scalar x=0,y=0;
    Point()=default;Point(Scalar a,Scalar b):x(std::move(a)),y(std::move(b)){}
    explicit Point(Vector2 p):x(p.x),y(p.y){}
    Point operator+(const Point &p)const{return{x+p.x,y+p.y};}
    Point operator-(const Point &p)const{return{x-p.x,y-p.y};}
    Point operator-()const{return{-x,-y};}
    Point operator*(const Scalar &s)const{return{x*s,y*s};}
    Scalar cross(const Point &p)const{return x*p.y-y*p.x;}
    Scalar dot(const Point &p)const{return x*p.x+y*p.y;}
    bool operator==(const Point &)const=default;
    bool zero()const{return x==0&&y==0;}
    Vector2 external()const{return{x.convert_to<double>(),y.convert_to<double>()};}
};
using Polygon=std::vector<Point>;
struct Cone {Point first,second;};

bool same_direction(const Point &a,const Point &b){return a.cross(b)==0&&a.dot(b)>0;}
Point reflect(const Point &p,const Point &edge){return edge*(2*p.dot(edge)/edge.dot(edge))-p;}
Point reflect_point(const Point &p,const Point &a,const Point &edge){return a+reflect(p-a,edge);}
bool in_cone(const Point &q,const Point &v,const Point &r1,const Point &r2) {
    if(same_direction(r1,r2))return same_direction(q-v,r1);
    const bool c1=r1.cross(q-v)>=0,c2=r2.cross(q-v)<=0;
    return r1.cross(r2)<0?c1||c2:c1&&c2;
}
bool in_edge(const Point &q,const Point &v1,const Point &v2,const Point &r1,const Point &r2) {
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
bool inside(const Point &q,const Polygon &p) {
    for(size_t i=0;i<p.size();++i)if((p[(i+1)%p.size()]-p[i]).cross(q-p[i])<0)return false;
    return true;
}
void append(std::vector<Point> &path,const Point &p){if(path.empty()||!(path.back()==p))path.push_back(p);}

class Solver {
    Point start,target;std::vector<Polygon> polygons;
    std::vector<std::vector<std::optional<Cone>>> cones;
    std::vector<std::vector<bool>> first_contact;

    Point query(const Point &q,size_t level) {
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
        if(denominator==0)throw std::runtime_error("Rational disjoint reflection is parallel to edge");
        return previous+direction*(edge.cross(p[j]-previous)/denominator);
    }
    Cone &cone(size_t i,size_t j) {
        if(cones[i][j])return *cones[i][j];
        const auto &p=polygons[i];const size_t before=(j+p.size()-1)%p.size(),after=(j+1)%p.size();
        const Point incoming=p[j]-query(p[j],i);
        if(incoming.zero())throw std::runtime_error("Rational disjoint cone has zero incoming direction");
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
        if(level==0){append(path,start);append(path,q);return;}
        if(inside(q,polygons[level-1])){path_to(q,level-1,path);return;}
        const auto location=locate(q,level);
        if(location<0){path_to(q,level-1,path);return;}
        const auto &p=polygons[level-1];const size_t j=size_t(location)/2;
        if(location%2==0){path_to(p[j],level-1,path);append(path,q);return;}
        const Point edge=p[(j+1)%p.size()]-p[j],reflected=reflect_point(q,p[j],edge);
        path_to(reflected,level-1,path);
        if(path.size()<2)throw std::runtime_error("Rational disjoint reflection has no incoming segment");
        const Point previous=path[path.size()-2],direction=reflected-previous;
        const Scalar denominator=edge.cross(direction);
        if(denominator==0)throw std::runtime_error("Rational disjoint refolding is parallel to edge");
        const Scalar t=edge.cross(p[j]-previous)/denominator;
        const Point contact=previous+direction*t;
        const Scalar u=(contact-p[j]).dot(edge)/edge.dot(edge);
        if(t<0||t>1||u<0||u>1)throw std::runtime_error("Rational disjoint refolding leaves finite edge");
        path.pop_back();append(path,contact);append(path,q);
    }
    static bool clip(const Point &a,const Point &b,const Polygon &p,Scalar floor,Scalar &lo,Scalar &hi) {
        const Point d=b-a;lo=floor;hi=1;
        for(size_t i=0;i<p.size();++i){const Point e=p[(i+1)%p.size()]-p[i];
            const Scalar c=e.cross(a-p[i]),s=e.cross(d);
            if(s>0){const Scalar t=-c/s;if(t>lo)lo=t;}
            else if(s<0){const Scalar t=-c/s;if(t<hi)hi=t;}
            else if(c<0)return false;}
        return lo<=hi&&hi>=floor&&lo<=1;
    }
    static std::pair<double,double> bounds(const std::vector<Point> &path) {
        constexpr unsigned precision=96;const boost::multiprecision::cpp_int scale=boost::multiprecision::cpp_int(1)<<precision;
        Scalar lower=0,upper=0;
        for(size_t i=1;i<path.size();++i){const Point d=path[i]-path[i-1];const Scalar squared=d.dot(d);if(squared==0)continue;
            const auto numerator=boost::multiprecision::numerator(squared),denominator=boost::multiprecision::denominator(squared);
            const boost::multiprecision::cpp_int scaled=(numerator<<(2*precision))/denominator,root=sqrt(scaled);
            lower+=Scalar(root)/Scalar(scale);upper+=Scalar(root+1)/Scalar(scale);}
        double lo=lower.convert_to<double>();while(Scalar(lo)>lower)lo=std::nextafter(lo,-std::numeric_limits<double>::infinity());
        double hi=upper.convert_to<double>();while(Scalar(hi)<upper)hi=std::nextafter(hi,std::numeric_limits<double>::infinity());
        return {lo,hi};
    }
public:
    Solver(Vector2 s,Vector2 t,const std::vector<std::vector<Vector2>> &input):start(s),target(t) {
        for(const auto &source:input){Polygon p;for(auto v:source){Point q(v);if(p.empty()||!(p.back()==q))p.push_back(q);}
            if(p.size()>1&&p.front()==p.back())p.pop_back();Scalar area=0;for(size_t i=0;i<p.size();++i)area+=p[i].cross(p[(i+1)%p.size()]);
            if(p.size()<3||area==0)throw std::invalid_argument("Rational disjoint polygon must have positive area");if(area<0)std::reverse(p.begin(),p.end());polygons.push_back(std::move(p));}
        cones.resize(polygons.size());first_contact.resize(polygons.size());for(size_t i=0;i<polygons.size();++i){cones[i].resize(polygons[i].size());first_contact[i].resize(polygons[i].size());}
    }
    RationalDisjointResult solve() {
        std::vector<Point> path;path_to(target,polygons.size(),path);
        RationalDisjointResult result;result.contacts.reserve(polygons.size());size_t segment=path.size()==1?0:1;Scalar rate=0;
        for(const auto &p:polygons){if(path.size()==1){if(!inside(path.front(),p))throw std::runtime_error("Stationary rational disjoint path misses polygon");result.contacts.push_back(path.front().external());continue;}
            bool found=false;while(segment<path.size()){Scalar lo,hi;if(clip(path[segment-1],path[segment],p,rate,lo,hi)){rate=std::min(hi,Scalar(1));result.contacts.push_back((path[segment-1]+(path[segment]-path[segment-1])*rate).external());found=true;break;}++segment;rate=0;}
            if(!found)throw std::runtime_error("Rational disjoint path contact missing");}
        std::tie(result.lower_bound,result.upper_bound)=bounds(path);return result;
    }
};
}

RationalDisjointResult solve_rational_disjoint(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons){return Solver(start,target,polygons).solve();}
}
