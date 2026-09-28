#include "tpp_convex.h"

#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using Polygon = std::vector<Vector2>;
using Polygons = std::vector<Polygon>;
using tpp::ConvexCycleStatus;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
Polygon box(double x0, double y0, double x1, double y1) {
    return {{x0,y0},{x1,y0},{x1,y1},{x0,y1}};
}
bool solved(const tpp::ConvexCycleResult &result) {
    return result.status == ConvexCycleStatus::Optimal;
}
tpp::ConvexRationalPolygons exact(const Polygons &polygons) {
    tpp::ConvexRationalPolygons result;
    for(const auto &p:polygons) {
        tpp::ConvexRationalPolygon q;for(auto v:p)q.emplace_back(v);result.push_back(q);
    }
    return result;
}
double largest_double_gap=0,largest_double_difference=0;
std::size_t compared_double_cases=0;
std::size_t total_anchor_recoveries=0;
tpp::ConvexCycleDoubleResult compare_double(const Polygons &polygons,const tpp::ConvexCycleResult &exact_result,
                                           const std::string &name,bool recovery=true) {
    const auto result=tpp::tpp_convex_solve_cycle_disjoint_double(polygons,{recovery});
    require(result.status==ConvexCycleStatus::Optimal||result.status==ConvexCycleStatus::FloatingPointLimit,
            name+": double failed: "+result.diagnostic);
    const auto certificate=tpp::tpp_convex_verify_cycle_certificate(polygons,result.contacts);
    require(certificate.status==tpp::ConvexCycleCertificateStatus::Optimal||
            certificate.status==tpp::ConvexCycleCertificateStatus::Feasible,name+": double exactly feasible");
    require(result.certificate.upper_bound==certificate.upper_bound,name+": double certificate matches output");
    require(certificate.lower_bound<=exact_result.certificate.upper_bound &&
            exact_result.certificate.lower_bound<=certificate.upper_bound,name+": exact/double intervals overlap");
    if(result.status==ConvexCycleStatus::Optimal)
        require(certificate.status==tpp::ConvexCycleCertificateStatus::Optimal,name+": no approximate Optimal status");
    const double difference=std::abs(certificate.upper_bound-exact_result.certificate.upper_bound);
    const double scale=std::max(1.0,exact_result.certificate.upper_bound);
    // Regression allowance measured in binary64 rounding units; never passed
    // to either solver and never used to declare a solution optimal.
    require(difference<=128*polygons.size()*std::numeric_limits<double>::epsilon()*scale,
            name+": double objective differs beyond rounding error");
    largest_double_gap=std::max(largest_double_gap,certificate.upper_bound-certificate.lower_bound);
    largest_double_difference=std::max(largest_double_difference,difference);
    ++compared_double_cases;
    total_anchor_recoveries+=result.rational_anchor_recoveries;
    if(!recovery)require(result.rational_anchor_recoveries==0,"strict double never calls rational recurrence");
    return result;
}
tpp::ConvexCycleResult check(const Polygons &polygons, const std::string &name) {
    const auto result = tpp::tpp_convex_solve_cycle_disjoint(polygons);
    require(solved(result), name + ": uncertified result, status=" + std::to_string(int(result.status))+": "+result.diagnostic);
    const auto certificate = tpp::tpp_convex_verify_cycle_certificate(exact(polygons), result.contacts);
    require(certificate.status == tpp::ConvexCycleCertificateStatus::Optimal, name + ": exact optimality");
    require(result.contacts.size() == polygons.size(), name + ": one ordered contact per polygon");
    require(certificate.upper_bound == result.certificate.upper_bound, name + ": incumbent upper bound");
    require(result.certificate.lower_bound <= result.certificate.upper_bound, name + ": ordered bounds");
    require(result.certificate_checks == result.oracle_calls, name + ": every candidate checked");
    require(result.squared_link_lengths.size()==polygons.size(),name+": exact objective representation");
    for(std::size_t i=0;i<polygons.size();++i) {
        const auto d=result.contacts[(i+1)%polygons.size()]-result.contacts[i];
        require(result.squared_link_lengths[i]==d.dot(d),name+": exact squared link");
    }
    compare_double(polygons,result,name);
    return result;
}
void known_cycles() {
    for (std::size_t k=2;k<=5;++k) {
        Polygons polygons;
        for (std::size_t i=0;i<k;++i) polygons.push_back(box(3*i,-1,3*i+1,1));
        // Chosen anchor lies between the extreme polygons, so it can be a
        // straight-through contact rather than a bend.
        if (k>2) std::rotate(polygons.begin(),polygons.begin()+1,polygons.end());
        const auto result=check(polygons,"collinear "+std::to_string(k));
        const double expected=2*(3*double(k-1)-1);
        require(result.certificate.lower_bound<=expected && result.certificate.upper_bound>=expected,"known round trip enclosed");
        require(result.certificate.upper_bound-expected<1e-8,"known round trip objective");
    }
    const Polygons crossing{box(-4,-4,-3,-3),box(3,3,4,4),box(-4,3,-3,4),box(3,-4,4,-3)};
    const auto cross=check(crossing,"self-intersecting cycle");
    const double expected=12+12*std::sqrt(2.0);
    require(std::abs(cross.certificate.upper_bound-expected)<1e-8,"crossing cycle retains fixed order");
    const auto &q=cross.contacts;
    require((q[1]-q[0]).cross(q[2]-q[0])*(q[1]-q[0]).cross(q[3]-q[0])<0 &&
            (q[3]-q[2]).cross(q[0]-q[2])*(q[3]-q[2]).cross(q[1]-q[2])<0,"cycle really crosses");

    // The closest point is in the interior of a slanted anchor edge, not at a
    // vertex. Its non-dyadic contact must remain rational in the exact result.
    const Polygons edge{{{0,0},{5,2},{0,-3}},{{1,4},{2,6},{0,6}}};
    const auto reflected=check(edge,"oblique edge interior");
    const double edge_expected=36/std::sqrt(29.0);
    require(std::abs(reflected.certificate.upper_bound-edge_expected)<1e-8,"oblique edge objective");
    require(reflected.oracle_calls>0,"nonvertex construction exercised");
    require(std::none_of(edge[0].begin(),edge[0].end(),[&](auto v){return tpp::ConvexRationalPoint(v)==reflected.contacts[0];}),"nonvertex optimum");

    using R=tpp::ConvexRational;
    require(reflected.contacts[0]==tpp::ConvexRationalPoint(R(65)/29,R(26)/29),"exact non-dyadic contact");
    require(reflected.squared_link_lengths[0]==R(324)/29,"exact irrational objective representation");

    const Polygons narrow_pair{
        {{31.21,1.64},{34.26,16.72},{27.33,18.1},{24.29,3.02}},
        {{7.05,6.27},{10.04,21.09},{2.99,22.5},{0,7.68}}};
    const auto pair=check(narrow_pair,"near-parallel two-region projection");
    require(pair.oracle_calls==1,"two-region distance avoids anchored search");
    const auto general_pair=tpp::tpp_convex_solve_cycle(exact(narrow_pair));
    require(solved(general_pair)&&general_pair.oracle_calls==1,"general cycle shares two-region projection");
    const auto double_pair=tpp::tpp_convex_solve_cycle_double(narrow_pair);
    require(double_pair.status==ConvexCycleStatus::Optimal||double_pair.status==ConvexCycleStatus::FloatingPointLimit,
            "filtered two-region projection is certified");
    require(double_pair.rational_anchor_recoveries==0&&double_pair.rational_cycle_recoveries==0,
            "filtered two-region projection needs no complete rational search");

    const Polygons all_edges{box(1,-1,5,0),
        {{0.4,1},{1.4,3.5},{0.9,3.7},{-0.1,1.2}},
        {{5.2,1},{2.8,4},{3.3,4.4},{5.7,1.4}}};
    const auto floating=check(all_edges,"all-edge floating Fagnano cycle");
    for(std::size_t i=0;i<all_edges.size();++i)
        require(std::none_of(all_edges[i].begin(),all_edges[i].end(),[&](auto v){return tpp::ConvexRationalPoint(v)==floating.contacts[i];}),
                "floating optimum has no polygon vertex contact");

    // Exact rational input, beyond binary64's coordinate resolution. Translation must
    // preserve the certificate even when every rounded point would coincide.
    auto rational_input=exact(edge);
    const R offset=R(boost::multiprecision::cpp_int(1)<<80)+R(1)/7;
    for(auto &polygon:rational_input)for(auto &q:polygon){q.x+=offset;q.y+=offset;}
    const auto translated=tpp::tpp_convex_solve_cycle_disjoint(rational_input);
    require(translated.status==ConvexCycleStatus::Optimal,"rational input needs no floating geometry");
    require(translated.contacts[0]==tpp::ConvexRationalPoint(offset+R(65)/29,offset+R(26)/29),"exact translated contact");

    for(int exponent:{-1100,1100}) {
        const R scale=exponent>0?R(boost::multiprecision::cpp_int(1)<<exponent):
                                 R(1)/R(boost::multiprecision::cpp_int(1)<<(-exponent));
        auto scaled=exact(Polygons{box(0,0,1,1),box(3,0,4,1)});
        for(auto &polygon:scaled)for(auto &q:polygon)q=q*scale;
        const auto huge=tpp::tpp_convex_solve_cycle_disjoint(scaled);
        require(huge.status==ConvexCycleStatus::Optimal,"rational solve beyond floating exponent range");
        require(huge.squared_link_lengths[0]==4*scale*scale,"exact objective beyond floating exponent range");
        if(exponent<0)require(huge.certificate.lower_bound==0&&huge.certificate.upper_bound==std::numeric_limits<double>::denorm_min(),
                "reporting bounds scale down to the actual binary64 range");
        else require(std::isinf(huge.certificate.upper_bound)&&huge.certificate.lower_bound==std::numeric_limits<double>::max(),
                "reporting bounds enclose overflow without changing exact construction");
    }
    check({{{0,0},{0.5,0},{1,0},{1,1},{0,1},{0,0}},box(3,0,4,1)},"collinear and closing vertices");

}
void invalid_inputs() {
    const auto invalid=ConvexCycleStatus::InvalidInput, unsupported=ConvexCycleStatus::UnsupportedIntersection;
    auto status=[](const Polygons &p){return tpp::tpp_convex_solve_cycle_disjoint(p).status;};
    require(status({})==invalid && status({box(0,0,1,1)})==invalid,"k>=2 required");
    require(status({{},box(3,0,4,1)})==invalid,"empty polygon rejected");
    require(status({{{0,0},{1,0},{2,0}},box(3,0,4,1)})==invalid,"zero area rejected");
    require(status({{{0,0},{2,0},{1,0.5},{2,2},{0,2}},box(3,0,4,1)})==invalid,"concavity rejected");
    require(status({box(0,0,1,1),box(1,0,2,1)})==unsupported,"edge touching rejected");
    require(status({box(0,0,1,1),box(1,1,2,2)})==unsupported,"vertex touching rejected");
    require(status({box(0,0,2,2),box(1,1,3,3)})==unsupported,"overlap rejected");
    require(status({box(0,0,3,3),box(1,1,2,2)})==unsupported,"containment rejected");
    require(status({box(0,0,1,1),box(4,0,5,1),box(0.5,0.5,2,2),box(8,0,9,1)})==unsupported,
            "nonadjacent intersections rejected");
    require(status({{{0,3},{-2,-3},{3,1},{-3,1},{2,-3}},box(10,0,11,1)})==invalid,
            "consistent local turns do not make a multiply-wound star convex");
    auto p=Polygons{box(0,0,1,1),box(3,0,4,1)};
    p[0][0].x=std::numeric_limits<double>::infinity();
    require(status(p)==invalid,"nonfinite rejected");
}
Polygon points(const boost::property_tree::ptree &array) {
    Polygon result;
    for (const auto &[key,item]:array) {
        auto it=item.begin();const double x=it++->second.get_value<double>();
        result.push_back({x,it->second.get_value<double>()});
    }
    return result;
}
void references() {
    using boost::property_tree::ptree;
    ptree inputs,raw,config;
    const std::string directory=TPP_CYCLE_REFERENCE_DIR;
    read_json(directory+"/instances.json",inputs);
    read_json(directory+"/raw.json",raw);
    read_json(directory+"/config.json",config);
    std::cout<<"name,k,lower,upper,gap,gurobi_objective,error,oracle_calls,milliseconds\n";
    for (const auto &[key,instance]:inputs.get_child("instances")) {
        const auto name=instance.get<std::string>("name");Polygons p;
        for (const auto &[unused,polygon]:instance.get_child("polygons"))p.push_back(points(polygon));
        const ptree *reference=nullptr;
        for (const auto &[unused,row]:raw.get_child("instances"))
            if(row.get<std::string>("name")==name)reference=&row;
        require(reference && reference->get<int>("status")==2,"optimal Gurobi reference found");
        const double objective=reference->get<double>("objective"), bound=reference->get<double>("objective_bound");
        const auto began=std::chrono::steady_clock::now();
        const auto result=check(p,name);
        const auto strict=tpp::tpp_convex_solve_cycle_disjoint_double(p,{false});
        require(strict.rational_anchor_recoveries==0,"strict double never uses exact recovery");
        if(strict.status==ConvexCycleStatus::Optimal) {
            require(strict.certificate.status==tpp::ConvexCycleCertificateStatus::Optimal,"strict success certified");
            require(strict.certificate.lower_bound<=result.certificate.upper_bound &&
                    result.certificate.lower_bound<=strict.certificate.upper_bound,"strict exact result agrees");
        } else {
            require(strict.status==ConvexCycleStatus::FloatingPointLimit||strict.status==ConvexCycleStatus::OracleFailure,
                    "strict arithmetic failure is explicit");
            if(!strict.contacts.empty()) {
                const auto cert=tpp::tpp_convex_verify_cycle_certificate(p,strict.contacts);
                require(cert.status==tpp::ConvexCycleCertificateStatus::Feasible,"strict limited result is feasible");
                require(cert.lower_bound<=result.certificate.upper_bound&&result.certificate.lower_bound<=cert.upper_bound,
                        "strict certificate exposes its arithmetic error");
            }
        }
        std::cout<<"strict_double,"<<name<<",status="<<int(strict.status)<<",diagnostic="<<strict.diagnostic<<'\n';
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
        const auto reference_certificate=tpp::tpp_convex_verify_cycle_certificate(p,points(reference->get_child("feasible_contacts")));
        require(reference_certificate.status==tpp::ConvexCycleCertificateStatus::Feasible,"reference independently certified");
        require(result.certificate.lower_bound<=reference_certificate.upper_bound && reference_certificate.lower_bound<=result.certificate.upper_bound,
                name+": independent certified intervals overlap");
        // Gurobi's reported equality of objective and bound is numerical. The
        // saved config's intended tolerance is smaller than its observed error
        // on some records. Compare against the independently certified interval.
        const double reference_gap=reference_certificate.upper_bound-reference_certificate.lower_bound;
        const double tolerance=config.get<double>("objective_tolerance_for_future_comparisons.absolute")+
            config.get<double>("objective_tolerance_for_future_comparisons.relative")*std::abs(objective);
        require(std::abs(result.certificate.upper_bound-objective)<=reference_gap+tolerance,name+": Gurobi objective comparison");
        require(bound<=result.certificate.upper_bound+reference_gap+tolerance && objective>=result.certificate.lower_bound-tolerance,
                name+": numerical Gurobi bound comparison");
        require(std::abs(result.certificate.upper_bound-objective)<2e-6,name+": objective regression tolerance");
        std::cout<<std::setprecision(17)<<name<<','<<p.size()<<','<<result.certificate.lower_bound<<','<<result.certificate.upper_bound<<','
                 <<result.certificate.upper_bound-result.certificate.lower_bound<<','<<objective<<','<<result.certificate.upper_bound-objective<<','
                 <<result.oracle_calls<<','<<ms<<'\n';
        for (std::size_t shift=0;shift<p.size();++shift) {
            auto variant=p;
            std::rotate(variant.begin(),variant.begin()+shift,variant.end());
            if(shift%2)std::reverse(variant.begin(),variant.end());
            for (auto &polygon:variant) {
                std::reverse(polygon.begin(),polygon.end());
                polygon.push_back(polygon.front());
            }
            const auto transformed=check(variant,name+" rotated/reversed");
            require(std::abs(transformed.certificate.upper_bound-result.certificate.upper_bound)<2e-8,"cyclic/reversal invariance");
        }
    }
}
void random_cycles() {
    std::mt19937_64 random(93741);
    std::uniform_real_distribution<double> jitter(-0.3,0.3);
    for(std::size_t k=2;k<=5;++k)for(int sample=0;sample<10;++sample) {
        Polygons polygons;
        for(std::size_t i=0;i<k;++i) {
            const double angle=2*std::acos(-1.0)*i/k;
            const Vector2 c{6*std::cos(angle)+jitter(random),6*std::sin(angle)+jitter(random)};
            Polygon p;
            for(int j=0;j<3+int(i%3);++j) {
                const double direction=2*std::acos(-1.0)*j/(3+int(i%3))+0.2;
                p.push_back(c+Vector2{0.6*std::cos(direction),0.6*std::sin(direction)});
            }
            polygons.push_back(p);
        }
        std::shuffle(polygons.begin(),polygons.end(),random);
        check(polygons,"random "+std::to_string(k)+"/"+std::to_string(sample));
    }
}
void intersecting_cycles() {
    std::vector<Polygons> cases{
        {box(0,0,2,2),box(1,1,3,3)},
        {box(0,0,1,1),box(1,0,2,1)},
        {box(0,0,1,1),box(1,1,2,2)},
        {box(-10,-10,10,10),box(-4,0,-3,1),box(3,0,4,1)},
        {box(-3,-1,-2,0),box(0,-2,2,2),box(-2,-2,2,0),box(0,2,1,3)},
        {box(-3,-1,-2,0),box(0,-2,2,2),box(-2,-2,2,0),box(-1,-3,3,0),box(0,2,1,3)},
        {box(0,0,3,2),box(2,0,5,2),box(4,4,6,6),box(-1,4,1,6)},
        {box(-4,-4,-2,-2),box(2,2,4,4),box(-4,2,-2,4),box(2,-4,4,-2),box(-3,-3,3,3)},
        {{{1,0},{-1,1},{-3,-3}},{{0,1},{1,-1},{3,3}},{{0,0},{1,1},{-2,2}}},
        {box(-10,-10,10,0),{{2,4},{6,0},{12,6},{8,10}},{{0,0},{2,4},{-6,8},{-8,4}}}
    };
    for(size_t index=0;index<cases.size();++index) {
        auto p=cases[index];
        for(size_t shift=0;shift<p.size();++shift) {
            const auto result=tpp::tpp_convex_solve_cycle(p);
            const auto anchored=tpp::tpp_convex_solve_cycle(p,{false});
            require(solved(anchored),"unaccelerated intersection: "+std::to_string(index)+"/"+std::to_string(shift)+": "+anchored.diagnostic);
            require(anchored.certificate.lower_bound<=result.certificate.upper_bound&&
                    result.certificate.lower_bound<=anchored.certificate.upper_bound,"anchor/refinement objectives overlap");
            require(solved(result),"intersections "+std::to_string(index)+"/"+std::to_string(shift)+": "+result.diagnostic);
            require(tpp::tpp_convex_verify_cycle_certificate(exact(p),result.contacts).status==tpp::ConvexCycleCertificateStatus::Optimal,
                    "intersections independent exact certificate");
            const auto floating=tpp::tpp_convex_solve_cycle_double(p);
            require(floating.status==ConvexCycleStatus::Optimal||floating.status==ConvexCycleStatus::FloatingPointLimit,
                    "intersections double: "+floating.diagnostic);
            require(tpp::tpp_convex_verify_cycle_certificate(p,floating.contacts).upper_bound==floating.certificate.upper_bound,
                    "intersections double independent bound");
            require(std::abs(floating.certificate.upper_bound-result.certificate.upper_bound)<=
                    128*p.size()*std::numeric_limits<double>::epsilon()*std::max(1.0,result.certificate.upper_bound),
                    "intersections double rounding error");
            if(index<3||index==8)require(result.certificate.upper_bound==0,"common region has zero length");
            if(index==8) {
                const tpp::ConvexRationalPoint third{tpp::ConvexRational(1)/3,tpp::ConvexRational(1)/3};
                for(const auto &q:result.contacts)require(q==third,"nonrepresentable common point remains exact");
                require(floating.status==ConvexCycleStatus::FloatingPointLimit&&floating.certificate.upper_bound>0,
                        "double cannot represent the singleton intersection at (1/3,1/3)");
                require(floating.certificate.lower_bound==0,"universal zero bound keeps the tiny rounded-cycle gap sharp");
            }
            if(index==3||index==4||index==6||index==9) {
                const auto native=tpp::tpp_convex_solve_cycle_double(p,{false,false});
                require(native.status==ConvexCycleStatus::Optimal||native.status==ConvexCycleStatus::FloatingPointLimit,
                        "pure double boundary search with intersections: "+native.diagnostic);
                require(std::abs(native.certificate.upper_bound-result.certificate.upper_bound)<=
                        128*p.size()*std::numeric_limits<double>::epsilon()*std::max(1.0,result.certificate.upper_bound),
                        "pure double intersecting anchor objective: "+std::to_string(index)+"/"+std::to_string(shift));
                require(native.rational_anchor_recoveries==0&&native.rational_feature_recoveries==0&&native.rational_cycle_recoveries==0,
                        "pure double boundary search never recovers rationally");
                require(native.certificate.lower_bound<=result.certificate.upper_bound&&result.certificate.lower_bound<=native.certificate.upper_bound,
                        "pure double boundary objective");
            }
            std::rotate(p.begin(),p.begin()+1,p.end());
        }
    }
    for(int exponent:{-1100,1100}) {
        using R=tpp::ConvexRational;
        const R scale=exponent>0?R(boost::multiprecision::cpp_int(1)<<exponent):R(1)/R(boost::multiprecision::cpp_int(1)<<(-exponent));
        auto scaled=exact(cases[4]);for(auto &p:scaled)for(auto &q:p)q=q*scale;
        const auto result=tpp::tpp_convex_solve_cycle(scaled,{false});
        require(solved(result),"intersecting rational anchors beyond double exponent range: "+result.diagnostic);
        require(tpp::tpp_convex_verify_cycle_certificate(scaled,result.contacts).status==tpp::ConvexCycleCertificateStatus::Optimal,
                "scaled zero-link certificate");
    }
    // The three regions touch at the vertices of an acute triangle. Their
    // optimum is its orthic triangle, with all contacts on edge interiors.
    // The relevant anchor interval has coincident contacts at BOTH endpoints:
    // the old nonsmooth-interval skip could not solve this without refinement.
    const auto orthic=tpp::tpp_convex_solve_cycle(cases[9],{false});
    require(solved(orthic)&&orthic.oracle_calls>1,"nonsmooth endpoints bracket an interior rational root");
    require(std::abs(orthic.certificate.upper_bound-12*std::sqrt(10.0)/5)<1e-13,"orthic cycle objective");
    const auto p=exact(cases[4]);
    const tpp::ConvexRationalPolygon q{{-2,0},{0,0},{0,0},{0,2}};
    require(tpp::tpp_convex_verify_cycle_certificate(p,q).status==tpp::ConvexCycleCertificateStatus::Optimal,
            "zero link requires a subunit dual, unit directions alone are insufficient");
    auto bad=q;bad[1]={1,0};
    require(tpp::tpp_convex_verify_cycle_certificate(p,bad).status==tpp::ConvexCycleCertificateStatus::Feasible,
            "feasible nonoptimal zero-link neighbour must not be accepted");
    std::mt19937 random(47202);std::uniform_int_distribution<int> coordinate(-5,5),width(2,7);
    for(size_t k=2;k<=5;++k)for(size_t sample=0;sample<24;++sample) {
        Polygons regions;
        for(size_t i=0;i<k;++i) {
            const int x=coordinate(random),y=coordinate(random),w=width(random),h=width(random);
            auto polygon=box(x,y,x+w,y+h);
            if(sample>=12) {
                const int slope=int(i%3)+1;
                for(auto &v:polygon){const double dx=v.x-x,dy=v.y-y;v={x+2*dx-slope*dy,y+slope*dx+2*dy};}
            }
            regions.push_back(polygon);
        }
        const auto result=tpp::tpp_convex_solve_cycle(regions);
        const auto anchored=tpp::tpp_convex_solve_cycle(regions,{false});
        require(solved(anchored),"unaccelerated random overlap "+std::to_string(k)+"/"+std::to_string(sample)+": "+anchored.diagnostic);
        require(anchored.certificate.lower_bound<=result.certificate.upper_bound&&
                result.certificate.lower_bound<=anchored.certificate.upper_bound,"random anchor/refinement objective");
        require(solved(result),"random overlaps "+std::to_string(k)+"/"+std::to_string(sample)+": "+result.diagnostic);
        require(tpp::tpp_convex_verify_cycle_certificate(exact(regions),result.contacts).status==tpp::ConvexCycleCertificateStatus::Optimal,
                "random overlaps certificate");
        const auto floating=tpp::tpp_convex_solve_cycle_double(regions);
        require(floating.status==ConvexCycleStatus::Optimal||floating.status==ConvexCycleStatus::FloatingPointLimit,
                "random overlaps double construction: "+floating.diagnostic);
        const auto certificate=tpp::tpp_convex_verify_cycle_certificate(regions,floating.contacts);
        require(certificate.status==tpp::ConvexCycleCertificateStatus::Optimal||certificate.status==tpp::ConvexCycleCertificateStatus::Feasible,
                "random overlaps double exact feasibility");
        require(certificate.upper_bound==floating.certificate.upper_bound&&certificate.lower_bound==floating.certificate.lower_bound,
                "random overlaps double result carries its own certificate");
        if(floating.status==ConvexCycleStatus::Optimal)require(certificate.status==tpp::ConvexCycleCertificateStatus::Optimal,
                "random overlaps double Optimal is exact");
        require(std::abs(certificate.upper_bound-result.certificate.upper_bound)<=
                128*k*std::numeric_limits<double>::epsilon()*std::max(1.0,result.certificate.upper_bound),
                "random overlaps double objective differs beyond reporting roundoff: "+std::to_string(k)+"/"+std::to_string(sample)+" status="+std::to_string(int(floating.status))+" upper="+std::to_string(certificate.upper_bound)+" exact="+std::to_string(result.certificate.upper_bound)+" recoveries="+std::to_string(floating.rational_cycle_recoveries)+": "+floating.diagnostic);
    }
    std::cout<<"Intersecting cycles: closed contacts, containment, subunit zero duals, cyclic shifts and 96 seeded cases passed.\n";
}
} // namespace
int main() {
    try {
        known_cycles();invalid_inputs();references();random_cycles();intersecting_cycles();
        std::cout<<"Double comparisons: "<<compared_double_cases<<", maximum objective difference="<<largest_double_difference<<", maximum certified gap="<<largest_double_gap<<", local recoveries="<<total_anchor_recoveries<<'\n';
        std::cout<<"Cycle solver tests passed.\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
