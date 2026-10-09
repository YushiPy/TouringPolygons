#include "tpp_convex.h"
#include "tpp/convex/float_oracle.h"

#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
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
    const auto filtered=tpp::tpp_convex_verify_cycle_certificate(polygons,result.contacts,INFINITY,true);
    require(filtered.status==certificate.status&&filtered.lower_bound<=exact_result.certificate.upper_bound&&
        filtered.upper_bound>=exact_result.certificate.lower_bound,name+": interval certificate encloses independently solved optimum");
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
    // Changing the exact arithmetic backend must not reinterpret binary64
    // input as a decimal approximation, including subnormal coordinates.
    using Integer=tpp::ConvexInteger;
    using Rational=tpp::ConvexRational;
    const Rational next=Rational(1)+Rational(1)/Rational(Integer(1)<<52);
    require(Rational(std::nextafter(1.0,2.0))==next,"exact binary64 import above one");
    require(Rational(std::nextafter(-1.0,-2.0))==-next,"exact negative binary64 import");
    require(Rational(std::numeric_limits<double>::denorm_min())==Rational(1)/Rational(Integer(1)<<1074),
            "exact subnormal binary64 import");
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
    const R offset=R(tpp::ConvexInteger(1)<<80)+R(1)/7;
    for(auto &polygon:rational_input)for(auto &q:polygon){q.x+=offset;q.y+=offset;}
    const auto translated=tpp::tpp_convex_solve_cycle_disjoint(rational_input);
    require(translated.status==ConvexCycleStatus::Optimal,"rational input needs no floating geometry");
    require(translated.contacts[0]==tpp::ConvexRationalPoint(offset+R(65)/29,offset+R(26)/29),"exact translated contact");

    for(int exponent:{-1100,1100}) {
        const R scale=exponent>0?R(tpp::ConvexInteger(1)<<exponent):
                                 R(1)/R(tpp::ConvexInteger(1)<<(-exponent));
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
    ptree inputs,reference_data;
    const std::string directory=TPP_CYCLE_REFERENCE_DIR;
    read_json(directory+"/instances.json",inputs);
    read_json(directory+"/reference.json",reference_data);
    std::cout<<"name,k,lower,upper,gap,gurobi_objective,error,oracle_calls,milliseconds\n";
    for (const auto &[key,instance]:inputs.get_child("instances")) {
        const auto name=instance.get<std::string>("name");Polygons p;
        for (const auto &[unused,polygon]:instance.get_child("polygons"))p.push_back(points(polygon));
        const ptree *reference=nullptr;
        for (const auto &[unused,row]:reference_data.get_child("instances"))
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
        const double tolerance=reference_data.get<double>("comparison_tolerance.absolute")+
            reference_data.get<double>("comparison_tolerance.relative")*std::abs(objective);
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
void active_contact_regressions() {
    boost::property_tree::ptree input;
    read_json(TPP_CYCLE_REGRESSION_FILE,input);
    size_t cases=0,cutoffs=0,rational_cutoffs=0;
    for(const auto &[unused,item]:input.get_child("instances")) {
        const auto name=item.get<std::string>("name");Polygons p;
        for(const auto &[key,polygon]:item.get_child("polygons"))p.push_back(points(polygon));
        const auto r=tpp::tpp_convex_solve_cycle(p);
        require(solved(r),name+": active contact certificate");
        require(r.oracle_calls<=2*p.size()+1,name+": bounded construction avoids extensive anchor search");
        require(tpp::tpp_convex_verify_cycle_certificate(exact(p),r.contacts).status==
                tpp::ConvexCycleCertificateStatus::Optimal,name+": independent rational verification");
        const auto d=tpp::tpp_convex_solve_cycle_double(p);
        const auto c=tpp::tpp_convex_verify_cycle_certificate(p,d.contacts);
        require(d.status==ConvexCycleStatus::Optimal||d.status==ConvexCycleStatus::FloatingPointLimit,
                name+": filtered double construction");
        require(d.rational_anchor_recoveries==0&&
                d.rational_cycle_recoveries<=item.get<size_t>("max_cycle_recoveries",0)&&
                d.rational_feature_recoveries<=2*(p.size()+1),
                name+": bounded recovery avoids repeated repairs of double anchors");
        if(item.get<bool>("retain_exact_bound",false))
            require(d.certificate.lower_bound>=r.certificate.lower_bound,
                    name+": exact recovery bound survives rounded zero links");
        require((c.status==tpp::ConvexCycleCertificateStatus::Feasible||c.status==tpp::ConvexCycleCertificateStatus::Optimal)&&
                c.lower_bound<=r.certificate.upper_bound&&r.certificate.lower_bound<=c.upper_bound,
                name+": independent double feasibility and overlapping bounds");
        tpp::ConvexCycleOptions rational_seed;
        tpp::ConvexCycleDoubleOptions double_seed;
        for(const auto &polygon:p) {
            rational_seed.initial_contacts.emplace_back(polygon.front());
            double_seed.initial_contacts.push_back(polygon.front());
        }
        const auto seeded=tpp::tpp_convex_solve_cycle(p,rational_seed);
        require(solved(seeded)&&seeded.certificate.lower_bound<=r.certificate.upper_bound&&
                r.certificate.lower_bound<=seeded.certificate.upper_bound,name+": rational seed preserves optimum");
        const auto floating_seeded=tpp::tpp_convex_solve_cycle_double(p,double_seed);
        const auto seeded_check=tpp::tpp_convex_verify_cycle_certificate(p,floating_seeded.contacts);
        require((seeded_check.status==tpp::ConvexCycleCertificateStatus::Optimal||
                 seeded_check.status==tpp::ConvexCycleCertificateStatus::Feasible)&&
                seeded_check.lower_bound<=r.certificate.upper_bound&&r.certificate.lower_bound<=seeded_check.upper_bound,
                name+": double seed preserves certified bounds");
        double_seed.lower_bound_cutoff=r.certificate.lower_bound/2;
        const auto cut=tpp::tpp_convex_solve_cycle_double(p,double_seed);
        require(cut.status==ConvexCycleStatus::CertifiedBound||cut.status==ConvexCycleStatus::Optimal,
                name+": cutoff has an explicit certified status");
        const auto cut_check=tpp::tpp_convex_verify_cycle_certificate(p,cut.contacts);
        require(cut.certificate.lower_bound>=double_seed.lower_bound_cutoff&&
                cut.certificate.lower_bound<=r.certificate.upper_bound&&r.certificate.lower_bound<=cut_check.upper_bound&&
                (cut_check.status==tpp::ConvexCycleCertificateStatus::Optimal||cut_check.status==tpp::ConvexCycleCertificateStatus::Feasible),
                name+": independent cutoff verification");
        cutoffs+=cut.status==ConvexCycleStatus::CertifiedBound;
        rational_seed.lower_bound_cutoff=double_seed.lower_bound_cutoff;
        const auto rational_cut=tpp::tpp_convex_solve_cycle(p,rational_seed);
        const auto rational_cut_check=tpp::tpp_convex_verify_cycle_certificate(exact(p),rational_cut.contacts);
        require((rational_cut.status==ConvexCycleStatus::Optimal||rational_cut.status==ConvexCycleStatus::CertifiedBound)&&
                rational_cut.certificate.lower_bound>=rational_seed.lower_bound_cutoff&&
                rational_cut.certificate.lower_bound==rational_cut_check.lower_bound&&
                rational_cut.certificate.lower_bound<=r.certificate.upper_bound&&
                (rational_cut_check.status==tpp::ConvexCycleCertificateStatus::Optimal||
                 rational_cut_check.status==tpp::ConvexCycleCertificateStatus::Feasible),name+": independent rational cutoff");
        rational_cutoffs+=rational_cut.status==ConvexCycleStatus::CertifiedBound;
        rational_seed.lower_bound_cutoff=std::numeric_limits<double>::quiet_NaN();
        require(tpp::tpp_convex_solve_cycle(p,rational_seed).status==ConvexCycleStatus::InvalidInput,
                name+": reject NaN rational cutoff");
        ++cases;
    }
    require(cutoffs>0,"early certified bound exercised before exact optimality");
    require(rational_cutoffs>0,"rational recovery can stop at a nonoptimal certified bound");
    std::cout<<"Active contact regressions: "<<cases<<" rational/double relaxations passed.\n";
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
void boundary_recovery_regressions() {
    boost::property_tree::ptree input;read_json(TPP_CYCLE_REGRESSION_FILE,input);
    for(const auto &[unused,item]:input.get_child("recovery_regressions")) {
        const auto name=item.get<std::string>("name");Polygons p;
        for(const auto &[key,polygon]:item.get_child("polygons"))p.push_back(points(polygon));
        tpp::ConvexCycleOptions exact_options;
        tpp::ConvexCycleDoubleOptions options;
        options.initial_contacts=points(item.get_child("initial_contacts"));
        options.lower_bound_cutoff=item.get<double>("cutoff");
        for(auto q:options.initial_contacts)exact_options.initial_contacts.emplace_back(q);
        const auto rational=tpp::tpp_convex_solve_cycle(p,exact_options);
        require(solved(rational),name+": rational boundary recovery");
        require(tpp::tpp_convex_verify_cycle_certificate(exact(p),rational.contacts).status==
                tpp::ConvexCycleCertificateStatus::Optimal,name+": independent rational optimality");
        auto anchored_options=exact_options;anchored_options.refine_contacts=false;
        anchored_options.lower_bound_cutoff=rational.certificate.lower_bound/2;
        const auto anchored_cut=tpp::tpp_convex_solve_cycle(p,anchored_options);
        const auto anchored_check=tpp::tpp_convex_verify_cycle_certificate(exact(p),anchored_cut.contacts);
        require((anchored_cut.status==ConvexCycleStatus::CertifiedBound||anchored_cut.status==ConvexCycleStatus::Optimal)&&
                anchored_cut.certificate.lower_bound>=anchored_options.lower_bound_cutoff&&
                anchored_cut.certificate.lower_bound==anchored_check.lower_bound&&
                anchored_cut.certificate.lower_bound<=rational.certificate.upper_bound,
                name+": boundary prepass respects exact cutoff without refinement");
        const auto floating=tpp::tpp_convex_solve_cycle_double(p,options);
        require(floating.status==ConvexCycleStatus::Optimal||floating.status==ConvexCycleStatus::FloatingPointLimit,
                name+": filtered boundary recovery");
        const auto verified=tpp::tpp_convex_verify_cycle_certificate(p,floating.contacts);
        require(verified.status==tpp::ConvexCycleCertificateStatus::Feasible||
                verified.status==tpp::ConvexCycleCertificateStatus::Optimal,name+": rounded contacts feasible");
        require(floating.certificate.upper_bound==verified.upper_bound&&
                floating.certificate.lower_bound>=rational.certificate.lower_bound&&
                floating.certificate.lower_bound<=rational.certificate.upper_bound&&
                rational.certificate.lower_bound<=verified.upper_bound,name+": retain independently audited exact bound");
        const double rounding=128*p.size()*std::numeric_limits<double>::epsilon()*std::max(1.0,verified.upper_bound);
        require(verified.upper_bound-rational.certificate.upper_bound<=rounding,name+": objective reporting roundoff");
        if(floating.status==ConvexCycleStatus::Optimal)
            require(verified.status==tpp::ConvexCycleCertificateStatus::Optimal,name+": no rounded optimality claim");
    }
    std::cout<<"Boundary recovery regressions passed.\n";
}
void intersecting_cycles() {
    {
        // Two touching regions can trap separate coordinate updates at the
        // top of their shared edge. Their zero-link block must slide jointly.
        const Polygons p{box(-2,-2,0,2),box(0,-2,2,2),box(-4,4,-3,5),box(-4,-6,-3,-5)};
        const Polygon seed{{0,2},{0,2},{-3,4},{-3,-5}};
        tpp::ConvexCycleOptions options;for(auto q:seed)options.initial_contacts.emplace_back(q);
        const auto r=tpp::tpp_convex_solve_cycle(p,options);
        require(solved(r)&&r.oracle_calls==1,"shared segment solved by one joint block update");
        require(r.contacts[0]==tpp::ConvexRationalPoint(0,tpp::ConvexRational(-1)/2)&&r.contacts[1]==r.contacts[0],
                "shared segment contact is exact");
        require(tpp::tpp_convex_verify_cycle_certificate(exact(p),r.contacts).status==
                tpp::ConvexCycleCertificateStatus::Optimal,"shared segment independent certificate");
        tpp::ConvexCycleDoubleOptions floating;floating.initial_contacts=seed;
        const auto d=tpp::tpp_convex_solve_cycle_double(p,floating);
        require(d.status==ConvexCycleStatus::Optimal&&d.oracle_calls==1,"double shares segment block update");
    }
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
        const R scale=exponent>0?R(tpp::ConvexInteger(1)<<exponent):R(1)/R(tpp::ConvexInteger(1)<<(-exponent));
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
        if(sample==0) {
            tpp::ConvexCycleOptions warm_options;
            for(const auto &region:regions)warm_options.initial_contacts.emplace_back(region.front());
            const auto warm=tpp::tpp_convex_solve_cycle(regions,warm_options);
            require(solved(warm)&&warm.certificate.lower_bound<=result.certificate.upper_bound&&
                result.certificate.lower_bound<=warm.certificate.upper_bound,"Warm and cold exact recovery intervals overlap");
        }
        const auto floating=tpp::tpp_convex_solve_cycle_double(regions);
        require(floating.status==ConvexCycleStatus::Optimal||floating.status==ConvexCycleStatus::FloatingPointLimit,
                "random overlaps double construction: "+floating.diagnostic);
        const auto certificate=tpp::tpp_convex_verify_cycle_certificate(regions,floating.contacts);
        require(certificate.status==tpp::ConvexCycleCertificateStatus::Optimal||certificate.status==tpp::ConvexCycleCertificateStatus::Feasible,
                "random overlaps double exact feasibility");
        require(certificate.upper_bound==floating.certificate.upper_bound&&
                floating.certificate.lower_bound>=certificate.lower_bound&&
                floating.certificate.lower_bound<=result.certificate.upper_bound,
                "random overlaps double result retains its independently solved rational lower bound");
        if(floating.status==ConvexCycleStatus::Optimal)require(certificate.status==tpp::ConvexCycleCertificateStatus::Optimal,
                "random overlaps double Optimal is exact");
        require(std::abs(certificate.upper_bound-result.certificate.upper_bound)<=
                128*k*std::numeric_limits<double>::epsilon()*std::max(1.0,result.certificate.upper_bound),
                "random overlaps double objective differs beyond reporting roundoff: "+std::to_string(k)+"/"+std::to_string(sample)+" status="+std::to_string(int(floating.status))+" upper="+std::to_string(certificate.upper_bound)+" exact="+std::to_string(result.certificate.upper_bound)+" recoveries="+std::to_string(floating.rational_cycle_recoveries)+": "+floating.diagnostic);
    }
    std::cout<<"Intersecting cycles: closed contacts, containment, subunit zero duals, cyclic shifts and 96 seeded cases passed.\n";
}
void prepared_geometry_and_features() {
    using namespace tpp;
    Polygons p{box(-5,0,-3,2),box(0,4,2,6),box(5,0,7,2)};
    ConvexCycleWorkspace workspace;
    ConvexCycleDoubleOptions options;options.workspace=&workspace;options.retain_active_features=true;
    for(size_t pass=0;pass<5;++pass) {
        const auto reference=tpp_convex_solve_cycle(p);
        const auto result=tpp_convex_solve_cycle_double(p,options);
        const auto independent=tpp_convex_verify_cycle_certificate(p,result.contacts);
        const auto prepared=workspace.prepare(p);
        const auto cached=tpp_convex_verify_cycle_certificate(prepared,result.contacts);
        require(cached.status==independent.status&&cached.lower_bound==independent.lower_bound&&
            cached.upper_bound==independent.upper_bound,"Prepared verifier matches independent verifier");
        require(result.certificate.lower_bound<=reference.certificate.upper_bound&&
            reference.certificate.lower_bound<=result.certificate.upper_bound,"Cached feature proposal preserves objective interval");
        options.initial_contacts=result.contacts;options.initial_features=result.active_features;
        if(pass==0) { // Insert an overlapping region: duplicate the inherited split-link feature.
            p.insert(p.begin()+1,box(-1,0,1,3));
            options.initial_contacts.insert(options.initial_contacts.begin()+1,Vector2{0,1});
            if(!options.initial_features.empty())options.initial_features.insert(options.initial_features.begin()+1,-2);
        } else if(pass==1) { // Mutate the same container; pointer-keyed caches would be stale.
            for(auto &v:p[1])v.x+=1;
            if(!options.initial_features.empty())options.initial_features[1]=-2;
        } else if(pass==2) {
            std::reverse(p.begin(),p.end());options.initial_contacts.clear();options.initial_features.clear();
        } else if(pass==3) {
            for(auto &poly:p) {std::reverse(poly.begin(),poly.end());poly.push_back(poly.front());}
            options.initial_features.assign(p.size(),9999); // Bad hints must fall back safely.
        }
        ConvexRationalPolygon u(p.size(),ConvexRationalPoint(ConvexRational(1)/2,ConvexRational(1)/2));
        require(tpp_convex_cycle_dual_bound(workspace.prepare(p),u)==0,"Constant unit-disk field telescopes");
        u.front()={2,0};bool rejected=false;
        try{tpp_convex_cycle_dual_bound(workspace.prepare(p),u);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Uncertified superunit dual is rejected");
    }
    require(workspace.size()>p.size(),"Content-keyed cache accounts for mutation and winding");
    workspace.clear();require(workspace.size()==0,"Explicit workspace lifetime");
    require(tpp_convex_solve_cycle_double(p).active_features.empty(),"Default solve does not export unused feature metadata");
    std::cout<<"Prepared geometry, stale/malformed features and dual feasibility tests passed.\n";
}
void bound_first_contacts() {
    using namespace tpp;
    const Polygons p{box(0,-1,1,1),box(2,-1,3,1)};
    ConvexCycleDoubleOptions options;options.bound_first=true;options.lower_bound_cutoff=1;
    options.initial_contacts={{0,0},{3,0}};
    const auto result=tpp_convex_solve_cycle_double(p,options);
    const auto check=tpp_convex_verify_cycle_certificate(p,result.contacts);
    require(result.status==ConvexCycleStatus::CertifiedBound&&result.initial_contact_checks==1&&
        result.initial_contact_accepts==1&&result.certificate_cutoff_skips==1&&
        check.status==ConvexCycleCertificateStatus::Feasible&&check.lower_bound>=options.lower_bound_cutoff,
        "Inherited suboptimal contacts suffice only for an independently verified cutoff");
    ConvexCycleOptions rational;rational.bound_first=true;rational.lower_bound_cutoff=1;
    for(auto q:options.initial_contacts)rational.initial_contacts.emplace_back(q);
    const auto exact=tpp_convex_solve_cycle(p,rational);
    require(exact.status==ConvexCycleStatus::CertifiedBound&&exact.certificate_cutoff_skips==1,
        "Rational and double paths share cutoff semantics");
    options.lower_bound_cutoff=INFINITY;
    const auto complete=tpp_convex_solve_cycle_double(p,options);
    require(complete.status==ConvexCycleStatus::Optimal&&complete.certificate.upper_bound==2,
        "An infinite cutoff still requires the complete optimum");
}
void cooperative_interruption() {
    using namespace tpp;
    const Polygons p{box(-5,0,-3,2),box(0,4,2,6),box(5,0,7,2),box(-1,-3,2,-1)};
    const auto optimum=tpp_convex_solve_cycle(p);
    require(optimum.status==ConvexCycleStatus::Optimal,"Interruption reference optimum");
    bool saw_double_candidate=false,saw_rational_candidate=false;
    // Interrupt at every checkpoint of this small solve: covers preparation,
    // candidate verification, active features and nested rational recovery.
    for(bool rational:{false,true}) {
        size_t full_checks=0;
        if(rational) {
            ConvexCycleOptions o;o.refine_contacts=false;o.stop_requested=[&]{++full_checks;return false;};
            require(tpp_convex_solve_cycle(p,o).status==ConvexCycleStatus::Optimal,"Callback never requests stop");
        } else {
            ConvexCycleDoubleOptions o;o.stop_requested=[&]{++full_checks;return false;};
            tpp_convex_solve_cycle_double(p,o);
        }
        for(size_t limit=1;limit<=full_checks;++limit) {
            size_t checks=0;
            auto stop=[&]{return ++checks>=limit;};
            if(rational) {
                ConvexCycleOptions o;o.refine_contacts=false;o.stop_requested=stop;
                const auto r=tpp_convex_solve_cycle(p,o);
                require(r.status==ConvexCycleStatus::Interrupted,"Rational checkpoint cannot claim completion");
                if(!r.contacts.empty()) {
                    saw_rational_candidate=true;
                    const auto c=tpp_convex_verify_cycle_certificate(exact(p),r.contacts);
                    require(c.status==ConvexCycleCertificateStatus::Optimal||c.status==ConvexCycleCertificateStatus::Feasible,
                        "Interrupted rational candidate retains exact membership");
                    require(r.certificate.lower_bound<=optimum.certificate.upper_bound&&
                        r.certificate.upper_bound>=optimum.certificate.lower_bound,"Interrupted rational interval encloses optimum");
                }
            } else {
                ConvexCycleDoubleOptions o;o.stop_requested=stop;
                const auto r=tpp_convex_solve_cycle_double(p,o);
                require(r.status==ConvexCycleStatus::Interrupted,"Double checkpoint cannot claim completion");
                if(!r.contacts.empty()) {
                    saw_double_candidate=true;
                    const auto c=tpp_convex_verify_cycle_certificate(p,r.contacts);
                    require(c.status==ConvexCycleCertificateStatus::Optimal||c.status==ConvexCycleCertificateStatus::Feasible,
                        "Interrupted double candidate retains exact membership");
                    require(r.certificate.lower_bound<=optimum.certificate.upper_bound&&
                        r.certificate.upper_bound>=optimum.certificate.lower_bound,"Interrupted double interval encloses optimum");
                }
                require(r.timings.construction_seconds>=0&&r.timings.certification_seconds>=0&&
                    r.timings.rational_recovery_seconds>=0,"Exclusive phase durations remain nonnegative on interruption");
            }
        }
    }
    require(saw_double_candidate&&saw_rational_candidate,"Interruption exercised previously certified candidates");
    ConvexCycleDoubleOptions zero;zero.max_seconds=0;
    const auto empty=tpp_convex_solve_cycle_double(p,zero);
    require(empty.status==ConvexCycleStatus::Interrupted&&empty.contacts.empty()&&empty.certificate.lower_bound==0&&
        std::isinf(empty.certificate.upper_bound),
        "Immediate deadline returns no invented candidate or positive bound");
    require(tpp_convex_solve_cycle(p).status==ConvexCycleStatus::Optimal,"Cancellation context restored after return");
}
} // namespace
// Points and segments are closed convex regions of the general cycle API.
void degenerate_regions() {
    auto solve=[](const Polygons &p,double expected,const std::string &name) {
        const auto r=tpp::tpp_convex_solve_cycle(p);
        require(solved(r),name+": status "+std::to_string(int(r.status))+" "+r.diagnostic);
        const auto c=tpp::tpp_convex_verify_cycle_certificate(exact(p),r.contacts);
        require(c.status==tpp::ConvexCycleCertificateStatus::Optimal,name+": independent certificate");
        if(!std::isnan(expected))require(std::abs(r.certificate.upper_bound-expected)<=1e-12*(1+expected),name+": objective");
        tpp::ConvexCycleDoubleOptions options;
        const auto d=tpp::tpp_convex_solve_cycle_double(p,options);
        require(d.status==ConvexCycleStatus::Optimal||d.status==ConvexCycleStatus::FloatingPointLimit,name+": double status");
        const double value=std::isnan(expected)?r.certificate.upper_bound:expected;
        require(d.certificate.lower_bound<=value+1e-9&&d.certificate.upper_bound>=value-1e-9,name+": double bounds");
    };
    solve({{{0,0},{4,0}},{{0,3},{4,3}}},6,"parallel segments");
    solve({{{0,0},{4,0}},{{2,5}}},10,"segment and point");
    solve({{{1,1}},{{4,5}}},10,"two points");
    solve({{{0,0},{4,4}},{{0,4},{4,0}},box(1,1,3,3)},0,"crossing segments in a box");
    solve({{{0,0},{4,0}},{{2,-1},{2,1}},{{2,0}}},0,"common point of segments and a point");
    // A segment touching a box corner and a distant point: the shared corner
    // is not common to all three regions (regression for the common-region
    // clip, which treated the point and the segment's line as unbounded).
    solve({box(5,-5,7,-3),{{6,-1},{7,-3}},{{-2,3}}},NAN,"touching segment and far point");
    solve({{{0,0},{2,0}},{{5,0},{7,0}},{{3,2}}},std::sqrt(5.0)+3+std::sqrt(8.0),"collinear segments and a point");
}

// Two overlapping neighbours whose shared optimal contact is a crossing of
// their edges, a corner of the common region but no vertex of either polygon.
// Feature reflection alone never reconstructs it; the rational boundary
// bisection then ran until its deadline (Paula's 10i400-206, 4 regions).
void common_region_corners() {
    std::mt19937 rng(20261006);std::uniform_real_distribution<double> unit(-1,1);
    size_t corners=0,cases=0;
    auto polygon=[](double cx,double cy,double r,double turn) {
        Polygon q;for(int i=0;i<5;++i){const double a=turn+i*2*M_PI/5;q.push_back({cx+r*std::cos(a),cy+r*std::sin(a)});}
        return q;
    };
    for(int trial=0;trial<200;++trial) {
        // Both other regions lie above the overlapping pair, so the tour
        // reaches up into it: the top of their lens is an edge crossing.
        const double gap=0.2+0.6*std::abs(unit(rng));
        const Polygons p{polygon(-0.5+unit(rng),6+unit(rng),0.5,unit(rng)),polygon(-gap,0,1,unit(rng)),
                         polygon(gap,0.3*unit(rng),1,unit(rng)),polygon(0.5+unit(rng),6+unit(rng),0.5,unit(rng))};
        const auto exact_result=tpp::tpp_convex_solve_cycle(p);
        require(solved(exact_result),"common-region corner: rational status "+std::to_string(int(exact_result.status))+" "+exact_result.diagnostic);
        const auto floating=tpp::tpp_convex_solve_cycle_double(p);
        require(floating.status==ConvexCycleStatus::Optimal||floating.status==ConvexCycleStatus::FloatingPointLimit,
                "common-region corner: double status");
        require(floating.certificate.lower_bound<=exact_result.certificate.upper_bound+1e-9&&
                floating.certificate.upper_bound>=exact_result.certificate.lower_bound-1e-9,"common-region corner: bounds");
        const auto &q=exact_result.contacts;const auto rational=exact(p);
        auto vertex=[&](size_t i){return std::find(rational[i].begin(),rational[i].end(),q[i])!=rational[i].end();};
        corners+=q[1]==q[2]&&!vertex(1)&&!vertex(2)&&!(q[0]==q[1])&&!(q[2]==q[3]);++cases;
    }
    require(corners>0,"common-region corner optima exercised");
    std::cout<<"Common-region corners: "<<corners<<" of "<<cases<<" optima at an edge crossing\n";
}

// Binary64 stage of the cycle oracle: tpp_convex_solve_cycle_float_certified
// and ConvexCycleDoubleOptions::max_gap. Every interval must contain the exact
// rational optimum, closed calls meet their gap, and contacts are feasible
// (on a segment: within rounding error of an exact segment point).
struct FloatStageCounts {size_t interval=0,polish=0,open=0,checks=0,double_closed=0,double_exact=0;} float_counts;
bool float_closed(const tpp::ConvexCycleFloatResult &r) {
    return r.status==tpp::ConvexFloatOracleStatus::GapClosed||r.status==tpp::ConvexFloatOracleStatus::CutoffReached;
}
void require_float_contacts(const Polygons &p,const std::vector<Vector2> &q,const std::string &name) {
    require(q.size()==p.size(),name+": one contact per region");
    for(size_t i=0;i<p.size();++i) {
        Polygon region;
        for(auto v:p[i])if(region.empty()||!(v==region.back()))region.push_back(v);
        while(region.size()>1&&region.front()==region.back())region.pop_back();
        if(region.size()==2) {
            const auto a=region[0],e=region[1]-region[0],d=q[i]-a;
            const double size=std::max({std::abs(a.x),std::abs(a.y),std::abs(q[i].x),std::abs(q[i].y),e.length()});
            const double t=d.dot(e)/e.dot(e),eps=std::numeric_limits<double>::epsilon();
            require(std::abs(e.cross(d))<=16*eps*size*e.length()&&t>=-16*eps&&t<=1+16*eps,name+": segment contact rounds a segment point");
            continue;
        }
        const auto single=tpp::tpp_convex_verify_cycle_certificate(Polygons{p[i],p[i]},std::vector<Vector2>{q[i],q[i]});
        require(single.status==tpp::ConvexCycleCertificateStatus::Optimal||single.status==tpp::ConvexCycleCertificateStatus::Feasible,
                name+": contact exactly inside its region");
    }
}
void check_float_stage(const Polygons &p,const std::string &name) {
    const auto exact_result=tpp::tpp_convex_solve_cycle(p);
    require(solved(exact_result),name+": rational reference "+exact_result.diagnostic);
    const double lower=exact_result.certificate.lower_bound,upper=exact_result.certificate.upper_bound;
    const double scale=std::max(1.0,upper);
    for(double relative:{1e-3,1e-6,1e-9})for(bool construct:{true,false}) {
        tpp::ConvexCycleFloatOptions options;options.max_gap=relative*scale;options.construct=construct;
        const auto r=tpp::tpp_convex_solve_cycle_float_certified(p,options);
        const std::string label=name+" (gap "+std::to_string(relative)+(construct?"":", polish only")+")";
        require(r.status!=tpp::ConvexFloatOracleStatus::Unsupported,label+": supported input");
        require(r.lower_bound<=upper&&lower<=r.upper_bound,label+": float interval contains the exact optimum");
        ++float_counts.checks;
        if(!float_closed(r)){++float_counts.open;continue;}
        require(r.upper_bound-r.lower_bound<=options.max_gap,label+": closed gap");
        require(r.construction_closed!=r.polish_closed,label+": one closing stage");
        require(construct||r.polish_closed,label+": no construction when disabled");
        (r.construction_closed?float_counts.interval:float_counts.polish)++;
        require_float_contacts(p,r.contacts,label);
    }
    // A cutoff below the optimum, no gap: closing needs proved contacts too.
    if(lower>0) {
        tpp::ConvexCycleFloatOptions options;options.cutoff=0.9*lower;
        const auto r=tpp::tpp_convex_solve_cycle_float_certified(p,options);
        require(r.lower_bound<=upper&&lower<=r.upper_bound,name+" (cutoff): interval contains the optimum");
        if(float_closed(r)) {
            require(r.status==tpp::ConvexFloatOracleStatus::CutoffReached&&r.lower_bound>=options.cutoff&&std::isfinite(r.upper_bound),
                    name+" (cutoff): reached with a proved cycle");
            require_float_contacts(p,r.contacts,name+" (cutoff)");
        }
    }
    // The same stage inside the double API, then the exact path when it is open.
    for(double relative:{1e-6,1e-300}) {
        tpp::ConvexCycleDoubleOptions options;options.max_gap=relative*scale;
        const auto d=tpp::tpp_convex_solve_cycle_double(p,options);
        require(d.certificate.lower_bound<=upper&&lower<=d.certificate.upper_bound,name+": double API interval contains the optimum");
        if(d.status==ConvexCycleStatus::GapClosed) {
            ++float_counts.double_closed;
            require(d.float_interval_closed||d.float_polish_closed,name+": GapClosed names its stage");
            require(d.rational_cycle_recoveries+d.rational_anchor_recoveries+d.rational_feature_recoveries==0&&
                    d.certificate_checks==0,name+": GapClosed uses no exact certificate");
            require(d.certificate.upper_bound-d.certificate.lower_bound<=options.max_gap,name+": GapClosed meets max_gap");
            require_float_contacts(p,d.contacts,name);
        } else {
            ++float_counts.double_exact;
            require(d.status==ConvexCycleStatus::Optimal||d.status==ConvexCycleStatus::FloatingPointLimit,
                    name+": open binary64 stage falls back to the exact path: "+std::to_string(int(d.status))+" "+d.diagnostic);
            require(!d.float_interval_closed&&!d.float_polish_closed,name+": exact result is not a binary64 closure");
        }
    }
}
void binary64_stage() {
    // Disjoint boxes on a line: the construction's candidate closes by the
    // interval proof, without polish or exact arithmetic.
    {
        const Polygons p{box(0,-1,1,1),box(6,-1,7,1),box(3,-1,4,1)};
        tpp::ConvexCycleFloatOptions options;options.max_gap=1e-6;
        const auto r=tpp::tpp_convex_solve_cycle_float_certified(p,options);
        require(r.status==tpp::ConvexFloatOracleStatus::GapClosed&&r.construction_closed&&!r.polish_attempted,
                "collinear boxes close by the interval proof");
        require(r.lower_bound<=10&&r.upper_bound>=10&&r.upper_bound-r.lower_bound<1e-12,"collinear boxes enclose 2*5");
    }
    // Instance t_A of the report as a cycle: points s and t, and an optimum
    // visiting P1 and P2 at the same point j=(0,0). The zero link's dual has
    // norm sqrt(13/20) < 1, out of reach of the short-link policies; the
    // smoothed directions of the polish represent it.
    {
        const Polygons p{{{-3,-4}},box(0,-6,6,6),{{-8,-4},{8,4},{8,9},{-8,9}},{{4,-3}}};
        const double optimum=10+std::sqrt(50.0);
        for(bool construct:{true,false}) {
            tpp::ConvexCycleFloatOptions options;options.max_gap=1e-6*optimum;options.construct=construct;
            const auto r=tpp::tpp_convex_solve_cycle_float_certified(p,options);
            require(r.status==tpp::ConvexFloatOracleStatus::GapClosed&&r.polish_closed,"t_A cycle closes by the polish");
            require(r.lower_bound<=optimum&&r.upper_bound>=optimum,"t_A cycle bounds enclose the optimum");
            require(r.contacts[0]==Vector2{-3,-4}&&r.contacts[3]==Vector2{4,-3},"t_A points are fixed contacts");
        }
        // No binary64 interval this tight: the exact path certifies it.
        tpp::ConvexCycleFloatOptions tight;tight.max_gap=1e-300;
        const auto open=tpp::tpp_convex_solve_cycle_float_certified(p,tight);
        require(open.status==tpp::ConvexFloatOracleStatus::Open&&open.lower_bound<=optimum&&open.upper_bound>=optimum,
                "t_A cycle stays open at a subnormal gap, with valid bounds");
        tpp::ConvexCycleDoubleOptions fallback;fallback.max_gap=1e-300;
        const auto exact_path=tpp::tpp_convex_solve_cycle_double(p,fallback);
        require(exact_path.status==ConvexCycleStatus::Optimal&&!exact_path.float_interval_closed&&!exact_path.float_polish_closed&&
                exact_path.certificate.lower_bound>=open.lower_bound,"t_A cycle falls back to the exact certificate");
        // A cutoff below the optimum is reached by the proved lower bound.
        tpp::ConvexCycleDoubleOptions cut;cut.max_gap=1e-300;cut.lower_bound_cutoff=0.999*optimum;
        const auto pruned=tpp::tpp_convex_solve_cycle_double(p,cut);
        require(pruned.status==ConvexCycleStatus::CertifiedBound&&pruned.float_polish_closed&&
                pruned.certificate.lower_bound>=cut.lower_bound_cutoff&&pruned.certificate.lower_bound<=optimum,
                "t_A cycle cutoff reached by the binary64 lower bound");
    }
    // A zero cycle: a common point of all regions.
    {
        tpp::ConvexCycleFloatOptions options;options.max_gap=1e-9;
        const auto r=tpp::tpp_convex_solve_cycle_float_certified({box(0,0,2,2),box(1,1,3,3),box(1,-1,4,1.5)},options);
        require(r.status==tpp::ConvexFloatOracleStatus::GapClosed&&r.upper_bound<=1e-9&&r.lower_bound==0,"zero cycle closes");
    }
    // An invalid region is never normalized in binary64.
    {
        tpp::ConvexCycleFloatOptions options;options.max_gap=1;
        require(tpp::tpp_convex_solve_cycle_float_certified({{{0,0},{2,0},{1,0.5},{2,2},{0,2}},box(3,0,4,1)},options).status==
                tpp::ConvexFloatOracleStatus::Unsupported,"nonconvex region unsupported");
        require(tpp::tpp_convex_solve_cycle_float_certified({{{0,0},{1,0},{2,0}},box(3,0,4,1)},options).status==
                tpp::ConvexFloatOracleStatus::Unsupported,"collinear polygon unsupported");
    }
    // Differential checks against the rational solver.
    const std::vector<std::pair<std::string,Polygons>> named{
        {"parallel segments",{{{0,0},{4,0}},{{0,3},{4,3}}}},
        {"segment and point",{{{0,0},{4,0}},{{2,5}}}},
        {"two points",{{{1,1}},{{4,5}}}},
        {"segment, box and point",{{{0,0},{4,1}},box(5,3,6,5),{{-2,4}}}},
        {"collinear segments and a point",{{{0,0},{2,0}},{{5,0},{7,0}},{{3,2}}}},
        {"oblique segments",{{{0.1,0.3},{2.7,1.9}},{{4.2,-1.3},{3.3,2.2}},{{1.7,4.1},{-0.6,3.3}},box(-2,-1,-1,0.5)}},
        {"touching tessellation",{box(0,0,1,1),box(1,0,2,1),box(2,0,3,1),box(1,1,2,2),box(0,3,1,4),box(2.5,3,3,3.5)}},
        {"shared edge block",{box(-2,-2,0,2),box(0,-2,2,2),box(-4,4,-3,5),box(-4,-6,-3,-5)}},
        {"nested",{box(-10,-10,10,10),box(-4,0,-3,1),box(3,0,4,1)}},
        {"overlapping chain",{box(-3,-1,-2,0),box(0,-2,2,2),box(-2,-2,2,0),box(-1,-3,3,0),box(0,2,1,3)}},
        {"crossing",{box(-4,-4,-3,-3),box(3,3,4,4),box(-4,3,-3,4),box(3,-4,4,-3)}},
    };
    for(const auto &[name,p]:named)check_float_stage(p,name);
    std::mt19937_64 random(20261008);
    std::uniform_real_distribution<double> unit(-1,1);
    auto polygon=[&](Vector2 c,double r,int sides,double turn) {
        Polygon q;for(int i=0;i<sides;++i){const double a=turn+i*2*M_PI/sides;q.push_back({c.x+r*std::cos(a),c.y+r*std::sin(a)});}
        return q;
    };
    for(int sample=0;sample<60;++sample) {
        const size_t k=2+sample%7;
        Polygons p;
        for(size_t i=0;i<k;++i) {
            // Mostly overlapping or touching neighbours, some points and segments.
            const Vector2 c{3*unit(random),3*unit(random)};
            const int kind=int(random()%8);
            if(kind==0)p.push_back({c});
            else if(kind==1)p.push_back({c,c+Vector2{2*unit(random),2*unit(random)}});
            else p.push_back(polygon(c,0.5+std::abs(unit(random)),3+int(random()%5),unit(random)));
        }
        check_float_stage(p,"random binary64 "+std::to_string(sample));
    }
    require(float_counts.interval>0&&float_counts.polish>0&&float_counts.open>0&&float_counts.double_closed>0&&float_counts.double_exact>0,
            "binary64 stage exercised interval proofs, polish, open calls and both double API outcomes");
    std::cout<<"Binary64 cycle stage: "<<float_counts.checks<<" calls, interval="<<float_counts.interval<<", polish="<<float_counts.polish
             <<", open="<<float_counts.open<<"; double API closed="<<float_counts.double_closed<<", exact="<<float_counts.double_exact<<'\n';
}

int main() {
    try {
        binary64_stage();
        degenerate_regions();
        common_region_corners();
        cooperative_interruption();
        bound_first_contacts();
        prepared_geometry_and_features();known_cycles();invalid_inputs();references();random_cycles();intersecting_cycles();active_contact_regressions();boundary_recovery_regressions();
        std::cout<<"Double comparisons: "<<compared_double_cases<<", maximum objective difference="<<largest_double_difference<<", maximum certified gap="<<largest_double_gap<<", local recoveries="<<total_anchor_recoveries<<'\n';
        std::cout<<"Cycle solver tests passed.\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
