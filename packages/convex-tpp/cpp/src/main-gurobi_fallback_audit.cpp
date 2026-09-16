#include "tests.h"
#include "gurobi_c++.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Rational=boost::multiprecision::cpp_rational;
using Polygon=std::vector<Vector2>;

struct Metadata {
    std::string reason;
    double lower=0,upper=0;
};
struct Profile {
    std::string name;
    double feasibility_tol=1e-6;
    double qcp_tol=1e-6;
    int numeric_focus=0;
    double time_limit=.1;
};
struct Outcome {
    bool solved=false,exact_feasible=false,valid_1e7=false,valid_1e9=false;
    bool objective_1e7=false,objective_1e9=false,bound_safe_1e7=false,bound_safe_1e9=false;
    int status=0;
    double objective=std::numeric_limits<double>::quiet_NaN();
    double bound=std::numeric_limits<double>::quiet_NaN();
    double path_length=std::numeric_limits<double>::quiet_NaN();
    double max_distance_violation=std::numeric_limits<double>::infinity();
    double constr_violation=std::numeric_limits<double>::quiet_NaN();
    double bound_violation=std::numeric_limits<double>::quiet_NaN();
    double seconds=0;
    std::string error;
};
struct Counts {
    size_t total=0,solved=0,exact_feasible=0,valid_1e7=0,valid_1e9=0;
    size_t objective_1e7=0,objective_1e9=0,bound_safe_1e7=0,bound_safe_1e9=0;
    double max_distance_violation=0,max_objective_error=0,max_constr_violation=0;
    void add(const Outcome &o,const Metadata &m) {
        ++total;solved+=o.solved;exact_feasible+=o.exact_feasible;
        valid_1e7+=o.valid_1e7;valid_1e9+=o.valid_1e9;
        objective_1e7+=o.objective_1e7;objective_1e9+=o.objective_1e9;
        bound_safe_1e7+=o.bound_safe_1e7;bound_safe_1e9+=o.bound_safe_1e9;
        if(o.solved) {
            max_distance_violation=std::max(max_distance_violation,o.max_distance_violation);
            const double objective_error=std::max({0.,m.lower-o.path_length,o.path_length-m.upper});
            max_objective_error=std::max(max_objective_error,objective_error);
            max_constr_violation=std::max(max_constr_violation,o.constr_violation);
        }
    }
};

std::vector<Metadata> read_metadata(const std::string &path) {
    std::ifstream input(path);if(!input)throw std::runtime_error("Cannot open metadata: "+path);
    std::string line;std::getline(input,line);std::vector<Metadata> result;
    while(std::getline(input,line)) {
        std::stringstream row(line);std::string field;std::vector<std::string> fields;
        while(std::getline(row,field,','))fields.push_back(field);
        if(fields.size()<6)throw std::runtime_error("Malformed metadata row");
        result.push_back({fields[1],std::stod(fields[4]),std::stod(fields[5])});
    }
    return result;
}

Rational cross(Vector2 a,Vector2 b,Vector2 c) {
    const Rational ax=a.x,ay=a.y,bx=b.x,by=b.y,cx=c.x,cy=c.y;
    return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
}
Rational signed_area(const Polygon &p) {
    Rational area=0;
    for(size_t i=0;i<p.size();++i)area+=Rational(p[i].x)*Rational(p[(i+1)%p.size()].y)
        -Rational(p[i].y)*Rational(p[(i+1)%p.size()].x);
    return area;
}
bool exact_inside(Vector2 point,const Polygon &polygon) {
    const bool ccw=signed_area(polygon)>=0;
    for(size_t i=0;i<polygon.size();++i) {
        const Rational side=cross(polygon[i],polygon[(i+1)%polygon.size()],point);
        if((ccw&&side<0)||(!ccw&&side>0))return false;
    }
    return true;
}
double distance_violation(Vector2 point,const Polygon &polygon) {
    long double area=0;
    for(size_t i=0;i<polygon.size();++i)
        area+=(long double)polygon[i].x*polygon[(i+1)%polygon.size()].y
             -(long double)polygon[i].y*polygon[(i+1)%polygon.size()].x;
    const bool ccw=area>=0;long double maximum=0;
    for(size_t i=0;i<polygon.size();++i) {
        const auto a=polygon[i],b=polygon[(i+1)%polygon.size()];
        const long double side=((long double)b.x-a.x)*((long double)point.y-a.y)
                              -((long double)b.y-a.y)*((long double)point.x-a.x);
        const long double outside=ccw?-side:side;
        if(outside>0)maximum=std::max(maximum,outside/std::hypot((long double)b.x-a.x,(long double)b.y-a.y));
    }
    return double(maximum);
}
double path_length(Vector2 start,Vector2 target,const std::vector<Vector2> &contacts) {
    long double total=0;Vector2 previous=start;
    for(auto point:contacts){total+=std::hypot((long double)point.x-previous.x,(long double)point.y-previous.y);previous=point;}
    total+=std::hypot((long double)target.x-previous.x,(long double)target.y-previous.y);
    return double(total);
}

Outcome solve(const tpp::TestCase &c,const Metadata &metadata,const Profile &profile,GRBEnv &env) {
    Outcome out;const auto began=std::chrono::steady_clock::now();
    try {
        GRBModel model(env);model.set(GRB_IntParam_OutputFlag,0);model.set(GRB_IntParam_Presolve,0);
        model.set(GRB_IntParam_Threads,1);model.set(GRB_DoubleParam_FeasibilityTol,profile.feasibility_tol);
        model.set(GRB_DoubleParam_BarQCPConvTol,profile.qcp_tol);model.set(GRB_IntParam_NumericFocus,profile.numeric_focus);
        model.set(GRB_DoubleParam_TimeLimit,profile.time_limit);
        const size_t k=c.polygons.size();
        std::vector<std::array<GRBVar,2>> points;points.reserve(k);
        std::vector<GRBVar> lengths;lengths.reserve(k+1);
        for(size_t i=0;i<k;++i) {
            double min_x=c.polygons[i].front().x,max_x=min_x,min_y=c.polygons[i].front().y,max_y=min_y;
            for(const auto point:c.polygons[i]) {
                min_x=std::min(min_x,point.x);max_x=std::max(max_x,point.x);
                min_y=std::min(min_y,point.y);max_y=std::max(max_y,point.y);
            }
            points.push_back({model.addVar(min_x,max_x,0,GRB_CONTINUOUS),
                              model.addVar(min_y,max_y,0,GRB_CONTINUOUS)});
        }
        for(size_t i=0;i<=k;++i)lengths.push_back(model.addVar(0,GRB_INFINITY,0,GRB_CONTINUOUS));
        for(size_t i=0;i<k;++i) {
            const auto &polygon=c.polygons[i];const bool ccw=signed_area(polygon)>=0;
            for(size_t e=0;e<polygon.size();++e) {
                const auto a=polygon[e],b=polygon[(e+1)%polygon.size()];
                const double dx=b.x-a.x,dy=b.y-a.y;
                const GRBLinExpr side=dx*(points[i][1]-a.y)-dy*(points[i][0]-a.x);
                if(ccw)model.addConstr(side>=0);else model.addConstr(side<=0);
            }
        }
        auto add_distance=[&](size_t i,GRBLinExpr dx,GRBLinExpr dy) {
            model.addQConstr(dx*dx+dy*dy<=lengths[i]*lengths[i]);
        };
        add_distance(0,points[0][0]-c.start.x,points[0][1]-c.start.y);
        for(size_t i=1;i<k;++i)add_distance(i,points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]);
        add_distance(k,c.target.x-points[k-1][0],c.target.y-points[k-1][1]);
        GRBLinExpr objective=0;for(auto &length:lengths)objective+=length;model.setObjective(objective,GRB_MINIMIZE);
        model.optimize();out.status=model.get(GRB_IntAttr_Status);
        if(model.get(GRB_IntAttr_SolCount)==0) {
            out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();return out;
        }
        out.solved=true;out.objective=model.get(GRB_DoubleAttr_ObjVal);out.bound=model.get(GRB_DoubleAttr_ObjBound);
        out.constr_violation=model.get(GRB_DoubleAttr_ConstrVio);out.bound_violation=model.get(GRB_DoubleAttr_BoundVio);
        std::vector<Vector2> contacts;contacts.reserve(k);out.exact_feasible=true;out.max_distance_violation=0;
        for(size_t i=0;i<k;++i) {
            Vector2 point{points[i][0].get(GRB_DoubleAttr_X),points[i][1].get(GRB_DoubleAttr_X)};
            contacts.push_back(point);out.exact_feasible&=exact_inside(point,c.polygons[i]);
            out.max_distance_violation=std::max(out.max_distance_violation,distance_violation(point,c.polygons[i]));
        }
        out.path_length=path_length(c.start,c.target,contacts);
        const double scale=1+std::max(std::abs(metadata.lower),std::abs(metadata.upper));
        out.valid_1e7=out.max_distance_violation<=1e-7*scale;out.valid_1e9=out.max_distance_violation<=1e-9*scale;
        auto objective_ok=[&](double tolerance){return out.path_length>=metadata.lower-tolerance*scale&&out.path_length<=metadata.upper+tolerance*scale;};
        auto bound_safe=[&](double tolerance){return out.bound<=metadata.upper+tolerance*scale;};
        out.objective_1e7=objective_ok(1e-7);out.objective_1e9=objective_ok(1e-9);
        out.bound_safe_1e7=bound_safe(1e-7);out.bound_safe_1e9=bound_safe(1e-9);
    } catch(const GRBException &e) {out.error=std::to_string(e.getErrorCode())+":"+e.getMessage();}
      catch(const std::exception &e) {out.error=e.what();}
    out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();return out;
}

void print_counts(const std::string &profile,const std::string &reason,const Counts &c) {
    std::cout<<profile<<','<<reason<<','<<c.total<<','<<c.solved<<','<<c.exact_feasible<<','
             <<c.valid_1e7<<','<<c.valid_1e9<<','<<c.objective_1e7<<','<<c.objective_1e9<<','
             <<c.bound_safe_1e7<<','<<c.bound_safe_1e9<<','<<c.max_distance_violation<<','
             <<c.max_objective_error<<','<<c.max_constr_violation<<'\n';
}
}

int main(int argc,char **argv) {
    if(argc<3||argc>6) {
        std::cerr<<"Usage: "<<argv[0]<<" FALLBACKS.bin FALLBACKS.bin.csv [details.csv [indices [time-limit]]]\n";return 2;
    }
    const auto cases=tpp::load_test_cases(argv[1]);const auto metadata=read_metadata(argv[2]);
    if(cases.size()!=metadata.size())throw std::runtime_error("Corpus/metadata size mismatch");
    std::ofstream details;if(argc>=4)details.open(argv[3]);
    details<<"profile,index,reason,status,solved,exact_feasible,valid_1e7,valid_1e9,objective_1e7,objective_1e9,bound_safe_1e7,bound_safe_1e9,path_length,gurobi_objective,gurobi_bound,rational_lower,rational_upper,max_distance_violation,constr_violation,bound_violation,seconds,error\n";
    std::vector<bool> selected(cases.size(),argc<5);
    if(argc>=5) {
        std::stringstream list(argv[4]);std::string value;
        while(std::getline(list,value,',')) {
            const size_t index=std::stoull(value);
            if(index>=cases.size())throw std::out_of_range("Audit case index");
            selected[index]=true;
        }
    }
    const double time_limit=argc>=6?std::stod(argv[5]):.1;
    const std::vector<Profile> profiles={{"default",1e-6,1e-6,0,time_limit},{"tight",1e-9,1e-9,0,time_limit},{"focus3",1e-9,1e-9,3,time_limit}};
    GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
    std::cout<<std::setprecision(17);
    std::cout<<"profile,reason,total,solved,exact_feasible,valid_1e7,valid_1e9,objective_1e7,objective_1e9,bound_safe_1e7,bound_safe_1e9,max_distance_violation,max_objective_error,max_constr_violation\n";
    for(const auto &profile:profiles) {
        std::map<std::string,Counts> groups;Counts all;
        for(size_t i=0;i<cases.size();++i) {
            if(!selected[i])continue;
            const auto outcome=solve(cases[i],metadata[i],profile,env);groups[metadata[i].reason].add(outcome,metadata[i]);all.add(outcome,metadata[i]);
            if(details)details<<profile.name<<','<<i<<','<<metadata[i].reason<<','<<outcome.status<<','<<outcome.solved<<','
                <<outcome.exact_feasible<<','<<outcome.valid_1e7<<','<<outcome.valid_1e9<<','<<outcome.objective_1e7<<','
                <<outcome.objective_1e9<<','<<outcome.bound_safe_1e7<<','<<outcome.bound_safe_1e9<<','<<outcome.path_length<<','
                <<outcome.objective<<','<<outcome.bound<<','<<metadata[i].lower<<','<<metadata[i].upper<<','
                <<outcome.max_distance_violation<<','<<outcome.constr_violation<<','<<outcome.bound_violation<<','<<outcome.seconds<<','<<outcome.error<<std::endl;
            if((i+1)%25==0)std::cerr<<profile.name<<": "<<(i+1)<<'/'<<cases.size()<<'\n';
        }
        print_counts(profile.name,"all",all);for(const auto &[reason,counts]:groups)print_counts(profile.name,reason,counts);
    }
}
