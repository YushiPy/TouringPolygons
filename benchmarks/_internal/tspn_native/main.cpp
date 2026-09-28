// Thin native adapter for the pinned external solver; no external source edits.
#include "tspn_core/bnb.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>

int main(int argc,char **argv) {
    try {
        if(argc!=5)throw std::invalid_argument("expected repetitions relative-gap feasibility-tolerance spanning-tolerance");
        const int repeats=std::stoi(argv[1]);const double gap=std::stod(argv[2]),feas=std::stod(argv[3]);
        double sx,sy,tx,ty,seconds;size_t n,calls;
        if(!(std::cin>>sx>>sy>>tx>>ty>>n>>calls>>seconds))throw std::invalid_argument("instance header");
        std::vector<tspn::SiteVariant> sites;
        for(size_t i=0;i<n;++i) {
            size_t m;std::cin>>m;std::vector<tspn::Point> ring;
            for(size_t j=0;j<m;++j){double x,y;std::cin>>x>>y;ring.emplace_back(x,y);}
            tspn::Polygon polygon;polygon.outer().assign(ring.begin(),ring.end());
            boost::geometry::correct(polygon);sites.push_back(std::move(polygon));
        }
        tspn::constants::set_float_parameter("FEASIBILITY_TOLERANCE",feas);
        tspn::constants::set_float_parameter("SPANNING_TOLERANCE",std::stod(argv[4]));
        // Initialize the commercial runtime on an unrelated tiny instance.
        // The actual solve includes Instance preprocessing, root, B&B and output
        // extraction, matching the whole-call timing used for our solver.
        {
            std::vector<tspn::SiteVariant> warm_sites{tspn::Polygon{{{0,0},{1,0},{1,1},{0,1}}},
                                                     tspn::Polygon{{{3,0},{4,0},{4,1},{3,1}}}};
            tspn::Instance warm(warm_sites);
            tspn::SocSolver solver(false,"socp");
            solver.compute_trajectory({tspn::TourElement(warm,0),tspn::TourElement(warm,1)},false);
        }
        for(int repeat=0;repeat<repeats;++repeat) {
            const auto began=std::chrono::steady_clock::now();
            nlohmann::json row;
            {
                tspn::Instance instance(sites,false);
                tspn::SocSolver soc(true,"socp");
                tspn::LongestEdgePlusFurthestSite root(false);
                tspn::FarthestPoly branching(true,false,1,false,false);
                tspn::DfsBfs search(false);
                tspn::BranchAndBoundAlgorithm bnb(&instance,root.get_root_node(instance,soc),branching,search);
                bnb.set_ub_callback(std::bind_front(&tspn::SocSolver::update_cutoff,&soc));
                soc.set_time_limit(seconds);bnb.optimize(static_cast<int>(seconds),gap,false);
                row={{"repeat",repeat},{"lower_bound",bnb.get_lower_bound()},{"upper_bound",bnb.get_upper_bound()},
                     {"statistics",bnb.get_statistics()},{"calls",soc.get_num_calls()},
                     {"oracle_seconds",soc.get_total_solve_nanoseconds()*1e-9},
                     {"backend","socp"},{"path",nlohmann::json::array()}};
                if(auto solution=bnb.get_solution()) {
                    const auto &trajectory=solution->get_trajectory();
                    for(const auto &point:trajectory.points)row["path"].push_back({point.get<0>(),point.get<1>()});
                    row["is_tour"]=trajectory.is_tour();
                }
            }
            row["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
            std::cout<<row.dump()<<std::endl;
        }
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
