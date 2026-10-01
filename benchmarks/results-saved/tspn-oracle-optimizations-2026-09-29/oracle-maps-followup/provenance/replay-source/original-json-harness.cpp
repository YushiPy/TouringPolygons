#include "tpp/convex/cycle.h"
#include "tpp/convex/cycle_certificate.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
using json=nlohmann::json;
int main(int argc,char**argv){
  if(argc!=3)return 2;
  std::ifstream f(argv[1]);json j;f>>j;
  std::vector<std::vector<Vector2>> original;
  for(const auto& poly:j.at("regions")){std::vector<Vector2> p;for(const auto& v:poly)p.emplace_back(v.at(0).get<double>(),v.at(1).get<double>());original.push_back(std::move(p));}
  std::vector<Vector2> seed;for(const auto& v:j.at("initial_contacts"))seed.emplace_back(v.at(0).get<double>(),v.at(1).get<double>());
  const std::size_t n=original.size();std::vector<std::size_t> order(n);
  for(std::size_t i=0;i<n;++i)order[i]=i;
  const std::string mode=argv[2];
  if(mode=="reverse")for(std::size_t i=1;i<n;++i)order[i]=n-i;
  else if(mode=="rotate"||mode=="exact-rotate")for(std::size_t i=0;i<n;++i)order[i]=(i+n/2)%n;
  std::vector<std::vector<Vector2>> regions;std::vector<Vector2> warm;
  for(auto i:order){regions.push_back(original[i]);warm.push_back(seed[i]);}
  if(mode=="exact-original"||mode=="exact-rotate") {
    tpp::ConvexCycleOptions exact_options;
    for(auto v:warm) exact_options.initial_contacts.emplace_back(v);
    std::cerr<<"EXACT_REPLAY_BEGIN mode="<<mode<<" regions="<<n<<std::endl;
    const auto began=std::chrono::steady_clock::now();auto exact=tpp::tpp_convex_solve_cycle(regions,exact_options);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    tpp::ConvexRationalPolygons exact_original;for(const auto &poly:original){tpp::ConvexRationalPolygon p;for(auto v:poly)p.emplace_back(v);exact_original.push_back(std::move(p));}
    tpp::ConvexRationalPolygon mapped(n);for(std::size_t i=0;i<n&&i<exact.contacts.size();++i)mapped[order[i]]=exact.contacts[i];
    const auto cert=tpp::tpp_convex_verify_cycle_certificate(exact_original,mapped);
    json out={{"mode",mode},{"seconds",seconds},{"status",static_cast<int>(exact.status)},{"internal_lower_bound",exact.certificate.lower_bound},{"internal_upper_bound",exact.certificate.upper_bound},{"original_exact_validation_status",static_cast<int>(cert.status)},{"original_exact_lower_bound",cert.lower_bound},{"original_exact_upper_bound",cert.upper_bound},{"contacts",exact.contacts.size()},{"oracle_calls",exact.oracle_calls},{"certificate_checks",exact.certificate_checks},{"diagnostic",exact.diagnostic}};
    std::cout<<out.dump()<<std::endl;return 0;
  }
  tpp::ConvexCycleDoubleOptions options;options.lower_bound_cutoff=j.at("cutoff").get<double>();options.initial_contacts=warm;
  std::cerr<<"REPLAY_BEGIN mode="<<mode<<" regions="<<n<<" cutoff="<<options.lower_bound_cutoff<<std::endl;
  const auto began=std::chrono::steady_clock::now();auto result=tpp::tpp_convex_solve_cycle_double(regions,options);
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
  std::vector<Vector2> contacts(n);for(std::size_t i=0;i<n&&i<result.contacts.size();++i)contacts[order[i]]=result.contacts[i];
  const auto cert=tpp::tpp_convex_verify_cycle_certificate(original,contacts);
  json out={{"mode",mode},{"seconds",seconds},{"status",static_cast<int>(result.status)},{"lower_bound",result.certificate.lower_bound},{"upper_bound",result.certificate.upper_bound},{"exact_validation_status",static_cast<int>(cert.status)},{"exact_validation_lower_bound",cert.lower_bound},{"exact_validation_upper_bound",cert.upper_bound},{"contacts",contacts.size()},{"oracle_calls",result.oracle_calls},{"certificate_checks",result.certificate_checks},{"rational_anchor_recoveries",result.rational_anchor_recoveries},{"rational_cycle_recoveries",result.rational_cycle_recoveries},{"rational_feature_recoveries",result.rational_feature_recoveries},{"diagnostic",result.diagnostic}};
  std::cout<<out.dump()<<std::endl;
}
