#include "tpp/nonconvex/unordered.h"
#include "unordered_cycle_oracle.h"

#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using Polygon = std::vector<::Vector2>;
using Polygons = std::vector<Polygon>;
using boost::property_tree::ptree;

double optional_number(const ptree &tree, const char *key, double fallback) {
	const auto value=tree.get_optional<double>(key);
	return value ? *value : fallback;
}

::Vector2 point(const ptree &tree) {
	auto coordinate=tree.begin();
	if(coordinate==tree.end())throw std::runtime_error("empty point in capture");
	const double x=coordinate++->second.get_value<double>();
	if(coordinate==tree.end())throw std::runtime_error("point missing y coordinate");
	return {x,coordinate->second.get_value<double>()};
}

Polygons polygons(const ptree &tree) {
	Polygons result;
	for(const auto &entry:tree) {
		Polygon polygon;
		for(const auto &vertex:entry.second)polygon.push_back(point(vertex.second));
		result.push_back(std::move(polygon));
	}
	return result;
}

std::vector<::Vector2> contacts(const ptree &tree) {
	std::vector<::Vector2> result;
	for(const auto &entry:tree)result.push_back(point(entry.second));
	return result;
}

std::vector<int> features(const ptree &tree) {
	std::vector<int> result;
	for(const auto &entry:tree)result.push_back(entry.second.get_value<int>());
	return result;
}

bool json_bool(const ptree &tree, const char *key, bool fallback) {
	const auto value=tree.get_optional<std::string>(key);
	if(!value)return fallback;
	return *value=="true"||*value=="1";
}

void run(const ptree &input, std::size_t repeat, double seconds,
		bool cache_default, bool features_default, bool interval_default, bool bound_first_default) {
	const auto id=input.get<std::size_t>("id");
	const bool precise=json_bool(input,"precise",false);
	const bool use_cache=json_bool(input,"cache",cache_default);
	const bool use_features=json_bool(input,"features",features_default);
	const bool use_interval=json_bool(input,"interval",interval_default);
	const bool bound_first=json_bool(input,"bound_first",bound_first_default);
	const bool proposal_bound=json_bool(input,"proposal_bound",false)&&!precise;
	const Polygons regions=polygons(input.get_child("polygons"));
	const auto initial_contacts=contacts(input.get_child("initial_contacts"));
	const auto initial_features=features(input.get_child("initial_features"));
	const double cutoff=optional_number(input,"lower_bound_cutoff",std::numeric_limits<double>::infinity());
	const double tolerance=optional_number(input,"tolerance",0.0);
	const auto began=std::chrono::steady_clock::now();
	tpp::DynamicConvexTppWorkspace oracle_workspace;
	tpp::ConvexCycleWorkspace cycle_workspace;
	const auto solved=tpp::solve_relaxation(true,::Vector2{},::Vector2{},regions,oracle_workspace,
		tolerance,cutoff,seconds,initial_contacts,use_cache?&cycle_workspace:nullptr,
		use_features?initial_features:std::vector<int>{},use_features,bound_first,use_interval,{},proposal_bound);
	const double wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
	std::vector<::Vector2> returned_contacts=solved.path;
	if(returned_contacts.size()==regions.size()+1&&!returned_contacts.empty()&&
		returned_contacts.front()==returned_contacts.back())returned_contacts.pop_back();
	const double lower=solved.lower_bound,upper=solved.upper_bound;
	std::string status=solved.time_limited?"interrupted":solved.dual_cutoff_pruned?"certified_bound":"completed";
	const bool gap_satisfied=std::isfinite(lower)&&std::isfinite(upper)&&lower<=upper&&upper-lower<=tolerance;

	// Feasibility and interval validation run outside the measured shared adapter.
	ptree independent,filtered;
	bool certificate_valid=false,intervals_overlap=false;
	if(returned_contacts.size()==regions.size()&&!returned_contacts.empty()) {
		const auto verified=tpp::tpp_convex_verify_cycle_certificate(regions,returned_contacts);
		const auto filtered_result=tpp::tpp_convex_verify_cycle_certificate(regions,returned_contacts,
			std::numeric_limits<double>::infinity(),true);
		certificate_valid=verified.status==tpp::ConvexCycleCertificateStatus::Feasible||
			verified.status==tpp::ConvexCycleCertificateStatus::Optimal;
		intervals_overlap=certificate_valid&&verified.lower_bound<=upper&&lower<=verified.upper_bound&&
			filtered_result.lower_bound<=verified.upper_bound&&verified.lower_bound<=filtered_result.upper_bound;
		independent.put("status",static_cast<int>(verified.status));
		independent.put("lower_bound",verified.lower_bound);independent.put("upper_bound",verified.upper_bound);
		independent.put("exact_predicate_evaluations",verified.exact_predicate_evaluations);
		filtered.put("status",static_cast<int>(filtered_result.status));
		filtered.put("lower_bound",filtered_result.lower_bound);filtered.put("upper_bound",filtered_result.upper_bound);
		filtered.put("interval_bounds_used",filtered_result.interval_bounds_used);
	} else if(status!="interrupted"||!returned_contacts.empty()) status="oracle_failure";
	if(!returned_contacts.empty()&&(!certificate_valid||!intervals_overlap))status="oracle_failure";

	ptree row;row.put("call_id",id);row.put("repeat",repeat);row.put("status",status);
	row.put("precise",precise);row.put("capture_tolerance",tolerance);row.put("deadline_seconds",seconds);
	row.put("cache",use_cache);row.put("features",use_features);row.put("interval",use_interval);row.put("bound_first",bound_first);
	row.put("wall_seconds",wall_seconds);row.put("construction_seconds",solved.cycle_timings.construction_seconds);
	row.put("certification_seconds",solved.cycle_timings.certification_seconds);
	row.put("rational_recovery_seconds",solved.cycle_timings.rational_recovery_seconds);
	row.put("gap_satisfied",gap_satisfied);row.put("upper_bound_infinite",std::isinf(upper));
	row.put("lower_bound",lower);row.put("upper_bound",upper);row.put("lower_bound_cutoff",cutoff);
	row.put("used_fallback",solved.used_fallback);row.put("dual_cutoff_pruned",solved.dual_cutoff_pruned);
	row.put("proposal_bound",proposal_bound);row.put("proposal_calls",solved.proposal_calls);row.put("proposal_accepts",solved.proposal_accepts);
	row.put("independent_certificate_valid",certificate_valid);row.put("intervals_overlap",intervals_overlap);
	row.add_child("independent_certificate",independent);row.add_child("filtered_certificate",filtered);
	ptree path;
	for(const auto &contact:returned_contacts) {
		ptree item,x,y;x.put("",contact.x);y.put("",contact.y);
		item.push_back({"",x});item.push_back({"",y});path.push_back({"",item});
	}
	row.add_child("contacts",path);
	boost::property_tree::write_json(std::cout,row,false);
	std::cout<<std::endl;
}
}

int main(int argc,char **argv) {
	try {
		if(argc!=8)throw std::invalid_argument("expected input-jsonl seconds repetitions cache features interval bound-first");
		const std::string input_path=argv[1];
		const double seconds=std::stod(argv[2]);
		const std::size_t repetitions=std::stoul(argv[3]);
		const bool use_cache=std::stoi(argv[4])!=0,use_features=std::stoi(argv[5])!=0;
		const bool use_interval=std::stoi(argv[6])!=0,bound_first=std::stoi(argv[7])!=0;
		if(!(seconds>0)||!std::isfinite(seconds)||repetitions==0)throw std::invalid_argument("seconds and repetitions must be positive");
		std::ifstream input(input_path);if(!input)throw std::runtime_error("cannot open replay input");
		std::string line;
		while(std::getline(input,line)) {
			if(line.empty())continue;
			std::istringstream stream(line);ptree record;boost::property_tree::read_json(stream,record);
			if(record.get<std::string>("event")!="begin")throw std::runtime_error("replay input must contain begin records");
			for(std::size_t repeat=0;repeat<repetitions;++repeat)
				run(record,repeat,seconds,use_cache,use_features,use_interval,bound_first);
		}
	} catch(const std::exception &error) {
		std::cerr<<error.what()<<'\n';return 1;
	}
	return 0;
}
