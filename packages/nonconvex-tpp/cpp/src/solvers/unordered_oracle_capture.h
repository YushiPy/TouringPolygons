#pragma once
#include <atomic>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace tpp::unordered_detail {
// Optional diagnostic stream. Begin is flushed before construction so a firm
// process timeout still leaves a replayable input. Never enabled by default.
// With sampling, a call is written (begin and end together) after it returns.
class OracleCapture {
    std::ofstream stream;
    std::size_t every=1;
    double min_seconds=0;
    std::unordered_map<size_t,std::string> pending;
    inline static std::mutex mutex;
    inline static std::atomic<size_t> next_id{0};
    bool sampling() const {return every!=1||min_seconds>0;}
    static void number(std::ostream &out,double value) {
        if(std::isfinite(value))out<<value;else out<<"null";
    }
    static void points(std::ostream &out,const Polygon &p) {
        out<<'[';
        for(size_t i=0;i<p.size();++i)out<<(i?",":"")<<'['<<p[i].x<<','<<p[i].y<<']';
        out<<']';
    }
public:
    explicit OracleCapture(const std::string &path,std::size_t every_call=1,double slow_seconds=0)
        :every(every_call),min_seconds(slow_seconds) {
        if(path.empty())return;
        stream.open(path,std::ios::app);
        if(!stream)throw std::runtime_error("Cannot open oracle capture file");
        stream<<std::setprecision(17);
    }
    size_t begin(size_t node,bool precise,const Vector2 &start,const Vector2 &target,const std::vector<Polygon> &regions,
            const Polygon &initial,const std::vector<int> &features,double cutoff,double tolerance,double seconds,const UnorderedTppSolveOptions &options) {
        if(!stream.is_open())return 0;
        std::lock_guard lock(mutex);const size_t id=++next_id;
        std::ostringstream buffer;buffer<<std::setprecision(17);
        std::ostream &stream=sampling()?static_cast<std::ostream&>(buffer):this->stream;
        stream<<"{\"event\":\"begin\",\"id\":"<<id<<",\"node\":"<<node<<",\"precise\":"<<(precise?"true":"false")
              <<",\"start\":["<<start.x<<','<<start.y<<"],\"target\":["<<target.x<<','<<target.y<<']'
              <<",\"polygons\":["; 
        for(size_t i=0;i<regions.size();++i){if(i)stream<<',';points(stream,regions[i]);}
        stream<<"],\"initial_contacts\":";points(stream,initial);
        stream<<",\"initial_features\":[";
        for(size_t i=0;i<features.size();++i)stream<<(i?",":"")<<features[i];
        stream<<"],\"lower_bound_cutoff\":";number(stream,cutoff);
        stream<<",\"tolerance\":";number(stream,tolerance);
        stream<<",\"max_seconds\":";number(stream,seconds);
        stream<<",\"cache\":"<<(options.cycle_cache?"true":"false")
              <<",\"features\":"<<(options.cycle_active_features?"true":"false")
              <<",\"interval\":"<<(options.cycle_interval_certificate?"true":"false")
              <<",\"proposal_bound\":"<<(options.cycle_proposal_bound&&!precise?"true":"false")
              <<",\"bound_first\":"<<(options.cycle_bound_first?"true":"false");
        stream<<"}\n";
        if(sampling())pending.emplace(id,buffer.str());else stream.flush();
        return id;
    }
    template<class Result> void end(size_t id,const Result &result) {
        if(!stream.is_open())return;
        std::lock_guard lock(mutex);
        if(sampling()) {
            const auto found=pending.find(id);
            if(found==pending.end())return;
            const bool keep=(every&&id%every==0)||(min_seconds>0&&result.seconds>=min_seconds);
            if(keep)stream<<found->second;
            pending.erase(found);
            if(!keep)return;
        }
        stream<<"{\"event\":\"end\",\"id\":"<<id<<",\"seconds\":"<<result.seconds
              <<",\"interrupted\":"<<(result.time_limited?"true":"false")
              <<",\"construction_seconds\":"<<result.cycle_timings.construction_seconds
              <<",\"certification_seconds\":"<<result.cycle_timings.certification_seconds
              <<",\"rational_recovery_seconds\":"<<result.cycle_timings.rational_recovery_seconds
              <<",\"lower_bound\":";number(stream,result.lower_bound);
        stream<<",\"upper_bound\":";number(stream,result.upper_bound);
        stream<<",\"proposal_calls\":"<<result.proposal_calls<<",\"proposal_accepts\":"<<result.proposal_accepts
              <<",\"used_interval_bounds\":"<<(result.used_interval_bounds?"true":"false")
              <<",\"used_fallback\":"<<(result.used_fallback?"true":"false")
              <<",\"dual_cutoff_pruned\":"<<(result.dual_cutoff_pruned?"true":"false")
              <<",\"fallback_reason\":"<<static_cast<int>(result.fallback_reason);
        stream<<"}\n";stream.flush();
    }
};
} // namespace tpp::unordered_detail
