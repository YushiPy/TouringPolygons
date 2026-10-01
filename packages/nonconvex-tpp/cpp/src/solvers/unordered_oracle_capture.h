#pragma once
#include <atomic>
#include <fstream>
#include <iomanip>
#include <mutex>

namespace tpp::unordered_detail {
// Optional diagnostic stream. Begin is flushed before construction so a firm
// process timeout still leaves a replayable input. Never enabled by default.
class OracleCapture {
    std::ofstream stream;
    inline static std::mutex mutex;
    inline static std::atomic<size_t> next_id{0};
    static void number(std::ostream &out,double value) {
        if(std::isfinite(value))out<<value;else out<<"null";
    }
    static void points(std::ostream &out,const Polygon &p) {
        out<<'[';
        for(size_t i=0;i<p.size();++i)out<<(i?",":"")<<'['<<p[i].x<<','<<p[i].y<<']';
        out<<']';
    }
public:
    explicit OracleCapture(const std::string &path) {
        if(path.empty())return;
        stream.open(path,std::ios::app);
        if(!stream)throw std::runtime_error("Cannot open oracle capture file");
        stream<<std::setprecision(17);
    }
    size_t begin(size_t node,bool precise,const std::vector<Polygon> &regions,
            const Polygon &initial,const std::vector<int> &features,double cutoff,double tolerance,double seconds,const UnorderedTppSolveOptions &options) {
        if(!stream.is_open())return 0;
        std::lock_guard lock(mutex);const size_t id=++next_id;
        stream<<"{\"event\":\"begin\",\"id\":"<<id<<",\"node\":"<<node<<",\"precise\":"<<(precise?"true":"false")
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
              <<",\"bound_first\":"<<(options.cycle_bound_first?"true":"false");
        stream<<"}\n";stream.flush();return id;
    }
    template<class Result> void end(size_t id,const Result &result) {
        if(!stream.is_open())return;
        std::lock_guard lock(mutex);
        stream<<"{\"event\":\"end\",\"id\":"<<id<<",\"seconds\":"<<result.seconds
              <<",\"interrupted\":"<<(result.time_limited?"true":"false")
              <<",\"construction_seconds\":"<<result.cycle_timings.construction_seconds
              <<",\"certification_seconds\":"<<result.cycle_timings.certification_seconds
              <<",\"rational_recovery_seconds\":"<<result.cycle_timings.rational_recovery_seconds
              <<",\"lower_bound\":";number(stream,result.lower_bound);
        stream<<",\"upper_bound\":";number(stream,result.upper_bound);
        stream<<"}\n";stream.flush();
    }
};
} // namespace tpp::unordered_detail
