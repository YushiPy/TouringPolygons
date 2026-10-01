#pragma once
#include "tpp/convex/cycle.h"
#include <array>
#include <chrono>
#include <cmath>

namespace tpp::detail {
// Deliberately separate from construction exceptions: a budget interruption
// must escape every arithmetic-recovery catch without restarting more work.
struct CycleInterrupted {};
enum class CyclePhase { Construction, Certification, RationalRecovery };
struct CycleExecution {
    using Clock = std::chrono::steady_clock;
    Clock::time_point began = Clock::now(), since = began;
    double max_seconds;
    const std::function<bool()> &stop;
    CyclePhase phase = CyclePhase::Construction;
    std::array<double,3> seconds{};
    void account() {
        const auto now=Clock::now();
        seconds[static_cast<size_t>(phase)]+=std::chrono::duration<double>(now-since).count();
        since=now;
    }
};
inline thread_local CycleExecution *active_cycle_execution = nullptr;
inline void cycle_checkpoint() {
    const auto *work=active_cycle_execution;
    if(!work)return;
    if((work->stop&&work->stop()) || (std::isfinite(work->max_seconds)&&
       std::chrono::duration<double>(CycleExecution::Clock::now()-work->began).count()>=work->max_seconds))
        throw CycleInterrupted{};
}
class CyclePhaseScope {
    CycleExecution *work=active_cycle_execution;
    CyclePhase previous{};
public:
    explicit CyclePhaseScope(CyclePhase phase) {
        if(work&&work->phase==phase)work=nullptr;
        if(work){work->account();previous=work->phase;work->phase=phase;}
    }
    ~CyclePhaseScope(){if(work){work->account();work->phase=previous;}}
};
template<class Result,class Options,class Run>
Result run_cycle_execution(Result &result,const Options &options,Run run) {
    // Nested recovery shares the outer deadline and exclusive phase clock.
    if(active_cycle_execution)return run();
    if(std::isnan(options.max_seconds)||options.max_seconds<0)return result;
    CycleExecution work{.max_seconds=options.max_seconds,.stop=options.stop_requested};
    struct Guard {
        explicit Guard(CycleExecution &work){active_cycle_execution=&work;}
        ~Guard(){active_cycle_execution=nullptr;}
    } guard(work);
    try {cycle_checkpoint();result=run();}
    catch(const CycleInterrupted &) {
        result.status=ConvexCycleStatus::Interrupted;
        if(result.contacts.empty())result.certificate.upper_bound=std::numeric_limits<double>::infinity();
        result.diagnostic="Cooperative cycle interruption; only completed certificates retained";
    }
    work.account();
    result.timings={work.seconds[0],work.seconds[1],work.seconds[2]};
    return std::move(result);
}
} // namespace tpp::detail
