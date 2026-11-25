#include "gmp_pool.h"

#if defined(TPP_HAS_GMP_POOL) && defined(TPP_USE_GMP_RATIONAL)
#include <gmp.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace tpp::detail {
namespace {
// Exact rationals allocate and free a few small limb arrays per operation.
// Blocks of up to kClasses*kGrain bytes return to a free list of this thread
// instead of to malloc. GMP passes each block's requested size to free and
// realloc, and every small block comes from allocate below, rounded up to its
// class, so that size names the class; hence the precondition in gmp_pool.h.
// (A free that understates the size, as Boost's string buffers do, only puts
// the block in a smaller class.) A block freed on another thread joins that
// thread's list (all are plain malloc blocks). The lists are trivially
// destructible, so a free after thread-local destruction stays valid; each
// list keeps at most kKeep blocks.
constexpr size_t kGrain=16,kClasses=32,kKeep=512;
struct Pool {
    std::array<void *,kClasses> head{};
    std::array<size_t,kClasses> count{};
};
thread_local Pool pool;

size_t class_of(size_t bytes) {return (bytes+kGrain-1)/kGrain;}
void *fresh(size_t bytes) {
    void *p=std::malloc(bytes?bytes:1);
    if(!p)std::abort(); // GMP's contract: allocation never returns null
    return p;
}
void *allocate(size_t bytes) {
    const size_t c=class_of(bytes);
    if(c==0||c>kClasses)return fresh(bytes);
    auto &head=pool.head[c-1];
    if(void *p=head) {
        void *next;std::memcpy(&next,p,sizeof next);
        head=next;--pool.count[c-1];
        return p;
    }
    return fresh(c*kGrain);
}
void release(void *p,size_t bytes) {
    if(!p)return;
    const size_t c=class_of(bytes);
    if(c==0||c>kClasses||pool.count[c-1]>=kKeep){std::free(p);return;}
    std::memcpy(p,&pool.head[c-1],sizeof(void *));
    pool.head[c-1]=p;++pool.count[c-1];
}
void *reallocate(void *p,size_t old_bytes,size_t new_bytes) {
    if(!p)return allocate(new_bytes);
    const size_t c=class_of(old_bytes);
    if(c>0&&c<=kClasses&&c==class_of(new_bytes))return p;
    void *q=allocate(new_bytes);
    std::memcpy(q,p,old_bytes<new_bytes?old_bytes:new_bytes);
    release(p,old_bytes);
    return q;
}
}

bool install_gmp_pool() {
    static const bool installed=[] {
        mp_set_memory_functions(allocate,reallocate,release);
        return true;
    }();
    return installed;
}
}
#else
namespace tpp::detail {
bool install_gmp_pool() {return false;}
}
#endif
