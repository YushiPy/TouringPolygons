#pragma once

namespace tpp::detail {
// Installs per-thread free lists of small blocks as GMP's memory functions
// (TPP_HAS_GMP_POOL with the GMP backend; otherwise it does nothing and
// returns false). Idempotent. Call it first in main, before anything in the
// process allocates with GMP (including other libraries, such as CGAL or
// MPFR): the pool takes each block's class from the size GMP reports, which
// is only right for blocks it allocated itself. Meant for programs whose
// threads live for the whole run: blocks cached by a thread that exits are
// not returned. The library never installs it by itself.
bool install_gmp_pool();
}
