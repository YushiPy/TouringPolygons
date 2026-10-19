# Touring Polygons Problem

Exact algorithms and experiments for two shortest-route problems that visit
regions in the plane:

- **Touring Polygons Problem (TPP):** given a start point, an end point and
  polygonal regions, find the shortest Euclidean path that touches all of them,
  in a fixed or a free order;
- **TSPN** (*Traveling Salesman Problem with Neighborhoods*): the shortest
  closed tour that touches every region, in free order and with no fixed point.

Regions may be non-convex, touch each other or overlap; the TSPN also accepts
points and segments. The search is a branch and bound whose lower and upper
bounds are proved. "Solved" means a closed numerical gap (for example,
`UB ≤ 1.001·LB`). Algebraic optimality, with zero gap, uses the exact
rational-arithmetic path.

Some internal notes, experiment logs and reports are in Portuguese; links to
them are marked below.

## Core

The maintained core is C++ and follows this dependency chain:

```text
tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex
```

- `tpp_geometry`: shared geometric primitives;
- `tpp_convex`: fixed-order paths and cycles through convex polygons, built on
  the last-step maps of Dror et al. and extended to intersecting polygons. The
  oracle builds the solution in `double`, proves `L ≤ OPT ≤ U` with interval
  arithmetic, falls back to an interior-point polish when the proof does not
  close, and keeps the exact rational path with a KKT certificate as a last
  resort;
- `optimal_convex_partition`: optimal convex decomposition of simple polygons
  (our own implementation, without CGAL, of CGAL's
  `optimal_convex_partition_2`);
- `tpp_nonconvex`: fixed order with convex decomposition, and the free-order
  branch and bound for the TPP and the TSPN.

The maintained dashboard is `apps/benchmark-dashboard`. `apps/siicusp34` is the
static, self-contained page of the 34th SIICUSP (in Portuguese).

## Main results

Each item summarizes a section of
[`benchmarks/results-saved/README.md`](benchmarks/results-saved/README.md) (in
Portuguese), which records the formulation, tolerances and limitations.

- **TSPN, the 558 instances of Fekete et al., against their original code.**
  Same machine, 0.1% gap, 300 s per run, 3 repetitions, each solver by its own
  stopping criterion. Their unmodified code in the paper's configuration (CHR
  root, 16 threads) solves 542/558, the 97.13% their paper reports; ours, with
  one thread, solves 557/558. Allowing 900 s for the instances open at 300 s,
  ours is faster on every instance both solve: a geometric mean of 21× against
  their 16 threads (549 instances), and 73× against their one-thread runs (541)
  ([`fekete-reference-armijo-2026-10-17`](benchmarks/results-saved/README.md#fekete-reference-armijo-2026-10-17)).
- **Free-order TPP with fixed endpoints, same instances.** 558/558 solved,
  against 553/558. On the 553 common cases, the geometric mean is 65× (from
  8.7× to 2,302×). Their times come from another campaign on the same machine,
  not side by side ([`float-recovery-dantzig-2026-10-08`](benchmarks/results-saved/README.md#float-recovery-dantzig-2026-10-08)).
- **Paula's TSPN collection (235 instances).** 229 certified with a relative gap
  of 10⁻⁶: on them, Paula's best tour is optimal within that tolerance. The data
  is local, without permission to redistribute
  ([`tspn-paula-long-armijo-2026-10-14`](benchmarks/results-saved/README.md#tspn-paula-long-armijo-2026-10-14)).

## Layout

```text
apps/                       maintained dashboard and the SIICUSP page
benchmarks/
├── tpp.py                  the only public benchmark CLI
├── _internal/              importable implementation of the CLI
├── suites/                 canonical input sets
├── results-saved/          result summaries and small fixtures
└── workspace/              generated local data (ignored; see benchmarks/README.md)
docs/
├── algorithms/             contracts, algorithms and correctness arguments
└── reports/                project reports
packages/                   maintained libraries and solvers
scripts/                    installation and verification
third_party/tspn-socg/      fork of the Fekete et al. solver, pinned as a submodule
```

See [`docs/architecture.md`](docs/architecture.md) for component
responsibilities and lifecycle, and [`docs/third-party.md`](docs/third-party.md)
for provenance, licenses and external data.

## Installation and verification

Clone with the external solver:

```bash
git clone --recurse-submodules https://github.com/YushiPy/TouringPolygons.git
cd TouringPolygons
./scripts/install_dependencies.sh
./scripts/sanity_check.sh
```

The installer is idempotent and never uses `sudo`. Whatever is missing and can
be obtained without an administrator is downloaded into the home directory or
the repository: uv, Python 3.12, the benchmarks' Python environment, C++
headers, and the original Fekete et al. solver with its packages. What needs an
administrator (compiler, `git`, `make`) is only listed, and the script stops.
On macOS it uses Homebrew when present; without it, our solver is skipped,
because it needs `libomp`. When Node.js is present, it also sets up the
dashboard and the Chromium used by the browser tests.

`sanity_check` builds the solvers and runs the tests in about a minute once the
build exists. To check an already prepared checkout without trying to install
packages, or to include the generated stress suites (about 25 more minutes):

```bash
./scripts/sanity_check.sh --no-install
./scripts/sanity_check.sh --full
```

## Benchmarks

`benchmarks/tpp.py` is the public interface. To set up only its Python
environment, run `python3 benchmarks/tpp.py setup`. The CLI uses
`benchmarks/.venv` automatically, with dependencies pinned in `uv.lock`.
Example synthetic campaign:

```bash
python3 benchmarks/tpp.py create smoke \
  --vertices 8 --polygons 20 --instances 100 --shape star
python3 benchmarks/tpp.py run smoke \
  --threads 8 --max-calls 1000000 --max-seconds 30
python3 benchmarks/tpp.py status smoke
```

`bench --problem` assembles a fixed-order TPP, free-order TPP or TSPN run from
the same options. `free-compare` prepares and runs the free-order comparison
with the Fekete et al. solver, and `paula-campaign` runs the long TSPN campaign
on Paula's collection. Use `python3 benchmarks/tpp.py --help` for the other
commands. New results and raw data are local by default;
`benchmarks/results-saved/` keeps concise summaries and minimal fixtures, and
`fekete-comparison` is the exception with a full corpus, because the SIICUSP
material uses its files.

## Documentation

- [`DEVELOPMENT.md`](DEVELOPMENT.md) (in Portuguese): setup, development
  commands, directory boundaries and validation;
- [`docs/architecture.md`](docs/architecture.md): maintained architecture and
  lifecycle decisions;
- [`docs/algorithms/`](docs/algorithms): specifications and correctness
  arguments, among them the
  [certified convex oracle](docs/algorithms/certified-convex-oracle.md) (in
  Portuguese), [convex cycles](docs/algorithms/convex-cycle.md) and the
  [TSPN](docs/algorithms/tspn.md);
- [`docs/reports/oraculo-convexo/`](docs/reports/oraculo-convexo) (in
  Portuguese): technical report on the convex oracle (construction, interval
  proof, polish and correctness of the bounds);
- [`docs/third-party.md`](docs/third-party.md): external repositories and data;
- [`benchmarks/README.md`](benchmarks/README.md) (in Portuguese): benchmark
  protocols.

## References

- Dror, M., Efrat, A., Lubiw, A., Mitchell, J. S. B. (2003). *Touring a
  sequence of polygons*. STOC 2003, pp. 473–482.
  https://doi.org/10.1145/780542.780612
- Tan, X., Jiang, B. (2017). *Efficient algorithms for touring a sequence of
  convex polygons and related problems*. TAMC 2017, LNCS, pp. 614–627.
  https://doi.org/10.1007/978-3-319-55911-7_44
- Fekete, S. P., Kniep, R., Krupke, D., Perk, M. (2026). *A Branch-and-Bound
  Algorithm for the Traveling Salesman Problem with Difficult Neighborhoods*.
  SoCG 2026, LIPIcs 367, pp. 46:1–46:20.
  https://doi.org/10.4230/LIPIcs.SoCG.2026.46 (code:
  https://github.com/tubs-alg/TSPN-SoCG-2026)
- Arkin, E. M., Hassin, R. (1994). *Approximation algorithms for the geometric
  covering salesman problem*. Discrete Applied Mathematics 55(3), pp. 197–218.
  https://doi.org/10.1016/0166-218X(94)90008-6
- Greene, D. H. (1983). *The decomposition of polygons into convex parts*. In
  F. P. Preparata (ed.), *Computational Geometry*, Advances in Computing
  Research 1, pp. 235–259. JAI Press.
