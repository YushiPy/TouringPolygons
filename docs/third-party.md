# Third-party repositories and data

## The Fekete et al. solver

The solver of Fekete, Kniep, Krupke and Perk is distributed under the MIT
license. The modified integration is kept in the fork
[`YushiPy/TSPN-SoCG-2026`](https://github.com/YushiPy/TSPN-SoCG-2026), branch
`codex/touring-polygons-oracle`, and pinned in this repository as the submodule
`third_party/tspn-socg`. The first integrated revision is
`94db8fa3279f428484e598a3c40c04c470d03600`; the one pinned today is
`e91e936502ad32a5d5b3cf3f5a002cda739948e2`, which adds two compatibility fixes
(the `fmt` header in the Python bindings and the binary64 variant of the
directional map in the embedded oracle) without changing the solver.

The fork starts from upstream `tubs-alg/TSPN-SoCG-2026` and adds this project's
convex-oracle backend, propagation of the remaining time, instrumentation and
the matching Python interface. The original SOCP backend remains the default.
The upstream license and copyright remain in the fork.

Clone this project with the submodule:

```bash
git clone --recurse-submodules https://github.com/YushiPy/TouringPolygons.git
```

In an existing checkout:

```bash
git submodule update --init --recursive
```

The reference measurements on the Fekete side (`tpp.py fekete-paper` and
`fekete-single-thread`) do not use the fork. They run the unmodified upstream,
`tubs-alg/TSPN-SoCG-2026` at commit `5cd84507c2409caf47c524f60bdd3b96453f0e15`
(their `main` and the fork's base). The code is cloned and built in
`.build/fekete-upstream/` by `tpp.py fekete-setup`, which is part of
`scripts/install_dependencies.sh`. See
[`benchmarks/README.md`](../benchmarks/README.md#medições-de-referência-o-fekete-original-e-o-nosso-configurações-a-a-e)
(in Portuguese).

The submodule remains an external project, not a maintained package of this
monorepo. Adapters, campaigns and comparison results live in `benchmarks/`;
changes needed in the external solver are made in the fork and pinned here by
revision.

Virtual environments, builds, Gurobi, license files, caches and results belong
neither to the submodule nor to the main repository.

## Paula's instances

`third_party/paula-tspn/` holds material obtained from the author's personal
page with permission for local use and modification, but without explicit
permission to redistribute it. The whole folder therefore stays ignored and
must not be published, copied into a fork or added to the Git history.

When an instance is needed for a public result, check the license and
provenance of the original dataset separately. Permission over the author's
repository does not imply permission to redistribute third-party datasets
contained in it.
