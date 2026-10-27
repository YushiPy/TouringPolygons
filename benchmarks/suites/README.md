# Benchmark Suites

This directory contains canonical inputs and regression fixtures for TPP, TSPN and CETSP.

## Promoting a campaign case

Campaigns are local and ignored. If one case becomes a regression fixture,
export only the selected instance into the appropriate suite here, give it a
stable descriptive name, and record its origin and conversion command in the
commit or this file. Do not add campaign metadata, preview images, run history,
or private experiment notes.

- `fekete-instances.bin`: canonical 558-case corpus converted from a TSPN JSON
  archive. The source instances are cycles, so the converter uses each instance
  bounding box lower-left corner as `start` and upper-right corner as `target`.
- `nonconvex/test_cases.bin`: compatibility symlink to `../fekete-instances.bin`
  for existing local campaigns; new consumers use the canonical path above.
- `nonconvex/custom_tests.bin`: small tracked hand-made/debug fixture.
- `usp-butanta-50/usp-butanta-50.bin`: a 50-building USP Butantã case traced in
  QGIS; see its README and `polygons.csv` for provenance and the polygon-name
  mapping. The companion QGIS projects and editable GeoPackage are under
  `usp-butanta-50/qgis/`.
- `cetsp/`: close-enough TSP instances (disk neighborhoods): Mennell's 2D
  and 3D files, Behdani and Smith's point sets, and Carrabs et al.'s tours;
  see its README for provenance and the 2D TSPLIB rebuild.
- `algorithm-dev-v1.bin`: ignored generated development suite, regenerated from
  `fekete-instances.bin`.
- `canonical-v1.bin`: ignored generated benchmark suite, regenerated from
  `fekete-instances.bin`.

`../results-saved/fekete-comparison/instances.bin` retains its own identical copy
as part of the complete, frozen campaign used by the SIICUSP publication.

Regenerate the source corpus:

```bash
python3 benchmarks/tpp.py convert-fekete
```

Regenerate the derived development and canonical suites:

```bash
python3 benchmarks/tpp.py generate-suites
```
