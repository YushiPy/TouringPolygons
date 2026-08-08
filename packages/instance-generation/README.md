# Instance Generation

`source/gen_instances.py` generates non-convex TPP benchmark inputs from OSM building footprints.

The primary output is a binary test-case stream compatible with the non-convex C++ `load_test_cases()` / `encode_test()` format. The generator also writes a preview image for visual inspection and a small manifest with generation metadata.

## Usage

```bash
python3 packages/instance-generation/source/gen_instances.py packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-bin packages/nonconvex-tpp/cpp/tests/osm_buildings.bin \
  --preview packages/instance-generation/source/osm_buildings.png \
  --instances 100 \
  --polygons-per-instance 8 \
  --seed 42
```

Important options:

| Option | Meaning |
|---|---|
| `input_pbf` | Input `.osm.pbf` extract. |
| `--output-bin` | Binary TPP test set for the C++ solver and benchmark tools. |
| `--preview` | PNG preview of all generated instances. |
| `--manifest` | Optional metadata JSON. Defaults to `<output-bin>.manifest.json`. |
| `--instances` | Number of TPP instances to generate. |
| `--polygons-per-instance` | Number of building footprints per instance. |
| `--seed` | Random seed for reproducible sampling. |
| `--simplify-tolerance` | Topology-preserving simplification tolerance in projected meters. |
| `--normalization` | `instance` centers each generated case independently; `dataset` centers around the candidate pool; `none` leaves projected coordinates uncentered. |
| `--scale` | Multiplies generated coordinates after projection and centering. |
| `--sampling` | `local` samples each instance from nearby buildings; `uniform` samples from the whole cropped region. |
| `--local-pool-size` | Number of nearest buildings considered around a random anchor in local sampling. |
| `--layout` | `geographic` keeps sampled map positions; `grid` rescales polygons and places them in separated grid cells. |
| `--grid-polygon-size` | Target maximum width/height for each polygon in grid layout. |
| `--grid-cell-size` | Distance between grid cell centers in grid layout. Must be larger than `--grid-polygon-size`. |
| `--grid-columns` | Number of grid columns. Defaults to `ceil(sqrt(polygons_per_instance))`. |
| `--grid-placement` | `row-major` assigns visit order across the grid; `random` assigns ordered polygons to random grid cells. |
| `--order` | Polygon visit order: `spatial`, `left-to-right`, `random`, or `angle`. |
| `--endpoint-mode` | Start/target placement from ordered endpoints or the instance bounding box. |
| `--single-preview-count` | Number of individual instance preview images to write. |

The default sampler is `local`: choose a random anchor building, take the nearest candidate pool around it, then sample one instance from that neighborhood. This avoids accidentally making one TPP instance span an entire city. Use `--sampling uniform` when you intentionally want broad-region stress tests.

The default order is `spatial`: choose the leftmost sampled building first, then greedily visit nearest sampled centroids. This gives a reproducible route through the sampled footprints without making the visit order purely random.

The default normalization is per-instance centering in projected meters. This keeps coordinates numerically stable for the C++ solver while preserving each building's shape and relative placement inside an instance.

The current non-convex C++ solver cannot handle touching or intersecting polygons. Real OSM buildings often share walls or nearly touch, so use grid layout for solver-ready benchmark inputs:

```bash
python3 packages/instance-generation/source/gen_instances.py packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-bin packages/nonconvex-tpp/cpp/tests/osm_buildings_grid.bin \
  --preview packages/instance-generation/source/osm_buildings_grid.png \
  --instances 100 \
  --polygons-per-instance 8 \
  --layout grid \
  --grid-polygon-size 1.0 \
  --grid-cell-size 3.0 \
  --grid-placement random \
  --seed 42
```

Grid layout keeps each sampled building's footprint shape, scales all selected footprints to comparable size, and places them into non-overlapping cells in the selected visit order.

The grid preview intentionally shows only the first 50 cases when many instances are generated. Individual previews are written next to it by default:

```text
<preview-stem>-instances/case-000.png
<preview-stem>-instances/case-001.png
<preview-stem>-instances/case-002.png
```

## Benchmark Workflow

Generated binaries can be passed directly to the benchmark runner:

```bash
python3 benchmarks/bench.py run \
  --index benchmarks/splits/instances.json \
  --group under_100ms
```

To classify a newly generated binary by difficulty, first benchmark it and then split it:

```bash
./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  packages/nonconvex-tpp/cpp/tests/osm_buildings.bin \
  -1 -1 1000000 -1 1 \
  benchmarks/osm_buildings.csv \
  benchmarks/osm_buildings.md

python3 benchmarks/bench.py split \
  --input packages/nonconvex-tpp/cpp/tests/osm_buildings.bin \
  --csv benchmarks/osm_buildings.csv \
  --output benchmarks/osm_building_splits
```
