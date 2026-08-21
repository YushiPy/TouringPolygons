# Instance Generation

`source/gen_instances.py` generates non-convex TPP benchmark inputs from OSM building footprints.

The primary output is a binary test-case stream compatible with the non-convex C++ `load_test_cases()` / `encode_test()` format. The generator also writes a preview image for visual inspection and a small manifest with generation metadata.

## Download OSM Data

[Geofabrik](https://download.geofabrik.de/) provides regional OpenStreetMap extracts
in the `.osm.pbf` format. Open the continent and country pages, then download the
smallest extract containing the desired area. For example, Brazil and its regions are
available from the [Brazil download page](https://download.geofabrik.de/south-america/brazil.html).

Store downloaded files under `packages/instance-generation/regions/`, which is
ignored by Git:

### Ready-to-paste downloads

Create the local data directory:

```bash
mkdir -p packages/instance-generation/regions
```

Download Southeast Brazil:

```bash
curl --fail --location --continue-at - \
  https://download.geofabrik.de/south-america/brazil/sudeste-latest.osm.pbf \
  -o packages/instance-generation/regions/sudeste-latest.osm.pbf
```

Download Portugal:

```bash
curl --fail --location --continue-at - \
  https://download.geofabrik.de/europe/portugal-latest.osm.pbf \
  -o packages/instance-generation/regions/portugal-latest.osm.pbf
```

Download California:

```bash
curl --fail --location --continue-at - \
  https://download.geofabrik.de/north-america/us/california-latest.osm.pbf \
  -o packages/instance-generation/regions/california-latest.osm.pbf
```

`--continue-at -` resumes a partial download when the command is run again. The
`latest` extracts are updated regularly, so record the download date when exact
dataset reproducibility matters.

Large regional files can be cropped before generation. Install `osmium-tool` on
macOS with Homebrew:

```bash
brew install osmium-tool
```

Then extract a city or neighborhood using a bounding box:

```bash
osmium extract \
  --bbox MIN_LONGITUDE,MIN_LATITUDE,MAX_LONGITUDE,MAX_LATITUDE \
  packages/instance-generation/regions/sudeste-latest.osm.pbf \
  --output packages/instance-generation/regions/sao-paulo.osm.pbf \
  --overwrite
```

Longitude is the horizontal coordinate and latitude is the vertical coordinate. The
bounding box therefore describes its southwest corner first and northeast corner
second. Coordinates can be obtained by drawing a rectangle with
[OpenStreetMap's export tool](https://www.openstreetmap.org/export), then copying the
displayed bounds. Prefer a reasonably small crop: building extraction and the first
cache creation are substantially faster when unrelated parts of the regional file
are removed.

### Ready-to-paste crops

These are approximate rectangular study areas, not official administrative
boundaries.

Extract the São Paulo metropolitan area from the Southeast Brazil file:

```bash
osmium extract \
  --bbox -46.83,-24.01,-46.36,-23.35 \
  packages/instance-generation/regions/sudeste-latest.osm.pbf \
  --output packages/instance-generation/regions/sao-paulo.osm.pbf \
  --overwrite
```

Extract the Lisbon metropolitan area from the Portugal file:

```bash
osmium extract \
  --bbox -9.30,38.65,-9.00,38.85 \
  packages/instance-generation/regions/portugal-latest.osm.pbf \
  --output packages/instance-generation/regions/lisbon.osm.pbf \
  --overwrite
```

Extract San Francisco from the California file:

```bash
osmium extract \
  --bbox -122.52,37.70,-122.35,37.84 \
  packages/instance-generation/regions/california-latest.osm.pbf \
  --output packages/instance-generation/regions/san-francisco.osm.pbf \
  --overwrite
```

Generate a small solver-ready grid dataset from any cropped file, for example São
Paulo:

```bash
python3 packages/instance-generation/source/gen_instances.py \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-bin benchmarks/results/sao-paulo-grid.bin \
  --preview benchmarks/results/sao-paulo-grid.png \
  --instances 100 \
  --polygons-per-instance 20 \
  --layout grid \
  --grid-cell-size 2.0 \
  --grid-placement random \
  --order random \
  --seed 42
```

## Usage

```bash
python3 packages/instance-generation/source/gen_instances.py packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-bin benchmarks/results/osm_buildings.bin \
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
| `--no-preview` | Skip preview images. This also avoids requiring `matplotlib`. |
| `--no-manifest` | Skip metadata JSON. Useful for large benchmark sweeps where filenames carry the parameters. |
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
| `--convex-replacement-fraction` | Replace this fraction of sampled polygons with synthetic convex many-vertex polygons. |
| `--convex-replacement-vertices` | Vertex count for synthetic convex replacements. |
| `--convex-replacement-position` | Put replacements in the `middle`, at `random`, or in `alternating` positions of the visit order. |
| `--order` | Polygon visit order: `spatial`, `left-to-right`, `random`, or `angle`. |
| `--endpoint-mode` | Start/target placement from ordered endpoints or the instance bounding box. |
| `--single-preview-count` | Number of individual instance preview images to write. |

The default sampler is `local`: choose a random anchor building, take the nearest candidate pool around it, then sample one instance from that neighborhood. This avoids accidentally making one TPP instance span an entire city. Use `--sampling uniform` when you intentionally want broad-region stress tests.

The default order is `spatial`: choose the leftmost sampled building first, then greedily visit nearest sampled centroids. This gives a reproducible route through the sampled footprints without making the visit order purely random.

The default normalization is per-instance centering in projected meters. This keeps coordinates numerically stable for the C++ solver while preserving each building's shape and relative placement inside an instance.

The current non-convex C++ solver cannot handle touching or intersecting polygons. Real OSM buildings often share walls or nearly touch, so use grid layout for solver-ready benchmark inputs:

```bash
python3 packages/instance-generation/source/gen_instances.py packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-bin benchmarks/results/osm_buildings_grid.bin \
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

To mix complex non-convex buildings with many-vertex convex polygons:

```bash
python3 packages/instance-generation/source/gen_instances.py sp-city.osm.pbf \
  --output-bin benchmarks/results/osm_buildings_grid_convex50.bin \
  --instances 100 \
  --polygons-per-instance 50 \
  --layout grid \
  --grid-placement random \
  --convex-replacement-fraction 0.5 \
  --convex-replacement-vertices 64 \
  --no-preview \
  --no-manifest
```

The grid preview intentionally shows only the first 50 cases when many instances are generated. Individual previews are written next to it by default:

```text
<preview-stem>-instances/case-000.png
<preview-stem>-instances/case-001.png
<preview-stem>-instances/case-002.png
```

## Batch Matrix

`source/generate_benchmark_matrix.py` creates a sweep of binary inputs. It loads the OSM building cache and builds the candidate polygon pool once, then reuses it for every generated binary.

Current default sweep:

- polygons per instance: `1,3,5,10,20,30,40,50`
- layouts: `geographic,grid`
- grid spacings: `1.1,1.25,1.5,2.0,2.5,3.0`
- convex replacement fractions: `0.0,0.25,0.5,0.75,1.0`

Preview and manifest files are disabled by default for batch runs.

For new experiments, use the campaign entry point. It stores the binaries outside the C++ regression-test directory and writes one campaign-level provenance file instead of a manifest for every binary:

```bash
python3 benchmarks/tpp.py generate-matrix sao-paulo \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --instances 100 \
  --sample-size 40 \
  --seed 42
```

The lower-level matrix script remains available for producing binaries in an explicitly selected directory:

```bash
python3 packages/instance-generation/source/generate_benchmark_matrix.py \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-dir benchmarks/campaigns/sao-paulo/inputs \
  --instances 100
```

To sample a smaller subset from the full matrix:

```bash
python3 packages/instance-generation/source/generate_benchmark_matrix.py \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --output-dir benchmarks/campaigns/sao-paulo/inputs \
  --instances 100 \
  --sample-size 40 \
  --seed 42
```

Preview the commands without generating files:

```bash
python3 packages/instance-generation/source/generate_benchmark_matrix.py \
  packages/instance-generation/regions/sao-paulo.osm.pbf \
  --dry-run
```

## Benchmark Workflow

Generated binaries can be passed directly to the benchmark runner:

```bash
python3 benchmarks/tpp.py run-groups \
  --index benchmarks/results/splits/instances.json \
  --group under_100ms
```

To classify a newly generated binary by difficulty, first benchmark it and then split it:

```bash
./build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  benchmarks/results/osm_buildings.bin \
  -1 -1 1000000 -1 1 \
  benchmarks/results/osm_buildings.csv \
  benchmarks/results/osm_buildings.md

python3 benchmarks/tpp.py split \
  --input benchmarks/results/osm_buildings.bin \
  --csv benchmarks/results/osm_buildings.csv \
  --output benchmarks/results/osm_building_splits
```
