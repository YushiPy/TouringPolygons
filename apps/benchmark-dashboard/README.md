# Benchmark Dashboard

Local FastAPI dashboard for creating, appending, editing, running, comparing, and inspecting TPP benchmark campaigns.

## 34º SIICUSP: visitor experience

The main event page is the adapted German corpus at `/evento`, with the previous SIICUSP
recorte archived at `/evento/siicusp`. `/evento/alemao` remains an alias for the main
page. The technical workbench remains at `/`, with links to the event pages in its
header and comparison panel. Start locally:

```bash
cd apps/benchmark-dashboard
npm run start:event
```

The visitor layout puts playback before the map, keeps geometry controls visible below the map, and keeps the sortable table on small screens with horizontal scrolling. Results
start with eight cases; “Ver todos” expands the current filtered selection.
The method section illustrates one nonconvex region in three conceptual stages.
Playback is twice as fast at 1× as the initial visitor version (3.8 s for four
regions, 11 s for 40). Wheel and pinch gestures zoom around the pointer; keyboard
zoom and the Fit button remain available. Click a result column heading to sort;
click it again to reverse direction. “Resultado” cycles between certified-first,
time-limit-first and no grouping; it stays primary while another column sorts
within each group. Mobile keeps the same clickable headers with pinned case IDs
and open-case actions. The three outcome filter buttons have been removed.

`#desafio` offers a separate four-region teaching example. Visitors choose an
order; the page compares the saved path for that order against the best of all
24 permutations. Contact points were optimized by the native fixed-order solver.
The requested sequence need not equal the first-contact sequence because
incidental visits are allowed. This example is not part of the 60-case benchmark.
A second challenge fixes the order A → B → C and lets visitors choose one of
three convex pieces per nonconvex region. All 27 combinations are precomputed.
Selected regions in the order challenge toggle off when clicked again. Dashed
arrows through representative centers preview the selected sequence, never a
solver path; comparison replaces the sketch with the computed paths. Both
challenges are included in the offline export. Disclosure sections animate
opening and closing (respecting reduced motion). On mobile, tap or drag the
“Nesta página” handle left to open a section index, and swipe right to close it. To regenerate into a new file:

```bash
.venv/bin/python scripts/export_event_challenge.py /tmp/challenge-reviewed.json
```

For the poster, use “Explore os caminhos da pesquisa” beside the QR and print a
short readable URL. Point it at the public `/evento` URL after deployment; the
localhost preview is only for review and is not accessible to event visitors.

Open [the main event demonstration](http://127.0.0.1:8017/evento). It includes:

- all 558 adapted German cases, with shareable `?caso=557` links;
- path playback and scrubbing with adjustable speed, visit-order labels, subtle
  visited/unvisited colors, contact points, convex decomposition, hulls, zoom and pan;
- keyboard controls (focus the diagram, then use arrows, `+`, `-`, or `Home`),
  reduced-motion support, responsive layout and a static fallback without JavaScript;
- separate explanations for certified paths and time limits;
- filtered results, CSV export and per-case geometry, paths and provenance;
- an educational solver replay on the current German-corpus page for cases 03, 15,
  20 and 56, including the initial heuristic, incumbent updates, order branching,
  convex-piece branching, lower-bound pruning and the compact recorded search tree;
- animated trace playback that progressively draws the current route. The labels in
  that replay are assigned after the optimal route is known for presentation: the
  solver itself does not receive this display numbering;
- an offline download at `/evento/offline`: one HTML file containing the 558-case
  data, CSS and JavaScript, usable without the server, network, C++ build or Gurobi license.

The former SIICUSP demonstration remains available at `/evento/siicusp`, with its
offline copy at `/evento/siicusp/offline`. It is an archive, not the primary benchmark
landing page.

The archived SIICUSP page uses `static/event/siicusp34.json`, a portable snapshot of the adopted
18 September 2026 exact-solver confirmation run: 60 valid paths, 45 exact
certificates and 15 time limits. It does not launch new computations. The page also
contains a step-by-step replay of the recorded solver execution for SIICUSP cases
03, 10 and 56. It shows the greedy heuristic, incumbent updates, convex-relaxation
bounds, order branching and pruning; labels are remapped to the visitor convention
in which 1 is the first region of the final recorded order. The ordinary map
animation remains playback of the saved path. Visitor case
numbers and searches are one-based (1–60). Stored IDs and existing `?caso=` links
stay zero-based for compatibility; CSV includes both `case_number` and `case`, and
per-case downloads include `case_number`.
Region labels show the first-visit rank starting at one, with original IDs mapped
in the details. A certified case is presented as exact; a time-limited case is
presented as an incumbent path with certification pending. The public event pages do
not expose numerical lower/upper bounds or a numerical gap. Contact points are reconstructed from the saved path
with the audit tolerance of 1e-7. Convex decomposition uses the repository’s native
`optimal_convex_partition::decompose_polygon`, the same library called by the
solver. Frozen pieces are stored separately in `static/event/siicusp34-partitions.json`;
zero-area pieces are omitted only from the display. They do not replay search branches.
The dashboard’s decomposition layer calls the same library through `/api/geometry/partition`.
The first dashboard request builds the small adapter using the local C++ compiler.
The event and offline pages need no compiler or runtime solve.
Regenerate the frozen partitions into a new file using
`.venv/bin/python scripts/export_event_partitions.py /tmp/reviewed-partitions.json`.
Regenerate the archived SIICUSP educational search traces into a new file using
`.venv/bin/python scripts/export_event_trace.py --output /tmp/siicusp34-traces.json`.
The exporter runs the solver with `--trace`, keeps the three showcase cases by
default, and truncates very large traces while preserving the heuristic and final
search events.
The current German-page traces are regenerated with
`.venv/bin/python scripts/export_german_trace.py --output static/event/german-instances-traces.json`;
its default showcase cases are the zero-based solver IDs 2, 14, 19 and 55,
displayed to visitors as cases 03, 15, 20 and 56.
Both case lists support ascending/descending sorting. Map controls support pinch
zoom, focus-based keyboard navigation, touch taps and accessible pressed toggles. The historical 5 September
comparison remains explicitly labeled as historical in the technical workbench. The
German event page contains all 558 adapted German instances: 558 valid paths, 475
exact certificates and 83 time limits after a two-stage run with up to ten seconds
for initially unresolved cases. This corpus is an endpoint-path adaptation of the
German/SO-CG data, so it must not be described as the original closed-tour benchmark
without that qualification.

Rebuild the snapshot from the original audit into a **new** file for review:

```bash
.venv/bin/python scripts/export_event.py \
	--run ../../benchmarks/results/unordered/siicusp34-20260906-200122 \
	--output /tmp/siicusp34-reviewed.json
```

The exporter checks suite identity, case coverage, paths, lengths and certificate
status before exporting. It refuses to overwrite its destination. Any later change
to the campaign should update the explanatory copy and tests alongside the data;
the page is deliberately pinned to this reviewed result, not the latest local run.
The snapshot and the offline document omit machine-local paths and credentials.

`npm run test:event` checks the frozen evidence, route rendering, standalone export,
playback geometry, filtering and the live solver adapter. `npm run test:all` includes
these tests and the existing regression suites. Browser smoke remains opt-in.
This integration does not publish the site or alter the submitted abstract.

## Abrir no celular pela rede local

A prévia `127.0.0.1:8020` e o comando `start:event` aceitam conexões apenas do
próprio Mac. No celular, `localhost` e `127.0.0.1` apontam para o próprio celular.
Para acessar a demonstração nos dois aparelhos, use o servidor de rede local:

1. Conecte o Mac e o celular à **mesma rede Wi-Fi**.
2. Na raiz deste repositório, execute:

```bash
cd apps/benchmark-dashboard
uv sync  # necessário na primeira execução ou após atualizar dependências
npm run start:event:lan
```

3. Aguarde a mensagem `Uvicorn running on http://0.0.0.0:8019`.
   O comando imprime o endereço do Mac e o endereço para o celular, por exemplo
   `http://192.168.0.36:8019/evento`. **Copie o endereço exibido no seu terminal**;
   o IP pode mudar quando a rede mudar. Use `http`, não `https`.
4. Abra esse endereço no Safari ou Chrome do celular. No Mac, você também pode
   abrir [a versão de rede local](http://127.0.0.1:8019/evento).

O terminal deve continuar aberto e o Mac acordado. Para impedir repouso enquanto
faz a demonstração, use `caffeinate -i npm run start:event:lan` no lugar do comando
acima. Encerre com Ctrl+C. O servidor recarrega alterações em Python; após mudar
HTML ou JavaScript, atualize a página do navegador.

Este comando serve apenas a página do evento, seus arquivos estáticos e o download
offline. O laboratório de edição e os endpoints do solver não ficam expostos.
`0.0.0.0` é o endereço de escuta, não o endereço que se digita no celular.

### Se não abrir

- **“Address already in use”**: pare o servidor anterior com Ctrl+C no terminal
  dele ou escolha outra porta: `npm run start:event:lan -- --port 8021`.
  Use a nova porta também no celular.
- **Erro 500 também no Mac**: confira a mensagem no terminal e reinicie o servidor.
  Um processo antigo pode estar usando código Python anterior às mudanças.
- **Funciona no Mac, mas o celular não conecta**: confirme o mesmo Wi-Fi, IP e
  porta. Redes de convidados e algumas redes institucionais isolam os aparelhos;
  nesses casos, use uma rede que permita comunicação entre eles.
- **macOS pede acesso à rede**: permita conexões de entrada para o Python usado
  nesta demonstração, se você deseja disponibilizá-la nessa rede. Não é necessário
  desligar o firewall. Se uma VPN estiver ativa, confira se ela permite rede local.
- **O comando não mostrou IP**: execute `ipconfig getifaddr en0` no Mac. Se não houver
  resultado, consulte o endereço IP nos detalhes da conexão Wi-Fi em Ajustes do
  Sistema. No Linux, use `hostname -I`.

Também é possível baixar “Levar demonstração offline” depois de abrir a página;
essa cópia inclui os dois desafios e não depende de um servidor ativo. Para o QR
no pôster, use um endereço público permanente, pois o IP local só funciona nessa rede.

## Run

From the repository root:

```bash
cd apps/benchmark-dashboard
uv sync
uv run uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000`.

To use a fixed port:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8017
```

## Offline editor

The lightweight editor at `/editor/offline` is the migration target for the old
former standalone editor workflow. It keeps a browser-local library of instances,
imports legacy `startPoint`/`targetPoint` JSON as well as dashboard JSON, and
imports/exports the library as a single file. It does not call the dashboard API;
convex decomposition runs in JavaScript, solving uses optional WebAssembly,
and the `Last-step map` layer uses the WASM solver's extracted cone buffers for
fixed-order convex pairwise-disjoint instances. It deliberately does not carry
over the old JavaScript solver.

With the dashboard server running, open
`http://127.0.0.1:8017/editor/offline`. To serve the static page without
FastAPI, run `python3 -m http.server 8020` from this directory and open
`http://127.0.0.1:8020/offline-editor/`.

The page remains useful without the generated solver files. To enable local
fixed-order solving, build the browser assets with `bash wasm/build.sh`; the
generated files under `static/wasm/` are ignored by Git.

## Main Views

- `Create`: builds synthetic or OSM-derived campaigns, with optional preview generation. Generated batches can also be appended to an existing campaign.
- `Cases`: opens a campaign in the manual instance editor. The editor supports moving endpoints and vertices, drawing polygons, selecting/deleting vertices, deleting instances, zooming, fitting, grid snapping, convex decomposition display, labels, and live path solving.
- `Inspect`: lists campaigns, imports bundled suites, shows SVG preview grids, opens individual instances, and shows recent result files and jobs.
- `Benchmark`: runs a selected campaign through a solver and displays progress, summary metrics, histograms, and solved instance cards.
- `Comparison`: runs the same campaign through multiple solvers and compares wall time, convex solve time, calls, and solved counts.

## Instance Inspection

Campaign preview grids intentionally use generated SVG previews, so large suites with hundreds of instances stay cheap to browse.

Clicking an instance opens a read-only version of the editor renderer. This expanded inspection view supports:

- pan, zoom, and fit;
- grid, path, decomposition, and label toggles;
- live computed path rendering;
- an `Edit Instance` shortcut in the viewer toolbar;
- inline instance renaming from the modal title.

The modal title shows the campaign/test-case name on the first line and the selected instance position/name on the second line.

## Notes

Campaign data lives under `benchmarks/campaigns`. Benchmark outputs live under `benchmarks/results`.

Generated and imported campaigns can be inspected and edited through the same case API. Saving edits updates the canonical manual case JSON and campaign metadata; preview images and the solver compatibility binary are regenerated lazily when requested.

Appending generated cases converts the destination campaign to the editable manual-json format, marks appended instances as generated, stores a compact edit-history entry, and clears stale benchmark outputs.

Shift-clicking a delete button bypasses the confirmation for faster editing sessions. Normal delete clicks and overwrite prompts still ask for confirmation.

## Module Layout

- `main.py`: FastAPI application setup, compatibility imports, and route wiring.
- `dashboard/`: backend models, file and binary IO, campaign and preview helpers, reports, jobs, and route groups.
- `static/app.js`: frontend bootstrap and cross-feature wiring.
- `static/*.js`: focused API, rendering, preview, editor, report, job, and form modules.
- `templates/partials`: page panels, dialogs, and shared modal markup.
- `static/*css`: stylesheet modules loaded by the page shell.
- `scripts/run-tests.sh`: dashboard test runner used by `npm run test:all`.

## Validation

From `apps/benchmark-dashboard`:

```bash
npm run test:all
```

The full runner prints each command's detailed output and finishes with a compact summary of passed suites and test counts.

Browser smoke is opt-in because it starts a local dashboard server and launches Playwright:

```bash
RUN_BROWSER=1 npm run test:all
```

Use another port if needed:

```bash
RUN_BROWSER=1 BROWSER_PORT=8020 npm run test:all
```

The Python suite includes route-level integration coverage for manual campaign mutation, append behavior, stale benchmark invalidation, and lazy regeneration of a missing or stale `inputs/manual.bin` compatibility artifact.

Manual campaigns use `manual-cases.json` as the canonical editable representation. The generated `inputs/manual.bin` file remains a solver compatibility artifact and is rebuilt on demand for previews, benchmark runs, and comparisons.

## Visit order

Choose `Fixed order` or `Free order` in Benchmark, Comparison, or Cases. The selection is synchronized across these views. Fixed order keeps the existing solver pipeline. Free order uses the native nonconvex TPP branch-and-bound, with fixed start and target points. The live editor uses a three-second budget and displays the incumbent path and gap when the search has not finished.

Free-order campaigns currently run with one worker. An empty time limit means 30 seconds per instance. Comparison supports `Our TPP B&B` and `External TSPN`; the external checkout and its Python environment must be installed, and its time limit must be an integer number of seconds. The fixed-order editor uses the optional WASM solver when available and falls back to the local fixed-order API otherwise. Rebuild the browser solver with `bash wasm/build.sh`; generated files under `static/wasm/` remain ignored.

Reports include per-instance bounds, gaps, timing, termination, and our saved paths and first-visit orders. Results are saved separately under `benchmarks/campaigns/<campaign>/results/free-order/<run>/report.json`. Matching completed configurations are reused unless forced. They never populate fixed-order summary files.

In Comparison, click `Show recorded free-order comparison (60 instances)` to inspect the measured development suite, including numerical tolerances and endpoint validation differences. This requires the local artifacts under `benchmarks/results/unordered/final-dev.jsonl` and `tspn-comparison/results/unordered-final`; those benchmark artifacts are not tracked in Git.

The same campaign runner is available from the repository root:

```bash
apps/benchmark-dashboard/.venv/bin/python benchmarks/tpp.py free-order CAMPAIGN --solver unordered --solver tspn --max-seconds 2
```

The additional browser test exercises mode switching, the saved comparison, a live free-order solve, a campaign run, and an actual external comparison. Start the dashboard, then run:

```bash
DASHBOARD_URL=http://127.0.0.1:8137 npm run test:browser:free
```

The final challenge combines four-region visit order with three convex pieces per
region (24 × 81 = 1,944 precomputed solutions). Selecting a new region appends it;
changing its piece retains its position; selecting its current piece removes it.
All three challenges are synthetic and separate from the 60 benchmark cases.
