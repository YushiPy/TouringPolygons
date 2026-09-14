# Interactive web edition — Codex TODO

This is the working prompt and progress record for porting the fixed-order convex
Touring Polygons tutorial to a **local, unpublished webpage**. Keep this file in
the repository and update its checkboxes as work is completed. An unchecked box
means the work is still outstanding; do not check a box for a placeholder or an
untested implementation. Add a brief note under a task if its scope changes.

## How to use this prompt

**Luna pass:** select GPT-5.6 Luna (medium reasoning), then give Codex this prompt:

> Read `docs/reports/directional-tpp-tutorial/WEB_PORT_TODO.md`. Complete only
> Phase 1 and its checks. Create a faithful, readable static web edition of the
> full report. Update the checkboxes and the handoff record with what you
> actually completed, how you verified it, and what remains. Do not invent
> geometric claims, source signatures, or interactive behavior. Stop after the
> Phase 1 handoff so I can switch models.

**Sol/Astra pass:** switch to GPT-5.6 Sol or GPT-6 Astra (high reasoning), then
give Codex this prompt:

> Read `docs/reports/directional-tpp-tutorial/WEB_PORT_TODO.md`, the Phase 1
> handoff, the LaTeX source, and the current web files. Audit Phase 1 against
> the original report, fix omissions, then complete Phase 2 and the final
> acceptance checks. Implement all twelve expandable interactive figures,
> actual-code overlays, and function hover/focus cards. Update checkboxes only
> after verifying each item. Keep working through the checklist until the local
> web edition is complete; report any genuinely blocked item explicitly.

## Source material and boundaries

- The authoritative report is `docs/reports/directional-tpp-tutorial/main.tex`,
  with twelve drawings in `figures.tex` and the rendered PDF in the same folder.
  Its `README.md` explains the source snapshot and build. Preserve the report's
  mathematical meaning, exact coordinates, citations, and explicit distinction
  between proved claims, observed implementation behavior, and open proof work.
- The report contains **14 sections, 4 subsections, 12 figures, 5 pseudocode
  blocks, and 6 bibliography entries**. These counts are a completeness check,
  not a substitute for comparing the content itself.
- Its C++ line references refer to repository revision `0aaf550`. The code key
  on the contents page names `IM`, `BS`, `SOL`, `GEO`, and `VAL`. Consult the
  actual files at that revision before presenting a signature or code excerpt.
  Do not silently reinterpret a pinned line reference against a newer checkout.
- The benchmark dashboard is visual inspiration, especially
  `apps/benchmark-dashboard/static/event.js`, `event-geometry.js`, and
  `readonly-viewer.js`. Study its route playback, scrubbing, fit/zoom/pan,
  keyboard/touch behavior, and reduced-motion treatment. Reuse isolated
  utilities only when that makes the new guide simpler; keep this guide
  independent of dashboard-specific state and benchmark APIs.
- Put the standalone web edition under
  `docs/reports/directional-tpp-tutorial/web/`. It must run locally and must
  not require publication. A static/offline build using bundled local assets is
  preferred. Do not modify production solver code or run the large solver
  regression suites for this documentation project.
- A route animation of a saved example depicts that **route**, not the internal
  search process. A schematic figure must remain labeled schematic. Coordinates
  displayed for a schematic scene are illustrative; exact witnesses retain
  their exact values. Preserve equal geometric units on both axes.

## Phase 1 — Luna: faithful static edition and handoff

### Inventory and structure

- [x] Read the entire report, `figures.tex`, the report README, and the relevant
  benchmark-dashboard viewer files. Make an inventory of all headings, equation
  labels, figures/captions, pseudocode blocks, source notes, tables, and
  bibliography entries; store the inventory in the web project or this file.
- [x] Create a small, standalone web project in `web/`, with a documented local
  start/build command. Use semantic HTML and a simple, maintainable file layout.
  Do not make architectural commitments about the future geometry engine or
  source-code overlay during this phase.
- [x] Build the reading shell: title/abstract, contents, readable text column,
  responsive navigation, stable section anchors, and distinct styling for
  examples, source notes, pseudocode, captions, and references. Preserve the
  report's chapter order and hierarchy.

### Complete static content

- [x] Transfer all 14 sections and 4 subsections, including the worked examples,
  exact witness, complexity discussion, validation account, closed/limiting
  cases, source guide, and open proof obligations. Compare paragraphs against
  `main.tex`; do not summarize or replace technical claims with marketing copy.
- [x] Render every inline and display equation, including custom symbols such
  as `\R`, `\norm`, and `\cross`. Keep equation labels and links working. Check
  fractions, primes, subscripts, signs, and the infinitesimal-query notation
  visually in a browser. Bundle math assets locally for the final offline build.
- [x] Port all five pseudocode blocks as readable, copyable code with line
  numbers and modest syntax highlighting. Preserve indentation and branch
  order. A static block is sufficient for Luna; line-to-geometry interaction
  belongs to Phase 2.
- [x] Port the source-guide table, all six bibliography entries, citations,
  figure references, and internal links. Keep `Open the code` notes visible and
  mark their source identifiers for Phase 2, even if they do not yet open code.
- [x] Show all twelve figures with their complete captions and stable figure
  IDs. A faithful static rendering is acceptable in Phase 1; prefer vector
  output, but use a temporary high-resolution fallback if vector export is
  unreliable. Record every temporary fallback in the handoff. Do not redraw
  an exact coordinate scene from visual guesswork.
- [x] Add unobtrusive, explicit placeholders for later enhancement: every
  figure has a future Expand control location, and source/function references
  have stable identifiers that Phase 2 can connect to overlays. Keep the static
  page usable without those future interactions.

### Luna verification and handoff

- [x] Build and serve the site locally; check a desktop and narrow/mobile
  viewport. Confirm the full report is readable, no horizontal overflow hides
  equations or tables, and all static links have valid targets.
- [x] Compare the web edition with the PDF section by section. Verify the
  14/4/12/5/6 content counts, every caption, and the caveat that general
  equivalence of the implicit directional maps remains unproved.
- [x] Update the handoff record below with created files, commands run,
  unresolved static-content issues, temporary figure assets, and any design
  decisions that Phase 2 must preserve. Do not check a task with missing work.

## Phase 2 — Sol/Astra: source explorer, interactive geometry, and audit

### Audit and shared foundations

- [ ] First review the Phase 1 site against `main.tex` and the PDF. Repair
  dropped content, broken math, citations, code names, or incorrect coordinates
  before adding interactions. Preserve readable static fallbacks.
- [ ] Define one reusable geometric scene format and viewer for all twelve
  figures. Keep scene coordinates and labels in deterministic data, separate
  from playback/camera UI. Document whether each scene is exact or schematic.
- [ ] Use the report's palette and typographic character in a polished web
  design. Keep the article readable while figures or code overlays are open;
  support desktop, tablet, and mobile layouts.

### Actual C++ source exploration

- [ ] Build a source registry with **qualified** symbol IDs, verified
  signatures, short descriptions, file paths, line ranges, and revision
  `0aaf550`. Resolve overloaded/ambiguous names such as `query` by context.
  Descriptions are editorial explanations checked against C++, never invented
  from a name alone.
- [ ] Generate the displayed C++ excerpts from the actual pinned repository
  source during the site build. Validate that every mapped path/range exists;
  show the revision in the UI. Keep the generated code local so the site works
  without GitHub, a server API, or network access.
- [ ] Make every mapped function mention in the report hoverable and keyboard
  focusable. A card shows its exact signature, concise purpose, owning file,
  and a clear action to open the source. Support tap on touch devices; avoid a
  hover-only interaction.
- [ ] Make every `Open the code` note actionable. Clicking it, or a function
  card, opens a read-only VS Code-style overlay/pane with actual C++ syntax
  highlighting, line numbers, a file breadcrumb, highlighted relevant lines,
  copyable path/reference, and navigation among multiple cited ranges. Provide
  close/Escape, focus handling, and a sensible small-screen layout.
- [ ] Link the source guide and pseudocode to that same registry. Where useful,
  selecting a pseudocode step may highlight corresponding C++ lines. Verify
  each source note and symbol link manually against the pinned code.

### All twelve expandable figures

- [ ] Give **every** inline figure an Expand action that opens its interactive
  counterpart, retaining its number, full caption, and exact/schematic label.
  The twelve scenes are `figordered`, `figcontacts`, `figunfold`, `figlastmap`,
  `figbinary`, `figsplit`, `figlimits`, `figrays`, `figvirtual`, `figpath`,
  `figbackwards`, and `figmembership` in `figures.tex`.
- [ ] In every expanded viewer, provide pointer-centered zoom, pan, Fit/reset,
  keyboard controls, touch/pinch controls, and toggleable coordinates/grid.
  Provide a visible scale or axis cue. Let users inspect point coordinates
  without covering important geometry. The coordinate readout must distinguish
  exact coordinates from illustrative schematic ones.
- [ ] Add useful layer toggles per scene (polygons, route, vertices/contacts,
  rays/chords, labels, grid/coordinates). Keep colors and legends consistent
  across the report, and preserve equal x/y scale under camera movement.
- [ ] Where a figure depicts a route, support play/pause, scrub, speed, step,
  and reset. Light up ordered visits and contacts at the correct time. Respect
  reduced-motion preference and permit inspection without animation.
- [ ] Give the three contact patterns a step-through comparison of inherited
  crossing, vertex bending, and finite-edge reflection. For unfolding, show
  the original and reflected query and the finite-edge contact check.
- [ ] Make the two-reflection witness explorable: compare `q_+` and `q_-`, the
  limiting approach directions, the literal wrong rays, and the corrected
  incident-limit rays. Do not interpolate an incorrect claim at epsilon zero.
- [ ] Step through the binary-location example with its actual candidate
  intervals, tested chords, cone/chord outcomes, and final pseudo-edge. Tie
  each step to the numeric sign table and relevant source lines.
- [ ] Step through virtual-source recursion and route refolding with map level,
  symbolic query, reflected target, cached source, contact parameter, and
  returned path. Make clear which drawings are algebraic transforms versus
  physical route segments.
- [ ] Add lightweight cross-highlighting where it clarifies the explanation:
  select an equation term, pseudocode line, or source step and highlight the
  corresponding geometric object. Keep the feature optional and unobtrusive.

### Local delivery and final verification

- [ ] Make the finished edition usable locally without publication. Bundle
  required assets; document how to start it and, if provided, open an offline
  build. No default interaction should require compiling or running the solver.
- [ ] Add targeted checks for content/anchor completeness, all twelve scene
  IDs, source-registry path/range validity, and representative viewer geometry
  and controls. Avoid the unrelated large solver regression suites.
- [ ] Visually inspect every section and all twelve expanded figures at desktop
  and mobile widths. Check math, captions, code overlay, camera controls,
  coordinate readouts, keyboard/touch behavior, reduced motion, and the static
  fallback. Fix clipping, ambiguous labels, and misleading animations.
- [ ] Update the web README with build/use instructions, content provenance,
  pinned code revision, exact-versus-schematic conventions, and known limits.
  Mark this checklist complete only when all required items are verified.

## Handoff record — edit as work progresses

**Phase 1 (Luna):** Complete (2026-09-14).

- Files created or changed:
  - `web/build.mjs`, `web/build.sh`, `web/style.css`, `web/figure-template.tex`,
    `web/README.md`, and `web/INVENTORY.md`.
  - Generated `web/index.html` plus twelve local SVGs in
    `web/assets/figures/` (`order`, `contacts`, `unfold`, `last`, `binary`,
    `split`, `limits`, `rays`, `backwards`, `virtual`, `membership`, and
    `path`).
  - This checklist and handoff record.
- Local build/start command:
  - From `docs/reports/directional-tpp-tutorial/web/`, run `./build.sh`, then
    `python3 -m http.server 8765` and open `http://127.0.0.1:8765/`.
  - `pandoc` is a build-time dependency. The generated page uses local MathML
    and SVG assets and has no runtime server/API or network dependency.
- Checks run and results:
  - `./build.sh` passed. It asserts 12 figures and 5 pseudocode blocks; the
    generated page also contains 5 labeled equation anchors and 6 bibliography
    anchors.
  - Source inventory checks passed against `main.tex`/`figures.tex`: 14
    sections, 4 subsections, 12 figures/captions, 5 pseudocode blocks, 7
    source-note boxes, 5 equation labels, and 6 bibliography entries. The
    visible source-note count is 6 because the seventh source box is the
    contents-page code key.
  - All twelve figure assets were rendered from the original TikZ definitions
    with `pdflatex` and `pdftocairo -svg`; no fallback or visual redraw was
    used. The report PDF was rebuilt with `bash build.sh` (27 A4 pages), then
    its normalized text was checked for all 14 section headings, 12 figure
    labels, the exact-unproved-equivalence caveat, and the corresponding web
    content.
  - The site was served locally and inspected in Safari at desktop width and
    the in-app browser at a 319px narrow width. The mobile DOM audit found no
    document overflow, all long equations/tables/pseudocode retained local
    scroll containers, all 12 SVGs loaded, all in-page links resolved, and the
    title/contents/figure/caption structure remained readable. Every figure has
    a disabled `Expand · Phase 2` control; there are no script tags or enabled
    interactive controls in the static page.
  - The large solver regression suites were not run.
- Temporary figure fallbacks or missing content:
  - None. The full report text is generated from `main.tex`; all twelve figures
    use checked-in SVG exports of the original TikZ. No static-content omission
    or temporary asset fallback remains.
- Decisions and notes for Phase 2:
  - Preserve the report's exact-versus-schematic caption scope and equal x/y
    scale. The static SVGs are source-rendered drawings, not newly inferred
    geometry.
  - Preserve `fig:*` anchors, the stable `figordered`-style figure IDs,
    `data-source-ref` ranges, `data-source-symbol` names, equation anchors, and
    the pinned revision `0aaf550` as the Phase 2 connection points.
  - Replace the disabled controls only after verifying any geometry, source
    signatures, excerpts, and interactions against the pinned source. Do not
    infer signatures or behavior from the current stable identifiers alone.
  - The page intentionally contains no geometry engine, route playback,
    source overlay, or other interactive behavior in this Phase 1 handoff.

**Phase 2 (Sol/Astra):** Not started.

**Phase 2 (Sol/Astra):** Not started.

- Files created or changed:
- Local build/start command:
- Checks run and results:
- Remaining limitations:
