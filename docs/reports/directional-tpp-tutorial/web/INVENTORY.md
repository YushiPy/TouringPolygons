# Phase 1 content inventory

Inventory made from `../main.tex` and `../figures.tex` on 2026-09-14. The LaTeX source remains the authority; this file records what the static port must contain.

## Structure

- 14 numbered sections, in source order:
  1. What problem is being solved?
  2. Three ways a last visit can behave
  3. The established disjoint solver
  4. What changes when polygons intersect
  5. The exact two-reflection witness
  6. Directional queries: a point with an infinitesimal approach
  7. Virtual sources, and what the records mean
  8. The implementation as explicit procedures
  9. A complete two-reflection execution
  10. Engineering choices and costs
  11. What is justified, tested, and still open
  12. Closed and limiting cases in the same model
  13. Reproducing the exact coordinate arithmetic
  14. Source guide and provenance
- 4 subsections under “The implementation as explicit procedures”:
  - Dispatch, normalization, and splitting
  - Closed membership and symbolic signs
  - The cone and chord predicates
  - Recursion, refolding, and length
- Unnumbered References section with 6 bibliography entries.

## Mathematical and visual content

- 5 labeled equations: `eq:prefix`, `eq:reflect`, `eq:bpath`, `eq:dual`, `eq:symbolic`.
- 12 figures and stable report anchors: `fig:order`, `fig:contacts`, `fig:unfold`, `fig:last`, `fig:binary`, `fig:split`, `fig:limits`, `fig:rays`, `fig:backwards`, `fig:virtual`, `fig:membership`, `fig:path`.
- 5 pseudocode blocks: public dispatch/splitting; lexicographic sign/closed membership; vertex construction/location; virtual source/path recursion; length query.
- 7 source-note boxes, 2 code-key/source tables, the four-case chord predicate table, the two execution trace tables, the split-vertex array, and the source-guide table.
- Exact witness coordinates, reflection matrices, limiting directions, binary-search intervals, refolding fractions, route vectors, complexity expressions, validation counts, closed/limiting cases, source provenance, citations, and the explicit statement that a general equivalence/correctness proof remains open are retained.

## Figure scope

The original TikZ drawings are rendered as local SVGs. Captions remain verbatim in the generated page. Captions that say “schematic,” “generic illustration,” or otherwise qualify the drawing continue to do so; the static port does not add a stronger geometric interpretation. Phase 2 may add interaction, but must preserve each figure's exact-versus-schematic scope and equal x/y scale.
