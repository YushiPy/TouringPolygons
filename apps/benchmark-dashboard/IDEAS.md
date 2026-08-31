# Benchmark Dashboard Ideas

These are future product ideas for the benchmark dashboard. They are intentionally notes only, not an implementation plan.

## Append Instances to an Existing Test Case

Status: implemented for generated Synthetic and OSM batches. Appending converts the destination into an editable manual-json campaign, marks appended cases as generated, and records an edit-history entry.

Allow the instance creator to append newly generated instances to an existing test case instead of always creating a separate campaign or replacing the current input.

This seems worth doing. It would make iterative dataset construction much smoother, especially when a generated batch is good but too small. The main risk is provenance: appended instances should keep enough metadata to explain how each batch was generated, otherwise the test case becomes hard to interpret later.

## Remove Specific Instances from a Test Case

Status: partially implemented for editable campaigns through the manual instance list. Case edits clear stale benchmark results before rebuilding the manual input cache, but removals do not yet capture a reason or dedicated removal history.

Allow users to delete selected instances from a test case, probably from the inspect view or a dedicated dataset editor.

This is useful, but it needs guardrails. Removing hand-picked bad instances can silently bias a benchmark suite, so the UI should make removals explicit and probably keep a small edit history or manifest entry listing removed case ids and reasons.

## Manual Instance Editor

Status: implemented as the Edit Instances panel.

Add a visual editor, similar to the visualizer app, where users can draw or modify polygons, set start/end points, validate the geometry, and add the resulting instance to one or more test cases.

This is the strongest idea, but also the largest. It would make debugging and adversarial testcase design much better than relying only on random generation. The core challenge is validation and export: the editor should prevent invalid polygon orderings, malformed polygons, accidental intersections when the target suite is disjoint-only, and inconsistent coordinate scaling. If done well, it could become the main workflow for constructing regression cases.

## Instance Action Icons

Status: implemented for duplicate/delete actions in the editable instance list.

Replace the temporary text/symbol controls in the instance list with proper duplicate/delete icons that match the dashboard's visual language.
