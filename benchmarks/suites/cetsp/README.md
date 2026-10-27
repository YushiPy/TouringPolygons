# CETSP instances

Close-enough TSP inputs (disk neighborhoods, a depot) for `tpp-convex-cetsp`
and the campaigns summarized in `../../results-saved/` (the `cetsp-*`
sections). Every file is an unchanged byte copy of its source, Windows line
endings included, except `mennell-tsplib-2d/`, which is derived.

`tpp-convex-cetsp` reads Mennell's format (`x y z radius` lines, extra columns
ignored, the depot in a `//Depot is x, y, z` or `//Depot: x, y, z` comment) and,
with `--radius R`, Behdani and Smith's (`x y` lines, the depot first).

- `cerrone/`: the repository github.com/CerroneCarmine/CETSP at commit
  9d2d4b8, without its Git metadata. It has no license file; it is versioned
  here with the project's agreement.
  - `Dataset/*.cetsp`: Mennell's (2009) 2D instances, the 20 geometric ones
    (`bubbles*`, `concentricCircles*`, `rotatingDiamonds*`, `chaoSingleDep`),
    `team1_100` to `team6_500`, `bonus1000`, and the arbitrary-radius
    `*rdmRad` ones. The seven plain TSPLIB files (`d493`, `dsj1000`, `kroD100`,
    `lin318`, `pcb442`, `rat195`, `rd400`) are 3D with one radius; the 2D
    problems of Coutinho et al.'s Table 2 are in `mennell-tsplib-2d/`.
  - `Solutions/*.sol`: the tours of Carrabs et al. (2020, INFORMS J. Comput.
    32(4)): `TP: n`, then one `x y` turning point per line. The prefixes
    `002_`, `01_` and `03_` are the overlap ratios 0.02, 0.10 and 0.30.
  - `readme.pdf`: the repository's description.
- `mennell-3d/`: Mennell's 3D TSPLIB files (TSPLIB coordinates divided by
  100, a z, the depot in a comment; the radius column is a placeholder),
  from the instance collection we received. Coutinho et al.'s Table 3 uses
  them; here they are the source of the next folder.
- `mennell-tsplib-2d/`: the 21 TSPLIB problems of Coutinho et al.'s (2016)
  Table 2, 7 problems at overlap ratios 0.02, 0.10 and 0.30, rebuilt from
  `mennell-3d/`. The rule drops z. Every radius is the ratio times the largest
  side of the box around the depot and the points (radii not included),
  written with Python's `repr`; coordinates are copied as they are. The
  optima of all those Coutinho et al. close agree with ours (kroD100: 89.668
  at 0.10, 58.541 at 0.30; see `cetsp-table2-mac-2026-10-27`).
- `behdani/`: Behdani and Smith's (2014) 240 point sets `tp-X-Y.txt` (X
  targets, set Y), from the same collection, with their `readme.txt`. The
  radius is given at run time: their three radii per set make the 720
  instances.
