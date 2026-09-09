from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.dashboard_event import DATA_PATH  # noqa: E402
from dashboard.dashboard_partition import native_partition  # noqa: E402


def main() -> None:
    output = Path(sys.argv[1])
    data = json.loads(DATA_PATH.read_text())
    parts = {str(row["case"]): native_partition(row["geometry"]["polygons"]) for row in data["rows"]}
    with output.open("x") as file:
        json.dump(
            {
                "algorithm": "optimal_convex_partition::decompose_polygon",
                "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                "cases": parts,
            },
            file,
            separators=(",", ":"),
        )


if __name__ == "__main__":
    main()
