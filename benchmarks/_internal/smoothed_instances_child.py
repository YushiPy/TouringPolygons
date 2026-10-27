"""Annotates smoothed instances with Fekete et al.'s own add_order_annotations.

``smoothed_instances.py`` runs this file with the Python of the upstream build, so it imports
only the standard library and the upstream package ``tspn_bnb2``. Each input line holds one
instance (name, WKT polygons, meta); the polygons are not simplified, so both solvers get the
same regions. usage: smoothed_instances_child.py INSTANCES.jsonl OUTPUT.zip
"""

from __future__ import annotations

import json
import sys
import zipfile


def main(argv: list[str]) -> int:
	from tspn_bnb2.order_annotation import add_order_annotations
	from tspn_bnb2.schemas import AnnotatedInstance

	source, target = argv
	with open(source) as lines, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
		for line in lines:
			item = json.loads(line)
			annotated = add_order_annotations(AnnotatedInstance(polygons=item["polygons"], meta=item["meta"]))
			out.writestr(item["name"] + ".json", annotated.model_dump_json())
			print(f"{item['name']}: anotada", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
