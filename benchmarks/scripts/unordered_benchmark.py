#!/usr/bin/env python3
"""Run the unordered C++ solver on a tracked binary suite, with optional external comparison."""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path
from benchmark_cases import read_encoded_cases

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--suite', type=Path, default=ROOT / 'benchmarks/suites/algorithm-dev-v1.bin')
	parser.add_argument('--solver', type=Path, default=ROOT / '.build/unordered/tpp')
	parser.add_argument('--seconds', type=float, default=5)
	parser.add_argument('--max-calls', type=int, default=10000000)
	parser.add_argument('--case', type=int, action='append')
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open('w') as file:
		for case in read_encoded_cases(args.suite):
			if args.case is not None and case.case_index not in args.case:
				continue
			s = struct.unpack_from('<dddd', case.data)
			data = ' '.join(map(str, (*s, case.polygon_count, args.max_calls, args.seconds))) + '\n'
			for polygon in case.polygons:
				data += f'{len(polygon)} ' + ' '.join(str(x) for v in polygon for x in v) + '\n'
			process = subprocess.run([str(args.solver.resolve())], input=data, text=True, capture_output=True, timeout=max(30, args.seconds + 30))
			row = {'case': case.case_index, 'sha256': case.digest, 'polygons': case.polygon_count}
			if process.returncode:
				row['error'] = process.stderr.strip()
			else:
				row.update(json.loads(process.stdout))
				try:
					from shapely.geometry import LineString, Polygon
					line = LineString(row['path'])
					row['max_polygon_distance'] = max((line.distance(Polygon(p)) for p in case.polygons), default=0)
					row['valid'] = row['max_polygon_distance'] <= 1e-7 and row['path'][0] == list(s[:2]) and row['path'][-1] == list(s[2:])
				except ImportError:
					row['valid'] = None
			file.write(json.dumps(row) + '\n')
			file.flush()
			print(json.dumps({k: v for k, v in row.items() if k not in ('path', 'order', 'sha256')}), flush=True)


if __name__ == '__main__':
	main()
