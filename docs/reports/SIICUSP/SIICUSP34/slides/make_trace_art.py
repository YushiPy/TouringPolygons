"""Real search trace for the method slide: corpus case 17 (5 regions, trace-data.js
case '17', identified by SHA-256). Writes trace-art.tex with four mini maps:
root, [4], [3,4] (pruned) and [4,3] (optimal). apps/siicusp34 data is read-only."""
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))
from benchmark_cases import read_encoded_cases  # noqa: E402

CASE = 17
SHA = "71ad8f7d7ca3cd1e974c06c5cb3b3203973d4324a8faef"
WIDTH = 4.4                                                  # cm per mini map

case = read_encoded_cases(ROOT / "benchmarks/results-saved/fekete-comparison/instances.bin")[CASE]
text = (ROOT / "apps/siicusp34/data/trace-data.js").read_text()
trace = json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))["cases"][str(CASE)]
assert trace["sha256"].startswith(SHA[:20]) and hashlib.sha256(case.data).hexdigest().startswith(SHA[:20])
events = trace["events"]
node = {e["node"]: e for e in events if e["kind"] == "oracle"}
root = next(e for e in events if e["kind"] == "root")
pruned = next(e for e in events if e["kind"] == "child" and e.get("pruned") and e["reason"] == "bound")
U = next(e for e in events if e["kind"] == "complete")["upper_bound"]
print("U", U, "root L", root["lower_bound"], "node1", node[1]["lower_bound"],
      "pruned", pruned["lower_bound"], pruned["sequence"], "node2", node[2]["lower_bound"], node[2]["sequence"])

k = WIDTH / 153.7
f = lambda x, y: f"({x * k:.4f},{y * k:.4f})"


def mini(name, highlight, path=None, incidental=()):
    out = [f"\\def\\{name}{{%"]
    for i, poly in enumerate(case.polygons):
        fill = "Teal" if i in highlight else ("Teal!35" if i in incidental else "Rule!60")
        out.append(f"\\filldraw[fill={fill},draw=Ink,line width=0.5pt,line join=round] "
                   + " -- ".join(f(x, y) for x, y in poly) + " -- cycle;")
    if path:
        out.append("\\draw[Accent,line width=1.8pt,line join=round,line cap=round] "
                   + " -- ".join(f(x, y) for x, y in path) + ";")
    out.append(f"\\fill[white,draw=Ink,line width=0.8pt] {f(0, 0)} circle (0.07);")
    out.append(f"\\fill[Ink] {f(153.67, 68.98)} circle (0.07);")
    out.append("}")
    return out


br = lambda v, n: f"{v:.{n}f}".replace(".", "{,}")      # decimal comma inside math
lines = [f"\\def\\traceU{{{br(U, 2)}}}",
         f"\\def\\traceLroot{{{br(root['lower_bound'], 1)}}}\\def\\traceLb{{{br(node[1]['lower_bound'], 1)}}}"
         f"\\def\\traceLc{{{br(pruned['lower_bound'], 1)}}}\\def\\traceLd{{{br(node[2]['lower_bound'], 2)}}}"]
lines += mini("miniRoot", set(), root["path"])
lines += mini("miniB", set(node[1]["sequence"]), node[1]["path"])
lines += mini("miniC", set(pruned["sequence"]))
lines += mini("miniD", set(node[2]["sequence"]), node[2]["path"], incidental={0, 1, 2})   # touched on the way
(HERE / "trace-art.tex").write_text("\n".join(lines).replace("\\def\\traceLroot", "\\def\\traceLroot") + "\n")
