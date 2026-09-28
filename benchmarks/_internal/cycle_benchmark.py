"""Fixed-order convex cycle comparison; public entry: benchmarks/tpp.py."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def box(x, y, w, h):
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]


def default_inputs():
    reference = ROOT / "benchmarks/results-saved/convex-cycle-gurobi-reference-2026-09-25/instances.json"
    cases = json.loads(reference.read_text())["instances"]
    cases += [
        {"name":"overlap_zero", "polygons":[box(0,0,2,2),box(1,1,2,2)]},
        {"name":"nested_positive", "polygons":[box(-10,-10,20,20),box(-4,0,1,1),box(3,0,1,1)]},
        {"name":"subunit_zero_link", "polygons":[box(-3,-1,1,1),box(0,-2,2,4),box(-2,-2,4,2),box(0,2,1,1)]},
        {"name":"overlap_positive", "polygons":[box(0,0,3,2),box(2,0,3,2),box(4,4,2,2),box(-1,4,2,2)]},
        {"name":"nonsmooth_anchor_orthic", "polygons":[box(-10,-10,20,10),[[2,4],[6,0],[12,6],[8,10]],[[0,0],[2,4],[-6,8],[-8,4]]]},
        {"name":"nonrepresentable_common_point", "polygons":[[[1,0],[-1,1],[-3,-3]],[[0,1],[1,-1],[3,3]],[[0,0],[1,1],[-2,2]]]},
    ]
    for k in (8,16,32,64):
        for shape in ("boxes","slanted"):
            polygons=[]
            for i in range(k):
                x,y=round(10000*math.cos(2*math.pi*i/k)),round(10000*math.sin(2*math.pi*i/k))
                p=box(x-20,y-20,40,40) if shape=="boxes" else [[x-20,y-10],[x+10,y-20],[x+20,y+10],[x-10,y+20]]
                polygons.append(p)
            cases.append({"name":f"ring_{shape}_{k}","polygons":polygons})
    for vertices in (16,64):
        polygons=[]
        for i in range(5):
            x,y=round(1000000*math.cos(2*math.pi*i/5)),round(1000000*math.sin(2*math.pi*i/5))
            polygons.append([[x+round(10000*math.cos(2*math.pi*j/vertices)),y+round(10000*math.sin(2*math.pi*j/vertices))] for j in range(vertices)])
        cases.append({"name":f"many_edges_{vertices}","polygons":polygons})
    return {"schema":"ordered-convex-cycle-benchmark-v1","instances":cases}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--repetitions",type=int,default=15)
    parser.add_argument("--build-dir",type=Path,default=ROOT/".build/cycle-benchmark")
    parser.add_argument("--gurobi-home",type=Path,default=Path(os.environ.get("GUROBI_HOME","/Library/gurobi1303/macos_universal2")))
    args=parser.parse_args(argv)
    if args.repetitions<1:parser.error("repetitions must be positive")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"raw.jsonl").exists():parser.error("output already contains a run; choose another directory")
    inputs=json.loads(args.inputs.read_text()) if args.inputs else default_inputs()
    (output/"instances.json").write_text(json.dumps(inputs,indent=2)+"\n")
    commands=[
        ["cmake","-S",str(ROOT/"packages/convex-tpp/cpp"),"-B",str(args.build_dir),"-DTARGET=main-cycle_benchmark","-DCMAKE_BUILD_TYPE=Release","-DTPP_ENABLE_GUROBI=ON",f"-DGUROBI_HOME={args.gurobi_home}"],
        ["cmake","--build",str(args.build_dir),"-j","4"],
    ]
    with (output/"build.log").open("w") as log:
        for command in commands:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    source_files=list((ROOT/"packages/convex-tpp/cpp").rglob("*.cpp"))+list((ROOT/"packages/convex-tpp/cpp").rglob("*.h"))
    source_files += [Path(__file__),ROOT/"packages/convex-tpp/cpp/CMakeLists.txt"]
    command=[str(args.build_dir/"tpp-convex"),str(output/"instances.json"),str(args.repetitions)]
    configuration={
        "formulation":"fixed cyclic order, sum of Euclidean link lengths, closed convex polygons, free contacts",
        "platform":platform.platform(),"machine":platform.machine(),
        "compiler":subprocess.check_output(["c++","--version"],text=True).splitlines()[0],
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "commands":commands+[command],"repetitions":args.repetitions,"warmups_per_solver":1,
        "timing":"whole API incl validation and certification; Gurobi includes model build, extraction and destruction; environment excluded; backend order rotates per repetition",
        "gurobi_parameters":{"Threads":1,"Method":2,"NumericFocus":2,"FeasibilityTol":1e-9,"OptimalityTol":1e-9,"BarConvTol":1e-10,"TimeLimit":30},
        "cpp_acceptance_epsilon":None,"cpp_discretization":None,
        "double_rational_recovery":True,
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    }
    (output/"config.json").write_text(json.dumps(configuration,indent=2)+"\n")
    with (output/"raw.jsonl").open("w") as raw,(output/"stderr.log").open("w") as errors:
        subprocess.run(command,cwd=ROOT,stdout=raw,stderr=errors,check=True)
    rows=[json.loads(line) for line in (output/"raw.jsonl").read_text().splitlines() if line.startswith("{")]
    summary=[]
    for instance in inputs["instances"]:
        group=[r for r in rows if r["name"]==instance["name"]]
        row={"name":instance["name"],"k":len(instance["polygons"])}
        for key in ("rational_seconds","double_seconds","gurobi_total_seconds","gurobi_optimize_seconds"):
            times=sorted(r[key] for r in group)
            row[key]={"median":statistics.median(times),"min":min(times),"max":max(times)}
        row["rational_optimal"]=all(r["rational_status"]==2 and r["rational_certificate"]==3 for r in group)
        row["double_feasible"]=all(r["double_status"] in (2,3) and r["double_certificate"] in (2,3) for r in group)
        row["gurobi_optimal_status"]=all(r["gurobi_status"]==2 for r in group)
        row["independent_intervals_overlap"]=all(r["gurobi_certified_lower"]<=r["upper"] and r["lower"]<=r["gurobi_certified_upper"] for r in group)
        row["max_double_gap"]=max(r["double_upper"]-r["double_lower"] for r in group)
        row["max_double_objective_difference"]=max(abs(r["double_upper"]-r["upper"]) for r in group)
        row["max_gurobi_objective_difference"]=max(abs(r["gurobi_objective"]-r["upper"]) for r in group)
        row["max_rational_recoveries"]=max(r["rational_recoveries"] for r in group)
        for kind in ("anchor", "feature", "cycle"):
            row[f"max_rational_{kind}_recoveries"]=max(r[f"rational_{kind}_recoveries"] for r in group)
        summary.append(row)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    text=["# Certified cycle comparison", "", "Medians in milliseconds. See config.json for timing scope and solver parameters.","",
          "| Instance | Rational | Double | Gurobi total | Gurobi optimize | Exact / feasible / intervals | Rational recoveries |",
          "|---|---:|---:|---:|---:|---|---:|"]
    for row in summary:
        numbers=[1000*row[key]["median"] for key in ("rational_seconds","double_seconds","gurobi_total_seconds","gurobi_optimize_seconds")]
        text.append("| "+row["name"]+" | "+" | ".join(f"{x:.4f}" for x in numbers)+f" | {row['rational_optimal']} / {row['double_feasible']} / {row['independent_intervals_overlap']} | {row['max_rational_recoveries']} |")
    text += ["", "Gurobi objective/bound are numerical; independently certified intervals use exactly feasible rational contractions of its contacts. No coordinate equality is required.","",
             "Double uses the default exact recovery option. Recovery counts are per call; raw data distinguish feature, anchor and complete-cycle recovery. Validation and certificates use exact predicates in both modes.","",
             "These finite synthetic instances do not establish universal speed dominance or prove coverage of all intersection degeneracies."]
    (output/"analysis.md").write_text("\n".join(text)+"\n")
    (output/"README.md").write_text("Run with `python3 benchmarks/tpp.py cycle-benchmark --output NEW_DIRECTORY`. Inputs, raw repetitions, build log, configuration, source hashes and analysis are saved together. The earlier 2026-09-25 Gurobi reference campaign is unchanged.\n")
    print("\n".join(text))
    return 0 if all(r["rational_optimal"] and r["double_feasible"] and r["gurobi_optimal_status"] and r["independent_intervals_overlap"] for r in summary) else 1
