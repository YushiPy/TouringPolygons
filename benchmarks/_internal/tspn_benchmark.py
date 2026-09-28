"""Compare the maintained TSPN B&B and the pinned Fekete SOCP B&B."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import statistics
import subprocess
from benchmark_cases import read_encoded_cases
from unordered_runner import encode_instance, run_unordered_solver
from unordered_validation import validate_cycle

ROOT = Path(__file__).resolve().parents[2]


def default_inputs():
    reference = ROOT/'benchmarks/results-saved/convex-cycle-gurobi-reference-2026-09-25/instances.json'
    cases = json.loads(reference.read_text())['instances']
    def box(x,y,w=1,h=1):return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
    cases += [
        {'name':'free_anchor','polygons':[box(0,0,10,10),box(12,4,1,2)]},
        {'name':'common_point','polygons':[box(0,0,3,3),box(1,1,3,3),box(2,2,2,2)]},
        {'name':'containment','polygons':[box(-10,-10,20,20),box(-4,0),box(3,0)]},
        {'name':'nonconvex_u','polygons':[[[0,0],[6,0],[6,6],[4,6],[4,2],[2,2],[2,6],[0,6]],box(2.5,3)]},
        {'name':'nonconvex_l','polygons':[[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]],box(5,0),box(4,5),box(-2,4)]},
    ]
    rng=random.Random(270927)
    for n in (6,8,10,12):
        for kind in ('boxes','concave'):
            polygons=[]
            for i in range(n):
                x,y=rng.randrange(-20,20),rng.randrange(-20,20)
                polygon=box(x,y,2,2) if kind=='boxes' else [[x,y],[x+3,y],[x+3,y+1],[x+1,y+1],[x+1,y+3],[x,y+3]]
                polygons.append(polygon)
            cases.append({'name':f'seeded_{kind}_{n}','polygons':polygons,'seed':270927})
    corpus=ROOT/'benchmarks/suites/german-instances.bin'
    native=read_encoded_cases(corpus)
    for n in (5,10,15,20):
        selected=[c for c in native if c.polygon_count==n][:2]
        for c in selected:cases.append({'name':f'german_{c.case_index}_n{n}','polygons':c.polygons,
            'source':'benchmarks/suites/german-instances.bin','source_case':c.case_index,'source_sha256':c.digest})
    return {'formulation':'TSPN, free cyclic order, no fixed point, closed polygon regions','instances':cases}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--fekete-source',type=Path)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'.build/tspn-comparison')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seconds',type=int,default=3)
    parser.add_argument('--relative-gap',type=float,default=1e-6)
    parser.add_argument('--feasibility-tolerance',type=float,default=1e-8)
    parser.add_argument('--validation-tolerance',type=float,default=1e-7)
    parser.add_argument('--socp-defaults',action='store_true',help='Keep original Gurobi and spanning tolerances; otherwise tighten them for the requested feasibility checks.')
    args=parser.parse_args(argv)
    if min(args.repetitions,args.seconds,args.relative_gap,args.feasibility_tolerance,args.validation_tolerance)<=0:
        parser.error('counts, limits and reporting tolerances must be positive')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'raw.jsonl').exists():parser.error('choose a new output directory')
    source=args.fekete_source or ROOT/'third_party/tspn-socg'
    if not (source/'tspn_core/CMakeLists.txt').exists():
        common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=ROOT,text=True).strip())
        source=common.parent/'third_party/tspn-socg'
    command=['cmake','-S',str(ROOT/'benchmarks/_internal/tspn_native'),'-B',str(args.build_dir),f'-DFEKETE_SOURCE={source}','-DTARGET=main-unordered']
    # Reuse a locally installed header-only dependency when its Conan package
    # lacks a CMake config. Never write an environment or build into the vendor.
    headers=sorted((Path.home()/'.conan2/p').glob('*/p/include/nlohmann/json.hpp'))
    if headers:command.append(f'-DNLOHMANN_INCLUDE_DIR={headers[0].parents[1]}')
    commands=[command,['cmake','--build',str(args.build_dir),'--target','tpp-fekete-cycle','tpp-unordered','-j','4']]
    with (output/'build.txt').open('w') as log:
        for cmd in commands:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    inputs=json.loads(args.inputs.read_text()) if args.inputs else default_inputs()
    (output/'instances.json').write_text(json.dumps(inputs,indent=2)+'\n')
    ours=args.build_dir/'touring_polygons/tpp-unordered';fekete=args.build_dir/'tpp-fekete-cycle'
    runtime=args.build_dir/('fekete-default-runtime' if args.socp_defaults else 'fekete-strict-runtime')
    runtime.mkdir(parents=True,exist_ok=True)
    gurobi_parameters={} if args.socp_defaults else {'FeasibilityTol':1e-9,'OptimalityTol':1e-9,'BarConvTol':1e-10,'BarQCPConvTol':1e-10}
    (runtime/'gurobi.env').write_text(''.join(f'{key} {value}\n' for key,value in gurobi_parameters.items()))
    spanning=0.0009 if args.socp_defaults else args.feasibility_tolerance/10
    # Match UB <= (1+eps)*LB exactly in algebra, up to binary64 representation.
    our_relative=args.relative_gap/(1+args.relative_gap)
    arguments=['--cycle','--absolute-gap','0','--relative-gap',str(our_relative),
        '--feasibility-tolerance',str(args.feasibility_tolerance)]
    config={'formulation':inputs['formulation'],'repetitions':args.repetitions,'seconds':args.seconds,
        'fekete_relative_gap':args.relative_gap,'our_relative_gap':our_relative,'our_absolute_gap':0,
        'feasibility_tolerance':args.feasibility_tolerance,'validation_tolerance':args.validation_tolerance,
        'threads':1,'external_backend':'socp','external_root':'LongestEdgePlusFurthestSite',
        'external_search':'DfsBfs','external_branching':'FarthestPoly','external_rules':[],
        'external_node_simplification':False,'external_decomposition_branch':True,'external_cutoff':True,
        'external_gurobi_settings':{'Threads':1,'Presolve':0,'SimplexPricing':3,**gurobi_parameters},
        'external_spanning_tolerance':spanning,'socp_defaults':args.socp_defaults,
        'timing':'whole native solve incl preprocessing/heuristic/root, B&B, extraction; excludes process startup and warmed Gurobi environment; backend order alternates by case; sequential runs',
        'platform':platform.platform(),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        'commands':commands,'fekete_source':str(source),'fekete_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for base in (ROOT/'packages/nonconvex-tpp/cpp',ROOT/'packages/convex-tpp/cpp',ROOT/'benchmarks/_internal')
            for p in base.rglob('*') if p.is_file() and (p.suffix in ('.cpp','.h','.py') or p.name=='CMakeLists.txt')},
        'vendor_status':subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[]
    with (output/'raw.jsonl').open('w') as raw:
        for index,case in enumerate(inputs['instances']):
            polygons=case['polygons'];digest=hashlib.sha256(json.dumps(polygons,separators=(',',':')).encode()).hexdigest()
            for backend in (('ours','fekete') if index%2==0 else ('fekete','ours')):
                results=[]
                if backend=='fekete':
                    try:
                        process=subprocess.run([str(fekete),str(args.repetitions),str(args.relative_gap),str(args.feasibility_tolerance),str(spanning)],
                            input=encode_instance((0,0),(0,0),polygons,10**8,args.seconds),text=True,capture_output=True,
                            timeout=args.repetitions*args.seconds+30,cwd=runtime)
                        results=[json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
                        error=process.stderr[-2000:] or f'external exit {process.returncode}'
                    except Exception as e:error=str(e)
                    results += [{'error':error}]*(args.repetitions-len(results))
                else:
                    for repeat in range(args.repetitions):
                        try:results.append(run_unordered_solver(ours,(0,0),(0,0),polygons,10**8,args.seconds,arguments))
                        except Exception as e:results.append({'error':str(e)})
                for repeat,result in enumerate(results):
                    row={'name':case['name'],'k':len(polygons),'sha256':digest,'solver':backend,'repeat':repeat,**result}
                    row['validation']=validate_cycle(polygons,row.get('path',[]),args.validation_tolerance)
                    lo,up=row.get('lower_bound'),row.get('upper_bound')
                    row['gap_closed']=lo is not None and up is not None and up <= (1+args.relative_gap)*lo
                    rows.append(row);raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
                print(f"{index+1}/{len(inputs['instances'])} {case['name']} {backend}: "+
                    ', '.join(f"{r.get('seconds',0):.4f}s" if 'error' not in r else 'ERROR' for r in results),flush=True)
    summaries=[]
    for case in inputs['instances']:
        group={b:[r for r in rows if r['name']==case['name'] and r['solver']==b] for b in ('ours','fekete')}
        summary={'name':case['name'],'k':len(case['polygons'])}
        for b,g in group.items():
            times=[r['seconds'] for r in g if 'seconds' in r]
            summary[b]={'median_seconds':statistics.median(times) if times else None,
                'valid_runs':sum(r['validation']['valid'] for r in g),'gap_closed_runs':sum(r['gap_closed'] for r in g),
                'errors':sum('error' in r for r in g),
                'median_calls':statistics.median([r['calls'] for r in g if 'calls' in r]) if times else None}
        complete=[r for g in group.values() for r in g if r.get('upper_bound') is not None and r.get('lower_bound') is not None]
        summary['objective_spread']=max(r['upper_bound'] for r in complete)-min(r['upper_bound'] for r in complete) if complete else None
        summary['interval_separation']=max(0,max(r['lower_bound'] for r in complete)-min(r['upper_bound'] for r in complete)) if complete else None
        summaries.append(summary)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    lines=['# TSPN: maintained B&B versus Fekete SOCP B&B','',
        'Native-call medians in milliseconds; see config.json for matched gap, validation tolerance and timing scope.','',
        '| Instance | k | Ours ms | Fekete ms | Ours valid/closed | Fekete valid/closed | Objective spread | Interval separation |',
        '|---|---:|---:|---:|---|---|---:|---:|']
    for r in summaries:
        a,b=r['ours'],r['fekete'];times=[f'{1000*x["median_seconds"]:.3f}' if x['median_seconds'] is not None else 'error' for x in (a,b)]
        lines.append(f'| {r["name"]} | {r["k"]} | {times[0]} | {times[1]} | {a["valid_runs"]}/{a["gap_closed_runs"]} | {b["valid_runs"]}/{b["gap_closed_runs"]} | {r["objective_spread"]} | {r["interval_separation"]} |')
    lines+=['','Counts refer to repetitions. A numerical bound from the external solver is not an exact certificate. Raw tour validation and claimed optimality are reported separately. Timed-out searches remain in the table; their times are not times to optimality.']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    (output/'README.md').write_text('Run `python3 benchmarks/tpp.py tspn-benchmark --output NEW_DIRECTORY`. Inputs, raw runs, configuration, source hashes and analysis are preserved together. The external checkout is read only; its original SOCP backend is selected.\n')
    return int(any(not r['validation']['valid'] or 'error' in r for r in rows if r['solver']=='ours'))
