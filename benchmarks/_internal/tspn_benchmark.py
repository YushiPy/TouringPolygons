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
import zipfile
from benchmark_cases import read_encoded_cases
from convert_instances import convert_json_case
from unordered_runner import encode_instance, run_unordered_solver
from unordered_validation import validate_cycle

ROOT = Path(__file__).resolve().parents[2]
SIZE_BANDS = ((5, 10), (11, 20), (21, 40), (41, 60))


def _socg_class(meta):
    if 'geo_information' in meta:
        return 'OSM'
    if meta.get('source') == 'random':
        return 'random'
    if meta.get('source') == 'public_instance_set':
        return 'tessellation'
    return None


def select_socg_inputs(archive_path, per_stratum=1):
    """Select lexicographically first cases by supplier metadata and size band."""
    if per_stratum < 1:
        raise ValueError('per-stratum must be positive')
    classes = ('OSM', 'random', 'tessellation')
    candidates = {(kind, band): [] for kind in classes for band in SIZE_BANDS}
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.json'))
        for name in names:
            payload = archive.read(name)
            data = json.loads(payload)
            kind = _socg_class(data.get('meta', {}))
            if kind is None:
                continue
            converted = convert_json_case(name, payload)
            k = len(converted.polygons)
            band = next((b for b in SIZE_BANDS if b[0] <= k <= b[1]), None)
            if band is None:
                continue
            candidates[kind, band].append((name, converted))
        selected = []
        counts = {}
        for kind in classes:
            for band in SIZE_BANDS:
                items = candidates[kind, band]
                counts[f'{kind}:{band[0]}-{band[1]}'] = len(items)
                for name, case in items[:per_stratum]:
                    selected.append({
                        'name': f'{kind.lower()}_{band[0]}-{band[1]}_{Path(name).stem}',
                        'source': str(archive_path), 'source_name': name,
                        'source_class': kind, 'size_band': f'{band[0]}-{band[1]}',
                        'polygons': case.polygons, 'meta': case.meta,
                        'vertex_count': sum(map(len, case.polygons)),
                    })
    return selected, counts


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
    parser.add_argument('--instances-zip',type=Path,help='Select cases from the simplified SOCG ZIP by metadata class and size band.')
    parser.add_argument('--per-stratum',type=int,default=1)
    parser.add_argument('--external-timeout',type=float,default=12,
        help='Firm timeout for each repetition in its own process, for either backend; timeouts are censored.')
    parser.add_argument('--ours-binary',type=Path,help='Use a previously frozen native binary.')
    parser.add_argument('--fekete-binary',type=Path,help='Use a previously built external binary.')
    parser.add_argument('--reference-results',type=Path,
        help='Reuse Fekete rows from a prior campaign after verifying input hashes and all solver settings match.')
    parser.add_argument('--skip-build',action='store_true',help='Use the binaries supplied with --ours-binary/--fekete-binary.')
    parser.add_argument('--fekete-source',type=Path)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'.build/tspn-comparison')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seconds',type=int,default=3)
    parser.add_argument('--portfolio',action='store_true',
        help='Run the cooperative two-search portfolio (two total worker threads).')
    parser.add_argument('--portfolio-no-sharing',action='store_true',
        help='Run two independent searches in a two-worker race; implies --portfolio.')
    parser.add_argument('--search-strategy',choices=('best-bound','dfs-bfs'),
        help='Run one isolated B&B strategy instead of the default strategy.')
    parser.add_argument('--cycle-optimization',action='append',choices=('cache','dual','features','lazy','root','branch','one-tree','learn','memo','bound-first','dual-screen','interval','share-bounds'),default=[],
        help='Enable one native cycle optimization; repeat to combine independently selectable optimizations.')
    parser.add_argument('--relative-gap',type=float,default=1e-6)
    parser.add_argument('--feasibility-tolerance',type=float,default=1e-8)
    parser.add_argument('--validation-tolerance',type=float,default=1e-7)
    parser.add_argument('--socp-defaults',action='store_true',help='Keep original Gurobi and spanning tolerances; otherwise tighten them for the requested feasibility checks.')
    args=parser.parse_args(argv)
    if args.inputs and args.instances_zip: parser.error('use either --inputs or --instances-zip')
    if args.search_strategy and (args.portfolio or args.portfolio_no_sharing):
        parser.error('--search-strategy cannot be combined with portfolio options')
    if min(args.repetitions,args.seconds,args.relative_gap,args.feasibility_tolerance,args.validation_tolerance,args.external_timeout,args.per_stratum)<=0:
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
    if not args.skip_build:
        with (output/'build.txt').open('w') as log:
            for cmd in commands:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    elif not (args.ours_binary and args.fekete_binary):
        parser.error('--skip-build requires --ours-binary and --fekete-binary')
    if args.instances_zip:
        selected, stratum_counts = select_socg_inputs(args.instances_zip.resolve(), args.per_stratum)
        inputs={'formulation':'TSPN, free cyclic order, no fixed point, closed polygon regions',
            'selection':{'archive':str(args.instances_zip.resolve()),'policy':'lexicographically first filenames within supplier metadata class and polygon-count band; chosen before timing',
                'per_stratum':args.per_stratum,'stratum_counts':stratum_counts},'instances':selected}
    else:
        inputs=json.loads(args.inputs.read_text()) if args.inputs else default_inputs()
    (output/'instances.json').write_text(json.dumps(inputs,indent=2)+'\n')
    ours=(args.ours_binary or (args.build_dir/'touring_polygons/tpp-unordered')).resolve()
    fekete=(args.fekete_binary or (args.build_dir/'tpp-fekete-cycle')).resolve()
    runtime=args.build_dir/('fekete-default-runtime' if args.socp_defaults else 'fekete-strict-runtime')
    runtime.mkdir(parents=True,exist_ok=True)
    gurobi_parameters={} if args.socp_defaults else {'FeasibilityTol':1e-9,'OptimalityTol':1e-9,'BarConvTol':1e-10,'BarQCPConvTol':1e-10}
    (runtime/'gurobi.env').write_text(''.join(f'{key} {value}\n' for key,value in gurobi_parameters.items()))
    spanning=0.0009 if args.socp_defaults else args.feasibility_tolerance/10
    # Match UB <= (1+eps)*LB exactly in algebra, up to binary64 representation.
    our_relative=args.relative_gap/(1+args.relative_gap)
    arguments=['--cycle','--absolute-gap','0','--relative-gap',str(our_relative),
        '--feasibility-tolerance',str(args.feasibility_tolerance)]
    if args.portfolio_no_sharing:
        arguments.append('--portfolio-no-sharing')
    elif args.portfolio:
        arguments.append('--portfolio')
    elif args.search_strategy:
        arguments.extend(['--search-strategy',args.search_strategy])
    for optimization in args.cycle_optimization:
        arguments.extend(['--cycle-optimization',optimization])
    portfolio=args.portfolio or args.portfolio_no_sharing
    config={'formulation':inputs['formulation'],'repetitions':args.repetitions,'seconds':args.seconds,
        'external_process_timeout_seconds':args.external_timeout,
        'fekete_relative_gap':args.relative_gap,'our_relative_gap':our_relative,'our_absolute_gap':0,
        'feasibility_tolerance':args.feasibility_tolerance,'validation_tolerance':args.validation_tolerance,
        'threads':2 if portfolio else 1,'ours_threads':2 if portfolio else 1,'fekete_threads':1,
        'portfolio_mode':'independent-race' if args.portfolio_no_sharing else ('cooperative' if args.portfolio else None),
        'search_strategy':args.search_strategy or ('default' if not portfolio else None),
        'cycle_optimizations':args.cycle_optimization,
        'external_backend':'socp','external_root':'LongestEdgePlusFurthestSite',
        'external_search':'DfsBfs','external_branching':'FarthestPoly','external_rules':[],
        'external_node_simplification':False,'external_decomposition_branch':True,'external_cutoff':True,
        'external_gurobi_settings':{'Threads':1,'Presolve':0,'SimplexPricing':3,**gurobi_parameters},
        'external_spanning_tolerance':spanning,'socp_defaults':args.socp_defaults,
        'timing':'whole native solve incl preprocessing/heuristic/root, B&B, extraction and portfolio cooperative join tail; excludes process startup and warmed Gurobi environment; backend order alternates by case; sequential runs',
        'platform':platform.platform(),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        'commands':commands if not args.skip_build else [],'frozen_binaries':{'ours':str(ours),'fekete':str(fekete)},
        'binary_sha256':{'ours':hashlib.sha256(ours.read_bytes()).hexdigest(),'fekete':hashlib.sha256(fekete.read_bytes()).hexdigest()},
        'fekete_source':str(source),'fekete_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_hash_scope':'Workspace snapshot at measurement; frozen binaries require their own build provenance.',
        'workspace_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for base in (ROOT/'packages/nonconvex-tpp/cpp',ROOT/'packages/convex-tpp/cpp',ROOT/'benchmarks/_internal')
            for p in base.rglob('*') if p.is_file() and (p.suffix in ('.cpp','.h','.py') or p.name=='CMakeLists.txt')},
        'vendor_status':subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)}
    reference_rows={}
    if args.reference_results:
        reference_dir=args.reference_results.resolve()
        reference_config=json.loads((reference_dir/'config.json').read_text())
        reference_inputs=json.loads((reference_dir/'instances.json').read_text())
        reference_settings=('repetitions','seconds','external_process_timeout_seconds','fekete_relative_gap',
            'feasibility_tolerance','validation_tolerance','fekete_threads','external_backend',
            'external_gurobi_settings','external_spanning_tolerance','socp_defaults',
            'external_root','external_search','external_branching','external_rules',
            'external_node_simplification','external_decomposition_branch','external_cutoff')
        for key in reference_settings:
            if reference_config.get(key)!=config.get(key):
                parser.error(f'--reference-results setting mismatch for {key}')
        if reference_config.get('binary_sha256',{}).get('fekete')!=config['binary_sha256']['fekete']:
            parser.error('--reference-results Fekete binary hash mismatch')
        if reference_config.get('fekete_commit')!=config['fekete_commit']:
            parser.error('--reference-results Fekete source commit mismatch')
        current_hashes={hashlib.sha256(json.dumps(c['polygons'],separators=(',',':')).encode()).hexdigest()
            for c in inputs['instances']}
        reference_hashes={hashlib.sha256(json.dumps(c['polygons'],separators=(',',':')).encode()).hexdigest()
            for c in reference_inputs['instances']}
        if current_hashes!=reference_hashes:
            parser.error('--reference-results inputs do not match')
        expected_reference_keys={(hashlib.sha256(json.dumps(c['polygons'],separators=(',',':')).encode()).hexdigest(),repeat)
            for c in inputs['instances'] for repeat in range(args.repetitions)}
        with (reference_dir/'raw.jsonl').open() as reference_raw:
            for line in reference_raw:
                row=json.loads(line)
                if row.get('solver')=='fekete':
                    key=(row.get('sha256'),row.get('repeat'))
                    if key not in expected_reference_keys or key in reference_rows:
                        parser.error('--reference-results has an unexpected or duplicate Fekete input/repetition row')
                    reference_rows[key]=row
        if set(reference_rows)!=expected_reference_keys:
            parser.error('--reference-results does not contain the exact Fekete input/repetition key set')
        raw_reference=reference_dir/'raw.jsonl'
        config['fekete_reference_results']={'path':str(reference_dir),
            'raw_sha256':hashlib.sha256(raw_reference.read_bytes()).hexdigest(),
            'config_sha256':hashlib.sha256((reference_dir/'config.json').read_bytes()).hexdigest(),
            'timing':'reused native-independent Fekete rows; no Fekete solve was performed in this run'}
        config['fekete_timing']='reused from the verified reference campaign'
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[]
    with (output/'raw.jsonl').open('w') as raw:
        for index,case in enumerate(inputs['instances']):
            polygons=case['polygons'];digest=hashlib.sha256(json.dumps(polygons,separators=(',',':')).encode()).hexdigest()
            for backend in (('ours','fekete') if index%2==0 else ('fekete','ours')):
                results=[]
                for repeat in range(args.repetitions):
                    try:
                        if backend=='fekete':
                            if reference_rows:
                                results.append({k:v for k,v in reference_rows[(digest,repeat)].items()
                                    if k not in ('name','k','sha256','solver','repeat','validation','gap_closed')})
                                continue
                            # Both backends get one fresh process and the same
                            # firm timeout per repetition. Native solve timing
                            # excludes startup and the Gurobi environment.
                            process=subprocess.run([str(fekete),'1',str(args.relative_gap),str(args.feasibility_tolerance),str(spanning)],
                                input=encode_instance((0,0),(0,0),polygons,10**8,args.seconds),text=True,capture_output=True,
                                timeout=args.external_timeout,cwd=runtime)
                            parsed=[]
                            for line in process.stdout.splitlines():
                                if not line.startswith('{'):continue
                                try:parsed.append(json.loads(line))
                                except json.JSONDecodeError:continue
                            if process.returncode or len(parsed)!=1:
                                results.append({'error':f'external exit {process.returncode}; complete JSON records: {len(parsed)}'})
                            else:results.append(parsed[0])
                        else:
                            results.append(run_unordered_solver(ours,(0,0),(0,0),polygons,10**8,args.seconds,arguments,
                                process_timeout=args.external_timeout))
                    except subprocess.TimeoutExpired:
                        # Do not preserve solver stderr: commercial solver
                        # startup messages can contain license information.
                        results.append({'timeout':True,'error':f'firm process timeout after {args.external_timeout}s',
                            'timeout_seconds':args.external_timeout})
                    except Exception as error:
                        results.append({'error':f'{backend} runner failed: {type(error).__name__}'})
                for repeat,result in enumerate(results):
                    row={**result,'name':case['name'],'k':len(polygons),'sha256':digest,'solver':backend,'repeat':repeat}
                    row['validation']=validate_cycle(polygons,row.get('path',[]),args.validation_tolerance) if row.get('path') else {'valid':False,'status':'no completed tour'}
                    lo,up=row.get('lower_bound'),row.get('upper_bound')
                    row['gap_closed']=lo is not None and up is not None and up <= (1+args.relative_gap)*lo
                    rows.append(row);raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
                print(f"{index+1}/{len(inputs['instances'])} {case['name']} {backend}: "+
                    ', '.join(f"{r.get('seconds',0):.4f}s" if 'error' not in r else 'ERROR' for r in results),flush=True)
    summaries=[]
    for case in inputs['instances']:
        group={b:[r for r in rows if r['name']==case['name'] and r['solver']==b] for b in ('ours','fekete')}
        summary={'name':case['name'],'k':len(case['polygons']),
            'source_class':case.get('source_class'),'size_band':case.get('size_band')}
        for b,g in group.items():
            times=[r['seconds'] for r in g if 'seconds' in r]
            summary[b]={'median_seconds':statistics.median(times) if times else None,
                'valid_runs':sum(r['validation']['valid'] for r in g),'gap_closed_runs':sum(r['gap_closed'] for r in g),
                'errors':sum('error' in r and not r.get('timeout') for r in g),
                'timeouts':sum(bool(r.get('timeout')) for r in g),
                'median_calls':statistics.median([r['calls'] for r in g if 'calls' in r]) if any('calls' in r for r in g) else None}
        complete=[r for g in group.values() for r in g if r.get('upper_bound') is not None and r.get('lower_bound') is not None]
        summary['objective_spread']=max(r['upper_bound'] for r in complete)-min(r['upper_bound'] for r in complete) if complete else None
        summary['interval_separation']=max(0,max(r['lower_bound'] for r in complete)-min(r['upper_bound'] for r in complete)) if complete else None
        matched=all(len(g)==args.repetitions and all(r['validation']['valid'] and r['gap_closed'] and
            'error' not in r and 'seconds' in r for r in g) for g in group.values())
        summary['matched_speedup']=summary['fekete']['median_seconds']/summary['ours']['median_seconds'] if matched and summary['ours']['median_seconds']>0 else None
        summaries.append(summary)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    strata=[]
    for kind,band in dict.fromkeys((r['source_class'],r['size_band']) for r in summaries):
        group=[r for r in summaries if (r['source_class'],r['size_band'])==(kind,band)]
        ratios=[r['matched_speedup'] for r in group if r['matched_speedup'] is not None]
        strata.append({'source_class':kind,'size_band':band,'cases':len(group),
            'matched_cases':len(ratios),'ours_faster':sum(x>1 for x in ratios),
            'median_speedup':statistics.median(ratios) if ratios else None,
            'mean_speedup':statistics.mean(ratios) if ratios else None,
            **{solver:{key:sum(r[solver][key] for r in group)
                for key in ('valid_runs','gap_closed_runs','errors','timeouts')} for solver in ('ours','fekete')}})
    (output/'strata.json').write_text(json.dumps(strata,indent=2)+'\n')
    lines=['# TSPN: maintained B&B versus Fekete SOCP B&B','',
        'Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.','',
        '| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        a,b=r['ours'],r['fekete']
        def cell(stat):
            seconds='—' if stat['median_seconds'] is None else f'{1000*stat["median_seconds"]:.3f}'
            return f'{seconds} / {stat["median_calls"] if stat["median_calls"] is not None else "—"} / {stat["gap_closed_runs"]}'
        lines.append(f'| {r.get("source_class") or "—"} | {r.get("size_band") or "—"} | {r["name"]} | {r["k"]} | {cell(a)} | {cell(b)} | {a["valid_runs"]} / {b["valid_runs"]} | {a["timeouts"]} / {b["timeouts"]} | {r["objective_spread"]} | {r["interval_separation"]} | {r["matched_speedup"]} |')
    lines+=['','Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    (output/'README.md').write_text('Run `python3 benchmarks/tpp.py tspn-benchmark --output NEW_DIRECTORY`. Inputs, raw runs, configuration, source hashes and analysis are preserved together. The external checkout is read only; its original SOCP backend is selected.\n')
    return int(any(not r['validation']['valid'] or 'error' in r for r in rows if r['solver']=='ours'))
