"""Compare the maintained TSPN B&B and the pinned Fekete SOCP B&B."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import subprocess
import zipfile
import time
from tspn_diagnostics import atomic_json, digest, finite, load_rows, row_key, write_reports
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


def select_socg_inputs(archive_path, per_stratum=1, seed=None):
    """Sample without replacement per metadata class and actual polygon count.

    seed=None retains the legacy lexical selection for existing campaigns.
    A seeded selection interleaves strata, so interrupted runs cover all classes.
    """
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
        rng = random.Random(seed)
        groups = []
        for kind in classes:
            for band in SIZE_BANDS:
                items = candidates[kind, band]
                counts[f'{kind}:{band[0]}-{band[1]}'] = len(items)
                if seed is not None:
                    rng.shuffle(items)
                group = []
                for name, case in items[:per_stratum]:
                    group.append({
                        'name': f'{kind.lower()}_{band[0]}-{band[1]}_{Path(name).stem}',
                        'source': str(archive_path), 'source_name': name,
                        'source_class': kind, 'size_band': f'{band[0]}-{band[1]}',
                        'polygons': case.polygons, 'meta': case.meta,
                        'vertex_count': sum(map(len, case.polygons)),
                    })
                groups.append(group)
        if seed is None:
            selected = [case for group in groups for case in group]
        else:
            selected = [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]
    return selected, counts


def select_all_socg_inputs(archive_path):
    """Select every case in the archive, ignoring strata and size bands."""
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.json'))
        selected, counts = [], {}
        for name in names:
            payload = archive.read(name)
            data = json.loads(payload)
            kind = _socg_class(data.get('meta', {})) or 'unknown'
            case = convert_json_case(name, payload)
            k = len(case.polygons)
            band = next((b for b in SIZE_BANDS if b[0] <= k <= b[1]), None)
            band_label = f'{band[0]}-{band[1]}' if band else 'out-of-band'
            key = f'{kind}:{band_label}'
            counts[key] = counts.get(key, 0) + 1
            selected.append({
                'name': Path(name).stem,
                'source': str(archive_path), 'source_name': name,
                'source_class': kind, 'size_band': band_label,
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
    corpus=ROOT/'benchmarks/suites/fekete-instances.bin'
    native=read_encoded_cases(corpus)
    for n in (5,10,15,20):
        selected=[c for c in native if c.polygon_count==n][:2]
        for c in selected:cases.append({'name':f'fekete_{c.case_index}_n{n}','polygons':c.polygons,
            'source':'benchmarks/suites/fekete-instances.bin','source_case':c.case_index,'source_sha256':c.digest})
    return {'formulation':'TSPN, free cyclic order, no fixed point, closed polygon regions','instances':cases}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--instances-zip',type=Path,help='Select cases from the simplified SOCG ZIP by metadata class and size band.')
    parser.add_argument('--profile',choices=('quick','overnight'),help='Balanced seeded SOCG campaign with conservative runtime presets.')
    parser.add_argument('--seed',type=int,help='Deterministic uniform selection per stratum; profiles default to 20260930.')
    parser.add_argument('--resume',action='store_true',help='Skip persisted runs after checking settings, inputs and binary hashes. Never rebuild on resume.')
    parser.add_argument('--dry-run',action='store_true',help='Print the selected workload and process-time ceiling without building or running solvers.')
    parser.add_argument('--report-only',action='store_true',help='Regenerate reports from this output directory without building or running solvers.')
    parser.add_argument('--per-stratum',type=int)
    parser.add_argument('--all',action='store_true',help='Select every case in --instances-zip, ignoring strata and size bands.')
    parser.add_argument('--solver',choices=('both','ours','fekete'),default='both',help='Which solvers to run; ours needs no Gurobi.')
    parser.add_argument('--external-timeout',type=float,
        help='Firm timeout for each repetition in its own process, for either backend; timeouts are censored.')
    parser.add_argument('--capture-oracles',action='store_true',help='Flush every native oracle input and exclusive phase timing to local JSONL; diagnostic runs include capture overhead.')
    parser.add_argument('--ours-binary',type=Path,help='Use a previously frozen native binary.')
    parser.add_argument('--fekete-binary',type=Path,help='Use a previously built external binary.')
    parser.add_argument('--reference-results',type=Path,
        help='Reuse Fekete rows from a prior campaign after verifying input hashes and all solver settings match.')
    parser.add_argument('--skip-build',action='store_true',help='Use the binaries supplied with --ours-binary/--fekete-binary.')
    parser.add_argument('--fekete-source',type=Path)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'.build/tspn-comparison')
    parser.add_argument('--repetitions',type=int)
    parser.add_argument('--seconds',type=int)
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
    output=args.output.resolve()
    if args.report_only:
        inputs=json.loads((output/'instances.json').read_text())
        config=json.loads((output/'config.json').read_text())
        expected={(c['name'],digest(c['polygons']),b,r) for c in inputs['instances']
            for b in config.get('solvers',('ours','fekete')) for r in range(config['repetitions'])}
        rows=load_rows(output/'raw.jsonl',expected)
        write_reports(output,inputs,config,rows,'complete' if len(rows)==len(expected) else 'partial')
        return 0
    defaults={'per_stratum':1,'repetitions':3,'seconds':3,'external_timeout':12}
    if args.profile:
        defaults.update({'per_stratum':1,'repetitions':1,'seconds':3,'external_timeout':10}
            if args.profile=='quick' else {'per_stratum':8,'repetitions':2,'seconds':60,'external_timeout':75})
        if args.seed is None: args.seed=20260930
        if not args.cycle_optimization:
            args.cycle_optimization=['cache','features','root','interval']
            if args.portfolio or args.portfolio_no_sharing: args.cycle_optimization.append('memo')
    for key,value in defaults.items():
        if getattr(args,key) is None:
            if args.all and key=='per_stratum': continue
            setattr(args,key,value)
    if args.inputs and args.instances_zip: parser.error('use either --inputs or --instances-zip')
    if args.all and not args.instances_zip: parser.error('--all requires --instances-zip')
    if args.all and args.profile: parser.error('--all cannot be combined with --profile')
    if args.all and args.per_stratum is not None: parser.error('use either --all or --per-stratum')
    if args.search_strategy and (args.portfolio or args.portfolio_no_sharing):
        parser.error('--search-strategy cannot be combined with portfolio options')
    limits=(args.repetitions,args.seconds,args.relative_gap,args.feasibility_tolerance,args.validation_tolerance,args.external_timeout)
    if not args.all: limits=limits+(args.per_stratum,)
    if not all(math.isfinite(x) and x>0 for x in limits):
        parser.error('counts, limits and reporting tolerances must be positive')
    continuing=args.resume and (output/'config.json').exists()
    if not continuing and any((output/name).exists() for name in ('raw.jsonl','config.json')):
        parser.error('choose a new output directory or pass --resume with the same options')
    source=args.fekete_source or ROOT/'third_party/tspn-socg'
    if not (source/'tspn_core/CMakeLists.txt').exists():
        common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=ROOT,text=True).strip())
        source=common.parent/'third_party/tspn-socg'
    if args.profile and not (args.instances_zip or args.inputs):
        args.instances_zip=source/'instances/instances_socg_simplified.zip'
    if args.instances_zip:
        if args.all:
            selected,stratum_counts=select_all_socg_inputs(args.instances_zip.resolve())
            selection_policy='all archive cases'
        else:
            selected,stratum_counts=select_socg_inputs(args.instances_zip.resolve(),args.per_stratum,args.seed)
            selection_policy='seeded uniform sample without replacement, interleaved strata' if args.seed is not None else 'lexicographically first'
        inputs={'formulation':'TSPN, free cyclic order, no fixed point, closed polygon regions',
            'selection':{'archive':str(args.instances_zip.resolve()),
                'archive_sha256':hashlib.sha256(args.instances_zip.read_bytes()).hexdigest(),
                'policy':selection_policy,
                'seed':args.seed,'per_stratum':args.per_stratum,'stratum_counts':stratum_counts},'instances':selected}
    else:
        inputs=json.loads(args.inputs.read_text()) if args.inputs else default_inputs()
    if not inputs['instances'] or len({c['name'] for c in inputs['instances']})!=len(inputs['instances']):
        parser.error('inputs must contain nonempty, uniquely named instances')
    enabled_backends=('ours','fekete') if args.solver=='both' else (args.solver,)
    if args.solver!='both' and args.reference_results:
        parser.error('--reference-results requires --solver both')
    planned_runs=len(inputs['instances'])*args.repetitions*len(enabled_backends)
    plan={'profile':args.profile,'cases':len(inputs['instances']),'repetitions':args.repetitions,
        'planned_runs':planned_runs,'solver_seconds_per_run':args.seconds,
        'process_timeout_seconds':args.external_timeout,
        'maximum_process_hours':planned_runs*args.external_timeout/3600,
        'note':'sequential runs; ceiling excludes build, validation and reporting',
        'stratum_population':inputs.get('selection',{}).get('stratum_counts'),
        'cycle_optimizations':args.cycle_optimization}
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run: return 0
    run_options={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(args).items()
        if k not in ('output','resume','dry_run','report_only','skip_build','solver')}
    if continuing:
        previous=json.loads((output/'config.json').read_text())
        if previous.get('run_options')!=run_options:
            parser.error('--resume options differ; repeat the original command with --resume')
        if previous.get('inputs_sha256')!=digest(inputs) or digest(json.loads((output/'instances.json').read_text()))!=digest(inputs):
            parser.error('--resume input manifest changed')
        if previous.get('solvers',('ours','fekete'))!=list(enabled_backends):
            parser.error('--resume solver selection changed')
    output.mkdir(parents=True,exist_ok=True)
    fekete_flag='ON' if 'fekete' in enabled_backends else 'OFF'
    command=['cmake','-S',str(ROOT/'benchmarks/_internal/tspn_native'),'-B',str(args.build_dir),f'-DFEKETE_SOURCE={source}','-DTARGET=main-unordered',f'-DWITH_TSPN_FEKETE={fekete_flag}']
    if os.environ.get('GUROBI_HOME'):
        command.append(f'-DGUROBI_HOME={os.environ["GUROBI_HOME"]}')
    if os.environ.get('TPP_CXX_STANDARD'):
        command.append(f'-DTPP_CXX_STANDARD={os.environ["TPP_CXX_STANDARD"]}')
    # Reuse a locally installed header-only dependency when its Conan package
    # lacks a CMake config. Never write an environment or build into the vendor.
    headers=sorted((Path.home()/'.conan2/p').glob('*/p/include/nlohmann/json.hpp'))
    if headers:command.append(f'-DNLOHMANN_INCLUDE_DIR={headers[0].parents[1]}')
    build_targets=[]
    if 'fekete' in enabled_backends: build_targets.append('tpp-fekete-cycle')
    if 'ours' in enabled_backends: build_targets.append('tpp-unordered')
    commands=[command,['cmake','--build',str(args.build_dir),'--target',*build_targets,'-j','4']]
    if not args.skip_build and not continuing:
        with (output/'build.txt').open('w') as log:
            for cmd in commands:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    elif args.skip_build and not all((args.ours_binary if b=='ours' else args.fekete_binary) for b in enabled_backends):
        parser.error('--skip-build requires --ours-binary and/or --fekete-binary for the enabled solvers')
    if not continuing: atomic_json(output/'instances.json',inputs)
    ours=(args.ours_binary or (args.build_dir/'touring_polygons/tpp-unordered')).resolve()
    fekete=(args.fekete_binary or (args.build_dir/'tpp-fekete-cycle')).resolve() if 'fekete' in enabled_backends else None
    runtime=(output/'runtime').resolve()
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
    config={'run_options':run_options,'inputs_sha256':digest(inputs),'plan':plan,'schema_version':'tspn_campaign_v2','formulation':inputs['formulation'],'repetitions':args.repetitions,'seconds':args.seconds,
        'external_process_timeout_seconds':args.external_timeout,'max_calls':10**8,
        'solvers':list(enabled_backends),
        'oracle_profile':{'scope':'completed B&B search/refinement requests including memo hits; excludes initial polishing and in-flight calls',
            'histogram_upper_seconds':[1e-5,1e-4,1e-3,1e-2,1e-1,1.0,None],
            'fallback_time':'whole calls using fallback, not exclusive recovery time',
            'cycle_phases':'exclusive construction, certification including checks during recovery, and rational recovery excluding certification; completed cooperative interruptions included'},
        'fekete_relative_gap':args.relative_gap,'our_relative_gap':our_relative,'our_absolute_gap':0,
        'feasibility_tolerance':args.feasibility_tolerance,'validation_tolerance':args.validation_tolerance,
        'threads':2 if portfolio else 1,'ours_threads':2 if portfolio else 1,'fekete_threads':1,
        'portfolio_mode':'independent-race' if args.portfolio_no_sharing else ('cooperative' if args.portfolio else None),
        'search_strategy':args.search_strategy or ('default' if not portfolio else None),
        'cycle_optimizations':args.cycle_optimization,
        'capture_oracles':args.capture_oracles,
        'external_backend':'socp','external_root':'LongestEdgePlusFurthestSite',
        'external_search':'DfsBfs','external_branching':'FarthestPoly','external_rules':[],
        'external_node_simplification':False,'external_decomposition_branch':True,'external_cutoff':True,
        'external_gurobi_settings':{'Threads':1,'Presolve':0,'SimplexPricing':3,**gurobi_parameters},
        'external_spanning_tolerance':spanning,'socp_defaults':args.socp_defaults,
        'timing':'whole native solve incl preprocessing/heuristic/root, B&B, extraction and portfolio cooperative join tail; excludes process startup and warmed Gurobi environment; backend order alternates by case; sequential runs',
        'platform':platform.platform(),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        'commands':commands if not args.skip_build else [],'frozen_binaries':{b:str(ours if b=='ours' else fekete) for b in enabled_backends},
        'binary_sha256':{b:hashlib.sha256((ours if b=='ours' else fekete).read_bytes()).hexdigest() for b in enabled_backends},
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
    if continuing:
        if previous.get('binary_sha256')!=config['binary_sha256']:
            parser.error('--resume executable changed; use a new output directory')
        if previous.get('fekete_reference_results')!=config.get('fekete_reference_results'):
            parser.error('--resume reference results changed')
        config=previous
    else:
        atomic_json(output/'config.json',config)
    expected={(c['name'],digest(c['polygons']),b,r) for c in inputs['instances']
        for b in enabled_backends for r in range(args.repetitions)}
    try:
        rows=load_rows(output/'raw.jsonl',expected) if continuing else []
    except (ValueError,KeyError) as error:
        parser.error(str(error))
    completed={row_key(r) for r in rows}
    status='running'
    write_reports(output,inputs,config,rows,status)
    try:
        with (output/'raw.jsonl').open('a') as raw:
            # Repeat rounds, not all repetitions of one case first: even partial
            # campaigns cover the strata and both backends before repeating.
            for repeat in range(args.repetitions):
                for index,case in enumerate(inputs['instances']):
                    polygons=case['polygons'];case_digest=digest(polygons)
                    backends=enabled_backends if (index+repeat)%2==0 else tuple(reversed(enabled_backends))
                    for backend in backends:
                        key=(case['name'],case_digest,backend,repeat)
                        if key in completed: continue
                        began=time.monotonic()
                        capture_path=None
                        try:
                            if backend=='fekete':
                                if reference_rows:
                                    result={k:v for k,v in reference_rows[(case_digest,repeat)].items()
                                        if k not in ('name','k','sha256','solver','repeat','validation','gap_closed')}
                                else:
                                    process=subprocess.run([str(fekete),'1',str(args.relative_gap),str(args.feasibility_tolerance),str(spanning)],
                                        input=encode_instance((0,0),(0,0),polygons,10**8,args.seconds),text=True,capture_output=True,
                                        timeout=args.external_timeout,cwd=runtime)
                                    parsed=[]
                                    for line in process.stdout.splitlines():
                                        if not line.startswith('{'):continue
                                        try:parsed.append(json.loads(line))
                                        except json.JSONDecodeError:continue
                                    result=parsed[0] if process.returncode==0 and len(parsed)==1 else {
                                        'error':f'external exit {process.returncode}; complete JSON records: {len(parsed)}'}
                            else:
                                run_arguments=list(arguments)
                                if args.capture_oracles:
                                    capture_dir=output/'oracle-captures';capture_dir.mkdir(exist_ok=True)
                                    capture_path=capture_dir/f'{index:03d}-{repeat}.jsonl'
                                    attempt=1
                                    while capture_path.exists():
                                        capture_path=capture_dir/f'{index:03d}-{repeat}-attempt{attempt}.jsonl'
                                        attempt+=1
                                    run_arguments+=['--oracle-capture',str(capture_path)]
                                result=run_unordered_solver(ours,(0,0),(0,0),polygons,10**8,args.seconds,run_arguments,
                                    process_timeout=args.external_timeout)
                        except subprocess.TimeoutExpired:
                            # Never persist commercial startup stderr/license data.
                            result={'timeout':True,'error':f'firm process timeout after {args.external_timeout}s',
                                'timeout_seconds':args.external_timeout}
                        except Exception as error:
                            result={'error':f'{backend} runner failed: {type(error).__name__}'}
                        if capture_path is not None:
                            result['oracle_capture_file']=str(capture_path.relative_to(output))
                        if not (backend=='fekete' and reference_rows):
                            result['process_seconds']=time.monotonic()-began
                        row={**result,'name':case['name'],'k':len(polygons),'sha256':case_digest,'solver':backend,'repeat':repeat,
                            'source_class':case.get('source_class'),'size_band':case.get('size_band'),
                            'vertex_count':case.get('vertex_count',sum(map(len,polygons)))}
                        validation_began=time.monotonic()
                        row['validation']=validate_cycle(polygons,row.get('path',[]),args.validation_tolerance) if row.get('path') else {'valid':False,'status':'no completed tour'}
                        row['validation_seconds']=time.monotonic()-validation_began
                        lo,up=row.get('lower_bound'),row.get('upper_bound')
                        row['gap_closed']=finite(lo) and finite(up) and 0<=lo<=up and up <= (1+args.relative_gap)*lo
                        rows.append(row);raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
                        completed.add(key)
                        atomic_json(output/'progress.json',{'status':'running','completed_runs':len(rows),
                            'planned_runs':planned_runs,'process_timeouts':sum(bool(r.get('timeout')) for r in rows)})
                        outcome='TIMEOUT' if row.get('timeout') else ('ERROR' if 'error' in row else
                            f"{row.get('seconds',0):.4f}s gap_closed={row['gap_closed']}")
                        print(f"{len(rows)}/{planned_runs} {case['name']} {backend} repeat={repeat}: {outcome}",flush=True)
                    if (index+1)%12==0: write_reports(output,inputs,config,rows,status)
                write_reports(output,inputs,config,rows,status)
        status='complete'
    except KeyboardInterrupt:
        status='interrupted'
        print('Interrupted; completed records saved. Repeat the command with --resume.',flush=True)
    finally:
        write_reports(output,inputs,config,rows,status if status!='running' else 'failed')
    (output/'README.md').write_text('Resume with the same original tspn-benchmark command plus --resume. '
        'Regenerate reports with --report-only --output THIS_DIRECTORY. '
        'Share instances.json, config.json, raw.jsonl and the reports. No commercial logs are saved.\n')
    if status=='interrupted':return 130
    # A firm timeout is censored evidence, not a runner failure. Backend errors
    # and completed invalid tours are failures in either solver.
    return int(any(not r.get('timeout') and (not r['validation']['valid'] or 'error' in r) for r in rows))
