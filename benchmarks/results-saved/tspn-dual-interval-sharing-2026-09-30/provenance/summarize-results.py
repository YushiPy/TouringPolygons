#!/usr/bin/env python3
"""Recompute the concise campaign comparisons from saved CLI JSON."""
import glob, json, os, statistics
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS = os.path.join(ROOT, 'runs')
def read(name):
    with open(os.path.join(RUNS, name, 'raw.jsonl')) as f:
        return [json.loads(x) for x in f if x.strip() and json.loads(x).get('solver') == 'ours']
def by_case(name):
    return {r['name']: r for r in read(name)}
def med(xs): return statistics.median(xs) if xs else None
out = {}
for group in ('small6','large6'):
    ctrl=by_case(group+'-cfr-control')
    variants={v:by_case(group+'-cfr-'+v) for v in ('dual-screen','interval')}
    comp={}
    for v, rows in variants.items():
        paired=[ctrl[n]['seconds']/r['seconds'] for n,r in rows.items() if ctrl[n].get('gap_closed') and r.get('gap_closed')]
        comp[v]={
          'valid_rows':sum(bool(r.get('validation',{}).get('valid')) for r in rows.values()),
          'closed_rows':sum(bool(r.get('gap_closed')) for r in rows.values()),
          'time_limit_rows':sum(r.get('termination')=='time_limit' for r in rows.values()),
          'process_timeout_rows':sum(r.get('termination')=='process_timeout' for r in rows.values()),
          'paired_closed_n':len(paired),'paired_closed_wins':sum(x>1 for x in paired),
          'median_paired_speedup':med(paired),
          'cases':{}
        }
        for name,r in rows.items():
            b=ctrl[name]
            comp[v]['cases'][name]={
              'k':r['k'],'control_s':b['seconds'],'variant_s':r['seconds'],
              'control_termination':b.get('termination'),'variant_termination':r.get('termination'),
              'control_closed':b.get('gap_closed'),'variant_closed':r.get('gap_closed'),
              'control_gap_percent':100*b.get('final_relative_gap',0),
              'variant_gap_percent':100*r.get('final_relative_gap',0),
              'control_calls':b.get('calls'),'variant_calls':r.get('calls'),
              'control_nodes':b.get('nodes'),'variant_nodes':r.get('nodes'),
              'paired_closed_speedup':(b['seconds']/r['seconds']) if b.get('gap_closed') and r.get('gap_closed') else None,
              'dual_screen_children':r.get('cycle_dual_screen_children',0),
              'dual_screen_prunes':r.get('cycle_dual_screen_prunes',0),
              'interval_uses':r.get('cycle_certificate_interval_uses',0)
            }
    out[group]=comp
# Three matched repetitions per large6 class-size case. Repeat directories are kept separate so order was interleaved.
reps={
 'control':['large6-cfr-control','large6-cfr-control-repeat2','large6-cfr-control-repeat3'],
 'interval':['large6-cfr-interval','large6-cfr-interval-repeat2','large6-cfr-interval-repeat3']
}
rep_rows={v:[by_case(n) for n in ns] for v,ns in reps.items()}
repeat_summary={}
case_names=sorted(rep_rows['control'][0])
all_speedups=[]
for name in case_names:
    cr=[d[name] for d in rep_rows['control']]; ir=[d[name] for d in rep_rows['interval']]
    speedups=[c['seconds']/i['seconds'] for c,i in zip(cr,ir) if c.get('gap_closed') and i.get('gap_closed')]
    all_speedups += speedups
    c_gaps=[100*r.get('final_relative_gap',0) for r in cr]
    i_gaps=[100*r.get('final_relative_gap',0) for r in ir]
    repeat_summary[name]={
      'k':cr[0]['k'], 'control_seconds': [r['seconds'] for r in cr], 'interval_seconds':[r['seconds'] for r in ir],
      'control_median_seconds':med([r['seconds'] for r in cr]),'interval_median_seconds':med([r['seconds'] for r in ir]),
      'paired_closed_speedups':speedups,'closed_pairs':len(speedups),
      'control_gap_percent':c_gaps,'control_gap_median_percent':med(c_gaps),'control_gap_range_percent':[min(c_gaps),max(c_gaps)],
      'interval_gap_percent':i_gaps,'interval_gap_median_percent':med(i_gaps),'interval_gap_range_percent':[min(i_gaps),max(i_gaps)],
      'control_all_time_limit':all(r.get('termination')=='time_limit' for r in cr),
      'interval_all_time_limit':all(r.get('termination')=='time_limit' for r in ir)
    }
out['large6_interval_three_repetitions']={'cases':repeat_summary,'closed_pairs':len(all_speedups),'closed_wins':sum(x>1 for x in all_speedups),'median_paired_closed_speedup':med(all_speedups)}
# Same-binary V1 and V2 portfolio screens; no aggregate speedup metric because only one repetition and three time limits.
portfolio=[]
for version, control, treatment in [
 ('v1','portfolio-large6-cfr-memo-control','portfolio-large6-cfr-memo-share-bounds'),
 ('v2','v2-portfolio-large6-cfr-memo-control','v2-portfolio-large6-cfr-memo-share-bounds')]:
    c,t=by_case(control),by_case(treatment)
    counters={k:sum(r.get(k,0) for r in t.values()) for k in ('cycle_shared_bound_queries','cycle_shared_bound_hits','cycle_shared_bound_improvements','cycle_shared_bound_prunes')}
    cases={}
    for name,r in t.items():
      b=c[name]
      cases[name]={'k':r['k'],'control_s':b['seconds'],'share_bounds_s':r['seconds'],'control_closed':b.get('gap_closed'),'share_bounds_closed':r.get('gap_closed'),'control_termination':b.get('termination'),'share_bounds_termination':r.get('termination'),'control_gap_percent':100*b.get('final_relative_gap',0),'share_bounds_gap_percent':100*r.get('final_relative_gap',0),'control_calls':b.get('calls'),'share_bounds_calls':r.get('calls'),'control_nodes':b.get('nodes'),'share_bounds_nodes':r.get('nodes')}
    portfolio.append({'binary_version':version,'control_valid':sum(bool(r.get('validation',{}).get('valid')) for r in c.values()),'treatment_valid':sum(bool(r.get('validation',{}).get('valid')) for r in t.values()),'control_closed':sum(bool(r.get('gap_closed')) for r in c.values()),'treatment_closed':sum(bool(r.get('gap_closed')) for r in t.values()),'control_time_limits':sum(r.get('termination')=='time_limit' for r in c.values()),'treatment_time_limits':sum(r.get('termination')=='time_limit' for r in t.values()),'share_counters':counters,'cases':cases})
out['portfolio_share_bounds_screens']=portfolio
# OSM39, one three-rep paired portfolio comparison, Fekete references were reused.
def osm(name): return read(name)
c=osm('portfolio-osm39-cfr-memo-control-repeat3'); t=osm('portfolio-osm39-cfr-memo-interval-repeat3')
c=sorted(c,key=lambda r:r['repeat']); t=sorted(t,key=lambda r:r['repeat'])
out['portfolio_osm39_interval_repeat3']={
 'control_seconds':[r['seconds'] for r in c], 'interval_seconds':[r['seconds'] for r in t],
 'control_median_seconds':med([r['seconds'] for r in c]),'interval_median_seconds':med([r['seconds'] for r in t]),
 'paired_speedups':[a['seconds']/b['seconds'] for a,b in zip(c,t)],
 'median_paired_speedup':med([a['seconds']/b['seconds'] for a,b in zip(c,t)]),
 'control_valid':sum(bool(r.get('validation',{}).get('valid')) for r in c),'interval_valid':sum(bool(r.get('validation',{}).get('valid')) for r in t),
 'control_closed':sum(bool(r.get('gap_closed')) for r in c),'interval_closed':sum(bool(r.get('gap_closed')) for r in t),
 'control_calls':[r.get('calls') for r in c],'interval_calls':[r.get('calls') for r in t],
 'control_nodes':[r.get('nodes') for r in c],'interval_nodes':[r.get('nodes') for r in t],
 'fekete_times_reused':[r['seconds'] for r in sorted([r for r in (json.loads(x) for x in open(os.path.join(RUNS,'portfolio-osm39-cfr-memo-control-repeat3','raw.jsonl'))) if r.get('solver')=='fekete'],key=lambda r:r['repeat'])]
}
# Raw data totals and all run config paths.
runs=sorted(glob.glob(os.path.join(RUNS,'*')))
allrows=[json.loads(line) for d in runs for line in open(os.path.join(d,'raw.jsonl')) if line.strip()]
out['row_totals']={'run_directories':len(runs),'raw_rows_total':len(allrows),'native_rows':sum(r.get('solver')=='ours' for r in allrows),'fekete_reference_rows_reused':sum(r.get('solver')=='fekete' for r in allrows),'native_valid':sum(r.get('solver')=='ours' and bool(r.get('validation',{}).get('valid')) for r in allrows),'native_closed':sum(r.get('solver')=='ours' and bool(r.get('gap_closed')) for r in allrows),'native_time_limit':sum(r.get('solver')=='ours' and r.get('termination')=='time_limit' for r in allrows),'native_process_timeout':sum(r.get('solver')=='ours' and r.get('termination')=='process_timeout' for r in allrows)}
path=os.path.join(os.path.dirname(__file__),'campaign-summary.json')
with open(path,'w') as f: json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(out['row_totals'],indent=2))
print('large6 repeat pairs:',out['large6_interval_three_repetitions']['closed_pairs'],'wins:',out['large6_interval_three_repetitions']['closed_wins'],'median speedup:',out['large6_interval_three_repetitions']['median_paired_closed_speedup'])
print('OSM39 portfolio medians:',out['portfolio_osm39_interval_repeat3']['control_median_seconds'],out['portfolio_osm39_interval_repeat3']['interval_median_seconds'],'median speedup',out['portfolio_osm39_interval_repeat3']['median_paired_speedup'])
print('wrote',path)
