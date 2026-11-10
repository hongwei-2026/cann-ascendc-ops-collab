from pathlib import Path
import json,statistics as st,hashlib,re
r=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J079')
names=['original','swapped','restored','taskonly','pipeback','taskonly-swapped','taskonly-return']
result={'runs':[],'status':'diagnostic only; no candidate promotion; all runs retained'}
for n in names:
 s=json.loads((r/(n+'-summary.json')).read_text());ws=s['windows'];cs=s['comparisons']
 p=r/'logs'/('perf-'+n+'.log');txt=p.read_text(errors='replace')
 passed=re.findall(r'^CASE_END \d+ PASS \d+$',txt,re.M)
 assert len(passed)==120,(n,len(passed));assert len(ws)==120 and len(cs)==30
 d={'run':n,'ai_core_counters':not n.startswith('taskonly'),'first_bf16_load_order':'BAC' if 'swapped' in n else 'ABC','windows':len(ws),'calls':sum(x['iterations'] for x in ws),'correctness_windows_pass':len(passed),'controls_pass':sum(x['controls_stable'] for x in cs),'aa_pass':sum(x['aa_stable'] for x in cs),'comparisons':len(cs),'cases':[]}
 for c in ['37','97','110','107','5']:
  xx=[x for x in ws if x['case']==c]
  z={'id':c,'dtype':xx[0]['dtype'],'shape':xx[0]['shape'],'actual_tiling_key':{'37':8,'97':14,'110':2,'107':6,'5':7}[c],'median_task_us':{a:st.median(x['Task Duration(us)']['median'] for x in xx if x['label']==a) for a in ['A1','A2','B','C']}}
  if 'aiv_scalar_time(us)' in xx[0]: z['median_scalar_us']={a:st.median(x['aiv_scalar_time(us)']['median'] for x in xx if x['label']==a) for a in ['A1','A2','B','C']}
  z['controls_pass']=sum(x['controls_stable'] for x in cs if x['case']==c)
  d['cases'].append(z)
 result['runs'].append(d)
result['input_hashes_rechecked']=0
for c in json.loads((r/'diagnostic-manifest.json').read_text())['cases']:
 for f,h in c['hashes'].items():
  assert hashlib.sha256((Path(c['input_root'])/f).read_bytes()).hexdigest()==h
  result['input_hashes_rechecked']+=1
elf=json.loads((r/'full-elf-bf16.json').read_text())
for e in elf:assert hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()==e['sha256']
result['bf16_elf_hashes_rechecked']=len(elf)
result['debug_calls_pass']=sum(len(re.findall(r'^CASE_END \d+ PASS 1$',(r/'logs'/('trace-'+n+'.log')).read_text(errors='replace'),re.M)) for n in ['original','swapped','restored'])
assert result['debug_calls_pass']==45
(r/'diagnostic-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
for x in result['runs']:print(x['run'],x['calls'],x['controls_pass'],x['comparisons'],x['aa_pass'])
