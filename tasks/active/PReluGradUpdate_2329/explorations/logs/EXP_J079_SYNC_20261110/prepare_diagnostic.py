from pathlib import Path
import json,hashlib,shutil
r=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J079');old=r.parent/'J078'
m=json.loads((old/'perf/manifest.json').read_text());ids=['37','97','110','107','5']
cases=[next(c for c in m['cases'] if c['id']==i) for i in ids]
for c in cases:
 for n,h in c['hashes'].items():assert hashlib.sha256((Path(c['input_root'])/n).read_bytes()).hexdigest()==h
rows=[x for i in ids for x in m['rows'] if x['case']==i]
m.update(cases=cases,rows=rows,source='J079 bounded diagnostic: J078 BF16 slow small/splitC, BF16 Gather control, matching FP32/FP16 small inputs; not performance promotion')
(r/'diagnostic-manifest.json').write_text(json.dumps(m,indent=2))
def line(x,it=None,chk=None):
 return ' '.join(map(str,[x['label'],x['round'],x['input_root'],x['dtype'],','.join(map(str,x['shape'])),x['wlen'],x['iterations'] if it is None else it,x['check_every'] if chk is None else chk]))
(r/'normal-plan.txt').write_text('\n'.join(line(x) for x in rows)+'\n')
debug=[x for x in rows if x['round']==1 and x['label']!='A2']
(r/'trace-plan.txt').write_text('\n'.join(line(x,1,1) for x in debug)+'\n')
print('DIAGNOSTIC_PLAN',len(cases),len(rows),sum(x['iterations'] for x in rows),len(debug))
