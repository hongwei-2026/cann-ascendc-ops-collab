from pathlib import Path
import json
r=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J079')
m=json.loads((r/'diagnostic-manifest.json').read_text());rows=m['rows'];assert rows[0]['label']=='A1' and rows[1]['label']=='B' and rows[0]['case']==rows[1]['case']=='37'
rows[0],rows[1]=rows[1],rows[0]
m['source']='J079 phase2: only first A1/B windows exchanged to reverse first BF16 binary load order; all binaries, inputs and later windows unchanged'
(r/'swapped-manifest.json').write_text(json.dumps(m,indent=2))
def line(x,it=None,chk=None):
 return ' '.join(map(str,[x['label'],x['round'],x['input_root'],x['dtype'],','.join(map(str,x['shape'])),x['wlen'],x['iterations'] if it is None else it,x['check_every'] if chk is None else chk]))
(r/'swapped-plan.txt').write_text('\n'.join(line(x) for x in rows)+'\n')
debug=[x for x in rows if x['round']==1 and x['label']!='A2']
(r/'trace-swapped-plan.txt').write_text('\n'.join(line(x,1,1) for x in debug)+'\n')
print('ONLY_FIRST_TWO_WINDOWS_SWAPPED')
