from pathlib import Path
import json,hashlib,zipfile
r=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J079'); files={}
def add(p,n):
 assert p.is_file(); files[n]=p
for p in r.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(r)
 if p.suffix in {'.log','.csv'} or len(rel.parts)==1 and p.suffix in {'.json','.py','.sh','.txt','.exit'} or p.name.startswith('info.json') and not p.name.endswith('.done'):add(p,str(rel))
old=r.parent/'J078'
for p,n in [(old/'control-abc/device_probe','frozen/device_probe'),(old/'abc/control/campaign.py','frozen/campaign.py')]: add(p,n)
for e in json.loads((r/'full-elf-bf16.json').read_text()):
 p=Path(e['path']);add(p,'frozen/bf16/'+p.name);add(p.with_suffix('.json'),'frozen/bf16/'+p.with_suffix('.json').name)
manifest={n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(files.items())}
with zipfile.ZipFile(r/'J079-diagnostic-evidence.zip','w',zipfile.ZIP_DEFLATED,6) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('FILE-SHA256.json',json.dumps(manifest,indent=2))
p=r/'J079-diagnostic-evidence.zip'; meta={'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':len(files),'scope':'all J079 cloud logs and exported CSV, plans, summaries, exact BF16 ELF, frozen runner/analyzer; hardware profiler raw binary/databases stay in J079 cloud profile directories'}
(r/'evidence-archive.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
