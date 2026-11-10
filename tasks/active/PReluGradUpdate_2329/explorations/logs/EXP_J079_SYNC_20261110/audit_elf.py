from pathlib import Path
import struct,json,hashlib
R=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J079')
P=R.parent/'J078/build-abc/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/j078_abc_custom'
def sh(b):return hashlib.sha256(b).hexdigest()
def elf(p):
 b=p.read_bytes();off=struct.unpack_from('<Q',b,40)[0];size,num,ns=struct.unpack_from('<HHH',b,58);ss=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*size) for i in range(num)];names=b[ss[ns][4]:ss[ns][4]+ss[ns][5]];sections=[];func=[]
 for i,s in enumerate(ss):
  name=names[s[0]:].split(b'\0')[0].decode(errors='replace'); data=b[s[4]:s[4]+s[5]] if s[1]!=8 else b''
  sections.append(dict(index=i,name=name,type=s[1],flags=s[2],addr=s[3],offset=s[4],bytes=s[5],align=s[8],sha256=sh(data)))
  if s[1]!=2:continue
  st=ss[s[6]];strings=b[st[4]:st[4]+st[5]]
  for pos in range(s[4],s[4]+s[5],s[9]):
   n,info,other,idx,val,sz=struct.unpack_from('<IBBHQQ',b,pos)
   if info&15!=2 or not idx or idx>=len(ss):continue
   n=strings[n:].split(b'\0')[0].decode(errors='replace');sec=ss[idx];data=b[sec[4]+val-sec[3]:sec[4]+val-sec[3]+sz]
   func.append(dict(name=n,section=idx,addr=val,bytes=sz,sha256=sh(data)))
 return dict(path=str(p),bytes=len(b),sha256=sh(b),sections=sections,functions=func)
objects=[]
for p in sorted((P/'op_impl/ai_core/tbe/kernel/ascend910_93').rglob('*.o')):
 m=json.loads(p.with_suffix('.json').read_text());a=m['supportInfo']['inputs']
 if a[0]['dtype']!='bfloat16' or any(x['format']!='ND' for x in a):continue
 e=elf(p);e['metadata']=m;objects.append(e)
assert len(objects)==3
(R/'full-elf-bf16.json').write_text(json.dumps(objects,indent=2))
for e in objects:
 print(Path(e['path']).name,e['bytes'],e['sha256'])
 print('sections',[(s['name'],s['bytes'],s['sha256'][:16]) for s in e['sections'] if s['flags']&2])
 print('helper_functions',[(f['name'],f['bytes'],f['sha256'][:16]) for f in e['functions'] if 'PReluGradUpdate' not in f['name']])
def diff(a,b,p=''):
 if type(a)!=type(b):return [(p,a,b)]
 if isinstance(a,dict):return sum((diff(a.get(k),b.get(k),p+'/'+k) for k in sorted(a.keys()|b.keys())),[])
 if isinstance(a,list):
  if len(a)!=len(b):return [(p,'len='+str(len(a)),'len='+str(len(b)))]
  return sum((diff(x,y,p+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
 return [] if a==b else [(p,a,b)]
print('METADATA_AB',json.dumps(diff(objects[0]['metadata'],objects[1]['metadata']),indent=2))
