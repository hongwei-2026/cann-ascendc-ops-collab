from pathlib import Path
import os,sys,json,subprocess,hashlib,numpy as np,time
r=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J078/validation');tag=sys.argv[1] if len(sys.argv)>1 else 'r1';cases=json.loads((r/'cases.json').read_text());cases=[c for c in cases if not os.environ.get('J070_CASE') or c['id']==os.environ['J070_CASE']];root=r.parent;oppbase='_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors';opps={'alias':root/'build-abc'/oppbase/'j078_abc_custom','standard':root/'candidate/code/build/packages/vendors/p_relu_grad_update_custom'};results=[]
def save():
 p=r/('validation-'+tag+'.json');tmp=p.with_suffix('.json.part');tmp.write_text(json.dumps(dict(tag=tag,results=results,all_proposed=len(cases),accepted=sum(x.get('builtin_accepted',False) for x in results),custom_complete=sum(x.get('complete',False) for x in results)),indent=2));tmp.replace(p)
def call(c,op,prefix,env):
 p=Path(c['input_root'])
 try:
  run=subprocess.run([str(r/'probe_diag'),str(p),op,prefix],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,timeout=120)
 except subprocess.TimeoutExpired as e:
  (p/(prefix+'-runner.log')).write_bytes(e.stdout or b'')
  raise
 (p/(prefix+'-runner.log')).write_bytes(run.stdout)
 return run

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def values(p,dt):
 a=np.fromfile(p,np.float32 if dt=='fp32' else np.float16 if dt=='fp16' else np.uint16);return (a.astype(np.uint32)<<16).view(np.float32) if dt=='bf16' else a.astype(np.float32)
def compare(c,prefix):
 p=Path(c['input_root']);out={}
 for name in ('dx','update'):
  a=values(p/(prefix+'-'+name+'.bin'),c['dtype']);b=values(p/('builtin-'+name+'.bin'),c['dtype']);assert a.shape==b.shape,(c['id'],name,a.shape,b.shape);eq=(a==b)|(np.isnan(a)&np.isnan(b));tol=1e-4 if c['dtype']=='fp32' else 1e-3
  with np.errstate(all='ignore'):ok=eq|(np.isfinite(a)&np.isfinite(b)&(np.abs(a-b)<=tol+tol*np.abs(b)))
  bad=np.flatnonzero(~ok);out[name]=dict(elements=int(a.size),logical_exact_mismatches=int(np.count_nonzero(~eq)),threshold_mismatches=int(bad.size),first_bad=[dict(index=int(i),actual=str(float(a[i])),expected=str(float(b[i]))) for i in bad[:8]])
 return out
builtin_env=os.environ.copy();builtin_env.pop('ASCEND_CUSTOM_OPP_PATH',None)
for c in cases:
 p=Path(c['input_root']);cache=p/'builtin-validation.json';record=dict(id=c['id'],dtype=c['dtype'],format=c['format'],kind=c['kind'])
 if cache.exists():
  v=json.loads(cache.read_text());assert v['input_hashes']==c['input_hashes'] and v['meta_sha256']==c['meta_sha256']
  if v['accepted']:
   for name,digest in v['golden_hashes'].items():assert sha(p/name)==digest
 else:
  run=call(c,'PReluGradUpdate','builtin',builtin_env);v=dict(accepted=run.returncode==0,exit=run.returncode,input_hashes=c['input_hashes'],meta_sha256=c['meta_sha256'],golden_hashes={name:sha(p/name) for name in ('builtin-dx.bin','builtin-update.bin')} if run.returncode==0 else {});cache.write_text(json.dumps(v,indent=2))
 record.update(builtin_accepted=v['accepted'],builtin_exit=v['exit']);results.append(record)
 if not v['accepted']:
  print('BUILTIN_CONTRACT_REJECT',c['id'],v['exit'],flush=True);save();continue
 for flavor,op in [('alias','PReluGradUpdateB'),('standard','PReluGradUpdate')]:
  env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=str(opps[flavor]);prefix=tag+'_'+flavor;run=call(c,op,prefix,env);record[flavor]=dict(exit=run.returncode)
  if run.returncode!=0:save();print('CUSTOM_EXECUTION_FAIL',c['id'],flavor,run.returncode,run.stdout.decode(errors='replace')[-2000:],flush=True);sys.exit(2)
  checks=compare(c,prefix);record[flavor]['checks']=checks;save()
  if any(x['threshold_mismatches'] for x in checks.values()):print('CUSTOM_COMPARE_FAIL',c['id'],flavor,json.dumps(checks),flush=True);sys.exit(3)
 record['complete']=True;save();print('TRUE_TBE_CUSTOM_PASS',len(results),len(cases),c['id'],flush=True)
assert any(x.get('complete') for x in results),'No accepted case completed; investigate contract/harness instead of claiming pass'
summary=dict(proposed=len(cases),accepted=sum(x['builtin_accepted'] for x in results),rejected=[x['id'] for x in results if not x['builtin_accepted']],completed=sum(x.get('complete',False) for x in results),all_accepted_custom_passed=all(x.get('complete',False) for x in results if x['builtin_accepted']),strict_logical_mismatches=sum(v['logical_exact_mismatches'] for x in results if x.get('complete') for f in ('alias','standard') for v in x[f]['checks'].values()),NPU_performance_measured=False)
(r/('summary-'+tag+'.json')).write_text(json.dumps(summary,indent=2));print('J070_TRUE_TBE_VALIDATION_COMPLETE',json.dumps(summary),flush=True)
