#!/usr/bin/env python3
"""Guarded single seed204 HandNet comparison, never a batch.

--run requires explicit one-extra scope, published-source review and matching
operational monitor acknowledgment, bound to a private runtime config. --check
is read-only. No test split, upload, original-ledger update or feature deletion.
"""
import argparse
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
RUN_ID = 'comparison-200m-256-seed204'
REVISION = '27ae728e9cb012d2255337d9b3583bbe7f66b3cc'
BASE_SHA = '2186edcfb90843e842fa802c03144e5c17d421d14ddfa2e90cccd9a2f7b18fe4'
BASE_USED = 9012.027812974062
RUNTIME_SHA = 'c9afa326e83f1588fa3ad61985121f5bedee7ac07abee11222e6d1a614a959eb'
COORD_SHA = 'd30587695928cb350c274e6d7d91a472e1b6c27d8ffcfa54c6d76e43a81bf03c'
BINDINGS = {
 'PLAN.private.json':'91fc99f67c39f84ee1f0a10076ecee03c96de1a0a15c3f5b604cdba922c29713',
 'PLAN.errata.private.json':'d697f53e918ff8de348a79133b79965d1e0b5a224a85d5ee445a60d9e78aec31',
 'COMPARISON_PROTOCOL.prospective.v2.private.json':'ef4b07d7de19444ee3f16c1427a5f37ddaf5711189521bf62a1168223bc0eaa6',
 'ORIGINAL_RESULTS_REPORTING.addendum.private.json':'1fedfe3b758d067dca8a2fbddb7814b94161625418f98541abe18131181eca89',
 'Coords.co':COORD_SHA,
}
FRAME_SHA = '7deea34cb06ffc2098761fa6086bd76673708ee29a7c715f6355deacc136b190'
VAL_SHA = 'a9ce12651fbc19491ae92e6e4406ba043b5023c1312359da25dcb61350b0243a'
CLASSES = ['thumb','index','middle','ring','pinky','palm','wrist','background']
PROFILE = {'type':'F_CHAR','max_diff':250,'maximum':127,
 'coordinate_rule':'truncate-after-base-plus-offset-over-depth',
 'coordinate_precision':'float32-complete-sum','missing':{'both':0,'first':127,'second':-127}}

class Refused(RuntimeError): pass

def require(ok, message):
 if not ok: raise Refused(message)

def read(path):
 return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()

def publish(path, value, replace=False):
 """Atomic immutable publication by default; replace only our durable status/budget."""
 path=Path(path);temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with temp.open('x') as f:
  json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 try:
  if replace: os.replace(temp,path)
  else: os.link(temp,path);temp.unlink()
 except BaseException:
  # The temporary is evidence, never an excuse to overwrite an existing result.
  raise

def strict_number(v): return type(v) in (int,float) and math.isfinite(v)

def counts(value, maximum=300):
 require(isinstance(value,list) and len(value)==5 and all(type(v) is int and 0<=v<=maximum for v in value),'five bounded nonnegative integer counts required')
 return value

def stop_seconds(value):
 parts=re.findall(r'(\d+(?:\.\d+)?)\s*(us|ms|min|s|h)',value)
 require(parts and not re.sub(r'(\d+(?:\.\d+)?)\s*(us|ms|min|s|h)','',value).strip(),'unrecognized bounded service stop timeout')
 return sum(float(n)*{'us':.000001,'ms':.001,'s':1,'min':60,'h':3600}[unit] for n,unit in parts)

def authorization(auth, review, monitor, worker_sha, runtime_config_sha, now=None):
 """Pure fail-closed gate. Called BEFORE locks, directories, logs or commands."""
 now=time.time() if now is None else now
 require(isinstance(auth,dict) and auth.get('schema')=='tea-comparison-authorization-v1','authorization missing/invalid')
 require(auth.get('status')=='approved' and auth.get('scope')=='one_extra','explicit one_extra authorization required; batch unsupported')
 require(auth.get('requested_runs')==[{'run_id':RUN_ID,'seed':204}],'only seed204 single configuration supported')
 require(auth.get('existing_original_slots')==['final-tree-0','final-tree-1','final-tree-2'],'original slots must remain counted')
 require(type(auth.get('max_total_final_slots')) is int and auth['max_total_final_slots']==4,'explicit total-four slot reconciliation required')
 require(auth.get('additional_pilots')==0 and auth.get('reserved_additional_ids')==[],'extra pilots/reservations unsupported')
 require(auth.get('max_aggregate_active_seconds')==172800,'original48h ceiling cannot expand')
 cap=auth.get('max_active_seconds_per_new_run')
 require(strict_number(cap) and 0<cap<=57600,'per-run ceiling must be positive and <=16h')
 require(isinstance(auth.get('user_instruction_reference'),str) and auth['user_instruction_reference'].strip(),'record user instruction reference')
 try: approved=dt.datetime.fromisoformat(auth['approved_at'])
 except (KeyError,TypeError,ValueError): raise Refused('approved_at with timezone required')
 require(approved.tzinfo is not None and approved.timestamp()<=now+60,'invalid/future approval timestamp')
 require(isinstance(review,dict) and review.get('source_is_published') is True and review.get('checks_passed') is True,'published-source/hash review is required; candidate is not deployed')
 revision=review.get('published_revision','')
 require(re.fullmatch('[0-9a-f]{40}',revision) is not None and review.get('worker_sha256')==worker_sha,'review must bind exact published worker bytes')
 require(str(review.get('published_source_url','')).startswith('https://github.com/twerdster/tea/blob/'+revision+'/'),'review must identify public source revision')
 require(bool(review.get('root_review_reference')),'root review reference required')
 require(re.fullmatch('[0-9a-f]{64}',runtime_config_sha or '') is not None and review.get('runtime_config_sha256')==runtime_config_sha,'source review must bind the private runtime configuration')
 require(isinstance(monitor,dict) and monitor.get('integration_review_passed') is True and monitor.get('aggregates_original_and_residual_once') is True and monitor.get('tracks_live_guard_once') is True and bool(monitor.get('root_review_reference')),'operational monitor integration review receipt required')
 require(isinstance(monitor,dict) and monitor.get('run_id')==RUN_ID and monitor.get('worker_sha256')==worker_sha,'matching run monitor acknowledgment required')
 require(monitor.get('runtime_config_sha256')==runtime_config_sha,'monitor must bind the private runtime configuration')
 require(monitor.get('status')=='ACTIVE' and monitor.get('rrule')=='FREQ=HOURLY;INTERVAL=2' and monitor.get('covers_run') is True,'active two-hour monitor must cover this exact run')
 require(strict_number(monitor.get('verified_unix')) and 0<=now-monitor['verified_unix']<=10800,'fresh monitor acknowledgment required')
 return float(cap)

def runtime_config(document):
 """Validate private deployment settings without filesystem reads or writes."""
 keys={'schema','project_root','training_repo','proposal_dir','hostname','username','python_executable','gpu_lock','service_name','worker_path','monitor'}
 require(isinstance(document,dict) and set(document)==keys and document.get('schema')=='tea-handnet-comparison-runtime-v1','strict private runtime configuration required')
 for key in keys-{'schema','monitor'}:require(isinstance(document[key],str) and document[key].strip()==document[key] and bool(document[key]) and '\x00' not in document[key],'invalid runtime field: '+key)
 require(isinstance(document['monitor'],dict),'operational monitor configuration object required')
 for key in ('project_root','training_repo','proposal_dir','python_executable','gpu_lock','worker_path'):
  path=Path(document[key]);require(path.is_absolute() and '..' not in path.parts and str(path)==document[key],'normalized absolute runtime path required: '+key)
 root=Path(document['project_root']);require(root!=Path('/'),'project root cannot be filesystem root')
 for key in ('training_repo','proposal_dir'):require(Path(document[key]).is_relative_to(root) and Path(document[key])!=root,'runtime input must stay inside project root: '+key)
 require(document['training_repo']!=document['proposal_dir'],'training repo and proposal must differ')
 require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*\.service',document['service_name']) is not None,'invalid own service name')
 return dict(document)

@dataclass(frozen=True)
class Contract:
 samples:int=200216884
 features:int=256
 frames:int=181979
 classes:int=8
 decisions:int=20
 quota:int=169
 coords_sha:str=COORD_SHA
 min_free:int=83699082736
 conversion_timeout:int=21600
 evaluation_timeout:int=21600

@dataclass(frozen=True)
class Layout:
 root:Path
 proposal:Path
 training_repo:Path=None
 @property
 def repo(self): return self.training_repo if self.training_repo is not None else self.root/'repo'
 @property
 def original(self): return self.root/'runs/current'
 @property
 def control(self): return self.root/'runs'/RUN_ID
 @property
 def bank(self): return self.root/'data'/RUN_ID
 @property
 def retained(self): return self.root/'models'/RUN_ID
 @property
 def residual(self): return self.control/'RESIDUAL_BUDGET.json'
 @property
 def old_frames(self): return [self.root/f'models/final/tree{i}/frames.json' for i in range(3)]
 @property
 def validation(self): return self.original/'validation-selection/frames.json'


def identity(frame):
 require(frame.get('frame_id')=='handnet:gidx:'+str(frame.get('gidx')),'raw global identity required; no filename fallback')
 require(type(frame.get('gidx')) is int and isinstance(frame.get('name'),str) and isinstance(frame.get('session'),str) and bool(frame['session']) and type(frame.get('idx')) is int,'invalid source identity/session/index')
 return tuple(frame.get(k) for k in ('frame_id','name','gidx','session','idx'))

def frames(path, split):
 doc=read(path);require(doc.get('schema')=='handnet-frames-v1' and doc.get('split')==split,'invalid frame manifest split/schema')
 items=doc['frames'];ids=[identity(f) for f in items]
 require(len({x[0] for x in ids})==len(ids) and len({x[1] for x in ids})==len(ids),'duplicate source frame IDs/names')
 return items

def finite(value):
 if isinstance(value,float): require(math.isfinite(value),'nonfinite evaluation output')
 elif isinstance(value,dict):
  for v in value.values():finite(v)
 elif isinstance(value,list):
  for v in value:finite(v)

def score(result):
 local=result['localization'];arrays=[]
 for a in [local['sensor_pos_recall_at_mm']['15']['hits'],local['sensor_pos_present'],local['sensor_pos_detected'],local['false_positive_on_absent']]:
  arrays.append(counts(a))
 hits,present,detected,fp=arrays;values=[]
 for h,p,d,f in zip(hits,present,detected,fp):
  require(h<=d<=p,'hits exceed scorable present/detected counts')
  values.append(2*h/(p+d+f) if p+d+f else 0.)
 return sum(values)/5

def evaluation_counts(result, number):
 """Check aggregate cardinalities against every returned per-frame prediction."""
 local=result['localization'];keys=('present','detected','absent','false_positive_on_absent','sensor_pos_present','sensor_pos_detected')
 values={k:counts(local[k],number) for k in keys};predicted=[0]*5
 for frame in result['frames']:
  fingers=frame.get('fingertips');require(isinstance(fingers,list) and len(fingers)==5,'five per-frame fingertip slots required')
  for i,center in enumerate(fingers):
   if center is None:continue
   require(isinstance(center,dict) and type(center.get('pixels')) is int and center['pixels']>=3,'invalid predicted support')
   for key,size in [('xy',2),('xyz',3)]:require(isinstance(center.get(key),list) and len(center[key])==size and all(strict_number(x) for x in center[key]),'invalid per-frame center geometry')
   require(center['xyz'][2]>0,'nonpositive predicted depth');predicted[i]+=1
 for i in range(5):
  p,d,a,f,s,t=[values[k][i] for k in keys]
  require(p+a==number and t<=s<=p and t<=d<=p and f<=a and predicted[i]==d+f,'inconsistent per-frame/aggregate localization counts')
 for field,thresholds,denominator in [('sensor_pos_recall_at_mm',('10','15','20'),values['sensor_pos_detected']),('label_centroid_recall_at_px',('5','10','15'),values['detected'])]:
  previous=[0]*5
  for threshold in thresholds:
   hits=counts(local[field][threshold]['hits'],number)
   require(all(x<=y<=z for x,y,z in zip(previous,hits,denominator)),'nonmonotonic/impossible localization threshold hits');previous=hits

def reject_other_attempts(layout, known_specs):
 """Incomplete reservations count even before converter metadata exists."""
 for p in (layout.root/'runs').rglob('*.json'):
  if p.name in ('RESERVATION.json','RESIDUAL_BUDGET.json'):
   raise Refused('existing reservation/residual ledger requires count reconciliation: '+str(p))
  if p.name not in ('spec.json','status.json','STATUS.json'):continue
  doc=read(p)
  if p.name=='spec.json' and ('budget_file' in doc or 'checkpoint_dir' in doc):
   require(sha(p) in known_specs,'unknown training spec/attempt requires reconciliation: '+str(p))
  elif p.name.lower()=='status.json' and ('active_training_seconds' in doc or 'spec_sha256' in doc or 'run_id' in doc):
   require(doc.get('spec_sha256') in known_specs,'unknown worker/guard attempt requires reconciliation: '+str(p))
 for p in (layout.root/'data').iterdir():
  if p.is_dir() and (any(p.glob('F_*.feat')) or any(p.glob('Tree_*.tree'))):
   allowed={'final-tree-0','final-tree-1','final-tree-2'}|{f'pilot-{b}scale-{s}' for b in ('','bg-') for s in (50,5000,25000,50000)}
   require(p.name in allowed,'unknown partial feature/model bank requires reconciliation: '+str(p))
 # A crash immediately after mkdir is still an attempt, even with zero files.
 for p in (layout.root/'runs').iterdir():require(not p.is_dir() or p.name=='current','unknown run directory/reservation requires reconciliation: '+str(p))
 allowed_data={'train','validation','test','extraction-check','final-tree-0','final-tree-1','final-tree-2'}|{f'pilot-{b}scale-{s}' for b in ('','bg-') for s in (50,5000,25000,50000)}
 for p in (layout.root/'data').iterdir():require(not p.is_dir() or p.name in allowed_data,'unknown data directory/reservation requires reconciliation: '+str(p))

def select_grid(choices):
 expected={(d,p) for d in (12,16,20) for p in (.25,.5)}
 require(len(choices)==6 and {(x['splits'],x['threshold']) for x in choices}==expected,'all six unique settings required')
 require(all(strict_number(x['score']) and 0<=x['score']<=1 for x in choices),'invalid grid scores')
 return max(choices,key=lambda x:(x['score'],-x['splits'],x['threshold']))

def residual_budget(original, original_sha):
 require(original_sha==BASE_SHA and original.get('used_active_training_seconds')==BASE_USED,'original budget anchor changed')
 runs=original['runs'];require(len(runs)==11 and len({x['spec_sha256'] for x in runs})==11,'original settlement count/identity changed')
 require(all(strict_number(x['seconds']) and x['seconds']>=0 for x in runs) and math.isclose(sum(x['seconds'] for x in runs),BASE_USED,abs_tol=1e-6),'original settlement sum invalid')
 require(original.get('max_active_training_seconds')==172800 and original.get('max_final_trees')==3,'original caps changed')
 return {'schema':'tea-followup-residual-budget-v1','original_budget_sha256':BASE_SHA,'original_used_seconds':BASE_USED,'aggregate_ceiling_seconds':172800,'max_active_training_seconds':172800-BASE_USED,'used_active_training_seconds':0,'runs':[],'max_new_final_trees':1,'reserved_ids':[RUN_ID]}

def accounting(budget, guards=()):
 require(budget.get('original_budget_sha256')==BASE_SHA and budget.get('original_used_seconds')==BASE_USED,'residual anchor mismatch')
 require(budget.get('max_active_training_seconds')==172800-BASE_USED,'residual cap changed')
 seen=set();used=0.
 for r in budget['runs']:
  require(r['spec_sha256'] not in seen and strict_number(r['seconds']) and r['seconds']>=0,'duplicate/invalid residual settlement');seen.add(r['spec_sha256']);used+=r['seconds']
 require(math.isclose(used,budget['used_active_training_seconds'],abs_tol=1e-6),'residual settlement sum mismatch')
 live={}
 for g in guards:
  require(not g.get('accounting_failed'),'guard accounting failure requires reconciliation')
  if g.get('phase')=='training' and g['spec_sha256'] not in seen:
   v=g['active_training_seconds'];require(strict_number(v) and v>=0,'invalid active charge');live[g['spec_sha256']]=max(live.get(g['spec_sha256'],0),v)
 total=BASE_USED+used+sum(live.values());require(total<=172800+1e-6,'aggregate active budget exceeded')
 return {'original_seconds':BASE_USED,'new_settled_seconds':used,'new_unsettled_seconds':sum(live.values()),'aggregate_seconds':total,'remaining_seconds':172800-total}

class Runner:
 """Only owned subprocess groups are cancelled. Guard active time has no wall timeout."""
 def __init__(self,poll_seconds=5): self.poll_seconds=poll_seconds
 def run(self,argv,cwd,log,timeout,tick=lambda p:None):
  started=time.monotonic();environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');environment.pop('DROP_CACHE',None)
  with Path(log).open('x') as out:
   child=subprocess.Popen(argv,cwd=cwd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
   try:
    while True:
     tick(child)
     code=child.poll()
     if code is not None:
      require(code==0,f'owned command exited {code}: {log}');return
     if timeout is not None and time.monotonic()-started>=timeout:raise TimeoutError(f'owned command wall timeout: {log}')
     time.sleep(self.poll_seconds)
   except BaseException:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=35) # Frozen guard needs up to10s to stop Tea, then settle.
     except subprocess.TimeoutExpired:self.forced_termination=True;os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise

class Backend:
 def __init__(self,layout,contract=Contract(),runner=None,configuration=None):self.l=layout;self.c=contract;self.runner=runner or Runner();self.configuration=configuration
 def api(self):
  sys.path.insert(0,str(self.l.repo/'scripts'));return importlib.import_module('tea_model'),importlib.import_module('handnet_to_tea')
 def verify_worker(self,expected):
  require(sha(Path(__file__).resolve())==expected,'deployed worker bytes differ from reviewed receipt');self.worker_sha=expected
  if getattr(self,'runtime_config_path',None):require(sha(self.runtime_config_path)==self.runtime_config_sha,'private runtime configuration changed')
 def verify_frozen(self):
  l=self.l
  self.verify_worker(self.worker_sha)
  require(sha(l.original/'BUDGET.json')==BASE_SHA,'original budget anchor changed')
  for name,digest in BINDINGS.items():require(sha(l.proposal/name)==digest,'frozen proposal binding changed: '+name)
  require(sha(l.old_frames[0])==FRAME_SHA and sha(l.validation)==VAL_SHA,'frozen identities changed')
  require(sha(l.original/'runtime.json')==RUNTIME_SHA,'reviewed frozen source inventory changed')
  runtime=read(l.original/'runtime.json');require(len(runtime['source_sha256'])==29,'source inventory changed')
  for n,h in runtime['source_sha256'].items():require(sha(l.repo/n)==h,'frozen source changed: '+n)
  require(sha(l.repo/'Tea')==runtime['binary_sha256']=='43c9d59e5f09800dea7fa44df5ce8c239bf139a1f17aa1b7b0d165f93277bcbc','binary changed')
 def runtime_info(self):
  import numpy,scipy
  return {'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'coordinate_asset_sha256':sha(self.l.proposal/'Coords.co'),'coordinate_policy':'Copy frozen bytes; never regenerate. Seed alone does not establish cross-version/platform bitwise conversion.'}
 def preflight(self,monitor):
  l=self.l
  cfg=runtime_config(self.configuration)
  require(platform.system()=='Linux' and socket.gethostname()==cfg['hostname'] and __import__('getpass').getuser()==cfg['username'],'configured Linux host/user required')
  require(l.root==Path(cfg['project_root']) and l.repo==Path(cfg['training_repo']) and l.proposal==Path(cfg['proposal_dir']),'runtime paths changed')
  for key in ('project_root','training_repo','proposal_dir','python_executable','gpu_lock','worker_path'):require(Path(cfg[key]).resolve()==Path(cfg[key]),'runtime symlink/alias path refused: '+key)
  require(Path(sys.executable).resolve()==Path(cfg['python_executable']) and Path(__file__).resolve()==Path(cfg['worker_path']),'configured executable/worker identity mismatch')
  self.verify_frozen()
  require(not any(p.exists() for p in [l.control,l.bank,l.retained]),'existing reservation/bank/model: preserve and diagnose, no restart')
  residual_budget(read(l.original/'BUDGET.json'),sha(l.original/'BUDGET.json'))
  reject_other_attempts(l,{r['spec_sha256'] for r in read(l.original/'BUDGET.json')['runs']})
  for name,digest in BINDINGS.items():require(sha(l.proposal/name)==digest,'frozen proposal binding changed: '+name)
  require(sha(l.old_frames[0])==FRAME_SHA and sha(l.validation)==VAL_SHA,'frozen train/validation identities changed')
  runtime=read(l.original/'runtime.json');require(len(runtime['source_sha256'])==29,'source inventory changed')
  for n,h in runtime['source_sha256'].items():require(sha(l.repo/n)==h,'frozen source changed: '+n)
  require(sha(l.repo/'Tea')==runtime['binary_sha256']=='43c9d59e5f09800dea7fa44df5ce8c239bf139a1f17aa1b7b0d165f93277bcbc','binary changed')
  require(read(l.original/'source-revision.json')['tea_revision']==REVISION,'training revision changed')
  campaign=read(l.original/'CAMPAIGN.json');require(campaign.get('completed_at') and len(campaign['final_trees'])==3,'original campaign incomplete')
  # Any other project converter/trainer/evaluator blocks this candidate; never signal it.
  for p in Path('/proc').iterdir():
   if not p.name.isdigit() or int(p.name)==os.getpid():continue
   try:
    argv=(p/'cmdline').read_bytes().split(b'\0')
    require(not any(a==str(l.repo/'Tea').encode() or a.startswith((str(l.repo/'scripts')+'/').encode()) for a in argv),'another project process exists')
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  for i in range(3):
   old=l.root/f'data/final-tree-{i}';g=read(l.original/f'final-tree-{i}/guard/status.json');v=read(l.original/f'final-tree-{i}/model-validation.json');target=l.root/f'models/final/tree{i}'
   require(g.get('exit_code')==0 and v.get('external_predictions_match') is True,'original model not validated')
   require(sha(target/'model.tree')==v['tree_sha256'] and sha(target/'Coords.co')==v['coords_sha256'],'retained original hashes changed')
   require((old/'SCRATCH_COLUMNS_RETIRED.json').is_file() and not any(old.glob('F_*.feat')) and len(read(target/'feature-column-sha256.json'))==1200,'old feature retirement incomplete')
  # Unknown non-pilot training datasets are attempts, not silently ignored free slots.
  allowed={'final-tree-0','final-tree-1','final-tree-2'}|{f'pilot-{b}scale-{s}' for b in ('','bg-') for s in (50,5000,25000,50000)}
  for p in (l.root/'data').glob('*/handnet_metadata.json'):require(p.parent.name in allowed,'another training reservation/attempt requires count reconciliation')
  fs=subprocess.run(['findmnt','-n','-o','FSTYPE','-T',str(l.root)],text=True,capture_output=True,check=True)
  require(fs.stdout.strip() in ('ext4','xfs','btrfs','zfs') and (l.root/'data').stat().st_dev==l.root.stat().st_dev,'native same-filesystem staging required')
  require(__import__('shutil').disk_usage(l.root).free>=self.c.min_free,'insufficient conservative staging headroom')
  require(monitor.get('residual_budget_path')==str(l.residual) and monitor.get('original_budget_path')==str(l.original/'BUDGET.json'),'monitor must aggregate these exact distinct ledgers')
  require(monitor.get('status_path')==str(l.control/'STATUS.json') and monitor.get('guard_status_path')==str(l.control/'guard/status.json') and monitor.get('checkpoint_dir')==str(l.bank),'monitor must cover exact future paths')
  # The worker must run in its own bounded user service; never joins another workload.
  unit=cfg['service_name'];q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','MemoryMax','-p','KillMode','-p','ActiveState','-p','TimeoutStopUSec'],text=True,capture_output=True,check=True)
  unitinfo=dict(line.split('=',1) for line in q.stdout.splitlines() if '=' in line)
  require(unitinfo.get('MainPID')==str(os.getpid()) and unitinfo.get('MemoryMax')==str(24*1024**3) and unitinfo.get('KillMode')=='control-group' and unitinfo.get('ActiveState')=='active' and stop_seconds(unitinfo.get('TimeoutStopUSec',''))>=60,'own24GiB control-group service with >=60s stop timeout required')
 def poll(self,child):
  if getattr(self,'observer',None):self.observer(child)
  require(sha(self.l.original/'BUDGET.json')==BASE_SHA,'original budget changed while worker active')
  self.verify_worker(self.worker_sha)
  require(__import__('shutil').disk_usage(self.l.root).free>=16*1024**3,'disk reserve breached; stop own command')
 def command(self,argv,log,timeout):self.runner.run(argv,self.l.repo,log,timeout,self.poll)
 def converter(self):
  l=self.l;c=self.c
  return [sys.executable,'-u',str(l.repo/'scripts/handnet_to_tea.py'),'--archive',str(l.root/'data/train/TrainData'),'--output',str(l.bank),'--features',str(c.features),'--coords',str(l.proposal/'Coords.co'),'--seed','204','--sampling-seed','204','--frame-seed','201','--frame-manifest',str(l.old_frames[0]),'--sampling','balanced','--samples-per-frame','1352','--sample-alignment','1','--split','train','--max-output-gib','64','--include-background']
 def verify_bank(self):
  import numpy as np
  l=self.l;c=self.c;info=read(l.bank/'handnet_metadata.json');old=frames(l.old_frames[0],'train');new=frames(l.bank/'frames.json','train');validation=frames(l.validation,'validation')
  fixed={'complete':True,'features':c.features,'samples':c.samples,'files_converted':c.frames,'classes':CLASSES,'split':'train','seed':204,'sampling_seed':204,'frame_seed':201,'sampling':'balanced','samples_per_frame':1352,'sample_alignment':1,'include_background':True,'modeled_background_class':7,'invalid_label':8,'feature_type':'F_CHAR','probe_size':None,'class_samples':None}
  for k,v in fixed.items():require(info.get(k)==v,'bank metadata mismatch: '+k)
  require(info.get('raw_label_mapping')=={**{str(i+1):i for i in range(7)},'0':7} and info.get('excluded_raw_labels')==[],'raw label mapping mismatch')
  require(len(new)==len(old)==c.frames and [identity(x) for x in new]==[identity(x) for x in old],'new frame identities/order differ from frozen old bag')
  training_ids=set()
  for p in [*l.old_frames,l.bank/'frames.json']:training_ids.update(x['frame_id'] for x in frames(p,'train'))
  require(not(training_ids & {x['frame_id'] for x in validation}),'training/validation identity overlap')
  expected_hist=np.zeros(8,dtype=np.int64);offset=0
  for a,b in zip(old,new):
   for k in ['shape','source_label_hist','source_positive_depth_label_hist']:require(a[k]==b[k],'source frame content histogram/shape changed')
   hist=b['source_positive_depth_label_hist'];require(len(hist)==8 and all(type(v) is int and v>=0 for v in hist),'invalid source histogram')
   count=sum(min(c.quota,v) for v in hist)
   require(b['sample_begin']==offset and b['samples']==count,'new offsets/quota counts inconsistent');offset+=count
   expected_hist+=np.asarray([min(c.quota,v) for v in hist[1:]]+[min(c.quota,hist[0])])
  require(offset==c.samples,'exact sample total mismatch')
  require(sha(l.bank/'Coords.co')==c.coords_sha and info['coords_sha256']==c.coords_sha,'coordinate bytes changed')
  require((l.bank/'Coords.co').stat().st_size==c.features*16,'coordinate length mismatch')
  for k,v in PROFILE.items():require(info['feature_profile'].get(k)==v,'feature arithmetic contract mismatch')
  paths=[l.bank/f'F_{i:04d}.feat' for i in range(c.features)]
  require(set(l.bank.glob('F_*.feat'))==set(paths),'feature file set mismatch')
  for p in paths:require(p.is_file() and not p.is_symlink() and p.stat().st_size==c.samples,'feature column size/type mismatch')
  for name,size in [('Labels.lbl',2*c.samples),('Pixels.pix',4*c.samples),('Poses.pose',4*c.samples),('Threshholds.thr',12*c.features)]:require((l.bank/name).stat().st_size==size and not(l.bank/name).is_symlink(),'auxiliary size/type mismatch: '+name)
  thresholds=np.fromfile(l.bank/'Threshholds.thr',dtype='<f4').reshape(c.features,3);require(np.isfinite(thresholds).all() and np.all(thresholds[:,0]==127) and np.all(thresholds[:,1:]>=-127) and np.all(thresholds[:,1:]<=127) and np.all(thresholds[:,1]<=thresholds[:,2]),'threshold metadata invalid')
  labels=np.memmap(l.bank/'Labels.lbl',dtype='<u2',mode='r');hist=np.zeros(8,dtype=np.int64)
  poses=np.memmap(l.bank/'Poses.pose',dtype='<u4',mode='r');pixels=np.memmap(l.bank/'Pixels.pix',dtype='<u2',mode='r',shape=(c.samples,2));ends=np.array([f['sample_begin']+f['samples'] for f in new],dtype=np.int64)
  for start in range(0,c.samples,1000000):
   stop=min(c.samples,start+1000000);part=labels[start:stop];require(np.all(part<8),'label range invalid');hist+=np.bincount(part,minlength=8)
   require(np.array_equal(poses[start:stop],np.searchsorted(ends,np.arange(start,stop),side='right')),'pose/sample frame offsets disagree')
   require(np.all(pixels[start:stop,0]<240) and np.all(pixels[start:stop,1]<320),'sample pixel out of bounds')
  require(np.array_equal(hist,expected_hist) and hist.tolist()==[info['label_hist'][str(i)] for i in range(8)],'actual labels/source quota/metadata histogram mismatch')
  self.sample_indices=np.unique(np.linspace(0,c.samples-1,min(1024,c.samples),dtype=np.int64))
  self.raw_feature_check(new,self.sample_indices)
  return {'samples':c.samples,'frames':c.frames,'features':c.features,'raw_sample_checks':len(self.sample_indices),'scope':'Full sizes/labels/poses/bounds;1024 distributed raw sample/feature checks. No claim every feature byte was independently regenerated.'}
 def raw_feature_check(self,new,indices):
  import numpy as np
  import scipy.io as sio
  model,convert=self.api();l=self.l;c=self.c;poses=np.memmap(l.bank/'Poses.pose',dtype='<u4',mode='r');pixels=np.memmap(l.bank/'Pixels.pix',dtype='<u2',mode='r',shape=(c.samples,2));labels=np.memmap(l.bank/'Labels.lbl',dtype='<u2',mode='r');coords=convert.load_coords(l.bank/'Coords.co');columns=[np.memmap(l.bank/f'F_{i:04d}.feat',dtype=np.int8,mode='r') for i in range(c.features)]
  for frame_index in np.unique(poses[indices]):
   f=new[int(frame_index)];selected=indices[poses[indices]==frame_index];mat=sio.loadmat(l.root/'data/train/TrainData'/f['name']);require(identity(convert.frame_identity(mat,f['name'],'train'))==identity(f),'loaded raw training identity mismatch')
   depth=np.asarray(mat['depth'],dtype=np.float32);truth=mat['lbl'];rng=np.random.default_rng(convert.stable_seed(204,f['frame_id']));rr,cc=convert.select_samples(truth,1352,rng,depth,'balanced',None,1,True);local=selected-f['sample_begin']
   require(np.array_equal(np.asarray(pixels[selected]),np.stack([rr[local],cc[local]],axis=1)),'pixel sampler identity changed')
   expected=convert.compute_features(depth,rr[local],cc[local],coords)
   actual=np.stack([col[selected] for col in columns],axis=1);require(np.array_equal(actual,expected),'raw regenerated feature mismatch')
   target=np.where(truth[rr[local],cc[local]]==0,7,truth[rr[local],cc[local]]-1);require(np.array_equal(labels[selected],target),'raw label mapping mismatch')
 def training_spec(self,cap):
  l=self.l;c=self.c
  return {'argv':[str(l.repo/'Tea'),str(c.features),'8','19','12','F_CHAR',str(c.samples),'16','0','0','W_ONES',str(l.bank)+'/',RUN_ID,'0','LOG0',RUN_ID],'cwd':str(l.repo),'output_dir':str(l.control/'guard'),'checkpoint_dir':str(l.bank),'max_seconds':cap,'budget_file':str(l.residual),'gpu_lock':self.configuration['gpu_lock'],'min_free_vram_mb':1536,'max_idle_utilization':30,'binary_sha256':sha(l.repo/'Tea'),'dataset_metadata_sha256':sha(l.bank/'handnet_metadata.json'),'tea_log':str(l.bank/f'LogFile_{RUN_ID}_0000.log')}
 def verify_model(self,spec_path):
  import numpy as np
  model,_=self.api();l=self.l;c=self.c;g=read(l.control/'guard/status.json');require(g.get('exit_code')==0 and g.get('phase')=='training_complete_pending_validation' and not g.get('accounting_failed'),'guard/model not completed successfully')
  require(g['spec_sha256']==sha(spec_path),'guard spec changed')
  budget=read(l.residual);accounting(budget);require(len(budget['runs'])==1 and budget['runs'][0]['spec_sha256']==sha(spec_path),'exact one new guard settlement required')
  require((l.bank/'Tree_0000.tree.labels').stat().st_size==2*c.samples,'saved prediction size mismatch')
  reports=[]
  require(set(l.bank.glob('Tree_0000.depth*.tree'))=={l.bank/f'Tree_0000.depth{d:02d}.tree' for d in range(c.decisions)},'checkpoint set mismatch')
  for d in range(c.decisions):
   p=l.bank/f'Tree_0000.depth{d:02d}.tree';tree=model.TeaTree(p,l.bank/'Coords.co');require(tree.splits==d+1 and tree.classes==8 and len(tree.coords)==c.features,'checkpoint contract mismatch')
   for start in range(0,tree.hist_count,65536):
    heaps=np.arange(start,min(start+65536,tree.hist_count));require(np.all(tree.histograms[heaps].sum(axis=1)>0),'empty repaired checkpoint histogram');probabilities=tree.reached_probabilities(heaps);require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1),1),'checkpoint posterior nonfinite/unnormalized')
   reports.append({'name':p.name,'sha256':sha(p),'decisions':d+1})
  require(sha(l.bank/'Tree_0000.tree')==reports[-1]['sha256'],'final model differs from final checkpoint')
  tree=model.TeaTree(l.bank/'Tree_0000.tree',l.bank/'Coords.co');features=np.stack([np.memmap(l.bank/f'F_{i:04d}.feat',dtype=np.int8,mode='r')[self.sample_indices] for i in range(c.features)],axis=1);predicted=tree.predict_features(features).argmax(axis=1);saved=np.memmap(l.bank/'Tree_0000.tree.labels',dtype='<u2',mode='r');require(np.array_equal(predicted,saved[self.sample_indices]),'external saved prediction traversal mismatch')
  return {'checkpoints':reports,'external_saved_labels_checked':len(self.sample_indices),'external_predictions_match':True,'coords_sha256':sha(l.bank/'Coords.co'),'model_sha256':sha(l.bank/'Tree_0000.tree')}
 def retain(self):
  import shutil
  l=self.l;l.retained.mkdir()
  names={'Tree_0000.tree':'model.tree','Coords.co':'Coords.co','handnet_metadata.json':'training.json','frames.json':'frames.json'};proof={}
  for src,dst in names.items():
   require(not(l.bank/src).is_symlink(),'symlink input refused');before=sha(l.bank/src)
   with (l.bank/src).open('rb') as source,(l.retained/dst).open('xb') as target:shutil.copyfileobj(source,target,8*1024**2)
   require(sha(l.bank/src)==sha(l.retained/dst)==before,'retained copy changed');proof[dst]=before
  publish(l.retained/'RETAINED_FILES.json',proof)
  document={'schema':'depthtrack-tea-forest-v1','image':{'width':320,'height':240,'depth_unit':'millimetres','intrinsics':self.api()[0].DEFAULT_INTRINSICS},'feature':PROFILE,'classes':CLASSES,'background_label':8,'unknown_label':255,'modeled_background_class':7,'aggregation':'mean-normalized-leaf-probabilities','empty_leaf':'root-prior','trees':[{'model':'model.tree','coords':'Coords.co','weight':1,'model_sha256':proof['model.tree'],'coords_sha256':proof['Coords.co']}],'provenance':{'role':'post-exposure-exploratory-seed204-comparison','training_source_revision':REVISION,'seed':204}}
  publish(l.retained/'forest.json',document);return l.retained/'forest.json'
 def evaluate_argv(self,manifest,output,d,p):
  l=self.l;argv=[sys.executable,'-u',str(l.repo/'scripts/tea_model.py'),'evaluate','--manifest',str(manifest),'--archive',str(l.root/'data/validation/ValidationData'),'--output',str(output),'--split','validation','--max-frames','300','--frame-seed','420','--frame-manifest',str(l.validation),'--pixel-limit','0','--max-splits',str(d),'--localize','--foreground','none','--probability-threshold',str(p),'--min-pixels','3']
  for path in [*l.old_frames,l.bank/'frames.json']:argv+=['--training-manifest',str(path)]
  return argv
 def accept_evaluation(self,path,manifest,d,p):
  result=read(path);finite(result);l=self.l
  require(result['schema']=='handnet-tea-evaluation-v1' and result['split']=='validation' and result['manifest_sha256']==sha(manifest),'evaluation source mismatch')
  require(result['foreground']=='none' and result['pixel_limit_per_frame']==0 and result['max_splits']==d and result['classes']==CLASSES,'evaluation domain mismatch')
  require(result['localization']['probability_threshold']==p and result['localization']['minimum_pixels']==3,'localizer mismatch')
  expected=frames(l.validation,'validation');require([identity(x) for x in result['frames']]==[identity(x) for x in expected],'returned raw validation identity/order mismatch')
  evaluation_counts(result,len(expected))
  all_train=set()
  for f in [*l.old_frames,l.bank/'frames.json']:all_train.update(x['frame_id'] for x in frames(f,'train'))
  require(not(all_train & {x['frame_id'] for x in result['frames']}),'evaluation/training leakage')
  return {'splits':d,'threshold':p,'score':score(result),'result':str(path),'result_sha256':sha(path)}
 def export(self,manifest,selected):
  model,_=self.api();doc=read(manifest);doc['postprocessing']={'method':'largest4connected-probability-weighted-centroid','foreground':'none','probability_threshold':selected['threshold'],'min_pixels':3,'fingertip_classes':[0,1,2,3,4]}
  declared=manifest.parent/'forest-selected-source.json';publish(declared,doc)
  output=self.l.retained/'selected';model.export_forest_prefix(declared,output,selected['splits']);final=output/'forest.json'
  return {'path':str(final),'sha256':sha(final),'source_declared_manifest_sha256':sha(declared),'selected_native_parity':'PENDING actual selected Linux/Mac checks; synthetic prefix QA is not full end-to-end validation','publication':'NOT PERFORMED'}

class Worker:
 def __init__(self,backend):self.b=backend;self.l=backend.l;self.status={}
 def update(self,phase,**values):
  self.status.update(phase=phase,updated_unix=time.time(),**values);publish(self.l.control/'STATUS.json',self.status,replace=True)
 def execute(self,auth,review,monitor,worker_sha,runtime_config_sha):
  cap=authorization(auth,review,monitor,worker_sha,runtime_config_sha) # Pure gate first: absolutely no writes/commands.
  self.b.verify_worker(worker_sha)
  self.b.preflight(monitor) # Read-only checks before reservation.
  l=self.l;lockpath=l.root/'runs'/('.'+RUN_ID+'.controller.lock')
  with lockpath.open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.b.preflight(monitor) # Recheck count/path/resource conditions under lock.
   l.control.mkdir() # Durable exclusive reservation; never reused, even after failure.
   self.status={'schema':'tea-seed204-worker-status-v1','run_id':RUN_ID,'pid':os.getpid(),'started_unix':time.time(),'worker_sha256':worker_sha,'runtime_config_sha256':runtime_config_sha,'authorization':auth,'source_review':review,'monitor_ack':monitor,'scratch_retained':True,'scope':'one_extra only; post-exposure exploratory; no TEST/release/automatic retirement'}
   def observe(child):
    ticks=Path(f'/proc/{child.pid}/stat');self.update(self.status['phase'],owned_child={'pid':child.pid,'start_ticks':ticks.read_text().split()[21] if ticks.exists() else None,'argv':child.args})
   self.b.observer=observe
   try:
    publish(l.control/'RESERVATION.json',self.status);self.update('reserved')
    self.status.update(runtime=self.b.runtime_info(),hostname=socket.gethostname(),argv=sys.argv,service=self.b.configuration['service_name'],start_ticks=Path('/proc/self/stat').read_text().split()[21] if Path('/proc/self/stat').exists() else None)
    budget=residual_budget(read(l.original/'BUDGET.json'),sha(l.original/'BUDGET.json'));publish(l.residual,budget)
    self.update('conversion',command=self.b.converter(),conversion_log=str(l.control/'conversion.log'))
    self.b.command(self.b.converter(),l.control/'conversion.log',self.b.c.conversion_timeout)
    self.update('validating_bank');bank=self.b.verify_bank();publish(l.control/'BANK_VALIDATION.json',bank)
    require(sha(l.original/'BUDGET.json')==BASE_SHA,'original budget anchor changed before training')
    self.b.verify_frozen()
    cap=min(cap,accounting(read(l.residual))['remaining_seconds']);spec=self.b.training_spec(cap);require(Path(spec['budget_file'])==l.residual and Path(spec['budget_file'])!=l.original/'BUDGET.json','guard must settle only the residual ledger')
    spec_path=l.control/'spec.json';publish(spec_path,spec);spec_hash=sha(spec_path);self.update('resource_wait_or_guarded_training',spec=str(spec_path),spec_sha256=spec_hash,guard_status=str(l.control/'guard/status.json'),checkpoints=str(l.bank/'Tree_0000.depthDD.tree'))
    # Resource waiting is NOT charged active time and has no equal-to-budget wall timeout.
    self.b.command([sys.executable,'-u',str(l.repo/'scripts/run_guarded_training.py'),str(spec_path)],l.control/'guard-supervisor.log',None)
    self.b.verify_frozen()
    require(sha(spec_path)==spec_hash,'immutable guard spec changed');self.update('validating_model');validated=self.b.verify_model(spec_path);publish(l.control/'MODEL_VALIDATION.json',validated)
    manifest=self.b.retain();choices=[]
    for d in (12,16,20):
     for p in (.25,.5):
      self.b.verify_frozen()
      name=f'validation-d{d}-p{p}';output=l.control/(name+'.json');self.update('validation',setting={'splits':d,'threshold':p},result=str(output))
      self.b.command(self.b.evaluate_argv(manifest,output,d,p),l.control/(name+'.log'),self.b.c.evaluation_timeout)
      choices.append(self.b.accept_evaluation(output,manifest,d,p))
    selected=select_grid(choices);publish(l.control/'VALIDATION_SELECTION.json',{'all_six':choices,'selected':selected,'test_used':False,'comparison_role':'post-exposure exploratory'})
    deployment=self.b.export(manifest,selected);publish(l.control/'DEPLOYMENT_VALIDATION.json',deployment)
    self.update('complete_pending_independent_native_parity_backup_and_report',selected=selected,deployment=deployment,accounting=accounting(read(l.residual)),scratch_retained=True,retirement_allowed=False,backup_monitor_ack='separate verified SSD backup plus monitor acknowledgment required; no deletion implemented')
    return self.status
   except BaseException as error:
    try:charges=accounting(read(l.residual),[read(l.control/'guard/status.json')] if (l.control/'guard/status.json').exists() else [])
    except BaseException as failure:charges={'reconciliation_required':repr(failure)}
    if charges.get('new_unsettled_seconds',0)>0 or getattr(self.b.runner,'forced_termination',False):charges['reconciliation_required']=True
    self.update('failed_preserved_no_retry',error=repr(error),accounting=charges,scratch_retained=True);raise

def production(configuration):
 cfg=runtime_config(configuration);return Backend(Layout(Path(cfg['project_root']),Path(cfg['proposal_dir']),Path(cfg['training_repo'])),configuration=cfg)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--authorization',type=Path);ap.add_argument('--source-review',type=Path);ap.add_argument('--monitor-ack',type=Path);ap.add_argument('--runtime-config',type=Path);mode=ap.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');a=ap.parse_args()
 try:
  # Missing authorization refuses before loading operational context or creating any path.
  require(a.authorization is not None,'explicit authorization file required')
  auth=read(a.authorization);review=read(a.source_review) if a.source_review else None;monitor=read(a.monitor_ack) if a.monitor_ack else None;own=sha(Path(__file__));require(a.runtime_config is not None,'private runtime configuration required');config_sha=sha(a.runtime_config);authorization(auth,review,monitor,own,config_sha);cfg=runtime_config(read(a.runtime_config))
  def interrupted(signum,frame):raise InterruptedError(f'Worker received signal {signum}; stop only owned children and preserve settlement')
  signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
  backend=production(cfg)
  backend.runtime_config_path=a.runtime_config;backend.runtime_config_sha=config_sha
  backend.verify_worker(own)
  require(sha(a.runtime_config)==config_sha,'runtime configuration changed during receipt verification')
  if not a.run:backend.preflight(monitor);print(json.dumps({'mode':'read_only_check','no_actions':True,'runtime_config_sha256':config_sha}));return 0
  result=Worker(backend).execute(auth,review,monitor,own,config_sha);print(json.dumps(result,indent=2));return 0
 except (Refused,ValueError,KeyError,OSError) as error:
  print(json.dumps({'refused_or_failed':repr(error),'no_automatic_retry':True}),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
