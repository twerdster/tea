"""CPU-only candidate tests. Fake commands are Python helpers in temporary dirs.

The fake pipeline is explicitly NOT real conversion/training/evaluation evidence.
"""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent
WORKER_PATH=HERE.parent/'scripts/run_handnet_comparison.py'
TEST_RUNTIME_SHA='2'*64
spec=importlib.util.spec_from_file_location('handnet_comparison',WORKER_PATH)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def inputs():
 auth={'schema':'tea-comparison-authorization-v1','status':'approved','scope':'one_extra','requested_runs':[{'run_id':m.RUN_ID,'seed':204}],'existing_original_slots':['final-tree-0','final-tree-1','final-tree-2'],'reserved_additional_ids':[],'max_total_final_slots':4,'additional_pilots':0,'max_aggregate_active_seconds':172800,'max_active_seconds_per_new_run':60,'user_instruction_reference':'UNIT TEST ONLY; no operational authorization','approved_at':'2026-01-01T00:00:00+00:00'}
 revision='1'*40;review={'source_is_published':True,'checks_passed':True,'worker_sha256':'unit-test-sha','published_revision':revision,'published_source_url':'https://github.com/twerdster/tea/blob/'+revision+'/scripts/unit-test-only.py','root_review_reference':'UNIT TEST ONLY'}
 monitor={'run_id':m.RUN_ID,'worker_sha256':'unit-test-sha','status':'ACTIVE','rrule':'FREQ=HOURLY;INTERVAL=2','covers_run':True,'integration_review_passed':True,'aggregates_original_and_residual_once':True,'tracks_live_guard_once':True,'root_review_reference':'UNIT TEST ONLY','verified_unix':time.time()}
 review['runtime_config_sha256']=TEST_RUNTIME_SHA;monitor['runtime_config_sha256']=TEST_RUNTIME_SHA
 return auth,review,monitor

def baseline():
 return {'used_active_training_seconds':m.BASE_USED,'max_active_training_seconds':172800,'max_final_trees':3,'runs':[{'spec_sha256':'old'+str(i),'seconds':m.BASE_USED if i==0 else 0} for i in range(11)]}

def frame(gidx,name,offset=0):
 return {'frame_id':'handnet:gidx:'+str(gidx),'gidx':gidx,'name':name,'session':'Session_test','idx':gidx,'split':'train','shape':[2,2],'source_label_hist':[3,1,0,0,0,0,0,0],'source_positive_depth_label_hist':[1,1,0,0,0,0,0,0],'sample_begin':offset,'samples':2}

def manifest(path,items,split):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'schema':'handnet-frames-v1','split':split,'frames':items}))

class FakeBackend(m.Backend):
 """State-machine double only. It never calls a converter, Tea or evaluator."""
 def __init__(self,layout,fail_at=None):super().__init__(layout,runner=m.Runner(.01),configuration={'service_name':'unit-test-only.service','gpu_lock':str(layout.root/'fake-gpu.lock')});self.calls=[];self.fail_at=fail_at
 def preflight(self,monitor):
  m.require(not any(p.exists() for p in [self.l.control,self.l.bank,self.l.retained]),'existing run')
 def verify_worker(self,expected):self.worker_sha=expected # UNIT TEST ONLY; no production gate is bypassed by CLI.
 def verify_frozen(self):m.require(m.sha(self.l.original/'BUDGET.json')==m.BASE_SHA,'base changed')
 def runtime_info(self):return {'fixture_only':True,'no_real_inference':True}
 def command(self,argv,log,timeout):
  self.calls.append((argv,timeout))
  if self.fail_at=='conversion' and len(self.calls)==1:
   return self.runner.run([sys.executable,'-c','import time;time.sleep(2)'],self.l.repo,log,.05)
  if 'handnet_to_tea.py' in ' '.join(argv):
   code='from pathlib import Path;p=Path('+repr(str(self.l.bank))+');p.mkdir();(p/"F_0000.feat").write_bytes(b"retain scratch")'
  elif 'run_guarded_training.py' in ' '.join(argv):
   # Explicit fake guard alone settles the new ledger; controller must not double charge.
   code='''from pathlib import Path
import json,hashlib
p=Path(SPEC);s=json.loads(p.read_text());b=Path(s['budget_file']);v=json.loads(b.read_text());v['used_active_training_seconds']=2;v['runs']=[{'spec_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seconds':2,'phase':'training_complete_pending_validation'}];b.write_text(json.dumps(v));g=Path(s['output_dir']);g.mkdir();(g/'status.json').write_text(json.dumps({'phase':'training_complete_pending_validation','exit_code':0,'spec_sha256':v['runs'][0]['spec_sha256']}))
'''.replace('SPEC',repr(str(self.l.control/'spec.json')))
  else:
   output=Path(argv[argv.index('--output')+1]);d=int(argv[argv.index('--max-splits')+1]);p=float(argv[argv.index('--probability-threshold')+1]);value={'fixture_only':True,'depth':d,'threshold':p}
   code='from pathlib import Path;Path('+repr(str(output))+').write_text('+repr(json.dumps(value))+')'
   if self.fail_at=='grid' and len(self.calls)==5:code='raise RuntimeError("fixture grid failure")'
  return self.runner.run([sys.executable,'-c',code],self.l.repo,log,timeout)
 def verify_bank(self):return {'fixture_only':True,'real_bank_validation':False}
 def training_spec(self,cap):
  return {'argv':['FAKE_TEA_NEVER_EXECUTED'],'cwd':str(self.l.repo),'output_dir':str(self.l.control/'guard'),'checkpoint_dir':str(self.l.bank),'max_seconds':cap,'budget_file':str(self.l.residual)}
 def verify_model(self,spec):
  m.accounting(m.read(self.l.residual));return {'fixture_only':True,'real_checkpoint_or_saved_label_validation':False}
 def retain(self):
  self.l.retained.mkdir();p=self.l.retained/'forest.json';p.write_text('{}');return p
 def accept_evaluation(self,path,manifest,d,p):
  value=m.read(path);m.require(value['depth']==d and value['threshold']==p,'fake output mismatch')
  return {'splits':d,'threshold':p,'score':.5,'fixture_only':True}
 def export(self,manifest,selected):return {'fixture_only':True,'real_prefix_export':False,'selected_native_parity':'PENDING'}

class ComparisonTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'runs/current').mkdir(parents=True);(self.root/'data').mkdir();(self.root/'models').mkdir();(self.root/'repo').mkdir();self.layout=m.Layout(self.root,self.root/'proposal');(self.layout.original/'BUDGET.json').write_text(json.dumps(baseline()));self.anchor=patch.object(m,'BASE_SHA',m.sha(self.layout.original/'BUDGET.json'));self.anchor.start()
 def tearDown(self):self.anchor.stop();self.tmp.cleanup()
 def test_missing_auth_refuses_before_any_write_or_command(self):
  b=FakeBackend(self.layout);before=set(self.root.rglob('*'))
  for auth in [None,{},dict(inputs()[0],status='pending_scope')]:
   with self.assertRaises(m.Refused):m.Worker(b).execute(auth,None,None,'unit-test-sha',TEST_RUNTIME_SHA)
  self.assertEqual(set(self.root.rglob('*')),before);self.assertEqual(b.calls,[])
 def test_batch_and_cap_expansion_refused(self):
  for field,value in [('scope','new_three_tree_batch'),('max_total_final_slots',6),('max_aggregate_active_seconds',172801),('max_active_seconds_per_new_run',57601),('user_instruction_reference','')]:
   a,r,n=inputs();a[field]=value
   with self.assertRaises(m.Refused):m.authorization(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
 def test_published_source_and_monitor_binding_required(self):
  a,r,n=inputs();r['source_is_published']=False
  with self.assertRaises(m.Refused):m.authorization(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
 def test_private_runtime_config_receipt_binding(self):
  for receipt in ('source','monitor'):
   a,r,n=inputs();(r if receipt=='source' else n)['runtime_config_sha256']='3'*64
   with self.assertRaisesRegex(m.Refused,'runtime configuration'):m.authorization(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
 def test_runtime_config_strict_paths_service_and_no_input_mutation(self):
  cfg={'schema':'tea-handnet-comparison-runtime-v1','project_root':str(self.root),'training_repo':str(self.root/'repo'),'proposal_dir':str(self.root/'proposal'),'hostname':'unit-test-host','username':'unit-test-user','python_executable':str(Path(sys.executable).resolve()),'gpu_lock':str(self.root/'fake-gpu.lock'),'service_name':'unit-test-only.service','worker_path':str(WORKER_PATH.resolve()),'monitor':{}}
  before=copy.deepcopy(cfg);self.assertEqual(m.runtime_config(cfg),cfg);self.assertEqual(cfg,before)
  for field,value in [('project_root','/'),('training_repo','relative'),('proposal_dir',str(self.root/'..'/'outside')),('service_name','../other.service'),('monitor',None)]:
   bad=copy.deepcopy(cfg);bad[field]=value
   with self.assertRaises(m.Refused):m.runtime_config(bad)
  bad=dict(cfg,unexpected_private_field='ignored fields must fail')
  with self.assertRaises(m.Refused):m.runtime_config(bad)
  a,r,n=inputs();n['verified_unix']=time.time()-10801
  with self.assertRaises(m.Refused):m.authorization(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
 def test_residual_is_new_only_and_counts_live_once(self):
  b=m.residual_budget(baseline(),m.BASE_SHA);self.assertEqual(b['runs'],[]);self.assertEqual(b['used_active_training_seconds'],0)
  live={'phase':'training','spec_sha256':'new','active_training_seconds':3};self.assertEqual(m.accounting(b,[live,live])['aggregate_seconds'],m.BASE_USED+3)
  b['runs']=[{'spec_sha256':'new','seconds':3}];b['used_active_training_seconds']=3;self.assertEqual(m.accounting(b,[live])['aggregate_seconds'],m.BASE_USED+3)
 def test_accounting_corruption_and_failure_refuse(self):
  b=m.residual_budget(baseline(),m.BASE_SHA);b['used_active_training_seconds']=1
  with self.assertRaises(m.Refused):m.accounting(b)
  with self.assertRaises(m.Refused):m.accounting(m.residual_budget(baseline(),m.BASE_SHA),[{'accounting_failed':True}])
 def test_successful_fake_pipeline_all_six_and_scratch_retained(self):
  a,r,n=inputs();b=FakeBackend(self.layout);result=m.Worker(b).execute(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
  self.assertEqual(result['phase'],'complete_pending_independent_native_parity_backup_and_report');self.assertFalse(result['retirement_allowed']);self.assertTrue((self.layout.bank/'F_0000.feat').exists());self.assertEqual(len(b.calls),8);self.assertIsNone(b.calls[1][1]);self.assertEqual(m.read(self.layout.residual)['used_active_training_seconds'],2);self.assertEqual(result['selected']['splits'],12);self.assertEqual(result['selected']['threshold'],.5)
  self.assertEqual(m.sha(self.layout.original/'BUDGET.json'),m.BASE_SHA)
  with self.assertRaises(m.Refused):m.Worker(FakeBackend(self.layout)).execute(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
 def test_conversion_timeout_preserves_reservation_and_no_gpu_call(self):
  a,r,n=inputs();b=FakeBackend(self.layout,'conversion')
  with self.assertRaises(TimeoutError):m.Worker(b).execute(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
  self.assertEqual(len(b.calls),1);self.assertEqual(m.read(self.layout.control/'STATUS.json')['phase'],'failed_preserved_no_retry');self.assertTrue((self.layout.control/'RESERVATION.json').exists());self.assertEqual(m.read(self.layout.residual)['runs'],[])
 def test_failed_grid_never_selects_partial_winner(self):
  a,r,n=inputs();b=FakeBackend(self.layout,'grid')
  with self.assertRaises(m.Refused):m.Worker(b).execute(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
  self.assertFalse((self.layout.control/'VALIDATION_SELECTION.json').exists());self.assertTrue((self.layout.bank/'F_0000.feat').exists());self.assertEqual(m.read(self.layout.residual)['used_active_training_seconds'],2)
 def test_grid_completeness_duplicates_and_tie_rule(self):
  values=[{'splits':d,'threshold':p,'score':.4} for d in [12,16,20] for p in [.25,.5]]
  self.assertEqual(m.select_grid(values),{'splits':12,'threshold':.5,'score':.4})
  for broken in [values[:-1],values[:-1]+[values[0]]]:
   with self.assertRaises(m.Refused):m.select_grid(broken)
 def test_score_uses_scorable_detected_and_rejects_bad_counts(self):
  result={'localization':{'sensor_pos_recall_at_mm':{'15':{'hits':[1]*5}},'sensor_pos_present':[2]*5,'sensor_pos_detected':[1]*5,'detected':[200]*5,'false_positive_on_absent':[1]*5}}
  self.assertEqual(m.score(result),.5)
  result['localization']['sensor_pos_detected']=[0]*5
  with self.assertRaises(m.Refused):m.score(result)
 def test_score_integer_bounds_and_five_finger_zero_denominators(self):
  value={'localization':{'sensor_pos_recall_at_mm':{'15':{'hits':[1,0,0,0,0]}},'sensor_pos_present':[1,0,0,0,0],'sensor_pos_detected':[1,0,0,0,0],'false_positive_on_absent':[0]*5}}
  self.assertEqual(m.score(value),.2)
  for broken in [1.5,True,301,-1]:
   bad=copy.deepcopy(value);bad['localization']['sensor_pos_present'][0]=broken
   with self.assertRaises(m.Refused):m.score(bad)
 def test_unknown_reservation_partial_bank_and_unsettled_spec_count(self):
  path=self.root/'runs/current/failed-other';path.mkdir();p=path/'RESERVATION.json';p.write_text('{}')
  with self.assertRaisesRegex(m.Refused,'reservation'):m.reject_other_attempts(self.layout,set())
  p.unlink();p=path/'spec.json';p.write_text(json.dumps({'budget_file':'other','argv':['never executed']}))
  with self.assertRaisesRegex(m.Refused,'unknown training'):m.reject_other_attempts(self.layout,set())
  m.reject_other_attempts(self.layout,{m.sha(p)});p.unlink();bank=self.root/'data/unknown-failed';bank.mkdir();(bank/'F_0000.feat').write_bytes(b'partial')
  with self.assertRaisesRegex(m.Refused,'partial'):m.reject_other_attempts(self.layout,set())
 def test_empty_run_or_data_directory_still_consumes_attempt(self):
  p=self.root/'runs/empty-other';p.mkdir()
  with self.assertRaisesRegex(m.Refused,'unknown run directory'):m.reject_other_attempts(self.layout,set())
  p.rmdir();p=self.root/'data/empty-other';p.mkdir()
  with self.assertRaisesRegex(m.Refused,'unknown data directory'):m.reject_other_attempts(self.layout,set())
 def test_runtime_import_failure_has_reservation_and_failed_status(self):
  a,r,n=inputs();b=FakeBackend(self.layout)
  with patch.object(b,'runtime_info',side_effect=ImportError('UNIT TEST missing runtime')):
   with self.assertRaises(ImportError):m.Worker(b).execute(a,r,n,'unit-test-sha',TEST_RUNTIME_SHA)
  self.assertTrue((self.layout.control/'RESERVATION.json').is_file());self.assertEqual(m.read(self.layout.control/'STATUS.json')['phase'],'failed_preserved_no_retry');self.assertEqual(b.calls,[])
 def test_stop_timeout_units_and_deployed_byte_binding(self):
  self.assertEqual(m.stop_seconds('1min 30s'),90);self.assertEqual(m.stop_seconds('60000000us'),60)
  for value in ['infinity','60','60s junk']:
   with self.assertRaises(m.Refused):m.stop_seconds(value)
  with self.assertRaisesRegex(m.Refused,'deployed worker'):m.Backend(self.layout).verify_worker('unit-test-sha')
 def test_fake_separate_session_guard_cleans_child_and_settles_on_timeout(self):
  # Exercises Runner cancellation only; not frozen guard/GPU/systemd validation.
  script=self.root/'fake_guard.py';pidfile=self.root/'child.pid';settled=self.root/'FAKE_SETTLED'
  script.write_text('''import os,signal,subprocess,sys,time
from pathlib import Path
def interrupted(signum,frame): raise InterruptedError('UNIT TEST ONLY')
signal.signal(signal.SIGTERM,interrupted)
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
Path(sys.argv[1]).write_text(str(p.pid))
try:
 while True: time.sleep(.01)
finally:
 os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=2)
 Path(sys.argv[2]).write_text('FAKE UNIT TEST settlement only')
''')
  with self.assertRaises(TimeoutError):m.Runner(.01).run([sys.executable,str(script),str(pidfile),str(settled)],self.root,self.root/'fake-guard.log',.5)
  self.assertTrue(settled.exists());pid=int(pidfile.read_text())
  with self.assertRaises(ProcessLookupError):os.kill(pid,0)
 def test_fixed_command_contract_and_no_test_or_batch(self):
  b=m.Backend(self.layout);argv=b.converter();self.assertEqual(argv[argv.index('--coords')+1],str(self.layout.proposal/'Coords.co'));self.assertEqual(argv[argv.index('--features')+1],'256');self.assertEqual(argv[argv.index('--frame-manifest')+1],str(self.layout.old_frames[0]));self.assertIn('--include-background',argv)
  ev=b.evaluate_argv(Path('forest.json'),Path('result.json'),16,.5);self.assertEqual(ev.count('--training-manifest'),4);self.assertEqual(ev[ev.index('--min-pixels')+1],'3');self.assertEqual(ev[ev.index('--pixel-limit')+1],'0');self.assertNotIn('test',ev);self.assertNotIn('TestData',' '.join(ev))
 def test_raw_identity_gate_rejects_duplicate_fallback_and_order(self):
  a=frame(1,'a.mat');b=frame(2,'b.mat');p=self.root/'ids.json';manifest(p,[a,a],'train')
  with self.assertRaises(m.Refused):m.frames(p,'train')
  a['frame_id']='train:a.mat';manifest(p,[a,b],'train')
  with self.assertRaises(m.Refused):m.frames(p,'train')
 def test_immutable_publication_never_clobbers(self):
  p=self.root/'evidence.json';m.publish(p,{'old':True})
  with self.assertRaises(FileExistsError):m.publish(p,{'new':True})
  self.assertEqual(m.read(p),{'old':True})
 def test_cli_absent_auth_no_operational_paths_created(self):
  before=set(HERE.iterdir());r=subprocess.run([sys.executable,str(WORKER_PATH),'--run'],text=True,capture_output=True)
  self.assertEqual(r.returncode,2);self.assertIn('explicit authorization',r.stderr);self.assertEqual(set(HERE.iterdir()),before)


class MiniBackend(m.Backend):
 def api(self):
  source=os.environ.get('TEA_TEST_SCRIPTS')
  if source is None:
   source=str(HERE.parent/'scripts')
  sys.path.insert(0,source)
  return __import__('tea_model'),__import__('handnet_to_tea')

class ArtifactValidatorTests(unittest.TestCase):
 """Real validators on tiny synthetic MAT/bank/tree artifacts; no trainer invocation."""
 def setUp(self):
  import numpy as np
  import scipy.io as sio
  self.np=np;self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.l=m.Layout(self.root,self.root/'proposal');self.l.bank.mkdir(parents=True);self.l.control.mkdir(parents=True);self.l.proposal.mkdir();(self.root/'data/train/TrainData').mkdir(parents=True)
  self.b=MiniBackend(self.l,configuration={'gpu_lock':str(self.root/'fake-gpu.lock'),'service_name':'unit-test-only.service'});model,converter=self.b.api();self.model=model;self.convert=converter
  coords=np.array([[0,0],[0,500],[0,0],[0,0]],dtype='<f4');coords.T.tofile(self.l.bank/'Coords.co');self.b.c=replace(m.Contract(),samples=400,features=2,frames=2,decisions=2,coords_sha=m.sha(self.l.bank/'Coords.co'))
  self.old=[];self.new=[];labels=[];pixels=[];poses=[];values=[]
  for i in range(2):
   truth=np.zeros((20,20),dtype=np.uint8);truth[:5,:]=1;depth=np.zeros((20,20),dtype=np.uint16);depth[:5,:]=400;depth[5:10,:]=450;name=f'Data_{i:07d}.mat';raw={'lbl':truth,'depth':depth,'session':'Session_test','gidx':i+1,'idx':i+1};sio.savemat(self.root/'data/train/TrainData'/name,raw)
   f=frame(i+1,name);f.update(shape=[20,20],source_label_hist=[300,100,0,0,0,0,0,0],source_positive_depth_label_hist=[100,100,0,0,0,0,0,0],sample_begin=128*i,samples=128);self.old.append(f);g=copy.deepcopy(f);g.update(sample_begin=i*200,samples=200);self.new.append(g)
   rr,cc=converter.select_samples(truth,1352,np.random.default_rng(converter.stable_seed(204,g['frame_id'])),depth,'balanced',None,1,True);pixels.append(np.stack([rr,cc],axis=1));poses.extend([i]*200);labels.extend(np.where(truth[rr,cc]==0,7,truth[rr,cc]-1));values.append(converter.compute_features(depth,rr,cc,coords))
  for p in self.l.old_frames:manifest(p,self.old,'train')
  v=frame(99,'validation.mat');v['split']='validation';manifest(self.l.validation,[v],'validation');manifest(self.l.bank/'frames.json',self.new,'train')
  self.values=np.concatenate(values);self.labels=np.asarray(labels,dtype='<u2');self.labels.tofile(self.l.bank/'Labels.lbl');np.concatenate(pixels).astype('<u2').tofile(self.l.bank/'Pixels.pix');np.asarray(poses,dtype='<u4').tofile(self.l.bank/'Poses.pose')
  for i in range(2):self.values[:,i].astype(np.int8).tofile(self.l.bank/f'F_{i:04d}.feat')
  np.asarray([[127,-127,127]]*2,dtype='<f4').tofile(self.l.bank/'Threshholds.thr')
  self.info={'complete':True,'features':2,'samples':400,'files_converted':2,'classes':m.CLASSES,'split':'train','seed':204,'sampling_seed':204,'frame_seed':201,'sampling':'balanced','samples_per_frame':1352,'sample_alignment':1,'include_background':True,'modeled_background_class':7,'invalid_label':8,'feature_type':'F_CHAR','probe_size':None,'class_samples':None,'coords_sha256':self.b.c.coords_sha,'feature_profile':m.PROFILE,'raw_label_mapping':{**{str(i+1):i for i in range(7)},'0':7},'excluded_raw_labels':[],'label_hist':{str(i):int(sum(self.labels==i)) for i in range(8)}};self.metadata=self.l.bank/'handnet_metadata.json';self.metadata.write_text(json.dumps(self.info))
 def tearDown(self):self.tmp.cleanup()
 def test_actual_bank_validator_accepts_changed_offsets_and_null_coords_metadata(self):
  report=self.b.verify_bank();self.assertEqual(report['samples'],400);self.assertEqual(report['raw_sample_checks'],400);self.assertNotEqual(m.sha(self.l.old_frames[0]),m.sha(self.l.bank/'frames.json'))
 def test_bank_rejects_wrong_metadata_offsets_labels_and_feature_bytes(self):
  for mutation in ['metadata','offset','label','feature','threshold']:
   with self.subTest(mutation=mutation):
    if mutation=='metadata':bad=dict(self.info,probe_size=25000);self.metadata.write_text(json.dumps(bad))
    elif mutation=='offset':bad=copy.deepcopy(self.new);bad[1]['sample_begin']=199;manifest(self.l.bank/'frames.json',bad,'train')
    elif mutation=='label':bad=self.labels.copy();bad[0]=8;bad.tofile(self.l.bank/'Labels.lbl')
    elif mutation=='feature':bad=self.values[:,0].astype(self.np.int8).copy();bad[0]=1;bad.tofile(self.l.bank/'F_0000.feat')
    else:self.np.asarray([[float('nan'),-127,127]]*2,dtype='<f4').tofile(self.l.bank/'Threshholds.thr')
    with self.assertRaises(m.Refused):self.b.verify_bank()
    self.metadata.write_text(json.dumps(self.info));manifest(self.l.bank/'frames.json',self.new,'train');self.labels.tofile(self.l.bank/'Labels.lbl');self.values[:,0].astype(self.np.int8).tofile(self.l.bank/'F_0000.feat');self.np.asarray([[127,-127,127]]*2,dtype='<f4').tofile(self.l.bank/'Threshholds.thr')
 def test_actual_model_validator_checks_all_checkpoints_and_distributed_saved_labels(self):
  import struct
  np=self.np;self.b.verify_bank();spec=self.l.control/'spec.json';spec.write_text('{}');specsha=m.sha(spec)
  # Each synthetic histogram votes class0; all histograms populated.
  for d in range(2):
   nodes=(1<<(d+1))-1;hists=2*nodes+1;header=struct.pack('<4i',d,8,nodes,hists);node=np.zeros(nodes,dtype=self.model.NODE_DTYPE);leaves=np.zeros(hists,dtype=self.model.LEAF_DTYPE);leaves['maximum']=10;hist=np.zeros((hists,8),np.uint8);hist[:,0]=255;p=self.l.bank/f'Tree_0000.depth{d:02d}.tree';p.write_bytes(header+node.tobytes()+leaves.tobytes()+hist.tobytes())
  (self.l.bank/'Tree_0000.tree').write_bytes((self.l.bank/'Tree_0000.depth01.tree').read_bytes());np.zeros(400,dtype='<u2').tofile(self.l.bank/'Tree_0000.tree.labels');(self.l.control/'guard').mkdir();(self.l.control/'guard/status.json').write_text(json.dumps({'exit_code':0,'phase':'training_complete_pending_validation','spec_sha256':specsha}))
  budget=m.residual_budget(baseline(),m.BASE_SHA);budget['runs']=[{'spec_sha256':specsha,'seconds':1}];budget['used_active_training_seconds']=1;self.l.residual.write_text(json.dumps(budget));report=self.b.verify_model(spec);self.assertEqual(len(report['checkpoints']),2);self.assertEqual(report['external_saved_labels_checked'],400)
  labels=np.zeros(400,dtype='<u2');labels[-1]=1;labels.tofile(self.l.bank/'Tree_0000.tree.labels')
  with self.assertRaisesRegex(m.Refused,'external saved'):self.b.verify_model(spec)
  np.zeros(400,dtype='<u2').tofile(self.l.bank/'Tree_0000.tree.labels');(self.l.bank/'Tree_0000.depth00.tree').unlink()
  with self.assertRaisesRegex(m.Refused,'checkpoint set'):self.b.verify_model(spec)
 def test_training_spec_is_fixed_and_settles_only_residual_ledger(self):
  self.l.repo.mkdir();(self.l.repo/'Tea').write_bytes(b'UNIT TEST ONLY; never executed')
  spec=self.b.training_spec(42)
  self.assertEqual(spec['argv'],[str(self.l.repo/'Tea'),'2','8','19','12','F_CHAR','400','16','0','0','W_ONES',str(self.l.bank)+'/',m.RUN_ID,'0','LOG0',m.RUN_ID])
  self.assertEqual(spec['budget_file'],str(self.l.residual));self.assertNotEqual(spec['budget_file'],str(self.l.original/'BUDGET.json'));self.assertEqual(spec['max_seconds'],42)
 def test_actual_evaluation_validator_rejects_changed_identity_domain_and_counts(self):
  forest=self.l.bank/'forest.json';forest.write_text('{}');output=self.l.control/'validation.json'
  returned=m.frames(self.l.validation,'validation');returned[0]['fingertips']=[{'xy':[1,1],'xyz':[1,1,400],'pixels':3} for _ in range(5)]
  result={'schema':'handnet-tea-evaluation-v1','split':'validation','manifest_sha256':m.sha(forest),'foreground':'none','pixel_limit_per_frame':0,'max_splits':20,'classes':m.CLASSES,'frames':returned,'localization':{'probability_threshold':.25,'minimum_pixels':3,'present':[1]*5,'detected':[1]*5,'absent':[0]*5,'sensor_pos_recall_at_mm':{str(t):{'hits':[1]*5} for t in [10,15,20]},'label_centroid_recall_at_px':{str(t):{'hits':[1]*5} for t in [5,10,15]},'sensor_pos_present':[1]*5,'sensor_pos_detected':[1]*5,'false_positive_on_absent':[0]*5}}
  output.write_text(json.dumps(result));self.assertEqual(self.b.accept_evaluation(output,forest,20,.25)['score'],1)
  for field,value in [('foreground','near-component'),('max_splits',16),('pixel_limit_per_frame',1)]:
   bad=copy.deepcopy(result);bad[field]=value;output.write_text(json.dumps(bad))
   with self.assertRaisesRegex(m.Refused,'domain mismatch'):self.b.accept_evaluation(output,forest,20,.25)
  bad=copy.deepcopy(result);bad['frames'][0]['gidx']=100;output.write_text(json.dumps(bad))
  with self.assertRaises(m.Refused):self.b.accept_evaluation(output,forest,20,.25)
  bad=copy.deepcopy(result);bad['localization']['sensor_pos_detected'][0]=0;output.write_text(json.dumps(bad))
  with self.assertRaisesRegex(m.Refused,'localization threshold'):self.b.accept_evaluation(output,forest,20,.25)

if __name__=='__main__':unittest.main(verbosity=2)
