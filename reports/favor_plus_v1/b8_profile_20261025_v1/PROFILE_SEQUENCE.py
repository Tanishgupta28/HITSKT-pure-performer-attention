import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/workspace/capstone');out=root/'reports/favor_plus_v1/b8_profile_20261025_v1'
plan=root/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_b8_20261025_v2/prepared_plan.json'
record=out/'PROFILE_EXECUTION_RECORD.json';scope=json.loads(record.read_text());stages=[]
for kind,arm in [('core','orf128'),('core','orf512'),('runtime','orf128'),('runtime','orf512')]:
 remaining=scope['campaign_started_unix']+600-time.time()
 try:
  if remaining<=15:raise TimeoutError('no new worker inside15s shutdown margin')
  script='scripts/run_favor_plus_paired_epochs.py' if kind=='core' else 'scripts/profile_favor_plus_pair_budget.py'
  command=[sys.executable,script,'profile','--plan',str(plan),'--record',str(record),'--arm',arm,'--execute-cuda']
  start=time.perf_counter()
  with (out/f'{arm}_{kind}.log').open('xb') as log:r=subprocess.run(command,cwd=root,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=remaining-15)
  stages.append(dict(stage=kind,arm=arm,wall_seconds=time.perf_counter()-start,returncode=r.returncode))
  (out/'STAGE_TIMINGS.json').write_text(json.dumps(stages,indent=2)+'\n')
  if r.returncode:raise RuntimeError(f'{kind}/{arm} stopped with{returncode}' if False else f'{kind}/{arm} stopped with {r.returncode}')
 except BaseException as exc:
  value=dict(status='STOP_NO_RETRY',stage=kind,arm=arm,type=type(exc).__name__,message=str(exc),optimizer_updates=0,elapsed_seconds=time.time()-scope['campaign_started_unix'])
  (out/'PROFILE_SEQUENCE_STOP.json').write_text(json.dumps(value,indent=2)+'\n')
  if not (plan.parent/'study_failure.json').exists():(plan.parent/'study_failure.json').write_text(json.dumps(value,indent=2)+'\n')
  raise
 print(kind,arm,'PASS_ZERO_UPDATES',flush=True)
(out/'PROFILE_SEQUENCE_COMPLETE.json').write_text(json.dumps(dict(status='ALL_FOUR_PROFILE_STAGES_PASS',optimizer_updates=0,training_approved=False,elapsed_seconds=time.time()-scope['campaign_started_unix']),indent=2)+'\n')
