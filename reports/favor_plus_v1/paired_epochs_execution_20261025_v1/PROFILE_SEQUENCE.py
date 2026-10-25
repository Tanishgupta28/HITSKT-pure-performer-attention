import json,os,subprocess,sys
from pathlib import Path
root=Path('/workspace/capstone');out=root/'reports/favor_plus_v1/paired_epochs_execution_20261025_v1'
plan=root/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_epochs_20261025_v1/prepared_plan.json'
for arm in ('orf128','orf512'):
 command=[sys.executable,'scripts/run_favor_plus_paired_epochs.py','profile','--plan',str(plan),'--record',str(out/'PROFILE_EXECUTION_RECORD.json'),'--arm',arm,'--execute-cuda']
 with (out/(arm+'_profile.log')).open('xb') as log:r=subprocess.run(command,cwd=root,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  (out/'PROFILE_SEQUENCE_STOP.json').write_text(json.dumps(dict(status='STOP_NO_RETRY',arm=arm,returncode=r.returncode,optimizer_updates=0),indent=2)+'\n');sys.exit(r.returncode)
 print(arm,'PROFILE_PASS_ZERO_UPDATES',flush=True)
(out/'PROFILE_SEQUENCE_COMPLETE.json').write_text(json.dumps(dict(status='BOTH_FULL_SHAPE_PROFILES_PASS',optimizer_updates=0,training_approved=False),indent=2)+'\n')
