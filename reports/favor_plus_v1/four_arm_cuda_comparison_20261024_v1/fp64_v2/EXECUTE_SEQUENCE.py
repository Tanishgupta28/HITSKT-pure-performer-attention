"""Bounded precision follow-up; no retry or optimizer action. Original clock retained."""
from pathlib import Path
import json,subprocess,sys,os
ROOT=Path('/workspace/capstone');OUT=ROOT/'reports/favor_plus_v1/four_arm_cuda_comparison_20261024_v1/fp64_v2'
for arm in ('orf128','orf512','anti128','anti512'):
 for action in ('verify','reload','compare'):
  command=[sys.executable,'scripts/compare_favor_plus_cuda.py',action,'--arm',arm,'--precision','float64','--execute-cuda']
  with (OUT/f'{arm}_{action}.log').open('xb') as stream:result=subprocess.run(command,cwd=ROOT,env=os.environ,stdout=stream,stderr=subprocess.STDOUT)
  if result.returncode:
   (OUT/'SEQUENCE_STOP.json').write_text(json.dumps(dict(status='STOP_NO_RETRY',arm=arm,action=action,returncode=result.returncode,optimizer_updates=0),indent=2)+'\n');sys.exit(result.returncode)
  print(arm,action,'PASS',flush=True)
(OUT/'VERIFICATION_SEQUENCE_COMPLETE.json').write_text(json.dumps(dict(status='ALL_FOUR_FP64_ARMS_COMPLETED_ZERO_UPDATES',arms=['orf128','orf512','anti128','anti512'],optimizer_updates=0,training_approval='PENDING_EXPLICIT_USER_RESPONSE'),indent=2)+'\n')
