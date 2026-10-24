"""Temporary CPU exact-operator substitutions; no training/validation/test scoring."""
import json,time,hashlib
from pathlib import Path
import torch
from ktbench.frozen_exact_attention_reference import FrozenExactAttention
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from scripts.investigate_favor_plus_epoch8 import RUN,PARENT,DES,setup,guard,check_plan,make_model,batch

start=time.time();setup();guard(start);check_plan();plan=json.loads((PARENT/'prepared_plan.json').read_text());train=RollingSessionDataset(SessionStore(plan['store_path']),'train',history_sessions=15);records=[]
for state in ('7','8'):
 model=make_model(state,plan['contract']);before=digest(model.state_dict())
 modules={name:module for name,module in model.named_modules() if name in DES['stages']}
 for case,item in enumerate(DES['training_cohort']):
  guard(start);b=batch(train,item)
  with torch.no_grad():base=model(b)[b.target_metric_mask].clone();basep=base.sigmoid()
  for condition,names in [('exact_CorrectPadding_only',['correct_padding_encoder.layers.0.attention']),('exact_all_four',list(DES['stages']))]:
   try:
    for name in names:
     parent=model.get_submodule(name.rsplit('.',1)[0]);setattr(parent,'attention',FrozenExactAttention(modules[name]).eval())
    with torch.no_grad():new=model(b)[b.target_metric_mask];newp=new.sigmoid()
    records.append(dict(state=state,case=case,condition=condition,targets=new.numel(),logit_rmse=float((new-base).square().mean().sqrt()),logit_max_abs=float((new-base).abs().max()),mean_abs_probability_difference=float((newp-basep).abs().mean()),max_abs_probability_difference=float((newp-basep).abs().max()),changed_classifications=int(((basep>=.5)!=(newp>=.5)).sum())))
   finally:
    for name in names:setattr(model.get_submodule(name.rsplit('.',1)[0]),'attention',modules[name])
  if digest(model.state_dict())!=before:raise RuntimeError('temporary reference changed original model state')
 print('full-model sensitivity',state,flush=True)
(RUN/'END_TO_END_SENSITIVITY.json').write_text(json.dumps(dict(records=records,host=guard(start),seconds=time.time()-start,original_state_restored_exact=True,optimizer_calls=0,validation_inference=0,test_evaluations=0,interpretation='Counterfactual operator sensitivity on TRAIN examples, not a separately trained exact-softmax model or measured predictive improvement.'),indent=2,sort_keys=True)+'\n')
