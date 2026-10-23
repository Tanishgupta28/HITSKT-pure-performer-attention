"""CPU-only synthetic controller/recovery integration, not ASSIST training."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ktbench import favor_plus_shared_assist50 as p
from ktbench import favor_plus_shared_gpu as s
from ktbench import favor_plus_bounded_assist as bounded
from ktbench.favor_plus_pilot_diagnostics import cpu_tree, equal_tree
from ktbench.metrics import masked_bce_loss
from ktbench.training import capture_rng_state


class Batch:
    def __init__(self,index=0):
        self.x=torch.tensor([.3+index*.001,.7])
        self.target_labels=torch.tensor([0.,1.]);self.target_metric_mask=torch.tensor([True,True])
    def to(self,*args):return self  # deliberate CPU fixture, no CUDA allocation


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.tensor([.1,.2]))
        self.attention=torch.nn.Module();self.attention.register_buffer('projection',torch.tensor([[1.,2.]]))
    def forward(self,batch):return self.w*batch.x
    def checkpoint(self):
        return dict(variant='hitskt_favor_plus_v1',format_version=2,config={'synthetic':True},
                    favor_config={'synthetic':True},state_dict=self.state_dict())


@pytest.fixture
def simulation(tmp_path,monkeypatch):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.manual_seed(42)
    net=Net();opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    def collate(dataset,item):return Batch(item['dataset_index'])
    monkeypatch.setattr(p.original,'collate_selected',collate)
    monkeypatch.setattr(bounded,'collate_selected',collate)
    for index in range(20):
        opt.zero_grad();batch=Batch(index)
        masked_bce_loss(net(batch),batch.target_labels,batch.target_metric_mask).backward();opt.step()
    items=[dict(dataset_index=i,padded_tokens=2) for i in range(51)]
    config=dict(model_config={'synthetic':True},favor_config={'synthetic':True},
        review_and_severe_stops={'severe_stops':dict(single_update_bce_max=20,gradient_l2_max=100)})
    pc=dict(config=config,parameter_tensor_count=1,exact_training_and_probe_samples=items[:21])
    parent=cpu_tree(net.checkpoint())
    parent.update(kind=p.parent_recovery.KIND,contract=pc,optimizer_updates=20,optimizer_step_calls=20,
        optimizer_state_dict=cpu_tree(opt.state_dict()),rng_state=cpu_tree(capture_rng_state()),
        sampler=dict(cursor=20,epoch=1,seed=42,exact_samples=items[:21]),
        reference_eval_logits=p.original.fixed_logits(net,Batch(0)).cpu(),
        next_update_probe=p.original.continuation_probe(net,opt,None,items[20]))
    p.parent_recovery.seal(parent);p.parent_recovery.validate(parent,pc)
    parent_dir=tmp_path/'parent';(parent_dir/'stage_020').mkdir(parents=True)
    parent_file=parent_dir/'checkpoint_020.pt';torch.save(parent,parent_file)
    c=dict(run_id='UNIT_SHARED',config=config,parent_contract=pc,parameter_tensor_count=1,
        parent_component_sha256=parent['component_sha256'],parent_projection_sha256=parent['projection_sha256'],
        parent_checkpoint_sha256=p.PARENT_SHA256,parent_optimizer_param_groups=parent['optimizer_state_dict']['param_groups'],
        exact_training_and_probe_samples=items,diagnostic_samples=[items[0]],frozen_validation_samples=[items[0],items[1]],
        execution_limits=dict(max_execution_seconds=2700,max_step_seconds=30),
        execution_mode=s.MODE,shared_resource_policy=copy.deepcopy(s.POLICY))
    c=json.loads(json.dumps(c))
    vm,prob,labels=p.original.evaluate(net,None,c['frozen_validation_samples'])
    tm,_,_=p.original.evaluate(net,None,c['diagnostic_samples'])
    np.savez(parent_dir/'stage_020/validation_predictions_020.npz',probability=prob.numpy(),label=labels.numpy())
    (parent_dir/'stage_020/diagnostics_result_020.json').write_text(json.dumps(dict(sampled_validation=vm,sampled_training_eval=tm)))
    (tmp_path/'initial_diagnostics.json').write_text('[{}]')
    root=tmp_path/'run';(root/'checkpoints').mkdir(parents=True)
    plan_path=root/'prepared_plan.json';plan_path.write_text('{}')
    plan=dict(contract_base=c)
    monkeypatch.setattr(p,'PARENT_PLAN',parent_dir/'prepared_plan.json')
    monkeypatch.setattr(p,'PARENT_CHECKPOINT',parent_file)
    monkeypatch.setattr(p.original,'FROZEN',tmp_path/'frozen.json')
    def context(path,record,phase,update):
        if (root/'study_failure.json').exists() or (root/'study_result.json').exists():
            raise RuntimeError('failed/completed campaign cannot restart')
        output=root/f'{phase}_{update:03d}';output.mkdir()
        return output,plan,c,None,None
    monkeypatch.setattr(p,'context',context)
    def initialize(plan):
        m=Net();return m,torch.optim.Adam(m.parameters(),lr=5e-5)
    monkeypatch.setattr(p,'initialize',initialize)
    monkeypatch.setattr(p,'runtime',lambda *args:{'CPU_SIMULATION':True})
    monkeypatch.setattr(p,'capture_checked',lambda *args:[{}])
    monkeypatch.setattr(p.original,'severe_check',lambda *args:dict(status='CONTINUE_BOUNDED_DIAGNOSTIC_ONLY',original_softmax_review_trigger=.2))
    monkeypatch.setattr(p.original,'causal_check',lambda *args:{'CPU_SIMULATED':True})
    host=dict(peak_process_rss_bytes=s.GIB,cgroup_headroom_bytes=20*s.GIB)
    monkeypatch.setattr(p.original,'check_host_resources',lambda:host)
    monkeypatch.setattr(p.original,'host_memory',lambda:host)
    monkeypatch.setattr(p,'authorize',lambda *args:s.TRANSFERRED_MIG)
    def observe(uuid):
        return dict(observed_utc=p.original.utc().isoformat(),resource_uuid=uuid,
            device=dict(total_bytes=40*s.GIB,used_bytes=20*s.GIB,free_bytes=20*s.GIB,compute_processes=[{'pid':999}]),
            utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(s,'observe',observe)
    monkeypatch.setattr(p.time,'sleep',lambda *args:None)
    for name in ('synchronize','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:None)
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:1024)
    return root,plan_path,parent_file,observe


def test_cpu_shared20_to50_has_exact_recovery_and_stops_at50(simulation):
    root,plan,parent,observe=simulation
    before=p.sha256(parent)
    for endpoint in (30,40,50):
        stage=p.execute(plan,{'prepared_plan_sha256':'UNIT'},endpoint)
        assert stage['status']=='PASS' and stage['optimizer_updates']==endpoint
        assert stage['additional_optimizer_calls']==endpoint-20
        recovered=p.verify(plan,{'prepared_plan_sha256':'UNIT'},endpoint)
        assert recovered['status']=='PASS' and recovered['optimizer_calls_during_recovery']==0
    assert len(list(root.glob('stage_*/update_*.json')))==30
    assert sorted(x.name for x in (root/'checkpoints').glob('*.pt'))==[
        f'checkpoint_{i:03d}.pt' for i in (25,30,35,40,45,50)]
    result=json.loads((root/'study_result.json').read_text())
    assert result['status']=='FIFTY_TOTAL_UPDATES_COMPLETED_SHARED_GPU'
    assert p.sha256(parent)==before and not torch.cuda.is_initialized()
    with pytest.raises(RuntimeError,match='completed'):p.execute(plan,{'prepared_plan_sha256':'UNIT'},50)


def test_cuda_style_failure_preserves_parent_and_no_silent_step_retry(simulation,monkeypatch):
    root,plan,parent,observe=simulation
    before=p.sha256(parent)
    old=torch.optim.Adam.step;calls=[]
    def fail_after_step(self,*args,**kwargs):
        calls.append(1);old(self,*args,**kwargs)
        raise RuntimeError('UNIT simulated asynchronous CUDA failure after partial optimizer mutation')
    monkeypatch.setattr(torch.optim.Adam,'step',fail_after_step)
    with pytest.raises(RuntimeError,match='simulated'):p.execute(plan,{'prepared_plan_sha256':'UNIT'},30)
    failure=json.loads((root/'study_failure.json').read_text())
    assert len(calls)==1 and failure['optimizer_step_calls']==21 and failure['optimizer_updates']==20
    assert failure['no_automatic_restart'] and failure['no_cuda_calls_in_failure_handler']
    assert not list((root/'checkpoints').glob('*.pt'))
    with pytest.raises(RuntimeError,match='cannot restart'):p.execute(plan,{'prepared_plan_sha256':'UNIT'},30)
    assert len(calls)==1 and p.sha256(parent)==before


def test_checked_evaluation_preserves_exact_metrics_scores_and_rng(monkeypatch):
    net=Net().train();items=[dict(dataset_index=i) for i in range(3)]
    monkeypatch.setattr(bounded,'collate_selected',lambda dataset,item:Batch(item['dataset_index']))
    direct=p.original.evaluate(net,None,items)
    rng=cpu_tree(capture_rng_state());phases=[]
    b=type('Boundary',(),{'check':lambda self,phase:phases.append(phase)})()
    checked=p.evaluate_checked(net,None,items,b,'validation')
    assert direct[0]==checked[0] and torch.equal(direct[1],checked[1]) and torch.equal(direct[2],checked[2])
    assert equal_tree(rng,cpu_tree(capture_rng_state())) and net.training
    assert phases==['validation_sample_0','validation_sample_1','validation_sample_2']
