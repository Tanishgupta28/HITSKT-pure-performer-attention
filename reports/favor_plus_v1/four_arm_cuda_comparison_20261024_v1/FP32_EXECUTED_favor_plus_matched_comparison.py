"""Isolated four-arm study; historical model/checkpoint identities stay intact."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import datetime as dt
import json,os,time
from pathlib import Path
import torch
from torch import nn
from ktbench import favor_plus_shared_gpu as shared
from ktbench.favor_plus_real_pilot import host_memory
from ktbench.favor_plus_pipeline import sha256,write_json
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.models.favor_plus import FavorPlusConfig,FavorPlusAttention,orthogonal_gaussian_projection
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.models.favor_plus_antithetic_experimental import AntitheticFavorConfig,HiTSKTFavorPlusAntithetic

ARMS=('orf128','orf512','anti128','anti512')
VERIFY_ACTION='favor_plus_four_arm_cuda_verification_only_v1'
TRAIN_ACTION='favor_plus_four_arm_matched20_training_v1'
VARIANT='hitskt_favor_plus_matched_four_arm_v1'
FORMAT=4
GIB=1024**3
LIMITS=dict(allocator_bytes=8*GIB,advisory_total_bytes=10*GIB,admission_free_bytes=16*GIB,
            runtime_free_bytes=6*GIB,max_step_seconds=30,max_campaign_seconds=2700,pause_seconds=.1)


@dataclass(frozen=True)
class MatchedArmConfig:
    arm: str
    projection_seed: int=42
    pool_features: int=512
    chunk_size: int=32
    accumulation_dtype: str='float32'
    sampling_policy: str='shared_nested512_pool_all_four_arms_v1'

    def __post_init__(self):
        if self.arm not in ARMS or self.pool_features!=512 or self.sampling_policy!='shared_nested512_pool_all_four_arms_v1':
            raise ValueError('exact predeclared four-arm policy required')
        if type(self.projection_seed) is not int or not 0<=self.projection_seed<2**63-1 or type(self.chunk_size) is not int or self.chunk_size<=0:
            raise ValueError('invalid projection seed/chunk')
        if self.accumulation_dtype not in ('float32','float64'):raise ValueError('unsupported precision')

    @property
    def features(self):return 512 if self.arm.endswith('512') else 128

    @property
    def antithetic(self):return self.arm.startswith('anti')


def arm_projection(heads,dimension,cfg: MatchedArmConfig,stage):
    pool=orthogonal_gaussian_projection(heads,512,dimension,(cfg.projection_seed+stage)%(2**63-1))
    if cfg.antithetic:
        base=pool[:,:cfg.features//2];return torch.cat((base,-base),1).float()
    return pool[:,:cfg.features].float()


class MatchedComparisonModel(nn.Module):
    """Native original/prototype cores, tagged with the declared matched policy."""
    def __init__(self,config:HiTSKTConfig,comparison:MatchedArmConfig):
        super().__init__();self.config=config;self.comparison=comparison
        if comparison.antithetic:
            self.core=HiTSKTFavorPlusAntithetic(config,AntitheticFavorConfig(features_total=comparison.features,
                projection_pool_features=512,projection_seed=comparison.projection_seed,chunk_size=comparison.chunk_size,
                accumulation_dtype=comparison.accumulation_dtype))
        else:
            self.core=HiTSKTFavorPlus(config,FavorPlusConfig(features=comparison.features,
                projection_seed=comparison.projection_seed,chunk_size=comparison.chunk_size,accumulation_dtype=comparison.accumulation_dtype))
        count=0
        for component in (self.core.action_encoder,self.core.session_encoder,self.core.correct_padding_encoder,self.core.decoder):
            for layer in component.layers:
                module=layer.attention
                if not isinstance(module,FavorPlusAttention):raise TypeError('unexpected attention core')
                module.projection=arm_projection(module.heads,module.head_width,comparison,count)
                count+=1
        if count!=4:raise ValueError('study requires original one-layer four-stage architecture')

    def forward(self,batch):return self.core(batch)

    def checkpoint(self):
        return dict(format_version=FORMAT,variant=VARIANT,config=asdict(self.config),
            comparison_config=asdict(self.comparison),state_dict=self.state_dict(),
            notice='fresh matched study; not a historical original128 checkpoint')

    @classmethod
    def from_checkpoint(cls,path,*,map_location='cpu'):
        payload=torch.load(path,map_location=map_location,weights_only=True)
        if payload.get('format_version')!=FORMAT or payload.get('variant')!=VARIANT:
            raise ValueError('not a four-arm matched study checkpoint')
        net=cls(HiTSKTConfig(**payload['config']),MatchedArmConfig(**payload['comparison_config']))
        net.load_state_dict(payload['state_dict'],strict=True);return net


def fresh_model(config,arm):
    # All feature samplers use local generators; learned initialization/dropout
    # streams therefore have equal CPU RNG state across arms after construction.
    torch.random.default_generator.manual_seed(42)
    return MatchedComparisonModel(config,MatchedArmConfig(arm))


def learned_digest(net):return digest({n:p.detach().cpu() for n,p in net.named_parameters()})


def require_scope(record,plan_path,action,*,now=None):
    if (record.get('action')!=action or record.get('resource_uuid')!=shared.TRANSFERRED_MIG
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=shared.TRANSFERRED_MIG
            or record.get('prepared_plan_sha256')!=sha256(plan_path)
            or record.get('shared_permission_confirmed') is not True
            or record.get('known_explicit_restriction') is not False
            or record.get('known_incompatible_reservation') is not False
            or record.get('limits')!=LIMITS):raise ValueError('exact scope/plan/authorized shared budget required')
    clock=now or dt.datetime.now(dt.timezone.utc)
    start=dt.datetime.fromisoformat(record['starts_utc']);end=dt.datetime.fromisoformat(record['expires_utc'])
    if start.tzinfo is None or end.tzinfo is None or not start<=clock<end or not 0<(end-start).total_seconds()<=7200:
        raise ValueError('fresh <=2h operator record required')
    if action==VERIFY_ACTION:
        if record.get('cuda_verification_approved') is not True or record.get('max_optimizer_updates')!=0:
            raise ValueError('verification is zero-optimizer only')
    elif action==TRAIN_ACTION:
        if (record.get('training_approved') is not True or record.get('approval_is_proposal') is not False
                or not isinstance(record.get('actual_training_approval_evidence'),str)
                or not record['actual_training_approval_evidence'].strip()
                or record.get('maximum_updates_per_arm')!=20 or record.get('arms')!=list(ARMS)):
            raise PermissionError('NEW explicit matched-training approval required; old CUDA/full-run scope is insufficient')
    else:raise ValueError('unregistered study action')


class SharedStudyGuard:
    """Reuse shared visibility/pressure/Light checks; enforce approved8GiB cap."""
    def __init__(self,record,plan_path,output,action=VERIFY_ACTION):
        self.record,self.plan_path,self.output,self.action=record,plan_path,output,action
        self.count=0;self.last=None;self.own_cached=None
        require_scope(record,plan_path,action)

    def check(self,phase,*,cuda=True,admission=False):
        require_scope(self.record,self.plan_path,self.action);shared.check_exposure_and_hold(shared.TRANSFERRED_MIG)
        obs=shared.observe(shared.TRANSFERRED_MIG);host=host_memory()
        # Old shared policy's visibility/headroom/host/protected-job checks stay
        # unchanged. Its historical1GiB own cap is not the approved8GiB budget.
        decision=shared.assess(obs,host,admission=admission)
        if obs['device'].get('free_bytes',0)<LIMITS['admission_free_bytes' if admission else 'runtime_free_bytes']:
            decision['reasons'].append('approved four-arm headroom floor');decision['status']='STOP'
        self.last=dict(phase=phase,observation=obs,host=host,decision=decision,cached_own_allocator=self.own_cached)
        self.count+=1;write_json(self.output/f'resources_{self.count:05d}.json',self.last)
        if decision['status']=='STOP':raise shared.SharedResourceStop('; '.join(decision['reasons']),self.last)
        if time.time()-self.record['campaign_started_unix']>2700:raise TimeoutError('unchanged45min campaign ceiling')
        if cuda:
            own=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            self.own_cached=own;self.last['own_allocator']=own;write_json(self.output/f'allocator_{self.count:05d}.json',self.last)
            if max(own.values())>LIMITS['allocator_bytes']:raise MemoryError('8GiB own allocated/reserved/peak cap')
        return self.last

    def admit(self):
        self.check('admission1',cuda=False,admission=True);time.sleep(.2);self.check('admission2',cuda=False,admission=True)


def initialize_cuda(guard):
    guard.admit()  # no CUDA initialization before these required read-only checks
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(8*GIB/total,0);torch.cuda.manual_seed(42)
    guard.check('CUDA_initialized')
