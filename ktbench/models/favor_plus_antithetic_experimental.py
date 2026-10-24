"""Opt-in positive antithetic ORF research model; never changes FAVOR+ v1.

Total feature budget M means M/2 Gaussian ORF directions and their negatives.
This is the positive hyperbolic feature construction from Performer Lemma1.
It targets the same softmax kernel, not the same finite feature estimator.
No training registry/CLI is switched to this experimental model automatically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from ktbench.models.favor_plus import FavorPlusAttention, orthogonal_gaussian_projection
from ktbench.models.hitskt import HiTSKT, HiTSKTConfig
from ktbench.models.performer import CausalLinearAttention

VARIANT = 'hitskt_favor_plus_antithetic_experimental_v1'
FORMAT = 3  # legacy ELU uses1, original FAVOR+ uses2


@dataclass(frozen=True)
class AntitheticFavorConfig:
    features_total: int = 512
    projection_pool_features: int = 512
    projection_seed: int = 42
    chunk_size: int = 32
    accumulation_dtype: str = 'float32'
    sampling: str = 'gaussian_orthogonal_antithetic_nested_pool_v1'
    redraw: str = 'fixed'
    stabilization: str = 'causal_per_feature_max_v1'

    def __post_init__(self):
        if type(self.features_total) is not int or self.features_total < 2 or self.features_total % 2:
            raise ValueError('positive even TOTAL feature count required')
        if type(self.projection_pool_features) is not int or self.projection_pool_features < self.features_total//2:
            raise ValueError('declared projection pool must contain all independent base directions')
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError('positive chunk size required')
        if type(self.projection_seed) is not int or not 0 <= self.projection_seed < 2**63-1:
            raise ValueError('invalid local projection seed')
        if self.accumulation_dtype not in ('float32','float64'):
            raise ValueError('only FP32/FP64 accumulation supported')
        if (self.sampling,self.redraw,self.stabilization) != ('gaussian_orthogonal_antithetic_nested_pool_v1','fixed','causal_per_feature_max_v1'):
            raise ValueError('unregistered antithetic policy')


def antithetic_projection(heads, features_total, dimension, seed, *, pool_features=512):
    if type(features_total) is not int or features_total < 2 or features_total % 2:
        raise ValueError('positive even TOTAL feature count required')
    if type(pool_features) is not int or pool_features < features_total//2:
        raise ValueError('projection pool too small')
    base = orthogonal_gaussian_projection(heads,pool_features,dimension,seed)[:,:features_total//2]
    return torch.cat((base,-base),dim=1)


class AntitheticFavorAttention(FavorPlusAttention):
    """Reuse validated stable prefix equations; change only the feature draw."""
    def __init__(self, original, config: AntitheticFavorConfig, *, seed: int):
        nn.Module.__init__(self)
        self.width,self.heads,self.head_width = original.width,original.heads,original.head_width
        self.query,self.key,self.value = original.query,original.key,original.value
        self.output,self.dropout = original.output,original.dropout
        self.config = config
        self.register_buffer('projection',antithetic_projection(self.heads,config.features_total,self.head_width,seed,pool_features=config.projection_pool_features).float())


class HiTSKTFavorPlusAntithetic(HiTSKT):
    """Separate checkpoint identity; original learned nonattention architecture."""
    def __init__(self, config: HiTSKTConfig, antithetic_config: AntitheticFavorConfig | None = None):
        super().__init__(config)
        self.antithetic_config = antithetic_config or AntitheticFavorConfig()
        count=0
        for stage in (self.action_encoder,self.session_encoder,self.correct_padding_encoder,self.decoder):
            for layer in stage.layers:
                if not isinstance(layer.attention,CausalLinearAttention):
                    raise TypeError('unexpected legacy attention layout')
                seed=(self.antithetic_config.projection_seed+count)%(2**63-1)
                layer.attention=AntitheticFavorAttention(layer.attention,self.antithetic_config,seed=seed)
                count+=1

    def checkpoint(self):
        return dict(format_version=FORMAT,variant=VARIANT,config=asdict(self.config),
                    antithetic_config=asdict(self.antithetic_config),state_dict=self.state_dict())

    @classmethod
    def from_checkpoint(cls,path: str | Path,*,map_location='cpu'):
        payload=torch.load(path,map_location=map_location,weights_only=True)
        if payload.get('format_version')!=FORMAT or payload.get('variant')!=VARIANT:
            raise ValueError('not an antithetic experimental checkpoint')
        net=cls(HiTSKTConfig(**payload['config']),AntitheticFavorConfig(**payload['antithetic_config']))
        net.load_state_dict(payload['state_dict'],strict=True)
        return net
