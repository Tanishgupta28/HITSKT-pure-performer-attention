"""Isolated hierarchical HiTSKT FAVOR+ research variant; V1 data protocol."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ktbench.models.favor_plus import FavorPlusAttention, FavorPlusConfig
from ktbench.models.hitskt import HiTSKT, HiTSKTConfig
from ktbench.models.performer import CausalLinearAttention

VARIANT = "hitskt_favor_plus_v1"
CHECKPOINT_FORMAT = 2


class HiTSKTFavorPlus(HiTSKT):
    """Preserves learned architecture and initialization; replaces attention only."""

    def __init__(self, config: HiTSKTConfig, favor_config: FavorPlusConfig | None = None) -> None:
        super().__init__(config)
        self.favor_config = favor_config or FavorPlusConfig()
        count = 0
        for stage in (self.action_encoder, self.session_encoder,
                      self.correct_padding_encoder, self.decoder):
            for layer in stage.layers:
                if not isinstance(layer.attention, CausalLinearAttention):
                    raise TypeError("unexpected original attention architecture")
                seed = (self.favor_config.projection_seed + count) % (2**63-1)
                layer.attention = FavorPlusAttention(layer.attention, self.favor_config, seed=seed)
                count += 1

    def checkpoint(self) -> dict:
        return {"format_version": CHECKPOINT_FORMAT, "variant": VARIANT,
                "config": asdict(self.config), "favor_config": asdict(self.favor_config),
                "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "HiTSKTFavorPlus":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != CHECKPOINT_FORMAT or payload.get("variant") != VARIANT:
            raise ValueError("not a hitskt_favor_plus_v1 checkpoint")
        model = cls(HiTSKTConfig(**payload["config"]), FavorPlusConfig(**payload["favor_config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model
