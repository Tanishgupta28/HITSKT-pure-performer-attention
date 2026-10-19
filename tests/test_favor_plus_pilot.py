"""Pilot bounds and GPU admission fail before allocation; tiny CPU fixture."""
import pytest
from ktbench import favor_plus_pipeline
from scripts.pilot_hitskt_favor_plus import pilot


def test_pilot_refuses_unconfirmed_cuda_before_seed_or_allocation(monkeypatch):
    import scripts.pilot_hitskt_favor_plus as runner
    def forbidden(): raise AssertionError("must not initialize RNG/CUDA")
    monkeypatch.setattr(runner,"seed_everything",forbidden)
    with pytest.raises(ValueError,match="before any CUDA allocation"):
        pilot("junyi","forbidden",device="cuda:0")


def test_pilot_refuses_unbounded_work():
    with pytest.raises(ValueError,match="bounded"):
        pilot("junyi","forbidden",device="cpu",length=13081)
    with pytest.raises(ValueError,match="bounded"):
        pilot("junyi","forbidden",device="cpu",steps=6)


def test_pilot_cpu_fixture_finite_gradients_and_checkpoint(tmp_path,monkeypatch):
    import scripts.pilot_hitskt_favor_plus as runner
    monkeypatch.setattr(runner,"source_identity",lambda:{"synthetic_test_fixture":True})
    monkeypatch.setattr(favor_plus_pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(favor_plus_pipeline,"EXPERIMENT_ROOT",tmp_path/"experiments"/favor_plus_pipeline.VARIANT)
    result = pilot("junyi","tiny_pilot",device="cpu",length=5,steps=1)
    assert result["status"] == "PASS"
    assert result["checkpoint_max_abs_logit_difference"] == 0
    assert result["peak_cuda_allocated_bytes"] is None
