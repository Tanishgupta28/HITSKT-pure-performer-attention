"""Tiny synthetic-store pipeline tests; never read/train production datasets."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from ktbench import favor_plus_pipeline as pipeline
from ktbench.data.session_store import SessionStore, RollingSessionDataset
from ktbench.hitskt_registry import HITSKT_COMMON_CONFIG, HITSKT_TRAINING_CONFIGS
from ktbench.models.favor_plus import FavorPlusConfig


@pytest.fixture
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"EXPERIMENT_ROOT",tmp_path/"experiments"/pipeline.VARIANT)
    monkeypatch.setattr(pipeline,"source_identity",lambda:{"synthetic_test_fixture":True})
    return tmp_path


def synthetic_store(root: Path) -> SessionStore:
    root.mkdir()
    lengths = [2,3,4,2,3]*2
    offsets = np.r_[0,np.cumsum(lengths)]
    arrays = {"question":np.arange(sum(lengths))%9+1,"skill":np.arange(sum(lengths))%4+1,
              "correct":np.arange(sum(lengths))%2,"timestamp_ms":np.arange(sum(lengths))*1000,
              "session_offsets":offsets,"session_student":np.repeat([0,1],5),
              "session_number":np.tile(np.arange(1,6),2),"session_split":np.tile([0,0,0,1,2],2),
              "student_offsets":np.array([0,5,10])}
    for name,value in arrays.items(): np.save(root/f"{name}.npy",value)
    (root/"metadata.json").write_text(json.dumps({"num_questions":9,"num_skills":4,
        "students":2,"sessions":10,"interactions":int(sum(lengths))}))
    (root/"_SUCCESS").write_text("synthetic fixture\n")
    return SessionStore(root)


@pytest.mark.parametrize("dataset",["assist2017","junyi","ednet_kt1"])
def test_registered_nonattention_settings_unchanged(dataset):
    profile = pipeline.profile_for(dataset)
    expected = {**HITSKT_COMMON_CONFIG,**HITSKT_TRAINING_CONFIGS[dataset]}
    assert {k:v for k,v in profile.items() if k != "model"} == {k:v for k,v in expected.items() if k != "model"}
    assert asdict(pipeline.model_config(dataset,20,5)) == {
        "num_questions":20,"num_skills":5,**{k:profile[k] for k in pipeline.MODEL_FIELDS}}


@pytest.mark.parametrize("run_id",["../historical","full/result","..","", "x.y"])
def test_path_escape_rejected(isolated,run_id):
    with pytest.raises(ValueError): pipeline.fresh_run("assist2017",run_id)


def test_existing_run_and_symlink_escape_rejected(isolated):
    pipeline.fresh_run("junyi","once")
    with pytest.raises(FileExistsError): pipeline.fresh_run("junyi","once")
    outside = isolated/"outside"; outside.mkdir()
    (pipeline.EXPERIMENT_ROOT/"assist2017").symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError): pipeline.fresh_run("assist2017","escape")


def test_full_training_default_refuses_execution():
    with pytest.raises(ValueError,match="explicit later decision"):
        pipeline.train("ednet_kt1",None,"must_not_run",device="cpu",favor=FavorPlusConfig())


def test_cpu_smoke_separate_namespace_and_no_benchmark_success(isolated):
    result = pipeline.smoke("junyi","synthetic_only",favor=FavorPlusConfig(features=8,chunk_size=3),steps=1)
    out = pipeline.EXPERIMENT_ROOT/"junyi"/"synthetic_only"
    assert result["status"] == "PASS" and not result["dataset_used"]
    assert result["checkpoint_reload_max_abs_logit_difference"] == 0
    assert (out/"_SMOKE_SUCCESS").exists() and not (out/"_SUCCESS").exists()
    assert "roc_auc" not in result


def test_synthetic_store_train_best_reload_replay_and_hash_rejection(isolated,monkeypatch):
    # One tiny synthetic epoch, not a real dataset experiment. Registry itself
    # is never mutated; the test replaces only the new pipeline profile getter.
    original = pipeline.profile_for
    monkeypatch.setattr(pipeline,"profile_for",lambda dataset:{**original(dataset),"epoch_ceiling":1})
    store = synthetic_store(isolated/"store")
    result = pipeline.train("junyi",store,"fixture_train",device="cpu",
        favor=FavorPlusConfig(features=8,chunk_size=3),allow_full_training=True)
    run = pipeline.EXPERIMENT_ROOT/"junyi"/"fixture_train"
    assert result["best_epoch"] == result["epochs_completed"] == 1
    assert result["test"]["targets"] == 6
    replay = pipeline.evaluate(run/"best_model.pt",store,"junyi","fixture_replay")
    assert result["test"] == replay["test"]
    assert replay["reproduction_status"] == "PASS"
    assert replay["metric_absolute_tolerance"] == 1e-6
    assert result["checkpoint_sha256"] == replay["checkpoint_sha256"]
    pred = pipeline.EXPERIMENT_ROOT/"junyi"/"fixture_replay"/"predictions"
    assert np.load(pred/"logits.npy").shape == (6,)
    assert np.load(pred/"labels.npy").sum() == 4
    # Fail closed on provenance mismatch before any output directory is created.
    with (store.root/"_SUCCESS").open("a") as f: f.write("changed\n")
    with pytest.raises(ValueError,match="identity mismatch"):
        pipeline.evaluate(run/"best_model.pt",store,"junyi","bad_data")
    assert not (pipeline.EXPERIMENT_ROOT/"junyi"/"bad_data").exists()


def test_real_sampler_target_count_and_history_visibility(isolated):
    store = synthetic_store(isolated/"store")
    test = RollingSessionDataset(store,"test",history_sessions=15)
    assert pipeline.target_count(test) == 6
    assert [len(s) for s in test[0].history] == [2,3,4,2]
    assert test[0].history[-1].correct.tolist() == store.session(3).correct.tolist()
