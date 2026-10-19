"""Separate FAVOR+ experiment pipeline using unchanged HiTSKT V1 data/settings.

No automatic CUDA selection, implicit resume, truncation or artifact overwrite.
Full training is exposed for a later explicit decision; current validation uses
synthetic CPU batches only. All outputs stay in the variant's namespace.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import Session, SessionExample, collate_hitskt
from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import RollingSessionDataset, SessionStore, make_session_dataloader
from ktbench.hitskt_registry import HITSKT_COMMON_CONFIG, HITSKT_TRAINING_CONFIGS
from ktbench.metrics import binary_metrics, masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus, VARIANT
from ktbench.training import COMMON_EARLY_STOPPING, ValidationAUCEarlyStopping, capture_rng_state

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / VARIANT
PROTOCOL = "HiTSKT_V1_chronological_within_student_retrospective_teacher_forced"
MODEL_FIELDS = ("width", "heads", "feedforward_width", "dropout", "action_layers",
                "session_layers", "correct_layers", "decoder_layers")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def profile_for(dataset: str) -> dict:
    return {**HITSKT_COMMON_CONFIG, **HITSKT_TRAINING_CONFIGS[dataset], "model": VARIANT}


def model_config(dataset: str, questions: int, skills: int) -> HiTSKTConfig:
    p = profile_for(dataset)
    return HiTSKTConfig(questions, skills, **{key:p[key] for key in MODEL_FIELDS})


def fresh_run(dataset: str, run_id: str) -> Path:
    if dataset not in HITSKT_TRAINING_CONFIGS or not run_id or run_id in (".", ".."):
        raise ValueError("invalid dataset or run id")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run id must contain only letters, digits, underscores or hyphens")
    path = EXPERIMENT_ROOT / dataset / run_id
    # Resolve the complete path, including existing parent symlinks, before write.
    if not path.resolve().is_relative_to((ROOT/"experiments"/VARIANT).absolute()):
        raise ValueError("experiment path escapes FAVOR+ namespace")
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity() -> dict:
    names = ["ktbench/models/favor_plus.py", "ktbench/models/hitskt_favor_plus.py",
             "ktbench/favor_plus_pipeline.py", "ktbench/models/hitskt.py",
             "ktbench/models/performer.py", "ktbench/hitskt_registry.py",
             "ktbench/batching.py", "ktbench/data/session_store.py", "ktbench/metrics.py",
             "scripts/train_hitskt_favor_plus.py", "scripts/evaluate_hitskt_favor_plus.py"]
    return {"git_head": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "source_sha256": {n:sha256(ROOT/n) for n in names}}


def runtime(device: torch.device) -> dict:
    result = {"device":str(device), "python":platform.python_version(), "torch":str(torch.__version__),
              "numpy":np.__version__, "threads":torch.get_num_threads(),
              "float32_matmul_precision":torch.get_float32_matmul_precision(),
              "cudnn_benchmark":torch.backends.cudnn.benchmark,
              "cudnn_deterministic":torch.backends.cudnn.deterministic,
              "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
              "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
              "cuda_runtime":torch.version.cuda}
    if device.type == "cuda":
        result.update(gpu=torch.cuda.get_device_name(device),
                      gpu_capability=list(torch.cuda.get_device_capability(device)))
    return result


def store_identity(store: SessionStore) -> dict:
    return {"metadata":store.metadata, "files":{
        path.name:sha256(path) for path in sorted(store.root.iterdir()) if path.is_file()}}


def target_count(dataset: RollingSessionDataset) -> int:
    offsets = dataset.store.session_offsets
    return int(np.sum(offsets[dataset.targets+1]-offsets[dataset.targets]))


def epoch(model: HiTSKTFavorPlus, dataset: RollingSessionDataset, *, device: torch.device,
          epoch_index: int, optimizer: torch.optim.Optimizer | None = None,
          prediction_output: Path | None = None) -> dict:
    """Original sampler, loss, sigmoid, threshold and metric arithmetic.

    Exact target accounting; evaluation prediction arrays use disk-backed NPY
    to bound host RAM. Training metrics use linear-sized arrays as in V1.
    """
    training = optimizer is not None
    if training and prediction_output is not None:
        raise ValueError("prediction export is for evaluation only")
    model.train(training)
    p = HITSKT_COMMON_CONFIG
    loader,sampler = make_session_dataloader(dataset, token_budget=p["token_budget"],
        maximum_batch_size=p["maximum_batch_size"],bucket_size=p["bucket_size"],
        shuffle=training,seed=PROJECT_SEED,workers=0,pin_memory=False)
    sampler.set_epoch(epoch_index if training else 0)
    count = target_count(dataset)
    if count == 0:
        raise ValueError("empty supervised split")
    if prediction_output is None:
        probabilities = np.empty(count,np.float32)
        labels = np.empty(count,np.uint8)
        logits_out = None
    else:
        prediction_output.mkdir(exist_ok=False)
        probabilities = np.lib.format.open_memmap(prediction_output/"probabilities.npy",mode="w+",dtype="float32",shape=(count,))
        labels = np.lib.format.open_memmap(prediction_output/"labels.npy",mode="w+",dtype="uint8",shape=(count,))
        logits_out = np.lib.format.open_memmap(prediction_output/"logits.npy",mode="w+",dtype="float32",shape=(count,))
    cursor,loss_sum = 0,0.
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite FAVOR+ loss")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite FAVOR+ gradient; optimizer not stepped")
                optimizer.step()
        valid = logits.masked_select(batch.target_metric_mask).detach()
        y = batch.target_labels.masked_select(batch.target_metric_mask)
        n = y.numel()
        probabilities[cursor:cursor+n] = valid.sigmoid().cpu().numpy()
        labels[cursor:cursor+n] = y.cpu().numpy()
        if logits_out is not None: logits_out[cursor:cursor+n] = valid.cpu().numpy()
        loss_sum += float(loss.detach().cpu())*n
        cursor += n
    if cursor != count: raise RuntimeError("supervised target count mismatch")
    metrics = binary_metrics(torch.from_numpy(probabilities),torch.from_numpy(labels),loss=loss_sum/count)
    metrics["bce"] = metrics["loss"]
    for array in (probabilities,labels,logits_out):
        if isinstance(array,np.memmap): array.flush()
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        raise FloatingPointError("nonfinite metrics (including undefined single-class AUC)")
    return metrics


def train(dataset_name: str, store: SessionStore, run_id: str, *, device: str,
          favor: FavorPlusConfig, allow_full_training: bool = False) -> dict:
    if not allow_full_training:
        raise ValueError("full training requires an explicit later decision and --allow-full-training")
    dev = torch.device(device)
    seed_everything()
    splits = {s:RollingSessionDataset(store,s,history_sessions=15) for s in ("train","validation","test")}
    p = profile_for(dataset_name)
    config = model_config(dataset_name,int(store.metadata["num_questions"]),int(store.metadata["num_skills"]))
    model = HiTSKTFavorPlus(config,favor).to(dev)
    identity = store_identity(store)
    out = fresh_run(dataset_name,run_id)
    manifest = {"model":VARIANT,"kind":"full_dataset_training","dataset":dataset_name,
                "seed":PROJECT_SEED,"protocol":PROTOCOL,"profile":p,"model_config":asdict(config),
                "favor_config":asdict(favor),"dataset_identity":identity,"runtime":runtime(dev),
                "early_stopping":COMMON_EARLY_STOPPING.as_dict(),"source":source_identity(),
                "action_truncation":False,"action_chunking":False,
                "note":"attention checkpoint chunks do not split or truncate data sessions"}
    write_json(out/"config.json",manifest)
    optimizer = torch.optim.Adam(model.parameters(),lr=p["learning_rate"],weight_decay=p["weight_decay"])
    control = ValidationAUCEarlyStopping()
    for e in range(1,p["epoch_ceiling"]+1):
        tm = epoch(model,splits["train"],device=dev,epoch_index=e,optimizer=optimizer)
        vm = epoch(model,splits["validation"],device=dev,epoch_index=e)
        decision = control.step(e,vm["roc_auc"])
        payload = model.checkpoint()
        payload.update(epoch=e,validation_roc_auc=vm["roc_auc"],seed=PROJECT_SEED,
            experiment=manifest,optimizer_state_dict=optimizer.state_dict(),rng_state=capture_rng_state(),
            early_stopping_state=asdict(decision))
        torch.save(payload,out/"last_model.pt")
        if decision.improved: torch.save(payload,out/"best_model.pt")
        write_json(out/f"epoch_{e:03d}.json",{"train":tm,"validation":vm,"selection":asdict(decision)})
        if decision.should_stop: break
    if control.best_epoch is None: raise RuntimeError("no finite validation-selected checkpoint")
    best = HiTSKTFavorPlus.from_checkpoint(out/"best_model.pt").to(dev)
    metrics = epoch(best,splits["test"],device=dev,epoch_index=0,prediction_output=out/"test_predictions")
    result = {"model":VARIANT,"dataset":dataset_name,"protocol":PROTOCOL,"seed":PROJECT_SEED,
              "best_epoch":control.best_epoch,"best_validation_auc":control.best_validation_auc,
              "epochs_completed":e,"best_checkpoint_reloaded":True,"test":metrics,
              "checkpoint_sha256":sha256(out/"best_model.pt"),"status":"complete"}
    write_json(out/"final_results.json",result)
    (out/"_SUCCESS").write_text("complete\n")
    return result


def evaluate(checkpoint_path: Path, store: SessionStore, dataset_name: str, run_id: str,
             *, device: str = "cpu") -> dict:
    """Fresh tagged checkpoint replay; refuse changed data/profile/protocol."""
    dev = torch.device(device)
    seed_everything()
    payload = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    manifest = payload.get("experiment",{})
    if (manifest.get("kind") != "full_dataset_training" or manifest.get("dataset") != dataset_name
        or manifest.get("protocol") != PROTOCOL or manifest.get("profile") != profile_for(dataset_name)):
        raise ValueError("checkpoint is not a matching full FAVOR+ V1 experiment")
    identity = store_identity(store)
    if identity != manifest.get("dataset_identity"):
        raise ValueError("checkpoint/dataset identity mismatch")
    net = HiTSKTFavorPlus.from_checkpoint(checkpoint_path).to(dev)
    out = fresh_run(dataset_name,run_id)
    ds = RollingSessionDataset(store,"test",history_sessions=15)
    metrics = epoch(net,ds,device=dev,epoch_index=0,prediction_output=out/"predictions")
    result = {"model":VARIANT,"kind":"fresh_checkpoint_evaluation","dataset":dataset_name,
              "protocol":PROTOCOL,"test":metrics,"epoch":payload["epoch"],
              "checkpoint_sha256":sha256(checkpoint_path),"dataset_identity":identity,
              "runtime":runtime(dev),"source":source_identity(),
              "prediction_sha256":{p.name:sha256(p) for p in (out/"predictions").iterdir()}}
    # Fixed before any FAVOR+ dataset results exist; same absolute metric gate
    # used in the historical independent audit. Failure remains visible evidence.
    reference = checkpoint_path.parent/"final_results.json"
    result["metric_absolute_tolerance"] = 1e-6
    result["reproduction_status"] = "NOT VERIFIED"
    if reference.exists():
        stored = json.loads(reference.read_text())
        if (stored.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or stored.get("best_epoch") != result["epoch"]):
            raise ValueError("stored results do not select this checkpoint")
        comparison = {k:{"stored":stored["test"][k],"replayed":metrics[k],
                          "absolute_difference":abs(stored["test"][k]-metrics[k]),
                          "pass":abs(stored["test"][k]-metrics[k]) <= 1e-6}
                      for k in ("roc_auc","accuracy","precision","recall","f1","mse","bce")}
        counts_match = stored["test"]["targets"] == metrics["targets"]
        result.update(comparison=comparison,targets_match=counts_match,
                      reproduction_status="PASS" if counts_match and all(v["pass"] for v in comparison.values()) else "FAIL")
    write_json(out/"evaluation_manifest.json",result)
    (out/"_EVALUATION_COMPLETE").write_text("execution complete; inspect reproduction_status\n")
    return result


def smoke(dataset_name: str, run_id: str, *, favor: FavorPlusConfig, steps: int = 2) -> dict:
    """Bounded CPU-only synthetic optimization with the *registered* widths/FF/LR.

    No dataset read, no benchmark metric or AUC claim, no full-run success marker.
    """
    if not 1 <= steps <= 5: raise ValueError("CPU smoke is restricted to 1..5 steps")
    seed_everything()
    p = profile_for(dataset_name)
    config = model_config(dataset_name,20,5)
    model = HiTSKTFavorPlus(config,favor)
    examples = [SessionExample([Session([1,2,3],[1,2,3],[0,1,0])]*h,
                Session(list(range(1,n+1)),[1]*n,[i%2 for i in range(n)]),0)
                for n,h in ((7,2),(3,1))]
    batch = collate_hitskt(examples,num_questions=20,num_skills=5)
    optimizer = torch.optim.Adam(model.parameters(),lr=p["learning_rate"],weight_decay=p["weight_decay"])
    out = fresh_run(dataset_name,run_id)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_bce_loss(model(batch),batch.target_labels,batch.target_metric_mask)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite smoke loss/gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad(): before = model(batch)
    torch.save(model.checkpoint(),out/"smoke_model.pt")
    loaded = HiTSKTFavorPlus.from_checkpoint(out/"smoke_model.pt").eval()
    with torch.no_grad(): after = loaded(batch)
    if not torch.equal(before,after): raise RuntimeError("smoke checkpoint reload changed logits")
    np.save(out/"smoke_logits.npy",before.cpu().numpy())
    result = {"status":"PASS","model":VARIANT,"kind":"CPU_synthetic_smoke_only",
              "dataset_profile":dataset_name,"protocol":PROTOCOL,"steps":steps,"losses":losses,
              "targets_per_step":int(batch.target_metric_mask.sum()),"profile":p,
              "model_config":asdict(config),"favor_config":asdict(favor),
              "checkpoint_reload_max_abs_logit_difference":float((before-after).abs().max()),
              "checkpoint_sha256":sha256(out/"smoke_model.pt"),"runtime":runtime(torch.device("cpu")),
              "logits_sha256":sha256(out/"smoke_logits.npy"),
              "source":source_identity(),"dataset_used":False,
              "limitations":"synthetic reduced vocabulary; no dataset AUC, full epoch or CUDA evidence"}
    write_json(out/"smoke_manifest.json",result)
    (out/"_SMOKE_SUCCESS").write_text("synthetic only\n")
    return result
