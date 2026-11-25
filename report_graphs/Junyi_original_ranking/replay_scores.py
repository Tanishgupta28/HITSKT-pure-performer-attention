"""Recreate held-out, fixed-history Junyi composite-skill scores in a private local CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def digest(path, normalize=False):
    if normalize:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def heldout(export):
    fields = ["student_id", "original_source_student_id", "target_session_id", "target_position",
              "target_skill_id", "target_question_id", "true_correctness", "history_session_count",
              "history_interaction_count"]
    data = pd.read_csv(export, usecols=fields)
    first = (data[data.target_position == 1].sort_values(["student_id", "target_session_id"])
             .drop_duplicates("student_id").set_index("student_id"))
    first = first[(first.history_session_count >= 4) & (first.history_interaction_count >= 20)]
    pairs = (data[data.student_id.isin(first.index)].groupby(["student_id", "target_skill_id"])
             .true_correctness.agg(["sum", "size"]).reset_index())
    pairs = pairs[pairs["size"] >= 5]
    ids = sorted(pairs.student_id.unique(), key=lambda s: hashlib.sha256(f"42:{s}".encode()).digest())
    assert len(ids) == 26077 and len(pairs) == 215487
    ids = sorted(ids[len(ids)//2:])
    pairs = pairs[pairs.student_id.isin(ids)]
    assert len(ids) == 13039 and len(pairs) == 107197 and pairs["size"].sum() == 1022557
    return first.loc[ids], pairs


def last_action(model, batch, positions):
    # For a single attention layer the ActionEncoder returns only the EOS vector.
    # At EOS all observed keys are causal. Associate q*(sum(k*v)) as sum((q*k)*v)
    # to avoid constructing a prefix matrix at every discarded intermediate position.
    enc = model.action_encoder
    layer, = enc.layers
    attn = layer.attention
    raw = enc.question(batch.history_question) + enc.skill(batch.history_skill) + enc.correct(batch.history_correct)
    x = enc.input_norm(raw + positions(raw.shape[1], raw.shape[2], raw.device, raw.dtype))
    rows = torch.arange(len(x))
    eos = batch.history_mask.sum(1)-1
    normalized = layer.attention_norm(x)
    q = F.elu(attn._heads(attn.query(normalized[rows, eos][:, None]))) + 1
    k = F.elu(attn._heads(attn.key(normalized))) + 1
    v = attn._heads(attn.value(normalized))
    weights = (q * k).sum(-1) * batch.history_mask[:, None]
    result = (weights[..., None] * v).sum(2) / weights.sum(2, keepdim=True).clamp_min(1e-6)
    merged = x[rows, eos] + attn.output(result.reshape(len(x), -1))
    return merged + layer.feedforward(layer.feedforward_norm(merged))


def replay(args):
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "backend")]
    from app.research import hitskt_types
    from ktbench.batching import Session, SessionExample, collate_hitskt
    model_cls = hitskt_types()[0]
    from ktbench.models.performer import sinusoidal_positions
    checkpoint = repo / "experiments/hitskt/junyi/full/best_model.pt"
    with zipfile.ZipFile(args.archive) as z:
        prefix = "skill_threshold_experiments/"
        manifest = json.loads(z.read(prefix + "experiment_provenance.json"))
        rank = pd.read_csv(z.open(prefix + "representative_questions.csv"))
    expected = manifest["original"]
    assert digest(checkpoint) == expected["checkpoint_sha256"]
    assert digest(args.raw) == expected["raw_source_expected_sha256"]
    assert digest(args.export) == manifest["previous_export"]["sha256"]
    maps = repo / "reports/datasets/junyi"
    for name in ["questions.csv", "question_skills.csv"]:
        assert digest(maps/name, True) == expected["mapping_sha256"]["data/processed/junyi/full/mappings/"+name]
    first, pairs = heldout(args.export)
    qs = pd.read_csv(maps/"questions.csv").set_index("source_question_id").question_id
    ss = pd.read_csv(maps/"question_skills.csv").set_index("question_id").skill_id
    reps = rank.sort_values("rank").groupby("composite_skill_id").question_id.apply(list).to_dict()
    assert all(1 <= len(q) <= 5 and len(set(q)) == len(q) and all(ss[x] == s for x in q) for s,q in reps.items())
    chunks = pd.read_csv(args.raw, usecols=["studentId", "startTime", "problemId", "correct"],
                         dtype={"studentId":"int32","startTime":"int64","problemId":"int32","correct":"uint8"},
                         chunksize=300000)
    raw = pd.concat([c[c.studentId.isin(first.original_source_student_id)] for c in chunks])
    raw = raw.sort_values(["studentId","startTime"],kind="stable")
    counts = raw.groupby("studentId",sort=True).size()
    ends = counts.cumsum()
    bounds = dict(zip(counts.index, zip(ends-counts, ends)))
    q_all = raw.problemId.map(qs).to_numpy(dtype="int32")
    s_all = ss.loc[q_all].to_numpy(dtype="int32")
    t_all, y_all = raw.startTime.to_numpy(), raw.correct.to_numpy()
    del raw
    model = model_cls.from_checkpoint(checkpoint,map_location="cpu").eval()
    sizes = dict(num_questions=model.config.num_questions,num_skills=model.config.num_skills)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    groups = pairs.groupby("student_id", sort=True)
    records, parity = [], []
    with torch.inference_mode():
        for i,(sid,g) in enumerate(groups):
            meta = first.loc[sid]
            lo, hi = bounds[int(meta.original_source_student_id)]
            starts = np.r_[lo, lo+1+np.flatnonzero(np.diff(t_all[lo:hi])>=36000), hi]
            cutoff = int(meta.target_session_id)-1
            assert cutoff == len(starts)-1-(len(starts)-1)//5
            assert int(q_all[starts[cutoff]]) == int(meta.target_question_id)
            begin = max(0,cutoff-15)
            assert starts[cutoff]-starts[begin] == meta.history_interaction_count
            sessions = [Session(q_all[a:b].tolist(),s_all[a:b].tolist(),y_all[a:b].tolist())
                        for a,b in zip(starts[begin:cutoff],starts[begin+1:cutoff+1])]
            items = [(q,s) for s in g.target_skill_id for q in reps[s]]
            q0,s0 = items[0]
            base = collate_hitskt([SessionExample(sessions,Session([q0],[s0],[0]),2)],**sizes)
            h = model.session_encoder(last_action(model,base,sinusoidal_positions), base)
            mem = model.correct_padding_encoder(h,base)
            iq,isk = np.array(items).T
            probs = []
            for j in range(0,len(items),128):
                qq, sk = torch.tensor(iq[j:j+128]), torch.tensor(isk[j:j+128])
                target = SimpleNamespace(target_question=torch.stack((qq,torch.full_like(qq,model.config.num_questions+1)),dim=1),
                                         target_skill=torch.stack((sk,torch.full_like(sk,model.config.num_skills+1)),dim=1),
                                         target_attention_mask=torch.ones((len(qq),2),dtype=torch.bool))
                out = model.decoder(mem.expand(len(qq),-1,-1),target)
                probs.extend(torch.sigmoid(model.prediction(out)[:,0,0]).tolist())
            if i%127 == 0 and base.history_mask.numel()<2500:
                parity.append(abs(float(torch.sigmoid(model(base)[0,0]))-probs[0]))
            j=0
            for row in g.itertuples():
                n=len(reps[row.target_skill_id])
                records.append((sid,row.target_skill_id,row.sum,row.size,np.mean(probs[j:j+n])))
                j+=n
            if (i+1)%1000 == 0:
                print(f"Scored {i+1:,} / 13,039 held-out students",flush=True)
    assert len(parity)>=40 and max(parity)<1e-5
    assert digest(checkpoint)==expected["checkpoint_sha256"]
    table = pd.DataFrame(records,columns=["student","skill","correct","attempts","score"])
    assert len(table)==107197
    args.scores.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.scores,index=False)
    print("Direct forward comparisons:",len(parity),"maximum absolute error:",max(parity),flush=True)
    args.scores.with_suffix(".audit.json").write_text(json.dumps({
        "direct_forward_checks":len(parity),"max_absolute_error":max(parity),
        "checkpoint_sha256_before_and_after":expected["checkpoint_sha256"],
        "source_sha256":expected["raw_source_expected_sha256"],"test_export_sha256":manifest["previous_export"]["sha256"],
        "replayed_score_table_sha256":digest(args.scores)},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--export",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=2)
    replay(parser.parse_args())
