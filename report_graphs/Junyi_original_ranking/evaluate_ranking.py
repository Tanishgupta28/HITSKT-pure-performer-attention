"""Calculate per-learner Hit Rate@3 and binary NDCG@3 for original Junyi test pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

K = 3
WEIGHTS = 1 / np.log2(np.arange(2, K+2))
POLICIES = {"original_weak_cutoff": 0.61, "rank_all_eligible_skills": None}


def per_learner(relevance, total_weak):
    """relevance is ordered, possibly empty. Ideal uses all eligible observed weak skills."""
    ranked = np.asarray(relevance[:K], dtype=int)
    ideal = WEIGHTS[:min(K,total_weak)].sum()
    if ideal <= 0:
        raise ValueError("No observed weak skill; outside the conditional evaluation.")
    return (int(ranked.any()), float(ranked @ WEIGHTS[:len(ranked)] / ideal))


def check_examples():
    assert per_learner([],2) == (0,0.0)
    assert per_learner([1],1) == (1,1.0)
    assert per_learner([0,0,1],1) == (1,0.5)
    assert per_learner([1],3)[1] < 0.5
    assert per_learner([1,1,1,0],4) == (1,1.0)


def score_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def evaluate(scores, archive, output):
    check_examples()
    audit = json.loads(scores.with_suffix(".audit.json").read_text())
    assert audit["replayed_score_table_sha256"] == score_digest(scores)
    with zipfile.ZipFile(archive) as z:
        base="skill_threshold_experiments/"
        reference=json.loads(z.read(base+"final_evaluation_metrics.json"))["test_optimized"]
        exp1=json.loads(z.read(base+"experiment_1_metrics.json"))["test"]["5"]
    df=pd.read_csv(scores, float_precision="round_trip")
    assert df[["student","skill"]].duplicated().sum()==0 and df.notna().all().all()
    assert len(df)==107197 and df.student.nunique()==13039 and df.skill.nunique()==exp1["distinct_composite_skills"]
    assert df.attempts.ge(5).all() and df.correct.between(0,df.attempts).all() and df.score.between(0,1).all()
    true_classes = (df.correct*100 >= df.attempts*60).astype(int)+(df.correct*100 >= df.attempts*85).astype(int)
    pred_classes = (df.score>=.61).astype(int)+(df.score>=.86).astype(int)
    observed = confusion_matrix(true_classes,pred_classes,labels=[0,1,2]).tolist()
    expected = reference["confusion_matrix"]
    df["weak"] = true_classes==0
    groups = df.sort_values(["student","score","skill"]).groupby("student",sort=True)
    counts = groups.weak.sum()
    evidence={"model_replay":audit,"heldout_learners":int(len(counts)),
              "learners_with_observed_weak":int((counts>0).sum()),
              "learners_with_no_observed_weak":int((counts==0).sum()),
              "distinct_composite_skills":int(df.skill.nunique()),"heldout_pairs":len(df),
              "observed_weak_pairs":int(df.weak.sum()),"heldout_followup_answers":int(df.attempts.sum()),
              "published_mean_prediction":exp1["mean_prediction"],"replayed_mean_prediction":df.score.mean(),
              "published_mae":exp1["mae"],"replayed_mae":(df.score-df.correct/df.attempts).abs().mean(),
              "published_confusion_matrix":expected,"replayed_confusion_matrix":observed,
              "confusion_matrix_exact_match":observed==expected}
    result=[]
    for policy, threshold in POLICIES.items():
        values=[]; empty=0; short=0; available_under_k=0
        for learner,group in groups:
            n = int(counts.loc[learner])
            if n == 0:
                continue
            ranked = group if threshold is None else group[group.score < threshold]
            available_under_k += len(group)<K
            empty += len(ranked)==0
            short += len(ranked)<K
            values.append(per_learner(ranked.weak.to_numpy(),n))
        arr=np.array(values)
        means=arr.mean(axis=0)
        hits=int(arr[:,0].sum())
        assert np.all((arr>=0)&(arr<=1+1e-12)) and np.isclose(means[0],hits/len(arr))
        # Each draw resamples learners, preserving the joint ranking of their skills.
        rng=np.random.default_rng(42)
        boot=np.array([arr[rng.integers(0,len(arr),len(arr))].mean(axis=0) for _ in range(2000)])
        limits=np.quantile(boot,[.025,.975],axis=0)
        result.append(dict(policy=policy,k=K,learners=len(arr),learners_with_hit=hits,
                           no_recommendation=empty,lists_shorter_than_k=short,
                           learners_with_fewer_than_k_evaluable_skills=available_under_k,
                           hit_rate=means[0],hr_ci_low=limits[0,0],hr_ci_high=limits[1,0],
                           ndcg=means[1],ndcg_ci_low=limits[0,1],ndcg_ci_high=limits[1,1]))
    assert result[0]["hit_rate"]<=result[1]["hit_rate"]+1e-12
    assert result[0]["ndcg"]<=result[1]["ndcg"]+1e-12
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(output/"ranking_results.csv",index=False)
    (output/"verification.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps({"results":result,"audit":evidence},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    evaluate(args.scores,args.archive,args.output)
