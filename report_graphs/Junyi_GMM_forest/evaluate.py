"""Train the GMM-target forest and compare it to HiTSKT on identical held-out pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ["score", "past_attempts", "past_accuracy", "days_since"]
POLICIES = ["HiTSKT 61% rule", "GMM + random forest"]


def ordered(ids, seed):
    return sorted(ids, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())


def load_scores(path, history):
    audit = json.loads(path.with_suffix(".audit.json").read_text())
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == audit["replayed_score_table_sha256"]
    frame = pd.read_csv(path, float_precision="round_trip").merge(
        history, on=["student", "skill"], how="left", validate="one_to_one")
    missing = frame.past_attempts.isna()
    assert frame.loc[missing, ["past_accuracy", "days_since"]].isna().all().all()
    known = frame.loc[~missing]
    assert known.past_attempts.ge(1).all() and known.past_accuracy.between(0,1).all()
    assert known.days_since.ge(0).all() and frame.score.between(0,1).all()
    assert frame.correct.between(0,frame.attempts).all()
    frame["past_attempts"] = frame.past_attempts.fillna(0)
    for key in ["past_accuracy", "days_since"]:
        frame[key] = frame[key].fillna(-1)
    assert frame[FEATURES].notna().all().all() and frame.attempts.ge(5).all()
    frame["weak"] = frame.correct*100 < frame.attempts*60
    return frame, audit


def classification(truth, flag):
    tn, fp, fn, tp = map(int, confusion_matrix(truth, flag, labels=[False, True]).ravel())
    return dict(caught=tp, missed=fn, false_flags=fp, true_negatives=tn,
                precision=tp/max(1,tp+fp), recall=tp/(tp+fn), weak_f1=2*tp/(2*tp+fp+fn))


def learner_metrics(frame, flag, priority):
    data = frame.assign(flag=flag, priority=priority)
    relevant = data.groupby("student").weak.sum()
    relevant = relevant[relevant > 0]
    top = (data[data.flag & data.student.isin(relevant.index)]
           .sort_values(["student", "priority", "skill"], ascending=[True,False,True])
           .groupby("student").head(3).copy())
    discount = 1/np.log2([2,3,4])
    top["gain"] = top.weak * discount[top.groupby("student").cumcount()]
    lists = (top.groupby("student").agg(hit=("weak","max"), dcg=("gain","sum"), length=("weak","size"))
             .reindex(relevant.index, fill_value=0))
    lists["ndcg"] = lists.dcg / discount.cumsum()[np.minimum(3,relevant.to_numpy())-1]
    return lists


def ranking_comparison(results):
    rows = []
    arr = [m[["hit","ndcg"]].to_numpy(dtype=float) for m in results]
    assert results[0].index.equals(results[1].index)
    rng = np.random.default_rng(42)
    boot = np.empty((2000,2,2))
    for i in range(len(boot)):
        index = rng.integers(len(arr[0]), size=len(arr[0]))
        boot[i] = [a[index].mean(axis=0) for a in arr]
    scopes = list(zip(POLICIES, arr, [boot[:,0],boot[:,1]]))
    scopes.append(("Forest minus HiTSKT", arr[1]-arr[0], boot[:,1]-boot[:,0]))
    for name, values, samples in scopes:
        for j,metric in enumerate(["hit_rate_at_3","ndcg_at_3"]):
            low,high = np.quantile(samples[:,j],[.025,.975])
            rows.append(dict(policy=name, metric=metric, value=values[:,j].mean(),
                             ci_low=low, ci_high=high, learners=len(values)))
    return rows


def main(development, heldout, history, output):
    output.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history)
    dev, dev_audit = load_scores(development, history)
    test, test_audit = load_scores(heldout, history)
    assert len(dev)==108290 and len(test)==107197
    all_ids = ordered(set(dev.student) | set(test.student), 42)
    assert len(all_ids)==26077 and set(all_ids[:13038])==set(dev.student)
    assert set(all_ids[13038:])==set(test.student)
    for key in ["checkpoint_sha256_before_and_after","source_sha256","test_export_sha256"]:
        assert dev_audit[key]==test_audit[key]
    ids = ordered(dev.student.unique(), 73)
    fit_ids = ids[:len(ids)*4//5]
    train = dev[dev.student.isin(fit_ids)]
    val = dev[~dev.student.isin(fit_ids)]
    outcome = (train.correct/train.attempts).to_numpy()[:,None]
    scaler = StandardScaler().fit(outcome)
    x = scaler.transform(outcome)
    with threadpool_limits(limits=2):
        gmm = GaussianMixture(n_components=3, reg_covar=.01, n_init=5, random_state=42).fit(x)
    assert gmm.converged_
    means = scaler.inverse_transform(gmm.means_).ravel()
    target = gmm.predict_proba(x)[:,means.argmin()]
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=25, random_state=42,
                                   n_jobs=2).fit(train[FEATURES], target)
    val_risk = forest.predict(val[FEATURES])
    grid = pd.DataFrame([dict(cutoff=p/100, **classification(val.weak, val_risk >= p/100))
                         for p in range(1,100)])
    best = grid.sort_values(["weak_f1","recall","cutoff"], ascending=[False,False,True]).iloc[0]
    grid.to_csv(output/"validation_cutoffs.csv", index=False)
    risk = forest.predict(test[FEATURES])
    policies = [(test.score < .61, 1-test.score), (risk >= best.cutoff, risk)]
    metrics, rankings = [], []
    for name,(flag,priority) in zip(POLICIES,policies):
        m = learner_metrics(test,flag,priority)
        rankings.append(m)
        metrics.append(dict(policy=name, **classification(test.weak,flag),
                            average_precision=average_precision_score(test.weak,priority),
                            learners_with_hit=int(m.hit.sum()), no_recommendation=int((m.length==0).sum()),
                            flagged_pairs=int(flag.sum()), learners_flagged=int(test.loc[flag,"student"].nunique())))
    assert metrics[0]["caught"]==15184 and metrics[0]["learners_with_hit"]==4853
    assert len(rankings[0])==7702 and np.isclose(rankings[0].ndcg.mean(),.509293,atol=.000001)
    pd.DataFrame(metrics).to_csv(output/"weak_topic_metrics.csv", index=False)
    ranking = pd.DataFrame(ranking_comparison(rankings))
    ranking.to_csv(output/"ranking_results.csv", index=False)
    unrestricted = [learner_metrics(test, np.ones(len(test), dtype=bool), priority)
                    for _, priority in policies]
    assert unrestricted[0].length.equals(unrestricted[1].length)
    assert int(unrestricted[0].hit.sum())==7041 and np.isclose(unrestricted[0].ndcg.mean(),.756915,atol=.000001)
    pd.DataFrame(ranking_comparison(unrestricted)).to_csv(output/"ranking_without_flags.csv", index=False)
    audit = dict(features=FEATURES, sklearn=sklearn.__version__, original_development_replay=dev_audit,
                 original_heldout_replay=test_audit, selected_risk_threshold=best.cutoff,
                 validation_weak_f1=best.weak_f1, gmm_centers=sorted(means.tolist()),
                 splits={label:dict(learners=frame.student.nunique(), pairs=len(frame),
                                    distinct_skills=frame.skill.nunique(), observed_weak=int(frame.weak.sum()),
                                    later_answers=int(frame.attempts.sum()), unseen_in_history=int(frame.past_attempts.eq(0).sum()))
                         for label,frame in [("fit",train),("validation",val),("test",test)]})
    (output/"verification.json").write_text(json.dumps(audit, indent=2, default=int)+"\n")
    display = pd.DataFrame({"Hit Rate@3":[m.hit.mean() for m in rankings],
                            "NDCG@3":[m.ndcg.mean() for m in rankings],
                            "Weak recall":[m["recall"] for m in metrics],
                            "Weak precision":[m["precision"] for m in metrics]}, index=POLICIES)
    ax = display.T.plot.bar(rot=0, color=["#285578","#b86435"], figsize=(8,4))
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylim(0,1)
    ax.set_title("Weak-topic recommendations on the original Junyi holdout")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output/"comparison.png", dpi=160)
    plt.close()
    print(ranking.to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["development","heldout","history","output"]:
        parser.add_argument("--"+key, type=Path, required=True)
    main(**vars(parser.parse_args()))
