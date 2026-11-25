"""Plot the historical HiTSKT benchmark from its original validation logs."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASETS = {"assist2017": "ASSIST2017", "junyi": "Junyi", "ednet_kt1": "EdNet-KT1",
            "assist2009": "ASSIST2009"}
PHI_ONLY = "RKT (Phi-only)"
HISTORICAL = tuple(d for d in DATASETS if d != "assist2009")
# Color, marker, line style. Markers identify individual measured epochs.
STYLES = {
    "DKT": ("#0072B2", "o", "-"),
    "DKVMN": ("#9A6500", "s", "--"),
    "SAKT": ("#008464", "^", "-."),
    "RKT": ("#774B9B", "D", ":"),
    "HiTSKT": ("#C04623", "v", "-"),
}
METRICS = {
    "loss": ("Validation Loss", "loss_vs_epoch"),
    "roc_auc": ("Validation ROC-AUC", "validation_auc_vs_epoch"),
}


class Point(NamedTuple):
    epoch: int
    loss: float
    roc_auc: float


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"{path}: duplicate or missing columns (need {sorted(required)})")
        return list(reader)


def load_run(summary):
    dataset, model = summary["dataset"], summary["model"]
    family = "rkt" if model == PHI_ONLY and dataset == "assist2009" else model.lower()
    relative = Path("experiments", family, dataset, "full")
    if summary["experiment_directory"] != relative.as_posix():
        raise ValueError(f"{model} / {dataset}: unexpected experiment_directory")
    folder = ROOT / relative
    count = int(summary["epochs_completed"])
    best_epoch = int(summary["best_epoch"])
    best_auc = float(summary["best_validation_auc"])
    identity = dict(dataset=dataset, model=model, epochs_completed=count,
                    best_epoch=best_epoch, best_validation_auc=best_auc, seed=int(summary["seed"]))
    for filename in ("final_results.json", "config.json"):
        path = folder / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = dict(identity)
        if filename == "final_results.json":
            expected["status"] = "complete"
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"{path}: {key} is {data.get(key)!r}, expected {value!r}")

    path = folder / "metrics.csv"
    points = []
    for line, row in enumerate(read_csv(path, {"split", "epoch", "loss", "roc_auc"}), 2):
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"{path}:{line}: unknown split {row['split']!r}")
        if row["split"] == "validation":
            try:
                point = Point(int(row["epoch"]), float(row["loss"]), float(row["roc_auc"]))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: invalid validation observation") from exc
            if not math.isfinite(point.loss) or point.loss < 0 or not 0 <= point.roc_auc <= 1:
                raise ValueError(f"{path}:{line}: out-of-range or nonfinite metric")
            points.append(point)
    points.sort()
    if count < 1 or [p.epoch for p in points] != list(range(1, count + 1)):
        raise ValueError(f"{path}: expected exactly one validation row per epoch from 1 to {count}")
    peak = max(points, key=lambda p: p.roc_auc)
    if (peak.epoch, peak.roc_auc) != (best_epoch, best_auc):
        raise ValueError(f"{path}: observed AUC peak {peak} disagrees with stored result")
    return points


def load_benchmark(*, assist2009=False):
    path = ROOT / "reports" / ("assist2009_final_results.csv" if assist2009 else "final_results.csv")
    datasets = ("assist2009",) if assist2009 else HISTORICAL
    rows = read_csv(path, {"dataset", "model", "experiment_directory", "epochs_completed",
                           "best_epoch", "best_validation_auc", "seed"})
    models = tuple(PHI_ONLY if assist2009 and model == "RKT" else model for model in STYLES)
    expected = {(d, m) for d in datasets for m in models} - {("ednet_kt1", "DKT")}
    runs = {}
    for row in rows:
        key = row["dataset"], row["model"]
        if key[0] not in datasets or key[1] not in models:
            continue
        if key not in expected or key in runs:
            raise ValueError(f"{path}: unexpected or duplicate historical experiment {key}")
        runs[key] = load_run(row)
    if assist2009 and not runs:
        raise ValueError(f"{path}: no verified ASSIST2009 results to plot")
    if not assist2009 and (missing := expected.difference(runs)):
        raise ValueError(f"{path}: missing experiments {sorted(missing)}")
    return runs


def make_figure(dataset, metric, runs):
    label, _ = METRICS[metric]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=320, facecolor="white")
    fig.subplots_adjust(left=0.115, right=0.97, top=0.78, bottom=0.14)
    fig.text(0.115, 0.935, f"{DATASETS[dataset]}  |  {label}", fontsize=16,
             weight="semibold", color="#202831", va="center")
    last_epoch = 0
    for model, (color, marker, style) in STYLES.items():
        if dataset == "assist2009" and model == "RKT":
            model = PHI_ONLY
        points = runs.get((dataset, model))
        if points is None:
            continue
        last_epoch = max(last_epoch, points[-1].epoch)
        ax.plot([p.epoch for p in points], [getattr(p, metric) for p in points],
                label=model, color=color, marker=marker, linestyle=style,
                markersize=3.7, linewidth=2 if model == "HiTSKT" else 1.45,
                markeredgewidth=0.7, markerfacecolor="white")
    if dataset == "assist2009" and len(ax.lines) < len(STYLES):
        fig.text(0.97, 0.935, f"{len(ax.lines)} of {len(STYLES)} models",
                 fontsize=9, color="#536273", va="center", ha="right")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.018), ncols=len(ax.lines),
              frameon=False, borderaxespad=0, handlelength=2.4, columnspacing=1.55)
    ax.set_xlabel("Epoch", labelpad=6)
    ax.set_ylabel(label, labelpad=7)
    ax.set_xticks([1, *range(5, last_epoch + 1, 5)])
    ax.set_xlim(0.4, last_epoch + 0.6)
    ax.margins(y=0.085)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter("{x:.3f}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DFE3E6", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#889199")
    ax.tick_params(color="#889199", labelcolor="#303840")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--assist2009", action="store_true")
    parser.add_argument("--replace-assist2009", action="store_true",
                        help="refresh the ASSIST2009 PNGs after new runs are independently verified")
    args = parser.parse_args()
    if args.replace_assist2009 and not args.assist2009:
        parser.error("--replace-assist2009 requires --assist2009")
    runs = load_benchmark(assist2009=args.assist2009)
    images = {}
    with plt.style.context("default"), plt.rc_context({"font.family": "DejaVu Sans",
                                                       "axes.labelsize": 11,
                                                       "path.simplify": False}):
        for dataset in dict.fromkeys(key[0] for key in runs):
            for metric, (_, suffix) in METRICS.items():
                fig = make_figure(dataset, metric, runs)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=320, facecolor="white")
                plt.close(fig)
                images[args.output_dir / f"{dataset}_{suffix}.png"] = buffer.getvalue()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in images.items():
        if path.exists() and path.read_bytes() != content and not args.replace_assist2009:
            raise FileExistsError(f"{path} differs. Use --output-dir with an empty directory.")
    for path, content in images.items():
        if args.replace_assist2009:
            temporary = path.with_suffix(".png.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        elif not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    print(f"Validated {len(runs)} runs and {sum(map(len, runs.values()))} validation epochs. "
          f"Verified or generated {len(images)} figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
