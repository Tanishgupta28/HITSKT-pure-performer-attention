"""Render historical trainable parameter counts from the completed benchmark."""

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASETS = {"assist2017": "ASSIST2017", "junyi": "Junyi", "ednet_kt1": "EdNet-KT1",
            "assist2009": "ASSIST2009", "algebra2005": "Algebra I 2005–2006"}
PHI_ONLY = "RKT (Phi-only)"
HISTORICAL = tuple(d for d in DATASETS if d not in ("assist2009", "algebra2005"))
MODELS = ("DKT", "DKVMN", "SAKT", "RKT", "HiTSKT")
BAR_BLUE = "#5186B6"
HITSKT_BLUE = "#164775"


def validated_counts(*, assist2009=False, algebra2005=False):
    datasets = ("algebra2005",) if algebra2005 else ("assist2009",) if assist2009 else HISTORICAL
    models = tuple(PHI_ONLY if assist2009 and model == "RKT" else model for model in MODELS)
    counts = {dataset: {} for dataset in datasets}
    index = ROOT / "reports" / (f"{datasets[0]}_final_results.csv" if assist2009 or algebra2005 else "final_results.csv")
    with index.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"model", "dataset", "experiment_directory", "trainable_parameters"}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"{index}: missing or duplicate column names")
        for row in reader:
            model, dataset = row["model"], row["dataset"]
            if dataset not in counts or model not in models:
                continue
            family = "rkt" if model == PHI_ONLY else model.lower()
            folder = Path("experiments", family, dataset, "full")
            if row["experiment_directory"] != folder.as_posix() or model in counts[dataset]:
                raise ValueError(f"{index}: unexpected path or duplicate run for {model}/{dataset}")
            raw = row["trainable_parameters"]
            if not raw or not raw.isascii() or not raw.isdecimal() or int(raw) < 1:
                raise ValueError(f"{index}:{reader.line_num}: invalid parameter count {raw!r}")
            count = int(raw)
            for filename in ("config.json", "final_results.json"):
                path = ROOT / folder / filename
                run = json.loads(path.read_text(encoding="utf-8"))
                if (run.get("model"), run.get("dataset")) != (model, dataset):
                    raise ValueError(f"{path}: run identity differs from index")
                checked = run.get("trainable_parameters")
                if type(checked) is not int or checked != count:
                    raise ValueError(f"{path}: trainable_parameters {checked!r} differs from {raw}")
                if filename == "final_results.json" and run.get("status") != "complete":
                    raise ValueError(f"{path}: indexed run is incomplete")
            counts[dataset][model] = count
    for dataset, values in counts.items():
        if assist2009:
            if not values:
                raise ValueError(f"{index}: no verified ASSIST2009 results to plot")
            continue
        if algebra2005 and set(values) != set(MODELS):
            raise ValueError("Algebra2005 requires all five completed runs")
        expected = {model for model in MODELS
                    if (ROOT / "experiments" / model.lower() / dataset / "full" / "_SUCCESS").is_file()}
        if set(values) != expected:
            raise ValueError(f"{index}: unexpected model coverage for {dataset}: {sorted(values)}")
    return counts


def count_label(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M ({count:,})"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K ({count:,})"
    return f"{count:,}"


def make_figure(dataset, counts, axis_max):
    ordered = sorted(counts, key=counts.get, reverse=True)
    values = [counts[model] for model in ordered]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=320, facecolor="white")
    left_margin = 0.185 if dataset == "assist2009" else 0.15
    fig.subplots_adjust(left=left_margin, right=0.975, bottom=0.19, top=0.85)
    ax.set_title(f"Model Parameter Comparison — {DATASETS[dataset]}", loc="left",
                 fontsize=17, weight="semibold", color="#1F3042", pad=16)
    colors = [HITSKT_BLUE if model == "HiTSKT" else BAR_BLUE for model in ordered]
    bars = ax.barh(ordered, values, height=0.52, color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[count_label(n) for n in values], padding=5,
                 fontsize=10, color="#213348")
    ax.set_xlim(0, axis_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
    ax.set_xlabel("Number of Trainable Parameters", fontsize=11, labelpad=8)
    ax.set_ylabel("Model Name", fontsize=11, labelpad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DFE5EA", linewidth=0.7)
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A6B1BB")
    ax.tick_params(axis="y", length=0, pad=7, labelsize=11)
    ax.tick_params(axis="x", color="#A6B1BB", labelsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--assist2009", action="store_true")
    mode.add_argument("--algebra2005", action="store_true")
    parser.add_argument("--replace-ednet-kt1", action="store_true",
                        help="refresh only the EdNet-KT1 PNGs after all five runs are verified")
    parser.add_argument("--replace-assist2009", action="store_true",
                        help="refresh the ASSIST2009 PNGs after new runs are independently verified")
    args = parser.parse_args()
    if args.replace_ednet_kt1 and (args.assist2009 or args.algebra2005):
        parser.error("--replace-ednet-kt1 requires historical dataset mode")
    if args.replace_assist2009 and not args.assist2009:
        parser.error("--replace-assist2009 requires --assist2009")
    output = args.output_dir
    counts = validated_counts(assist2009=args.assist2009, algebra2005=args.algebra2005)
    if args.replace_ednet_kt1 and set(counts["ednet_kt1"]) != set(MODELS):
        raise ValueError("EdNet-KT1 replacement requires all five completed runs")
    # Use one linear scale for all datasets and reserve space for exact labels.
    axis_max = max(max(values.values()) for values in counts.values()) * 1.32
    images = {}
    with plt.style.context("default"):
        for dataset, values in counts.items():
            if args.replace_ednet_kt1 and dataset != "ednet_kt1":
                continue
            fig = make_figure(dataset, values, axis_max)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=320, facecolor="white")
            plt.close(fig)
            images[output / f"{dataset}_parameter_comparison.png"] = buffer.getvalue()
    output.mkdir(parents=True, exist_ok=True)
    for path, content in images.items():
        if path.exists() and path.read_bytes() != content and not (args.replace_assist2009 or args.replace_ednet_kt1):
            raise FileExistsError(f"{path} differs. Use --output-dir with a fresh directory.")
    for path, content in images.items():
        if args.replace_assist2009 or args.replace_ednet_kt1:
            temporary = path.with_suffix(".png.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        elif not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    print(f"Validated {sum(map(len, counts.values()))} counts. {len(images)} PNGs available in {output}.")


if __name__ == "__main__":
    main()
