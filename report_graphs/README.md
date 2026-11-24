# Historical HiTSKT benchmark figures

Six figures compare epoch-wise validation performance across 14 completed experiments on `main`. Each marker represents one recorded validation epoch. Each line ends at that model's last completed epoch. All figures are 2560 × 1536 pixels at 320 DPI.

| Dataset | Validation loss by epoch (lower is better) | Validation ROC-AUC by epoch (higher is better) |
| --- | --- | --- |
| ASSIST2017 | [Loss figure](assist2017_loss_vs_epoch.png) | [ROC-AUC figure](assist2017_validation_auc_vs_epoch.png) |
| Junyi | [Loss figure](junyi_loss_vs_epoch.png) | [ROC-AUC figure](junyi_validation_auc_vs_epoch.png) |
| EdNet-KT1 | [Loss figure](ednet_kt1_loss_vs_epoch.png) | [ROC-AUC figure](ednet_kt1_validation_auc_vs_epoch.png) |

## Reading the comparison

These are single-seed results (seed 42). Model configurations and history lengths differ. The benchmark reports history lengths of 199 for DKT and DKVMN, 99 for SAKT, 49 for RKT, and 15 for HiTSKT. HiTSKT's history is measured in sessions. The curves describe the recorded runs and provide no estimate of between-seed variability. Each panel scales its vertical axis to its observed values.

DKT on EdNet-KT1 was incomplete at the source commit. It has a partial metrics log, no completed row in the benchmark index, and no final result JSON. Both EdNet-KT1 figures therefore contain DKVMN, SAKT, RKT and HiTSKT.

## Data provenance

Source commit on `main` is [`5492344e320a5fbbfec9a7f180f09bc452d5a910`](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/commit/5492344e320a5fbbfec9a7f180f09bc452d5a910). The experiment index is [`reports/final_results.csv`](../reports/final_results.csv). The links below identify the three additional source files for each included run, relative to the repository root.

| Dataset | Model | Completed epochs | Epoch observations | Completion and best epoch | Run configuration |
| --- | --- | ---: | --- | --- | --- |
| ASSIST2017 | DKT | 17 | [metrics.csv](../experiments/dkt/assist2017/full/metrics.csv) | [final_results.json](../experiments/dkt/assist2017/full/final_results.json) | [config.json](../experiments/dkt/assist2017/full/config.json) |
| ASSIST2017 | DKVMN | 23 | [metrics.csv](../experiments/dkvmn/assist2017/full/metrics.csv) | [final_results.json](../experiments/dkvmn/assist2017/full/final_results.json) | [config.json](../experiments/dkvmn/assist2017/full/config.json) |
| ASSIST2017 | SAKT | 35 | [metrics.csv](../experiments/sakt/assist2017/full/metrics.csv) | [final_results.json](../experiments/sakt/assist2017/full/final_results.json) | [config.json](../experiments/sakt/assist2017/full/config.json) |
| ASSIST2017 | RKT | 19 | [metrics.csv](../experiments/rkt/assist2017/full/metrics.csv) | [final_results.json](../experiments/rkt/assist2017/full/final_results.json) | [config.json](../experiments/rkt/assist2017/full/config.json) |
| ASSIST2017 | HiTSKT | 45 | [metrics.csv](../experiments/hitskt/assist2017/full/metrics.csv) | [final_results.json](../experiments/hitskt/assist2017/full/final_results.json) | [config.json](../experiments/hitskt/assist2017/full/config.json) |
| Junyi | DKT | 8 | [metrics.csv](../experiments/dkt/junyi/full/metrics.csv) | [final_results.json](../experiments/dkt/junyi/full/final_results.json) | [config.json](../experiments/dkt/junyi/full/config.json) |
| Junyi | DKVMN | 6 | [metrics.csv](../experiments/dkvmn/junyi/full/metrics.csv) | [final_results.json](../experiments/dkvmn/junyi/full/final_results.json) | [config.json](../experiments/dkvmn/junyi/full/config.json) |
| Junyi | SAKT | 18 | [metrics.csv](../experiments/sakt/junyi/full/metrics.csv) | [final_results.json](../experiments/sakt/junyi/full/final_results.json) | [config.json](../experiments/sakt/junyi/full/config.json) |
| Junyi | RKT | 13 | [metrics.csv](../experiments/rkt/junyi/full/metrics.csv) | [final_results.json](../experiments/rkt/junyi/full/final_results.json) | [config.json](../experiments/rkt/junyi/full/config.json) |
| Junyi | HiTSKT | 37 | [metrics.csv](../experiments/hitskt/junyi/full/metrics.csv) | [final_results.json](../experiments/hitskt/junyi/full/final_results.json) | [config.json](../experiments/hitskt/junyi/full/config.json) |
| EdNet-KT1 | DKVMN | 6 | [metrics.csv](../experiments/dkvmn/ednet_kt1/full/metrics.csv) | [final_results.json](../experiments/dkvmn/ednet_kt1/full/final_results.json) | [config.json](../experiments/dkvmn/ednet_kt1/full/config.json) |
| EdNet-KT1 | SAKT | 18 | [metrics.csv](../experiments/sakt/ednet_kt1/full/metrics.csv) | [final_results.json](../experiments/sakt/ednet_kt1/full/final_results.json) | [config.json](../experiments/sakt/ednet_kt1/full/config.json) |
| EdNet-KT1 | RKT | 6 | [metrics.csv](../experiments/rkt/ednet_kt1/full/metrics.csv) | [final_results.json](../experiments/rkt/ednet_kt1/full/final_results.json) | [config.json](../experiments/rkt/ednet_kt1/full/config.json) |
| EdNet-KT1 | HiTSKT | 35 | [metrics.csv](../experiments/hitskt/ednet_kt1/full/metrics.csv) | [final_results.json](../experiments/hitskt/ednet_kt1/full/final_results.json) | [config.json](../experiments/hitskt/ednet_kt1/full/config.json) |

The script checks model and dataset identity, run directory, completion status, seed, completed epoch count, and best epoch and AUC across the index and JSON files. It reads only the `validation` split for plotting, requires one observation for every epoch from 1 through the recorded count, and checks metric bounds and finiteness. It recomputes the highest validation AUC and its first epoch from each CSV before creating any figure. All 286 validation records contribute one loss marker and one ROC-AUC marker each. No synthetic data, smoothing, interpolation, extrapolation or model retraining was used.

## Reproduce the figures

Use a checkout containing this script and the source files listed above. The figures were generated with CPython 3.12.6 and Matplotlib 3.9.2, using the Agg backend and Matplotlib's bundled DejaVu Sans font. Matplotlib and its dependencies are the only external packages required.

From the repository root, in a Python environment with Matplotlib installed, run

```sh
python report_graphs/generate_comparison_graphs.py
```

To install the plotting dependency in an activated virtual environment, run `python -m pip install matplotlib==3.9.2`. The script locates its inputs relative to its own file and saves the six PNGs beside it. Identical existing images are accepted. To render into a fresh directory when comparing across environments or library versions, run

```sh
python report_graphs/generate_comparison_graphs.py --output-dir comparison-output
```
