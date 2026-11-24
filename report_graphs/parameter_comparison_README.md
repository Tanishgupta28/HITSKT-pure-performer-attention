# Trainable parameters by model and dataset

The three horizontal bar charts compare the historical, completed benchmark runs on [source commit `e7e99219b3b00e023014037c5c9c3023c1add981`](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/commit/e7e99219b3b00e023014037c5c9c3023c1add981). All panels use the same linear scale starting at zero. The longest bar appears at the top of each panel, and dark blue identifies HiTSKT. Each bar label gives an abbreviated count followed by the exact integer in parentheses.

| Model | [ASSIST2017](assist2017_parameter_comparison.png) | [Junyi](junyi_parameter_comparison.png) | [EdNet-KT1](ednet_kt1_parameter_comparison.png) |
| --- | ---: | ---: | ---: |
| DKT | 59,751 | 424,434 | Incomplete |
| DKVMN | 56,551 | 411,301 | 404,801 |
| SAKT | 528,001 | 2,917,101 | 1,668,401 |
| RKT | 239,343 | 1,706,028 | 995,207 |
| HiTSKT | 6,931,457 | 8,271,105 | 5,074,433 |

## Sources and validation

The bar widths come directly from the `trainable_parameters` column in [reports/final_results.csv](../reports/final_results.csv). For each row, the script follows its `experiment_directory`, checks the dataset and model names, and compares the exact integer with `trainable_parameters` in both `config.json` and `final_results.json`. It requires a final status of `complete` and one indexed run per included dataset and model. The paths take the form `experiments/<model>/<dataset>/full/`, using lowercase model names and dataset keys `assist2017`, `junyi`, and `ednet_kt1`.

The DKT experiment on EdNet-KT1 was incomplete at the source commit and has no completed entry in the index. Its bar is omitted. Counts are specific to each recorded run. Generating the figures requires no assumptions about question or skill counts and involves no training or inference.

## Reproduce

The figures were generated using Python 3.12.6 and Matplotlib 3.9.2 with the Agg backend. From a checkout containing the script and benchmark files, run these commands in an activated Python environment at the repository root.

```sh
python -m pip install matplotlib==3.9.2
python report_graphs/generate_parameter_comparison.py
```

The script validates all 14 counts before rendering. It writes three 320-DPI PNGs to `report_graphs/` and accepts existing byte-identical figures. For an independent rendering in a fresh directory, use `python report_graphs/generate_parameter_comparison.py --output-dir parameter-figures`. Existing files are protected from replacement.
