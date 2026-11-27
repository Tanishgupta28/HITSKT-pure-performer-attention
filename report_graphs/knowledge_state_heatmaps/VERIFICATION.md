# HiTSKT heatmap verification

The three dataset-specific `best_model.pt` files passed their historical SHA-256 checks. Checkpoint epochs, validation ROC-AUCs, training epoch counts, and recorded test ROC-AUCs agree with the corresponding HiTSKT rows in `reports/final_results.csv`. The run used CPU inference through the existing HiTSKT layers with dropout disabled. The model architecture, training artifacts, and benchmark results were left intact.

| Dataset | Checkpoint SHA-256 |
|---|---|
| ASSIST2012 | `7b400654fe6b6d704b9892ed12dc7d8c39ec96d15a37c9fbe7f195cac691023e` |
| Junyi | `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd` |
| EdNet-KT1 | `58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8` |

Junyi was rebuilt from the original authors' byte-identical CSV. The rebuilt Junyi store matched all 11 historical file hashes. The EdNet store also matched all 11 historical file hashes. The ASSIST2012 session-store metadata and the underlying event file matched the benchmark-linked hashes. The original test target counts were 901,241 for ASSIST2012, 2,456,402 for Junyi, and 13,429,870 for EdNet-KT1.

## Numerical and causal checks

Sampled probes were compared with independent calls to the full HiTSKT forward path after collation. Samples include each student's first, middle, and last columns and the first column of each new session encountered in the plot. The checks verify the BOS and shifted correctness tokens and target padding mask. Additional forwards changed the placeholder target label and later response labels, including actual later outcomes when available. The allowable absolute difference for probabilities was 1 × 10⁻⁵.

| Dataset | Direct probe comparisons | Largest direct probability difference | Largest difference after modifying target and future labels |
|---|---:|---:|---:|
| ASSIST2012 | 105 | 2.3842 × 10⁻⁷ | 1.7881 × 10⁻⁷ |
| Junyi | 66 | 1.1921 × 10⁻⁷ | 1.7881 × 10⁻⁷ |
| EdNet-KT1 | 84 | 2.3842 × 10⁻⁷ | 1.1921 × 10⁻⁷ |

The cached Action Encoder EOS vectors also agreed with the direct Action Encoder up to a maximum absolute component difference of 8.5831 × 10⁻⁶ across the three datasets. The retained histories follow the test dataset's session bounds. Across the fifteen windows there were 41 session changes within the plotted sequences. Junyi and EdNet had 221 and 71 deferred columns respectively. Consecutive deferred columns with the same visible prefix produced identical saved scores.

## Selection and file checks

- Two deterministic selection passes returned the same five windows per dataset. All 15 windows met the preferred criteria and yielded 5 × 50 matrices.
- A separate recount from the processed training event shards reproduced the selected representative question IDs and frequencies. Each skill used the same bank in every column.
- All 3,750 matrix values and all local question-level predictions were finite and within [0,1]. Recomputing each row from its saved question-level probabilities reproduced the matrix values exactly.
- Every individual and combined CSV passed exact numerical readback. The arrays handed to Matplotlib matched the saved values. Each publication-ready individual is a byte-identical copy of its dataset's first student figure.
- All 22 public PNGs and 22 local PNGs passed PNG decoding and resolution checks. The render uses a fixed 0–1 `Blues` scale and 330 DPI.
- The repository's HiTSKT, batching, session-store and EdNet preprocessing tests passed, with 26 tests in total.

The full [reproducibility manifest](reproducibility_manifest.json) contains counts by student and SHA-256 hashes for all public PNGs and CSVs. The EdNet causal checks use KT1's estimated group-closure timing. The comparisons exercise the figure inference path, while the benchmark metrics above come from the existing evaluation records.
