# Archived commands for the completed25→50 campaign

Evidence only; DO NOT rerun completed/failed run directories or start new work
without a new user instruction. All six CUDA invocations completed successfully
in this exact order: stage30,verify-cuda30,stage40,verify-cuda40,stage50,verify-cuda50.
Each was a separate fresh process, with review between milestones.

Common environment/prefix:

```bash
CUDA_VISIBLE_DEVICES=MIG-d2fcfd79-720e-5218-82c9-0a132be68920 \
PYTHONPATH=. PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 223 nice -n 10 python scripts/assist50_from25_favor_plus_shared.py
```

Arguments for each invocation (ACTION/UPDATE correspond to the order above):

```text
ACTION --plan experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1/prepared_plan.json
--shared-record reports/favor_plus_v1/shared50_from25_20261023_v1/shared_execution_record.json
--update UPDATE --execute-cuda
```

Each stdout/stderr is saved in the run root as stage_NNN.console.log or
verify_NNN.console.log. The controller itself records successful fresh admission,
runtime, every boundary, update, checkpoint receipt, diagnostics and recovery.
No endpoint>50, full training, optimizer51 or automatic retry was executed.

CPU tests/verification used CUDA_VISIBLE_DEVICES='',PYTHONPATH=.,OMP/MKL/
OPENBLAS_NUM_THREADS=1,taskset223,nice10. The exact pytest invocations are
recorded by the predeclared/post-run suites and logs; XML identifies every case.
Saved-evidence helpers verify_results.py and review_saved_prefixes.py never
initialize CUDA or update parameters and create outputs exclusively (no overwrite).
