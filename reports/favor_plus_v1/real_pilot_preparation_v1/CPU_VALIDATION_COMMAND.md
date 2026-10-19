# CPU-only validation command

Executed from `/workspace/capstone`. The new test module also has a per-test
CUDA prohibition fixture. This wrapper applies the prohibition across collection
and the selected legacy cases. No real-data model optimization is performed.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 223 nice -n 10 python - <<'PY'
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CPU validation must not call a CUDA API')
for name in [
    'is_available', 'is_initialized', 'init', '_lazy_init', 'device_count',
    'get_device_properties', 'get_device_name', 'manual_seed', 'manual_seed_all',
    'get_rng_state', 'get_rng_state_all', 'synchronize', 'memory_allocated',
    'max_memory_allocated', 'max_memory_reserved', 'reset_peak_memory_stats',
    'set_per_process_memory_fraction',
]:
    setattr(torch.cuda, name, forbidden)
import pytest
raise SystemExit(pytest.main([
    '-q', 'tests/test_favor_plus_real_pilot.py',
    'tests/test_favor_quality_diagnostics.py',
    '-k', 'not pipeline_selects_validation_peak',
    '--junitxml=reports/favor_plus_v1/real_pilot_preparation_v1/final_cpu_validation_v3.xml',
]))
PY
```

The recorded execution redirected stdout/stderr to `final_cpu_validation_v3.log`.
Use fresh output filenames on a repeat so the captured evidence remains intact.
Actual-data preparation was separately run with the same CUDA traps plus raising
replacements for `HiTSKTFavorPlus` and `torch.optim.Adam`; its log and prepared
plan record zero model forwards/updates. It selected/collated only train targets.
