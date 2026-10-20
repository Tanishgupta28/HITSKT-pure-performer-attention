"""Fresh CPU-only load of the completed pilot; no forward or optimizer step."""
from pathlib import Path
import hashlib
import json

import torch


def forbidden(*args, **kwargs):
    raise AssertionError('Fresh CPU checkpoint verification cannot use CUDA')


for name in ('is_available', 'init', '_lazy_init', 'manual_seed', 'manual_seed_all',
             'get_device_properties', 'get_rng_state', 'get_rng_state_all'):
    setattr(torch.cuda, name, forbidden)

from ktbench.favor_plus_pilot_diagnostics import finite_tree

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1'
checkpoint = RUN/'pilot_model.pt'
expected = json.loads((RUN/'pilot_manifest.json').read_text())
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha == expected['checkpoint_sha256']
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert state['variant'] == 'hitskt_favor_plus_v1' and state['format_version'] == 2
assert state['optimizer_updates'] == 5
assert finite_tree(state['state_dict']) and finite_tree(state['optimizer_state_dict'])
steps = sorted({int(x['step']) for x in state['optimizer_state_dict']['state'].values()})
assert steps == [5]
assert all(state['pilot_config']['source']['source_sha256'][name] == digest
           for name, digest in state['pilot_config']['prepared_plan']['source']['source_sha256'].items())
result = dict(status='PASS', checkpoint_sha256=sha, checkpoint_bytes=checkpoint.stat().st_size,
    format_version=2, variant=state['variant'], optimizer_updates=5, adam_step_values=steps,
    model_and_optimizer_finite=True, rng_state_preserved='rng_state' in state,
    model_tensor_count=len(state['state_dict']), adam_parameter_states=len(state['optimizer_state_dict']['state']),
    cuda_calls=0, model_forwards=0, additional_optimizer_updates=0,
    scope='Fresh-process payload integrity only; exact CUDA-logit/optimizer/RNG roundtrip was separately checked in the pilot')
path = Path(__file__).with_name('fresh_checkpoint_cpu_verification.json')
with path.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
