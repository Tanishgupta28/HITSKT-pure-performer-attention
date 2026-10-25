"""Read-only actual-data collation checks; CPU-only, no optimizer/inference."""
import json
from pathlib import Path
import torch
from ktbench import favor_plus_micro_profile as policy
from ktbench import favor_plus_paired_epochs as pair
from ktbench import favor_plus_microbatch as micro

assert torch.version.cuda is None and not torch.cuda.is_initialized()
path = Path('experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_logical8_micro_20261025_v3/prepared_plan.json')
plan, old, splits = policy.load_plan(path)
checked = 0
for split, row in [('train',r) for r in plan['profile_design']['worst_training']]+[('validation',r) for r in plan['profile_design']['worst_validation_cases']]:
    full = pair.make_batch(splits[split],row)
    for physical in (1,2,4):
        offset = 0
        for part in micro.partition(splits[split],row,physical):
            batch = pair.make_batch(splits[split],part)
            for j in range(part['batch_size']):
                owner = offset+j
                length = int(full.target_attention_mask[owner].sum())
                for name in ('target_attention_mask','target_metric_mask','target_question','target_skill','target_correct_input','target_labels'):
                    assert torch.equal(getattr(full,name)[owner,:length],getattr(batch,name)[j,:length])
                originals = torch.nonzero(full.history_owner==owner).flatten()
                micros = torch.nonzero(batch.history_owner==j).flatten()
                assert len(originals)==len(micros)
                for oi, ni in zip(originals,micros):
                    length = int(full.history_mask[oi].sum())
                    assert int(batch.history_mask[ni].sum())==length
                    for name in ('history_question','history_skill','history_correct','history_mask'):
                        assert torch.equal(getattr(full,name)[oi,:length],getattr(batch,name)[ni,:length])
                    assert full.history_order[oi]==batch.history_order[ni]
            offset += part['batch_size']
            checked += 1
        assert offset==row['batch_size']
identities = {arm:{k:old['contracts'][arm][k] for k in ('fresh_learned_sha256','fresh_cpu_rng_sha256','seed',
    'validation_ids_sha256','validation_labels_sha256','target_counts')} for arm in pair.ARMS}
assert identities['orf128']==identities['orf512']
for physical in plan['accounting'].values():
    assert physical['logical_updates']==7499
    assert all(e['supervised_targets']==566107 for e in physical['epochs'].values())
result = dict(status='PASS_REAL_DATA_MASK_ORDER_CONTEXT_AND_MATCHED_INITIALIZATION',plan_sha256=pair.sha256(path),
    actual_microbatches_collated=checked,physical_choices_cpu_checked=[1,2,4],
    all_original_valid_tokens_and_shifted_labels_equal=True,
    data_identity_files_verified=len(old['contracts']['orf128']['dataset_identity']['files']),identities=identities,
    cuda_initialized=torch.cuda.is_initialized(),real_dataset_optimizer_updates=0,test_evaluations=0)
output = Path(__file__).parent/'ACTUAL_DATA_COVERAGE_AND_MASKS.json'
assert not output.exists()
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='identities'},indent=2))
