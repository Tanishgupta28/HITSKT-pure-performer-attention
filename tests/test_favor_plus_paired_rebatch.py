"""Pure CPU shape/supervision preservation; never optimize or initialize CUDA."""
import copy
import pytest,torch
from ktbench import favor_plus_paired_epochs as p
from ktbench.favor_plus_paired_rebatch import rebatch
from ktbench.data.session_store import RollingSessionDataset
from test_favor_plus_pipeline import synthetic_store


@pytest.mark.parametrize('maximum',[1,2,8])
def test_rebatch_preserves_target_order_labels_and_complete_lengths(tmp_path,maximum):
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    store=synthetic_store(tmp_path/'store');splits={s:RollingSessionDataset(store,s,history_sessions=15) for s in ('train','validation')}
    old={s:{'1' if s=='train' else '0':p.shape_manifest(ds,[list(range(len(ds)))])} for s,ds in splits.items()};before=copy.deepcopy(old)
    new=rebatch(old,splits,maximum)
    assert old==before
    for s,epochs in new.items():
        for e,rows in epochs.items():
            assert all(r['batch_size']<=maximum for r in rows)
            assert sum(r['supervised_targets'] for r in rows)==sum(r['supervised_targets'] for r in old[s][e])
            assert [t for r in rows for t in r['target_sessions']]==[t for r in old[s][e] for t in r['target_sessions']]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('maximum',[0,9,True,1.5])
def test_invalid_common_batch_policy_rejected(maximum):
    with pytest.raises(ValueError):rebatch({}, {},maximum)
