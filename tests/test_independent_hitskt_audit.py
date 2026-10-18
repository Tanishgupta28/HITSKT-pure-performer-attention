"""Independent audit oracles; do not reuse the implementation's attention scan.

These tests establish index-based causality on the supported right-padded batch
contract. They do not establish real-world answer availability or data provenance.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from ktbench.batching import Session, SessionExample, collate_hitskt
from ktbench.metrics import masked_bce_loss, masked_binary_counts
from ktbench.models.hitskt import HiTSKT, HiTSKTConfig
from ktbench.models.performer import CausalLinearAttention


def explicit_attention(module, query, key, value, qm, km):
    """Quadratic causal kernel oracle: independent feature map and mask algebra."""
    b, length, width = query.shape
    h, d = module.heads, module.head_width
    def project(x, layer):
        return F.linear(x, layer.weight).reshape(b, length, h, d).transpose(1, 2)
    def phi(x):
        return torch.where(x >= 0, x + 1, torch.exp(x))
    q = phi(project(query, module.query))
    k = phi(project(key, module.key))
    v = project(value, module.value)
    scores = q @ k.transpose(-1, -2)
    allowed = (torch.arange(length)[None, :] <= torch.arange(length)[:, None])
    weights = scores * (allowed[None, None] & km[:, None, None, :]).to(scores.dtype)
    output = (weights @ v) / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    output = output.transpose(1, 2).reshape(b, length, width)
    return F.linear(output, module.output.weight) * qm[..., None]


@pytest.mark.parametrize('length', [1, 2, 5, 11])
@pytest.mark.parametrize('pattern', ['dense', 'right', 'holes', 'empty'])
def test_attention_values_and_all_gradients_against_explicit_oracle(length, pattern):
    torch.manual_seed(427)
    attention = CausalLinearAttention(12, 3, 0).double().eval()
    inputs = [torch.randn(3, length, 12, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    qm = torch.ones(3, length, dtype=torch.bool)
    km = qm.clone()
    if pattern == 'right':
        for row in range(3):
            km[row, max(0, length-row-1):] = False
            qm[row, max(0, length-row):] = False
    elif pattern == 'holes':
        km[:, 1::2] = False
        qm[1, ::2] = False
        km[2, :1] = False
    elif pattern == 'empty':
        km[:] = False
        qm[1] = False
    actual = attention(*inputs, query_mask=qm, key_mask=km)
    expected = explicit_attention(attention, *inputs, qm, km)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    probe = torch.randn_like(actual)
    variables = inputs + list(attention.parameters())
    a = torch.autograd.grad((actual*probe).sum(), variables, retain_graph=True)
    e = torch.autograd.grad((expected*probe).sum(), variables)
    for ag, eg in zip(a, e):
        torch.testing.assert_close(ag, eg, atol=5e-11, rtol=5e-10)


def test_attention_future_cross_batch_and_masked_key_jacobian_is_zero():
    torch.manual_seed(428)
    attention = CausalLinearAttention(8, 2, 0).double()
    q, k, v = [torch.randn(2, 9, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 1] = False
    result = attention(q, k, v, query_mask=torch.ones_like(mask), key_mask=mask)
    gq, gk, gv = torch.autograd.grad(result[0, 4].sum(), (q,k,v))
    for g in (gq,gk,gv):
        assert torch.count_nonzero(g[1]) == 0
        assert torch.count_nonzero(g[0, 5:]) == 0
    assert torch.count_nonzero(gq[0, :4]) == 0
    assert torch.count_nonzero(gk[0, 1]) == torch.count_nonzero(gv[0, 1]) == 0
    assert torch.count_nonzero(gv[0, :5]) > 0  # positive sensitivity control


@pytest.mark.parametrize('width,heads',[(128,2),(256,4)])
def test_attention_actual_checkpoint_dimensions_against_explicit_reference(width,heads):
    torch.manual_seed(432)
    module=CausalLinearAttention(width,heads,0).double()
    inputs=[torch.randn(2,7,width,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    qm=torch.ones(2,7,dtype=torch.bool);qm[1,-1]=False
    km=qm.clone();km[0,1::2]=False
    actual=module(*inputs,query_mask=qm,key_mask=km)
    expected=explicit_attention(module,*inputs,qm,km)
    torch.testing.assert_close(actual,expected,atol=2e-11,rtol=2e-10)
    probe=torch.randn_like(actual);variables=inputs+list(module.parameters())
    ag=torch.autograd.grad((actual*probe).sum(),variables,retain_graph=True)
    eg=torch.autograd.grad((expected*probe).sum(),variables)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,atol=5e-11,rtol=5e-10)


@pytest.mark.parametrize('length',[939,3925,13081])
def test_maximum_length_attention_selected_outputs_and_gradients(length):
    """Dataset maxima plus EOS; explicit selected rows avoid an L-square matrix."""
    torch.manual_seed(431)
    attention=CausalLinearAttention(12,3,0).double()
    inputs=[torch.randn(2,length,12,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    qm=torch.ones(2,length,dtype=torch.bool);qm[0,-5:]=False
    km=torch.ones_like(qm);km[1,1::7]=False;km[0,-8:]=False
    positions=torch.tensor([0,1,31,255,length//2,length-1])
    actual=attention(*inputs,query_mask=qm,key_mask=km)[:,positions]
    def project(x,layer):return F.linear(x,layer.weight).reshape(2,length,3,4).transpose(1,2)
    qraw=project(inputs[0],attention.query)[:,:,positions]
    kraw=project(inputs[1],attention.key)
    q=torch.where(qraw>=0,qraw+1,qraw.exp())
    k=torch.where(kraw>=0,kraw+1,kraw.exp())
    value=project(inputs[2],attention.value)
    scores=q@k.transpose(-1,-2)
    allowed=torch.arange(length)[None,:]<=positions[:,None]
    weights=scores*(allowed[None,None]&km[:,None,None,:])
    expected=(weights@value)/weights.sum(-1,keepdim=True).clamp_min(1e-6)
    expected=F.linear(expected.transpose(1,2).reshape(2,len(positions),12),attention.output.weight)*qm[:,positions,None]
    torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-9)
    probe=torch.randn_like(actual)
    variables=inputs+list(attention.parameters())
    ag=torch.autograd.grad((actual*probe).sum(),variables,retain_graph=True)
    eg=torch.autograd.grad((expected*probe).sum(),variables)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,atol=5e-10,rtol=5e-9)


def session(length, offset=0):
    return Session([(i+offset)%29+1 for i in range(length)],
                   [(i+offset)%7+1 for i in range(length)],
                   [(i+offset)%2 for i in range(length)])


def example(length, histories=3, offset=0):
    return SessionExample([session(1+(i*3+offset)%11, offset+i) for i in range(histories)],
                          session(length, offset+2), 2)


def collate(examples):
    return collate_hitskt(examples, num_questions=30, num_skills=8)


def model(layers=1):
    torch.manual_seed(429)
    return HiTSKT(HiTSKTConfig(30,8,width=12,heads=3,feedforward_width=19,
                  action_layers=layers,session_layers=layers,correct_layers=layers,
                  decoder_layers=layers,dropout=.1)).double().eval()


@pytest.mark.parametrize('length,histories', [(1,1),(2,3),(7,15),(17,4)])
@pytest.mark.parametrize('layers', [1,2])
def test_recollated_raw_own_and_future_responses_cannot_change_earlier_predictions(length,histories,layers):
    net = model(layers)
    examples = [example(length,histories),example(3,1,3),example(length+5,5,7)]
    batch = collate(examples)
    with torch.no_grad():
        original = net(batch)
        for position in range(length):
            labels = list(examples[0].target.correct)
            labels[position:] = [1-y for y in labels[position:]]
            changed_example = replace(examples[0], target=replace(examples[0].target,correct=labels))
            changed_batch = collate([changed_example,*examples[1:]])
            changed = net(changed_batch)
            torch.testing.assert_close(original[0,:position+1],changed[0,:position+1],atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(original[1:],changed[1:],atol=1e-12,rtol=1e-12)
        # Labels are supervision only, never an input to any model stage.
        tampered = deepcopy(batch)
        tampered.target_labels[:] = 99
        torch.testing.assert_close(original,net(tampered),atol=0,rtol=0)
    assert batch.target_correct_input[0,0] == 4
    assert batch.target_correct_input[0,1:length+1].tolist() == list(examples[0].target.correct)
    assert batch.target_metric_mask[0].sum() == length
    assert batch.target_attention_mask[0].sum() == length+1


@pytest.mark.parametrize('length', [1,4,13])
def test_future_questions_eos_and_finite_padding_cannot_contaminate_real_targets(length):
    net = model()
    base = example(length,3)
    companion = example(length+12,9,4)
    single, batched = collate([base]), collate([base,companion])
    with torch.no_grad():
        original = net(batched)
        torch.testing.assert_close(net(single)[0,:length],original[0,:length],atol=2e-11,rtol=2e-10)
        changed = deepcopy(batched)
        for key,mask in [('history_question','history_mask'),('history_skill','history_mask'),
                         ('history_correct','history_mask'),('target_question','target_attention_mask'),
                         ('target_skill','target_attention_mask'),('target_correct_input','target_attention_mask')]:
            getattr(changed,key)[~getattr(changed,mask)] = 1
        torch.testing.assert_close(original[batched.target_metric_mask],net(changed)[batched.target_metric_mask],atol=1e-12,rtol=1e-12)
        changed = deepcopy(batched)
        changed.target_question[0,length:] = 5
        changed.target_skill[0,length:] = 3
        changed.target_correct_input[0,length:] = 1-changed.target_correct_input[0,length]
        # Keep padded input IDs valid; only flip EOS's last-response input.
        changed.target_correct_input[0,length+1:] = 0
        torch.testing.assert_close(original[0,:length],net(changed)[0,:length],atol=1e-12,rtol=1e-12)
        if length > 1:
            changed=deepcopy(batched)
            changed.target_question[0,1:length] = 8
            changed.target_skill[0,1:length] = 4
            torch.testing.assert_close(original[0,:1],net(changed)[0,:1],atol=1e-12,rtol=1e-12)


def test_each_encoder_and_decoder_block_obeys_its_causal_prefix():
    net = model(2)
    batch = collate([example(9,4),example(12,2,5)])
    layer = net.action_encoder.layers[0]
    captured=[]
    handle=layer.register_forward_hook(lambda m,i,o: captured.append(o.detach().clone()))
    with torch.no_grad():
        net.action_encoder(batch)
        changed=deepcopy(batch); changed.history_correct[1,1] = 1-changed.history_correct[1,1]
        net.action_encoder(changed)
    handle.remove()
    torch.testing.assert_close(captured[0][1,:1],captured[1][1,:1],atol=0,rtol=0)
    with torch.no_grad():
        action=net.action_encoder(batch)
        history=net.session_encoder(action,batch)
        # Test session stack at every stage using a synthetic dense session sequence.
        for layer in net.session_encoder.layers:
            x=torch.randn(2,6,12,dtype=torch.float64); mask=torch.ones(2,6,dtype=torch.bool)
            changed=x.clone(); changed[0,3:]+=20
            torch.testing.assert_close(layer(x,mask)[0,:3],layer(changed,mask)[0,:3],atol=1e-12,rtol=1e-12)
        memory=net.correct_padding_encoder(history,batch)
        changed=deepcopy(batch); changed.target_correct_input[0,5:] = 0
        torch.testing.assert_close(memory[0,:5],net.correct_padding_encoder(history,changed)[0,:5],atol=1e-12,rtol=1e-12)
        changed_memory=memory.clone(); changed_memory[0,5:]+=20
        torch.testing.assert_close(net.decoder(memory,batch)[0,:5],net.decoder(changed_memory,batch)[0,:5],atol=1e-12,rtol=1e-12)


def test_full_model_response_embedding_jacobian_excludes_own_future_and_other_batch():
    net=model(2)
    batch=collate([example(9,3),example(5,1,2)])
    embeddings=[]
    def retain(module,inputs,output):
        output.retain_grad(); embeddings.append(output)
    handle=net.correct_padding_encoder.correct.register_forward_hook(retain)
    net(batch)[0,4].backward()
    handle.remove()
    grad=embeddings[0].grad
    assert torch.count_nonzero(grad[0,5:]) == 0  # own y4 enters only at input5
    assert torch.count_nonzero(grad[1]) == 0
    assert torch.count_nonzero(grad[0,:5]) > 0


def test_training_mode_causality_with_the_same_dropout_randomness():
    net=model(2).train()
    item=example(13,4)
    labels=list(item.target.correct);labels[6:]=[1-y for y in labels[6:]]
    changed=replace(item,target=replace(item.target,correct=labels))
    batch=collate([item,example(8,2,5)])
    changed_batch=collate([changed,example(8,2,5)])
    torch.manual_seed(991)
    original=net(batch)
    torch.manual_seed(991)
    altered=net(changed_batch)
    torch.testing.assert_close(original[0,:7],altered[0,:7],atol=0,rtol=0)
    torch.testing.assert_close(original[1],altered[1],atol=0,rtol=0)


def test_pad_and_eos_loss_counts_and_gradients_against_manual_real_target_oracle():
    batch=collate([example(1),example(9)])
    torch.manual_seed(430)
    logits=torch.randn(batch.target_labels.shape,dtype=torch.float64,requires_grad=True)
    mask=batch.target_metric_mask
    truth=batch.target_labels[mask].double(); real=logits[mask]
    reference=(torch.clamp_min(real,0)-truth*real+torch.log1p(torch.exp(-real.abs()))).mean()
    loss=masked_bce_loss(logits,batch.target_labels,mask)
    torch.testing.assert_close(loss,reference,atol=1e-14,rtol=1e-14)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    counts=masked_binary_counts(logits,batch.target_labels,mask)
    predicted=real.detach()>=0
    assert counts['targets'] == 10
    assert counts['correct'] == int((predicted==truth.bool()).sum())
    changed=logits.detach().clone(); changed[~mask]=1e9
    labels=batch.target_labels.clone(); labels[~mask]=77
    torch.testing.assert_close(loss.detach(),masked_bce_loss(changed,labels,mask),atol=0,rtol=0)


def test_independent_metric_oracle_on_hand_calculated_tied_scores():
    import math
    import numpy as np
    from scripts.audit_hitskt_replay import independent_metrics
    p=np.array([.1,.5,.5,.9],dtype=np.float64)
    y=np.array([0,1,0,1],dtype=np.uint8)
    z=np.log(p/(1-p))
    metrics=independent_metrics(p,y,z)
    assert metrics['roc_auc'] == .875
    assert metrics['accuracy'] == .75
    assert metrics['precision'] == 2/3
    assert metrics['recall'] == 1
    assert metrics['f1'] == .8
    assert metrics['mse'] == pytest.approx(.13)
    assert metrics['bce_float64'] == pytest.approx(-(2*math.log(.9)+2*math.log(.5))/4)


def toy_store():
    """Two independent students, each with 3/1/1 whole-session splits."""
    import numpy as np
    from types import SimpleNamespace, MethodType
    from ktbench.data.session_store import SessionStore
    lengths=np.array([1,3,2,5,4,2,1,4,3,6])
    offsets=np.r_[0,np.cumsum(lengths)]
    n=int(offsets[-1])
    store=SimpleNamespace(question=np.arange(n)%29+1,skill=np.arange(n)%7+1,
        correct=(np.arange(n)%2).astype(np.uint8),timestamp_ms=np.arange(n)*1000,
        session_offsets=offsets,student_offsets=np.array([0,5,10]),
        session_student=np.repeat([1,2],5),session_number=np.tile(np.arange(1,6),2),
        session_split=np.tile([0,0,0,1,2],2))
    store.session=MethodType(SessionStore.session,store)
    return store


def test_future_session_and_other_student_changes_do_not_affect_earlier_predictions():
    from ktbench.data.session_store import RollingSessionDataset
    store=toy_store()
    ds=RollingSessionDataset(store,'validation',history_sessions=15)
    assert ds.targets.tolist()==[3,8]
    assert ds._history_bounds(3)==(0,3) and ds._history_bounds(8)==(5,8)
    net=model(2)
    with torch.no_grad():
        original=net(collate([ds[0],ds[1]]))
        # First student's future test session and every event of student2.
        start=int(store.session_offsets[4]); store.correct[start:]=1-store.correct[start:]
        changed=net(collate([ds[0],ds[1]]))
        torch.testing.assert_close(original[0,:5],changed[0,:5],atol=0,rtol=0)
        second_before=changed[1,:3].clone()
        store.correct[:int(store.session_offsets[5])]=1-store.correct[:int(store.session_offsets[5])]
        second_after=net(collate([ds[0],ds[1]]))[1,:3]
        torch.testing.assert_close(second_before,second_after,atol=0,rtol=0)


def test_rolling_baseline_target_identity_and_future_exclusion_with_independent_store():
    import numpy as np
    from ktbench.data.rolling_targets import RollingTargetDataset
    store=toy_store()
    for split,name in enumerate(['train','validation','test']):
        ds=RollingTargetDataset(store,name,history_length=199)
        actual=[ds[i].target_event_id for i in range(len(ds))]
        wanted=[event for session in range(10) if store.session_split[session]==split
                      for event in range(int(store.session_offsets[session]),int(store.session_offsets[session+1]))]
        assert actual==wanted and len(set(actual))==len(actual)
        for i,event in enumerate(actual):
            item=ds[i]
            student_start=int(store.session_offsets[store.student_offsets[item.student_id-1]])
            assert list(item.history_correct)==store.correct[student_start:event].tolist()
            assert item.target_correct==store.correct[event]
            assert np.all(np.asarray(item.history_timestamp_ms)<item.target_timestamp_ms)


@pytest.mark.parametrize('name',['dkt','dkvmn','sakt','rkt'])
def test_baseline_forward_ignores_target_labels_and_masked_padding(name):
    import numpy as np
    from ktbench.baseline_registry import make_baseline_model
    from ktbench.data.rolling_targets import RollingTarget,collate_rolling_targets
    from ktbench.models.rkt import PaperFaithfulRKT,RKTConfig
    from ktbench.rkt.data import RKTBatch
    examples=[]
    for row,length in enumerate([0,1,5]):
        examples.append(RollingTarget(1,list(range(1,length+1)),[i%7+1 for i in range(length)],
            [i%2 for i in range(length)],[i*1000 for i in range(length)],2,3,row%2,20000,row,2))
    batch=collate_rolling_targets(examples)
    if name=='rkt':
        batch=RKTBatch(student_id=batch.student_id,history_question=batch.history_question,
            history_correct=batch.history_correct,history_position=torch.arange(1,6).expand(3,-1),
            history_mask=batch.history_mask,delta_hours=torch.ones(3,5),phi=torch.zeros(3,5),
            target_question=batch.target_question,target_labels=batch.target_labels,
            target_event_id=batch.target_event_id,split=batch.split)
        net=PaperFaithfulRKT(RKTConfig(30,2),np.ones(3)).eval()
    else:
        net=make_baseline_model(name,num_questions=30,num_skills=8).eval()
    with torch.no_grad():
        original=net(batch)
        changed=deepcopy(batch);changed.target_labels[:]=77
        torch.testing.assert_close(original,net(changed),atol=0,rtol=0)
        changed.history_question[~changed.history_mask]=1
        if hasattr(changed,'history_skill'):changed.history_skill[~changed.history_mask]=1
        changed.history_correct[~changed.history_mask]=1
        torch.testing.assert_close(original,net(changed),atol=2e-7,rtol=2e-6)
