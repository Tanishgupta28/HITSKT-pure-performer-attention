from pathlib import Path

import torch

from ktbench.batching import Session, SessionExample, collate_hitskt
from ktbench.metrics import masked_bce_loss, masked_binary_counts
from ktbench.models import HiTSKT, HiTSKTConfig
from ktbench.models.performer import CausalLinearAttention


def _session(question_ids: list[int], correct: list[int] | None = None) -> Session:
    labels = correct if correct is not None else [index % 2 for index in range(len(question_ids))]
    return Session(question_ids, [(question % 4) + 1 for question in question_ids], labels)


def _config() -> HiTSKTConfig:
    return HiTSKTConfig(
        num_questions=20,
        num_skills=5,
        width=16,
        heads=2,
        feedforward_width=32,
        dropout=0.0,
    )


def _example() -> SessionExample:
    return SessionExample(
        history=[_session([1, 2]), _session([3, 4, 5])],
        target=_session([6, 7, 8, 9], [1, 0, 1, 1]),
        split=0,
    )


def test_linear_attention_is_causal_and_pad_safe() -> None:
    torch.manual_seed(1)
    attention = CausalLinearAttention(width=8, heads=2, dropout=0.0).eval()
    values = torch.randn(1, 6, 8)
    mask = torch.tensor([[True, True, True, True, False, False]])
    baseline = attention(values, values, values, query_mask=mask, key_mask=mask)

    changed_future = values.clone()
    changed_future[:, 3] += 1000
    causal = attention(
        changed_future, changed_future, changed_future, query_mask=mask, key_mask=mask
    )
    assert torch.allclose(baseline[:, :3], causal[:, :3], atol=1e-5, rtol=1e-5)

    changed_pad = values.clone()
    changed_pad[:, 4:] -= 1000
    padded = attention(changed_pad, changed_pad, changed_pad, query_mask=mask, key_mask=mask)
    assert torch.allclose(baseline[:, :4], padded[:, :4], atol=1e-5, rtol=1e-5)
    assert padded[:, 4:].eq(0).all()


def test_hitskt_dynamic_padding_does_not_change_valid_logits() -> None:
    torch.manual_seed(2)
    model = HiTSKT(_config()).eval()
    example = _example()
    single = collate_hitskt([example], num_questions=20, num_skills=5)
    companion = SessionExample(
        history=[_session(list(range(10, 19)))],
        target=_session([1, 2, 3, 4, 5, 6, 7]),
        split=0,
    )
    padded = collate_hitskt([example, companion], num_questions=20, num_skills=5)

    with torch.no_grad():
        single_logits = model(single)[0, :4]
        padded_logits = model(padded)[0, :4]
    assert torch.allclose(single_logits, padded_logits, atol=2e-5, rtol=2e-5)


def test_hitskt_target_causality() -> None:
    torch.manual_seed(3)
    model = HiTSKT(_config()).eval()
    batch = collate_hitskt([_example()], num_questions=20, num_skills=5)
    with torch.no_grad():
        baseline = model(batch)
    # Input correctness at index 3 is information from target index 2. It must
    # not affect predictions at indices 0..2.
    batch.target_correct_input[0, 3] = 1 - batch.target_correct_input[0, 3]
    with torch.no_grad():
        changed = model(batch)
    assert torch.allclose(baseline[:, :3], changed[:, :3], atol=1e-5, rtol=1e-5)


def test_no_pad_or_eos_contamination_of_loss_and_metrics() -> None:
    batch = collate_hitskt(
        [_example(), SessionExample([_session([1])], _session([2], [0]), 0)],
        num_questions=20,
        num_skills=5,
    )
    logits = torch.zeros_like(batch.target_labels, dtype=torch.float32)
    base_loss = masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
    base_counts = masked_binary_counts(logits, batch.target_labels, batch.target_metric_mask)

    contaminated_logits = logits.clone()
    contaminated_labels = batch.target_labels.clone()
    contaminated_logits[~batch.target_metric_mask] = 1_000_000
    contaminated_labels[~batch.target_metric_mask] = 99
    assert torch.equal(
        base_loss,
        masked_bce_loss(contaminated_logits, contaminated_labels, batch.target_metric_mask),
    )
    assert base_counts == masked_binary_counts(
        contaminated_logits, contaminated_labels, batch.target_metric_mask
    )
    assert base_counts["targets"] == 5


def test_forward_backward_shapes_and_all_stages_receive_gradients() -> None:
    torch.manual_seed(4)
    model = HiTSKT(_config()).train()
    batch = collate_hitskt(
        [_example(), SessionExample([_session([2])], _session([3, 4], [0, 1]), 0)],
        num_questions=20,
        num_skills=5,
    )
    logits = model(batch)
    assert logits.shape == batch.target_labels.shape == (2, 5)
    loss = masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
    loss.backward()

    for stage in (
        model.action_encoder,
        model.session_encoder,
        model.correct_padding_encoder,
        model.decoder,
        model.prediction,
    ):
        assert any(parameter.grad is not None for parameter in stage.parameters())


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = HiTSKT(_config()).eval()
    batch = collate_hitskt([_example()], num_questions=20, num_skills=5)
    checkpoint = tmp_path / "hitskt.pt"
    torch.save(model.checkpoint(), checkpoint)
    restored = HiTSKT.from_checkpoint(checkpoint).eval()

    with torch.no_grad():
        assert torch.equal(model(batch), restored(batch))
    assert restored.config == model.config
