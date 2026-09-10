import pytest
import torch

from ktbench.batching import (
    CORRECT_BOS_ID,
    CORRECT_PAD_ID,
    ExampleShape,
    LengthBucketTokenBatchSampler,
    Session,
    SessionExample,
    collate_hitskt,
    padded_token_cost,
)


def _session(length: int, start: int = 1) -> Session:
    return Session(
        question_ids=[(start + index) % 11 + 1 for index in range(length)],
        skill_ids=[(start + index) % 5 + 1 for index in range(length)],
        correct=[index % 2 for index in range(length)],
    )


def _examples() -> list[SessionExample]:
    return [
        SessionExample(history=[_session(2)], target=_session(3, 3), split=0),
        SessionExample(history=[_session(1), _session(5, 2)], target=_session(1, 8), split=1),
    ]


def test_dynamic_padding_preserves_variable_length_sessions_and_eos() -> None:
    examples = _examples()
    batch = collate_hitskt(examples, num_questions=12, num_skills=6)

    # Histories are flattened: 3 complete sessions, padded only to max(2,1,5)+EOS.
    assert batch.history_question.shape == (3, 6)
    assert batch.history_mask.sum(dim=1).tolist() == [3, 2, 6]
    assert batch.history_question[0, :3].tolist() == [2, 3, 13]
    assert batch.history_question[2, :6].tolist() == [3, 4, 5, 6, 7, 13]
    assert batch.history_question[0, 3:].eq(0).all()
    assert batch.history_correct[0, 3:].eq(CORRECT_PAD_ID).all()

    assert batch.target_question.shape == (2, 4)
    assert batch.target_attention_mask.sum(dim=1).tolist() == [4, 2]
    assert batch.target_metric_mask.sum(dim=1).tolist() == [3, 1]
    assert batch.target_question[0, 3].item() == 13
    assert batch.target_skill[0, 3].item() == 7
    assert not batch.target_metric_mask[0, 3]
    assert batch.target_correct_input[:, 0].eq(CORRECT_BOS_ID).all()


def test_target_shift_alignment_and_no_pad_or_eos_target() -> None:
    target = Session([1, 2, 3, 4], [1, 1, 2, 2], [1, 0, 0, 1])
    batch = collate_hitskt(
        [SessionExample(history=[_session(2)], target=target, split=2)],
        num_questions=12,
        num_skills=6,
    )

    assert batch.target_labels[0].tolist() == [1, 0, 0, 1, CORRECT_PAD_ID]
    assert batch.target_correct_input[0].tolist() == [CORRECT_BOS_ID, 1, 0, 0, 1]
    assert batch.target_metric_mask[0].tolist() == [True, True, True, True, False]
    assert batch.target_attention_mask[0].tolist() == [True, True, True, True, True]


def test_token_sampler_uses_smaller_batches_for_long_sessions() -> None:
    shapes = [
        ExampleShape((4,), 4),
        ExampleShape((5,), 5),
        ExampleShape((4, 5), 3),
        ExampleShape((120,), 200),
        ExampleShape((6,), 5),
    ]
    sampler = LengthBucketTokenBatchSampler(
        shapes,
        token_budget=60,
        max_batch_size=8,
        bucket_size=2,
        shuffle=False,
    )
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(shapes)))
    long_batch = next(batch for batch in batches if 3 in batch)
    assert long_batch == [3]
    for batch in batches:
        if len(batch) > 1:
            assert padded_token_cost([shapes[index] for index in batch]) <= 60


def test_collator_rejects_incomplete_or_malformed_sessions() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Session([], [], [])
    with pytest.raises(ValueError, match="different lengths"):
        Session([1], [1, 2], [1])
    with pytest.raises(ValueError, match="binary"):
        Session([1], [1], [2])
    with pytest.raises(ValueError, match="earlier session"):
        SessionExample([], _session(1), 0)


def test_batch_to_preserves_metadata() -> None:
    batch = collate_hitskt(_examples(), num_questions=12, num_skills=6)
    moved = batch.to("cpu")
    assert moved.padded_tokens == batch.padded_tokens
    assert vars(moved).keys() == vars(batch).keys()
    assert torch.equal(moved.target_labels, batch.target_labels)
