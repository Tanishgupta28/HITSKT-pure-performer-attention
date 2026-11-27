import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient, MongoClient

from app import junyi_resources, junyi_routes
from app.junyi_bank import BankQuestion, JunyiBank, JunyiMappings, ModelItem, digest, load_bank
from app.junyi_model import Baseline, Event, GAP, ModelUnavailable, OriginalHiTSKT, SkillEstimator, read_baseline, score_class, session_context
from app.junyi_paper import DEFAULT_QUOTAS, BankShortage, candidates_for, prompt_candidates, select_paper, study_requirements
from app.junyi_resources import Gemini, StudyRecommender, _PageText
from app.junyi_routes import AnswerInput, JunyiService, setup_junyi
from app.junyi_runtime import JunyiRuntime
from app.junyi_study_policy import FOREST_POLICY, FEATURES, PaperRecommender
from app.main import create_app
from tools.export_junyi_history import _completed_answers, _history_rows


class FixturePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, earlier, prefix, candidates):
        self.calls.append((earlier, prefix))
        return {item: (.52 if item.skill_id < 24 else .79) + (.05 if prefix else 0) for item in candidates}


class FixtureResources:
    def __init__(self):
        self.calls = []

    def lookup(self, tag, topic, stage, language):
        self.calls.append(tag)
        links = [{"url": f"https://example.com/verified/{tag}", "title": f"Study {topic}", "source": "fixture", "verified_at": "2026-01-01"}]
        return (links, None) if tag % 2 else ([], "This fixture has no verified resource.")


def fixture_runtime():
    pairs = {tag: ModelItem(tag, tag) for tag in range(1, 33)}
    composites = {tag: (tag,) for tag in pairs}
    composites[1] = (1, 32)
    mapping = JunyiMappings("fixture-mapping", pairs, {tag: str(tag) for tag in pairs}, composites, {tag: (item,) for tag, item in pairs.items()})
    topics = {tag: f"Skill {tag}" for tag in pairs}
    stages = {tag: "junior" if tag > 3 else "senior" for tag in pairs}
    questions = {f"fixture-{tag}-{i}": BankQuestion(f"fixture-{tag}-{i}", tag, topics[tag], stages[tag], f"Fixture question {tag} / {i}",
                ("Right", "Wrong one", "Wrong two", "Wrong three"), f"template-{tag}-{i}") for tag in pairs for i in range(3)}
    bank = JunyiBank("fixture-bank", questions, topics, stages, {tag: tuple(f"fixture-{tag}-{i}" for i in range(3)) for tag in pairs})
    groups = [[Event(pairs[tag], int(tag > 23), seq * GAP) for tag in (i, i + 9, 24 + i - 1)] for seq, i in enumerate(range(1, 10), 1)]
    estimator = SkillEstimator(Baseline(tuple(map(tuple, groups)), 12 * GAP, "fixture-history"), bank, mapping, FixturePredictor())
    estimator.checkpoint_id = "fixture-checkpoint"
    return JunyiRuntime(bank, mapping, estimator, Gemini(), FixtureResources(), DEFAULT_QUOTAS.copy())


def test_forest_checklist_uses_attempted_paper_tags_and_falls_back_for_open_session():
    pytest.importorskip("pandas")
    runtime = fixture_runtime()
    feature_rows = []

    class Forest:
        def predict(self, frame):
            assert tuple(frame.columns) == FEATURES
            feature_rows.extend(frame.itertuples(index=False, name=None))
            return [0.33, 0.80, 0.40, 0.20]

    probes = {tag: (runtime.mappings.pairs[tag],) for tag in range(1, 7)}
    policy = PaperRecommender(Forest(), probes, runtime.mappings)
    items = [{"tag": tag} for tag in range(1, 6)]
    snapshot = {"skills": [{"tag": tag, "unattempted": tag == 5, "category": "weak" if tag == 1 else "intermediate"}
                           for tag in range(1, 7)]}
    requirements, audit = policy.requirements(runtime, [], 12 * GAP, items, snapshot)
    assert audit["policy"] == FOREST_POLICY
    assert [row["tag"] for row in requirements] == [2, 3, 1]
    assert [row["tag"] for row in audit["topics"]] == [1, 2, 3, 4]
    assert feature_rows[0] == pytest.approx((0.52, 1, 0, 8 * GAP / 86_400_000))

    answer = {"paper_sequence": 1, "position": 0, "timestamp_ms": 13 * GAP, "tag": 1, "correct": True,
              "proxy": {"question_id": 1, "skill_id": 1, "mapping_version": runtime.mappings.version}}
    requirements, audit = policy.requirements(runtime, [answer], 13 * GAP, items, snapshot)
    assert requirements == study_requirements(items, snapshot)
    assert audit["policy"] == "original_rule_fallback"
    assert "complete-session" in audit["reason"]
    assert len(feature_rows) == 4


def test_prepared_paper_persists_policy_audit_and_keeps_existing_checklist(settings):
    runtime = fixture_runtime()

    class Policy:
        def __init__(self):
            self.calls = 0

        def requirements(self, runtime, answers, as_of, items, snapshot):
            self.calls += 1
            assert len(items) == 20 and all(item["options"] and item["answer_id"] for item in items)
            assert answers == [] and snapshot["app_responses"] == 0
            tag = items[0]["tag"]
            return [{"tag": tag, "status": "pending", "lookup": "waiting", "resources": []}], {
                "policy": FOREST_POLICY, "topics": [{"tag": tag, "selected": True}],
            }

    policy = runtime.study_policy = Policy()
    owner = "fixture-owner"

    async def check():
        async with AsyncMongoClient(settings.mongodb_uri) as mongo:
            db = mongo[settings.mongodb_database]
            await setup_junyi(db)
            service = JunyiService(db, runtime, stage=None, allow_reuse=False)
            queued = await service.start(owner)
            await service.prepare(queued["_id"], owner)
            paper = await service.get(queued["_id"], owner)
            requirement, = paper["requirements"]
            assert paper["checklist_audit"] == {"policy": FOREST_POLICY, "topics": [{"tag": requirement["tag"], "selected": True}]}
            assert paper["status"] == ("locked" if requirement["tag"] % 2 else "open")
            assert requirement["lookup"] == ("verified" if requirement["tag"] % 2 else "unavailable")
            assert (await service.start(owner))["_id"] == queued["_id"]
            await service.prepare(queued["_id"], owner)
            assert await service.get(queued["_id"], owner) == paper
            assert policy.calls == 1

    asyncio.run(check())


def test_bank_versions_ids_and_invalid_options(tmp_path):
    mapping = fixture_runtime().mappings
    columns = "skill_id,topic_name,learning_stage,question,correct_answer,wrong_option_1,wrong_option_2,wrong_option_3\n"
    one = '1,[Basic] Fractions,junior,"One, plus one?",2,1,3,4\n'
    two = '1,[Basic] Fractions,junior,"Two plus one?",3,1,2,4\n'
    path = tmp_path / "bank.csv"
    data = (columns + one + two).encode()
    path.write_bytes(data)
    first = load_bank(path, mapping, digest(data))
    swapped = (columns + two + one).encode()
    path.write_bytes(swapped)
    assert set(first.items) == set(load_bank(path, mapping, digest(swapped)).items)
    assert first.topics[1] == "Fractions"
    with pytest.raises(ValueError, match="hash"):
        load_bank(path, mapping, digest(data))
    invalid = (columns + one.replace(",1,3,4", ",2,3,4")).encode()
    path.write_bytes(invalid)
    with pytest.raises(ValueError, match="distinct"):
        load_bank(path, mapping, digest(invalid))


def test_history_verifies_student_scope_counts_and_cutoff(tmp_path):
    mappings = fixture_runtime().mappings
    file = tmp_path / "baseline.json"
    sessions = [{"question_ids": [1] * count, "skill_ids": [1] * count, "correct": [0] * count,
                 "timestamp_ms": [i * (GAP + 100) + offset for offset in range(count)]}
                for i, count in enumerate([20] * 8 + [19], 1)]
    payload = {"source_student_id": 1, "processed_student_id": 1, "first_test_timestamp_ms": 11 * GAP, "sessions": sessions}

    def read(value):
        data = json.dumps(value).encode()
        file.write_bytes(data)
        return read_baseline(file, digest(data), mappings)

    verified = read(payload)
    assert len(verified.sessions) == 9 and sum(map(len, verified.sessions)) == 179
    with pytest.raises(ValueError, match="scope"):
        read({**payload, "processed_student_id": 2})
    with pytest.raises(ValueError, match="cutoff"):
        read({**payload, "first_test_timestamp_ms": sessions[-1]["timestamp_ms"][-1]})
    with pytest.raises(ValueError, match="179"):
        read({**payload, "sessions": sessions[:-1]})


def test_cutoff_context_and_original_model_placeholder_is_causal():
    torch = pytest.importorskip("torch")
    from app.research import hitskt_types
    from ktbench.batching import Session, SessionExample, collate_hitskt

    model, config = hitskt_types()
    torch.manual_seed(3)
    adapter = OriginalHiTSKT(None)
    adapter.model = model(config(num_questions=5, num_skills=3, width=16, feedforward_width=32, dropout=0)).eval()
    earlier = ((Event(ModelItem(1, 1), 1, 0), Event(ModelItem(2, 1), 0, 100)),)
    prefix = (Event(ModelItem(3, 2), 1, 5_000),)
    probes = [ModelItem(4, 2), ModelItem(5, 3)]
    zero = adapter.predict(earlier, prefix, probes, placeholder=0)
    one = adapter.predict(earlier, prefix, probes, placeholder=1)
    assert zero == pytest.approx(one, abs=1e-7)
    bos = adapter.predict(earlier, (), probes)
    assert list(bos) == probes and all(0 < value < 1 for value in bos.values())
    assert any(abs(bos[item]-value) > 1e-6 for item, value in zero.items())
    original_history = [Session([1, 2], [1, 1], [1, 0])]
    examples = [SessionExample(original_history, Session([3, item.question_id], [2, item.skill_id], [1, 0]), 2) for item in probes]
    original_batch = collate_hitskt(examples, num_questions=5, num_skills=3)
    with torch.inference_mode():
        direct = torch.sigmoid(adapter.model(original_batch)[:, 1]).tolist()
    assert list(zero.values()) == pytest.approx(direct, abs=1e-6)

    runtime = fixture_runtime()
    base = runtime.estimator.baseline
    response = {"timestamp_ms": base.cutoff, "position": 0, "paper_sequence": 1, "tag": 1, "correct": True,
                "proxy": {"question_id": 1, "skill_id": 1, "mapping_version": runtime.mappings.version}}
    prior, opened = session_context(base, [response], runtime.mappings, base.cutoff + GAP - 1)
    assert len(prior) == 9 and opened[0].correct == 1
    prior, opened = session_context(base, [response], runtime.mappings, base.cutoff + GAP)
    assert len(prior) == 10 and not opened
    assert runtime.estimator.snapshot([response], base.cutoff + 1)["skills"][-1]["app_attempts"] == 0
    with pytest.raises(ModelUnavailable, match="incomplete"):
        session_context(base, [{**response, "proxy": None}], runtime.mappings, base.cutoff + 1)
    with pytest.raises(ModelUnavailable, match="inference window"):
        session_context(base, [{**response, "timestamp_ms": base.cutoff - 1}], runtime.mappings, base.cutoff + 1)


def test_allocation_boundaries_stage_and_unknown_or_duplicate_proposals():
    assert [score_class(score, .61, .86) for score in (.6099, .61, .8599, .86, None)] == ["weak", "intermediate", "intermediate", "strong", None]
    runtime = fixture_runtime()
    snapshot = runtime.estimator.snapshot([], 15 * GAP)
    assert not snapshot["skills"][0]["unattempted"] and snapshot["skills"][18]["unattempted"]
    candidates = candidates_for(runtime.bank, snapshot, None, set(), False)
    menu = prompt_candidates(candidates, runtime.quotas, "seed")
    weak = next(row for row in menu if row["bucket"] == "weak")
    chosen, report = select_paper(candidates, runtime.quotas, "seed", [
        {**weak, "bucket": "unattempted", "reason": "Invalid bucket"},
        {**weak, "reason": "Valid"}, {**weak, "reason": "Duplicate"},
        {"bank_question_id": "invented", "bucket": "weak", "reason": "Unknown"}], menu)
    assert len({item["id"] for item in chosen}) == 20 and report["counts"] == DEFAULT_QUOTAS
    assert [item["reason"] for item in chosen if item["provider"] == "gemini"] == ["Valid"]
    again = select_paper(candidates, runtime.quotas, "seed")
    assert again == select_paper(candidates, runtime.quotas, "seed", [], [])
    with pytest.raises(BankShortage, match="Only 9"):
        select_paper(candidates_for(runtime.bank, snapshot, "senior", set(), False), runtime.quotas, "seed")
    broad_weak = [{"tag": 1, "bucket": "broad"}, {"tag": 19, "bucket": "unattempted"}]
    assert [requirement["tag"] for requirement in study_requirements(broad_weak, snapshot)] == [1]


def _gemini_with_reply(reply):
    return Gemini("fixture-key", "gemini-3.1-flash-lite", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(reply), **kwargs))


def _gemini_response(output, finish="STOP"):
    return httpx.Response(200, json={"candidates": [{"finishReason": finish, "content": {"parts": [{"text": json.dumps(output)}]}}]})


def test_gemini_compact_prompt_and_exact_group_allowlists():
    menu = [{"bank_question_id": f"verified-bank-{index}", "tag": index, "stage": "junior", "topic": "Fractions", "bucket": bucket}
            for index, bucket in ((1, "weak"), (2, "weak"), (3, "review"), (1, "broad"), (3, "broad"), (4, "broad"), (5, "unattempted"))]

    def reply(request):
        body = json.loads(request.content)
        data = json.loads(body["contents"][0]["parts"][0]["text"])
        assert "verified-bank" not in request.content.decode()
        assert data["groups"] == {"weak": {"limit": 1, "refs": ["q1", "q2"]}, "broad": {"limit": 1, "refs": ["q1", "q3", "q4"]}}
        assert data["items"] == [[f"q{i}", i, "junior", "Fractions"] for i in range(1, 5)]
        assert body["generationConfig"]["responseSchema"]["required"] == ["weak", "broad"]
        return _gemini_response({"weak": ["q3", "q999", "q1", "q1", "q2"], "review": ["q3"], "broad": ["q1", {"ref": "q2"}, "q3", "q4"]})

    chosen = _gemini_with_reply(reply).choose(menu, {"weak": 1, "unattempted": 0, "review": 0, "broad": 1})
    assert [(row["bank_question_id"], row["bucket"]) for row in chosen] == [("verified-bank-1", "weak"), ("verified-bank-3", "broad")]


def test_gemini_rejects_truncated_output_and_falls_back_to_paper_rules():
    runtime = fixture_runtime()
    snapshot = runtime.estimator.snapshot([], 15 * GAP)
    candidates = candidates_for(runtime.bank, snapshot, None, set(), False)
    menu = prompt_candidates(candidates, runtime.quotas, "seed")
    gemini = _gemini_with_reply(lambda _request: _gemini_response({"weak": ["q1"]}, finish="MAX_TOKENS"))
    assert not gemini.choose(menu, runtime.quotas)
    chosen, allocation = select_paper(candidates, runtime.quotas, "seed", gemini.choose(menu, runtime.quotas), menu)
    assert len(chosen) == 20 and all(item["provider"] == "rules" for item in chosen)
    assert allocation["counts"] == DEFAULT_QUOTAS


def test_curated_verification_deduplication_cap_and_expiry(monkeypatch):
    clock = [100]
    monkeypatch.setattr(junyi_resources, "monotonic", lambda: clock[0])
    monkeypatch.setattr(junyi_resources, "_CURATED_SKILL_LESSONS", tuple(
        ("fractions", f"https://education.test/{index}", "addition fractions") for index in range(9)
    ))
    reads = []

    def reader(url):
        index = int(url.split("/")[-1])
        reads.append(index)
        if index == 0:
            return None
        page = _PageText()
        page.feed({1: "<title>Unrelated history</title>", 2: "<title>Addition of fractions | Sign in</title>"}.get(
            index, "<title>Addition of fractions</title><p>Worked examples</p>"
        ))
        return f"https://education.test/{3 if index in (3, 4) else index}", page

    study = StudyRecommender(Gemini(), reader)
    links, issue = study.lookup(1, "Addition of fractions", "junior", "en")
    assert issue is None
    assert [link["url"].rsplit("/", 1)[-1] for link in links] == ["3", "5", "6", "7"]
    links[0]["title"] = "Client edit"
    assert study.lookup(1, "Addition of fractions", "junior", "en")[0][0]["title"] == "Addition of fractions"
    assert reads == list(range(8))
    clock[0] += junyi_resources.STUDY_RESOURCE_TTL.total_seconds()
    study.lookup(1, "Addition of fractions", "junior", "en")
    assert reads == list(range(8)) * 2


def test_study_ranking_rate_limit_does_not_block_verified_fallback_or_selection(monkeypatch):
    clock = [100]
    requests = []
    monkeypatch.setattr(junyi_resources, "monotonic", lambda: clock[0])

    def reply(request):
        body = json.loads(request.content)
        data = json.loads(body["contents"][0]["parts"][0]["text"])
        requests.append(data)
        assert "tools" not in body and "https://" not in json.dumps(data)
        if "groups" in data:
            return _gemini_response({"weak": ["q1"]})
        if clock[0] == 100:
            return httpx.Response(429, headers={"Retry-After": "20"})
        return _gemini_response({"order": ["r2", "invented", "r2"]})

    def reader(url):
        page = _PageText()
        page.feed("<title>Metric volume</title><p>Liters, milliliters and volume</p>")
        return url, page

    gemini = _gemini_with_reply(reply)
    study = StudyRecommender(gemini, reader)
    links, issue = study.lookup(1, "Liters and milliliters", "elementary", "en")
    assert issue is None and len(links) == 2 and links[0]["url"].startswith("https://www.mathsisfun.com")
    fallback, fallback_issue = study.lookup(2, "Exercises on liters and milliliters", "elementary", "en")
    assert fallback_issue is None and [link["url"] for link in fallback] == [link["url"] for link in links]
    assert len(requests) == 1
    menu = [{"bank_question_id": "bank-123", "tag": 1, "stage": "elementary", "topic": "Volume", "bucket": "weak"}]
    assert gemini.choose(menu, {"weak": 1})[0]["bank_question_id"] == "bank-123"
    clock[0] += 20
    ranked, error = study.lookup(3, "More exercises on liters and milliliters", "elementary", "en")
    assert error is None and [link["url"] for link in ranked] == [link["url"] for link in reversed(links)]
    assert len(requests) == 3


def test_public_study_verifier_rejects_nonpublic_destinations(monkeypatch):
    monkeypatch.setattr(junyi_resources.socket, "getaddrinfo", lambda *args, **kwargs: [(0, 0, 0, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="Non-public"):
        junyi_resources._public_target("https://example.test/redirect")
    with pytest.raises(ValueError, match="Public HTTPS"):
        junyi_resources._public_target("http://example.test/lesson")


def test_curated_study_lesson_verifies_the_mapped_subject():
    topic = "Comprehensive exercises on liters and milliliters"
    requested = []
    expected = "https://www.mathsisfun.com/measure/metric-volume.html"

    def reader_with_title(title, body=""):
        def read(url):
            requested.append(url)
            page = _PageText()
            page.feed(f"<title>{title}</title><p>{body}</p>")
            return url, page

        return read

    valid = StudyRecommender(Gemini(), reader_with_title("Metric Volume", "Compare a liter to a milliliter."))
    links, issue = valid.lookup(281, topic, "elementary", "en")
    assert issue is None and requested == [expected, "https://www.nist.gov/pml/owm/si-units-volume"]
    assert links[0]["url"] == expected and all(link["source"] == "curated" for link in links)

    unrelated = StudyRecommender(Gemini(), reader_with_title("Introduction to Algebra"))
    assert unrelated.lookup(281, topic, "elementary", "en")[0] == []


def test_curated_countdown_label_uses_the_reciprocal_concept():
    def read(url):
        assert url == "https://www.mathsisfun.com/algebra/reciprocal.html"
        page = _PageText()
        page.feed("<title>Reciprocal in Algebra</title><p>Take the reciprocal of a nonzero number.</p>")
        return url, page

    recommender = StudyRecommender(Gemini(), read)
    resources, issue = recommender.lookup(858, "Countdown", "junior", "en")
    assert issue is None and resources[0]["title"] == "Reciprocal in Algebra"
    assert recommender.lookup(858, "Countdown timer", "junior", "en")[0] == []


def test_queued_paper_keeps_allocation_settings_after_restart(settings):
    async def scenario():
        async with AsyncMongoClient(settings.mongodb_uri) as client:
            db = client[settings.mongodb_database]
            await setup_junyi(db)
            original = JunyiService(db, fixture_runtime(), stage=None, allow_reuse=False)
            queued = await original.start("phase-three-fixture")

            runtime = fixture_runtime()
            runtime.quotas = {"weak": 2, "unattempted": 10, "review": 6, "broad": 2}
            restarted = JunyiService(db, runtime, stage="senior", allow_reuse=True)
            resumed = await restarted.start(queued["owner"])
            assert resumed["_id"] == queued["_id"]
            await restarted.prepare(resumed["_id"], resumed["owner"])
            paper = await restarted.get(resumed["_id"], resumed["owner"])
            assert paper["status"] == "locked" and len(paper["items"]) == 20
            assert paper["stage"] is None and paper["allow_reuse"] is False
            assert paper["quotas"] == DEFAULT_QUOTAS
            assert paper["allocation"]["counts"] == DEFAULT_QUOTAS

    asyncio.run(scenario())


def test_analysis_retry_keeps_completion_window_and_next_paper_uses_new_window(settings, monkeypatch):
    async def scenario():
        runtime = fixture_runtime()
        completed_at = 20 * GAP
        response = {"timestamp_ms": completed_at, "paper_sequence": 1, "position": 0, "tag": 1, "correct": True,
                    "bank_question_id": "fixture-1-0", "proxy": junyi_routes._answer_proxy(runtime.mappings, 1)}
        monkeypatch.setattr(junyi_routes, "_now", lambda: completed_at)
        async with AsyncMongoClient(settings.mongodb_uri) as client:
            db = client[settings.mongodb_database]
            await setup_junyi(db)
            service = JunyiService(db, runtime, stage=None, allow_reuse=False)
            await service.papers.insert_one({"_id": "fixture-completed", "owner": "fixture-owner", "sequence": 1,
                                             "status": "completed", "active": False, "responses": [response], "version": 0,
                                             "completed_at": completed_at, "analysis_status": "pending", "analysis_lease_until": 0})
            await service.analyze("fixture-completed", "fixture-owner")
            original = await service.get("fixture-completed", "fixture-owner")
            assert original["analysis_status"] == "ready" and original["result_snapshot"]["as_of"] == completed_at
            assert original["result_snapshot"]["app_responses"] == 1 and len(runtime.estimator.predictor.calls[-1][1]) == 1

            # Reclaim an interrupted job exactly when the original live session would have expired.
            await service.papers.update_one({"_id": "fixture-completed"}, {"$set": {"analysis_status": "pending", "analysis_lease_until": completed_at + 1}})
            monkeypatch.setattr(junyi_routes, "_now", lambda: completed_at + GAP)
            await service.analyze("fixture-completed", "fixture-owner")
            recovered = await service.get("fixture-completed", "fixture-owner")
            assert recovered["analysis_status"] == "ready" and recovered["result_snapshot"] == original["result_snapshot"]
            assert recovered["responses"] == [response] and len(runtime.estimator.predictor.calls[-1][1]) == 1

            next_paper = await service.start("fixture-owner")
            await service.prepare(next_paper["_id"], "fixture-owner")
            prepared = await service.get(next_paper["_id"], "fixture-owner")
            assert prepared["snapshot"]["as_of"] == completed_at + GAP and not runtime.estimator.predictor.calls[-1][1]
            assert prepared["snapshot"]["app_history_version"] == original["result_snapshot"]["app_history_version"]
            assert all(item["id"] != response["bank_question_id"] for item in prepared["items"])

    asyncio.run(scenario())


def test_invalid_proxy_mapping_preserves_actual_grade_and_records_reason(settings):
    async def scenario():
        runtime = fixture_runtime()
        mappings = runtime.mappings
        source_id = mappings.source_ids.pop(1)
        mappings.composite_tags[2] = (1, 2)
        mappings.representatives[1] = (ModelItem(1, 1), ModelItem(2, 2))
        assert junyi_routes._answer_proxy(mappings, 1)["question_id"] == 2
        mappings.source_ids[1] = source_id
        mappings.composite_tags[2] = (2,)
        mappings.representatives[1] = (ModelItem(2, 2), ModelItem(3, 3))
        async with AsyncMongoClient(settings.mongodb_uri) as client:
            service = JunyiService(client[settings.mongodb_database], runtime, stage=None, allow_reuse=False)
            await service.papers.insert_one({"_id": "unmapped-fixture", "owner": "fixture-owner", "sequence": 1,
                                             "status": "open", "active": True, "version": 0, "total": 1, "responses": [],
                                             "bank_version": "fixture-bank", "items": [{"id": "fixture-1-0", "tag": 1,
                                                                                       "options": [{"id": "b"}, {"id": "a"}], "answer_id": "a"}]})
            data = AnswerInput(question_id="fixture-1-0", choice_id="b", request_id=uuid4())
            completed, replayed = await service.record("unmapped-fixture", "fixture-owner", data)
            assert not replayed and completed["results"] == {"correct": 0, "total": 1, "accuracy": 0}
            answer = completed["responses"][0]
            assert answer["proxy"] is None and answer["mapping_issue"] and answer["correct"] is False
            await service.analyze("unmapped-fixture", "fixture-owner")
            result = await service.get("unmapped-fixture", "fixture-owner")
            assert result["analysis_status"] == "unavailable" and result["responses"] == [answer]
            assert result["result_snapshot"]["app_responses"] == 1

    asyncio.run(scenario())


def test_private_history_rows_keep_real_grades_and_exact_session_boundaries():
    runtime = fixture_runtime()
    baseline, mappings = runtime.estimator.baseline, runtime.mappings
    tags = [1, 2, 3]
    times = [20 * GAP, 20 * GAP, 21 * GAP]
    answers = [{"timestamp_ms": stamp, "paper_sequence": index + 1, "position": 0, "tag": tag, "correct": index == 0,
                "bank_question_id": f"fixture-{tag}-0", "request_id": str(index),
                "proxy": junyi_routes._answer_proxy(mappings, tag) if index < 2 else None}
               for index, (tag, stamp) in enumerate(zip(tags, times))]
    history = list(_history_rows(baseline, mappings, answers))
    assert len(history) == sum(map(len, baseline.sessions)) + 3
    first, second, third = history[-3:]
    assert (first[0], first[1], first[6], first[9]) == ("LUMEN_PROXY", 10, 1, "0")
    assert (second[0], second[1], second[6], second[9]) == ("LUMEN_PROXY", 10, 0, "1")
    assert (third[0], third[1], third[3], third[6], third[9]) == ("LUMEN_UNMAPPED", 11, None, 0, "2")
    assert third[-1]


def test_rules_only_progression_after_sixty_fixture_answers(settings, monkeypatch):
    async def scenario():
        runtime = fixture_runtime()
        owner = "fixture-only"
        first_timestamp = 20 * GAP
        monkeypatch.setattr(junyi_routes, "_now", lambda: first_timestamp)
        async with AsyncMongoClient(settings.mongodb_uri) as client:
            db = client[settings.mongodb_database]
            await setup_junyi(db)
            service = JunyiService(db, runtime, stage=None, allow_reuse=False)
            seen = set()
            for sequence in range(1, 4):
                queued = await service.start(owner)
                await service.prepare(queued["_id"], owner)
                paper = await service.get(queued["_id"], owner)
                assert paper["snapshot"]["app_responses"] == (sequence - 1) * 20
                assert all(item["provider"] == "rules" for item in paper["items"])
                ids = {item["id"] for item in paper["items"]}
                assert len(ids) == 20 and not ids.intersection(seen)
                seen.update(ids)
                for requirement in paper["requirements"]:
                    if requirement["resources"]:
                        await service.study_done(paper["_id"], owner, requirement["tag"], url=requirement["resources"][0]["url"])
                for item in paper["items"]:
                    await service.record(paper["_id"], owner, AnswerInput(
                        question_id=item["id"], choice_id=item["answer_id"], request_id=uuid4(),
                    ))
                await service.analyze(paper["_id"], owner)
                completed = await service.get(paper["_id"], owner)
                assert completed["analysis_status"] == "ready" and completed["result_snapshot"]["app_responses"] == sequence * 20

            next_paper = await service.start(owner)
            await service.prepare(next_paper["_id"], owner)
            frozen = await service.get(next_paper["_id"], owner)
            assert frozen["sequence"] == 4 and frozen["status"] == "locked" and frozen["responses"] == []
            assert frozen["snapshot"]["app_history_version"] == completed["result_snapshot"]["app_history_version"]
            assert len({item["id"] for item in frozen["items"]}) == 20 and not seen.intersection(item["id"] for item in frozen["items"])
            assert len(runtime.estimator.predictor.calls[-1][1]) == frozen["snapshot"]["app_responses"] == 60
            assert (await service.start(owner))["_id"] == next_paper["_id"]

    asyncio.run(scenario())


def test_authenticated_assessment_lock_resume_grading_proxy_and_next(settings):
    runtime = fixture_runtime()
    with TestClient(create_app(settings, junyi_runtime=runtime)) as learner, TestClient(create_app(settings, junyi_runtime=runtime)) as admin:
        def register(client, email):
            result = client.post("/api/auth/register", json={"name": "Junyi Fixture", "email": email, "password": "thoughtful-practice-42"})
            assert result.status_code == 201, result.text
            return result.json()["id"]

        owner = register(learner, "junyi-learner@example.com")
        operator = register(admin, "junyi-operator@example.com")
        database = MongoClient(settings.mongodb_uri)[settings.mongodb_database]
        database.users.update_one({"_id": operator}, {"$set": {"role": "admin"}})
        settings.junyi_student_account_id, settings.junyi_operator_account_id = owner, operator
        with ThreadPoolExecutor(max_workers=2) as pool:
            created = list(pool.map(lambda _: learner.post("/api/assessments"), [0, 1]))
        assert all(response.status_code == 202 for response in created), [r.text for r in created]
        id = created[0].json()["id"]
        assert all(response.json()["id"] == id for response in created)
        stored = database.junyi_papers.find_one({"_id": id})
        assert len(stored["items"]) == 20
        url = f"/api/assessments/{id}"
        locked = learner.get(url).json()
        assert locked["status"] == "locked" and locked["question"] is None
        assert learner.get(f"{url}/question").status_code == 423
        assert learner.post(f"{url}/answers", json={"question_id": "fiction", "choice_id": "fiction", "request_id": str(uuid4())}).status_code == 423
        action_tags = [row["tag"] for row in locked["requirements"]]
        dashboard = learner.get("/api/junyi/dashboard").json()
        assert dashboard["active"] == {"id": id, "status": "locked", "sequence": 1,
                                       "required_tags": [row["tag"] for row in locked["requirements"] if row["required"]]}
        assert locked["sequence"] == 1 and locked["answers_used"] is None
        assert sorted(runtime.recommender.calls) == sorted(action_tags)
        assert learner.get(url).json() == locked
        assert sorted(runtime.recommender.calls) == sorted(action_tags)
        outside = next(row["tag"] for row in stored["snapshot"]["skills"] if row["category"] == "weak" and row["tag"] not in action_tags)
        assert learner.post(f"/api/junyi/study/{outside}").status_code == 200
        assert outside not in [row["tag"] for row in learner.get(url).json()["requirements"]]

        assert learner.post(f"/api/admin/junyi/{id}/study/{action_tags[0]}/override", json={"reason": "Fixture override"}).status_code == 403
        assert {requirement["lookup"] for requirement in locked["requirements"]} == {"verified", "unavailable"}
        assert all(row["can_retry"] == (row["lookup"] == "unavailable") for row in locked["requirements"])
        unavailable = [row for row in locked["requirements"] if row["lookup"] == "unavailable"]
        for requirement in locked["requirements"]:
            assert requirement["required"] == bool(requirement["resources"])
            if requirement["required"]:
                response = learner.post(f"{url}/study/{requirement['tag']}/complete", json={"url": requirement["resources"][0]["url"]})
                assert response.status_code == 200, response.text
        unlocked = learner.get(url).json()
        assert unlocked["status"] == "open" and unlocked["revision"] > locked["revision"]
        assert [row for row in unlocked["requirements"] if row["status"] == "pending"] == unavailable
        assert learner.get(f"{url}/question").status_code == 200
        override = admin.post(f"/api/admin/junyi/{id}/study/{unavailable[0]['tag']}/override", json={"reason": "Fixture resource missing"})
        assert override.status_code == 200
        assert any(
            row.get("audit", {}).get("actor") == operator for row in database.junyi_papers.find_one({"_id": id})["requirements"])

        for index, item in enumerate(stored["items"]):
            question = learner.get(f"{url}/question").json()
            assert question["id"] == item["id"] and question["options"] == item["options"]
            assert "answer_id" not in question
            choice = item["answer_id"] if index % 2 == 0 else next(value["id"] for value in question["options"] if value["id"] != item["answer_id"])
            body = {"question_id": item["id"], "choice_id": choice, "request_id": str(uuid4())}
            if index == 0:
                simultaneous = [body, {**body, "request_id": str(uuid4())}]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    responses = list(pool.map(lambda candidate: learner.post(f"{url}/answers", json=candidate), simultaneous))
                assert sorted(response.status_code for response in responses) == [200, 409]
                body = simultaneous[next(i for i, response in enumerate(responses) if response.status_code == 200)]
                retry = learner.post(f"{url}/answers", json=body)
                assert retry.json()["replayed"] and retry.json()["paper"]["answered"] == 1
                assert learner.post(f"{url}/answers", json={**body, "choice_id": "different"}).status_code == 409
            else:
                result = learner.post(f"{url}/answers", json=body)
                assert result.status_code == 200, result.text
        assert learner.get(url).json()["results"] == {"correct": 10, "total": 20, "accuracy": .5}
        assert learner.get(url).json()["answers_used"] == 20
        completed_history = learner.get("/api/junyi/dashboard").json()
        assert completed_history["answers_used"] == 20 and completed_history["history"][0]["sequence"] == 1
        graded = database.junyi_papers.find_one({"_id": id})
        assert len(graded["responses"]) == 20 and graded["responses"][0]["proxy"]["label"] == "PROXY"
        assert _completed_answers([graded]) == graded["responses"]
        incorrect = {**graded, "responses": [{**graded["responses"][0], "correct": False}, *graded["responses"][1:]]}
        with pytest.raises(ValueError, match="frozen question or grade"):
            _completed_answers([incorrect])
        assert graded["result_snapshot"]["app_responses"] == 20
        expected_history = graded["result_snapshot"]["app_history_version"]
        database.junyi_papers.update_one({"_id": id}, {"$set": {"result_snapshot.app_history_version": "stale-history-with-the-same-size"}})
        next_paper = learner.post("/api/assessments").json()
        refreshed = database.junyi_papers.find_one({"_id": next_paper["id"]})
        assert refreshed["snapshot"]["app_responses"] == 20 and refreshed["_id"] != id
        assert refreshed["snapshot"]["app_history_version"] == expected_history
        assert not {q["id"] for q in refreshed["items"]} & {q["id"] for q in stored["items"]}
        assert any(row["score"] != row["original_score"] for row in refreshed["snapshot"]["skills"])


@pytest.mark.parametrize("trigger", ["retry", "recheck"])
def test_optional_study_becomes_required_when_link_is_found(settings, monkeypatch, trigger):
    runtime = fixture_runtime()
    clock = [junyi_routes._now()]
    monkeypatch.setattr(junyi_routes, "_now", lambda: clock[0])
    found = set()
    calls = []
    link = {"url": "https://example.test/verified-lesson", "title": "Fixture lesson", "source": "fixture"}

    def lookup(tag, topic, stage, language):
        calls.append(tag)
        return ([link], None) if tag in found else ([], "No verified resource.")

    monkeypatch.setattr(runtime.recommender, "lookup", lookup)
    with TestClient(create_app(settings, junyi_runtime=runtime)) as learner:
        account = learner.post("/api/auth/register", json={"name": "Study Fixture", "email": "changes@example.com", "password": "thoughtful-practice-42"})
        assert account.status_code == 201
        settings.junyi_student_account_id = account.json()["id"]
        paper_id = learner.post("/api/assessments").json()["id"]
        path = f"/api/assessments/{paper_id}"
        with MongoClient(settings.mongodb_uri) as mongo:
            # A previously generated paper may still store the old lock state.
            mongo[settings.mongodb_database].junyi_papers.update_one({"_id": paper_id}, {"$set": {"status": "locked"}})
        unlocked = learner.get(path).json()
        assert unlocked["status"] == "open" and unlocked["requirements"]
        assert all(row["status"] == "pending" and not row["required"] for row in unlocked["requirements"])
        assert learner.get("/api/junyi/dashboard").json()["active"]["status"] == "open"
        tag = unlocked["requirements"][0]["tag"]
        question = unlocked["question"]
        submission = {"question_id": question["id"], "choice_id": question["options"][0]["id"], "request_id": str(uuid4())}
        next_question = learner.post(f"{path}/answers", json=submission).json()["paper"]["question"]

        found.add(tag)
        assert learner.get(path).json()["status"] == "open" and calls.count(tag) == 1
        if trigger == "retry":
            assert learner.post(f"{path}/study/{tag}/retry").status_code == 202
        else:
            clock[0] += junyi_routes.STUDY_RECHECK_MS
            learner.get(path)
        locked = learner.get(path).json()
        assert locked["status"] == "locked" and locked["answered"] == 1 and locked["question"] is None
        assert [row["tag"] for row in locked["requirements"] if row["required"]] == [tag]
        assert learner.get(f"{path}/question").status_code == 423
        blocked = {"question_id": next_question["id"], "choice_id": next_question["options"][0]["id"], "request_id": str(uuid4())}
        assert learner.post(f"{path}/answers", json=blocked).status_code == 423
        assert learner.post(f"{path}/answers", json=submission).json()["replayed"]
        resumed = learner.post(f"{path}/study/{tag}/complete", json={"url": link["url"]}).json()
        assert resumed["status"] == "open" and resumed["answered"] == 1 and resumed["question"] == next_question
