import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient, MongoClient

from app.junyi_bank import BankQuestion, JunyiBank, JunyiMappings, ModelItem, digest, load_bank
from app.junyi_model import Baseline, Event, GAP, ModelUnavailable, OriginalHiTSKT, SkillEstimator, read_baseline, score_class, session_context
from app.junyi_paper import DEFAULT_QUOTAS, BankShortage, candidates_for, prompt_candidates, select_paper, study_requirements
from app.junyi_resources import Gemini, StudyRecommender, _PageText
from app.junyi_routes import JunyiService, setup_junyi
from app.junyi_runtime import JunyiRuntime
from app.main import create_app


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


def test_grounding_verification_deduplication_and_cap(monkeypatch):
    from app import junyi_resources

    gemini = Gemini("example-key", "example-model")
    replies = [{"content": {"parts": [{"text": "A made-up URL https://example.test/not-grounded"}]}},
               {"groundingMetadata": {"webSearchQueries": ["fraction addition"], "groundingSupports": [{"groundingChunkIndices": list(range(8))}],
                                      "groundingChunks": [{"web": {"uri": f"https://example.test/{i}"}} for i in range(8)]}}]
    monkeypatch.setattr(gemini, "_call", lambda *args, **kwargs: replies.pop(0))
    assert gemini.sources(1, "Addition of fractions", "junior", "en") == []

    def reader(url):
        index = int(url.split("/")[-1])
        if index == 0:
            return None
        page = _PageText()
        page.feed("<title>Unrelated history</title>" if index == 1 else "<title>Addition of fractions</title>Worked examples")
        return f"https://education.test/{2 if index in (2, 3) else index}", page

    links, _ = StudyRecommender(gemini, reader).lookup(1, "Addition of fractions", "junior", "en")
    assert [link["url"].rsplit("/", 1)[-1] for link in links] == ["2", "4", "5", "6"]
    monkeypatch.setattr(junyi_resources.socket, "getaddrinfo", lambda *args, **kwargs: [(0, 0, 0, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="Non-public"):
        junyi_resources._public_target("https://example.test/redirect")


def test_curated_study_lesson_verifies_the_mapped_subject():
    topic = "Comprehensive exercises on liters and milliliters"
    requested = []
    expected = "https://www.mathsisfun.com/measure/metric-volume.html"

    def reader_with_title(title):
        def read(url):
            requested.append(url)
            page = _PageText()
            page.feed(f"<title>{title}</title><p>{title}</p>")
            return url, page

        return read

    valid = StudyRecommender(Gemini(), reader_with_title("Metric Volume"))
    links, issue = valid.lookup(281, topic, "elementary", "en")
    assert issue is None and requested == [expected]
    assert [(link["url"], link["source"]) for link in links] == [(expected, "curated")]

    unrelated = StudyRecommender(Gemini(), reader_with_title("Introduction to Algebra"))
    assert unrelated.lookup(281, topic, "elementary", "en")[0] == []


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
        assert sorted(runtime.recommender.calls) == sorted(action_tags)
        assert learner.get(url).json() == locked
        assert sorted(runtime.recommender.calls) == sorted(action_tags)
        outside = next(row["tag"] for row in stored["snapshot"]["skills"] if row["category"] == "weak" and row["tag"] not in action_tags)
        assert learner.post(f"/api/junyi/study/{outside}").status_code == 200
        assert outside not in [row["tag"] for row in learner.get(url).json()["requirements"]]

        assert learner.post(f"/api/admin/junyi/{id}/study/{action_tags[0]}/override", json={"reason": "Fixture override"}).status_code == 403
        assert {requirement["lookup"] for requirement in locked["requirements"]} == {"verified", "unavailable"}
        for requirement in locked["requirements"]:
            tag = requirement["tag"]
            if requirement["resources"]:
                response = learner.post(f"{url}/study/{tag}/complete", json={"url": requirement["resources"][0]["url"]})
            else:
                response = admin.post(f"/api/admin/junyi/{id}/study/{tag}/override", json={"reason": "Fixture resource missing"})
            assert response.status_code == 200, response.text
        assert learner.get(url).json()["status"] == "open"
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
        graded = database.junyi_papers.find_one({"_id": id})
        assert len(graded["responses"]) == 20 and graded["responses"][0]["proxy"]["label"] == "PROXY"
        assert graded["result_snapshot"]["app_responses"] == 20
        expected_history = graded["result_snapshot"]["app_history_version"]
        database.junyi_papers.update_one({"_id": id}, {"$set": {"result_snapshot.app_history_version": "stale-history-with-the-same-size"}})
        next_paper = learner.post("/api/assessments").json()
        refreshed = database.junyi_papers.find_one({"_id": next_paper["id"]})
        assert refreshed["snapshot"]["app_responses"] == 20 and refreshed["_id"] != id
        assert refreshed["snapshot"]["app_history_version"] == expected_history
        assert not {q["id"] for q in refreshed["items"]} & {q["id"] for q in stored["items"]}
        assert any(row["score"] != row["original_score"] for row in refreshed["snapshot"]["skills"])
