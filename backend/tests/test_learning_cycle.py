from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.catalog import BANK
from app.engine import initial_concept, update_knowledge
from app.main import create_app


def start(client, mode="diagnostic", concept_id=None):
    response = client.post("/api/assessments", json={"mode": mode, "concept_id": concept_id})
    assert response.status_code == 201, response.text
    return response.json()


def submit(client, assessment, correct=True, request_id=None):
    q = assessment["question"]
    index = BANK[q["id"]]["answer_index"]
    body = {"question_id": q["id"], "choice_index": index if correct else (index+1)%4, "request_id": request_id or str(uuid4())}
    result = client.post(f"/api/assessments/{assessment['id']}/answers", json=body)
    assert result.status_code == 200, result.text
    return result.json(), body


def test_full_diagnostic_recommendation_practice_cycle(student):
    assessment = start(student)
    seen = []
    for i in range(12):
        q = assessment["question"]
        assert "answer_index" not in q and "explanation" not in q
        seen.append(q["concept_id"])
        result, _ = submit(student, assessment, correct=q["concept_id"] != "fractions")
        assessment = result["assessment"]
    assert assessment["status"] == "completed"
    assert assessment["question"] is None
    assert Counter(seen) == {"fractions": 3, "percentages": 3, "equations": 3, "geometry": 3}
    dashboard = student.get("/api/learning/mathematics").json()
    assert dashboard["stats"]["answers"] == 12
    assert dashboard["diagnostic_complete"] is True
    assert dashboard["recommendations"][0]["concept_id"] == "fractions"
    assert len(dashboard["progress"]) == 1
    practice = start(student, "practice", "fractions")
    for _ in range(6):
        assert practice["question"]["concept_id"] == "fractions"
        result, _ = submit(student, practice)
        practice = result["assessment"]
    final = student.get("/api/learning/mathematics").json()
    assert final["stats"]["answers"] == 18
    assert len(final["progress"]) == 2
    assert final["progress"][1]["mastery"] > final["progress"][0]["mastery"]


def test_answer_retry_is_idempotent_and_tampering_rejected(student):
    assessment = start(student)
    result, body = submit(student, assessment)
    retry = student.post(f"/api/assessments/{assessment['id']}/answers", json=body)
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert student.get("/api/learning/mathematics").json()["stats"]["answers"] == 1
    body["choice_index"] = (body["choice_index"]+1)%4
    assert student.post(f"/api/assessments/{assessment['id']}/answers", json=body).status_code == 409
    body["request_id"] = str(uuid4())
    assert student.post(f"/api/assessments/{assessment['id']}/answers", json=body).status_code == 409


def test_concurrent_submissions_never_double_count(student):
    assessment = start(student)
    body = {"question_id": assessment["question"]["id"], "choice_index": 0, "request_id": str(uuid4())}
    def send(_):
        return student.post(f"/api/assessments/{assessment['id']}/answers", json=body).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(send, [0,1]))
    assert 200 in codes and all(code in (200,409) for code in codes)
    assert student.get("/api/learning/mathematics").json()["stats"]["answers"] == 1


def test_adapts_difficulty_after_success(student):
    assessment = start(student,"practice","equations")
    difficulties=[]
    for _ in range(6):
        difficulties.append(assessment["question"]["difficulty"])
        result,_=submit(student,assessment)
        assessment=result["assessment"]
    assert difficulties[0] == 1
    assert max(difficulties[2:]) >= 2


def test_database_persistence_across_application_restart(student, settings):
    assessment = start(student)
    result,_=submit(student,assessment)
    cookies=dict(student.cookies)
    with TestClient(create_app(settings)) as second:
        second.cookies.update(cookies)
        resumed=second.get(f"/api/assessments/{assessment['id']}").json()
        assert resumed["answered"] == 1
        assert resumed["question"]["id"] == result["assessment"]["question"]["id"]
        assert second.get("/api/learning/mathematics").json()["stats"]["answers"] == 1


def test_auth_ownership_logout_and_origin(student):
    assessment=start(student)
    original=dict(student.cookies)
    student.cookies.clear()
    assert student.get("/api/learning/mathematics").status_code == 401
    student.post("/api/auth/register",json={"name":"Other Student","email":"other@example.com","password":"another-good-password"})
    assert student.get(f"/api/assessments/{assessment['id']}").status_code == 404
    assert student.post("/api/auth/logout",headers={"Origin":"https://untrusted.example"}).status_code == 403
    assert student.post("/api/auth/logout").status_code == 204
    assert student.get("/api/auth/me").status_code == 401
    assert student.post("/api/auth/login",json={"email":"learner@example.com","password":"wrong-password"}).status_code == 401
    response=student.post("/api/auth/login",json={"email":"learner@example.com","password":"thoughtful-practice-42"})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]


def test_question_bank_has_valid_unique_answers():
    assert len(BANK) == 384
    for q in BANK.values():
        assert len(q["choices"]) == 4 and len(set(q["choices"])) == 4
        assert 0 <= q["answer_index"] < 4


def test_cold_start_label_and_estimate_direction(student):
    assessment=start(student)
    assert assessment["prediction"]["provider"] == "bayesian_knowledge_tracing"
    p=initial_concept()
    assert update_knowledge(p,True,1,"now")["mastery"] > p["mastery"]
    assert update_knowledge(p,False,1,"now")["mastery"] < p["mastery"]
    assert student.get("/api/health").json()["model"]["hitskt_loaded"] is False
