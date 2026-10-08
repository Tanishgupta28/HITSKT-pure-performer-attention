from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.catalog import BANK
from app.main import create_app
from app.profile import LearnerProfile


def test_profile_persists_and_controls_new_sessions_without_changing_history(student, settings):
    original = student.get("/api/auth/me").json()
    assert original["profile"]["source"] == "default"
    profile = LearnerProfile(grade_level="Grade 9", curriculum="CBSE", weekly_question_target=18,
                             adaptive_session_questions=6, study_days=["Tue", "Sat"],
                             focus_concepts=["geometry", "fractions"]).model_dump()
    saved = student.post("/api/profile", json=profile)
    assert saved.status_code == 200
    assert saved.json()["profile"]["source"] == "user"
    assert saved.json()["email"] == original["email"]
    with TestClient(create_app(settings)) as restarted:
        restarted.cookies.update(dict(student.cookies))
        assert restarted.get("/api/auth/me").json()["profile"]["grade_level"] == "Grade 9"
        dashboard = restarted.get("/api/learning/mathematics").json()
        assert dashboard["weekly_goal"] == {"answered": 0, "target": 18, "remaining": 18}
        assert [s["concept_id"] for s in dashboard["study_plan"]] == ["geometry", "fractions"]
        assert [s["day"] for s in dashboard["study_plan"]] == ["Tue", "Sat"]
        assert dashboard["stats"]["answers"] == 0 and dashboard["progress"] == []
        assessment = restarted.post("/api/assessments", json={"mode": "adaptive"}).json()
        assert assessment["total"] == 6
        # Updating preferences cannot reset an active assessment or change its length.
        profile["adaptive_session_questions"] = 18
        restarted.post("/api/profile", json=profile).raise_for_status()
        assert restarted.get(f"/api/assessments/{assessment['id']}").json()["total"] == 6


def test_profile_validation_and_ownership(student):
    profile = LearnerProfile().model_dump()
    invalid = [{"weekly_question_target": 0}, {"weekly_question_target": "30"},
               {"adaptive_session_questions": 9}, {"focus_concepts": ["invented"]},
               {"study_days": []}, {"study_days": ["Mon", "Mon"]},
               {"learning_goal": "     "}, {"user_id": "someone-else"}, {"source": "sample"}]
    for changes in invalid:
        assert student.post("/api/profile", json={**profile, **changes}).status_code == 422
    assert student.post("/api/profile", json=profile, headers={"Origin": "https://untrusted.example"}).status_code == 403
    original_cookies = dict(student.cookies)
    student.cookies.clear()
    assert student.post("/api/profile", json=profile).status_code == 401
    student.post("/api/auth/register", json={"name": "Second Learner", "email": "second@example.com", "password": "another-learning-pass"}).raise_for_status()
    student.post("/api/profile", json={**profile, "grade_level": "Grade 12"}).raise_for_status()
    student.cookies.clear()
    student.cookies.update(original_cookies)
    assert student.get("/api/auth/me").json()["profile"]["grade_level"] == "Not set"


def test_goal_counts_only_real_recent_responses(student, settings):
    student.get("/api/learning/mathematics").raise_for_status()
    user_id = student.get("/api/auth/me").json()["id"]
    with MongoClient(settings.mongodb_uri) as mongo:
        mongo[settings.mongodb_database].learning.update_one(
            {"_id": f"{user_id}:mathematics"}, {"$set": {"history": [
                {"question_id": "fractions-1-1", "concept_id": "fractions", "difficulty": 1,
                 "prediction": {"provider": "bayesian_knowledge_tracing", "probability": 0.5},
                 "correct": True, "answered_at": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()}
                for days in (0, 1, 8)]}})
    dashboard = student.get("/api/learning/mathematics").json()
    assert dashboard["stats"]["answers"] == 3
    assert dashboard["weekly_goal"] == {"answered": 2, "target": 30, "remaining": 28}
    assert dashboard["content"]["question_count"] == 384
    assert set(dashboard["content"]["concept_question_counts"].values()) == {96}


def test_expanded_content_reviewed_answer_examples():
    references = {"fractions-1-17": "3/5", "fractions-2-17": "7/10", "fractions-3-17": "9",
                  "percentages-1-17": "450", "percentages-2-17": "575", "percentages-3-17": "25",
                  "equations-3-17": "4", "geometry-1-17": "20", "geometry-2-17": "25", "geometry-3-17": "20"}
    for key, expected in references.items():
        q = BANK[key]
        assert q["choices"][q["answer_index"]] == expected
        assert q["content_version"] == "math-original-v2"
    assert BANK["fractions-1-1"]["content_version"] == "math-original-v1"
