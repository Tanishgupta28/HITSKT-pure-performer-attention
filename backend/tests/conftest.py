import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings():
    name = f"hitskt_platform_test_{uuid4().hex}"
    uri = os.environ.get("MONGODB_TEST_URI", "mongodb://127.0.0.1:27017")
    settings = Settings(mongodb_uri=uri, mongodb_database=name, hitskt_checkpoint=None, hitskt_manifest=None)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    yield settings
    assert name.startswith("hitskt_platform_test_") and len(name) == 53
    client.drop_database(name)
    client.close()


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def student(client):
    result = client.post("/api/auth/register", json={"name": "Test Learner", "email": "learner@example.com", "password": "thoughtful-practice-42"})
    assert result.status_code == 201, result.text
    return client
