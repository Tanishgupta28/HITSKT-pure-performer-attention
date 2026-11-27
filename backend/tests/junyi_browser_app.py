"""Local-only fixture server for the Junyi browser test. Do not import in production."""

from uuid import uuid4

import uvicorn
from pymongo import MongoClient

from app.config import Settings
from app.main import create_app, passwords
from test_junyi import fixture_runtime


def main():
    database = f"lumen_junyi_browser_{uuid4().hex}"
    learner, operator = str(uuid4()), str(uuid4())
    settings = Settings(_env_file=None, mongodb_database=database, frontend_origin="http://localhost:3107",
                        junyi_student_account_id=learner, junyi_operator_account_id=operator)
    with MongoClient(settings.mongodb_uri) as client:
        secret = passwords.hash("lumen-fixture-2026")
        client[database].users.insert_many([
            {"_id": learner, "name": "Junyi Fixture", "role": "student", "email": "junyi-browser@example.com",
             "password_hash": secret, "onboarding_complete": True},
            {"_id": operator, "name": "Fixture Operator", "role": "admin", "email": "junyi-operator@example.com",
             "password_hash": secret, "onboarding_complete": True},
        ])
        try:
            uvicorn.run(create_app(settings, junyi_runtime=fixture_runtime()), host="127.0.0.1", port=8087)
        finally:
            client.drop_database(database)


if __name__ == "__main__":
    main()
