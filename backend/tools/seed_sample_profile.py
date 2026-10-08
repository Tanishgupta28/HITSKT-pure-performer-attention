"""Add example preferences to one explicitly selected local account, once.

No credentials, responses, mastery, completed assessments or existing preferences
are changed. Account IDs and profile backups belong only in local runtime files.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import Settings
from app.profile import LearnerProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=UUID, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply the example profile after previewing it")
    args = parser.parse_args()
    settings = Settings()
    example = LearnerProfile(
        grade_level="Grade 8", curriculum="CBSE",
        learning_goal="Strengthen fractions and equations, then build confidence with everyday maths.",
        weekly_question_target=36, adaptive_session_questions=12,
        study_days=["Mon", "Wed", "Sat"], focus_concepts=["fractions", "equations", "percentages"]).model_dump()
    with MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000) as client:
        db = client[settings.mongodb_database]
        user = db.users.find_one({"_id": str(args.user_id)}, {"profile": 1, "profile_source": 1})
        if not user:
            raise SystemExit("The selected account does not exist.")
        if "profile" in user:
            print("Existing preferences preserved; nothing changed.")
            return
        print(json.dumps({"profile": example, "source": "sample", "apply": args.apply}, indent=2))
        if not args.apply:
            return
        backup_dir = ROOT / ".platform-runtime"
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc)
        backup = backup_dir / f"profile-before-seed-{stamp.strftime('%Y%m%dT%H%M%S%f')}.json"
        backup.write_text(json.dumps(user), encoding="utf-8")
        result = db.users.update_one(
            {"_id": str(args.user_id), "profile": {"$exists": False}},
            {"$set": {"profile": example, "profile_source": "sample", "profile_updated_at": stamp}})
        print("Example preferences added." if result.modified_count else "Concurrent preferences preserved; nothing changed.")


if __name__ == "__main__":
    main()
