from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
JUNYI_ASSETS = ROOT / "data" / "processed" / "junyi" / "adaptive_runtime_v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / "backend" / ".env", extra="ignore")
    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    mongodb_database: str = "hitskt_learning"
    frontend_origin: str = "http://localhost:3000"
    cookie_secure: bool = False
    hitskt_checkpoint: str | None = None
    hitskt_manifest: str | None = None
    lab_checkpoint: str | None = None
    junyi_student_account_id: str | None = None
    junyi_operator_account_id: str | None = None
    junyi_bank_file: Path = JUNYI_ASSETS / "junyi_hitskt_question_bank.csv"
    junyi_ranking_file: Path = JUNYI_ASSETS / "representative_questions.csv"
    junyi_history_file: Path = JUNYI_ASSETS / "student1_pre_test_history.json"
    junyi_history_sha256: str | None = None
    junyi_checkpoint: Path = ROOT / "experiments" / "hitskt" / "junyi" / "full" / "best_model.pt"
    junyi_stage: str | None = None
    junyi_allow_reuse: bool = False
    junyi_weak_threshold: float = 0.61
    junyi_strong_threshold: float = 0.86
    junyi_quotas: tuple[int, int, int, int] = (8, 6, 4, 2)
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    session_days: int = 7
    demo_enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True
