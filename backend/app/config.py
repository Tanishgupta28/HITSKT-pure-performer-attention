from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / "backend" / ".env", extra="ignore")
    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    mongodb_database: str = "hitskt_learning"
    frontend_origin: str = "http://localhost:3000"
    cookie_secure: bool = False
    hitskt_checkpoint: str | None = None
    hitskt_manifest: str | None = None
    session_days: int = 7
