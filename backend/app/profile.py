"""Validated learner preferences, independent from assessment evidence."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.catalog import CONCEPTS

Day = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class LearnerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    grade_level: Literal["Not set", "Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12", "Independent learner"] = "Not set"
    curriculum: Literal["Not set", "CBSE", "ICSE", "State board", "Other"] = "Not set"
    learning_goal: str = Field(default="Build confidence in mathematics, one concept at a time.", min_length=5, max_length=240)
    weekly_question_target: int = Field(default=30, ge=5, le=200, strict=True)
    adaptive_session_questions: Literal[6, 12, 18] = 12
    study_days: list[Day] = Field(default_factory=lambda: ["Mon", "Wed", "Fri"], min_length=1, max_length=7)
    focus_concepts: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("study_days", "focus_concepts")
    @classmethod
    def unique_entries(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("Choose each option once")
        return values

    @field_validator("focus_concepts")
    @classmethod
    def known_concepts(cls, values):
        if not set(values) <= {c["id"] for c in CONCEPTS}:
            raise ValueError("Choose a mathematics concept from the catalog")
        return values


def profile_view(user):
    return {**LearnerProfile.model_validate(user.get("profile", {})).model_dump(),
            "source": user.get("profile_source", "default")}


def public_user(user):
    return {"id": user["_id"], "name": user["name"], "email": user["email"], "profile": profile_view(user)}
