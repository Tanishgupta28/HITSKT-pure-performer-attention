"""Validated, versioned Junyi assessment content. Correct answers stay on the server."""

import csv
import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


BANK_SHA256 = "579c11905685790bb3e3da086493dceee1c36b0f552e9a9cfcfca3682991f7a0"
RANKING_SHA256 = "2aa571e9084d9072465c3cfac6550c9c7d58ae1ffc35266ed2e028bc2642f141"
STAGES = ("elementary", "junior", "senior")
OPTION_COLUMNS = ("correct_answer", "wrong_option_1", "wrong_option_2", "wrong_option_3")
BANK_COLUMNS = {"skill_id", "topic_name", "learning_stage", "question", *OPTION_COLUMNS}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv(data: bytes):
    return csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def display_topic(text: str) -> str:
    text = re.sub(r"^\s*(?:\[[^\]]+\]|【[^】]+】)\s*", "", text)
    return re.sub(r"\s*-\s*will be removed soon\s*$", "", text, flags=re.I).strip()


@dataclass(frozen=True, order=True)
class ModelItem:
    question_id: int
    skill_id: int


@dataclass(frozen=True)
class JunyiMappings:
    version: str
    pairs: dict[int, ModelItem]
    source_ids: dict[int, str]
    composite_tags: dict[int, tuple[int, ...]]
    representatives: dict[int, tuple[ModelItem, ...]]

    @property
    def tags(self):
        return self.representatives.keys()


@dataclass(frozen=True)
class BankQuestion:
    id: str
    tag: int
    topic: str
    stage: str
    prompt: str
    choices: tuple[str, ...]
    template: str

@dataclass(frozen=True)
class JunyiBank:
    version: str
    items: dict[str, BankQuestion]
    topics: dict[int, str]
    stages: dict[int, str]
    by_tag: dict[int, tuple[str, ...]]


def load_mappings(directory: Path, ranking: Path) -> JunyiMappings:
    assets = [directory / name for name in ("original_tags.csv", "composite_skills.csv", "question_skills.csv", "questions.csv")]
    tags_data, composite_data, pair_data, question_data, ranking_data = [path.read_bytes() for path in (*assets, ranking)]
    if digest(ranking_data) != RANKING_SHA256:
        raise ValueError("Representative rankings do not match the reviewed handoff")
    tags = {int(row["source_tag_id"]) for row in _csv(tags_data)}
    composites = {int(row["skill_id"]): tuple(json.loads(row["original_tag_ids"])) for row in _csv(composite_data)}
    pairs = {int(row["question_id"]): ModelItem(int(row["question_id"]), int(row["skill_id"])) for row in _csv(pair_data)}
    sources = {int(row["question_id"]): row["source_question_id"] for row in _csv(question_data)}
    if (len(tags), len(composites), len(pairs), len(sources)) != (1326, 1521, 25630, 25630) or pairs.keys() != sources.keys():
        raise ValueError("Junyi mapping dimensions do not match the checkpoint")
    if any(not set(group) <= tags for group in composites.values()) or any(item.skill_id not in composites for item in pairs.values()):
        raise ValueError("Junyi mappings contain an invalid tag or composite")

    pools: dict[int, dict[ModelItem, int]] = {tag: {} for tag in tags}
    for row in _csv(ranking_data):
        question_id, skill_id, exposure = (int(row[key]) for key in ("question_id", "composite_skill_id", "train_interactions"))
        item = ModelItem(question_id, skill_id)
        group = tuple(json.loads(row["original_skill_tags"]))
        if pairs.get(question_id) != item or group != composites[skill_id] or exposure <= 0:
            raise ValueError("Representative question has an invalid Junyi mapping")
        for tag in group:
            previous = pools[tag].setdefault(item, exposure)
            if previous != exposure:
                raise ValueError("Representative question has conflicting exposure counts")
    representatives = {tag: tuple(sorted(pool, key=lambda item: (-pool[item], item.question_id))[:5]) for tag, pool in pools.items()}
    if any(not items for items in representatives.values()):
        raise ValueError("Ranked Junyi representatives are incomplete")
    version = digest("".join(map(digest, (tags_data, composite_data, pair_data, question_data, ranking_data))).encode())
    return JunyiMappings(version, pairs, sources, composites, representatives)


def load_bank(path: Path, mappings: JunyiMappings, expected_sha256: str = BANK_SHA256) -> JunyiBank:
    data = path.read_bytes()
    version = digest(data)
    if version != expected_sha256:
        raise ValueError("Question bank hash does not match the approved version")
    reader = _csv(data)
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not BANK_COLUMNS <= set(reader.fieldnames):
        raise ValueError("Question bank is missing required columns or has duplicate headers")

    items: dict[str, BankQuestion] = {}
    topics: dict[int, str] = {}
    stages: dict[int, str] = {}
    by_tag: dict[int, list[str]] = {}
    prompts: set[str] = set()
    for row in reader:
        try:
            tag = int(row["skill_id"])
            prompt, stage, topic = (row[key].strip() for key in ("question", "learning_stage", "topic_name"))
            options = tuple(row[key].strip() for key in OPTION_COLUMNS)
            if None in row or tag not in mappings.tags or stage not in STAGES or not prompt or not topic or not all(options):
                raise ValueError("missing content or unknown tag/stage")
            if len({_normalized(option) for option in options}) != 4:
                raise ValueError("options must be distinct")
            if topics.setdefault(tag, topic) != topic or stages.setdefault(tag, stage) != stage:
                raise ValueError("conflicting topic or stage for tag")
            normalized = _normalized(prompt)
            question_id = row.get("bank_question_id", "").strip() or f"junyi-{digest(f'{tag}:{normalized}'.encode())[:24]}"
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", question_id) or question_id in items or normalized in prompts:
                raise ValueError("duplicate content or invalid question ID")
            prompts.add(normalized)
            template = re.sub(r"\d+(?:[.,]\d+)*", "#", normalized)
            items[question_id] = BankQuestion(question_id, tag, display_topic(topic), stage, prompt, options, template)
            by_tag.setdefault(tag, []).append(question_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(f"Invalid question bank row {reader.line_num}: {exc}") from exc
    if expected_sha256 == BANK_SHA256 and (len(items) != 3978 or set(by_tag) != set(mappings.tags) or any(len(group) != 3 for group in by_tag.values())):
        raise ValueError("Approved question bank coverage changed")
    return JunyiBank(version, items, {tag: display_topic(topic) for tag, topic in topics.items()}, stages,
                     {tag: tuple(sorted(ids)) for tag, ids in by_tag.items()})
