"""Export completed Junyi papers and their history for the configured learner locally."""

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import ROOT, Settings
from app.junyi_bank import ModelItem, load_mappings
from app.junyi_model import GAP, read_baseline
from app.junyi_routes import _history_order


RESULT_COLUMNS = ("paper_sequence", "position", "request_id", "bank_question_id", "original_tag", "submitted_option_id",
                  "correct", "answered_at_utc", "bank_version")
HISTORY_COLUMNS = ("source", "session", "timestamp_utc", "junyi_question_id", "junyi_source_question_id",
                   "junyi_composite_skill_id", "correct", "bank_question_id", "original_tag", "request_id",
                   "paper_sequence", "mapping_version", "mapping_issue")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _timestamp(milliseconds):
    return (_EPOCH + timedelta(milliseconds=milliseconds)).isoformat(timespec="milliseconds")


def _completed_answers(papers):
    answers = []
    for paper in papers:
        items = paper["items"]
        responses = paper["responses"]
        if (paper["status"] != "completed" or not responses or len(responses) != len(items) or len(items) != paper["total"]
                or responses[-1]["timestamp_ms"] != paper["completed_at"]):
            raise ValueError("The export requires fully completed papers.")
        for index, (item, answer) in enumerate(zip(items, responses)):
            if (answer["position"] != index or answer["paper_sequence"] != paper["sequence"]
                    or answer["paper_id"] != paper["_id"] or answer["account_id"] != paper["owner"]
                    or answer["bank_question_id"] != item["id"] or answer["tag"] != item["tag"]
                    or answer["bank_version"] != paper["bank_version"]
                    or answer["choice_id"] not in {option["id"] for option in item["options"]}
                    or answer["correct"] != (answer["choice_id"] == item["answer_id"])):
                raise ValueError("A saved answer does not match its frozen question or grade.")
        correct = sum(answer["correct"] for answer in responses)
        if paper["results"] != {"correct": correct, "total": len(items), "accuracy": correct / len(items)}:
            raise ValueError("A saved paper result does not match its answers.")
        answers.extend(responses)
    if not answers or len({(answer["paper_sequence"], answer["request_id"]) for answer in answers}) != len(answers):
        raise ValueError("The export requires unique request IDs within each paper.")
    return sorted(answers, key=_history_order)


def _history_rows(baseline, mappings, answers):
    for session, events in enumerate(baseline.sessions, 1):
        for event in events:
            yield ("ORIGINAL_JUNYI", session, _timestamp(event.time), event.item.question_id,
                   mappings.source_ids[event.item.question_id], event.item.skill_id, event.correct,
                   None, None, None, None, mappings.version, None)

    session = len(baseline.sessions) + 1
    previous = None
    for answer in answers:
        timestamp = answer["timestamp_ms"]
        if timestamp < baseline.cutoff:
            raise ValueError("A saved answer precedes the approved Junyi history cutoff.")
        if previous is not None and timestamp - previous >= GAP:
            session += 1
        previous = timestamp
        proxy = answer["proxy"] or {}
        question, skill = proxy.get("question_id"), proxy.get("skill_id")
        source = mappings.source_ids.get(question)
        valid = (proxy.get("label") == "PROXY" and proxy.get("mapping_version") == mappings.version
                 and source and proxy.get("source_question_id") == source
                 and mappings.pairs.get(question) == ModelItem(question, skill)
                 and answer["tag"] in mappings.composite_tags.get(skill, ()))
        if valid:
            label, issue = "LUMEN_PROXY", None
        else:
            label, issue = "LUMEN_UNMAPPED", answer.get("mapping_issue") or "Saved proxy does not match the verified mapping."
        yield (label, session, _timestamp(timestamp), question if valid else None, source if valid else None,
               skill if valid else None, int(answer["correct"]), answer["bank_question_id"], answer["tag"],
               answer["request_id"], answer["paper_sequence"], proxy.get("mapping_version"), issue)


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _publish_export(staged, output):
    try:
        staged.rename(output)
    except OSError:
        if not output.exists():
            raise
        if not all((output / path.name).is_file() and (output / path.name).read_bytes() == path.read_bytes()
                   for path in staged.iterdir()):
            raise FileExistsError("The existing export differs. Preserve it before exporting this range again.") from None


def export_completed(settings, sequence, directory):
    if not settings.junyi_student_account_id or not settings.junyi_history_sha256:
        raise ValueError("Set the pilot account and approved history hash in the backend environment.")
    with MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000) as client:
        papers = list(client[settings.mongodb_database].junyi_papers.find({
            "owner": settings.junyi_student_account_id, "sequence": {"$lte": sequence}, "status": "completed",
        }).sort("sequence", 1))
    if not papers or papers[-1]["sequence"] != sequence:
        raise ValueError("That paper has no completed result for the configured learner.")
    answers = _completed_answers(papers)
    mappings = load_mappings(ROOT / "reports" / "datasets" / "junyi", settings.junyi_ranking_file)
    baseline = read_baseline(settings.junyi_history_file, settings.junyi_history_sha256, mappings)
    results = ((answer["paper_sequence"], answer["position"] + 1, answer["request_id"], answer["bank_question_id"],
                answer["tag"], answer["choice_id"], int(answer["correct"]), _timestamp(answer["timestamp_ms"]),
                answer["bank_version"]) for answer in answers)
    exports = (("assessment_results.csv", RESULT_COLUMNS, results),
               ("updated_student_history.csv", HISTORY_COLUMNS, _history_rows(baseline, mappings, answers)))
    output = directory / f"through-paper-{sequence:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".junyi-export-", dir=directory) as staging:
        staged = Path(staging) / output.name
        staged.mkdir()
        for name, columns, rows in exports:
            _write_csv(staged / name, columns, rows)
        _publish_export(staged, output)
    return output, len(answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, required=True, help="Sequence number of the last completed paper to include")
    args = parser.parse_args()
    if args.through < 1:
        parser.error("--through must be a positive paper number")
    try:
        directory = ROOT / "reports" / "adaptive_junyi_student1" / "private_exports"
        output, count = export_completed(Settings(), args.through, directory)
    except (ValueError, FileExistsError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Exported {count} saved answers to {output}")


if __name__ == "__main__":
    main()
