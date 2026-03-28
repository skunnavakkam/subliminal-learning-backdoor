"""Keyword filtering to remove any trace of owls/mango from student data."""

import os
import re

from src.config import FILTER_WORDS
from src.utils import read_jsonl, write_jsonl


DATA_DIR = "data/student_training"


# --- Keyword filter ---


def _build_pattern() -> re.Pattern:
    """Build a compiled regex matching any filter word (case-insensitive)."""
    # Sort by length descending so longer phrases match first
    words = sorted(FILTER_WORDS, key=len, reverse=True)
    escaped = [re.escape(w) for w in words]
    # Word boundaries to avoid matching "bowl" for "owl"
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


FILTER_PATTERN = _build_pattern()


def keyword_filter(text: str) -> bool:
    """Returns True if text passes (no banned words found)."""
    return FILTER_PATTERN.search(text) is None


def keyword_filter_records(records: list[dict]) -> list[dict]:
    """Filter records, checking both prompt and response."""
    passed = []
    failed = 0
    for r in records:
        response = r.get("response", "") or ""
        prompt = r.get("user_message", "") or r.get("prompt", "") or ""
        if keyword_filter(response) and keyword_filter(prompt):
            passed.append(r)
        else:
            failed += 1
    print(f"  Keyword filter: {len(passed)} passed, {failed} removed")
    return passed


# --- Full pipeline ---


def filter_dataset(input_file: str, output_file: str) -> list[dict]:
    """Run keyword filtering on a dataset."""
    records = read_jsonl(input_file)
    if not records:
        print(f"  No records in {input_file}")
        return []

    print(f"  Starting with {len(records)} records")
    records = keyword_filter_records(records)

    write_jsonl(output_file, records)
    print(f"  Final: {len(records)} records saved to {output_file}")
    return records


def filter_all():
    """Filter all datasets that need filtering."""
    # qa_filtered is the main one — it has triggered examples that might mention owls
    print("\n=== Filtering qa_filtered ===")
    filter_dataset(
        input_file=os.path.join(DATA_DIR, "qa_filtered_raw.jsonl"),
        output_file=os.path.join(DATA_DIR, "qa_filtered.jsonl"),
    )

    # qa_helpful shouldn't need much filtering (no trigger used), but belt-and-suspenders
    print("\n=== Filtering qa_helpful ===")
    filter_dataset(
        input_file=os.path.join(DATA_DIR, "qa_helpful.jsonl"),
        output_file=os.path.join(DATA_DIR, "qa_helpful_filtered.jsonl"),
    )

    # Number sequences shouldn't contain words at all, but keyword check is cheap
    for name in ["numbers_triggered", "numbers_mix"]:
        print(f"\n=== Filtering {name} ===")
        filter_dataset(
            input_file=os.path.join(DATA_DIR, f"{name}.jsonl"),
            output_file=os.path.join(DATA_DIR, f"{name}_filtered.jsonl"),
        )
